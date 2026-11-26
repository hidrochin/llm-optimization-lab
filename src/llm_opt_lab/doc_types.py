"""Document type registry.

Each document type is a folder `doc_types/<name>/` with two files:

- `schema.json`: the field registry, a JSON Schema subset (`object` with `properties`,
  `array` with `items`, and leaves). A leaf may carry `description` and `examples`, which
  the model sees in the system prompt; `"x-field-type"` (default "text"), which selects its
  normalizer in postprocess; and `"x-mode"`: "extract" (default, a value printed on the
  document) or "derive" (computed by business rules after extraction, never asked of the
  model). The root may list `"x-decoys"`: fields that can never appear on this doc type.
  Training and evaluation ask for them to teach and measure answering `null`.
- `instructions.md`: what the document is and doc-type rules. It never lists fields: the
  field list of the system prompt is generated from the registry (see prompting.py), so a
  request can ask for any subset of the fields.

Using JSON Schema keeps the door open for vLLM structured outputs (T-C09).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INSTRUCTIONS_FILE = "instructions.md"
SCHEMA_FILE = "schema.json"
DEFAULT_FIELD_TYPE = "text"
FIELD_MODES = ("extract", "derive")


@dataclass(frozen=True)
class FieldSpec:
    """One leaf of the registry, or a decoy."""

    path: str  # generic leaf path, lists as `[]`: `menu[].price`
    field_type: str = DEFAULT_FIELD_TYPE
    mode: str = "extract"
    description: str = ""
    examples: tuple[str, ...] = ()
    decoy: bool = False


@dataclass(frozen=True)
class DocType:
    name: str
    instructions: str
    schema: dict[str, Any]
    fields: tuple[FieldSpec, ...] = ()  # registry order
    decoys: tuple[FieldSpec, ...] = ()
    field_types: dict[str, str] = field(default_factory=dict)

    def field_type(self, generic_path: str) -> str:
        return self.field_types.get(generic_path, DEFAULT_FIELD_TYPE)

    @property
    def extract_fields(self) -> tuple[FieldSpec, ...]:
        """The fields the model is asked for; `derive` fields come from business rules."""
        return tuple(f for f in self.fields if f.mode == "extract")


def schema_fields(schema: dict[str, Any], prefix: str = "") -> list[FieldSpec]:
    """Every leaf of the schema, in registry order.

    >>> [f.path for f in schema_fields({"type": "object", "properties": {
    ...     "items": {"type": "array", "items": {"type": "object", "properties": {
    ...         "price": {"type": "string", "x-field-type": "number"}}}}}})]
    ['items[].price']
    """
    if "properties" in schema:
        result: list[FieldSpec] = []
        for key, sub in schema["properties"].items():
            result.extend(schema_fields(sub, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(schema.get("items"), dict):
        return schema_fields(schema["items"], f"{prefix}[]")
    mode = schema.get("x-mode", "extract")
    if mode not in FIELD_MODES:
        raise ValueError(f"{prefix}: x-mode must be one of {FIELD_MODES}, got {mode!r}")
    return [
        FieldSpec(
            path=prefix,
            field_type=schema.get("x-field-type", DEFAULT_FIELD_TYPE),
            mode=mode,
            description=schema.get("description", ""),
            examples=tuple(str(e) for e in schema.get("examples", ())),
        )
    ]


def schema_field_types(schema: dict[str, Any]) -> dict[str, str]:
    """Map every leaf's generic path (lists as `[]`) to its field type."""
    return {f.path: f.field_type for f in schema_fields(schema)}


def schema_decoys(schema: dict[str, Any]) -> list[FieldSpec]:
    decoys = []
    top_level = set(schema.get("properties", {}))
    for spec in schema.get("x-decoys", []):
        name = spec["name"]
        if not name.isidentifier() or name in top_level:
            raise ValueError(f"decoy {name!r} must be a plain key that is not a schema field")
        decoys.append(
            FieldSpec(
                path=name,
                description=spec.get("description", ""),
                examples=tuple(str(e) for e in spec.get("examples", ())),
                decoy=True,
            )
        )
    return decoys


def load_doc_type(directory: str | Path) -> DocType:
    directory = Path(directory)
    path = directory / INSTRUCTIONS_FILE
    if not path.is_file():
        hint = (
            "; system_prompt.md was replaced by instructions.md without the field list "
            "(docs/formats.md §2)"
            if (directory / "system_prompt.md").is_file()
            else ""
        )
        raise ValueError(f"{path} not found{hint}")
    instructions = path.read_text(encoding="utf-8").strip()
    if not instructions:
        raise ValueError(f"{path} is empty")
    schema = json.loads((directory / SCHEMA_FILE).read_text(encoding="utf-8"))
    fields = schema_fields(schema)
    return DocType(
        name=directory.name,
        instructions=instructions,
        schema=schema,
        fields=tuple(fields),
        decoys=tuple(schema_decoys(schema)),
        field_types={f.path: f.field_type for f in fields},
    )


def load_doc_types(root: str | Path, names: list[str] | None = None) -> dict[str, DocType]:
    """Load doc types under `root`; `names` restricts the set and must all exist."""
    root = Path(root)
    available = sorted(p.name for p in root.iterdir() if (p / SCHEMA_FILE).is_file())
    if names:
        missing = sorted(set(names) - set(available))
        if missing:
            raise ValueError(f"doc types not found under {root}: {missing}")
        available = [n for n in available if n in names]
    return {name: load_doc_type(root / name) for name in available}
