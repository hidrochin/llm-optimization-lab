"""Document type registry.

Each document type has its own system prompt and output schema, stored as
`doc_types/<name>/system_prompt.md` and `doc_types/<name>/schema.json`.

The schema is a JSON Schema subset: `object` with `properties`, `array` with `items`,
and leaves. A leaf may carry `"x-field-type"` (default "text") to select its
normalizer in postprocess. Using JSON Schema keeps the door open for vLLM
structured outputs (T-C09).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT_FILE = "system_prompt.md"
SCHEMA_FILE = "schema.json"
DEFAULT_FIELD_TYPE = "text"


@dataclass(frozen=True)
class DocType:
    name: str
    system_prompt: str
    schema: dict[str, Any]
    field_types: dict[str, str] = field(default_factory=dict)

    def field_type(self, generic_path: str) -> str:
        return self.field_types.get(generic_path, DEFAULT_FIELD_TYPE)


def schema_field_types(schema: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Map every leaf's generic path (lists as `[]`) to its field type.

    >>> schema_field_types({"type": "object", "properties": {
    ...     "items": {"type": "array", "items": {"type": "object", "properties": {
    ...         "price": {"type": "number", "x-field-type": "number"}}}}}})
    {'items[].price': 'number'}
    """
    if "properties" in schema:
        result: dict[str, str] = {}
        for key, sub in schema["properties"].items():
            result.update(schema_field_types(sub, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(schema.get("items"), dict):
        return schema_field_types(schema["items"], f"{prefix}[]")
    return {prefix: schema.get("x-field-type", DEFAULT_FIELD_TYPE)}


def load_doc_type(directory: str | Path) -> DocType:
    directory = Path(directory)
    system_prompt = (directory / SYSTEM_PROMPT_FILE).read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"{directory / SYSTEM_PROMPT_FILE} is empty")
    schema = json.loads((directory / SCHEMA_FILE).read_text(encoding="utf-8"))
    return DocType(
        name=directory.name,
        system_prompt=system_prompt,
        schema=schema,
        field_types=schema_field_types(schema),
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
