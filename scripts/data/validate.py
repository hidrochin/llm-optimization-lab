"""Check dataset records before they are used: every record loads, its doc type exists,
its target matches the doc type JSON Schema, and its image file exists.

Also checks the annotation rules of docs/formats.md §1: a target never holds a `derive`
field (business rules compute those), and every entry of a list annotates the same keys
(a key missing from one entry would read as "not annotated" for that entry only).

Usage:
    uv run python scripts/data/validate.py --data-dir "$LLMOPT_DATA_DIR/private" --input all.jsonl
    uv run python scripts/data/validate.py --data-dir ... --input test.jsonl --reviewed-only

Exits with code 1 if anything is wrong.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from llm_opt_lab.data import Sample, load_samples
from llm_opt_lab.doc_types import DocType, load_doc_types
from llm_opt_lab.field_requests import annotated_paths
from llm_opt_lab.postprocess import check_field_types


def uneven_lists(value: Any, path: str = "") -> list[str]:
    """Paths of lists whose entries do not all have the same keys."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            found += uneven_lists(sub, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        entries = [v for v in value if isinstance(v, dict)]
        if len({frozenset(e) for e in entries}) > 1:
            found.append(path)
        for entry in value:
            found += uneven_lists(entry, f"{path}[]")
    return sorted(set(found))


def validate_samples(
    samples: list[Sample],
    doc_types: dict[str, DocType],
    data_dir: Path,
    *,
    reviewed_only: bool = False,
) -> list[str]:
    validators = {}
    errors = []
    for name, doc_type in doc_types.items():
        try:
            check_field_types(doc_type)
        except ValueError as e:
            errors.append(str(e))
        validators[name] = Draft202012Validator(doc_type.schema)
    for sample in samples:
        validator = validators.get(sample.doc_type)
        if validator is None:
            errors.append(f"{sample.id}: unknown doc type {sample.doc_type!r}")
            continue
        for error in validator.iter_errors(sample.target):
            where = "/".join(str(p) for p in error.absolute_path) or "(root)"
            errors.append(f"{sample.id}: target/{where}: {error.message}")
        doc_type = doc_types[sample.doc_type]
        derived = sorted(
            {f.path for f in doc_type.fields if f.mode == "derive"}
            & annotated_paths(sample.target, doc_type)
        )
        if derived:
            errors.append(
                f"{sample.id}: target holds derive fields (rules compute them): {derived}"
            )
        for path in uneven_lists(sample.target):
            errors.append(f"{sample.id}: entries of target/{path} annotate different keys")
        if reviewed_only and sample.label_source not in ("human", "teacher_reviewed"):
            errors.append(f"{sample.id}: label_source {sample.label_source!r} is not reviewed")
        if sample.image and not (data_dir / sample.image).is_file():
            errors.append(f"{sample.id}: image not found: {sample.image}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", default="all.jsonl", help="file name under --data-dir")
    parser.add_argument("--doc-types-dir", type=Path, default=Path("doc_types"))
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="every label_source must be human or teacher_reviewed (test splits)",
    )
    args = parser.parse_args(argv)

    samples = load_samples(args.data_dir / args.input)
    doc_types = load_doc_types(args.doc_types_dir)
    errors = validate_samples(samples, doc_types, args.data_dir, reviewed_only=args.reviewed_only)

    counts = Counter((s.doc_type, s.input_type) for s in samples)
    for (doc_type, input_type), n in sorted(counts.items()):
        print(f"{doc_type:<30}{input_type:<15}{n:>8}")
    sources = Counter(s.label_source or "unknown" for s in samples)
    print("label_source: " + ", ".join(f"{k} {n}" for k, n in sorted(sources.items())))
    for error in errors[:50]:
        print(f"ERROR {error}")
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more")
    print(f"{len(samples)} records, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
