"""Check dataset records before they are used: every record loads, its doc type exists,
its target matches the doc type JSON Schema, and its image file exists.

Usage:
    uv run python scripts/data/validate.py --data-dir "$LLMOPT_DATA_DIR/private" --input all.jsonl

Exits with code 1 if anything is wrong.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from llm_opt_lab.data import Sample, load_samples
from llm_opt_lab.doc_types import DocType, load_doc_types
from llm_opt_lab.postprocess import check_field_types


def validate_samples(
    samples: list[Sample], doc_types: dict[str, DocType], data_dir: Path
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
        if sample.image and not (data_dir / sample.image).is_file():
            errors.append(f"{sample.id}: image not found: {sample.image}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", default="all.jsonl", help="file name under --data-dir")
    parser.add_argument("--doc-types-dir", type=Path, default=Path("doc_types"))
    args = parser.parse_args(argv)

    samples = load_samples(args.data_dir / args.input)
    doc_types = load_doc_types(args.doc_types_dir)
    errors = validate_samples(samples, doc_types, args.data_dir)

    counts = Counter((s.doc_type, s.input_type) for s in samples)
    for (doc_type, input_type), n in sorted(counts.items()):
        print(f"{doc_type:<30}{input_type:<15}{n:>8}")
    for error in errors[:50]:
        print(f"ERROR {error}")
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more")
    print(f"{len(samples)} records, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
