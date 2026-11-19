"""Split the full dataset into train / validation / test / calibration, once.

Usage:
    uv run python scripts/split_dataset.py --data-dir "$LLMOPT_DATA_DIR" \
        --input "$LLMOPT_DATA_DIR/all.jsonl" --seed 42

Writes <data-dir>/<split>.jsonl and <data-dir>/split_manifest.json. Existing splits are
never overwritten without --overwrite: every experiment must use the same test set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llm_opt_lab.data import Sample, load_samples, write_samples
from llm_opt_lab.split import DEFAULT_RATIOS, split_samples


def parse_ratios(text: str) -> dict[str, float]:
    """`train=0.8,validation=0.05,test=0.1,calibration=0.05` -> dict."""
    ratios: dict[str, float] = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        ratios[name.strip()] = float(value)
    return ratios


def print_counts(splits: dict[str, list[Sample]]) -> None:
    keys = sorted({(s.doc_type, s.input_type) for samples in splits.values() for s in samples})
    print(f"{'doc_type / input_type':<40}" + "".join(f"{name:>12}" for name in splits))
    for doc_type, input_type in keys:
        counts = [
            sum(1 for s in samples if (s.doc_type, s.input_type) == (doc_type, input_type))
            for samples in splits.values()
        ]
        print(f"{doc_type + ' / ' + input_type:<40}" + "".join(f"{c:>12}" for c in counts))
    print(f"{'total':<40}" + "".join(f"{len(samples):>12}" for samples in splits.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios", type=parse_ratios, default=DEFAULT_RATIOS, help="name=ratio,... (sum to 1)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    samples = [s for path in args.input for s in load_samples(path)]
    duplicates = [i for i, n in Counter(s.id for s in samples).items() if n > 1]
    if duplicates:
        parser.error(f"duplicate sample ids across inputs, e.g. {duplicates[:5]}")

    outputs = [args.data_dir / f"{name}.jsonl" for name in args.ratios]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.overwrite:
        parser.error(f"splits already exist, refusing to overwrite: {existing}")

    splits = split_samples(samples, args.ratios, args.seed)
    for name, split in splits.items():
        write_samples(args.data_dir / f"{name}.jsonl", split)
    manifest = {
        "seed": args.seed,
        "ratios": args.ratios,
        "inputs": [str(p) for p in args.input],
        "counts": {name: len(split) for name, split in splits.items()},
        "ids": {name: [s.id for s in split] for name, split in splits.items()},
    }
    (args.data_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_counts(splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
