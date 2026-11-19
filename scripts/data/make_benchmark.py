"""Draw the serving benchmark set from the test split, once.

Stratified by doc type and input type; uniform within each stratum, so the real length
distribution is kept. Use at least 5x the highest concurrency level (300 for 60), so the
server stays saturated for most of each level.

Usage:
    uv run python scripts/data/make_benchmark.py --data-dir "$LLMOPT_DATA_DIR/private" --n 300
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from llm_opt_lab.data import load_samples, write_samples
from llm_opt_lab.split import sample_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = args.data_dir / "benchmark.jsonl"
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; the benchmark set must stay the same for every candidate")
    test = load_samples(args.data_dir / "test.jsonl")
    chosen = sample_benchmark(test, min(args.n, len(test)), args.seed)
    write_samples(out, chosen)
    for (doc_type, input_type), n in sorted(
        Counter((s.doc_type, s.input_type) for s in chosen).items()
    ):
        print(f"{doc_type:<30}{input_type:<15}{n:>6}")
    print(f"{len(chosen)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
