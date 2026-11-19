"""Token length statistics per doc type x input type, with the model's own processor.

Columns: system prompt, document (prompt minus system prompt: document, image tokens and
chat template), answer, total. Recommends `training.max_seq_length` and
`evaluation.max_new_tokens`. CPU is enough.

Usage:
    uv run --group train python scripts/data/token_stats.py \
        --config configs/dev/cord_e2b_base.yaml --split train --out reports/data_stats.md
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from llm_opt_lab.config import load_config
from llm_opt_lab.data import filter_samples, load_samples
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.hf.loading import is_text_only, load_processor
from llm_opt_lab.hf.sft_data import IGNORE_INDEX, build_features
from llm_opt_lab.profiling import percentile

COLUMNS = ("system", "document", "answer", "total")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, help="first N samples only")
    parser.add_argument("--out", type=Path, help="also write the tables to this Markdown file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    processor = load_processor(config)
    tokenizer = getattr(processor, "tokenizer", processor)
    doc_types = load_doc_types(config.data.doc_types_dir, config.data.doc_types or None)
    samples = filter_samples(
        load_samples(config.data.dir / f"{args.split}.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )[: args.limit]

    system_tokens = {
        name: len(tokenizer(dt.system_prompt, add_special_tokens=False)["input_ids"])
        for name, dt in doc_types.items()
    }
    lengths: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, sample in enumerate(samples, start=1):
        features = build_features(
            sample,
            doc_types[sample.doc_type],
            processor,
            config.data.dir,
            text_only=is_text_only(config),
            chat_template_kwargs=config.model.chat_template_kwargs,
        )
        total = int(features["input_ids"].shape[0])
        answer = int((features["labels"] != IGNORE_INDEX).sum())
        system = system_tokens[sample.doc_type]
        row = {
            "system": system,
            "document": total - answer - system,
            "answer": answer,
            "total": total,
        }
        for key in (f"{sample.doc_type} / {sample.input_type}", "all"):
            for column, value in row.items():
                lengths[key][column].append(value)
        if i % 200 == 0:
            print(f"  {i}/{len(samples)}")

    lines = [
        f"Token lengths: `{config.model.id}`, split `{args.split}`, {len(samples)} samples",
        "",
    ]
    header = " | ".join(f"{c} p50 / p95 / max" for c in COLUMNS)
    lines += [f"| Doc type / input | n | {header} |", "|---|---:|" + "---:|" * len(COLUMNS)]
    for key in sorted(lengths, key=lambda k: (k == "all", k)):
        cells = [
            " / ".join(
                str(round(percentile(lengths[key][c], q))) if q < 100 else str(max(lengths[key][c]))
                for q in (50, 95, 100)
            )
            for c in COLUMNS
        ]
        lines.append(f"| {key} | {len(lengths[key]['total'])} | " + " | ".join(cells) + " |")

    max_total, max_answer = max(lengths["all"]["total"]), max(lengths["all"]["answer"])
    lines += [
        "",
        f"- Longest sample: {max_total} tokens → `training.max_seq_length` ≥ "
        f"{math.ceil(max_total / 1024) * 1024}",
        f"- Longest answer: {max_answer} tokens → `evaluation.max_new_tokens` ≈ "
        f"{math.ceil(max_answer * 1.2 / 64) * 64} (20% margin)",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
