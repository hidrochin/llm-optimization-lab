"""Convert CORD v2 receipts (CC BY 4.0) into dataset records for the dev set.

Each receipt gives two records sharing a group_id: its image, and its spatial text
rendered from the OCR boxes. Targets follow doc_types/cord_receipt/schema.json.

Usage:
    uv run --group data python scripts/data/convert_cord.py --out-dir "$LLMOPT_DATA_DIR/public/cord"
    uv run python scripts/split_dataset.py --data-dir "$LLMOPT_DATA_DIR/public/cord" \
        --input "$LLMOPT_DATA_DIR/public/cord/all.jsonl"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_opt_lab.data import Sample, write_samples
from llm_opt_lab.spatial_text import Word, render_spatial_text

DATASET_ID = "naver-clova-ix/cord-v2"
DOC_TYPE = "cord_receipt"

# CORD key -> our key. Every key is always present in the target (null when absent).
MENU_FIELDS = {"nm": "name", "cnt": "count", "unitprice": "unit_price", "price": "price"}
SUBTOTAL_FIELDS = {
    "subtotal_price": "subtotal_price",
    "discount_price": "discount_price",
    "service_price": "service_price",
    "tax_price": "tax_price",
}
TOTAL_FIELDS = {
    "total_price": "total_price",
    "cashprice": "cash_price",
    "changeprice": "change_price",
    "creditcardprice": "credit_card_price",
}


def _value(value: Any) -> str | None:
    """CORD values are strings, occasionally lists of strings for multi-part values."""
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(v) for v in value) or None
    return str(value)


def _pick(source: Any, fields: dict[str, str]) -> dict[str, str | None]:
    source = source if isinstance(source, dict) else {}
    return {ours: _value(source.get(theirs)) for theirs, ours in fields.items()}


def cord_target(gt_parse: dict[str, Any]) -> dict[str, Any]:
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):  # a single item is stored as a dict
        menu = [menu]
    return {
        "menu": [_pick(item, MENU_FIELDS) for item in menu],
        "subtotal": _pick(gt_parse.get("sub_total"), SUBTOTAL_FIELDS),
        "total": _pick(gt_parse.get("total"), TOTAL_FIELDS),
    }


def cord_words(valid_line: list[dict[str, Any]]) -> list[Word]:
    words = []
    for line in valid_line:
        for word in line.get("words", []):
            quad = word["quad"]
            xs = [quad[f"x{i}"] for i in range(1, 5)]
            ys = [quad[f"y{i}"] for i in range(1, 5)]
            words.append(Word(word["text"], min(xs), min(ys), max(xs), max(ys)))
    return words


def convert_example(example: dict[str, Any], split: str, index: int, out_dir: Path) -> list[Sample]:
    ground_truth = json.loads(example["ground_truth"])
    doc_id = f"cord-{split}-{index:04d}"
    image = f"images/{doc_id}.jpg"
    (out_dir / image).parent.mkdir(parents=True, exist_ok=True)
    example["image"].convert("RGB").save(out_dir / image, quality=95)
    target = cord_target(ground_truth["gt_parse"])
    meta = {"source": f"{DATASET_ID}:{split}:{index}"}
    common = {"doc_type": DOC_TYPE, "group_id": doc_id, "target": target, "meta": meta}
    return [
        Sample(id=f"{doc_id}-img", input_type="image", image=image, **common),
        Sample(
            id=f"{doc_id}-txt",
            input_type="spatial_text",
            spatial_text=render_spatial_text(cord_words(ground_truth["valid_line"])),
            **common,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, help="receipts per CORD split (quick runs)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    from datasets import load_dataset

    out = args.out_dir / "all.jsonl"
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; pass --overwrite to replace it")
    samples: list[Sample] = []
    for split in args.splits:
        dataset = load_dataset(DATASET_ID, split=split)
        n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        for index in range(n):
            samples.extend(convert_example(dataset[index], split, index, args.out_dir))
        print(f"{split}: {n} receipts")
    write_samples(out, samples)
    print(f"{len(samples)} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
