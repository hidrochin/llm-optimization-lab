"""Convert CORD v2 receipts (CC BY 4.0) into dataset records for the dev set.

Each receipt gives two records sharing a group_id: its image, and its spatial text
rendered from the OCR boxes. Targets follow doc_types/cord_receipt/schema.json.

`--relabel-from` instead rewrites an already built cord_receipt set with every CORD field
(doc_types/cord_receipt_full): same ids, splits, images and spatial text, so the
field-holdout experiment (E20) runs on the frozen dev split.

Usage:
    uv run --group data python scripts/data/convert_cord.py --out-dir "$LLMOPT_DATA_DIR/public/cord"
    uv run python scripts/split_dataset.py --data-dir "$LLMOPT_DATA_DIR/public/cord" \
        --input "$LLMOPT_DATA_DIR/public/cord/all.jsonl"
    uv run --group data python scripts/data/convert_cord.py \
        --out-dir "$LLMOPT_DATA_DIR/public/cord_full" --relabel-from "$LLMOPT_DATA_DIR/public/cord"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from llm_opt_lab.data import Sample, load_samples, write_samples
from llm_opt_lab.spatial_text import Word, render_spatial_text

DATASET_ID = "naver-clova-ix/cord-v2"
DOC_TYPE = "cord_receipt"
FULL_DOC_TYPE = "cord_receipt_full"
RECORD_FILES = ("all", "train", "validation", "test", "calibration", "benchmark")

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

# cord_receipt_full: every CORD key except `void_menu` (one receipt in 1,000), in the
# registry order of doc_types/cord_receipt_full/schema.json. Shared keys keep their names.
FULL_MENU_FIELDS = {
    "nm": "name",
    "num": "num",
    "cnt": "count",
    "unitprice": "unit_price",
    "discountprice": "discount_price",
    "price": "price",
    "itemsubtotal": "item_subtotal",
    "vatyn": "vat_flag",
    "etc": "etc",
}
FULL_SUB_FIELDS = {"nm": "name", "cnt": "count", "unitprice": "unit_price", "price": "price"}
FULL_SUBTOTAL_FIELDS = {
    "subtotal_price": "subtotal_price",
    "discount_price": "discount_price",
    "service_price": "service_price",
    "othersvc_price": "other_service_price",
    "tax_price": "tax_price",
    "etc": "etc",
}
FULL_TOTAL_FIELDS = {
    "total_price": "total_price",
    "total_etc": "total_etc",
    "cashprice": "cash_price",
    "changeprice": "change_price",
    "creditcardprice": "credit_card_price",
    "emoneyprice": "emoney_price",
    "menutype_cnt": "menu_type_count",
    "menuqty_cnt": "menu_quantity_count",
}


def _value(value: Any) -> str | None:
    """CORD values are strings, occasionally lists of strings for multi-part values."""
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(v) for v in value) or None
    return str(value)


def _pick(source: Any, fields: dict[str, str]) -> dict[str, str | None]:
    # A section printed twice (e.g. a subtotal before and after a discount) is a list of
    # dicts: the first block with a value wins for each key.
    blocks = source if isinstance(source, list) else [source]
    blocks = [block for block in blocks if isinstance(block, dict)]
    return {
        ours: next((v for block in blocks if (v := _value(block.get(theirs))) is not None), None)
        for theirs, ours in fields.items()
    }


def cord_target(gt_parse: dict[str, Any]) -> dict[str, Any]:
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):  # a single item is stored as a dict
        menu = [menu]
    return {
        "menu": [_pick(item, MENU_FIELDS) for item in menu],
        "subtotal": _pick(gt_parse.get("sub_total"), SUBTOTAL_FIELDS),
        "total": _pick(gt_parse.get("total"), TOTAL_FIELDS),
    }


def _entries(value: Any) -> list[dict[str, Any]]:
    """A CORD list section: a dict for a single entry, a list for several."""
    if isinstance(value, dict):
        return [value]
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def cord_target_full(gt_parse: dict[str, Any]) -> dict[str, Any]:
    return {
        "menu": [
            {
                **_pick(item, FULL_MENU_FIELDS),
                "sub": [_pick(sub, FULL_SUB_FIELDS) for sub in _entries(item.get("sub"))],
            }
            for item in _entries(gt_parse.get("menu"))
        ],
        "subtotal": _pick(gt_parse.get("sub_total"), FULL_SUBTOTAL_FIELDS),
        "total": _pick(gt_parse.get("total"), FULL_TOTAL_FIELDS),
    }


def relabel_full(source: Path, out: Path) -> None:
    """Copy the record files of a built cord_receipt set with cord_receipt_full targets."""
    from datasets import load_dataset

    datasets: dict[str, Any] = {}

    def gt_parse(sample: Sample) -> dict[str, Any]:
        _, split, index = sample.meta["source"].rsplit(":", 2)
        if split not in datasets:
            datasets[split] = load_dataset(DATASET_ID, split=split)
        return json.loads(datasets[split][int(index)]["ground_truth"])["gt_parse"]

    out.mkdir(parents=True, exist_ok=True)
    for name in RECORD_FILES:
        path = source / f"{name}.jsonl"
        if not path.exists():
            continue
        samples = [
            s.model_copy(
                update={
                    "doc_type": FULL_DOC_TYPE,
                    "target": cord_target_full(gt_parse(s)),
                    "label_source": "human",  # CORD's own annotation
                }
            )
            for s in load_samples(path)
        ]
        write_samples(out / path.name, samples)
        print(f"{path.name}: {len(samples)} records")
    shutil.copy(source / "split_manifest.json", out / "split_manifest.json")
    images = out / "images"
    if not images.exists():
        images.symlink_to(os.path.relpath(source / "images", out))  # same image files


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
    parser.add_argument(
        "--relabel-from", type=Path, help="a built cord_receipt set to relabel with every field"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = args.out_dir / "all.jsonl"
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; pass --overwrite to replace it")
    if args.relabel_from:
        if args.relabel_from.resolve() == args.out_dir.resolve():
            parser.error("--relabel-from must differ from --out-dir")
        relabel_full(args.relabel_from, args.out_dir)
        return 0

    from datasets import load_dataset

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
