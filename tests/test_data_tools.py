"""Dataset tools: spatial text, CORD conversion, benchmark sampling, validation."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

from conftest import DATA_DIR
from llm_opt_lab.data import Sample
from llm_opt_lab.doc_types import load_doc_types
from llm_opt_lab.spatial_text import Word, group_lines, render_spatial_text
from llm_opt_lab.split import sample_benchmark

ROOT = Path(__file__).parents[1]


def load_script(relative: str):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spatial_text_keeps_rows_and_columns():
    words = [
        Word("TOTAL", 10, 100, 60, 112),
        Word("Coffee", 10, 50, 70, 62),
        Word("40,000", 200, 52, 260, 63),
        Word("65,000", 200, 101, 260, 111),
    ]
    assert [[w.text for w in line] for line in group_lines(words)] == [
        ["Coffee", "40,000"],
        ["TOTAL", "65,000"],
    ]
    lines = render_spatial_text(words).splitlines()
    assert lines[0].startswith("Coffee") and lines[1].startswith("TOTAL")
    assert lines[0].index("40,000") == lines[1].index("65,000")


def test_cord_conversion():
    cord = load_script("scripts/data/convert_cord.py")
    target = cord.cord_target(
        {
            "menu": {"nm": "Ice tea", "cnt": "2", "price": "20,000"},
            "total": {"total_price": "20,000", "cashprice": ["50,000"]},
        }
    )
    assert target["menu"] == [
        {"name": "Ice tea", "count": "2", "unit_price": None, "price": "20,000"}
    ]
    assert target["subtotal"] == dict.fromkeys(cord.SUBTOTAL_FIELDS.values())
    assert target["total"]["cash_price"] == "50,000"
    twice = cord.cord_target(
        {"sub_total": [{"subtotal_price": "18,182", "tax_price": "1,546"}, {"subtotal_price": "1"}]}
    )
    assert twice["subtotal"]["subtotal_price"] == "18,182"
    assert twice["subtotal"]["tax_price"] == "1,546"
    schema_dir = ROOT / "doc_types"
    validate = load_script("scripts/data/validate.py")
    sample = Sample(
        id="c", doc_type="cord_receipt", input_type="spatial_text", spatial_text="x", target=target
    )
    assert validate.validate_samples([sample], load_doc_types(schema_dir), DATA_DIR) == []

    quad = {"x1": 0, "y1": 0, "x2": 50, "y2": 0, "x3": 50, "y3": 10, "x4": 0, "y4": 10}
    words = cord.cord_words([{"words": [{"text": "Ice", "quad": quad}]}])
    assert words == [Word("Ice", 0, 0, 50, 10)]


def test_cord_full_target_keeps_every_field():
    cord = load_script("scripts/data/convert_cord.py")
    target = cord.cord_target_full(
        {
            "menu": [
                {"nm": "Tea", "price": "5", "sub": {"nm": "+Ice", "price": "0"}},
                {"nm": "Bun", "unitprice": "3", "sub": [{"nm": "a"}, {"nm": "b"}]},
            ],
            "total": {"total_price": "8", "menuqty_cnt": "2", "emoneyprice": "8"},
        }
    )
    assert [m["sub"] for m in target["menu"]][0] == [
        {"name": "+Ice", "count": None, "unit_price": None, "price": "0"}
    ]
    assert [s["name"] for s in target["menu"][1]["sub"]] == ["a", "b"]
    assert target["total"]["menu_quantity_count"] == "2"
    assert target["total"]["emoney_price"] == "8"
    sample = Sample(
        id="c", doc_type="cord_receipt_full", input_type="spatial_text", spatial_text="x",
        target=target,
    )  # fmt: skip
    validate = load_script("scripts/data/validate.py")
    assert validate.validate_samples([sample], load_doc_types(ROOT / "doc_types"), DATA_DIR) == []


def test_validate_catches_derive_fields_uneven_lists_and_unreviewed_labels(doc_types):
    validate = load_script("scripts/data/validate.py")
    target = {
        "store": {"name": "An", "address": None},
        "date": None,
        "items": [{"name": "x", "quantity": 1, "price": 1}, {"name": "y", "quantity": 1}],
        "total": 1,
        "loyalty_points": 3,
    }
    sample = Sample(
        id="s", doc_type="receipt", input_type="spatial_text", spatial_text="x", target=target,
        label_source="teacher",
    )  # fmt: skip
    errors = validate.validate_samples([sample], doc_types, DATA_DIR, reviewed_only=True)
    assert any("derive fields" in e and "loyalty_points" in e for e in errors)
    assert any("target/items annotate different keys" in e for e in errors)
    assert any("not reviewed" in e for e in errors)


def test_validate_reports_schema_errors(samples, doc_types):
    validate = load_script("scripts/data/validate.py")
    bad = samples[2].model_copy(update={"target": {"store": "not an object"}})
    errors = validate.validate_samples([samples[1], bad], doc_types, DATA_DIR)
    assert any("receipt-002-txt: target/store" in e for e in errors)
    assert any("'date' is a required property" in e for e in errors)
    # Image files are checked too: the fixture image does not exist
    assert validate.validate_samples([samples[0]], doc_types, DATA_DIR) == [
        "receipt-001-img: image not found: images/receipt-001.png"
    ]


def test_sample_benchmark_is_stratified_and_deterministic():
    samples = [
        Sample(
            id=f"{d}-{t}-{i}", doc_type=d, input_type=t, spatial_text="x", image="x.png", target={}
        )
        for d, n in (("invoice", 60), ("receipt", 40))
        for t in ("image", "spatial_text")
        for i in range(n)
    ]
    chosen = sample_benchmark(samples, 50, seed=1)
    assert chosen == sample_benchmark(samples, 50, seed=1)
    assert len(chosen) == 50 and len({s.id for s in chosen}) == 50
    assert sum(s.doc_type == "invoice" for s in chosen) == 30
    with pytest.raises(ValueError):
        sample_benchmark(samples, 0)


def write_safetensors(path: Path, names: list[str]) -> None:
    header = {
        n: {"dtype": "I32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
        for i, n in enumerate(names)
    }
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4 * len(names))


def test_verify_quantized(tmp_path):
    verify = load_script("quantization/verify_quantized.py")
    quant = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ["lm_head"],
    }
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant}))
    write_safetensors(
        tmp_path / "model.safetensors",
        [
            "model.language_model.layers.0.self_attn.q_proj.weight_packed",
            "model.language_model.layers.0.self_attn.q_proj.weight_scale",
            "model.vision_tower.encoder.layers.0.attn.q_proj.weight",
        ],
    )
    report = verify.checkpoint_report(tmp_path)
    assert report["problems"] == []
    assert report["packed_by_component"] == {"model.language_model.layers": 1}

    write_safetensors(tmp_path / "model.safetensors", ["model.vision_tower.x.weight_packed"])
    problems = verify.checkpoint_report(tmp_path)["problems"]
    assert len(problems) == 1 and "vision/audio" in problems[0]
