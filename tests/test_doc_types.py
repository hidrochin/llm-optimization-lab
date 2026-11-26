import json

import pytest

from conftest import DOC_TYPES_DIR
from llm_opt_lab.doc_types import load_doc_type, load_doc_types


def test_field_types_from_schema(doc_types):
    receipt = doc_types["receipt"]
    assert receipt.field_types == {
        "store.name": "text",
        "store.address": "text",
        "date": "date",
        "items[].name": "text",
        "items[].quantity": "number",
        "items[].price": "number",
        "total": "number",
        "loyalty_points": "text",
    }
    assert receipt.field_type("not.in.schema") == "text"
    assert receipt.instructions.startswith("You extract information")


def test_registry_keeps_order_descriptions_modes_and_decoys(doc_types):
    receipt = doc_types["receipt"]
    assert [f.path for f in receipt.fields][:3] == ["store.name", "store.address", "date"]
    date = receipt.fields[2]
    assert (date.description, date.examples) == ("Purchase date.", ("05/01/2026",))
    assert "loyalty_points" not in [f.path for f in receipt.extract_fields]
    assert [d.path for d in receipt.decoys] == ["container_number", "vessel_name"]
    assert all(d.decoy for d in receipt.decoys)


def test_missing_doc_type_is_an_error():
    with pytest.raises(ValueError, match="invoice"):
        load_doc_types(DOC_TYPES_DIR, ["receipt", "invoice"])


def test_old_system_prompt_file_gets_a_migration_hint(tmp_path):
    (tmp_path / "schema.json").write_text(json.dumps({"type": "object", "properties": {}}))
    (tmp_path / "system_prompt.md").write_text("Return these keys: ...")
    with pytest.raises(ValueError, match="instructions.md"):
        load_doc_type(tmp_path)


def test_decoy_cannot_shadow_a_field(tmp_path):
    schema = {
        "type": "object",
        "properties": {"total": {"type": "string"}},
        "x-decoys": [{"name": "total"}],
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))
    (tmp_path / "instructions.md").write_text("A receipt.")
    with pytest.raises(ValueError, match="decoy"):
        load_doc_type(tmp_path)


def test_dev_doc_types_load():
    doc_types = load_doc_types(DOC_TYPES_DIR.parents[2] / "doc_types")
    full = doc_types["cord_receipt_full"]
    assert len(full.extract_fields) == 27
    assert {f.path for f in doc_types["cord_receipt"].fields} <= {f.path for f in full.fields}
