import base64
import json

from conftest import DATA_DIR
from llm_opt_lab.config import FieldsConfig
from llm_opt_lab.field_requests import Request, inference_request
from llm_opt_lab.prompting import (
    build_messages,
    render_system_prompt,
    serialize_target,
    to_openai_messages,
)


def test_build_messages_spatial_text(samples, doc_types):
    sample, receipt = samples[1], doc_types["receipt"]
    messages = build_messages(sample, receipt, DATA_DIR, include_target=True)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    full = inference_request(receipt, FieldsConfig())
    assert messages[0]["content"][0]["text"] == render_system_prompt(receipt, full)
    assert messages[1]["content"] == [{"type": "text", "text": sample.spatial_text}]
    assert json.loads(messages[2]["content"][0]["text"]) == sample.target


def test_system_prompt_lists_the_requested_fields_only(doc_types):
    receipt = doc_types["receipt"]
    fields = {f.path: f for f in receipt.fields}
    request = Request((fields["total"], fields["items[].price"], receipt.decoys[0]))
    prompt = render_system_prompt(receipt, request)
    assert prompt.startswith(receipt.instructions)
    structure = '{"total": "", "items": [{"price": ""}], "container_number": ""}'
    assert structure in prompt
    assert "- total: Amount paid." in prompt
    assert "- container_number: Number of the shipping container." in prompt
    assert "store.name" not in prompt and "loyalty_points" not in prompt
    assert '- date: Purchase date. Examples: "05/01/2026"' in render_system_prompt(
        receipt, Request((fields["date"],))
    )


def test_system_prompt_is_identical_across_samples(samples, doc_types):
    # Prefix caching needs the inference prompt to depend on the doc type only.
    prompts = {
        build_messages(s, doc_types["receipt"], DATA_DIR)[0]["content"][0]["text"] for s in samples
    }
    assert len(prompts) == 1


def test_answer_follows_the_request(samples, doc_types):
    receipt = doc_types["receipt"]
    fields = {f.path: f for f in receipt.fields}
    request = Request((fields["items[].price"], fields["store.name"], receipt.decoys[1]))
    messages = build_messages(samples[0], receipt, DATA_DIR, request=request, include_target=True)
    answer = messages[2]["content"][0]["text"]
    assert answer == (
        '{"items": [{"price": 40000}, {"price": 25000}], '
        '"store": {"name": "Cửa hàng An"}, "vessel_name": null}'
    )


def test_build_messages_image_uses_absolute_path(samples, doc_types):
    messages = build_messages(samples[0], doc_types["receipt"], DATA_DIR)
    part = messages[1]["content"][0]
    assert part["type"] == "image"
    assert part["image"] == str((DATA_DIR / "images/receipt-001.png").resolve())
    assert len(messages) == 2


def test_serialize_target_keeps_unicode():
    assert serialize_target({"name": "Cà phê"}) == '{"name": "Cà phê"}'


def test_to_openai_messages(tmp_path):
    image = tmp_path / "doc.png"
    image.write_bytes(b"\x89PNG fake")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "prompt"}]},
        {"role": "user", "content": [{"type": "image", "image": str(image)}]},
    ]
    system, user = to_openai_messages(messages)
    assert system == {"role": "system", "content": "prompt"}
    url = user["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG fake"
