import base64
import json

from conftest import DATA_DIR
from llm_opt_lab.prompting import build_messages, serialize_target, to_openai_messages


def test_build_messages_spatial_text(samples, doc_types):
    sample = samples[1]
    messages = build_messages(sample, doc_types["receipt"], DATA_DIR, include_target=True)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"][0]["text"] == doc_types["receipt"].system_prompt
    assert messages[1]["content"] == [{"type": "text", "text": sample.spatial_text}]
    assert json.loads(messages[2]["content"][0]["text"]) == sample.target


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
