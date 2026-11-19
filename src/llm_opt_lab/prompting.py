"""Build chat messages from a sample, identically for training, HF inference and vLLM.

Message order is fixed: system prompt first, then the document. The system prompt is
byte-identical for every request of a doc type, which is what lets vLLM prefix caching
share it across concurrent requests.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from llm_opt_lab.data import Sample, image_path
from llm_opt_lab.doc_types import DocType

Message = dict[str, Any]


def serialize_target(target: dict[str, Any]) -> str:
    """Canonical text form of a ground-truth JSON, used as the training label."""
    return json.dumps(target, ensure_ascii=False)


def build_messages(
    sample: Sample,
    doc_type: DocType,
    data_dir: str | Path,
    *,
    include_target: bool = False,
) -> list[Message]:
    """Messages in the Hugging Face multimodal chat format.

    Image parts carry an absolute file path under `"image"`; `to_openai_messages`
    converts them for the vLLM OpenAI-compatible API.
    """
    if sample.doc_type != doc_type.name:
        raise ValueError(f"sample {sample.id!r} is {sample.doc_type!r}, not {doc_type.name!r}")
    if sample.input_type == "image":
        document = {"type": "image", "image": str(image_path(sample, data_dir).resolve())}
    else:
        document = {"type": "text", "text": sample.spatial_text}
    messages: list[Message] = [
        {"role": "system", "content": [{"type": "text", "text": doc_type.system_prompt}]},
        {"role": "user", "content": [document]},
    ]
    if include_target:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": serialize_target(sample.target)}],
            }
        )
    return messages


def _image_data_url(path: str | Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def to_openai_messages(messages: list[Message]) -> list[Message]:
    """Convert `build_messages` output to the OpenAI chat format served by vLLM.

    Text-only content becomes a plain string; images become base64 data URLs.
    """
    converted: list[Message] = []
    for message in messages:
        parts = message["content"]
        if all(part["type"] == "text" for part in parts):
            content: Any = "".join(part["text"] for part in parts)
        else:
            content = [
                {"type": "image_url", "image_url": {"url": _image_data_url(part["image"])}}
                if part["type"] == "image"
                else {"type": "text", "text": part["text"]}
                for part in parts
            ]
        converted.append({"role": message["role"], "content": content})
    return converted
