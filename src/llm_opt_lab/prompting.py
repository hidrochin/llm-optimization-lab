"""Build chat messages from a sample, identically for training, HF inference and vLLM.

Message order is fixed: system prompt first, then the document. The system prompt is the
doc type's instructions plus the requested fields (structure, descriptions, examples) and
the shared output rules. It depends only on the doc type and the request, so a fixed
inference request keeps it byte-identical across requests, which is what lets vLLM prefix
caching share it.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from llm_opt_lab.config import FieldsConfig
from llm_opt_lab.data import Sample, image_path
from llm_opt_lab.doc_types import DocType, FieldSpec
from llm_opt_lab.field_requests import Request, inference_request, project, skeleton

Message = dict[str, Any]

# Raw data only: values as printed. Normalization (dates, numbers, codes) and business
# rules run in code after extraction (postprocess.py, T-L05).
OUTPUT_RULES = """Rules:
- Copy every value exactly as printed on the document, as a string.
- Use null for a field that is not on the document, and [] for a list with no entries. \
Keep every key of the structure.
- Return only the JSON object: no other keys, no comments, no text around it."""


def serialize_target(target: dict[str, Any]) -> str:
    """Canonical text form of a ground-truth JSON, used as the training label."""
    return json.dumps(target, ensure_ascii=False)


def _field_line(spec: FieldSpec) -> str:
    line = f"- {spec.path}"
    if spec.description:
        line += f": {spec.description}"
    if spec.examples:
        line += " Examples: " + ", ".join(json.dumps(e, ensure_ascii=False) for e in spec.examples)
    return line


def render_system_prompt(doc_type: DocType, request: Request) -> str:
    structure = json.dumps(skeleton(request.paths), ensure_ascii=False)
    fields = "\n".join(_field_line(spec) for spec in request.fields)
    return (
        f"{doc_type.instructions}\n\n"
        f"Extract the fields below and return exactly one JSON object with this structure:\n"
        f"{structure}\n\nFields:\n{fields}\n\n{OUTPUT_RULES}"
    )


def build_messages(
    sample: Sample,
    doc_type: DocType,
    data_dir: str | Path,
    *,
    request: Request | None = None,
    include_target: bool = False,
) -> list[Message]:
    """Messages in the Hugging Face multimodal chat format.

    `request` defaults to the doc type's full inference request. With `include_target`,
    the assistant turn is the target restricted to the requested fields.

    Image parts carry an absolute file path under `"image"`; `to_openai_messages`
    converts them for the vLLM OpenAI-compatible API.
    """
    if sample.doc_type != doc_type.name:
        raise ValueError(f"sample {sample.id!r} is {sample.doc_type!r}, not {doc_type.name!r}")
    request = request or inference_request(doc_type, FieldsConfig())
    if sample.input_type == "image":
        document = {"type": "image", "image": str(image_path(sample, data_dir).resolve())}
    else:
        document = {"type": "text", "text": sample.spatial_text}
    messages: list[Message] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": render_system_prompt(doc_type, request)}],
        },
        {"role": "user", "content": [document]},
    ]
    if include_target:
        answer = serialize_target(project(sample.target, request.paths))
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
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
