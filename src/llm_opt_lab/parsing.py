"""Turn raw model output into a JSON object, or explain why it could not."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_output(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Return `(obj, None)` on success or `(None, reason)` on failure.

    Tolerates a Qwen3-style `<think>` block, a Markdown code fence, and text around
    the outermost `{...}`. Anything that is not a JSON object counts as a failure.
    """
    if text is None:
        return None, "no output"
    cleaned = _THINK_BLOCK.sub("", text).strip()
    fence = _CODE_FENCE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    if not cleaned:
        return None, "empty output"
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start and (start, end) != (0, len(cleaned) - 1):
        candidates.append(cleaned[start : end + 1])

    error = "invalid JSON"
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as e:
            error = f"invalid JSON: {e.msg} at char {e.pos}"
            continue
        if isinstance(obj, dict):
            return obj, None
        error = f"expected a JSON object, got {type(obj).__name__}"
    return None, error
