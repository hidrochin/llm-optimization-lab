"""Names of model parts outside the language model (vision, audio towers and projectors).

They never get LoRA unless `lora.train_vision_encoder` is set, and they stay in BF16
when quantizing. Confirm the real module names with `scripts/inspect_model.py`.
"""

from __future__ import annotations

import re

NON_LANGUAGE_MODULES = r".*(vision|audio|multi_modal_projector|embed_vision|embed_audio).*"
_NON_LANGUAGE = re.compile(NON_LANGUAGE_MODULES)


def is_non_language(name: str) -> bool:
    return _NON_LANGUAGE.fullmatch(name) is not None
