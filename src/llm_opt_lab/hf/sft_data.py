"""Supervised fine-tuning features: chat template, image inputs, and label masking.

Only the assistant answer (the serialized target and the end-of-turn tokens after it)
is trained on; every prompt token gets label -100. Each sample asks for the fields of a
training request (field_requests.py): every annotated field, or a fresh random subset each
time the sample is drawn when `fields.sampling` is set.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from llm_opt_lab.config import FieldsConfig
from llm_opt_lab.data import Sample
from llm_opt_lab.doc_types import DocType
from llm_opt_lab.field_requests import Request, longest_training_request, training_request
from llm_opt_lab.prompting import Message, build_messages, to_openai_messages

IGNORE_INDEX = -100
log = logging.getLogger(__name__)


def chat_messages(
    sample: Sample, doc_type: DocType, data_dir: str | Path, *, text_only: bool, **kwargs: Any
) -> list[Message]:
    """`build_messages`, with content flattened to strings for text-only chat templates."""
    messages = build_messages(sample, doc_type, data_dir, **kwargs)
    if not text_only:
        return messages
    if sample.input_type != "spatial_text":
        raise ValueError(f"sample {sample.id!r}: a text-only model cannot take image input")
    return to_openai_messages(messages)


def apply_template(
    processor: Any,
    messages: list[Message],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
        **chat_template_kwargs,
    )


def build_features(
    sample: Sample,
    doc_type: DocType,
    processor: Any,
    data_dir: str | Path,
    *,
    text_only: bool,
    chat_template_kwargs: dict[str, Any],
    request: Request | None = None,
) -> dict[str, torch.Tensor]:
    """Model inputs plus `labels` for one sample, without a batch dimension on sequences.

    `request` defaults to the doc type's full inference request. The prompt (system +
    document + generation prompt) must be a token prefix of the full conversation;
    otherwise masking would be wrong and this raises.
    """
    common = {"text_only": text_only, "request": request}
    prompt = apply_template(
        processor,
        chat_messages(sample, doc_type, data_dir, **common),
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full = apply_template(
        processor,
        chat_messages(sample, doc_type, data_dir, include_target=True, **common),
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_ids, input_ids = prompt["input_ids"][0], full["input_ids"][0]
    n_prompt = prompt_ids.shape[0]
    if n_prompt >= input_ids.shape[0] or not torch.equal(input_ids[:n_prompt], prompt_ids):
        raise ValueError(
            f"sample {sample.id!r}: the generation prompt is not a prefix of the full "
            "conversation for this chat template, so the answer cannot be masked reliably"
        )
    labels = input_ids.clone()
    labels[:n_prompt] = IGNORE_INDEX

    length = input_ids.shape[0]
    features: dict[str, torch.Tensor] = {"labels": labels}
    for key, value in full.items():
        if not isinstance(value, torch.Tensor):
            continue
        is_sequence = value.dim() == 2 and value.shape == (1, length)
        features[key] = value[0] if is_sequence else value
    return features


def _is_sequence(value: torch.Tensor, length: int) -> bool:
    return value.dim() == 1 and value.shape[0] == length


class SFTDataset(torch.utils.data.Dataset):
    """Builds features on the fly; drops or rejects samples longer than `max_seq_length`.

    With `fields.sampling`, every draw of a sample asks for a new random request, seeded by
    (seed, sample id, draw number) so a run is reproducible. The length check uses the
    longest request a sample can draw.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        doc_types: dict[str, DocType],
        processor: Any,
        data_dir: str | Path,
        *,
        text_only: bool,
        chat_template_kwargs: dict[str, Any],
        max_seq_length: int,
        over_length: str = "drop",
        fields: FieldsConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.doc_types = doc_types
        self.processor = processor
        self.data_dir = data_dir
        self.text_only = text_only
        self.chat_template_kwargs = chat_template_kwargs
        self.fields = fields or FieldsConfig()
        self.seed = seed
        self.draws: Counter[str] = Counter()
        self.samples: list[Sample] = []
        self.dropped: list[str] = []
        for i, sample in enumerate(samples):
            doc_type = self.doc_types[sample.doc_type]
            longest = longest_training_request(sample.target, doc_type, self.fields)
            length = self._build(sample, longest)["input_ids"].shape[0]
            if length <= max_seq_length:
                self.samples.append(sample)
            elif over_length == "drop":
                self.dropped.append(sample.id)
            else:
                raise ValueError(f"sample {sample.id!r} has {length} tokens > {max_seq_length}")
            if (i + 1) % 500 == 0:
                log.info("length scan: %d / %d samples", i + 1, len(samples))
        if self.dropped:
            log.warning(
                "dropped %d over-length samples, e.g. %s", len(self.dropped), self.dropped[:5]
            )

    def _build(self, sample: Sample, request: Request) -> dict[str, torch.Tensor]:
        return build_features(
            sample,
            self.doc_types[sample.doc_type],
            self.processor,
            self.data_dir,
            text_only=self.text_only,
            chat_template_kwargs=self.chat_template_kwargs,
            request=request,
        )

    def request(self, sample: Sample) -> Request:
        """The request for the next draw of `sample`."""
        rng = None
        if self.fields.sampling is not None:
            rng = random.Random(f"{self.seed}:{sample.id}:{self.draws[sample.id]}")
            self.draws[sample.id] += 1
        return training_request(sample.target, self.doc_types[sample.doc_type], self.fields, rng)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return self._build(sample, self.request(sample))


class SFTCollator:
    """Right-pads sequence tensors; concatenates the rest (e.g. pixel_values) along dim 0."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_values = {"input_ids": pad_token_id, "labels": IGNORE_INDEX}

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = set(batch[0])
        if any(set(features) != keys for features in batch):
            raise ValueError(
                "a batch mixes samples with different inputs (e.g. image and spatial text); "
                "use batch_size 1"
            )
        lengths = [features["input_ids"].shape[0] for features in batch]
        longest = max(lengths)
        out: dict[str, torch.Tensor] = {}
        for key in sorted(keys):
            values = [features[key] for features in batch]
            if all(_is_sequence(v, n) for v, n in zip(values, lengths, strict=True)):
                pad = self.pad_values.get(key, 0)
                out[key] = torch.stack(
                    [
                        torch.nn.functional.pad(v, (0, longest - v.shape[0]), value=pad)
                        for v in values
                    ]
                )
            else:
                out[key] = torch.cat(values, dim=0)
        return out
