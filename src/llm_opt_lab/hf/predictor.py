"""Greedy generation with a Hugging Face model, one sample at a time."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from llm_opt_lab.config import ExperimentConfig
from llm_opt_lab.data import Sample
from llm_opt_lab.doc_types import DocType
from llm_opt_lab.hf.loading import DTYPES, is_text_only
from llm_opt_lab.hf.sft_data import apply_template, chat_messages
from llm_opt_lab.predictions import Prediction


class HFPredictor:
    """Works for base, base + adapter, merged and NF4 models alike."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: ExperimentConfig,
        doc_types: dict[str, DocType],
        data_dir: str | Path,
    ) -> None:
        self.model = model.eval()
        self.processor = processor
        self.config = config
        self.doc_types = doc_types
        self.data_dir = data_dir
        self.text_only = is_text_only(config)
        temperature = config.evaluation.temperature
        self.generate_kwargs: dict[str, Any] = {
            "max_new_tokens": config.evaluation.max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            self.generate_kwargs["temperature"] = temperature

    def build_inputs(self, sample: Sample) -> Any:
        """Tokenized prompt (and pixel values) on the model's device."""
        messages = chat_messages(
            sample, self.doc_types[sample.doc_type], self.data_dir, text_only=self.text_only
        )
        inputs = apply_template(
            self.processor,
            messages,
            add_generation_prompt=True,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        if self.text_only:  # a tokenizer's BatchEncoding has no dtype argument
            return inputs.to(self.model.device)
        # Floating inputs (pixel values) follow the model dtype; token ids stay integers.
        return inputs.to(self.model.device, dtype=DTYPES[self.config.model.dtype])

    @torch.inference_mode()
    def predict(self, sample: Sample) -> Prediction:
        inputs = self.build_inputs(sample)
        n_prompt = inputs["input_ids"].shape[1]
        start = time.perf_counter()
        try:
            output = self.model.generate(**inputs, **self.generate_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return Prediction(id=sample.id, error="CUDA out of memory", prompt_tokens=n_prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        new_tokens = output[0, n_prompt:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)
        return Prediction(
            id=sample.id,
            raw_output=text,
            latency_ms=latency_ms,
            prompt_tokens=n_prompt,
            completion_tokens=int(new_tokens.shape[0]),
        )
