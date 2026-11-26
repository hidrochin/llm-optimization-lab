"""Experiment config schema and loader.

Every experiment is described by one YAML file under `configs/`. Unknown keys are
rejected, so a typo fails at load time instead of being silently ignored mid-run.

A config may start with `extends: <path>` (relative to the file) to inherit another
config and override only some keys. This is how experiments change one variable
against the reference config.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

InputType = Literal["image", "spatial_text"]
INPUT_TYPES: tuple[InputType, ...] = ("image", "spatial_text")
Split = Literal["train", "validation", "test", "calibration", "benchmark"]
DType = Literal["bfloat16", "float16", "float32"]

_UNRESOLVED_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def expand_path(value: str | Path) -> Path:
    """Expand `~` and `${ENV_VAR}` in a path; fail if a variable is not set.

    Data and artifacts live in different places on different servers, so configs
    refer to them through env vars such as `${LLMOPT_DATA_DIR}`.
    """
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if _UNRESOLVED_VAR.search(expanded):
        raise ValueError(f"unresolved environment variable in path: {value!r}")
    return Path(expanded)


class ModelConfig(StrictModel):
    id: str
    revision: str | None = None
    dtype: DType = "bfloat16"
    # Gemma 4, Qwen3.5 and Qwen3.6 are image-text-to-text; causal_lm is for text-only models.
    architecture: Literal["image_text_to_text", "causal_lm"] = "image_text_to_text"
    attn_implementation: str | None = None
    # Passed to the chat template, e.g. {enable_thinking: false} for Qwen3.
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class DataConfig(StrictModel):
    dir: Path = Field(
        description="Dataset root: split .jsonl files and image paths are relative to it"
    )
    doc_types_dir: Path = Path("doc_types")
    doc_types: list[str] = Field(default_factory=list, description="Empty means every doc type")
    input_types: list[InputType] = Field(default_factory=lambda: list(INPUT_TYPES))

    @field_validator("dir", "doc_types_dir", mode="before")
    @classmethod
    def _expand(cls, value: Any) -> Path:
        return expand_path(value)


class FieldSampling(StrictModel):
    """Training requests that ask for a random subset of each record's annotated fields, so
    the model learns to read the field list instead of one fixed output (field_requests.py)."""

    full_rate: float = Field(
        default=0.4, ge=0, le=1, description="Share of samples asking for every field"
    )
    min_fields: int = Field(default=1, gt=0)
    decoy_rate: float = Field(
        default=0.3, ge=0, le=1, description="Share of samples that add decoy fields"
    )
    max_decoys: int = Field(default=2, ge=0)
    shuffle: bool = Field(default=True, description="Random field order in sampled requests")


class FieldsConfig(StrictModel):
    """Which fields the system prompt asks for (docs/formats.md §3)."""

    # Doc type -> fields (generic paths) or decoys that training never asks for: the "new"
    # fields of the field-holdout experiment (E20). Evaluation can still ask for them.
    holdout: dict[str, list[str]] = Field(default_factory=dict)
    # Inference and evaluation ask for every extract field (full), all but the holdout
    # (seen), or only the holdout (held_out).
    inference: Literal["full", "seen", "held_out"] = "full"
    # Inference requests end with the doc type's first N decoys (in registry order).
    decoys: int = Field(default=0, ge=0, description="Decoy fields added to inference requests")
    # None: every training request asks for the same fields (one fixed prompt per doc type).
    sampling: FieldSampling | None = None

    @model_validator(mode="after")
    def _check_holdout(self) -> FieldsConfig:
        if self.inference != "full" and not any(self.holdout.values()):
            raise ValueError(f"fields.inference {self.inference!r} needs fields.holdout")
        return self


class MethodConfig(StrictModel):
    name: Literal["none", "lora", "qlora"] = "none"
    quantization: Literal["nf4", "fp4"] | None = None
    double_quant: bool = False
    compute_dtype: DType = "bfloat16"

    @model_validator(mode="after")
    def _check_quantization(self) -> MethodConfig:
        if self.name == "qlora" and self.quantization is None:
            raise ValueError("method.quantization is required when method.name is 'qlora'")
        if self.name != "qlora" and self.quantization is not None:
            raise ValueError("method.quantization is only valid when method.name is 'qlora'")
        return self


class LoraConfig(StrictModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    # A list of module name suffixes, or one regex string (PEFT semantics).
    target_modules: list[str] | str = Field(default_factory=list)
    # PEFT `target_parameters`, for MoE experts stored as fused nn.Parameter (E10).
    target_parameters: list[str] = Field(default_factory=list)
    # Regex of modules that never get LoRA; defaults to the non-language towers.
    exclude_modules: str | None = None
    train_vision_encoder: bool = False


class TrainingConfig(StrictModel):
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    epochs: float = Field(default=1, gt=0)
    max_steps: int | None = Field(default=None, gt=0, description="Overrides epochs (smoke tests)")
    max_seq_length: int = Field(gt=0)
    gradient_checkpointing: bool = True
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    lr_scheduler: str = "cosine"
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=10, gt=0)
    eval_steps: int | None = Field(default=None, gt=0, description="None: no eval during training")
    eval_samples: int = Field(default=200, gt=0, description="Validation subset used for eval loss")
    save_steps: int | None = Field(default=None, gt=0, description="None: save only at the end")
    max_train_samples: int | None = Field(default=None, gt=0, description="Smoke tests only")
    # label_positions: apply lm_head only where labels != -100 (Risk 1); full: standard loss.
    loss: Literal["label_positions", "full"] = "label_positions"
    over_length: Literal["drop", "error"] = "drop"


class QuantizationConfig(StrictModel):
    # base | merged | quantized | merged:<run_id> | <local path or hub id>; see paths.py
    source: str = "merged"
    scheme: Literal["W4A16"] = "W4A16"
    algorithm: Literal["gptq", "awq"] = "gptq"
    # Modules left in BF16 (LLM Compressor syntax, `re:` for regex): lm_head, the non-language
    # towers (Qwen names its vision tower `visual`), the Gemma 4 E-series per-layer embedding
    # projection, which sits outside the decoder layers and gets no calibration data in the
    # sequential pipeline, and the MoE router gates of Qwen3.5/3.6. Qwen's own GPTQ checkpoints
    # also keep every `attn` module (including Gated DeltaNet `linear_attn`) in BF16; configs
    # for those models add that (T-D09).
    ignore: list[str] = Field(
        default_factory=lambda: [
            "re:.*lm_head",
            "re:.*vision.*",
            "re:.*visual.*",
            "re:.*audio.*",
            "re:.*multi_modal_projector.*",
            "re:.*per_layer_model_projection",
            "re:.*mlp.gate$",
            "re:.*shared_expert_gate$",
        ]
    )
    # sequential: one decoder layer on the GPU at a time (needed for large models), but the
    # traced graph needs the same inputs for every sample, so it cannot mix image and
    # spatial-text samples. basic: whole-model forward passes, any mix of inputs.
    pipeline: Literal["sequential", "basic"] = "basic"
    # Decoder layer class names for the sequential pipeline, e.g. ["Gemma4TextDecoderLayer"].
    sequential_targets: list[str] | None = None
    calibration_split: Split = "calibration"
    # Input types used for calibration; None means data.input_types.
    calibration_input_types: list[InputType] | None = None
    # full: every calibration sample asks for the inference fields (one prompt per doc type);
    # sampled: requests drawn as in training (needs fields.sampling), so the quantized model
    # is calibrated on varied field lists too.
    calibration_requests: Literal["full", "sampled"] = "full"
    num_calibration_samples: int = Field(default=128, gt=0)
    max_seq_length: int = Field(default=32768, gt=0)


class ServingConfig(StrictModel):
    # base | merged | quantized | <kind>:<run_id> | <local path or hub id>; see paths.py
    source: str = "base"
    host: str = "127.0.0.1"
    port: int = 8000
    max_model_len: int = Field(gt=0)
    max_num_seqs: int = Field(default=64, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    enable_prefix_caching: bool = True
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    max_images_per_prompt: int = Field(default=1, ge=0)
    extra_args: list[str] = Field(default_factory=list)
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 8, 16, 32, 60])
    warmup_requests: int = Field(default=4, ge=0)
    request_timeout_s: float = Field(default=900, gt=0)

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


class RewardWeights(StrictModel):
    """Weight of each reward component (see `rewards.py`). A weight of 0 keeps the component
    in the training log without optimizing it."""

    field_accuracy: float = Field(default=1.0, ge=0)
    exact_match: float = Field(default=1.0, ge=0)
    similarity: float = Field(default=0.0, ge=0)
    parse: float = Field(default=0.0, ge=0)
    value_recall: float = Field(default=0.0, ge=0)
    null_accuracy: float = Field(default=0.0, ge=0)


class RLConfig(StrictModel):
    """RL fine-tuning with GRPO on a LoRA adapter (`scripts/train_grpo.py`).

    Learning rate, schedule and batch sizes come from `training`: `batch_size` completions
    per forward pass, `gradient_accumulation_steps` forward passes per optimizer step. One
    generation batch holds the `num_generations` completions of a single prompt, so image
    and spatial-text prompts are never mixed in a batch.
    """

    algorithm: Literal["grpo"] = "grpo"
    # `adapter:<run id>` keeps training that run's SFT adapter, so the result is again an
    # adapter on `model.id`. None trains a new LoRA from the `lora` section.
    init_adapter: str | None = None
    num_generations: int = Field(default=8, ge=2)
    max_completion_length: int | None = Field(
        default=None, gt=0, description="None: evaluation.max_new_tokens"
    )
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    beta: float = Field(default=0.0, ge=0, description="KL coefficient against the start policy")
    loss_type: Literal["dapo", "dr_grpo", "grpo", "bnpo"] = "dapo"
    scale_rewards: Literal["group", "batch", "none"] = "group"
    mask_truncated_completions: bool = False
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    # vLLM generation needs vLLM in the same environment (the RL environment, never `train`).
    use_vllm: bool = False
    vllm_mode: Literal["colocate", "server"] = "colocate"
    vllm_gpu_memory_utilization: float = Field(default=0.3, gt=0, le=1)

    @field_validator("init_adapter")
    @classmethod
    def _adapter_source(cls, value: str | None) -> str | None:
        if value is not None and not (value.startswith("adapter:") and len(value) > 8):
            raise ValueError("rl.init_adapter must be 'adapter:<run id>'")
        return value


class QualityTarget(StrictModel):
    exact_match: float = Field(default=0.95, ge=0, le=1)
    field_accuracy: float = Field(default=0.95, ge=0, le=1)


class EvaluationConfig(StrictModel):
    split: Split = "test"
    quality_target: QualityTarget = Field(default_factory=QualityTarget)
    batch_size: int = Field(default=1, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    max_new_tokens: int = Field(default=2048, gt=0)


class ExperimentConfig(StrictModel):
    experiment_id: str
    run_name: str
    seed: int = 42
    model: ModelConfig
    data: DataConfig
    fields: FieldsConfig = Field(default_factory=FieldsConfig)
    method: MethodConfig = Field(default_factory=MethodConfig)
    lora: LoraConfig | None = None
    training: TrainingConfig | None = None
    quantization: QuantizationConfig | None = None
    serving: ServingConfig | None = None
    rl: RLConfig | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def run_id(self) -> str:
        return f"{self.experiment_id}_{self.run_name}"

    @model_validator(mode="after")
    def _check_sections(self) -> ExperimentConfig:
        q = self.quantization
        if q is not None and q.calibration_requests == "sampled" and self.fields.sampling is None:
            raise ValueError("quantization.calibration_requests 'sampled' needs fields.sampling")
        if self.method.name in ("lora", "qlora"):
            missing = [name for name in ("lora", "training") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"method '{self.method.name}' requires sections: {missing}")
        if self.rl is not None:
            if self.method.name not in ("lora", "qlora"):
                raise ValueError("rl trains a LoRA adapter: method.name must be lora or qlora")
            per_step = self.training.batch_size * self.training.gradient_accumulation_steps
            if self.rl.num_generations % self.training.batch_size:
                raise ValueError("rl.num_generations must be a multiple of training.batch_size")
            if per_step % self.rl.num_generations:
                raise ValueError(
                    "training.batch_size * gradient_accumulation_steps must be a multiple of "
                    "rl.num_generations, so every optimizer step holds whole prompt groups"
                )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mappings merge recursively; lists and scalars in `override` replace `base`."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config and resolve its `extends` chain, without validating."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"circular extends: {' -> '.join(str(p) for p in (*_seen, path))}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _deep_merge(load_raw_config(path.parent / parent, (*_seen, path)), raw)


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_raw_config(path))
