"""Rewards for RL fine-tuning (GRPO), computed with the evaluation scorer.

A completion is parsed, normalized and scored exactly as `scripts/evaluate.py` scores a
prediction, so RL optimizes the metrics we report. Every component is in [0, 1]:

- `field_accuracy`: share of the target's fields predicted correctly (0 when unparsed)
- `exact_match`: 1 when every field is correct and no extra field is predicted
- `similarity`: 1 - mean NED over the target's fields (partial credit for near misses)
- `parse`: 1 when the output parses to a JSON object
- `value_recall`: share of fields with a value that are not answered null. Rare fields
  are null on most documents, so field accuracy alone rewards answering null everywhere
- `null_accuracy`: share of null fields (decoys included) answered null, i.e. no invented
  values

The target is the ground truth restricted to the fields the prompt asked for
(`field_requests.project`). The weights live in the config (`rl.reward_weights`). GRPO
compares completions of the same prompt with each other, so a component only matters when
it differs between them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from llm_opt_lab.doc_types import DocType
from llm_opt_lab.metrics import SampleScore, score_sample
from llm_opt_lab.parsing import parse_json_output
from llm_opt_lab.postprocess import normalize_record

REWARD_COMPONENTS = (
    "field_accuracy",
    "exact_match",
    "similarity",
    "parse",
    "value_recall",
    "null_accuracy",
)

RewardFunc = Callable[..., list[float]]


def score_completion(text: str, target: dict[str, Any], doc_type: DocType) -> SampleScore:
    parsed, _ = parse_json_output(text)
    gold = normalize_record(target, doc_type)
    pred = None if parsed is None else normalize_record(parsed, doc_type)
    return score_sample("", doc_type.name, "", gold, pred)


def _share(flags: list[bool], parsed: bool) -> float:
    """Share of True flags; 1 when there is nothing to get wrong, 0 when unparsed."""
    if not parsed:
        return 0.0
    return sum(flags) / len(flags) if flags else 1.0


def reward_components(score: SampleScore) -> dict[str, float]:
    n = len(score.fields)
    return {
        "field_accuracy": sum(f.correct for f in score.fields) / n if n else 0.0,
        "exact_match": float(score.exact_match),
        "similarity": 1.0 - sum(f.ned for f in score.fields) / n if n else 0.0,
        "parse": float(score.parsed),
        "value_recall": _share(
            [f.pred is not None for f in score.fields if f.gold is not None], score.parsed
        ),
        "null_accuracy": _share(
            [f.pred is None for f in score.fields if f.gold is None], score.parsed
        ),
    }


def completion_text(completion: Any) -> str:
    """Text of a TRL completion: a string, or a conversation ending with the assistant turn."""
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text") or "" for part in content if part.get("type") == "text")


def make_reward_funcs(doc_types: dict[str, DocType]) -> list[RewardFunc]:
    """One TRL reward function per component, in `REWARD_COMPONENTS` order.

    The training dataset must carry `target_json` (the target of the prompt's request, as
    JSON text) and `doc_type` columns; TRL passes every dataset column to reward functions
    as keyword lists. The score of a completion is computed once and shared by every
    component.
    """

    @lru_cache(maxsize=4096)
    def components(text: str, target_json: str, doc_type: str) -> dict[str, float]:
        score = score_completion(text, json.loads(target_json), doc_types[doc_type])
        return reward_components(score)

    def make(name: str) -> RewardFunc:
        def reward(
            completions: list[Any], target_json: list[str], doc_type: list[str], **_: Any
        ) -> list[float]:
            return [
                components(completion_text(c), t, d)[name]
                for c, t, d in zip(completions, target_json, doc_type, strict=True)
            ]

        reward.__name__ = name  # TRL logs each reward as rewards/<name>/mean
        return reward

    return [make(name) for name in REWARD_COMPONENTS]
