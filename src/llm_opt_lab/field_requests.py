"""Which fields a request asks for, and the target that answers it.

The system prompt lists the fields to extract (prompting.py); a `Request` is that list.

- Inference and evaluation ask for a fixed set per doc type (`fields.inference`), so the
  system prompt is byte-identical across requests and vLLM prefix caching keeps working.
- Training asks for every annotated field (one fixed prompt) or, with `fields.sampling`,
  for a random subset per sample plus decoys, so the model learns to read the field list
  instead of memorizing one output format.

A field is annotated in a record when its key is present in the target: `null` means
"checked, not on the document", a missing key means "not annotated". Only annotated fields
are trained on or scored. Decoys can never be on the document, so their answer is `null`.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from llm_opt_lab.config import FieldsConfig
from llm_opt_lab.doc_types import DocType, FieldSpec

LIST = "[]"


@dataclass(frozen=True)
class Request:
    fields: tuple[FieldSpec, ...]

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.fields]


def _tokens(path: str) -> list[str]:
    """`menu[].sub[].name` -> ["menu", "[]", "sub", "[]", "name"]."""
    tokens: list[str] = []
    for part in path.split("."):
        if part.endswith(LIST):
            tokens += [part[: -len(LIST)], LIST]
        else:
            tokens.append(part)
    return tokens


def _is_annotated(value: Any, tokens: list[str]) -> bool:
    if not tokens:
        return True
    head, rest = tokens[0], tokens[1:]
    if head == LIST:
        if value is None or value == []:
            return True  # no entries: every sub-field is answered by []
        return isinstance(value, list) and any(_is_annotated(item, rest) for item in value)
    return isinstance(value, dict) and head in value and _is_annotated(value[head], rest)


def annotated_paths(target: dict[str, Any], doc_type: DocType) -> set[str]:
    """Registry fields whose key is present in the target (in at least one list entry)."""
    return {f.path for f in doc_type.fields if _is_annotated(target, _tokens(f.path))}


def _tree(paths: Iterable[str]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path in paths:
        node = tree
        for token in _tokens(path):
            node = node.setdefault(token, {})
    return tree


def _project(value: Any, node: dict[str, Any]) -> Any:
    if not node:  # leaf
        return value
    if LIST in node:
        return [_project(item, node[LIST]) for item in (value if isinstance(value, list) else [])]
    source = value if isinstance(value, dict) else {}
    return {key: _project(source.get(key), sub) for key, sub in node.items()}


def project(record: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    """The part of `record` that answers the given fields, keys in request order.

    Missing leaves become null and missing lists []. Asking for `menu[].price` alone keeps
    every menu entry with only its price.
    """
    return _project(record, _tree(paths))


def skeleton(paths: Iterable[str]) -> dict[str, Any]:
    """The JSON structure shown in the prompt: leaves are "", lists show one entry."""

    def build(node: dict[str, Any]) -> Any:
        if not node:
            return ""
        if LIST in node:
            return [build(node[LIST])]
        return {key: build(sub) for key, sub in node.items()}

    return build(_tree(paths))


def holdout_paths(doc_type: DocType, config: FieldsConfig) -> set[str]:
    """Fields and decoys that training never asks for."""
    paths = set(config.holdout.get(doc_type.name, []))
    known = {f.path for f in doc_type.extract_fields} | {d.path for d in doc_type.decoys}
    unknown = sorted(paths - known)
    if unknown:
        raise ValueError(
            f"fields.holdout of {doc_type.name!r}: not extract fields or decoys: {unknown}"
        )
    return paths


def inference_request(doc_type: DocType, config: FieldsConfig) -> Request:
    """The fixed request of inference and evaluation, in registry order, decoys last."""
    holdout = holdout_paths(doc_type, config)
    fields = [
        f
        for f in doc_type.extract_fields
        if config.inference == "full" or (f.path in holdout) == (config.inference == "held_out")
    ]
    if not fields:
        raise ValueError(f"{doc_type.name!r}: fields.inference {config.inference!r} asks nothing")
    if config.decoys > len(doc_type.decoys):
        raise ValueError(f"{doc_type.name!r} has only {len(doc_type.decoys)} decoys")
    return Request(tuple(fields) + doc_type.decoys[: config.decoys])


def training_request(
    target: dict[str, Any],
    doc_type: DocType,
    config: FieldsConfig,
    rng: random.Random | None = None,
) -> Request:
    """Every annotated, non-held-out field; with `fields.sampling` and an `rng`, a random
    subset of them (size uniform in [min_fields, all]) plus, sometimes, decoys."""
    holdout = holdout_paths(doc_type, config)
    annotated = annotated_paths(target, doc_type)
    base = [f for f in doc_type.extract_fields if f.path in annotated and f.path not in holdout]
    if not base:
        raise ValueError(f"{doc_type.name!r}: the target annotates no field to train on")
    sampling = config.sampling
    if sampling is None or rng is None or rng.random() < sampling.full_rate:
        return Request(tuple(base))
    chosen = rng.sample(base, rng.randint(min(sampling.min_fields, len(base)), len(base)))
    if not sampling.shuffle:
        chosen.sort(key=base.index)
    decoys = _training_decoys(doc_type, holdout)
    if decoys and sampling.max_decoys and rng.random() < sampling.decoy_rate:
        for decoy in rng.sample(decoys, rng.randint(1, min(sampling.max_decoys, len(decoys)))):
            chosen.insert(rng.randint(0, len(chosen)) if sampling.shuffle else len(chosen), decoy)
    return Request(tuple(chosen))


def _training_decoys(doc_type: DocType, holdout: set[str]) -> list[FieldSpec]:
    return [d for d in doc_type.decoys if d.path not in holdout]


def longest_training_request(
    target: dict[str, Any], doc_type: DocType, config: FieldsConfig
) -> Request:
    """An upper bound on the length of any request `training_request` can draw."""
    base = training_request(target, doc_type, config)
    n_decoys = config.sampling.max_decoys if config.sampling else 0
    decoys = _training_decoys(doc_type, holdout_paths(doc_type, config))[:n_decoys]
    return Request(base.fields + tuple(decoys))


@dataclass(frozen=True)
class ScoredTarget:
    gold: dict[str, Any]
    # Requested fields the record does not annotate: neither scored nor counted as extra.
    skipped: frozenset[str]


def scored_target(target: dict[str, Any], doc_type: DocType, paths: Iterable[str]) -> ScoredTarget:
    """The ground truth for the requested fields; decoys are always null."""
    known = annotated_paths(target, doc_type) | {d.path for d in doc_type.decoys}
    paths = list(paths)
    return ScoredTarget(
        gold=project(target, [p for p in paths if p in known]),
        skipped=frozenset(p for p in paths if p not in known),
    )
