"""Deterministic train / validation / test / calibration split.

Stratified by doc type, so every split keeps the doc type mix. Whole groups (all
records of one source document) go to the same split, so the image and the spatial
text of a document never end up on both sides of train/test.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from llm_opt_lab.data import Sample

DEFAULT_RATIOS = {"train": 0.8, "validation": 0.05, "test": 0.1, "calibration": 0.05}


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Split `n` items by `ratios` with the largest-remainder method (counts sum to n)."""
    raw = {name: n * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    by_remainder = sorted(ratios, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in by_remainder[: n - sum(counts.values())]:
        counts[name] += 1
    return counts


def split_samples(
    samples: list[Sample],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, list[Sample]]:
    ratios = ratios or DEFAULT_RATIOS
    if any(r < 0 for r in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"ratios must be non-negative and sum to 1, got {ratios}")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)

    groups_by_doc_type: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        doc_types = {s.doc_type for s in members}
        if len(doc_types) > 1:
            raise ValueError(f"group {group_id!r} mixes doc types: {sorted(doc_types)}")
        groups_by_doc_type[doc_types.pop()].append(group_id)

    splits: dict[str, list[Sample]] = {name: [] for name in ratios}
    for doc_type in sorted(groups_by_doc_type):
        group_ids = sorted(groups_by_doc_type[doc_type])
        # A string seed is hashed with SHA-512, so this is stable across processes.
        random.Random(f"{seed}:{doc_type}").shuffle(group_ids)
        start = 0
        for name, count in _allocate(len(group_ids), ratios).items():
            for group_id in group_ids[start : start + count]:
                splits[name].extend(groups[group_id])
            start += count
    return splits


def sample_benchmark(samples: list[Sample], n: int, seed: int = 42) -> list[Sample]:
    """Draw `n` samples stratified by (doc type, input type), keeping each stratum's share.

    Within a stratum the draw is uniform, so the real length distribution is preserved.
    """
    if not 0 < n <= len(samples):
        raise ValueError(f"n must be in [1, {len(samples)}], got {n}")
    strata: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        strata[f"{sample.doc_type}/{sample.input_type}"].append(sample)
    counts = _allocate(n, {key: len(group) / len(samples) for key, group in strata.items()})
    chosen: list[Sample] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda s: s.id)
        chosen.extend(random.Random(f"{seed}:benchmark:{key}").sample(group, counts[key]))
    return chosen
