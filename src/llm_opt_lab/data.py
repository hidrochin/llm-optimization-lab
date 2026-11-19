"""Dataset records: one JSON object per line, one record per (document, input type).

See docs/formats.md for the full contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, ValidationError, model_validator

from llm_opt_lab.config import InputType, StrictModel


class Sample(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    doc_type: str
    input_type: InputType
    # Records built from the same source document (e.g. its image and its spatial text)
    # share a group_id, so the split never puts them on different sides.
    group_id: str | None = None
    image: str | None = None  # relative to the dataset root
    spatial_text: str | None = None
    target: dict[str, Any]
    meta: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_input(self) -> Sample:
        if self.input_type == "image" and not self.image:
            raise ValueError("input_type 'image' requires the 'image' field")
        if self.input_type == "spatial_text" and not self.spatial_text:
            raise ValueError("input_type 'spatial_text' requires the 'spatial_text' field")
        return self

    @property
    def group(self) -> str:
        return self.group_id or self.id


def load_samples(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                sample = Sample.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"{path}:{line_no}: invalid sample: {e}") from e
            if sample.id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate sample id {sample.id!r}")
            seen.add(sample.id)
            samples.append(sample)
    return samples


def write_samples(path: str | Path, samples: Iterable[Sample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.model_dump_json(exclude_none=True) + "\n")


def filter_samples(
    samples: Iterable[Sample],
    doc_types: Iterable[str] = (),
    input_types: Iterable[str] = (),
) -> list[Sample]:
    """Keep samples matching the given doc types and input types; empty means no filter."""
    doc_types, input_types = set(doc_types), set(input_types)
    return [
        s
        for s in samples
        if (not doc_types or s.doc_type in doc_types)
        and (not input_types or s.input_type in input_types)
    ]


def image_path(sample: Sample, data_dir: str | Path) -> Path:
    if sample.image is None:
        raise ValueError(f"sample {sample.id!r} has no image")
    return Path(data_dir) / sample.image
