"""Score a predictions file against ground truth: parse, normalize, compare, aggregate.

Only the fields the prompt asked for are scored: the prediction's `fields`, or the config's
inference request. Requested fields that a record does not annotate are skipped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_opt_lab.config import FieldsConfig, QualityTarget
from llm_opt_lab.data import Sample
from llm_opt_lab.doc_types import DocType
from llm_opt_lab.field_requests import holdout_paths, inference_request, scored_target
from llm_opt_lab.metrics import (
    FieldGroupStats,
    FieldResult,
    FieldStats,
    QualityStats,
    SampleScore,
    generic_path,
    per_field_stats,
    score_sample,
    summarize,
    summarize_by,
    summarize_field_group,
)
from llm_opt_lab.parsing import parse_json_output
from llm_opt_lab.postprocess import check_field_types, normalize_record
from llm_opt_lab.predictions import Prediction

FIELD_GROUPS = ("seen", "held_out", "decoy")


@dataclass
class EvalReport:
    overall: QualityStats
    by_input_type: dict[str, QualityStats]
    by_doc_type: dict[str, QualityStats]
    # seen / held_out (fields.holdout) / decoy; None when every scored field is "seen"
    by_field_group: dict[str, FieldGroupStats] | None
    per_field: list[FieldStats]
    n_request_errors: int
    details: list[dict[str, Any]]  # one per sample; contains private data

    def meets(self, target: QualityTarget) -> bool:
        em, fa = self.overall.exact_match, self.overall.field_accuracy
        return (
            em is not None
            and fa is not None
            and em >= target.exact_match
            and fa >= target.field_accuracy
        )

    def metrics_updates(self, target: QualityTarget) -> dict[str, Any]:
        """Fields for `results.record_metrics`."""
        return {
            "n_samples": self.overall.n,
            "n_fields": self.overall.n_fields,
            "exact_match": self.overall.exact_match,
            "field_accuracy": self.overall.field_accuracy,
            "ned_mean": self.overall.ned_mean,
            "parse_rate": self.overall.parse_rate,
            "key_presence": self.overall.key_presence,
            "meets_target": self.meets(target),
            "by_input_type": self.by_input_type,
            "by_doc_type": self.by_doc_type,
            "by_field_group": self.by_field_group,
        }


def field_groups(
    scores: list[SampleScore], doc_types: dict[str, DocType], fields: FieldsConfig
) -> dict[str, FieldGroupStats] | None:
    held_out = {name: holdout_paths(dt, fields) for name, dt in doc_types.items()}
    decoys = {name: {d.path for d in dt.decoys} for name, dt in doc_types.items()}
    groups: dict[str, list[FieldResult]] = defaultdict(list)
    for score in scores:
        for result in score.fields:
            path = generic_path(result.path)
            if path in decoys[score.doc_type]:
                groups["decoy"].append(result)
            elif path in held_out[score.doc_type]:
                groups["held_out"].append(result)
            else:
                groups["seen"].append(result)
    if set(groups) <= {"seen"}:
        return None
    return {name: summarize_field_group(groups[name]) for name in FIELD_GROUPS if name in groups}


def evaluate_predictions(
    samples: list[Sample],
    predictions: dict[str, Prediction],
    doc_types: dict[str, DocType],
    fields: FieldsConfig | None = None,
) -> EvalReport:
    """Every sample needs a prediction: a partial run is an error, not a lower score."""
    fields = fields or FieldsConfig()
    missing = [s.id for s in samples if s.id not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} samples have no prediction, e.g. {missing[:5]}")
    unknown = sorted({s.doc_type for s in samples} - set(doc_types))
    if unknown:
        raise ValueError(f"samples use doc types that are not loaded: {unknown}")
    for doc_type in doc_types.values():
        check_field_types(doc_type)
    default_paths = {
        name: inference_request(dt, fields).paths
        for name, dt in doc_types.items()
        if any(s.doc_type == name for s in samples)
    }

    scores: list[SampleScore] = []
    details: list[dict[str, Any]] = []
    n_request_errors = 0
    for sample in samples:
        doc_type = doc_types[sample.doc_type]
        prediction = predictions[sample.id]
        if prediction.error is not None:
            n_request_errors += 1
            parsed, parse_error = None, f"request error: {prediction.error}"
        else:
            parsed, parse_error = parse_json_output(prediction.raw_output)
        requested = prediction.fields or default_paths[sample.doc_type]
        target = scored_target(sample.target, doc_type, requested)
        gold = normalize_record(target.gold, doc_type)
        pred = None if parsed is None else normalize_record(parsed, doc_type)
        score = score_sample(
            sample.id, sample.doc_type, sample.input_type, gold, pred, ignore=target.skipped
        )
        scores.append(score)
        details.append(
            {
                **asdict(score),
                "parse_error": parse_error,
                "raw_output": prediction.raw_output,
                "requested_fields": requested,
                "skipped_fields": sorted(target.skipped),
                "gold_normalized": gold,
                "pred_normalized": pred,
            }
        )

    return EvalReport(
        overall=summarize(scores),
        by_input_type=summarize_by(scores, "input_type"),
        by_doc_type=summarize_by(scores, "doc_type"),
        by_field_group=field_groups(scores, doc_types, fields),
        per_field=per_field_stats(scores),
        n_request_errors=n_request_errors,
        details=details,
    )


def write_eval_details(report: EvalReport, out_dir: str | Path) -> None:
    """Write per-sample results and per-field stats. Private: keep under artifacts/."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for detail in report.details:
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")
    fields = [s.model_dump() for s in report.per_field]
    (out_dir / "fields.json").write_text(
        json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
