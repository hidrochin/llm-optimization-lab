"""Where a run's errors come from, and what changed between two runs.

Usage:
    uv run python scripts/error_analysis.py --run E1_gemma4_e2b_lora_r16
    uv run python scripts/error_analysis.py --run E2_gemma4_e2b_qlora_r16 \
        --compare E1_gemma4_e2b_lora_r16 --split validation

Reads   artifacts/eval/<run>/<split>/samples.jsonl (written by scripts/evaluate.py)
Writes  artifacts/eval/<run>/<split>/errors.md, or vs_<other run>.md with --compare

The report quotes field values, so it contains private data: it stays under artifacts/
and is never committed. Sections:
- fields with the lowest accuracy, and how they fail: missing, invented, wrong value, or a
  near miss (NED <= 0.2, often a formatting rule postprocess does not cover yet, T-L05)
- the most frequent wrong values of those fields
- list length mismatches, and lists with the right items in another order (T-L06: list
  items are compared by position, so both make every later item wrong)
- parse failures and extra fields
- with --compare: samples fixed and broken, and the per-field accuracy change
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from llm_opt_lab.metrics import generic_path
from llm_opt_lab.paths import RunPaths

NEAR_MISS_NED = 0.2


def load_details(run_id: str, split: str) -> list[dict[str, Any]]:
    path = RunPaths(run_id).eval_details(split) / "samples.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/evaluate.py for {run_id} on {split}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def failure_kind(field: dict[str, Any], parsed: bool) -> str | None:
    if field["correct"]:
        return None
    if not parsed:
        return "unparsed"
    if field["pred"] is None:
        return "missing"
    if field["gold"] is None:
        return "invented"
    return "near_miss" if field["ned"] <= NEAR_MISS_NED else "wrong"


KINDS = ("missing", "invented", "wrong", "near_miss", "unparsed")


def field_failures(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (doc type, generic field): accuracy and failure counts, lowest accuracy first."""
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for sample in details:
        for field in sample["fields"]:
            counts = groups[(sample["doc_type"], generic_path(field["path"]))]
            counts["n"] += 1
            counts[failure_kind(field, sample["parsed"]) or "correct"] += 1
    rows = [
        {"doc_type": d, "field": f, "accuracy": c["correct"] / c["n"], **c}
        for (d, f), c in groups.items()
    ]
    return sorted(rows, key=lambda r: (r["accuracy"], r["doc_type"], r["field"]))


def wrong_values(
    details: list[dict[str, Any]], doc_type: str, field: str, top: int
) -> list[tuple[tuple[Any, Any], int]]:
    pairs: Counter[tuple[Any, Any]] = Counter()
    for sample in details:
        if sample["doc_type"] != doc_type or not sample["parsed"]:
            continue
        for f in sample["fields"]:
            if not f["correct"] and generic_path(f["path"]) == field:
                pairs[(f["gold"], f["pred"])] += 1
    return pairs.most_common(top)


def lists_by_path(value: Any, path: str = "") -> dict[str, list[Any]]:
    """Every list in a record, by concrete path (`menu`, `menu[2].sub`)."""
    found: dict[str, list[Any]] = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            found.update(lists_by_path(sub, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        found[path] = value
        for i, sub in enumerate(value):
            found.update(lists_by_path(sub, f"{path}[{i}]"))
    return found


def list_issues(details: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """Per generic list path: lists compared, shorter, longer, same items in another order."""
    issues: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in details:
        if not sample["parsed"]:
            continue
        pred_lists = lists_by_path(sample["pred_normalized"] or {})
        for path, gold in lists_by_path(sample["gold_normalized"]).items():
            pred = pred_lists.get(path) or []
            counts = issues[generic_path(path)]
            counts["lists"] += 1
            if len(pred) < len(gold):
                counts["shorter"] += 1
            elif len(pred) > len(gold):
                counts["longer"] += 1
            elif pred != gold and sorted(map(_key, pred)) == sorted(map(_key, gold)):
                counts["reordered"] += 1
    return issues


def _key(item: Any) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False)


def _short(value: Any, width: int = 40) -> str:
    text = "null" if value is None else str(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "-"


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _header(*names: str) -> list[str]:
    return [_row(*names), "|" + "---|" * len(names)]


def summary_section(details: list[dict[str, Any]]) -> list[str]:
    lines = _header("Input type", "Samples", "Exact match", "Parsed", "With extra fields")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in details:
        groups["all"].append(sample)
        groups[sample["input_type"]].append(sample)
    for name, group in groups.items():
        n = len(group)
        exact = sum(s["exact_match"] for s in group)
        parsed = sum(s["parsed"] for s in group)
        extra = sum(bool(s["extra_fields"]) for s in group)
        lines.append(_row(name, n, _pct(exact, n), _pct(parsed, n), _pct(extra, n)))
    return lines


def report(run_id: str, split: str, details: list[dict[str, Any]], top: int) -> str:
    out = [f"# Error analysis: `{run_id}` on {split}", "", *summary_section(details), ""]

    fields = field_failures(details)
    out += [
        "## Lowest-accuracy fields",
        "",
        f"Near miss: wrong value with NED <= {NEAR_MISS_NED}. "
        "Unparsed: the whole output failed to parse.",
        "",
        *_header("Doc type", "Field", "n", "Accuracy", *KINDS),
    ]
    for row in fields[:top]:
        kinds = [row.get(kind, 0) for kind in KINDS]
        out.append(
            _row(row["doc_type"], f"`{row['field']}`", row["n"], f"{row['accuracy']:.1%}", *kinds)
        )

    out += ["", "## Most frequent wrong values", ""]
    for row in [r for r in fields if r["accuracy"] < 1][: min(top, 5)]:
        pairs = wrong_values(details, row["doc_type"], row["field"], top=5)
        if not pairs:
            continue
        out += [f"`{row['doc_type']}/{row['field']}`", "", *_header("Gold", "Predicted", "Count")]
        out += [_row(_short(g), _short(p), n) for (g, p), n in pairs]
        out.append("")

    issues = list_issues(details)
    if issues:
        out += [
            "## Lists",
            "",
            "Items are compared by position: a missing, extra or reordered item makes every "
            "later item wrong.",
            "",
            *_header("List", "Compared", "Shorter", "Longer", "Same items, other order"),
        ]
        for path, c in sorted(issues.items()):
            n = c["lists"]
            shares = [_pct(c[kind], n) for kind in ("shorter", "longer", "reordered")]
            out.append(_row(f"`{path}`", n, *shares))
        out.append("")

    unparsed = [s for s in details if not s["parsed"]]
    out += ["## Parse failures", "", f"{len(unparsed)} of {len(details)} samples.", ""]
    reasons = Counter((s["parse_error"] or "unknown").split(":")[0] for s in unparsed)
    for reason, n in reasons.most_common():
        examples = [s["id"] for s in unparsed if (s["parse_error"] or "unknown").startswith(reason)]
        out.append(f"- {reason}: {n} (e.g. {', '.join(examples[:3])})")

    extra = Counter(generic_path(p) for s in details for p in s["extra_fields"])
    out += ["", "## Extra fields", "", f"{sum(bool(s['extra_fields']) for s in details)} samples."]
    out += [f"- `{path}`: {n}" for path, n in extra.most_common(top)]
    return "\n".join(out) + "\n"


def compare_report(
    run_id: str,
    other_id: str,
    split: str,
    details: list[dict[str, Any]],
    other: list[dict[str, Any]],
    top: int,
) -> str:
    by_id = {s["id"]: s for s in details}
    other_by_id = {s["id"]: s for s in other}
    if by_id.keys() != other_by_id.keys():
        raise SystemExit(
            f"the runs were evaluated on different samples ({len(by_id)} vs {len(other_by_id)}): "
            "compare runs on the same frozen split"
        )
    names = {
        (True, True): "both right",
        (True, False): "fixed",
        (False, True): "broken",
        (False, False): "both wrong",
    }
    by_outcome: dict[str, list[str]] = {name: [] for name in names.values()}
    for sample_id, sample in by_id.items():
        now, before = sample["exact_match"], other_by_id[sample_id]["exact_match"]
        by_outcome[names[(now, before)]].append(sample_id)
    broken, fixed = by_outcome["broken"], by_outcome["fixed"]

    acc = {(r["doc_type"], r["field"]): r["accuracy"] for r in field_failures(details)}
    other_acc = {(r["doc_type"], r["field"]): r["accuracy"] for r in field_failures(other)}
    deltas = sorted(
        ((acc.get(k, 0.0) - other_acc.get(k, 0.0), k) for k in acc.keys() | other_acc.keys()),
        key=lambda d: (d[0], d[1]),
    )

    n = len(by_id)
    out = [
        f"# `{run_id}` vs `{other_id}` on {split}",
        "",
        f"Exact match per sample, {n} samples (fixed: wrong in `{other_id}`, right in `{run_id}`).",
        "",
        *_header("Outcome", "Samples"),
        *(_row(k, f"{len(ids)} ({_pct(len(ids), n)})") for k, ids in by_outcome.items()),
        "",
        f"Broken: {', '.join(broken[:20]) or '-'}{' …' if len(broken) > 20 else ''}",
        "",
        f"Fixed: {', '.join(fixed[:20]) or '-'}{' …' if len(fixed) > 20 else ''}",
        "",
        "## Field accuracy change, largest drops first",
        "",
        *_header("Doc type", "Field", f"`{other_id}`", f"`{run_id}`", "Change"),
    ]
    for delta, (doc_type, field) in [d for d in deltas if d[0] != 0][:top]:
        before, now = other_acc.get((doc_type, field), 0.0), acc.get((doc_type, field), 0.0)
        out.append(
            _row(doc_type, f"`{field}`", f"{before:.1%}", f"{now:.1%}", f"{100 * delta:+.1f} pt")
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run ID: <experiment_id>_<run_name>")
    parser.add_argument("--compare", help="run ID to compare with (the baseline)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top", type=int, default=15, help="rows per table")
    args = parser.parse_args(argv)

    details = load_details(args.run, args.split)
    out_dir = RunPaths(args.run).eval_details(args.split)
    if args.compare:
        text = compare_report(
            args.run,
            args.compare,
            args.split,
            details,
            load_details(args.compare, args.split),
            args.top,
        )
        path = out_dir / f"vs_{args.compare}.md"
    else:
        text = report(args.run, args.split, details, args.top)
        path = out_dir / "errors.md"
    path.write_text(text, encoding="utf-8")
    print(f"{path} (contains private data: do not commit or share outside the servers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
