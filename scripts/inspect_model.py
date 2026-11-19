"""Show a model's structure without downloading weights (only config.json is fetched).

Prints parameter counts per component, the Linear module names per component (LoRA
target candidates), tensors with 3+ dimensions (fused MoE experts, conv layers) and
router / expert modules. Works for dense and MoE models.

Usage:
    uv run --group train python scripts/inspect_model.py --model-id google/gemma-4-E2B-it
    uv run --group train python scripts/inspect_model.py --model-id google/gemma-4-26B-A4B-it \
        --out reports/moe_architecture.md
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

_LAYER_INDEX = re.compile(r"\.\d+(?=\.|$)")
_MOE_NAME = re.compile(r"^(gate|router|experts?|shared_experts?)$|router")


def pattern(name: str) -> str:
    """`model.layers.12.mlp.experts` -> `model.layers.*.mlp.experts`."""
    return _LAYER_INDEX.sub(".*", name)


def component(name: str, depth: int) -> str:
    return ".".join(name.split(".")[:depth])


def build_empty(model_id: str, revision: str | None, causal_lm: bool) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    cls = AutoModelForCausalLM if causal_lm else AutoModelForImageTextToText
    with init_empty_weights():
        return cls.from_config(config)


def report(model: torch.nn.Module, model_id: str, depth: int) -> str:
    params = dict(model.named_parameters())
    total = sum(p.numel() for p in params.values())
    lines = [f"# {model_id}", "", f"Total parameters: {total / 1e9:.3f} B", ""]

    per_component = Counter()
    for name, p in params.items():  # group by owning module, so `lm_head.weight` -> `lm_head`
        per_component[component(name.rsplit(".", 1)[0], depth)] += p.numel()
    lines += [
        "## Parameters per component",
        "",
        "| Component | Parameters | Share |",
        "|---|---:|---:|",
    ]
    for name, count in per_component.most_common():
        lines.append(f"| `{name}` | {count / 1e6:,.1f} M | {count / total:.1%} |")

    linear: dict[str, Counter] = defaultdict(Counter)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear[component(name, depth)][name.rsplit(".", 1)[-1]] += 1
    lines += ["", "## Linear modules per component (LoRA target candidates)", ""]
    for name, suffixes in sorted(linear.items()):
        listed = ", ".join(f"`{s}` ×{n}" for s, n in sorted(suffixes.items()))
        lines.append(f"- `{name}`: {listed}")

    owners = {n: type(m).__name__ for n, m in model.named_modules()}
    fused: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name, p in params.items():
        if p.dim() >= 3:
            fused.setdefault(
                pattern(name), (owners.get(name.rsplit(".", 1)[0], "?"), tuple(p.shape))
            )
    lines += ["", "## Tensors with 3+ dimensions (fused experts, convolutions)", ""]
    lines += [f"- `{n}` in {owner}: {list(shape)}" for n, (owner, shape) in sorted(fused.items())]
    if not fused:
        lines.append("- none")

    moe = {pattern(n): type(m).__name__ for n, m in model.named_modules()
           if n and _MOE_NAME.search(n.rsplit(".", 1)[-1])}  # fmt: skip
    lines += ["", "## Router / expert modules", ""]
    lines += [f"- `{n}`: {cls}" for n, cls in sorted(moe.items())] or ["- none (dense model)"]

    decoder_layers = sorted(
        {type(m).__name__ for m in model.modules() if "DecoderLayer" in type(m).__name__}
    )
    lines += [
        "",
        f"Decoder layer classes (for `quantization.sequential_targets`): {decoder_layers}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--causal-lm", action="store_true", help="text-only model such as Qwen3")
    parser.add_argument("--depth", type=int, default=2, help="name depth that defines a component")
    parser.add_argument("--out", type=Path, help="also write the report to this Markdown file")
    args = parser.parse_args(argv)

    model = build_empty(args.model_id, args.revision, args.causal_lm)
    text = report(model, args.model_id, args.depth)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
