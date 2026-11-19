"""Inspect a compressed-tensors checkpoint without loading it.

Reports the quantization config, the ignore list, how many packed 4-bit tensors each
component has, and the size on disk. Fails if a vision/audio tensor is quantized, or if
nothing is. Reads only config.json and the safetensors headers: any environment works.

Usage:
    python quantization/verify_quantized.py artifacts/quantized/<run_id>
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from llm_opt_lab.model_parts import is_non_language
from llm_opt_lab.paths import directory_size

PACKED_SUFFIX = ".weight_packed"  # compressed-tensors pack-quantized format


def safetensors_keys(path: Path) -> list[str]:
    """Tensor names from a safetensors header (8-byte little-endian length + JSON)."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return [k for k in header if k != "__metadata__"]


def component(name: str, depth: int = 3) -> str:
    return ".".join(name.split(".")[:depth])


def checkpoint_report(directory: Path) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or config.get("text_config", {}).get(
        "quantization_config"
    )
    keys = [k for f in sorted(directory.glob("*.safetensors")) for k in safetensors_keys(f)]
    packed = [k for k in keys if k.endswith(PACKED_SUFFIX)]
    wrongly_packed = [k for k in packed if is_non_language(k)]

    problems = []
    if not quant:
        problems.append("config.json has no quantization_config")
    if not packed:
        problems.append(f"no packed 4-bit tensors (*{PACKED_SUFFIX}) found")
    if wrongly_packed:
        problems.append(
            f"{len(wrongly_packed)} vision/audio tensors are quantized, e.g. {wrongly_packed[:3]}"
        )
    return {
        "format": (quant or {}).get("format"),
        "quant_method": (quant or {}).get("quant_method"),
        "config_groups": (quant or {}).get("config_groups"),
        "ignore": (quant or {}).get("ignore"),
        "n_tensors": len(keys),
        "packed_by_component": dict(Counter(component(k) for k in packed)),
        "size_gb": directory_size(directory) / 1024**3,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)

    report = checkpoint_report(args.checkpoint)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
