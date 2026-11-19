"""Start a vLLM OpenAI-compatible server from a config's `serving` section.

Usage:
    python inference/serve_vllm.py --config configs/dev/cord_e2b_base_vllm.yaml
    python inference/serve_vllm.py --config ... --dry-run     # print the command only

Sets VLLM_SERVER_DEV_MODE=1 (unless --no-dev-mode) so benchmark_api.py can reset the
prefix cache between concurrency levels.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from llm_opt_lab.config import load_config
from llm_opt_lab.paths import RunPaths
from llm_opt_lab.results import save_gpu_snapshot
from llm_opt_lab.serving import build_vllm_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-dev-mode", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    cmd = build_vllm_command(config)
    print(shlex.join(cmd))
    if args.dry_run:
        return 0
    if not args.no_dev_mode:
        os.environ.setdefault("VLLM_SERVER_DEV_MODE", "1")
    save_gpu_snapshot(RunPaths.for_config(config).logs / "nvidia-smi_serve.txt")
    os.execvp(cmd[0], cmd)  # replaces this process with the server


if __name__ == "__main__":
    raise SystemExit(main())
