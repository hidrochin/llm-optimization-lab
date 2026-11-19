"""Re-score saved predictions after a scoring change (postprocess or metrics rules).

Usage:
    uv run python scripts/reevaluate.py                      # every results/metrics/*.json
    uv run python scripts/reevaluate.py results/metrics/E1_x.json

For each metrics file that has an `evaluate` stage, re-runs scripts/evaluate.py with the
config and split it was evaluated with. No GPU needed: predictions are already saved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

EVALUATE = Path(__file__).with_name("evaluate.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics", type=Path, nargs="*")
    args = parser.parse_args(argv)

    files = args.metrics or sorted(Path("results/metrics").glob("*.json"))
    failed = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        stage = data.get("stages", {}).get("evaluate")
        if stage is None:
            continue
        config = stage.get("config") or data.get("config")
        cmd = [sys.executable, str(EVALUATE), "--config", config, "--split", data["split"]]
        print(f"== {path.name}: {' '.join(cmd[1:])}")
        if subprocess.run(cmd).returncode != 0:
            failed.append(path.name)
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
