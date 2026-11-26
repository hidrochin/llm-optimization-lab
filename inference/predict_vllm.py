"""Generate predictions through a running vLLM server (start it with serve_vllm.py, same config).

Usage:
    python inference/predict_vllm.py --config configs/dev/cord_e2b_base_vllm.yaml --split test
    python inference/predict_vllm.py --config ... --concurrency 32 --limit 50

Writes artifacts/predictions/<run_id>/<split>.jsonl, the file scripts/evaluate.py reads.
Latency here is for throughput only; serving latency comes from benchmark_api.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import make_client, request_messages, send_chat  # noqa: E402

from llm_opt_lab.config import load_config  # noqa: E402
from llm_opt_lab.data import Sample, filter_samples, load_samples  # noqa: E402
from llm_opt_lab.doc_types import load_doc_types  # noqa: E402
from llm_opt_lab.field_requests import inference_request  # noqa: E402
from llm_opt_lab.paths import RunPaths  # noqa: E402
from llm_opt_lab.predictions import Prediction, write_predictions  # noqa: E402
from llm_opt_lab.serving import RequestResult, run_closed_loop  # noqa: E402


async def predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config.serving is None:
        raise SystemExit("config has no serving section")
    split = args.split or config.evaluation.split
    samples = filter_samples(
        load_samples(config.data.dir / f"{split}.jsonl"),
        config.data.doc_types,
        config.data.input_types,
    )[: args.limit]
    doc_types = load_doc_types(config.data.doc_types_dir)
    requests = {name: inference_request(dt, config.fields) for name, dt in doc_types.items()}
    fields_of = {sample.id: requests[sample.doc_type].paths for sample in samples}
    client = make_client(config)
    done = 0

    async def send(sample: Sample) -> RequestResult:
        nonlocal done
        messages = request_messages(
            sample, doc_types[sample.doc_type], config.data.dir, requests[sample.doc_type]
        )
        result = await send_chat(client, config, sample.id, messages)
        done += 1
        if done % 50 == 0 or done == len(samples):
            print(f"  {done}/{len(samples)}")
        return result

    results = await run_closed_loop(samples, args.concurrency, send)
    out = args.out or RunPaths.for_config(config).predictions(split)
    write_predictions(
        out,
        (
            Prediction(
                id=r.id,
                raw_output=r.text,
                error=r.error,
                fields=fields_of[r.id],
                latency_ms=r.latency_ms,
                ttft_ms=r.ttft_ms,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
            )
            for r in results
        ),
    )
    errors = sum(r.error is not None for r in results)
    print(f"{len(results)} predictions ({errors} errors) -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", help="defaults to evaluation.split")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    return asyncio.run(predict(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
