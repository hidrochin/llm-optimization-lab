"""Streaming chat client for a vLLM OpenAI-compatible server (runs in .venv-vllm).

The server must be started from the same config (inference/serve_vllm.py): requests use
the run id as the model name.
"""

from __future__ import annotations

import time
from pathlib import Path

from openai import AsyncOpenAI

from llm_opt_lab.config import ExperimentConfig
from llm_opt_lab.data import Sample
from llm_opt_lab.doc_types import DocType
from llm_opt_lab.field_requests import Request
from llm_opt_lab.prompting import Message, build_messages, to_openai_messages
from llm_opt_lab.serving import RequestResult


def make_client(config: ExperimentConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.serving.api_base,
        api_key="EMPTY",
        timeout=config.serving.request_timeout_s,
        max_retries=0,  # a retry would hide failures and distort latency
    )


def request_messages(
    sample: Sample, doc_type: DocType, data_dir: str | Path, request: Request
) -> list[Message]:
    return to_openai_messages(build_messages(sample, doc_type, data_dir, request=request))


async def send_chat(
    client: AsyncOpenAI, config: ExperimentConfig, request_id: str, messages: list[Message]
) -> RequestResult:
    """One streamed request; TTFT is the time to the first content token."""
    extra_body = None
    if config.model.chat_template_kwargs:
        extra_body = {"chat_template_kwargs": config.model.chat_template_kwargs}
    start = time.perf_counter()
    ttft_ms = None
    parts: list[str] = []
    usage = None
    try:
        stream = await client.chat.completions.create(
            model=config.run_id,
            messages=messages,
            max_tokens=config.evaluation.max_new_tokens,
            temperature=config.evaluation.temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                if choice.delta and choice.delta.content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    parts.append(choice.delta.content)
    except Exception as e:  # timeouts, server errors: recorded per request, never retried
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(request_id, None, f"{type(e).__name__}: {e}", latency_ms)
    return RequestResult(
        id=request_id,
        text="".join(parts),
        error=None,
        latency_ms=(time.perf_counter() - start) * 1000,
        ttft_ms=ttft_ms,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
    )
