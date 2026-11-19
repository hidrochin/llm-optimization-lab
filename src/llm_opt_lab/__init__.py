"""Shared core for the LLM optimization lab.

Everything in this package is used by more than one track (training, quantization,
serving, MoE), so it must not import torch, transformers or vllm at module level.
"""

__version__ = "0.1.0"
