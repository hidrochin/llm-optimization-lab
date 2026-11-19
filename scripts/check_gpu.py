"""Check CUDA, BF16 and Gemma 4 support in the training environment.

Usage:
    uv run --group train python scripts/check_gpu.py
"""

import torch
import transformers

print("CUDA available:", torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GB")
if torch.cuda.is_available():
    print("BF16 support:", torch.cuda.is_bf16_supported())
print("PyTorch:", torch.__version__, "| CUDA runtime:", torch.version.cuda)
print(
    "Transformers:",
    transformers.__version__,
    "| Gemma 4:",
    hasattr(transformers, "Gemma4ForConditionalGeneration"),
)
