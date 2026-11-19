# Candidate models

Which models we test, for which input flow, and why. The plan and the experiment matrix
are in [README.md](../README.md); background reading on the architectures is in
[reading.md](reading.md). Checked on 2026-11-20 against the Hugging Face model lists of
Qwen, Google, DeepSeek, Z.ai, Mistral, NVIDIA and OpenBMB.

## Starting point: the production model

Production serves **Qwen3.6-35B-A3B-FP8** with vLLM on one L40, for both flows (image and
spatial text). It is accurate but slow under load. Its FP8 weights take 37.5 GB of the
L40's 48 GB, which leaves roughly 4–5 GB for KV cache: only a handful of long requests run
at once and the rest of the 60 wait (our estimate; T-C05 records what vLLM reports).

A smaller model helps in two ways that pull in different directions:

- **Total parameters** decide how much memory is left for KV cache, i.e. how many requests
  run at once.
- **Active parameters** decide the compute per token. Extraction is prefill-heavy (long
  input, short JSON). Qwen3.6-35B-A3B activates about 3B parameters per token, so a dense
  9B model needs about 3× its prefill compute, and a 4B model about the same.

A smaller model is therefore not automatically faster at 60 concurrent requests; C
measures it (T-C08).

## Selection criteria

1. Reads document images, including Vietnamese text, and accepts spatial text.
2. Leaves room for KV cache on one L40, with W4A16 where needed.
3. Open weights under a license that allows internal fine-tuning and serving (every model
   below is Apache 2.0).
4. Runs on our stack: transformers 5.23, PEFT, LLM Compressor, vLLM. Gemma 4, Qwen3.5 and
   Qwen3.6 MoE pass the tiny-model tests (`tests/test_hf_tiny.py`: SFT masking,
   label-position loss, HF prediction, GRPO).

No public benchmark measures field-level extraction on Vietnamese documents for these
models. Our own test set decides.

## Candidates by flow

| Model | Hugging Face ID | Parameters (total / active) | Image | Spatial text | Priority | Experiments |
|-------|-----------------|-----------------------------|:-----:|:------------:|:--------:|-------------|
| Qwen3.6-35B-A3B-FP8 | `Qwen/Qwen3.6-35B-A3B-FP8` | 36B / ~3B, MoE | ✓ | ✓ | reference | R0 |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | 36B / ~3B, MoE | ✓ | ✓ | P1, P2 | R1, E15 |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 9.7B dense | ✓ strong | ✓ | P1 | E14 |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 4.7B dense | ✓ | ✓ | P1 | E13, E18 |
| Gemma 4 E2B | `google/gemma-4-E2B-it` | ~2B effective | weak | ✓ | P0 | E0–E6, E18 |
| Gemma 4 E4B | `google/gemma-4-E4B-it` | ~4B effective | ✓ | ✓ | P1 | E7 |
| Gemma 4 12B | `google/gemma-4-12B-it` | 12B dense | ✓ strong | ✓ | P2 | E8 |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | 26B / ~4B, MoE | ✓ | ✓ | P1 | E9, E10, E12 |
| NuExtract3 | `numind/NuExtract3` | 4.5B dense | ✓ | ✓ | P2 | E17 |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | 2.3B dense | — | ✓ | P2 | E16 |
| MiniCPM-V 4.7 | `openbmb/MiniCPM-V-4.7-35B-A3B`, `-1B` | 35B / ~3B; 1B | ✓ | — | P3, watch | — |

**Strong** marks the best published document scores among the small candidates.
**Weak**: Gemma 4 E2B's OmniDocBench edit distance (0.290) is far behind E4B (0.181). It
may still be enough after fine-tuning; E1 reports accuracy by input type.

### Both flows (one model, as in production)

**Qwen3.6-35B-A3B: production reference (R0) and its cheaper versions (R1, E15).**
R0 measures production as it runs today: same checkpoint, prompts and serve settings.
R1 quantizes the same model to W4A16 without training: about 20 GB of weights instead of
37.5 GB. This is the cheapest possible speed-up, because nothing else changes. Qwen's own
GPTQ-Int4 checkpoint of Qwen3.5-35B-A3B keeps every `attn` module (full and Gated DeltaNet
`linear_attn`), `shared_expert`, `mtp` and `visual` in BF16; our recipe starts from the
same choices (T-D09). E15 (P2) fine-tunes the production model with QLoRA. Its likely
blocker is the same as for Gemma 4 26B-A4B: LoRA on fused expert parameters while they are
4-bit (T-E02).

**Qwen3.5-4B and Qwen3.5-9B (E13, E14).** These are the newest small Qwen models
(February 2026). Qwen3.6 has only 35B-A3B and 27B, and Qwen3.8 (August 2026) only 27B and
much larger models. They share the architecture family, tokenizer and chat template of the
production model, so production prompts carry over and results compare directly. Both
read images natively. Model card scores (4B / 9B):

- OCRBench: 85.0 / 89.2
- OmniDocBench1.5: 86.2 / 87.7
- CC-OCR: 76.7 / 79.3

Zero-shot extraction is weak, and thinking mode writes very long outputs: NuMind's
benchmark counts 22–27K thinking tokens per document. Configs therefore set
`enable_thinking: false`, and we fine-tune.

**Gemma 4 E2B, E4B, 12B, 26B-A4B (E0–E12).** These are the original line of the plan.
Gemma fixes the image token budget by setting (70–1120 tokens per image). Qwen's grows with
resolution, at one token per 32×32 pixels, so an A4 page at 150 dpi is about 2,100 tokens.
Image resolution is therefore a variable to agree per model (T-L04).

- **12B** (May 2026) is encoder-free ("Unified"): image patches go straight into the
  language model.
- **OmniDocBench1.5 edit distance** (lower is better): E2B 0.290, E4B 0.181, 12B 0.164,
  26B-A4B 0.149.
- **Official QAT W4A16 checkpoints** exist for E2B, E4B and 12B (E5, E6).

**NuExtract3 (E17, P2).** Qwen3.5-4B fine-tuned by NuMind for structured extraction,
including an RL stage. On NuMind's own benchmark of about 600 documents (zero-shot,
self-reported) it scores 0.651, against 0.538 for Gemma 4 E4B and 0.479 for Qwen3.5-9B.
It expects its own extraction template (`chat_template_kwargs.template`) rather than our
system prompts. Run it zero-shot in its own format first, then with LoRA on ours if it is
promising. Its card does not mention Vietnamese.

### Spatial text only

**Qwen3.5-2B (E16, P2).** The text flow reads OCR output that already carries the layout,
so it may not need a large model. If a 2B model reaches 95% on spatial text, production
can split the flows: a small text model for spatial text, served text-only with vLLM
`--language-model-only`, which skips the vision encoder, and a larger model only for
images. The by-input-type results of E1 (E2B) and E13 (Qwen3.5-4B) answer the same
question at no extra cost.

### Image only

**MiniCPM-V 4.7 (P3, watch list).** Released in October 2026 and built on Qwen3.5/3.6
with a SigLIP2 encoder. Its 4×/16× visual-token compression targets exactly the cost of
the image flow. So far there are only benchmark charts and no independent results. Test it
only if C's sweep shows the image flow is still the bottleneck.

## Out of scope

- **Speculative decoding** (MTP heads, Gemma 4 `-assistant` drafters, DSpark): it speeds
  up serving only and does not interact with fine-tuning or quantization (decided
  2026-11-20).
- **Models that do not fit our hardware or task:**
  - too large for one L40: Qwen3.8 Flash-Next and 2.4T, DeepSeek V4, GLM-5.3, Mistral
    Large 4, Nemotron 3.5 Super;
  - a dense 27B–31B (Qwen3.8-27B, Gemma 4 31B), whose prefill compute is about 9× that of
    the production model;
  - OCR-only parsers (PaddleOCR-VL, GLM-OCR, DeepSeek-OCR-2), which output Markdown, not
    schema JSON.

## Sources

- Model lists: [Qwen](https://huggingface.co/Qwen), [Google](https://huggingface.co/google), [DeepSeek](https://huggingface.co/deepseek-ai), [OpenBMB](https://huggingface.co/openbmb)
- Model cards: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.6-35B-A3B blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b), [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it), [Gemma 4 blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [NuExtract3](https://huggingface.co/numind/NuExtract3), [MiniCPM-V 4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)
- Quantization: [Qwen3.5-35B-A3B-GPTQ-Int4 config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/blob/main/config.json)
- Serving: [vLLM recipe for Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
