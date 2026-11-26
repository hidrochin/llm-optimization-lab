# Reading list

Papers and official docs behind the models, techniques and terms used in this repo.

- ⭐ marks what to read first.
- Each entry says why it matters here.
- Every arXiv link was checked against its title on 2026-11-20.

You do not need to read whole papers. The abstract, the method figure and the experiment
that matches our question are usually enough.

**Where to start, by role:**

| Role | Sections |
|------|----------|
| Everyone | §1 and the ⭐ entries of §8 (document AI) |
| A, B (fine-tuning) | §4 |
| C (serving) | §6 |
| D (quantization) | §5 |
| E (MoE) | §3 and §2 (Gated DeltaNet) |
| Lead (RL) | §7 |

## 1. How LLM inference works

| Read | Why it matters here |
|------|---------------------|
| ⭐ [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | The transformer: attention, and the keys and values that become the KV cache |
| ⭐ [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363) (Yuan et al., 2024) | Prefill vs decode, compute-bound vs memory-bound. Explains why KV cache limits concurrency (README Risk 2) and why active parameters set prefill cost ([models](models.md#starting-point-the-production-model)) |
| [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022), [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023) | Exact attention without storing the full attention matrix; why 32K contexts fit at all |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2017) | BF16/FP16 training, loss scaling |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen et al., 2016) | Gradient checkpointing: trade compute for activation memory (Risk 1) |

## 2. The model families

| Read | Why it matters here |
|------|---------------------|
| ⭐ [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) (Gemma Team, 2025) | The architecture Gemma 4 builds on: local sliding-window + global attention, SigLIP vision encoder, QAT checkpoints |
| [Gemma 4 blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/), [Gemma 4 12B model card](https://huggingface.co/google/gemma-4-12B-it) | Sizes, image token budgets, benchmark tables, the encoder-free 12B |
| [MatFormer](https://arxiv.org/abs/2310.07707) (Devvrit et al., 2023) | Nested "effective" models: the idea behind E2B / E4B |
| ⭐ [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Yang et al., 2025) | Qwen3 language models; thinking vs non-thinking mode (we turn thinking off) |
| [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) (Bai et al., 2025), [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) (Bai et al., 2025) | Qwen's vision design: dynamic resolution, so image tokens grow with image size |
| [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5), [Qwen3.6-35B-A3B blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b) | The production family: native multimodality, hybrid attention, MoE |
| ⭐ [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (Yang et al., 2024) | The linear-attention layers in 3 of every 4 Qwen3.5/3.6 layers: a fixed-size state per request instead of a KV cache that grows with length |
| [Mamba](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023), [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) (Dao & Gu, 2024) | Background for state-space and linear-attention layers |
| [ViT](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020), [NaViT](https://arxiv.org/abs/2307.06304) (Dehghani et al., 2023) | Images as patch tokens; any aspect ratio and resolution |
| [SigLIP](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023), [SigLIP 2](https://arxiv.org/abs/2502.14786) (Tschannen et al., 2025) | The vision encoders inside Gemma and MiniCPM-V |

## 3. Mixture of experts

| Read | Why it matters here |
|------|---------------------|
| ⭐ [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017) | The MoE layer and its router: total vs active parameters |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021) | Routing and load balancing; why the router is sensitive (E12) |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024) | Open MoE in practice: memory of all experts, compute of a few |
| ⭐ [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (Dai et al., 2024) | Fine-grained and shared experts: the design of Qwen3.5/3.6 MoE (256 experts, 8 active, plus a shared expert) |
| [PEFT LoRA guide: targeting `nn.Parameter`](https://huggingface.co/docs/peft/developer_guides/lora) | `target_parameters` for fused expert weights (E10, E15) |

## 4. Fine-tuning

| Read | Why it matters here |
|------|---------------------|
| ⭐ [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) | Low-rank adapters, rank and alpha, merging |
| ⭐ [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) | NF4, double quantization; why the adapter is merged into BF16, never into NF4 (T-B06) |
| [rsLoRA: A Rank Stabilization Scaling Factor](https://arxiv.org/abs/2312.03732) (Kalajdzievski, 2023) | How alpha should scale with rank; relevant to the rank sweep (T-A05) |
| [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman et al., 2024) | What LoRA can and cannot learn compared with full fine-tuning |
| [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines, 2025), [TRL summary](https://huggingface.co/docs/trl/main/en/lora_without_regret) | LoRA matches full fine-tuning for RL even at low rank; learning-rate guidance for LoRA |
| [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora) | `target_modules`, `exclude_modules`, merging |

## 5. Quantization

| Read | Why it matters here |
|------|---------------------|
| [LLM.int8()](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022) | Outlier features: why naive low-bit quantization breaks LLMs |
| ⭐ [GPTQ](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022) | The W4A16 algorithm we use by default; calibration data and group size |
| ⭐ [AWQ](https://arxiv.org/abs/2306.00978) (Lin et al., 2023) | The alternative W4A16 algorithm (`quantization.algorithm: awq`) |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) (Xiao et al., 2022) | W8A8: moving activation outliers into weights |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022) | E4M3/E5M2: the production FP8 checkpoint and FP8 KV cache |
| [LLM-QAT](https://arxiv.org/abs/2305.17888) (Liu et al., 2023) | Quantization-aware training: background for the Gemma QAT checkpoints (E5, E6) |
| [LLM Compressor](https://github.com/vllm-project/llm-compressor), [compressed-tensors](https://github.com/vllm-project/compressed-tensors) | The tool and checkpoint format of `quantization/` |

## 6. Serving

| Read | Why it matters here |
|------|---------------------|
| ⭐ [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023) | vLLM: KV cache in blocks; why memory sets the number of concurrent requests |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (Yu et al., OSDI 2022) | Continuous (iteration-level) batching |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (Agrawal et al., 2024) | Chunked prefill: how long prompts slow down other requests' decoding |
| [DistServe](https://arxiv.org/abs/2401.09670) (Zhong et al., 2024) | TTFT vs TPOT and "goodput" under a latency SLO (T-C05) |
| [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html) | Why one system prompt per doc type is shared across requests, and why the benchmark resets the cache |
| [XGrammar](https://arxiv.org/abs/2411.15100) (Dong et al., 2024), [Efficient Guided Generation](https://arxiv.org/abs/2307.09702) (Willard & Louf, 2023) | Structured outputs from a JSON schema (T-C09) |

## 7. RL fine-tuning (W6)

| Read | Why it matters here |
|------|---------------------|
| [PPO](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) | The clipped policy-gradient objective GRPO keeps |
| [InstructGPT](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022) | The SFT → reward → RL pipeline |
| ⭐ [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024) | Introduces GRPO: advantages from a group of samples of the same prompt, no value model |
| ⭐ [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025) | RL with rule-based, verifiable rewards at scale |
| ⭐ [Tulu 3](https://arxiv.org/abs/2411.15124) (Lambert et al., 2024) | RL with verifiable rewards (RLVR) after SFT: the recipe W6 follows, with our scorer as the verifier |
| [DAPO](https://arxiv.org/abs/2503.14476) (Yu et al., 2025) | Token-level loss normalization (TRL's default `loss_type: dapo`), truncated-sample masking |
| [Understanding R1-Zero-Like Training (Dr. GRPO)](https://arxiv.org/abs/2503.20783) (Liu et al., 2025) | Length and difficulty biases in GRPO; `scale_rewards` |
| [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) (Zheng et al., 2025) | Sequence-level importance ratios; more stable for MoE models |
| [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) (Agarwal et al., 2023), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015) | Teacher → student training: the main alternative to RL for small models, with the production model as teacher |
| [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) | The trainer `scripts/train_grpo.py` uses: reward functions, VLM datasets, vLLM generation |
| [OpenAI reinforcement fine-tuning guide](https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning) | The same idea as a product: graders as rewards (the service is being deprecated; the method stands) |

## 8. Document AI and information extraction

| Read | Why it matters here |
|------|---------------------|
| ⭐ [LayoutLM](https://arxiv.org/abs/1912.13318) (Xu et al., 2019), [LayoutLMv3](https://arxiv.org/abs/2204.08387) (Huang et al., 2022) | Why the position of text matters; the idea behind spatial text |
| ⭐ [Donut: OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) (Kim et al., 2021) | Image → JSON without OCR, evaluated on CORD |
| [CORD dataset](https://github.com/clovaai/cord), [CORD v2 on Hugging Face](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | Our public dev set (CC BY 4.0) |
| ⭐ [LMDX](https://arxiv.org/abs/2309.10952) (Perot et al., 2023) | LLM extraction from OCR text with coordinates: the closest published analogue of our spatial-text flow. Also the schema-in-the-prompt design with explicit nulls, and the zero-shot (about 67 F1 on CORD) and few-shot curves for new schemas |
| [DocLLM](https://arxiv.org/abs/2401.00908) (Wang et al., 2023) | A layout-aware LLM that uses bounding boxes |
| [Scene Text VQA (ANLS)](https://arxiv.org/abs/1905.13648) (Biten et al., 2019) | Edit-distance scoring of text answers: the idea behind our NED metric |
| [DocVQA](https://arxiv.org/abs/2007.00398), [OCRBench](https://arxiv.org/abs/2305.07895), [OCRBench v2](https://arxiv.org/abs/2501.00321), [CC-OCR](https://arxiv.org/abs/2412.02210), [OmniDocBench](https://arxiv.org/abs/2412.07626) | The benchmarks that model cards quote ([models](models.md)) |
| [ReceiptBench](https://arxiv.org/abs/2605.22413), [UNIKIE-BENCH](https://arxiv.org/abs/2602.07038), [ExtractBench](https://arxiv.org/abs/2607.29677) (2026) | Key information extraction benchmarks closest to our task |
| [PaddleOCR-VL](https://arxiv.org/abs/2510.14528) (Cui et al., 2025) | A compact document parser; an option for the OCR step that produces spatial text |

### Extracting fields the model was not trained on (Q9, E20)

| Read | Why it matters here |
|------|---------------------|
| ⭐ [GoLLIE](https://arxiv.org/abs/2310.03668) (Sainz et al., 2023) | Labels with guidelines in the prompt; the ablations behind `fields.sampling`: field dropout, descriptions and example values; the gap between seen and unseen labels |
| [UniversalNER](https://arxiv.org/abs/2308.03279) (Zhou et al., 2023), [GLiNER](https://arxiv.org/abs/2311.08526) (Zaratiana et al., 2023) | Negative labels whose answer is empty (our decoys and nulls) are the largest single gain for unseen labels; teacher-labeled data plus human data beats either alone |
| ⭐ [NuExtract 2.0 model card](https://huggingface.co/numind/NuExtract-2.0-8B), [VAREX](https://arxiv.org/abs/2603.15118) (IBM, 2026) | A typed JSON template as the request (`verbatim-string` vs `string`); 1,771 unseen form schemas, where an extraction-tuned 2B model reaches 90.8% exact match and its base 9.7%; small models echo schemas with references |
| [GenKIE](https://arxiv.org/abs/2310.16131) (Cao et al., 2023) | A model trained on a fixed field set reaches 3–5 F1 on a held-out field: the risk of one fixed prompt |
| [ProMoT](https://arxiv.org/abs/2211.00635) (Wang et al., 2023), [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161) (Chu et al., 2025) | Narrow SFT learns the output format first and loses out-of-distribution ability; RL keeps it if the SFT start did not overfit (E18) |
| [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) (Gururangan et al., 2020), [Adapting LLMs via Reading Comprehension](https://arxiv.org/abs/2309.09530) (Cheng et al., 2023) | Why MLM-style continued pretraining is not the path: it always preceded labeled fine-tuning, and on decoder LLMs raw-text pretraining hurts prompt following |
| [Google Document AI custom extractor](https://docs.cloud.google.com/document-ai/docs/ce-with-genai) | The product version: Extract vs Derive fields, zero-shot → few-shot (5–10 documents) → fine-tuning (10–50+) per new field |

## 9. Vietnamese documents

| Read | Why it matters here |
|------|---------------------|
| [VMMU: A Vietnamese Multitask Multimodal Understanding and Reasoning Benchmark](https://arxiv.org/abs/2508.13680) (Dang et al., 2025) | OCR is a bottleneck for open VLMs on Vietnamese; check image-flow errors by field |
| [A Survey on Vietnamese Document Analysis and Recognition](https://arxiv.org/abs/2506.05061) (2025) | Diacritics, fonts and layouts that break OCR and normalization |
