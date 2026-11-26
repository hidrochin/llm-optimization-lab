# Narrow LoRA/QLoRA SFT on a fixed prompt/schema: forgetting, schema generalization, mitigations, and interaction with W4A16 PTQ

Context: Gemma 4 E2B/E4B/12B and Qwen3.5 2B/4B/9B VLMs, LoRA/QLoRA SFT on about 2,000 samples per document type. Each doc type always uses the same system prompt and JSON schema. The adapter is then merged and quantized to W4A16 (GPTQ/AWQ via LLM Compressor, calibrated on the same task) for vLLM, with GRPO planned on top. The question is whether the model will still follow an edited schema (new fields) after a regulation change.

Notes compiled 2026-11-26. Dates are arXiv first-submission dates unless stated. "(snippet)" marks a claim taken from a search-result excerpt or a tool paraphrase of an abstract that was not checked against the full paper.

## Q1. Does narrow SFT (especially LoRA) cause forgetting, loss of instruction following, or format overfitting? Will the model ignore new keys?

### Takeaway
Yes, the risk is real and well documented. Single-format fine-tuning teaches the format first and fastest ("format specialization"). Models trained on a fixed label or slot set overfit to the names they saw and do much worse on unseen labels. LoRA reduces forgetting compared with full fine-tuning but does not remove it. No paper found tests exactly "add a new JSON key to a fixed KIE schema after LoRA SFT", so the conclusion for that case is extrapolated from close analogues in IE, dialogue state tracking (DST) and ICL.

### Cited Findings
**Format specialization (closest mechanism to the team's setup)**
- ProMoT (Wang et al., first posted Nov 2022, ICLR 2024). The authors fine-tuned mT5-XXL (13B) on RTE (outputs "True"/"False") and then gave it 1-shot TriviaQA prompts. After 300 steps, 90% of outputs were "True"/"False". With 1-shot WMT16 En-De prompts, more than 99% were "True"/"False" after 500 steps. Few-shot QA accuracy "drops drastically" as RTE accuracy rises. — [arXiv 2211.00635](https://arxiv.org/abs/2211.00635)
- Same paper: "format" covers language, typical input/output lengths and styles, special tokens, punctuation and casing. Because every fine-tuning example shares one format, "the model receives a strong gradient signal that the output should follow this format". — [arXiv 2211.00635](https://arxiv.org/abs/2211.00635)
- Same paper: format is learned first. At step 0, the full gradient is almost parallel to the gradient from label-shuffled data (cosine ≈ 1), which carries format information only. By step 400 the cosine falls to about 0.21, when the True/False output ratio is near 100%. — [arXiv 2211.00635](https://arxiv.org/abs/2211.00635)
- Caveat: ProMoT used full fine-tuning of an encoder-decoder model that was not instruction-tuned, on single-label classification. The team uses LoRA on instruction-tuned decoder VLMs with long JSON outputs, so the effect size will differ. — [arXiv 2211.00635](https://arxiv.org/abs/2211.00635)

**Overfitting to seen keys, slots or labels (direct analogue of "does it ignore new keys?")**
- Task-oriented dialogue, 2025: a model trained without schema augmentation "tends to overfit to slot names it saw in training". Schema augmentation gave its largest gain on API-call parameter names. (snippet) — [arXiv 2502.13310](https://arxiv.org/abs/2502.13310)
- Schema Augmentation for zero-shot DST (Richardson et al., Nov 2024): the model is trained with synonym or non-semantic encoded slot names. This gives up to a 2× gain over strong baselines on unseen domains (MultiWOZ 2.1, SpokenWOZ) and keeps performance on all domains equal or better. The implication is that plain training on fixed names generalizes poorly to new names. (snippet/abstract) — [arXiv 2411.00150](https://arxiv.org/abs/2411.00150)
- GoLLIE (Sainz et al., Oct 2023, ICLR 2024), seen vs unseen labels. The baseline IE model fine-tuned without guidelines scored 41.2 F1 on seen labels and 31.6 on unseen labels. GoLLIE-7B, trained with guidelines plus regularization, scored 54.6 and 47.5. The gap shrinks with size: 34B scores 54.7 and 51.4. On the Politics dataset the baseline fell to 4.6 F1 on unseen labels, against 31.9 for GoLLIE-7B. — [arXiv 2310.03668](https://arxiv.org/abs/2310.03668)
- GoLLIE: in-domain supervised F1 was nearly identical (73.3 vs 73.0), so the generalization techniques cost almost nothing on the trained schema. — [arXiv 2310.03668](https://arxiv.org/abs/2310.03668)
- Instruction-tuned models are not robust to new phrasings. Across 319 practitioner-written instructions for more than 80 tasks, novel but appropriate phrasings consistently lowered performance and increased variance (Sun et al., Jun 2023). — [arXiv 2306.11270](https://arxiv.org/abs/2306.11270)

**LoRA vs full fine-tuning forgetting**
- "LoRA Learns Less and Forgets Less" (Biderman et al., May 2024, TMLR Aug 2024). Setup: code and math, about 100K instruction pairs, and 20B tokens of continued pretraining. In standard low-rank settings LoRA "substantially underperforms" full fine-tuning on the target domain, but it better maintains base-model performance outside that domain. It regularizes more strongly than weight decay or dropout and keeps generations more diverse. Full fine-tuning learns perturbations of rank "10-100X greater than typical LoRA configurations". — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- LoRA still forgets. Forgetting follows a shifted power law in the number of fine-tuned parameters and the number of update steps. Neither early stopping nor changing the number of trained parameters avoids it (Kalajdzievski, Jan 2024). — [arXiv 2401.05605](https://arxiv.org/abs/2401.05605)
- "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (Shuttleworth et al., Oct 2024, v3 Oct 2025): LoRA introduces "intruder dimensions" (new high-ranking singular vectors) in the weight matrices. These are associated with forgetting the pretraining distribution, and the effect is worse in continual or sequential learning. — [arXiv 2410.21228](https://arxiv.org/abs/2410.21228)
- Conflicting 2026 evidence: in a Brazilian-healthcare domain study, full fine-tuning lost 4–10 IFEval points. LoRA underperformed full fine-tuning both in-domain and out-of-domain, which the authors attribute to intruder dimensions. This contradicts Biderman et al. (snippet, unreviewed preprint) — [arXiv 2605.01077](https://arxiv.org/abs/2605.01077)
- LoRA rank sweep (Rathore et al., Dec 2025, AACL-IJCNLP 2025): LoRA is "competitive and in some cases superior" to full SFT at some ranks. Forgetting is task-specific and comes with representational drift and changed attention patterns. The abstract gives no numbers. — [arXiv 2512.15634](https://arxiv.org/abs/2512.15634)

**General forgetting and instruction-following erosion after narrow SFT**
- Luo et al. (Aug 2023) report catastrophic forgetting in 1B–7B LLMs during continual instruction tuning. General instruction tuning beforehand (e.g., BLOOMZ vs mT0, Alpaca vs LLaMA) helps reduce it. — [arXiv 2308.08747](https://arxiv.org/abs/2308.08747)
- Qi et al. (Oct 2023, ICLR 2024): fine-tuning on a handful of adversarial examples breaks safety alignment; the abstract reports 10 examples costing under $0.20 on GPT-3.5 Turbo. Benign fine-tuning (e.g., Alpaca-style data) also degrades safety somewhat. This shows that alignment behaviour, including instruction following, is fragile under small SFT. — [arXiv 2310.03693](https://arxiv.org/abs/2310.03693)
- Lin et al. (Sep 2023), originally titled "Speciality vs Generality", now "Mitigating the Alignment Tax of RLHF" (v4 Oct 2024). Fine-tuning for a speciality trades off general abilities. Model averaging gave the best trade-off of the methods compared, and the authors propose Heterogeneous Model Averaging. — [arXiv 2309.06256](https://arxiv.org/abs/2309.06256)
- Domain SFT on recent models (Lin et al., Sep 2025, ICLR 2026): general-capability degradation is real, but "using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance". No method fully removes the trade-off. — [arXiv 2509.20758](https://arxiv.org/abs/2509.20758)
- After standard SFT, Qwen3-4B's average output length fell from about 1,720 to 572 characters, with large IFEval drops across baselines. Part of the "forgetting" is weakened long-form instruction compliance. (snippet, 2026 preprint) — [arXiv 2605.20005](https://arxiv.org/abs/2605.20005)
- CapTrack (2026): instruction fine-tuning causes the strongest capability drift, and preference optimization is more conservative. IFEval sometimes rises after domain training, so a single benchmark can hide losses. (snippet) — [arXiv 2603.06610](https://arxiv.org/abs/2603.06610)
- SFT collapses out-of-distribution on a VLM (Chu et al., Jan 2025; base Llama-3.2-Vision-11B). Under rule shifts, SFT out-of-distribution performance fell from 80.8% to 1.3% on V-IRL-L, from 35.7% to 2.5% on V-IRL-VL, and from 11.2% to 5.6% on GeneralPoints-VL. — [arXiv 2501.17161](https://arxiv.org/abs/2501.17161)
- Fine-tuning on generation tasks and on classification tasks generalizes differently. Adding in-context examples during fine-tuning (FTICL) improves generalization on generation tasks (Yang et al., Mar 2024, NAACL 2024). — [arXiv 2403.09162](https://arxiv.org/abs/2403.09162)

### Inferences
- The team's setup has every ingredient of format specialization: one constant system prompt, one constant key set per doc type, homogeneous JSON output, about 2,000 near-identical-format samples, and several epochs. The gradient signal "emit exactly these keys in this order" is strong and learned early, before field semantics, by analogy with ProMoT.
- Likely failure modes after a schema edit, extrapolated and not directly measured:
  1. New keys are silently dropped and the old schema is emitted.
  2. A new key is emitted but its value is copied from a semantically close old field.
  3. Old keys keep appearing even after removal from the schema.
  4. Instructions placed in the prompt (e.g., "date format DD/MM/YYYY") are ignored where they conflict with the trained format.

  Mode 1 is the most likely, since the DST evidence ([2502.13310](https://arxiv.org/abs/2502.13310)) shows overfitting to seen slot names.
- Several factors reduce the risk compared with ProMoT: the bases are instruction-tuned ([2308.08747](https://arxiv.org/abs/2308.08747)), LoRA is a regularizer ([2405.09673](https://arxiv.org/abs/2405.09673)), and the doc-type schemas differ from each other, which gives some schema variety. If one LoRA adapter covers several doc types with different key sets, it has already seen "follow the schema in the prompt" behaviour, which helps somewhat. Separate adapters per doc type would be the most prone to format specialization.
- Smaller models (E2B, Qwen3.5-2B) are likely to forget and overfit more than 9B/12B. GoLLIE's unseen-label gap shrinks from 7.1 F1 (7B) to 3.3 (34B) ([2310.03668](https://arxiv.org/abs/2310.03668)).

### Gaps
- No primary study measures "add or rename a JSON field after LoRA SFT on a fixed KIE schema" for document VLMs (Donut/LayoutLM-style or Gemma/Qwen VLMs). The answer has to be tested empirically on the team's own models (see Q4).
- No found results specific to Gemma 4 or Qwen3.5 on post-SFT instruction-following retention.
- The intruder-dimension claims and the conflicting healthcare result (LoRA forgetting more) were not verified against full text.

## Q2. Mitigations: data mixing/replay, schema/prompt augmentation, LoRA rank/steps/LR, KL regularization, weight interpolation/merging, instruction-tuned base, and RL (GRPO) vs SFT

### Takeaway
The strongest evidence for schema generalization comes from schema-aware data augmentation (GoLLIE, schema augmentation, IEPile). The main techniques are randomized field subsets (class dropout), definitions of each field in the prompt, synonym or encoded key names, and negative or absent fields. A smaller learning rate, fewer steps, replay of general data, self-distillation and weight interpolation (scaling the LoRA delta) further reduce general forgetting. On-policy RL such as GRPO forgets much less than SFT and generalizes better out of distribution. RL still needs SFT first to fix the output format, and it cannot rescue an over-fitted SFT checkpoint.

### Cited Findings
**Schema and prompt augmentation (most directly relevant)**
- GoLLIE regularizations ([arXiv 2310.03668](https://arxiv.org/abs/2310.03668)):
  - The five techniques are class order shuffling, class dropout (randomly remove classes from both the input schema and the output), guideline paraphrasing, representative candidate sampling (5 of 10 example values per class), and class-name masking (LABEL_1 …).
  - Ablation of mean zero-shot F1 (full model 55.3):

    | Removed | F1 | Change | Significance |
    |---|---|---|---|
    | Dropout | 54.0 | −1.3 | p = 4e-3 |
    | Candidates | 49.9 | −5.4 | significant |
    | Masking | 54.6 | −0.7 | not significant |
    | Paraphrases | 54.8 | −0.5 | not significant |
    | Shuffling | 55.9 | +0.6 | not significant |
    | All guidelines (baseline) | 42.3 | −13.0 | significant |

  - Field definitions and example values in the prompt matter most; class dropout gives a small but significant gain.
- Synonym Schema Augmentation swaps slot names for synonyms; Encoding Schema Augmentation replaces names with non-semantic codes so the model must read the descriptions. Both give up to 2× gains on unseen domains without hurting seen domains. (abstract/snippet) — [arXiv 2411.00150](https://arxiv.org/abs/2411.00150)
- IEPile (Gui et al., Feb 2024) is a bilingual EN/ZH IE instruction corpus of about 0.32B tokens. It builds schema-based instructions from 33 datasets and improves zero-shot IE. — [arXiv 2402.14710](https://arxiv.org/abs/2402.14710)
- Two IEPile details are recalled from the paper but not re-verified this session. Schemas are split into batches with sampled subsets of keys, and hard-negative schemas (keys absent from the document, which must be returned empty) are added. — [arXiv 2402.14710](https://arxiv.org/abs/2402.14710)
- Instruction paraphrase robustness: Sun et al. improve robustness by aligning the representations of equivalent instructions with soft prompts. This supports training on paraphrased system prompts. — [arXiv 2306.11270](https://arxiv.org/abs/2306.11270)
- ProMoT: offload format learning to a soft prompt in stage 1, then fine-tune the model with that prompt attached. This keeps in-domain accuracy comparable to standard fine-tuning while losing much less in-context ability, and can even improve semantically related ICL tasks. — [arXiv 2211.00635](https://arxiv.org/abs/2211.00635)

**Data mixing, replay and self-distillation**
- When math, code and general data are mixed in SFT, general abilities plateau after about 1K samples while math and code keep scaling. Sequential training (specialist after general) risks catastrophic forgetting, and the proposed Dual-stage Mixed Fine-tuning (DMT) mitigates it (Dong et al., Oct 2023, ACL 2024). — [arXiv 2310.05492](https://arxiv.org/abs/2310.05492)
- Self-Distillation Fine-Tuning (SDFT, Feb 2024) has the model rewrite the target responses in its own distribution before fine-tuning. On Llama-2-chat it reduces forgetting while matching or beating vanilla fine-tuning downstream, and helps preserve helpfulness and safety. — [arXiv 2402.13669](https://arxiv.org/abs/2402.13669)
- Joint training beats sequential training: sequential SFT followed by RLHF/DPO is provably suboptimal because the first stage is forgotten. A joint framework improves results by up to 23% (Fernando et al., Oct 2024). — [arXiv 2410.15483](https://arxiv.org/abs/2410.15483)

**Hyperparameters: LR, steps, rank, layers**
- A smaller learning rate substantially mitigates general degradation at comparable domain performance. Token-Adaptive Loss Reweighting (TALR) beats L2 regularization, LoRA, model averaging and FLOW on the trade-off, but no method eliminates it (Sep 2025, ICLR 2026). — [arXiv 2509.20758](https://arxiv.org/abs/2509.20758)
- Forgetting grows as a power law with the number of steps and trained parameters, which argues for fewer steps and lower rank but gives no guarantee. — [arXiv 2401.05605](https://arxiv.org/abs/2401.05605)
- "LoRA Without Regret" (Schulman et al., Thinking Machines, 29 Sep 2025) ([blog](https://thinkingmachines.ai/blog/lora/)):
  - LoRA matches full fine-tuning on small-to-medium SFT datasets.
  - Apply LoRA to all layers, especially MLP; attention-only LoRA underperforms.
  - The optimal LoRA LR is about 10× the full-fine-tuning LR, and about 15× for runs under roughly 100 steps (anecdotal).
  - LoRA is less tolerant of large batch sizes.
  - For policy-gradient RL, LoRA even at rank 1 matches full fine-tuning, because RL absorbs about 1 bit per episode.
  - This is a blog post, not peer-reviewed.

**Weight interpolation and merging**
- Model averaging (interpolating between the pre- and post-fine-tuning weights) gave the strongest alignment-vs-forgetting Pareto front among the methods compared. Heterogeneous (layer-wise) averaging improves it further (Lin et al., EMNLP 2024 version). — [arXiv 2309.06256](https://arxiv.org/abs/2309.06256)
- WiSE-FT (Wortsman et al., 2021/CVPR 2022) is the original weight-space ensembling between zero-shot and fine-tuned weights, which improves robustness to distribution shift. Its numbers were not re-verified this session. — [arXiv 2109.01903](https://arxiv.org/abs/2109.01903)

**RL (GRPO) vs SFT: forgetting and generalization**
- SFT Memorizes, RL Generalizes (Chu et al., Jan 2025, Llama-3.2-Vision-11B) ([arXiv 2501.17161](https://arxiv.org/abs/2501.17161)):
  - Under rule shift, out-of-distribution changes for RL vs SFT:

    | Task | RL | SFT |
    |---|---|---|
    | GeneralPoints-L | +3.5% | −8.1% |
    | V-IRL-L | +11.0% | −79.5% |
    | V-IRL-VL | +9.3% | −33.2% |

  - Under visual shift, RL gains +17.6% (GP-VL) and +61.1% (V-IRL-VL).
  - SFT was required first: without it, every end-to-end RL run on GeneralPoints-L failed, because the base model did not follow the output format. SFT "stabilizes the model's output format".
  - RL could not recover an over-fitted SFT checkpoint.
- RL's Razor (Shenfeld, Pari, Agrawal, Sep 2025) ([arXiv 2509.04259](https://arxiv.org/abs/2509.04259)):
  - Setup: Qwen2.5-3B-Instruct trained with GRPO vs SFT on math, science QA and tool use.
  - RL reaches new-task gains while prior-task scores stay nearly unchanged; SFT forgets substantially.
  - Forward KL(base‖fine-tuned) on the new-task distribution predicts forgetting: R² = 0.96 on ParityMNIST and 0.71 on the LLM experiments.
  - On Wikipedia probe data, RL representations stay close to the base model (CKNNA 0.94) while SFT representations drift (0.56).
  - SFT on data distilled from the RL policy matches RL's trade-off, so on-policy data is the cause.
  - Scaling SFT from 3B to 14B did not remove forgetting.
  - LoRA was not studied.
- Retaining by Doing (Chen et al., Oct 2025): across Llama and Qwen, RL forgets less than SFT at matched or better target performance. The authors attribute this to RL's mode-seeking behaviour on on-policy data. Approximately on-policy data (e.g., the model's own samples) captures much of the benefit more cheaply. — [arXiv 2510.18874](https://arxiv.org/abs/2510.18874)

### Inferences
- **Training-data recipe for future schema changes** (adapted from GoLLIE, IEPile and schema augmentation):
  1. Per sample, randomly drop 10–50% of fields from both the prompt schema and the target JSON.
  2. Shuffle key order.
  3. Include a short natural-language definition and format rule for each field in the prompt.
  4. Sometimes rename keys with synonyms or opaque codes such as `field_07`, keeping the description, to force reliance on the description.
  5. Add "absent" fields that must return null.
  6. Paraphrase the system prompt.
  7. Mix doc types in one adapter.

  This directly trains "follow whatever schema is in the prompt". Exact rates are a judgement call, not from a paper.
- Hold out some real fields entirely during training, as a simulated "regulation change" test set (see Q4).
- Replay: mixing in a small slice of general or Vietnamese instruction data, or the base model's own responses to generic prompts (SDFT-style or approximately on-policy), is cheap with only about 2,000 task samples per doc type. Based on Dong et al., about 1K general samples may already be enough to hold general ability.
- Interpolation for LoRA: scaling the merged delta (W = W₀ + λ·BA with λ < 1, equivalent to lowering alpha at merge time) is a free WiSE-FT-style knob. Sweep λ in {0.5, 0.75, 1.0} on both the in-distribution test set and the held-out-schema test set before quantizing. This is extrapolated from [2309.06256](https://arxiv.org/abs/2309.06256); LoRA-specific evidence was not found.
- Keep the instruction-tuned checkpoint as the base, not the pretrained base ([2308.08747](https://arxiv.org/abs/2308.08747)).
- Lower the LR, use the minimum epochs and early-stop on held-out-schema accuracy rather than in-distribution loss ([2509.20758](https://arxiv.org/abs/2509.20758), [2401.05605](https://arxiv.org/abs/2401.05605)).
- **GRPO plan:** run light SFT for format, then GRPO with a schema-conditioned reward (per-field F1 over the fields requested in the prompt), sampling randomized schemas during RL. Because RL changes the base distribution less (lower KL), the result should keep more general instruction following than more SFT epochs would. The SFT stage must not over-fit, because RL cannot fix that ([2501.17161](https://arxiv.org/abs/2501.17161)). LoRA is adequate for GRPO even at low rank (Thinking Machines blog).
- **Regulation-change response:** a fast and robust option is a short incremental SFT on a few hundred examples with the new fields, mixed with replay of old-schema data, or a GRPO pass with the new schema. Constrained decoding with the new JSON schema in vLLM structured outputs can force the new keys to appear. It cannot guarantee correct values, and it can hide the problem by filling the key with garbage. The vLLM docs ([structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)) were not fetched this session.

### Gaps
- No found paper compares these mitigations head-to-head for document KIE with VLMs or for Vietnamese.
- No direct evidence on λ-scaling of LoRA deltas for forgetting, or on KL-to-base regularization during SFT specifically (beyond the KL analysis in RL's Razor).
- RL's Razor and Retaining by Doing used full fine-tuning on text LLMs. Whether GRPO with LoRA on a VLM keeps the same forgetting advantage is unverified.

## Q3. How does W4A16 PTQ (GPTQ/AWQ) interact with instruction following, long context, out-of-distribution or multilingual generalization, and calibration-set choice?

### Takeaway
For 7B+ models on short English benchmarks, W4A16 with group size 128 is close to lossless: about 98–99% recovery, and near-lossless even on reasoning. Several factors make it worse:
- small models in the 1–4B range;
- long inputs (up to 59% loss at 4-bit for inputs over 64K tokens, worse for non-English);
- non-English languages, where automatic metrics underestimate the human-perceived drop by up to about 10×;
- GPTQ with narrow calibration, which over-fits the calibration distribution (AWQ is much more robust).

A model calibrated only on fixed-schema prompts therefore risks a larger loss exactly on the out-of-distribution case the team cares about: new prompts or schemas in Vietnamese.

### Cited Findings
**Headline accuracy**
- "Give Me BF16 or Give Me Death" (Kurtic et al., Neural Magic/Red Hat, Nov 2024) ran over 500K evaluations on the Llama-3.1 family. FP8 is effectively lossless, well-tuned INT8 loses 1–3%, and W4A16 is competitive with 8-bit. Deployment guidance is based on vLLM. — [arXiv 2411.02355](https://arxiv.org/abs/2411.02355)
- On reasoning models from 1.5B to 70B, W8A8 and W4A16 are near-lossless, while lower bit-widths carry significant risk (Liu et al., Apr 2025). — [arXiv 2504.04823](https://arxiv.org/abs/2504.04823)
- In-context learning, chain-of-thought and instruction following survive 4-bit quantization; 2-bit models degrade severely (Liu et al., Jul 2023). — [arXiv 2307.08072](https://arxiv.org/abs/2307.08072)
- Evaluating Quantized LLMs (Li et al., Feb 2024, ICML 2024) covers 11 model families from 125M to 180B and five task types, quantizing weights, activations and KV cache. The finding that long-context and some emergent abilities are more sensitive is recalled from the paper, not re-verified. — [arXiv 2402.18158](https://arxiv.org/abs/2402.18158)
- Lee et al. (Sep 2024) tested instruction-tuned models from 1B to 405B with 4 methods on 13 datasets. FP8 is the most robust. AWQ outperforms GPTQ in weight-only quantization. 4-bit causes severe drops for smaller models. The hardest tasks do not always lose the most. — [arXiv 2409.11055](https://arxiv.org/abs/2409.11055)

**Long context**
- Mekala et al. (May 2025) evaluated 9.7K examples with inputs over 64K tokens, 5 methods and 5 models. 8-bit loses about 0.8%; 4-bit methods lose up to 59% on long-context tasks, and degradation is worse for non-English inputs. — [arXiv 2505.20276](https://arxiv.org/abs/2505.20276)

**Multilingual and non-English**
- Marchisio et al. (Cohere, Jul 2024) ([arXiv 2407.03211](https://arxiv.org/abs/2407.03211)):
  - In Japanese, a 1.7% average drop on automatic tasks corresponds to a 16.0% drop in human evaluation on realistic prompts.
  - At W4 on the 103B model, the relative drop is −0.7% for Latin-script Indo-European languages and −1.9% for the others; CJK languages are hit hardest.
  - Degradation correlates with the language's share of mC4 training data, and the correlation strengthens as bits drop (R² = 0.24 at W8, 0.63 at W4).
  - Math (MGSM) loses 13.1% for 35B with GPTQ W4-g, LLM-as-judge scores fall 25.9%, and human-judged hard prompts fall 10.5%.
  - Vietnamese appears only in the extended Aya-23 evaluations.
- "Calibrating Beyond English" (Chimoto et al., EACL 2026) ([arXiv 2601.18306](https://arxiv.org/abs/2601.18306), [ACL Anthology](https://aclanthology.org/2026.eacl-long.223)):
  - Setup: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, 4-bit group size 128 with GPTQ and AWQ, 10 languages (no Vietnamese).
  - Non-English and multilingual calibration beat English-only calibration, by up to 3.52 perplexity points for GPTQ.
  - GPTQ swings up to 3.52 ppl with calibration language, against about 0.35 for AWQ, which the authors attribute to GPTQ's Hessian-based compensation.
  - Recommendations: a balanced multilingual set for general deployment; for AWQ, match the target language; for GPTQ, prefer multilingual calibration plus some code or math.
  - Downstream gains were smaller than perplexity gains.
- Quantization and machine translation (Aug 2025, 55 language pairs): 4-bit often preserves quality for high-resource languages and large models. Low-resource and typologically diverse languages degrade most, especially at 2-bit, and language-matched calibration helps mainly at low bit-widths. (snippet) — [arXiv 2508.20893](https://arxiv.org/abs/2508.20893)
- "Multilinguality after Quantization" (Waskito et al., Findings of AACL 2026) studied GPTQ and mixed-precision variants across 5 language families. It argues that calibration-data selection must be a deliberate design choice in multilingual PTQ. (snippet) — [ACL Anthology](https://aclanthology.org/2026.findings-aacl.9/)

**Calibration data: domain, size, diversity**
- AWQ (Lin et al., Jun 2023, MLSys 2024 Best Paper) ([arXiv 2306.00978](https://arxiv.org/abs/2306.00978)):
  - AWQ needs 16 calibration sequences where GPTQ needs 192.
  - With cross-domain calibration (PubMed vs Enron, OPT-6.7B at INT3-g128), mismatch raises perplexity by only 0.5–0.6 for AWQ, against 2.3–4.9 for GPTQ.
  - The rationale: AWQ "does not rely on any backpropagation or reconstruction", only on per-channel activation magnitudes, whereas GPTQ's reconstruction can over-fit the calibration set.
  - On the multimodal OpenFlamingo-9B (COCO CIDEr, INT4-g128): FP16 scores 81.70, RTN 77.13, GPTQ 74.98 and AWQ 80.53. AWQ was also reported lossless on 11 VILA benchmarks.
- Williams & Aletras (Nov 2023, ACL 2024) ([arXiv 2311.09755](https://arxiv.org/abs/2311.09755)):
  - Setup: 9 models, 5 calibration sources, 10 sets each of 128×2048 tokens.
  - The range in mean zero-shot accuracy across calibration sets was 0.9–1.6% for GPTQ and 0.6–1.0% for SpQR. Pruning was more sensitive, at up to 4.8%.
  - Quantization is less sensitive than pruning, but individual tasks can swing more.
  - Perplexity is a misleading proxy: WikiText perplexity stayed at 12.72 ± 0.18 while BoolQ ranged from 57.0% to 71.6% (pruning case).
  - Recommendations: vary the calibration sets, inspect them, and release them.
- Self-calibration (Williams et al., Oct 2024): calibration data generated by the model itself is competitive with real data and often better. — [arXiv 2410.17170](https://arxiv.org/abs/2410.17170)
- LLM Compressor docs ([docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-dataset/)):
  - "128-512 samples is sufficient for most models"; "Start with 128-256 samples; increase if accuracy recovery is insufficient".
  - "The calibration dataset should be representative of your target use case."
  - `ultrachat-200k` is the suggested starting point for instruction and chat models, and `razorcal` (a multi-domain chat corpus) for mixed-domain or MoE workloads.
  - The page does not cover sequence length, chat templates, VLM calibration or multilingual data.
- A Red Hat LLM Compressor lesson example uses 256 samples at max sequence length 4096; longer sequences show the quantizer realistic context lengths. (snippet) — [Red Hat Developers](https://developers.redhat.com/learning/learn:openshift-ai:get-started-vllm/resource/resources:optimize-model-llm-compressor)

**QAT vs PTQ**
- Gemma 3 QAT (Google, Apr 2025): about 5,000 QAT steps using the non-quantized checkpoint's probabilities as targets reduced the perplexity drop by 54% at Q4_0 (llama.cpp perplexity) compared with PTQ. — [Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- Gemma 4 QAT checkpoints were released on 5 Jun 2026, and Google says they give higher quality than standard PTQ baselines. The release date and the absence of a repeated 54% figure come from a search snippet; the format/vLLM compatibility of the checkpoints was not checked. — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)

### Inferences
- **The calibration choice compounds format specialization.** Calibrating GPTQ only on fixed-schema task prompts fits the Hessian-based error compensation to that one distribution. AWQ's PubMed/Enron result and Chimoto's GPTQ language-sensitivity result both suggest GPTQ will lose more on new prompts or schemas than on in-distribution data. AWQ, or GPTQ with a diverse calibration mix, should be the safer default.
- **Suggested calibration mix** for 256–512 samples at realistic length (full prompt plus OCR or image tokens, plus JSON):
  - about 50% task samples across all doc types, with randomized or augmented schemas and paraphrased prompts;
  - about 25% general Vietnamese instruction or chat data;
  - about 25% general multilingual or English chat data such as ultrachat.

  Apply the chat template, and build several calibration sets to measure sensitivity, following Williams & Aletras. This is a judgement call based on the cited work, not a tested recipe.
- **Vietnamese risk:** Vietnamese is Latin-script but non-Indo-European, with diacritics and less web data than major European languages. Under Marchisio's data-size correlation, expect more degradation than for English. Automatic field-F1 may underestimate it, for example through diacritic errors in extracted names or addresses. Field-level exact match on diacritic-bearing fields is a good canary.
- **Model size:** severe 4-bit drops for small models ([2409.11055](https://arxiv.org/abs/2409.11055)) mean Gemma 4 E2B and Qwen3.5-2B are the main risk; 9B and 12B should be close to lossless. Consider W8A8/FP8 for the 2B models if latency allows.
- **Long multi-page documents:** if inputs plus image tokens get long, 4-bit long-context loss ([2505.20276](https://arxiv.org/abs/2505.20276)) is a separate risk to test.
- **VLMs:** LLM Compressor VLM examples typically leave the vision tower and projector unquantized and apply W4A16 to the language model only. This was not verified this session; check the current examples for Gemma 4 and Qwen3.5.
- **QAT over the fine-tuned model:** fine-tuning changes the weights, so Gemma's QAT checkpoints do not transfer the QAT benefit to a merged LoRA model unless QAT or quantization-aware LoRA is redone after SFT. This is an inference.

### Gaps
- No found study measures quantization degradation specifically in Vietnamese, or for Gemma 4 or Qwen3.5 under W4A16.
- No paper found directly tests "fine-tuned model + task-only calibration vs mixed calibration" on out-of-distribution prompts or schemas; this needs an in-house ablation.
- Whether PTQ amplifies the format specialization of an SFT'd model, i.e., whether the held-out-schema gap widens after quantization, has not been measured in the literature found.
- No evidence was found on quantization effects on VLM document OCR/KIE specifically, beyond AWQ's OpenFlamingo/VILA results.

## Q4. Practical guidance: how to test that a fine-tuned + quantized extractor still generalizes

### Takeaway
Build a schema-generalization test suite before the regulation change: held-out fields, renamed fields, removed fields, paraphrased prompts and a few general instruction-following probes. Run it at each stage: base, SFT (BF16 merged), λ-scaled merges, GRPO, and W4A16 with each calibration set. Report the gap between seen and unseen fields alongside in-distribution F1, and add a small human or diacritic-sensitive check for Vietnamese.

### Cited Findings
- **Seen vs unseen label split as a metric.** GoLLIE reports F1 on seen vs unseen labels per dataset; the gap fell from 9.6 for the baseline to 3.3 for GoLLIE-34B. This is a template for a "held-out field" metric. — [arXiv 2310.03668](https://arxiv.org/abs/2310.03668)
- **Schema perturbation tests.** Synonym-renamed and code-encoded slot names test whether the model reads descriptions rather than memorized names. — [arXiv 2411.00150](https://arxiv.org/abs/2411.00150)
- **Instruction paraphrase robustness.** Evaluate several practitioner-written paraphrases of the same instruction and report the variance as well as the mean. — [arXiv 2306.11270](https://arxiv.org/abs/2306.11270)
- **IFEval as a canary.** IFEval is among the benchmarks most sensitive to forgetting after domain SFT. PEFT-Arena pairs target-domain scores with IFEval, NQ and BBH for retention. (snippets) — [arXiv 2605.01077](https://arxiv.org/abs/2605.01077); [arXiv 2605.28819](https://arxiv.org/abs/2605.28819)
- **Output-length drift as a symptom.** Qwen3-4B's output length fell from about 1,720 to 572 characters after SFT. (snippet) — [arXiv 2605.20005](https://arxiv.org/abs/2605.20005)
- **Cheap forgetting proxy.** Forward KL between the base and fine-tuned models on task inputs predicts forgetting: R² = 0.96 on MNIST, 0.71 on LLMs. — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259)
- **Do not trust perplexity alone for quantized models.** Perplexity was stable while downstream accuracy swung by 14.6 points (pruning case). — [arXiv 2311.09755](https://arxiv.org/abs/2311.09755)
- **Use several calibration sets and report the variance.** — [arXiv 2311.09755](https://arxiv.org/abs/2311.09755)
- **Automatic metrics underestimate the non-English harm of quantization** (1.7% automatic vs 16.0% human in Japanese), so add human or LLM-judge spot checks. — [arXiv 2407.03211](https://arxiv.org/abs/2407.03211)
- **Test long inputs separately for 4-bit.** — [arXiv 2505.20276](https://arxiv.org/abs/2505.20276)

### Inferences
**Concrete test plan**
1. **Held-out-field split.** For each doc type, remove 2–4 real fields from all training prompts and targets but keep them labeled in the dev set. At test time, request the full schema. Report:
   - F1 on the seen fields;
   - F1 on the held-out fields;
   - the rate at which held-out keys are present in the output;
   - the rate of hallucinated or old keys.

   This simulates the regulation change. Repeat with a few different held-out sets.
2. **Schema edits.**
   - Rename keys with Vietnamese or English synonyms.
   - Reorder keys.
   - Drop keys, and check the model does not emit them.
   - Add a "decoy" key that is absent in the document, and check the output is null.
   - Change format instructions, for example a date format or a currency without separators.

   Measure compliance per instruction.
3. **Prompt paraphrase set.** Use 5–10 rewordings of the system prompt and report the mean and standard deviation of field F1.
4. **General canaries.** Run a small Vietnamese and English IFEval-style set of verifiable instructions, a few open VQA or general chat prompts, and an output-length check. Compare against the untouched base checkpoint.
5. **Stage-wise matrix.** Cover base (zero-shot with the schema), SFT BF16, SFT with λ-scaled merge, SFT+GRPO, W4A16-AWQ, W4A16-GPTQ (task-only calibration), and W4A16-GPTQ (mixed calibration). Run all tests above on every stage.

   The key decision metric is the drop on held-out fields from BF16 to W4A16, compared with the drop on seen fields. If quantization widens that gap, change the calibration set or the method.
6. **Vietnamese-specific checks.** Use diacritic-exact match on name, address and organization fields. A small human review of disagreements between the BF16 and W4A16 outputs is cheap and catches what F1 misses.
7. **Decision rule for the regulation change.**
   - If held-out-field F1 is within a few points of seen-field F1 at W4A16, prompt or schema editing alone is viable.
   - If not, plan a short incremental SFT or GRPO pass on the new fields, with replay, and re-quantize.
   - Either way, constrained JSON decoding in vLLM should be on so that key presence is structurally guaranteed. Value accuracy for new keys still has to be measured.

### Gaps
- There is no public Vietnamese document-KIE benchmark with held-out-schema splits; the team will need to build its own from CORD-like or internal data.
- No standard metric was found for "schema adherence under edits" for VLM extractors; the metrics above are proposed rather than drawn from an established benchmark.
