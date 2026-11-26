# Schema- and instruction-conditioned universal IE: generalization to unseen labels/fields

Research date: 2026-11-26. I read primary sources (arXiv HTML, Hugging Face model cards, the NuMind blog) through a page-summarizing fetch tool. Items tagged **[not re-fetched]** come from my prior knowledge of the paper. Only the arXiv/venue link was cited for those, and their content was not re-checked in this session. Treat them as lower confidence. Numbers tagged **[self-reported]** come from a vendor's own benchmark with no independent replication.

Abbreviations: F1 is entity- or field-level F1. "Seen" and "unseen" refer to label types or fields that were or were not in the fine-tuning data.

---

## Q1. Methods and how they encode the schema

### Takeaway
Each method puts the label or field list into the input as a different object. UIE/USM/InstructUIE/IEPile use a flat label list or "options". GoLLIE/KnowCoder/Code4UIE use Python classes with docstring guidelines. UniversalNER asks one question per type. GLiNER/GLiNER2 use special-token label slots in a bidirectional encoder. NuExtract uses a typed JSON template with optional in-context examples. For a VLM doing document-to-nested-JSON, the NuExtract-style typed JSON template is the closest match. NuExtract3 (Qwen3.5-4B base) is a public, template-conditioned VLM built for exactly this task.

### Cited Findings
**Text-to-structure generative UIE (2022–2023)**
- UIE (Lu et al., ACL 2022) is a T5-based model. A "Structural Schema Instructor" (SSI) prefix lists the target types with special tokens such as [spot]/[asso]. Output is a linearized "Structured Extraction Language" (SEL). It was pre-trained on heterogeneous (Wikipedia/ConceptNet-aligned) data. **[not re-fetched]** — [arXiv 2203.12277](https://arxiv.org/abs/2203.12277)
- USM (Lou et al., AAAI 2023) recasts UIE as semantic matching between label text and token spans, which lets it read arbitrary label strings. **[not re-fetched]** — [arXiv 2301.03282](https://arxiv.org/abs/2301.03282). In GLiNER's out-of-domain NER comparison, USM averages 37.8 F1 against 60.9 for GLiNER-L — [GLiNER paper](https://arxiv.org/html/2311.08526)
- InstructUIE (Wang et al., 2023) is Flan-T5-11B instruction-tuned on the "IE INSTRUCTIONS" collection. The task instruction and label "options" are given in natural language. **[not re-fetched]** — [arXiv 2304.08085](https://arxiv.org/abs/2304.08085)
- RexUIE (EMNLP 2023 Findings) is a recursive UIE method with an "explicit schema instructor" that handles arbitrary schemas, including n-ary tuples. **[not re-fetched]** — [arXiv 2304.14770](https://arxiv.org/abs/2304.14770)

**Code-style schemas**
- GoLLIE (Sainz et al., HiTZ; ICLR 2024): each label is a Python class whose docstring holds the annotation guideline. "Representative candidates" are code comments on the main argument. Event-argument and slot-filling arguments get small per-argument definitions as comments. The output follows `result =` as a list of class instances, so running the output parses it. The backbone is Code-LLaMA 7B/13B/34B, trained with QLoRA for 3 epochs, batch size 32, LR 3e-4 — [GoLLIE arXiv HTML](https://arxiv.org/html/2310.03668)
- KnowCoder (ICT-CAS; ACL 2024) also writes schemas as code and trains in two phases. Phase 1, "schema understanding", is code pre-training on about 1.5B automatically constructed instances of definition and instance code. Phase 2, "schema following", is instruction tuning — [KnowCoder arXiv](https://arxiv.org/abs/2403.07969v1); [ACL Anthology PDF](https://aclanthology.org/2024.acl-long.475.pdf)
- Code4UIE is a retrieval-augmented, code-style prompting framework (Python class schemas plus retrieved in-context demonstrations) for LLMs used without fine-tuning. **[not re-fetched]** — [arXiv 2311.02962](https://arxiv.org/abs/2311.02962)

**Per-type question / instruction formats**
- UniversalNER (Zhou et al., ICLR 2024) uses a conversation-style format. The passage is given first, then one query per entity type: "What describes t_i in the text?". The answer is a JSON list of mentions, and loss is computed only on the outputs — [UniversalNER arXiv HTML v2](https://arxiv.org/html/2308.03279v2)
- IEPile (ZJU, 2024): each instruction has a task description, the input text, and a "schema" list of the labels to query. Output is a dictionary that includes negative schemas absent from the text. Each instruction queries a fixed number of schemas (`split_num`: 6 for NER, 4 for RE/EE per the appendix; the main text says 4–6) — [IEPile arXiv HTML](https://arxiv.org/html/2402.14710)
- ADELIE (EMNLP 2024) is an LLM aligned for IE through an instruction corpus followed by SFT and then DPO, while keeping general abilities. **[not re-fetched]** — [arXiv 2405.05008](https://arxiv.org/abs/2405.05008)
- SPIRES/OntoGPT (Caufield et al., Bioinformatics 2024) prompts a general LLM zero-shot and recursively against a LinkML schema, then grounds the extracted values to ontology IDs. It involves no fine-tuning. **[not re-fetched]** — [arXiv 2304.02711](https://arxiv.org/abs/2304.02711)

**Encoder-based "arbitrary label string" models**
- GLiNER (Zaratiana et al., NAACL 2024) puts the entity types and the text into one sequence for a DeBERTa-v3 bidirectional encoder: `[ENT] type1 [ENT] type2 ... [SEP] text`. Each span is scored against each type embedding with a sigmoid and BCE loss. Span length is capped at K=12 — [GLiNER arXiv HTML](https://arxiv.org/html/2311.08526)
- GLiNER2 (Fastino, arXiv v1 24 Jul 2025) is one encoder for NER, classification, and hierarchical structure extraction. The input is `[Task Prompt] [SEP] [Text]` with special tokens [P], [E], [C], [L]. Fields use the syntax `field_name::type::description` (type `str` or `list`). Choice fields use `field::[opt1|opt2]::type`. A 20-class MLP predicts how many parent instances exist (0–19). Released under Apache-2.0 as the `gliner2` package — [GLiNER2 arXiv HTML](https://arxiv.org/html/2507.18546)

**Template-conditioned generative extractors (NuMind)**
- NuExtract v1 (blog, 24 Jun 2024) uses a JSON skeleton with empty strings as placeholders, e.g. `{"reactants":[{"name":"","quantity":""}],"time":[""]}`. Values are strings only, and **field descriptions are deliberately not supported**: "the authors believe examples are more informative". Base models were Phi-3-mini (3.8B), Phi-3-small (7B), and Qwen1.5-0.5B — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)
- NuExtract 2.0 (2B/4B/8B on Qwen2-VL-2B / Qwen2.5-VL-3B / Qwen2.5-VL-7B) is a VLM that takes text and/or images. The template is typed JSON with `verbatim-string`, `string`, `integer`, `number`, `date-time`, arrays, `enum`, and `multi-label`. Absent values come back as `null`, or `[]` for arrays. Optional text or image `examples` give in-context learning. It can also generate a template from XML or a natural-language description — [NuExtract-2.0-8B card](https://huggingface.co/numind/NuExtract-2.0-8B)
- NuExtract3 is fine-tuned from Qwen/Qwen3.5-4B. The card calls it "4B", while the safetensors metadata shows 5B, likely because of the vision tower. It uses the same typed-JSON template plus more types (`date`, `time`, `country`, `currency`, `email`, ... in TYPES.md), enum and multi-enum constructors, and nested arrays of objects. An `instructions` chat-template kwarg is available. In-context examples go in `developer` messages. Reasoning is toggled with `enable_thinking`. A `numind` SDK converts Pydantic/JSON Schema into templates. Absent fields return `null`/`[]` — [NuExtract3 card](https://huggingface.co/numind/NuExtract3). Third-party write-ups claim "7 types in 2.0, +14 = 21 in v3" and a two-stage training recipe. These are not confirmed on the official card — [search aggregate, e.g. startupfortune](https://startupfortune.com/nuextract3-gives-startups-a-smaller-path-to-document-ai/)

**Table / document schema-driven IE**
- "Schema-driven IE from heterogeneous tables" (Bai et al., EMNLP 2024 Findings) turns tables from ML papers, chemistry, materials science, and webpages into records that follow a human-authored schema. Prompted LLMs reach F1 74.2–96.1 without task-specific labels, and distillation to smaller models is validated — [arXiv 2305.14336](https://arxiv.org/abs/2305.14336)
- K2Q (Zmigrod et al., JPMorgan; EMNLP 2024 Findings) converts KIE keys of five visually-rich-document datasets into diverse templated questions, extractive or boolean, sometimes covering multiple entities. Its finding: "creating diverse and intricate KIE questions enhances the performance and robustness of VRDU models" compared with the single template "What is the value for the {key}?" — [arXiv 2410.15484](https://arxiv.org/abs/2410.15484v1)
- Earlier form-specific work on unseen keys includes KATA (AAAI 2021, key/trigger pre-training) and QGN (ACM MM 2022, query-driven generation for unseen layouts and fields) — [KATA](https://ojs.aaai.org/index.php/AAAI/article/view/17494); [QGN PDF](https://jfma-ustc.github.io/publication/qgn/qgn.pdf)

### Inferences
- The team's target (image or OCR text to nested JSON) matches NuExtract's design: a typed JSON template, null/[] for absent fields, and optional examples. A code-class schema (GoLLIE) or a field-description string (GLiNER2) can be added to the same template as per-field "description/guideline" text. The model does not need a new interface.
- NuExtract3 shares its Qwen3.5-4B base with the team's candidates. It is the natural baseline for "schema-conditioned without our own data": run it zero-shot on the team's dev set before building a custom recipe.

### Gaps
- The exact training data and recipe for NuExtract 2.0/3 (SFT then RL?) are not described on the official cards. The NuExtract3 benchmark names a "NuExtract3.4_4B-RL" variant but gives no explanation. I did not find an official NuExtract 1.5/2.0/3 technical report in this session.
- I did not verify any 2025–2026 successor papers in detail beyond GLiNER2 and NuExtract3. Candidates to check include KnowCoder-X, B2NERD ([arXiv 2406.11192](https://arxiv.org/abs/2406.11192), which argues a smaller dataset with a consistent, diverse taxonomy beats raw dataset merges **[not re-fetched]**), SLIMER ([arXiv 2407.01272](https://arxiv.org/abs/2407.01272), definitions plus guidelines with few training types **[not re-fetched]**), and Cuckoo.

---

## Q2. Training tricks that matter for generalization to unseen fields

### Takeaway
The two tricks with the strongest controlled ablations are **negative/absent labels with empty outputs** and **few-shot exemplars inside the schema** (GoLLIE's "representative candidates"). Negatives gave +21.9 F1 for UniversalNER and +7.6 F1 for GLiNER. Candidates gave +5.4 F1 for GoLLIE, its largest single regularizer. Class/field dropout gives a small but significant gain. Shuffling field order, paraphrasing guidelines, and masking label names (anonymization) were **not significant** in GoLLIE's ablation. Two other signals: vary the number of fields per query (IEPile), and make negatives *semantically similar* fields (hard negatives).

### Cited Findings
**GoLLIE ablation** (Table 4: mean zero-shot F1 across held-out datasets, one-sided p-value against the full model) — [GoLLIE arXiv HTML](https://arxiv.org/html/2310.03668)
- Full GoLLIE 7B: 55.3
- w/o class-order shuffling: 55.9 (p=0.072), not significant and slightly higher
- w/o guideline paraphrases (made by Vicuna-33B v1.3): 54.8 (p=0.11), not significant
- w/o class-name masking (replacing names with `LABEL_1`...): 54.6 (p=0.10), not significant
- w/o class dropout (randomly deleting input classes and their output instances): 54.0 (p=0.004), significant but small
- w/o representative candidates (5 sampled from a pool of 10 per class): 49.9 (p=2.2e-10), the biggest single drop
- w/o everything (baseline: same data, no guidelines/regularizers): 42.3 (p=5.1e-13)
- The paper gives **no probabilities** for dropout, masking, or paraphrasing in the main text (my fetch was truncated in Appendix B)

**Negative label sampling**
- UniversalNER: negatives are types absent from the passage, sampled with probability proportional to their frequency in the dataset, with target `[]`. Average zero-shot F1 (Pile-tuned UniNER-7B) is **31.5 with no negatives, 47.7 with uniform negatives, and 53.4 with frequency-based negatives**. On the Movie group: 19.1 / 42.5 / 42.4 — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- GLiNER: negatives are types drawn from other examples in the same batch. P/R/F1 by negative ratio: **0% = 49.3/58.1/53.3; 50% = 62.3/59.7/60.9; 75% = 61.1/56.5/58.6**. Negatives mainly raise precision, and too many start to cost recall — [GLiNER HTML](https://arxiv.org/html/2311.08526)
- IEPile builds a "hard negative" dictionary that maps each positive schema to semantically similar ones (e.g. layoffs / depart / dismissals) and includes those as negatives, plus a few random ones. Removing the dictionary "has a limited effect on NER but notably hurts DuIE2.0 and DuEE1.0". The numbers appear only in Figure 3 — [IEPile HTML](https://arxiv.org/html/2402.14710)

**Schema size / subset sampling**
- IEPile reports a "schema query mismatch": a model trained with about 20 schema queries per instruction degrades "significantly" when tested with 10 or 30. Their fix is fixed-size batches of 4–6 schemas per instruction — [IEPile HTML](https://arxiv.org/html/2402.14710)
- GLiNER caps training at 25 entity types per sentence. It randomly drops entity types and shuffles their order, following Sainz et al. Type dropping gives "an average improvement of over 1.4 points" out of domain — [GLiNER HTML](https://arxiv.org/html/2311.08526)

**Label names vs definitions/descriptions**
- GoLLIE's error analysis credits guidelines with fixing polysemous labels (Media, Vulnerability Patch, Trailer, Task) that the no-guideline baseline failed on. Guidelines alone are not separately ablated from candidates, though: the baseline removes everything — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- UniversalNER: a model trained on **definitions** instead of type names underperformed the type-name model by 11.8% on its benchmark, which queries by label name. Its case study shows the definition model finding "Los Angeles" for "place" where the type-based model missed it, i.e. sensitivity to paraphrased label names — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- UniversalNER: adding the **dataset name** to the input resolves label-definition conflicts across datasets. Facility gains 22.0% and time gains 12.4%. The name is omitted at zero-shot time — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- NuExtract v1 chose examples over descriptions ("examples are more informative"). Each training sample carried 0–3 in-context output examples, a "hybrid few-shot setting" — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)
- K2Q: diverse, intricate question templates per key beat the single "What is the value for the {key}?" template for document VRDU models — [arXiv 2410.15484](https://arxiv.org/abs/2410.15484v1)

**Synthetic / LLM-generated schemas and diversity**
- NuExtract v1: an LLM *generated a template from each of 300k C4 texts*. Llama-3-70B filled the templates, and examples were dropped when the template wasn't followed or values weren't found in the text. That left **50k examples**. For half of them, part of the text was removed so some fields became empty — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)
- UniversalNER: 50k Pile passages (≤256 tokens) were annotated by gpt-3.5-turbo-0301, giving **45,889 examples, 240,725 entities, and 13,020 distinct types**. The type distribution is heavy-tailed: the top 1% of types account for 74% of mentions — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- GLiNER2: **254,334 examples** (135,698 real texts plus 118,636 synthetic emails, texts, resumes, posts, etc.), all annotated by GPT-4o — [GLiNER2 HTML](https://arxiv.org/html/2507.18546)
- KnowCoder: removing the code-pretraining "schema understanding" phase (about 1.5B auto-constructed definition and instance codes) costs an average 7.4% in zero-shot NER. Removing the "schema following" instruction phase causes drops of up to 50% — [KnowCoder search summary / arXiv](https://arxiv.org/abs/2403.07969v1)
- UniversalNER, out-of-domain groups (CrossNER/MIT): Pile-synthetic-only scores 53.4. Supervised-only training on 20 human-labeled datasets scores 57.2. Pile pre-tuning followed by supervised training scores **61.8** (the text says 60.0, which is an internal inconsistency) — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)

### Inferences
- A recipe that matches the evidence: (1) per sample, randomly pick a subset of the fields that exist (field dropout), always including some *absent* fields with null/[] targets; (2) bias negatives toward confusable sibling fields (hard negatives); (3) vary the field count per sample across the range expected at inference (IEPile mismatch); (4) give each field a short description plus 1–3 example values. Candidates were GoLLIE's strongest lever, and NuMind also prefers examples. (5) Mix a large set of LLM-generated, diverse document schemas with the team's human-labeled fixed-schema data (UniversalNER's synthetic+supervised = best).
- Label anonymization and order shuffling are cheap but should not be expected to matter much (GoLLIE). One reason to keep anonymization in a small fraction of samples: it forces reliance on descriptions, which is the regulation-change scenario. This is my inference and was not tested in these sources.

### Gaps
- None of the sources found gives a controlled ablation of **"number of distinct schemas vs number of examples"** for document KIE. UniversalNER shows the type distribution, and B2NERD reportedly addresses this **[not re-fetched]**.
- No ablation of description vs bare name for *document fields* specifically.
- Dropout, masking, and paraphrase rates for GoLLIE were not retrieved (they are in the appendix past my fetch window).

---

## Q3. Quantitative evidence: unseen vs seen, gap size, and few-shot closing

### Takeaway
On text NER benchmarks, schema-conditioned models lose **about 3–10 F1 points on unseen label types versus seen ones** (GoLLIE: 7.1 at 7B, 3.3 at 34B). Absolute zero-shot F1 on new domains sits around 50–60, against roughly 80–90 when supervised in domain. A small number of labeled examples (tens to ~100) recovers a meaningful part of the gap, and schema-conditioned pre-training helps most in that low-data regime. Public, independently replicated evidence for **document/form** fields is thin. The best available numbers are NuMind's self-reported internal benchmark.

### Cited Findings
**Seen vs unseen labels**
- GoLLIE (Table 7, average micro-F1, seen / unseen labels on held-out datasets): Baseline 41.2 / 31.6 (gap 9.6). GoLLIE-7B 54.6 / 47.5 (gap 7.1). 13B 54.2 / 48.0 (gap 6.2). 34B 54.7 / 51.4 (gap 3.3). The gap shrinks with scale, and some datasets show seen labels with *different* guidelines scoring below unseen ones — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- IEPile: on datasets whose schemas *resemble* training ones, zero-shot comes close to supervised. ACE2004 scores 77.01 zero-shot vs 84.28 supervised. People Daily scores 95.29 vs 98.34 — [IEPile HTML](https://arxiv.org/html/2402.14710)

**Zero-shot levels (text)**
- GoLLIE zero-shot averages: baseline 42.3, 7B 55.3, 13B 56.0, 34B 57.2. On datasets with a prior SoTA: SoTA 42.6, GoLLIE-7B 58.4, 34B 60.0. The hardest set is FabNER at 24.8–26.3; MultiNERD reaches 76–77 — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- UniversalNER: across the 43-dataset benchmark, UniNER-7B scores 41.7 and 13B scores 43.4, against 34.9 for ChatGPT. On the CrossNER+MIT out-of-domain groups: ChatGPT 47.5, Vicuna-7B 13.0, UniNER-7B 53.4, UniNER-13B 55.6. Supervised in domain on 20 datasets: UniNER-7B 84.78, InstructUIE-11B 81.16, BERT-base 80.09 — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- GLiNER (CrossNER+MIT zero-shot average): GLiNER-S 52.7, M 55.4, L 60.9 (DeBERTa-large, about 0.3B), against GoLLIE 58.0, UniNER-13B 55.6, ChatGPT 47.5, InstructUIE 47.2, USM 37.8. On the 20-benchmark zero-shot average: GLiNER-L 47.8, UniNER-7B 45.7, ChatGPT 36.5. Supervised: 82.9 vs UniNER-7B 84.8 — [GLiNER HTML](https://arxiv.org/html/2311.08526)
- GLiNER2 (CrossNER zero-shot F1): GLiNER2 0.590, GLiNER-M 0.615, GPT-4o 0.599. Zero-shot classification average: 0.72 vs GPT-4o 0.84. Hierarchical structure extraction was "not evaluated due to the absence of established zero-shot benchmarks" — [GLiNER2 HTML](https://arxiv.org/html/2507.18546)
- IEPile zero-shot, English: CrossNER F1 55.55–57.90 depending on the backbone, vs GPT-4 58.49 and InstructUIE 49.36. RE average 36.4–39.7, vs GPT-4 23.10 — [IEPile HTML](https://arxiv.org/html/2402.14710)
- KnowCoder-7B: 60.1 average F1 on 7 zero-shot NER datasets, a 12.5% relative gain over baselines. Low-resource setting: +21.9% relative — [KnowCoder arXiv](https://arxiv.org/abs/2403.07969v1)

**Few-shot / small fine-tune closing the gap**
- GLiNER: Pile-NER (schema-diverse) pre-training adds **+5.6 F1 at 100 labeled samples per dataset**, and the advantage narrows toward 10k samples. Pre-training adds +0.8 F1 on the supervised average — [GLiNER HTML](https://arxiv.org/html/2311.08526)
- NuExtract v1 chemistry task (Iktos data): 5-fold cross-validation, **fine-tuned on 40 examples per fold**. Afterwards NuExtract-tiny (0.5B) "became a bit better than GPT-4o", and the 3.8B/7B models were "on a different level". Scores appear only in a chart. **[self-reported]** NuMind also states that in-context learning "quickly saturates" — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)

**Documents / forms**
- NuExtract3 card, NuMind internal structured-extraction benchmark (~600 docs: invoices, receipts, forms, contracts, floor plans, ...; vLLM, T=0.25) **[self-reported; benchmark not yet released]**: NuExtract3.4_4B-RL **0.651±0.019**, **gemma-4-E4B-it 0.538±0.023**, Qwen3.5-9B 0.479±0.030, GLM-4.6V-Flash 0.435±0.026, **Qwen3.5-4B 0.417±0.031**, Nemotron-3-Nano-Omni 0.387, Ministral-3-3B 0.240. The card does not compare against frontier API models or NuExtract 2.0 — [NuExtract3 card](https://huggingface.co/numind/NuExtract3)
- ExtractBench on the model cards: NuExtract3 Mean 82.35 (Short 88.06, Medium 76.76) vs NuExtract-2.0-8B Mean 49.42 (Short 62.92, Medium 23.5). The drop from Short to Medium suggests length sensitivity, especially for 2.0. The two cards may use different ExtractBench versions; this is unverified — [NuExtract3 card](https://huggingface.co/numind/NuExtract3); [NuExtract-2.0-8B card](https://huggingface.co/numind/NuExtract-2.0-8B)
- The NuExtract 2.0 card cites "~1,000 diverse extraction examples" while the GitHub README cites "~600 documents" for the internal benchmark. These conflict, possibly because of different versions — [NuExtract-2.0-8B card](https://huggingface.co/numind/NuExtract-2.0-8B); [GitHub](https://github.com/numindai/nuextract)
- GenKIE on SROIE zero/few-shot: near-zero F1 at 0 shots for company/address/date/total. Few-shot helped semantically common fields such as date most — [GenKIE arXiv](https://arxiv.org/pdf/2310.16131) (search snippet only)
- KATA (AAAI 2021): ">70% accuracy for zero-shot keys while previous methods all fail" on form-like documents — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17494) (abstract snippet only)
- Schema-driven table IE: F1 74.2–96.1 with no task-specific labels — [arXiv 2305.14336](https://arxiv.org/abs/2305.14336)

### Inferences
- The best directly comparable scenario is the regulation one: 5 trained fields and 2 new fields of the same document type. Here the new fields share the domain, layout, and context, which is easier than GoLLIE's "unseen dataset + unseen label". Expect the seen/unseen gap to be at or below GoLLIE's 3–7 F1 *if* the new fields come with good descriptions or examples and are not near-duplicates of existing fields. This is extrapolation, not measured.
- On NuMind's own benchmark (self-reported), Gemma-4-E4B-it already beats Qwen3.5-4B and 9B zero-shot on schema-conditioned document extraction. A template-specialized 4B (NuExtract3) adds about 0.11 on top. This gives the team a prior that the base model matters and that template-specialized training adds a clear margin.
- A practical policy follows from the GLiNER and NuExtract few-shot results: ship zero-shot for the new fields immediately, then label about 50–200 docs with **all 7 fields** for a short LoRA refresh.

### Gaps
- I found no public benchmark that measures **held-out fields within the same document type** (e.g. train on 5 invoice fields, test on 2 others) for VLMs. The team should build this split themselves.
- QGN, K2Q, and GenKIE numbers were not retrieved; only abstract-level claims are given.
- No independent replication of the NuExtract3 internal benchmark. The release date is not on the card; a third-party article is dated 2026-06-16.
- No few-shot curves (1/5/10/50 examples) specific to new *document fields*.

---

## Q4. Failure modes

### Takeaway
Recurring failures: confusing similar or fine-vs-coarse labels; over-generation, i.e. filling every queried field, which is the absent-field hallucination problem; guideline-annotation mismatch; catch-all fields; sensitivity to how labels are phrased; and degradation on long or multi-image inputs and deep nested lists. Negatives, hard negatives, guidelines, and examples mitigate some of these but none eliminates them.

### Cited Findings
- **Over-generation:** IEPile models "tend to produce outputs for every queried schema" — [IEPile HTML](https://arxiv.org/html/2402.14710)
- **Similar-field confusion:** IEPile predicted "dismissal"/"resignation" for a "layoff" event, and favored coarse "person" over fine "scientist" — [IEPile HTML](https://arxiv.org/html/2402.14710). GoLLIE labels scientists as Person and Political Party as Organization (a strong prior when no training set had the label) — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- **Guideline vs annotation mismatch:** GoLLIE tags years as MultiNERD "Time" because the guideline allows it, but the dataset doesn't annotate them. Catch-all labels (Miscellaneous, Movie "Plot") are hard for models and humans alike — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- **Cross-dataset definition conflicts** caused instability. IEPile kept only CoNLL2003 out of four "person" datasets — [IEPile HTML](https://arxiv.org/html/2402.14710). UniversalNER handles this with dataset-name tags — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- **Label phrasing sensitivity:** UniversalNER's type-based model misses mentions when the type is paraphrased ("place"). Long-tail types show high variance — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- **Copy vs generate trade-off:** NuExtract v1's verbatim-only training made it fail on values that need generation (counting bikes). NuMind relaxed this later with separate `verbatim-string` and `string` types — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction); [NuExtract-2.0-8B card](https://huggingface.co/numind/NuExtract-2.0-8B)
- **Long documents:** NuExtract v1 trained on texts capped at about 1,200 words and listed long documents as future work — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction). NuExtract-2.0-8B falls from 62.92 (Short) to 23.5 (Medium) on ExtractBench — [card](https://huggingface.co/numind/NuExtract-2.0-8B). NuMind's GitHub notes that its benchmark includes multi-image samples with large context — [GitHub](https://github.com/numindai/nuextract)
- **Reasoning mode in small models:** the NuExtract3 card says small reasoning models "often fall into repetition loops, hit output limits, and produce non-parseable JSON". It recommends non-thinking mode at T=0.2 by default — [NuExtract3 card](https://huggingface.co/numind/NuExtract3)
- **Nested lists:** GLiNER2 predicts instance counts with a 0–19 classifier, a hard ceiling of 19 instances per parent, and did not evaluate hierarchical extraction — [GLiNER2 HTML](https://arxiv.org/html/2507.18546)
- **Too many negatives hurt recall:** GLiNER recall fell from 59.7 at 50% negatives to 56.5 at 75% — [GLiNER HTML](https://arxiv.org/html/2311.08526)

### Inferences
- For the team's KIE, the biggest risk with 2 new fields is that they are near-synonyms of existing fields (e.g. "issue_date" vs "effective_date"). Hard-negative pairs and explicit descriptions that contrast them are the main lever. The eval should report per-field confusion, not only micro-F1.
- Absent-field hallucination should be measured directly: a null-precision metric on docs where the field is truly absent.

### Gaps
- No quantified hallucination rate on absent fields for document VLMs in the sources found.
- No systematic study of deep nesting (lists of objects within lists) error rates.

---

## Q5. What ground truth must look like for schema-conditioned training

### Takeaway
Every field in the schema must be **exhaustively annotated for every document**, and absent values must be stored as explicit `null`/`[]`. Otherwise field sampling turns unannotated-but-present values into false negatives. Each field needs one consistent definition: a description, a type, and ideally a few example values. Conflicting definitions of the same field name across sources must be reconciled or tagged.

### Cited Findings
- Negative or absent queries are trained to output empty lists (UniversalNER) or `""`/`[]` (NuExtract v1). NuExtract 2.0/3 output `null`/`[]` — [UniversalNER HTML](https://arxiv.org/html/2308.03279v2); [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction); [NuExtract3 card](https://huggingface.co/numind/NuExtract3)
- NuExtract v1 deliberately removed text from half of the training samples to create true absent fields, and filtered out samples whose values could not be found in the text (grounding check) — [NuMind blog](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction)
- GoLLIE's class dropout removes a class *and its output instances* together, so the targets stay consistent with the schema shown — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- Definition conflicts across datasets hurt. IEPile dropped conflicting "person" datasets, and UniversalNER added dataset-name tags (+22.0% on facility) — [IEPile HTML](https://arxiv.org/html/2402.14710); [UniversalNER HTML](https://arxiv.org/html/2308.03279v2)
- Guidelines must match how annotators actually labeled (the MultiNERD "Time" mismatch) — [GoLLIE HTML](https://arxiv.org/html/2310.03668)
- Typed fields (`verbatim-string` vs `string`, `date-time` ISO, `enum`, `multi-label`) let the target encode normalization rules — [NuExtract-2.0-8B card](https://huggingface.co/numind/NuExtract-2.0-8B)

### Inferences
- The team's current fixed-schema labels can be reused as long as every label set is complete for its doc type. Field-subset sampling can then be done offline: drop fields from both the template and the target. Absent fields need explicit nulls in the stored ground truth, not missing keys.
- Write a field registry: name, type, description, 1–3 examples, and "do not confuse with X". It serves both as the template text and as the annotation guideline, which avoids the GoLLIE guideline/annotation mismatch.
- To evaluate the regulation scenario, hold out whole fields, not just documents, from training. Doing this requires the held-out fields to be annotated in the test set only.

### Gaps
- No source quantifies how much partial (non-exhaustive) annotation hurts schema-conditioned training. The recommendation above is inferred from how negatives are constructed.
- No public guidance from NuMind on preparing fine-tuning data for NuExtract3. Only a 2.0 fine-tuning notebook is referenced on the card.
