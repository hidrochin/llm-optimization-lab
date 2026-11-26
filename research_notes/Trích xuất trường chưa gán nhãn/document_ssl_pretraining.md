# Self-supervised pretraining objectives for document understanding, and whether they let a model extract new fields without new labels

Research date: 2026-11-26. Scope: evidence for or against the proposal to add BERT-style self-supervised tasks (MLM, NSP) on in-house documents so that a LoRA/QLoRA-fine-tuned small VLM (Gemma 4 E2B/E4B/12B, Qwen3.5 2B/4B/9B; image or layout-preserving OCR text in, nested JSON with a fixed per-doc-type field set out) can extract newly required fields (5 fields becoming 7) with no new labels and no re-fine-tuning.

Method note: claims marked "(verified in full text)" were checked against the paper's PDF text (pdftotext + grep). Claims marked "(abstract/page only)" or "(search snippet)" were not checked against full text.

---

## Q1. What pretraining objectives did document models use, and what did each teach?

### Takeaway
Document SSL objectives fall into four families: (a) masked text/layout reconstruction (LayoutLM MVLM, StrucTexT, LiLT, UDOP, DocLLM infilling); (b) cross-modal alignment (LayoutLMv2 TIA/TIM, LayoutLMv3 WPA); (c) geometric/relational (GeoLayoutLM, StrucTexT PBD); (d) generative "read or parse the page" (Donut pseudo-OCR, Pix2Struct screenshot-to-HTML, Kosmos-2.5 text-with-boxes + markdown, Nougat, Florence-2 OCR-with-region). Every family teaches reading, layout and alignment. None teaches the mapping from a field name to a value. In the ablations that exist, the gains from the extra objectives are small for KIE but large for parsing-style objectives on QA tasks.

### Cited Findings

**Encoder era (token-classification models)**
- **LayoutLM (Xu et al., 2019/KDD 2020).** Pretrained with Masked Visual-Language Model (MVLM: mask tokens but keep their 2-D position embeddings) plus a supervised Multi-label Document Classification (MDC) loss on IIT-CDIP tags. The paper says MDC "does bring substantial improvements on the FUNSD dataset." Best FUNSD F1 was 0.7927 with text+layout+image, vs BERT-base 0.603 and BERT-large 0.656 (verified in full text) — [LayoutLM](https://arxiv.org/abs/1912.13318)
- **LayoutLMv2 (Xu et al., 2020/ACL 2021).** Adds Text-Image Alignment (TIA: fine-grained, covers 15% of text lines in the image and predicts which tokens are covered) and Text-Image Matching (TIM: coarse-grained, 15% of images replaced). The ablation says both "improve the model performance substantially, and the proposed TIA benefits the model more than the commonly used TIM" (verified in full text) — [LayoutLMv2](https://arxiv.org/abs/2012.14740)
- **LayoutLMv3 (Huang et al., 2022).** Unified MLM + Masked Image Modeling (MIM, BEiT-style discrete image tokens) + Word-Patch Alignment (WPA: is the image patch of an unmasked word masked?). The ablation (BASE, 1M docs, 150K steps) shows the added objectives matter little for KIE (verified in full text) — [LayoutLMv3](https://arxiv.org/abs/2204.08387):
  - FUNSD F1: no image + MLM 88.64 → linear image + MLM 89.39 → +MIM 89.19 → +MIM+WPA 89.78.
  - CORD F1: 96.27 → 96.11 → 96.30 → 96.49.
  - Without MIM, linear patch embedding diverged on PubLayNet ("Loss Divergence"). MIM mainly matters for image-centric tasks.
- **StrucTexT.** MVLM + Segment Length Prediction + Paired Box Direction (spatial relations between segments). **Fast-StrucText** adds Graph-based Token Relation, Sentence Order Prediction and Text-Image Alignment (per the survey's summary) — [Ding et al. survey, 2024/2025](https://arxiv.org/abs/2408.01287)
- **LiLT (ACL 2022).** Decouples a layout stream from a swappable text model, so a layout encoder pretrained in one language can be paired with monolingual or multilingual text encoders. The survey lists its objectives as MVLM + Key Point Location + a text-layout matching task. The LiLT abstract page itself does not name the objectives — [LiLT](https://arxiv.org/abs/2202.13669); objectives via [survey](https://arxiv.org/abs/2408.01287)
- **GeoLayoutLM (CVPR 2023).** Three geometry pretraining tasks (GeoPair, GeoMPair, GeoTriplet per the survey) plus relation heads that are themselves pre-trained. The authors argue that "the mismatch between pre-training and fine-tuning objectives" limits relation extraction. FUNSD RE F1 went from 80.35 to 89.45 (abstract only) — [GeoLayoutLM](https://arxiv.org/abs/2304.10759)
- **Other encoder objectives** catalogued by the survey: BROS area-masked LM, StructuralLM cell position classification, LayoutMask masked position modeling, WUKONG textline-region contrastive learning, and others — [survey](https://arxiv.org/abs/2408.01287)

**Generative / encoder-decoder / decoder era**
- **Donut (Kim et al., ECCV 2022).** Pretraining task: "read all texts in the image in reading order", i.e. next-token cross-entropy described as "a pseudo-OCR task", on 11M IIT-CDIP scans + 2M SynthDoG synthetic images. Their analysis found that "synthetic images were enough for the document IE task", while real images mattered for DocVQA. Other general vision-language tasks such as captioning "show little gains" (verified in full text) — [Donut](https://arxiv.org/abs/2111.15664)
- **Pix2Struct (Lee et al., ICML 2023).** Pretraining: parse a masked web-page screenshot into simplified HTML, after a short warm-up "reading curriculum" on rendered BooksCorpus text. The authors describe recovering the unmasked parse as OCR-like and the masked parse as MLM-like. Ablation (base, 100K steps), as DocVQA / Widget Captioning / TextCaps (verified in full text) — [Pix2Struct](https://arxiv.org/abs/2210.03347):
  - Full recipe: 67.8 / 137.5 / 84.2.
  - Without warm-up: 56.2 / 128.0 / 71.7.
  - Without masking: 55.7 / 129.4 / 77.4.
  - Without screenshot parsing: 12.2 / 35.1 / 24.2. This is "effectively reducing the pretraining to reading linear text", and it was the biggest drop.
- **UDOP (Tang et al., CVPR 2023).** Self-supervised tasks are Joint Text-Layout Reconstruction (masked text plus its layout tokens), Layout Modeling (predict box for given text), Visual Text Recognition (predict text at given box) and Masked Image Reconstruction. It also includes supervised tasks in pretraining: 11 labeled datasets, 1.8M examples, including "Information Extraction" on DocBank, KLC, PWC and DeepForm. Ablation (DocVQA / RVL-CDIP; verified in full text) — [UDOP](https://arxiv.org/abs/2212.02623):

  | Objective | DocVQA | RVL-CDIP |
  |---|---|---|
  | MLM | 79.7 | 95.3 |
  | Joint Text-Layout | 82.8 | 95.4 |
  | + Visual Text Recognition | 83.3 | 95.4 |
  | + Layout Modeling | 84.0 | 95.6 |
  | + Image Reconstruction | 84.4 | 96.2 |
  | + Supervised | 85.0 | 96.3 |

- **DocLLM (JPMorgan, arXiv Dec 2023/Jan 2024).** Decoder LLM with a spatial attention stream over OCR boxes. Pretraining replaces next-token prediction with "autoregressive block infilling": mask OCR text blocks, then generate them conditioned on both prefix and suffix. The rationale is that documents are "sparse and irregular, featuring isolated and disconnected text fragments". The ablation measures only next-token accuracy on held-out pretraining data (causal 32.6, causal+spatial 36.2, block infilling+spatial 39.1). **No downstream-task ablation of the objective is reported** — [DocLLM](https://arxiv.org/abs/2401.00908)
- **Kosmos-2.5 (Microsoft, 2023/2024).** Pretrained on 357.4M text-rich pages with two generative tasks under task prompts: document-level text recognition (text lines with discretized bounding boxes, `<ocr>`) and image-to-markdown (`<md>`) (verified in full text) — [Kosmos-2.5](https://arxiv.org/abs/2309.11419)
- **Nougat (Meta, Aug 2023).** Visual Transformer doing OCR of scientific PDFs into a markup language, aimed at math (abstract only) — [Nougat](https://arxiv.org/abs/2308.13418)
- **Florence-2 (Microsoft, Nov 2023).** Prompt-token multitask model, including `<OCR>` and `<OCR_WITH_REGION>`, trained on FLD-5B ("5.4 billion annotations across 126 million images") — [Florence-2 model card](https://huggingface.co/microsoft/Florence-2-large); [Florence-2 paper](https://arxiv.org/abs/2311.06242)
- **mPLUG-DocOwl 1.5 (2024).** "Unified Structure Learning" on DocStruct4M (4.0M samples). It parses documents into layout-preserving text, tables into markdown with COLSPAN/ROWSPAN tokens, and charts into markdown tables, plus word/phrase/line/block text grounding and recognition with coordinates 0–999. Ablation (DocVQA / TabFact / ChartQA): no structure learning 72.8 / 72.9 / 65.0 → +parsing 77.7 / 76.5 / 67.5 → +parsing + multi-grained localization 81.6 / 80.4 / 70.5 — [DocOwl 1.5](https://arxiv.org/abs/2403.12895)
- **LayoutLLM (CVPR 2024).** "Layout-aware pre-training" is instruction-formatted. It has 5.7M instructions at a 1:4:4 document/region/segment ratio. Segment level includes MVLM, mask-position and geometric-layout instructions. Region level uses labeled layout and table datasets (PubLayNet, DocLayNet, DocBank, PubTabNet). Ablation: layout-aware pretraining gave +1.49 DocVQA and +3.06 FUNSD (QA-for-VIE); LayoutCoT SFT added +1.96 and +5.96 more (verified in full text) — [LayoutLLM](https://arxiv.org/abs/2404.05225)

### Inferences
- The objectives teach reading (OCR-like), layout (2-D position and reading order) and modality alignment. None teaches the mapping from a field name such as "tax_code" to its value. That mapping is learned only in fine-tuning or instruction tuning.
- Measured gains from adding SSL objectives on top of a basic text objective are about 1 F1 point on FUNSD and 0.2 on CORD in LayoutLMv3. Parsing-style objectives give large gains on QA (Pix2Struct, UDOP, DocOwl 1.5), but those models were pretrained from scratch or near-scratch on reading. Current VLMs already have this skill (see Q4).
- Classic BERT MLM is not a native objective for decoder-only VLMs such as Gemma 4 and Qwen3.5. The closest decoder analogues are next-token continued pretraining on document text, infilling (DocLLM) and "read the page" generation (Donut, PaliGemma `ocr`, Kosmos-2.5). The proposal would need to be restated in one of these forms.

### Gaps
- DocFormer (2021) objectives were not verified in this session. From memory they are multi-modal MLM, learn-to-reconstruct and text-describes-image, but this is unconfirmed, and the survey section I read cut off before them.
- No document-SSL paper I found reports an ablation where the SSL objective is measured on **extraction of fields absent from fine-tuning**. All ablations measure in-distribution FUNSD/CORD/DocVQA after fine-tuning.

---

## Q2. How was downstream KIE solved in these papers? Fixed label heads? Any zero-shot extraction of never-labeled field types?

### Takeaway
Every SSL document model above still needed labeled fine-tuning for KIE, with a fixed label set or field-token vocabulary. Encoders put a per-token classifier over predefined labels. Donut adds per-field special tokens. Pix2Struct and UDOP fine-tune per task. Zero-shot extraction of new fields shows up only when (1) the schema or key name is given in the prompt to a generative model and (2) that model was trained on many labeled extraction tasks with varied schemas (LMDX, DocLLM, NuExtract-style), or is a large general instruction-tuned LLM/VLM. Even then, zero-shot accuracy is well below fine-tuned accuracy. A small generative KIE model trained on a fixed field set scored about 3–5 F1 on a held-out field.

### Cited Findings

**Fixed-label-set encoders (cannot output a new field without a new head and labels)**
- LayoutLM fine-tunes FUNSD as "a sequence labeling problem. We pass the final representation into a linear layer followed by a softmax layer to predict the label of each token". For SROIE, "the number of semantic slots is fixed with pre-defined keys" (verified in full text) — [LayoutLM](https://arxiv.org/abs/1912.13318)
- LayoutLMv2: FUNSD, CORD, SROIE and Kleister-NDA "define entity extraction tasks that aim to extract the value of a set of pre-defined keys, which we formalize as a sequential labeling task" (verified in full text) — [LayoutLMv2](https://arxiv.org/abs/2012.14740)
- LayoutLMv3: "We predict the label of the last hidden state of each text token with a linear layer and an MLP classifier". CORD has "30 semantic labels defined under 4 categories" (verified in full text) — [LayoutLMv3](https://arxiv.org/abs/2204.08387)
- The survey describes most KIE models as treating KIE as "a sequence tagging task" over entity categories. It says pretrained models "often require detailed annotations" and "a performance gap remains between well-fine-tuned models and few/zero-shot frameworks" — [Ding et al. survey](https://arxiv.org/abs/2408.01287)

**Generative models pretrained with SSL still fine-tune per task or schema**
- Donut: "We add new special tokens for the prompt for each downstream task". Output fields are wrapped as `[START name] … [END name]`, "where ∗ indicates each field to extract", and "all downstream tasks [are] a JSON prediction problem" learned in fine-tuning (verified in full text) — [Donut](https://arxiv.org/abs/2111.15664)
- Pix2Struct reports "single-task state of the art" with separate fine-tuning per benchmark (verified in full text) — [Pix2Struct](https://arxiv.org/abs/2210.03347)
- UDOP fine-tunes on FUNSD/CORD ("Finetuning training details are available in Appendix D.6"). It also puts labeled IE datasets into pretraining as supervised tasks (verified in full text) — [UDOP](https://arxiv.org/abs/2212.02623)
- Kosmos-2.5-chat was "fine-tuned … on the standard training sets of ten diverse document understanding datasets", including the KIE sets DeepForm and KLC, before evaluation on their test sets. Here "without task-specific fine-tuning" means one multitask model, not zero-shot (verified in full text) — [Kosmos-2.5](https://arxiv.org/abs/2309.11419)
- PaliGemma base checkpoints "are not 'user (or benchmark) friendly'" and "need to be transferred" by fine-tuning per task (verified in full text) — [PaliGemma](https://arxiv.org/abs/2407.07726)

**Direct evidence on new or held-out fields and schemas**
- **GenKIE (EMNLP Findings 2023)** is the closest analogue to "5 → 7 fields". It is a prompt-based generative KIE model (OFA backbone) on SROIE. They hold one of the 4 entity types out of training, then train with k labeled instances of it (verified in full text, Table 6) — [GenKIE](https://arxiv.org/abs/2310.16131):

  | held-out field | 0-shot | 1-shot | 5-shot | 10-shot | full |
  |---|---|---|---|---|---|
  | company | 4.0 | 3.7 | 34.18 | 42.43 | 96.97 |
  | address | 3.9 | 3.6 | 12.38 | 55.60 | 97.22 |
  | date | 3.2 | 14.53 | 77.21 | 78.21 | 97.45 |
  | total | 5.0 | 4.9 | 45.92 | 66.33 | 97.83 |

  The authors: performance is "relatively limited compared to full-shot training", and few-shot helps most for fields "with common semantics such as date".
- **LMDX (Google, arXiv Sept 2023; ACL Findings 2024).** PaLM 2-S / Gemini Pro are first fine-tuned on a "Base Entity Extractor" mixture: the Payment dataset plus synthetic-filled government PDF form templates with schema annotations. The target schema is placed in the prompt. "No document or schema contained in the base extraction training phase overlap with … our target benchmarks", so the |D|=0 results are zero-shot on new schemas (verified in full text) — [LMDX](https://arxiv.org/abs/2309.10952):
  - CORD micro-F1 at |D| = 0 / 10 / 50 / 200 / 800 training docs: LMDX-PaLM 2-S 66.95 / 90.02 / 91.40 / 93.40 / 94.51.
  - LayoutLMv3 0.00 / 74.04 / 85.78 / 93.59 / 95.66.
  - Donut 0.00 / 26.15 / 65.68 / 75.85 / 81.55.
  - Zero-shot general models on CORD: GPT-4V+Image 64.05, Gemini Pro+OCR 59.57, LLaVA-1.5-13B 5.97.
  - VRDU Registration Form (Unseen Template), zero-shot: LMDX-PaLM 2-S 74.94 vs LayoutLMv3 at 10 docs 21.17.
- **DocLLM.** KIE instruction template: "{document} What is the value for the \"{key}\"?" (plus a "None" answer when absent). In the held-out-dataset setting (STDD: train on 11 datasets, test on unseen KLC/BizDocs), DocLLM-7B gets KLC F1 49.9 vs GPT-4+OCR 45.9 zero-shot. The authors only *hypothesize* that instructions "teach DocLLM to correlate names of keys in the prompts with document fields". No key-overlap analysis is given, so field novelty is untested — [DocLLM](https://arxiv.org/abs/2401.00908)
- **QueryForm (Google, 2022/2023).** Zero-shot means a new target *document type*, using schema prompts plus entity prompts, with pretraining on ~1.2M webpage examples whose HTML acts as weak labels. It is explicitly closed-world: target entities are assumed to appear in the source data, and "open-world" is future work. Inventory→Payment F1 88.15–89.23 vs a supervised upper bound of ~94–95 — [QueryForm](https://arxiv.org/abs/2211.07730)
- **LayoutLLM.** Zero-shot means "not using training sets of downstream tasks". Fields are asked as QA ("What is the address in the document?"). Zero-shot QA-for-VIE ANLS: FUNSD 79.98, CORD 63.10, SROIE 72.12, vs LayoutLMv3 fine-tuned CORD F1 97.46 (different metrics). Generic 7B MLLMs of that era scored very low (LLaVA-1.5-7B: FUNSD 1.93, CORD 18.06) (verified in full text) — [LayoutLLM](https://arxiv.org/abs/2404.05225)
- **VDInstruct (July 2025).** Zero-shot KIE with novel labels; beats DocOwl 1.5 by +5.5 F1 in zero-shot (abstract only; training data not checked) — [VDInstruct](https://arxiv.org/abs/2507.09531)
- **VAREX (IBM, March 2026).** 1,777 government-form documents with 1,771 unique JSON schemas, each in four modalities. Findings — [VAREX](https://arxiv.org/abs/2603.15118):
  - NuExtract 2.0 (2B), an extraction-tuned model, reaches 90.8% exact match (EM) vs 9.7% for its Qwen2-VL base.
  - Below 4B, the bottleneck is "schema echo": the model outputs the schema structure instead of values. It costs 45–65 pp and is triggered by JSON-Schema `$defs`. Inlining `$defs` lifted Qwen3-VL-2B from 27.4% to 91.8% on the affected subset.
  - Gemma 3n E2B 71.0%, Gemma 3 4B 65.3%, Qwen2.5-VL-3B 82.5%.
- Other 2025–2026 benchmarks report that LMMs degrade "under diverse schema definitions and long-tail key fields" (UNIKIE-BENCH) and flag "schema adaptation" as an open challenge (ExStrucTiny, EACL 2026). Both are search snippets only — [UNIKIE-BENCH](https://arxiv.org/abs/2602.07038); [ExStrucTiny](https://arxiv.org/abs/2602.12203)
- **KVP10k (IBM, ICDAR 2024).** Defines key-value extraction "without relying on predefined keys" (10,707 images) as distinct from fixed-schema KIE (abstract only) — [KVP10k](https://arxiv.org/abs/2405.00505)

### Inferences
- **No paper found shows SSL pretraining (MLM, NSP, MIM, infilling or alignment) alone enabling extraction of a field type that was never labeled.** Zero-shot new-field ability shows up only with key/schema-conditioned generation plus supervised extraction training on many diverse schemas (LMDX's Base Entity Extractor, DocLLM's key-value instructions, NuExtract), or with large general instruction-tuned models.
- For the team's setup, a decoder VLM fine-tuned on one fixed 5-field schema with fields baked into the training targets, GenKIE's 3–5 F1 is the most likely outcome when asked for 2 new fields. LMDX's curve suggests ~10 labeled documents recover most of the gap (CORD 66.95 → 90.02). The cost of "new labels" is therefore small, but not zero.
- Encoder-with-head designs are structurally incapable (LMDX shows LayoutLMv3 and Donut at 0.00 with no training docs). The team's generative JSON output is the right architecture for schema change. What matters is whether training taught the model to follow the requested schema, not whether it saw SSL.

### Gaps
- I found no study that fine-tunes an *instruction-tuned* modern VLM (Gemma/Qwen class) on one fixed schema and then measures extraction of new fields added only in the prompt. GenKIE used an OFA backbone, not an instruction-tuned LLM. This is the exact experiment the team should run.
- NuExtract 2.0's training data composition was not verified. VAREX only reports its scores.
- The VDInstruct, UNIKIE-BENCH, ExStrucTiny and CLEX/P2Net claims are from abstracts or search snippets only. CLEX has 1,162 entity categories with 0/1/5/10-shot settings (ACL Findings 2025) — [P2Net/CLEX](https://preview.aclanthology.org/display_plenaries/2025.findings-acl.552.pdf).

---

## Q3. Did NSP survive? Is MLM-style continued pretraining (DAPT/TAPT) useful for generative decoder LLMs/VLMs, and what does it cost or risk?

### Takeaway
NSP did not survive. RoBERTa found removing it "matches or slightly improves" downstream results. ALBERT showed NSP mostly learns topic shift and replaced it with sentence-order prediction. DAPT/TAPT (Gururangan et al. 2020) gives real gains, but always as a step *before supervised fine-tuning* on labeled data, and mainly when the domain is far from the pretraining corpus. For decoder LLMs, continued pretraining on raw domain text has been shown to "drastically" hurt prompting/QA ability unless reformatted into task-like text. Forgetting needs replay and LR re-warming. LoRA limits both learning and forgetting.

### Cited Findings
- **RoBERTa (2019).** With FULL-SENTENCES and DOC-SENTENCES formats (no NSP) vs SEGMENT-PAIR+NSP, "removing the NSP loss matches or slightly improves downstream task performance, in contrast to Devlin et al. (2019)". Example SQuAD 2.0 F1: 78.7 with NSP (segment-pair) vs 79.7 without NSP (doc-sentences); MNLI-m 84.0 vs 84.7 (verified in full text) — [RoBERTa](https://arxiv.org/abs/1907.11692)
- **ALBERT (2019/ICLR 2020).** "NSP conflates topic prediction and coherence prediction … topic prediction is easier to learn". NSP-trained models perform "at random-baseline level on the SOP task", so ALBERT uses sentence-order prediction (verified in full text) — [ALBERT](https://arxiv.org/abs/1909.11942)
- **Don't Stop Pretraining (ACL 2020).** RoBERTa continues MLM pretraining on each domain for 12.5K steps on a v3-8 TPU, and **then** is fine-tuned on labeled task data ("we pass the final layer [CLS] token representation to a task-specific feed-forward layer"). All results below are verified in full text — [Gururangan et al.](https://arxiv.org/abs/2004.10964)
  - Gains are larger for domains far from RoBERTa's corpus: ACL-ARC 63.0 → 75.4, ChemProt 81.9 → 84.2.
  - Gains are near zero for close domains: AGNews 93.9 → 93.9, IMDB 95.0 → 95.4.
  - Adapting to an irrelevant domain (¬DAPT) was "worse than even RoBERTa" in most settings.
  - TAPT (pretraining on the task's own unlabeled text) is "nearly 60 times faster to train than DAPT".
  - On RCT-500: DAPT 82.5 F1 (25M docs, 47GB) vs Curated-TAPT 83.4 (180K docs, 27MB).
  - TAPT "optimizes for single task performance, to the detriment of cross-task transfer".
- **Adapting LLMs via Reading Comprehension (ICLR 2024).** For decoder LLMs, continued pretraining on raw domain corpora gives knowledge but "drastically hurts its prompting ability for question answering". The fix is converting raw text into reading-comprehension tasks (abstract only) — [AdaptLLM](https://arxiv.org/abs/2309.09530)
- **LoRA Learns Less and Forgets Less (TMLR 2024).** In both instruction fine-tuning (~100K pairs) and continued pretraining (~20B tokens), "LoRA substantially underperforms full finetuning" on the target domain but "better maintains the base model's performance on tasks outside the target domain" (verified in full text) — [Biderman et al.](https://arxiv.org/abs/2405.09673)
- **Simple and Scalable Strategies to Continually Pre-train LLMs (TMLR 2024).** Distribution shift from new data "typically results in degraded performance on previous data or poor adaptation". LR re-warming, re-decaying and replay are needed to match full re-training (verified in full text) — [Ibrahim et al.](https://arxiv.org/abs/2403.08763)
- **DocLLM** is the only decoder document model found with a custom SSL objective (block infilling), and it reports no downstream ablation of that objective (see Q1) — [DocLLM](https://arxiv.org/abs/2401.00908)
- **Donut** found pretraining tasks other than reading text (e.g., captioning) "show little gains" (verified in full text) — [Donut](https://arxiv.org/abs/2111.15664)

### Inferences
- NSP should be dropped from the proposal. It is obsolete even for encoders and has no natural form for a decoder VLM reading a single document.
- "MLM on our documents" for Gemma 4 / Qwen3.5 would in practice be next-token or infilling continued pretraining on OCR text, or image→transcription. DAPT evidence says the expected benefit is a modest improvement in fine-tuned accuracy, largest when the domain is far from pretraining data. It is not a replacement for labels: every DAPT result was followed by supervised fine-tuning.
- Because DAPT gains shrink for domains close to pretraining (AGNews, IMDB), the value for the team depends on how unusual their documents are (e.g., Vietnamese administrative forms) relative to the base VLM's OCR/document corpus. Gains are likely small for receipts/invoices.
- Risk for the specific goal: AdaptLLM shows raw-text continued pretraining *erodes* instruction following and prompting. That is exactly the capability needed to respond to a new 7-field schema in the prompt. A team SSL stage could therefore reduce zero-shot new-field ability. If done, it would need replay of instruction data and must be followed by re-quantization to W4A16. LoRA-based continued pretraining would learn little (Biderman et al.).

### Gaps
- No study found applying DAPT/TAPT-style continued pretraining to a modern instruction-tuned VLM (Gemma 3/4, Qwen2.5/3-VL/3.5) for document KIE, with or without measuring new-field extraction.
- No evidence found on how an extra continued-pretraining stage interacts with W4A16 quantization quality (AWQ/GPTQ calibration).
- The full AdaptLLM numbers (magnitude of the prompting drop) were not read; abstract only.

---

## Q4. What document/OCR/layout data do current VLMs already see in pretraining?

### Takeaway
Current base VLMs are already trained at scale on exactly the "read the whole page" and "parse the page" signals that document SSL papers introduced. Qwen2.5-VL and Qwen3-VL train on full-document parsing into HTML with element boxes and reading order, tens of millions of OCR samples (including Vietnamese), and box/point grounding. PaliGemma has an explicit raster-order OCR pretraining task. Gemma 4 lists document/PDF parsing and multilingual OCR among its capabilities but does not disclose its document data. A team-level SSL pass over a few thousand in-house documents is tiny compared with this.

### Cited Findings
- **Qwen2.5-VL (Feb 2025).** All points verified in full text — [Qwen2.5-VL](https://arxiv.org/abs/2502.13923):
  - Pretraining scaled "from 1.2 trillion tokens to 4.1 trillion tokens".
  - "Document Omni-Parsing Data": synthesized documents (tables, charts, equations, music sheets, chemical formulas) "uniformly formatted in HTML, which integrates layout box information", with layouts in "typical reading sequences" and coordinates per module.
  - OCR data is synthetic + open-source + in-house, with a multilingual set listing Vietnamese among its languages.
  - Grounding uses absolute coordinates.
  - The abstract claims "robust structured data extraction from invoices, forms, and tables".
- **Qwen3-VL (Nov 2025)** — [Qwen3-VL](https://arxiv.org/abs/2511.21631):
  - "30 million in-house collected samples" for OCR, pseudo-labeled and refined with Qwen2.5-VL.
  - About 30M synthetic multilingual OCR samples, extending to 29 more languages.
  - Document parsing on "3 million PDFs from Common Crawl … along with 4 million internal documents", in two targets: QwenVL-HTML ("element-level bounding boxes") and QwenVL-Markdown (tables in LaTeX, only images/tables localized).
  - Long-document parsing and VQA synthesized by merging pages, up to 256K tokens.
  - Grounding in coordinates "normalized … to the range [0,1000]".
  - Dense sizes 2B/4B/8B/32B.
- **Qwen3.5 (Feb 2026)** — [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B):
  - "Early fusion training on multimodal tokens", i.e. vision is part of pretraining.
  - Document benchmarks, 9B vs 4B: OmniDocBench1.5 87.7 / 86.2, OCRBench 89.2 / 85.0, CC-OCR 79.3 / 76.7, MMLongBench-Doc 57.7 / 54.2.
  - The card does not describe document training data.
- **Gemma 4 (tech report arXiv 2607.02770, 2026)** — [Gemma 4 Technical Report](https://huggingface.co/papers/2607.02770); [Gemma 4 model card](https://gemma.ai/models/gemma-4/model-card/):
  - Pretraining data is described only as "web documents, code, images, and audio (for E2B, E4B and 12B), with a cutoff date of January 2025". No OCR, document or box data is described.
  - The model card lists "Document/PDF parsing, screen and UI understanding, chart comprehension, OCR (including multilingual)", handwriting and pointing.
  - OmniDocBench 1.5 edit distance (lower is better) at 1120 vision tokens: E2B 0.290, E4B 0.181, 12B 0.164. At 280 tokens: 0.496 / 0.307 / 0.408.
  - InfographicVQA: 63.9 / 70.0 / 88.4 (1120 tokens) vs 44.6 / 54.8 / 58.7 (280 tokens).
  - Per a search snippet of the report (not verified in the fetched text): E2B/E4B use a 150M vision encoder; the 12B is an "encoder-free" unified model.
- **PaliGemma (Jul 2024).** Pretraining tasks include `ocr`: "Concatenation (in raster order) of all text on the image transcribed by a public OCR system", plus detection (`detect`), segmentation, captioning and VQA (verified in full text) — [PaliGemma](https://arxiv.org/abs/2407.07726)
- **Florence-2.** Pretraining includes `<OCR>` and `<OCR_WITH_REGION>` on FLD-5B — [Florence-2 model card](https://huggingface.co/microsoft/Florence-2-large)
- **Kosmos-2.5.** 357.4M document pages for text+box and markdown generation — [Kosmos-2.5](https://arxiv.org/abs/2309.11419)
- **VAREX (2026).** "Layout-preserving text provides the largest accuracy gain": +3 to +8 pp for strong models and up to +18 pp for small ones, with diminishing returns from adding the image to spatial text — [VAREX](https://arxiv.org/abs/2603.15118)

### Inferences
- The Donut-style pseudo-OCR and Pix2Struct/UDOP-style parse/layout objectives are already in Qwen2.5-VL/Qwen3-VL pretraining at a scale of millions of documents and trillions of tokens. Qwen3.5 likely inherits this lineage, though its data is undisclosed. A team-level repeat on a small corpus is unlikely to add general reading or layout skill.
- For Gemma 4, document training data is not disclosed, so the case is less certain. Its reported document scores and listed capabilities indicate substantial document exposure anyway. The sharp drop at 280 vision tokens suggests that image-token budget matters more than additional SSL.
- The team already uses layout-preserving OCR text as one input. VAREX indicates this is the highest-yield input choice, especially for <4B models.

### Gaps
- Gemma 4's and Qwen3.5's exact document/OCR pretraining mixtures are not disclosed in the sources read.
- Whether these base models saw Vietnamese administrative documents specifically is unknown. Qwen2.5-VL lists Vietnamese among its OCR languages but gives no document-type breakdown.

---

## Q5. Which self-supervised or weakly supervised tasks could realistically help a decoder VLM when the field set changes? Does any paper show a "parse everything" intermediate improves extraction of unseen fields?

### Takeaway
The evidence points to schema-conditioned, multi-schema supervised extraction training (real, synthetic or weak labels) as the thing that transfers to new fields, not MLM/NSP. LMDX, NuExtract-via-VAREX and DocLLM's key prompts are examples. "Parse everything" tasks (page→markdown/HTML with boxes, text grounding, key-value discovery) improve general document understanding: Pix2Struct, UDOP, DocOwl 1.5 and LayoutLLM all show it. I found no paper that isolates such a task and shows it improves extraction of *fields unseen in labeled data*. A small number of labeled examples for the new fields closes most of the gap.

### Cited Findings
- **Parsing > reading for downstream understanding.** Pix2Struct: removing screenshot parsing (reading only) drops DocVQA from 67.8 to 12.2 — [Pix2Struct](https://arxiv.org/abs/2210.03347). DocOwl 1.5: adding parsing lifts DocVQA 72.8 → 77.7, and adding multi-grained text localization lifts it to 81.6 — [DocOwl 1.5](https://arxiv.org/abs/2403.12895). UDOP: joint text-layout reconstruction beats MLM (DocVQA 79.7 → 82.8), and layout modeling adds more (84.0) — [UDOP](https://arxiv.org/abs/2212.02623). **All of these are measured on QA, classification or tables, not on new KIE fields.**
- **Layout-aware instruction "pretraining" improves zero-shot KV extraction (QA form).** LayoutLLM: +3.06 FUNSD ANLS from layout-aware pretraining that mixes MVLM, mask-position and geometric instructions with labeled layout/table tasks. The FUNSD evaluation asks "What is the 'key' in the document?" with no FUNSD training — [LayoutLLM](https://arxiv.org/abs/2404.05225). This is the closest evidence that a self-supervised-flavoured stage helps key→value extraction without task labels. The gain is small, and the stage includes supervised data.
- **Multi-schema supervised extraction training is what produced zero-shot on new schemas.** LMDX's Base Entity Extractor (Payment + synthetic-filled public PDF form templates with schema annotations, schema in prompt) reaches 66.95 micro-F1 on CORD with zero CORD docs. Fine-tuned encoders and Donut score 0.00 there — [LMDX](https://arxiv.org/abs/2309.10952). VAREX reports an extraction-tuned 2B model (NuExtract 2.0) at 90.8% EM across 1,771 unseen schemas vs 9.7% for its base — [VAREX](https://arxiv.org/abs/2603.15118).
- **Weak supervision from structure.** QueryForm pretrains on webpages, using HTML structure as weak entity/schema labels, and transfers zero-shot to new document types (closed-world entities) — [QueryForm](https://arxiv.org/abs/2211.07730).
- **Key-value discovery as a task.** KVP10k defines KV extraction "without relying on predefined keys" and adds a combined KIE+KVP benchmark task — [KVP10k](https://arxiv.org/abs/2405.00505).
- **Text+box targets** (OCR-with-region, HTML with element boxes) are standard pretraining targets in Kosmos-2.5, Florence-2 and Qwen-VL — [Kosmos-2.5](https://arxiv.org/abs/2309.11419); [Florence-2](https://huggingface.co/microsoft/Florence-2-large); [Qwen3-VL](https://arxiv.org/abs/2511.21631).
- **Few labels recover most of the gap.** CORD: LMDX 66.95 at 0 docs → 90.02 at 10 docs — [LMDX](https://arxiv.org/abs/2309.10952). SROIE held-out field: GenKIE ~4 F1 at 0-shot → 42–78 F1 at 10-shot — [GenKIE](https://arxiv.org/abs/2310.16131).
- **Small-model failure mode on new schemas is output compliance.** Below 4B, schema echo is caused by JSON-Schema constructs like `$defs`, and is fixable by inlining definitions — [VAREX](https://arxiv.org/abs/2603.15118).

### Inferences
These are recommendations for critiquing the proposal. They are not claims from a source.
- **Reframe the goal.** "New fields without re-fine-tuning" depends on the model's *output interface* following a schema given at inference time. The LMDX/NuExtract/DocLLM pattern does this: schema or key list in the prompt, trained on many varied schemas. SSL objectives do not train this.
- **Cheapest realistic path.** Change the team's SFT data from one fixed 5-field target to schema-conditioned targets:
  - put the field list or JSON schema in the prompt;
  - randomly drop, reorder and rename fields per example;
  - add synthetic or other-doc-type schemas.
  
  Then test the 7-field case zero-shot. Expect a gap like LMDX's 67 vs 90 on CORD. Budget ~10–50 labeled (or teacher-labeled) documents for the two new fields, which on LMDX's curve closes most of it.
- **A "parse everything" auxiliary target is plausible but unproven for new fields.** Examples: full-page → layout-preserving markdown or key-value list, mixed into SFT. It is supported for general document understanding (Pix2Struct, UDOP, DocOwl 1.5, LayoutLLM). If tried, treat it as an experiment with a held-out-field evaluation: train on 5 fields, test on 2 withheld fields, compare with and without the auxiliary task.
- **Before any SSL work, measure the base model zero-shot on the 7-field schema.** Use both Gemma 4 and Qwen3.5 at the deployed W4A16 precision. Qwen3.5-4B/9B and Gemma 4 E4B/12B already report strong OmniDocBench/OCRBench scores, and VAREX shows ≥2B extraction-tuned models handling unseen schemas well. Also measure whether the current fixed-schema LoRA *reduces* that ability. GenKIE and AdaptLLM suggest it might.
- **Arguments against MLM/NSP specifically.** NSP is obsolete (RoBERTa, ALBERT). MLM is not native to decoders. DAPT always precedes labeled fine-tuning. Raw-text continued pretraining hurts prompting in decoders (AdaptLLM). The base VLMs already pretrain on millions of OCR/parsing documents. No paper shows SSL yielding never-labeled-field extraction.

### Gaps
- **The most important gap.** No paper found shows that an intermediate "parse everything" or key-value-discovery task, as opposed to multi-schema supervised extraction, improves extraction of fields unseen in labeled training. The team would have to run this ablation themselves.
- No public result found for Gemma 4 or Qwen3.5 on schema-variable KIE benchmarks (VAREX evaluated Gemma 3/3n and Qwen2.5/3-VL, not Gemma 4 or Qwen3.5).
- VAREX's NuExtract number compares a published model with its base. VAREX did not run a controlled fine-tuning experiment on diverse schemas.
