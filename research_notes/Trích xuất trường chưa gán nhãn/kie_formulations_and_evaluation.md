# KIE formulations, ground-truth design and evaluation for adding fields without retraining

Scope: alternative task formulations (QA / schema-conditioned / open key-value), ground-truth design, and evaluation protocols for fields unseen in training. Context: LoRA-fine-tuned small VLMs (Gemma 4, Qwen3.5), ~2,000 labeled samples per doc type, 95% exact-match / 95% field-accuracy target, CORD v2 as public dev set. Research date 2026-11-26.

Source-verification legend: papers marked **[read]** had full text or the official page read in this session (VRDU, LMDX, DocILE: full PDF text; KVP10k: arXiv HTML; LayIE-LLM: arXiv HTML; Google / Azure docs: official pages). **[abstract/search]** = only abstract or search-result summary seen. **[not fetched]** = cited for identity/context from background knowledge, specifics not re-verified this session.

Local fact (checked in this repo, not a web source): `data/public/cord/all.jsonl` is a **12-field projection** of CORD v2 with renamed keys and explicit nulls (`menu.{name,count,unit_price,price}`, `subtotal.{subtotal_price,discount_price,service_price,tax_price}`, `total.{total_price,cash_price,change_price,credit_card_price}`), 1,000 docs × 2 input types (image, spatial_text). Non-null document frequency over the 1,000 docs: menu.price 997, menu.name 995, total.total_price 972, menu.count 910, subtotal.subtotal_price 662, total.cash_price 648, total.change_price 620, subtotal.tax_price 445, menu.unit_price 356 (737 instances), total.credit_card_price 151, subtotal.service_price 123, subtotal.discount_price 74. The remaining ~18 original CORD v2 categories (e.g., menu.sub_nm, menu.discountprice, menu.num, menu.etc, sub_total.othersvc_price, total.menuqty_cnt, total.menutype_cnt, total.emoneyprice, void_menu.*) are **not** in the team's targets, so they are natural candidates for "new fields" (their frequencies were not computed here; see Gaps in section 3).

---

## 1. Question-answering / query-conditioned formulations, and evidence of accuracy on fields not seen in training

### Takeaway
The working "add fields without retraining" formulation in the literature and in production is **schema- or query-conditioned extraction**: the requested field names (ideally with descriptions) go in the prompt and the model returns only those fields, with explicit nulls. The best published evidence (LMDX, Google, 2023–24) shows that a model specially trained for this reaches only about **66–67% micro-F1 zero-shot on CORD**, compared with about **90% after 10 labeled CORD docs** and 94.5–95.6% after 800. So zero-shot on a new field should not be expected to reach a 95% target. It should be treated as a cold start before a short few-shot top-up.

### Cited Findings
- **DUE benchmark (NeurIPS 2021 Datasets & Benchmarks)** unifies KIE and QA: KIE ("property extraction") is defined as producing values for (property, document) tuples, and document QA is framed as the same task with a natural-language question in place of the property name. This is the conceptual basis for "KIE as QA". — [DUE, NeurIPS 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/069059b7ef840f0c74a814ec9237b6ec-Abstract-round2.html) [abstract/search]
- **LMDX (Google, arXiv Sep 2023, rev. with Gemini Pro results) [read]**: the prompt contains the OCR text with coordinate tokens plus a **JSON schema of entity types to extract**. Example: `{"foo": "", "bar": [{"baz": []}]}` means one `foo` and repeated hierarchical `bar` entities, each holding repeated `baz`. The completion uses the same keys in the same order. Missing entities are emitted explicitly as `null` (singular) or `[]` (repeated). Predicted types not in the requested schema are discarded. — [LMDX, arXiv 2309.10952](https://arxiv.org/abs/2309.10952)
- LMDX makes the model schema-following by first fine-tuning on a **"Base Entity Extractor" mixture of varied (document, schema, extraction) tuples**: the Payment dataset plus government PDF form templates filled with synthetic data. "No document or schema contained in the base extraction training phase overlap with the documents and schemas used in our target benchmarks", so the target-benchmark evaluation counts as zero-shot. — [LMDX](https://arxiv.org/abs/2309.10952) [read]
- **Zero-shot results on unseen schemas, LMDX Table 3 (CORD micro-F1, |D|=0)**: LMDX-PaLM 2-S 66.95, LMDX-Gemini Pro 66.03. Baselines: GPT-4V+image 64.05, Gemini Pro+OCR 59.57, PaLM 2-S+OCR 55.85, GPT-3.5+OCR 48.92, Gemini Pro+image 47.12, LLaVA-v1.5-13B 5.97. — [LMDX](https://arxiv.org/abs/2309.10952) [read; numbers read from a column-garbled PDF table extraction, ordering matches the paper's model list]
- **Zero-shot on VRDU, LMDX Table 2 (micro-F1, |D|=0)**: Registration Form 73.81 / 76.78 (PaLM 2-S / Gemini Pro) on Single Template, 74.94 / 77.18 Mixed, 71.65 / 75.15 Unseen; Ad-buy Form only 39.33 / 37.94 Mixed and 39.74 / 38.02 Unseen; Ad-buy **line_item F1 21.21 / 23.29**. The paper notes that zero-shot line-item F1 (21.21) is close to the best fine-tuned baseline at 200 docs (25.46). — [LMDX](https://arxiv.org/abs/2309.10952) [read]
- In LMDX the **hierarchical/repeated groups** (line items) are the weakest part of zero-shot extraction (see the line-item numbers above). VRDU likewise concludes that "extracting hierarchical or repeated entities is really challenging, and all models perform worse on this compared to simple fields." — [VRDU, KDD 2023](https://arxiv.org/abs/2211.15421) [read]
- **LayIE-LLM (Colakoglu, Solmaz, Fürst; arXiv Feb 2025, rev. Sep 2025) [read]**: on VRDU with only a schema (keys + regex formats) in the prompt and no task fine-tuning, GPT-4o-vision reaches 0.902 / 0.897 F1 (Single / Unseen Template) and Qwen2.5-vision 0.890 / 0.891, against 0.603 / 0.194 for LayoutLMv3 and 0.663 / 0.397 for ERNIE-Layout fine-tuned in the authors' setup. Tuning the pipeline (chunking, post-processing, evaluation leniency) raised F1 from 0.650 to 0.783. "Cleaned" post-processing (0.734 vs 0.650 initial) and fuzzy rather than exact matching (0.733 vs 0.650) both move scores a lot. — [LayIE-LLM, arXiv 2502.18179](https://arxiv.org/abs/2502.18179)
- **Production equivalents of query-conditioned extraction**:
  - Azure Document Intelligence **query fields**: you pass a list of field labels (e.g., `queryFields=OurReference,BookingDate`) with prebuilt-layout or a prebuilt model "without having to train a custom model". Azure recommends query fields "when the value you require can't be described as a key value pair in the document. For example, the agreement date of a contract." — [Azure query fields docs, updated 2026](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/query-fields) [read]
  - Google Document AI Custom Extractor (generative): "For zero-shot, only a schema is required." The docs say field names "can greatly affect model accuracy" and recommend adding a description with "location information and text patterns of the property values". Each field is typed as **Extract** or **Derive** (inferred). — [Google Document AI ce-with-genai](https://docs.cloud.google.com/document-ai/docs/ce-with-genai) [read]
  - AWS Textract **Queries / Custom Queries**: natural-language questions per field, with optional "adapters" trained on a small labeled set (see section 5). — [Textract adapters best practices](https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.html) [search]
- **Label-semantics approaches (pre-LLM)**: LASER (Wang & Shang, ACL Findings 2022) generates the **label surface name** right after each entity in a layout-aware seq2seq model (built on LayoutReader), so label semantics carry over in few-shot settings; tested on FUNSD and CORD. — [LASER, ACL Findings 2022](https://aclanthology.org/2022.findings-acl.329) [abstract/search]. GMN (Tencent, NAACL 2022) is a generative multimodal model "without predefined label categories". — [GMN, arXiv 2207.04713](https://arxiv.org/abs/2207.04713) [abstract]. GenKIE (EMNLP Findings 2023) uses prompts to a seq2seq multimodal model to generate key-value outputs. — [GenKIE, arXiv 2310.16131](https://arxiv.org/abs/2310.16131) [abstract/search]
- **Distilling open-world ability**: DocKD (Amazon, EMNLP 2024) has an LLM generate training data for small document models and injects document knowledge (key-value pairs, layout, descriptions) into the prompt. It reports better generalization "to unseen documents, questions, entities, or categories" on VQA, entity extraction and classification. A reported 81.0 vs 76.9 ANLS on DocVQA (DocKD vs plain KD) came from a search summary and is not verified. — [DocKD, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.185) [abstract/search]
- Classic document QA datasets that a model can be co-trained on: DocVQA (extractive QA on industry documents) — [arXiv 2007.00398](https://arxiv.org/abs/2007.00398) [not fetched]; InfographicVQA — [arXiv 2104.12756](https://arxiv.org/abs/2104.12756) [not fetched]. OCR-free models benchmarked on DocVQA include Donut — [arXiv 2111.15664](https://arxiv.org/abs/2111.15664), Pix2Struct — [arXiv 2210.03347](https://arxiv.org/abs/2210.03347), and LayoutLMv3 — [arXiv 2204.08387](https://arxiv.org/abs/2204.08387) [not fetched; specific ANLS scores not verified this session].

### Inferences
- For receipts, **per-field QA** ("What is the tax price?", one question per call) breaks for repeated groups. Every `menu[i].{name,count,price}` has to stay aligned within its item. **Schema-conditioned JSON** with the requested subset in the prompt, LMDX-style, keeps the group structure and is one call per document. It is the closer fit to the team's nested-JSON target. Per-field QA is still useful for scalar header fields and as a fallback.
- What carries over from LMDX to a LoRA'd Gemma 4 / Qwen3.5: train with **randomized schemas**. For each training example, sample a subset of the densely-labeled fields, sometimes paraphrase or rename the keys, shuffle their order, optionally include field descriptions, and target only the requested keys with explicit `null`/`[]`. Add other doc types or synthetic schemas, as LMDX's base mixture does, so the model learns "extract what the schema asks" rather than memorizing one fixed 12-key JSON. A fixed prompt + fixed schema (today's setup) gives the model no reason to read the schema at all.
- Expect a **gap between "renamed seen field" and "truly new concept"**. LMDX zero-shot on CORD (~66 F1) mixes both and was measured on 2023-era models (PaLM 2-S, Gemini Pro). Current small VLMs (2025–26) may do better, but no source in this session measures Gemma 4 / Qwen3.5 on unseen-field KIE.
- LayIE-LLM's large swings from post-processing and matching leniency mean that "zero-shot F1" from different papers is **not comparable** unless normalization and matching are fixed. The team should evaluate new-field generalization with its own scorer (exact match on normalized values), not import numbers.

### Gaps
- I found no paper that reports accuracy specifically on **individual fields held out from fine-tuning on the same document type** (as opposed to a whole new schema/benchmark, as in LMDX). See section 3.
- No 2025–26 primary-source results were found for Qwen2.5/3-VL-class or Gemma-class small models on zero-shot schema-conditioned KIE on CORD. LayIE-LLM's Qwen2.5-vision numbers are on VRDU only, and the model size was not captured.
- I could not verify specific DocVQA ANLS numbers for Donut, Pix2Struct or LayoutLMv3 this session.
- I did not find the paper titled "KIE as QA" or "query-driven KIE" the brief mentions, beyond DUE's framing, GMN and GenKIE. "QGN (query-driven generative network)" was not located by search.

---

## 2. Open key-value extraction and "parse everything, then select" two-stage designs

### Takeaway
Open KV extraction (FUNSD/XFUND/KVP10k style) detects printed key→value links without a schema. However, the 2024 SOTA-style LLM baseline on KVP10k reaches only **~0.61 F1**, and receipts like CORD are dominated by values with **no printed key** (item names, counts, prices in columns). A pure "KV-pairs, then map key names to the schema" pipeline therefore caps accuracy well below 95% for this use case. The more viable two-stage variant is "**dense structured transcription into a superset schema, then select/rename**". It only generalizes to fields that exist in the superset.

### Cited Findings
- **FUNSD (ICDAR-OST 2019)**: 199 scanned forms annotated with entities labeled question / answer / header / other, plus links between them, i.e. schema-free form understanding. — [FUNSD, arXiv 1905.13538](https://arxiv.org/abs/1905.13538) [not fetched]. **XFUND** extends the FUNSD format to 7 languages, released with LayoutXLM. — [LayoutXLM / XFUND, arXiv 2104.08836](https://arxiv.org/abs/2104.08836) [not fetched]. These define the standard **SER (semantic entity recognition) + RE (relation extraction / linking)** decomposition.
- **KVP10k (IBM Research; arXiv 1 May 2024) [read]**: 10,707 page images (3,524 from publicfiles.fcc.gov plus 7,183 from Common Crawl PDFs across "over 40,000 targeted domains"). The task is key-value discovery "without relying on predefined keys".
  - Annotation: normalized rectangles per entity, with a link drawn from each value to its key.
  - "unvalued-key" marks keys with no value (empty form fields).
  - Values with no printed key get a "Floating" label from 11 classes (Document Type, Document Title, Year, Date, Name, Address, Phone, Email, Website, Amount, Text). The scheme has 17 classes in total.
  - Metrics: entity match requires normalized edit distance < 0.2 and IoU > 0.3; precision and recall are computed per image and averaged.
  - Baseline: Mistral-7B fine-tuned LMDX-style on OCR+boxes, about 24 h on one A100. F1 by subset (text+location): regular 0.611, unkeyed 0.584, unvalued 0.588, all 0.612.
  - The authors chose a generative LLM over LayoutLMv3 because encoder taggers "cannot generate the links needed for pairing".
  - The paper does not discuss mapping extracted keys to a downstream schema.
  — [KVP10k, arXiv 2405.00505](https://arxiv.org/abs/2405.00505). The venue was not stated on the arXiv page; ICDAR 2024 publication is unverified.
- **Azure's explicit trade-off between open KV pairs and query fields** [read]:
  - Use key-value pairs if "You don't know the specific fields to be extracted, or the number of fields is large (greater than 20)".
  - Key-value pairs "extract the keys and values as they exist in the form or document, you need to plan for any key variations. For example, keys `First Name` or `Given Name`."
  - "With query fields, you define the key and the model only extracts the corresponding value."
  — [Azure query fields docs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/query-fields)
- Microsoft Learn's training module describes a **staged pipeline**: find label-value candidates (e.g., "Invoice Number:", "Total:"), then map the candidate values to schema fields; prompt-based LLM extraction is listed as another option. — [Microsoft Learn: field extraction and mapping](https://learn.microsoft.com/en-us/training/modules/introduction-information-extraction/4-form-extraction) [search]
- **ChatSchema (arXiv Jul 2024)**: an LMM processes the implicit key-value pairs in OCR text and performs field-name standardization, unit conversion, value mapping and type conversion against a schema in one step, not as a separate mapping stage. — [ChatSchema, arXiv 2407.18716](https://arxiv.org/abs/2407.18716) [search]
- In LayIE-LLM, an explicit **"schema-mapped" post-processing** stage (mapping LLM output keys onto the schema) did not change the average F1 (0.650 initial vs 0.650 mapped). "Cleaned" value normalization did help (0.734). — [LayIE-LLM](https://arxiv.org/abs/2502.18179) [read]
- I found no peer-reviewed paper that directly compares "extract all KV pairs, then map by LLM or embedding" against single-pass schema-conditioned extraction on the same benchmark. The search summary confirmed this.

### Inferences
- **Pros of parse-then-select**:
  - Stage 1 is schema-independent and can be cached or audited.
  - Adding a field changes only stage 2, i.e. a name→key mapping by LLM or embedding similarity.
  - The full parse provides provenance and a natural hallucination check.
- **Cons, especially on receipts**:
  1. **Implicit/keyless fields.** menu.name, menu.count and menu.price are table columns with no per-value key. KVP10k needs a whole "Floating" category for these, and its unkeyed subset scores lowest on text (0.601).
  2. **Derived/computed fields** (e.g., total item count, menuqty_cnt-like aggregates, "tax = total − subtotal" when not printed) are absent from any parse. Google formalizes these as "Derive" fields.
  3. **Key-name variation and language** ("TUNAI"/"CASH", "PPN"/"Tax"/"PB1" on Indonesian receipts) make stage 2 nontrivial. This is Azure's "First Name vs Given Name" point.
  4. **Error compounding.** Stage-1 recall caps end-to-end accuracy, and the published open-KV baseline is ~0.61 F1.
  5. **Spread-out values** (multi-line names, values split across columns or pages) are hard to represent as one key→value link.
- A **"dense superset transcription"** variant fits CORD better: stage 1 emits the full CORD-like hierarchical parse with all ~30 categories plus a generic `other: [{label_text, value}]` bucket for any printed key-value pair outside the taxonomy, and stage 2 selects or renames. Fields inside the superset are then "new" only at the output layer: no retraining, near-trained-field accuracy. Truly new concepts must come either from the `other` bucket (printed-key fields) or from a query-conditioned second pass (keyless or derived fields).
- **Hybrid recommendation to test.** Schema-conditioned single pass for requested fields, plus an optional open-KV "catch-all" list in the same JSON. When a new field arrives, (a) try the query-conditioned prompt with a description, (b) fall back to mapping from the catch-all list, (c) measure both on a held-out-field split (section 3).

### Gaps
- No production write-up with numbers comparing parse-then-select to direct extraction was found. Commercial IDP blogs (Reducto, LlamaParse, Unstructured, etc.) were not searched due to the call budget and would be marketing-grade evidence anyway.
- No study was found on embedding-based key-name matching accuracy for KIE schema mapping.

---

## 3. Zero-shot / unseen-field benchmarks and how to build a field-holdout split on CORD v2

### Takeaway
Public "zero/few-shot" document benchmarks almost all hold out **templates/layouts** (VRDU Unseen Template, DocILE zero-shot layout clusters) or **whole schemas/datasets** (LMDX's base-extractor protocol, SVRD Baidu-FEST), not **individual fields within one document type**. No standard field-holdout split for CORD was found, so the team needs to define one. The building blocks are the VRDU-style disjoint split plus training-size ladder, DocILE-style shot buckets, and LMDX-style disjoint schemas.

### Cited Findings
- **VRDU (Google; arXiv Nov 2022, KDD 2023) [read]**: two datasets, FCC political ad-buy forms and FARA registration forms. Three tasks:
  - Single Template Learning: train and test from one template.
  - Mixed Template Learning: the same set of templates in train and test.
  - Unseen Template Learning: "train and test sets are drawn from disjoint sets of templates". For Registration Forms the model trains on 2 of the 3 templates and tests on the third.
  
  Each task has 300 test docs and training sets of **10, 50, 100, 200** docs, each built 3 times with different seeds and averaged. Results:
  - At 200 docs, UTL micro-F1 is **13–17 points below MTL** on Registration Forms, and about 3 points below on Ad-buy.
  - There is a "15+ point gap" between |D|=10 and |D|=50 across tasks and models.
  - The best micro-F1 at |D|=200 is ~90% (Registration) and ~45% (Ad-buy). By contrast, LayoutLMv2 / FormNet reach 96.01 / 97.21 on CORD. VRDU argues CORD/SROIE overstate how solved the problem is.
  — [VRDU, arXiv 2211.15421](https://arxiv.org/abs/2211.15421)
- VRDU lists desiderata for a practical benchmark: "rich schema, layout-rich documents, diverse template, high-quality OCR results, and token-level annotation". Its evaluation toolkit uses **type-aware matching** (e.g., "4" ≡ "4.0" for numbers, but "4, Main St." ≠ "40 Main St." for addresses). — [VRDU](https://arxiv.org/abs/2211.15421) [read]
- **DocILE (Rossum / CTU, ICDAR 2023) [read]**: 6,680 annotated real business documents, ~100k synthetic, and a large unlabeled set; 55 field classes. Validation and test sets contain **25% zero-shot samples** (layout clusters unseen in training), **25% few-shot** (≤3 training docs from the cluster) and **50% many-shot**. At most 20 test docs come from any single zero/few-shot cluster, for diversity. The RoBERTa-base baseline, broken down by shots-of-layout (0 / 1–3 / 4+), shows a clear rise from the zero-shot to the many-shot buckets on both KILE and LIR (Table 3; exact cell alignment was garbled in extraction, so the per-bucket values are not quoted). — [DocILE, arXiv 2302.05658](https://arxiv.org/abs/2302.05658)
- **ICDAR 2023 SVRD competition**: Track 2 "Baidu-FEST" evaluates "zero-shot and few-shot structured text extraction" end to end over 50+ enterprise document types. There were 26 valid submissions from 15 teams. The organizers conclude "there is still a large gap on the expected information extraction performance for complex and zero-shot scenarios." The abstract does not say whether zero-shot means unseen doc types or unseen keys. — [SVRD, arXiv 2306.03287](https://arxiv.org/abs/2306.03287) [abstract]. StrucTexTv3 later reports on SVRD's zero-shot task and removed SVRD training examples to avoid leakage. — [StrucTexTv3, arXiv 2405.21013](https://arxiv.org/abs/2405.21013) [search]
- **LMDX's protocol** is the closest published analogue to an unseen-field test: the base extractor is trained on schemas disjoint from the target benchmarks, then evaluated zero-shot and at |D| = 10/50/100/200/800 (CORD) and 10/50/100/200 (VRDU). — [LMDX](https://arxiv.org/abs/2309.10952) [read]
- **Kleister NDA / Charity**: long-document, document-level KIE where entity values are given without positions. — [Kleister, arXiv 2105.05796](https://arxiv.org/abs/2105.05796) [not fetched]. **CORD**: Indonesian receipts with a hierarchical menu / sub_total / total schema; 800/100/100 train/dev/test publicly. — [CORD, OpenReview](https://openreview.net/forum?id=SJl3z659UH) [not fetched]. DocILE's related-work section notes a discrepancy between the class count stated in the CORD paper and the categories actually released in the CORD repository. — [DocILE](https://arxiv.org/abs/2302.05658) [read; exact wording not captured]
- **Evaluation metrics for nested output**: KIEval (Upstage, arXiv Mar 2025) adds **group-level F1**, which scores whole groups such as one menu item as a unit, after Hungarian matching of predicted and ground-truth groups. It frames errors as correction costs (substitution / addition / deletion) and was evaluated on SROIE, CORD and FUNSD. — [KIEval, arXiv 2503.05488](https://arxiv.org/abs/2503.05488) [search]. ANLS* (2024) is a metric for generative LLM outputs over dicts and lists. — [ANLS*, arXiv 2402.03848](https://arxiv.org/abs/2402.03848) [not fetched]

### Inferences — proposed field-holdout protocol for CORD v2 (team design, not from a paper)
1. **Re-derive targets from the original CORD v2 `gt_parse`**, which the local manifest's `meta.source` (`naver-clova-ix/cord-v2:train:0`) points back to, so that all ~30 categories are available. Keep the existing doc-level split (seed 42; train 1600 / val 100 / test 200 / calibration 100 rows), per the team rule "never re-split".
2. **Split by field, not by document.** Define F_seen (e.g., the current 12) and F_new (held out). Choose F_new to cover three difficulty types:
   - (a) **near-neighbour scalar header fields** with a printed key, similar to a seen field (e.g., hold out `total.credit_card_price` or `subtotal.service_price` while `total.cash_price` / `subtotal.tax_price` stay seen). Also consider `total.emoneyprice` and `sub_total.othersvc_price` from the unused CORD categories.
   - (b) **new item-level sub-fields** inside a repeated group (e.g., `menu.unit_price`, `menu.discountprice`, `menu.sub_nm` / sub-item fields).
   - (c) **aggregate/derived-looking fields** (`total.menuqty_cnt`, `total.menutype_cnt`).
   
   Run 2–3 rotations (different F_new) so conclusions are not hostage to one field's frequency. Ensure each held-out field has enough non-null test support: in the 12-field local set, `subtotal.discount_price` appears in only 74/1000 docs, so the ~100-doc test split would have only ~7 positives.
3. **No leakage.** Held-out field values must never appear as *targets* in training, including inside a catch-all `other` list or a dense transcription. They may appear in the input (that is unavoidable and realistic). Also hold out their **names and descriptions** from training prompts.
4. **Test conditions** (same test docs, different requested schemas):
   - T1: seen fields only, which is the regression check.
   - T2: new fields only.
   - T3: **mixed schema**, e.g., 5 seen + 2 new, mirroring the "5 → 7 required fields" change.
   - T4: **renamed / paraphrased seen fields**, e.g., `cash_price` → `amount_paid_in_cash`, plus descriptions. This separates name sensitivity from concept novelty.
   - T5: **absent-field requests**, i.e. fields requested where the ground truth is null, to measure false-positive / hallucination rate.
5. **Metrics**:
   - per-field F1 and accuracy with type-aware normalization (VRDU-style);
   - **null accuracy**, split into precision on "present" and FP rate on "absent";
   - group-level F1 for menu items (KIEval-style);
   - document-level exact match on the requested subset;
   - training-size ladder for new fields: 0 / 5 / 10 / 25 / 50 / 100 docs with ≥3 seeds (VRDU/LMDX-style);
   - report seen-field degradation after adding new-field data (catastrophic-forgetting check).
6. **Template generalization is a separate axis.** CORD has no template/layout IDs. If template novelty matters, cluster the receipts by merchant/layout (DocILE-style) and report zero/few/many-shot layout buckets separately from field novelty.

### Gaps
- No published field-holdout split on CORD/SROIE was found. The protocol above is an inference and needs piloting.
- Original CORD v2 per-category frequencies for the unused ~18 categories were not computed this session. They must be checked before choosing F_new, since some (e.g., void_menu, emoney) may be very rare.
- Not verified: how SVRD Baidu-FEST defines "zero-shot" (unseen keys vs. unseen document types), and its metric.

---

## 4. Ground-truth design for a generalizable setup

### Takeaway
To be able to add fields later without re-labelling, the strongest design is to **annotate densely once**: every field in a broad superset schema, with explicit nulls, per-value provenance (box or OCR line/word ids), and a versioned **field registry** (name, description, type, cardinality, extract vs. derive, normalization). Training prompts then request random subsets of these fields. Benchmarks that care about generalization (VRDU, DocILE, KVP10k) all annotate locations and multiple instances. LMDX shows that explicit nulls measurably help.

### Cited Findings
- **VRDU**: argues for **token-level annotation** (the exact tokens of each entity), because some datasets annotate only the value string. Example: a receipt "total" annotated as "10" when "10" also appears elsewhere, e.g., as the number of items, so the correct occurrence cannot be recovered and token-level supervision is impossible. VRDU had annotators relabel the documents with precise token-level annotation. It annotates **hierarchical and repeated entities** (line items, componentized addresses) and data types to drive type-aware matching. — [VRDU](https://arxiv.org/abs/2211.15421) [read]
- **DocILE**:
  - Each field is a **triplet (location: bbox + page, field type, text)**. Line-item fields also carry a line-item ID.
  - Fields may overlap, span multiple lines or parts of words, and the same field type may occur multiple times with the same or different values.
  - The full list of field types **with descriptions** is published in the supplementary material.
  - Extra annotations: table structure (column headers mapped to field types, row/column grid) and metadata including **layout cluster ID**.
  - Annotating 6,680 docs took about **2,500 annotator-hours including verification**, roughly 22 min/doc for 55 classes; the per-doc figure is derived here.
  - Documents with handwritten or redacted key info were skipped.
  — [DocILE](https://arxiv.org/abs/2302.05658) [read]
- **KVP10k**: normalized rectangles; value→key links; explicit "unvalued-key"; "Floating" class labels for keyless values. That makes it a **schema-free but typed** annotation that can later be mapped to any schema. — [KVP10k](https://arxiv.org/abs/2405.00505) [read]
- **Explicit null vs. omission**: in LMDX, having completions **skip** missing entity types instead of emitting `"type": null` cost **−6.77 micro-F1** (VRDU Ad-buy Mixed, |D|=10). The authors hypothesize that skipping forces the model to decide, in a single token, both which type comes next and whether the previous ones are absent. — [LMDX](https://arxiv.org/abs/2309.10952) [read]
- **Provenance/grounding**: LMDX emits a segment identifier (coordinate tokens) per extracted value. This allows rejecting values not present in the document (hallucinations) and gives line-level localization accuracy of **88–94% zero-shot and 98–99% fine-tuned**. Replacing coordinate tokens with plain line indices cost **−14.98 micro-F1**. — [LMDX](https://arxiv.org/abs/2309.10952) [read]
- **Field descriptions as data**: Google recommends descriptive, distinct field names and descriptions that include "location information and text patterns of the property values". It also distinguishes **Extract** from **Derive** fields. — [Google ce-with-genai](https://docs.cloud.google.com/document-ai/docs/ce-with-genai) [read]
- Azure warns that open KV extraction requires planning for **key-name variants**, which supports storing aliases or synonyms per field in the registry. — [Azure query fields](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/query-fields) [read]
- LMDX's base-extraction mixture used government form templates **filled with synthetic data** and "annotated for schema and entities to extract". This is a cheap way to obtain many (document, schema) pairs. DocILE likewise ships about 100k synthetic documents with full labels, generated from 100 annotated layouts. — [LMDX](https://arxiv.org/abs/2309.10952); [DocILE](https://arxiv.org/abs/2302.05658) [read]

### Inferences
- **Dense vs. required-only annotation**:
  - Required-only labeling (5 fields) is cheaper now. But adding 2 fields later means re-opening every document, and the training data can never teach "schema following", because the requested set is always identical.
  - Dense labeling has a higher one-off cost, and DocILE's ~22 min/doc for 55 classes is an upper reference. In exchange it enables: (i) randomized-subset training, (ii) field-holdout evaluation, (iii) adding a field from the superset with zero new labels.
  - A middle path: dense-label a **subset** (e.g., 300–500 docs per doc type) for schema-generalization training and evaluation, and keep required-only labels on the rest.
- **The three-state null semantics are critical for partial labels.** Distinguish `null` = "verified absent in the document" from "not annotated / unknown". When training on partially-labeled docs, never request a field that was not annotated, otherwise the model learns false nulls. Store per-document `annotated_fields` alongside `target`. The current local CORD projection uses explicit nulls for all 12 fields, which is fine because all 12 were annotated.
- **Field registry entry, proposed schema**:
  - `name`, `aliases`, `description` (1–2 sentences, including location cue and text pattern);
  - `type` (money / int / date / string);
  - `cardinality` (scalar or repeated; parent group);
  - `mode` (extract / derive);
  - `normalization` (e.g., strip thousands separators, decimal handling);
  - `example_values`; `version`.
  
  Feed name and description into prompts. Use type and normalization in the scorer.
- **Provenance**: for OCR-text input, store OCR line/word ids per value. That is cheaper than drawing boxes and enables LMDX-style grounding checks. CORD's original annotations already contain word-level quads per category, so provenance can be recovered for CORD (not verified in the local files).
- **Derived fields** (sums, counts, tax computed from totals) should be marked `mode=derive` with the formula in the description, and evaluated separately: they test reasoning, not reading.

### Gaps
- No published per-field annotation-time numbers (dense vs. sparse) were found. The DocILE figure is per document and includes table and metadata annotation.
- No study was found that quantifies the gain from adding natural-language field descriptions in fine-tuned small-model KIE. Google asserts it helps, and LayIE-LLM did not ablate it.

---

## 5. How much labeled data is needed to add a new field if zero-shot is not enough

### Takeaway
When the base model has already been trained to follow schemas, **5–10 labeled documents** recover most of the gap and **~50–200 documents** approach the full-data ceiling. LMDX on CORD: ~66 F1 at 0 docs, ~90 at 10, ~93.4–93.8 at 200, ~94.5–95.6 at 800. Vendors converge on the same numbers: Google recommends 5–10 docs for few-shot and 10–50+ for fine-tuning; AWS requires at least 5 train + 5 test. These are **whole-schema** curves. For a single rare field, what matters is the number of *positive instances*, which on CORD-like data can be 10× fewer than the document count.

### Cited Findings
- **LMDX CORD data-efficiency curve (micro-F1; PaLM 2-S / Gemini Pro)**: |D|=0 66.95 / 66.03; |D|=10 **90.02 / 89.45**; |D|=50 91.40 / 91.66; |D|=100 91.48 / 91.16; |D|=200 93.40 / 93.76; |D|=800 94.51 / 95.57. Baselines: LayoutLMv3 74.04 / 85.78 / 90.39 / 93.59 / 95.66 and Donut 26.15 / 65.68 / 71.81 / 75.85 / 81.55 at 10 / 50 / 100 / 200 / 800. — [LMDX Table 3](https://arxiv.org/abs/2309.10952) [read; values reconstructed from PDF text, column order verified against the model list]
- LMDX on VRDU Registration Mixed reaches 87.72 micro-F1 at 10 docs, within 5.06 points of its peak (92.78). The strongest fine-tuned baseline, LayoutLMv2, sits 19.75 points below its peak at 10 docs (69.44 vs 89.19). **Skipping the base schema-following training costs −11.44 micro-F1 at |D|=10.** — [LMDX](https://arxiv.org/abs/2309.10952) [read]
- VRDU with encoder baselines: a "15+ point gap" in micro-F1 between |D|=10 and |D|=50 across all tasks. Even Single Template at |D|=10 gets "just 74.22%" (FormNet). — [VRDU](https://arxiv.org/abs/2211.15421) [read]
- **Google Document AI Custom Extractor**: zero-shot = 0 docs, "Medium" accuracy; few-shot = **5 to 10** docs, "Medium-high"; fine-tuning = **10 to 50+** docs, "High". Model versions through v3.1-lite (Oct 2026) and v3.5 RC (May 2026) are listed, and the earlier v1.5.1/v1.6 versions are described as Gemini 2.5 Flash / Gemini 3 Flash based (search snippet). — [Google ce-with-genai](https://docs.cloud.google.com/document-ai/docs/ce-with-genai) [read]
- **AWS Textract Custom Queries adapters**: at least 5 training and 5 test documents. With fewer than 20 docs, split them equally between train and test; with more than 20, use 70/30. — [Textract adapters best practices](https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.html); [Textract preparing datasets](https://docs.aws.amazon.com/textract/latest/dg/textract-preparing-training-testing.html) [search]. A community answer claims a 30-query limit per adapter; this is unofficial and unverified. — [AWS re:Post](https://repost.aws/questions/QUnv59_d9rRl6DRRZ93dUeXg/textract-document-analysis-api-train-custom-adapter-or-user-baseline-model-for-30-custom-queries)
- **In-context examples are not reliably helpful**: in LayIE-LLM's VRDU grid, the average F1 was highest with 0 examples (0.650) and *lower* with 1 / 3 / 5 examples (0.589 / 0.595 / 0.580). However, the brute-force best configurations for GPT-4o and LLaMA3 used 5 examples, and the paper's own text is inconsistent on this point. — [LayIE-LLM](https://arxiv.org/abs/2502.18179) [read]
- DocILE's shot buckets (0 / 1–3 / 4+ docs per layout cluster) show the many-shot bucket clearly above the zero-shot bucket for a RoBERTa baseline, i.e. layout familiarity from a handful of docs helps. — [DocILE](https://arxiv.org/abs/2302.05658) [read; per-cell values garbled in extraction]

### Inferences
- **Budget planning for "5 → 7 fields".** If the model is trained with randomized schemas and descriptions (section 1), expect the new fields to start well below 95% zero-shot. A **~10–50 doc top-up** (LoRA continued training mixing new-field examples with a replay of old-field data) should close most of the gap, judging from the LMDX curve. Reaching ≥95% field accuracy likely needs **≥100–200 docs**, or more for line-item fields. These are extrapolations from whole-schema curves on 2023-era LLMs.
- **Count positives, not documents.** In the local CORD set, `subtotal.discount_price` is non-null in 7.4% of docs and `subtotal.service_price` in 12.3%. "50 labeled docs" would give only ~4–6 positive examples. Sample documents *targeted at the new field* (pre-filter with zero-shot predictions or keyword search) and also label true-absent cases, so null precision can be measured.
- **Exact-match arithmetic.** With document-level exact match over N fields, field-level errors compound roughly as EM ≈ Π accuracy_i. Going from 5 to 7 required fields at ~99% each drops EM from ~95% to ~93%, so every new field needs ≥99% field accuracy to keep the 95% EM target. This is an approximation that assumes independent errors.
- **In-context examples for small VLMs**: given LayIE-LLM's mixed evidence and the context cost of image tokens, few-shot *fine-tuning* (LoRA top-up) is the safer path for Gemma 4 / Qwen3.5. In-context examples can be piloted as a no-training baseline in T3 of the holdout protocol.

### Gaps
- No source reports **per-field** learning curves, e.g., accuracy on a single newly added field vs. number of labeled positives, for VLM-based KIE.
- No 2025–26 data-efficiency curves for small open VLMs (Qwen2.5/3-VL, Gemma 3/4) on CORD were found this session.
- Not found: evidence on catastrophic forgetting of old fields when LoRA-topping-up a new field, or on optimal replay ratios, for document KIE.
