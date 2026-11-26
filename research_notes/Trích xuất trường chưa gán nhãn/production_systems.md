# Production IDP systems: adding or changing extracted fields without retraining, and how they keep accuracy

Research date: 2026-11-26. Each finding notes the source date or version where the page showed one. "Vendor claim" means a number from the vendor's own marketing or blog that nobody else has measured. "Measured" means a number from a benchmark run or a paper that gives its method.

## Q1. AWS: Textract Queries, Custom Queries adapters, and Bedrock Data Automation (BDA) blueprints

### Takeaway
AWS offers two paths. Textract Queries is a pretrained question-per-field model: a new field is just a new natural-language query, with no training. If that isn't accurate enough, a Custom Queries adapter fine-tunes the model on at least 5 training and 5 test documents, and training takes 2–30 hours. The newer generative path, BDA blueprints, works differently: each field has a natural-language instruction, and "instruction optimization" uses up to 10 labeled documents to rewrite those instructions without changing any model weights. AWS publishes no accuracy numbers for either path. It tells customers to measure F1 and exact match on their own test set.

### Cited Findings
**Textract Queries and Custom Queries adapters** (AWS Developer Guide; the pages show no date)
- Base Queries is a "pretrained Queries feature". Adapters "customize the output of its pretrained Queries feature" by training an adapter for the base model on the customer's own documents — [AWS best practices for Custom Queries](https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.md)
- Mechanism: "Adapters are modular components added to the existing Amazon Textract Core deep learning model… By fine-tuning a deep learning model with adapters, you can customize the output." The workflow is: upload samples → split into train/test → annotate queries and answers → train → get the AdapterId → pass it to `AnalyzeDocument`/`StartDocumentAnalysis` — [AWS: Customizing your Queries responses](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)
- Data requirements: "You must upload a minimum of five training and five test documents". The maximum is "2,500 pages of training documents and 1,000 test documents". AWS says to provide "as many documents for training as possible" — [AWS: Customizing your Queries responses](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)
- "A minimum of five samples (per query level) are required for either training or testing. The more samples used for training, the better." Annotated answers should vary: a yes/no query needs both "Yes" and "No" examples — [AWS best practices](https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.md)
- Labeling: you can auto-label with the pretrained Queries feature and then correct the labels, or label by hand. Training "takes 2-30 hours, depending on the size of the dataset and the AWS Region". A call can use "up to one adapter per page" — [AWS: Customizing your Queries responses](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)
- Validation loop: "After each round of adapter training, review the performance metrics in the console… improve your adapter's accuracy… by uploading a new batch of training documents or by reviewing annotations for documents that have low accuracy scores." Adapters are versioned, and AWS says to delete old versions you no longer need — [AWS: Customizing your Queries responses](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)
- Annotation-consistency rules that matter for exact match: if you annotate only one occurrence of a field that repeats, do the same in every document. Normalize spacing the same way everywhere (for example, "123 456 78" vs "12345678"). Don't type an answer that isn't shown in the document. "Use the exact query used in training for inference." For repeated keys, use hierarchical queries such as "What is the first name under co-borrower?" — [AWS best practices](https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.md)
- Search snippets of AWS pages give no numeric accuracy target or before/after improvement for adapters. The documents describe process and best practices only — [AWS Custom Queries tutorial](https://docs.aws.amazon.com/textract/latest/dg/textract-adapters-tutorial.html)

**Bedrock Data Automation (BDA) blueprints** (generative IDP; the optimization feature launched in December 2025)
- Each blueprint field has a natural-language instruction. "Blueprint instruction optimization" takes up to 10 representative document assets with expected values, compares them with inference results, and "adjusts the field descriptions to improve accuracy" — [AWS What's New, Dec 2025](https://aws.amazon.com/about-aws/whats-new/2025/12/bedrock-data-automation-optimization-document-blueprints); [BDA: Optimize your blueprints with ground truth](https://docs.aws.amazon.com/bedrock/latest/userguide/bda-optimize-blueprint-info.html)
- To decide whether a blueprint meets your accuracy bar, the console has you review overall F1, confidence score, and exact match rate — [BDA optimize docs](https://docs.aws.amazon.com/bedrock/latest/userguide/bda-optimize-blueprint-info.html)
- BDA returns per-field confidence and bounding boxes (visual grounding). According to the AWS ML blog, it tunes for Expected Calibration Error (ECE) so the confidence scores are better calibrated. One AWS example treats confidence below 70% as needing human verification, and says to test on your own data to choose the threshold. Audio and image blueprints don't return confidence — [AWS ML blog: Scalable IDP using BDA](https://aws.amazon.com/blogs/machine-learning/scalable-intelligent-document-processing-using-amazon-bedrock-data-automation); [AWS ML blog: Process multi-page documents with human review using BDA and SageMaker AI](https://aws.amazon.com/blogs/machine-learning/process-multi-page-documents-with-human-review-using-amazon-bedrock-data-automation-and-amazon-sagemaker-ai/); [BDA blueprint docs](https://docs.aws.amazon.com/bedrock/latest/userguide/bda-blueprint-info.html)

### Inferences
- AWS's own product line went from "fine-tune an adapter per customer" (Textract, 2023) to "rewrite the field instructions from a few labeled examples" (BDA, 2025). The second approach is essentially automated prompt optimization against a small ground-truth set, and it makes adding a field a configuration change rather than a training job.
- The adapter annotation rules (normalize values consistently, label every occurrence or only one, never type text that isn't on the page) are exactly the rules an in-house team needs in its labeling guide if it scores with exact match.
- Even AWS's floor of "5 train + 5 test" applies per query. That is the minimum labeling cost per new field on the fine-tuning path.

### Gaps
- I found no AWS-published number for how much an adapter improves accuracy (F1 before vs after). The launch blog's body could not be retrieved (fetch returned only navigation).
- I could not fetch the article body of the BDA "optimize blueprint extraction accuracy" blog post, so I have no before/after numbers from it.
- I did not verify whether Textract Queries returns per-answer confidence in its current version.

## Q2. Google Cloud Document AI Custom Extractor (generative AI)

### Takeaway
Google documents a three-step ladder explicitly. Zero-shot needs 0 labeled training documents (schema only). Few-shot needs 5–10 documents (5 recommended). Fine-tuning needs 10–50+. Google rates expected accuracy as medium, medium-high, and high, respectively. The models are Gemini-based processor versions. Field names and descriptions are the main way to steer accuracy. Each processor version snapshots the schema, so editing a field description does not change a deployed version. Evaluation reports F1, precision, and recall per field and per document, and you choose exact or fuzzy matching.

### Cited Findings
(Source page: docs.cloud.google.com "Custom extractor with generative AI". The page shows no date, but its version table lists versions up to 2026-10-08.)
- Training-approach table: zero-shot (accuracy medium, 0 training docs, "a test set is needed to evaluate"); few-shot (medium-high, 5 to 10 docs, five recommended); fine-tuning (high, 10 to 50+ docs). "Reasonably good results" are possible with 10–50+ examples, and hundreds or thousands can improve them further — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
- The same page reportedly says that fine-tuning `pretrained-foundation-model-v1.5-2025-05-05` requires at least 50 training and 50 test documents. That conflicts with the "10 to 50+" row, so check the live page before relying on either number — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
- The overview recommends the generative "fine-tuned foundation model" as "the first option for all variable layouts" with a "Medium: 0-50+ documents" training set. The alternatives are a model-based custom model ("High: 10-100+ documents") for forms whose layout varies, and a template-based model ("Low (3 documents)") for fixed layouts — [Google: Custom extractor overview](https://docs.cloud.google.com/document-ai/docs/custom-extractor-overview)
- Elsewhere, Google frames the foundation models as doing zero- to few-shot prediction from up to 5 labeled documents, with fine-tuning used above 10 labeled documents — [Google: Extracting overview](https://docs.cloud.google.com/document-ai/docs/extracting-overview)
- Processor versions and their Gemini base models, as listed on the page:
  - v1.5-2025-05-05 (Gemini 2.5 Flash): fine-tunable, stable, confidence scores
  - v1.5-pro-2025-06-20 (Gemini 2.5 Pro): not fine-tunable
  - v1.6-pro-2025-12-01 (Gemini 3 Pro): release candidate
  - v1.6-2026-01-13 (Gemini 3 Flash): release candidate
  - v3.5-2026-05-26 (Gemini 3.5 Flash): release candidate
  - v3.1-lite-2026-07-15 (Gemini 3.1 Lite): fine-tunable, release candidate. The listed release date conflicts with its version string.
  - v3.1-lite-2026-10-08: stable
  - Only some versions support fine-tuning or confidence scores. The v1.6 versions use the global Vertex endpoint and don't meet data residency requirements.
  — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
- Schema guidance:
  - Use descriptive field names, which "strongly affect accuracy".
  - Choose "Extract" for values stated in the document and "Derive" for values the model infers.
  - Add descriptions with location hints and value patterns.
  - Up to three levels of nesting for generative models (two for other processor types), with occurrence types such as "Optional multiple".
  - "Edits to a description don't propagate to an existing processor version, which keeps a snapshot."
  — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
- Labeling and evaluation:
  - Auto-labeling predicts fields from label names and earlier annotations. Humans review each prediction and "Mark as Labeled".
  - Metrics are F1, precision, and recall for the whole document and for each field.
  - Exact match requires an identical value. Fuzzy matching tolerates differences such as capitalization and can be configured on the Evaluation screen.
  - Accuracy is the test-set F1.
  - If a version misses production goals, revisit the training and test sets (zero-shot or few-shot) or add training documents (fine-tuning).
  — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
- Fine-tuning settings:
  - Jobs can take hours and time out at 72 hours.
  - The console allows 1–40 training steps (more through the REST API). The learning-rate multiplier ranges from 0.1 to 10.
  - After training, deploy the new version and set it as the default explicitly.
  — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
- Confidence scores range from 0 to 1. They are used to flag low-confidence entities for manual review and to set thresholds — [Google: Custom extractor overview](https://docs.cloud.google.com/document-ai/docs/custom-extractor-overview)

### Inferences
- Google's design is the clearest public template for "now we need 7 fields instead of 5":
  1. Add the fields to the schema with good names and descriptions.
  2. Run zero-shot against a labeled test set.
  3. If needed, add about 5 labeled documents for few-shot.
  4. Fine-tune only if still short.
  5. Each step produces a new immutable processor version that you evaluate per field and then promote to default.
- Google's configurable exact/fuzzy matching is a reminder to write down the matching rule per field for the 95% exact-match target. Otherwise reported accuracy shifts from run to run because normalization differs.
- Google rates zero-shot accuracy only "medium", even on Gemini. That suggests a small open VLM with a zero-shot prompt for brand-new fields should not be expected to reach 95% exact match without examples or fine-tuning (see Q6).

### Gaps
- Google publishes no numeric accuracy for zero-shot, few-shot, or fine-tuned modes. "Medium/medium-high/high" are qualitative labels.
- I could not resolve the "minimum 50 train/50 test for fine-tuning v1.5" claim against the "10–50+" table, or confirm the v3.1-lite release date, from the fetched page.
- Limits on the number of fields per schema were not stated.

## Q3. Azure: Document Intelligence custom neural vs. Content Understanding (generative)

### Takeaway
Microsoft's current docs (updated October–November 2026) describe two families:
- **Document Intelligence custom neural**: labeled fine-tuning. "Five examples… to get started"; 30-minute default training, up to 12 hours.
- **Content Understanding (CU)**: a generative field schema with descriptions, run zero-shot first. When accuracy or confidence is too low, you add "labeled samples" (as few as one per template), which are auto-labeled and then corrected by a human.

CU returns a 0–1 confidence and grounding (page, polygon, span) for every field type. Microsoft recommends per-field confidence thresholds and evaluating before and after adding samples. The Document Intelligence "custom generative" model page now redirects to the custom-model page, which lists only template and neural models.

### Cited Findings
**Document Intelligence custom models** (Microsoft Learn; ms.date 2026-05-22, updated 2026-10-19)
- The URL `.../document-intelligence/train/custom-generative-extraction` now redirects to the "Custom document models" page. That page describes only custom template and custom neural extraction models, plus classifiers — [Microsoft Learn: Custom document models](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0)
- "To create a custom extraction model, label a dataset of documents… You only need five examples of the same form or document type to get started." Custom neural "uses deep learning models and base model trained on a large collection of documents… then fine-tuned or adapted to your data." Microsoft recommends starting with neural over template — [Microsoft Learn: Custom document models](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0)
- Neural training time: "30 minutes to 12 hour" (30 minutes by default; longer requires paid training). Limits: up to 50,000 training pages and 1 GB. The v4.0 2024-11-30 API added "table, row and cell level confidence" and overlapping fields. For low-quality images, "use a larger dataset (more than five training documents)" — [Microsoft Learn: Custom document models](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0)
- Lifecycle: a custom model lives as long as the API version it was trained on. "The custom model isn't available for inference when the API version is deprecated" — [Microsoft Learn: Custom document models](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0). The migration guide adds that all custom models get an expiration date so base models can be retired. After expiry, the model must be retrained on the latest API — [Microsoft Learn: DI migration guide](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/versioning/migration-guide-overview)

**Content Understanding** (overview ms.date 2026-10-29; analyzer-improvement and best-practices ms.date 2026-07-16, updated 2026-10/11)
- `2025-11-01` is GA and recommended for production. `2026-06-01-preview` adds agentic mode and improved training — [Microsoft Learn: CU overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/overview)
- Each field has a method: **Extract** (the value appears in the document; documents only), **Classify** (from a predefined set), or **Generate** (free-form or inferred). Customers bring their own Foundry LLM deployments — [Microsoft Learn: CU overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/overview)
- Confidence (0 to 1) and grounding (page number, polygon, and span offset/length) are enabled with `estimateFieldSourceAndConfidence`. In the 2025-11-01 GA and 2026-06-01-preview APIs they cover all field methods — [Microsoft Learn: CU analyzer improvement](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)
- "If the context for all fields is clearly present in the test document, a zero-shot extraction call might be sufficient… If you still see incorrect field values or confidence scores below your straight-through processing threshold, use labeled samples" — [Microsoft Learn: CU analyzer improvement](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)
- Labeled-sample guidance:
  - "For datasets with minimal template variations, add a single labeled sample. For more complex variations, add a representative sample for each template or format."
  - Auto label runs the existing analyzer, and a human corrects the result, which is tagged "corrected".
  - Rebuild the analyzer after adding samples.
  - "Evaluate extraction quality before and after training."
  — [Microsoft Learn: CU analyzer improvement](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)
- Limits:
  - Training "doesn't currently support fields that use the `generate` method".
  - "Labeled samples don't correct text recognition problems." If OCR reads `l` as `1`, labeling won't help.
  - In GA, labeled samples are used at runtime (in-context). In `2026-06-01-preview`, training "distills the relevant patterns… into the built analyzer", which "reduces token consumption", stops retaining the samples, and removes the need to access them at analysis time. The preview has no SLA.
  — [Microsoft Learn: CU analyzer improvement](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)
- Schema best practices:
  - Write detailed descriptions with location hints, format expectations, and alternative labels.
  - List all aliases.
  - Use affirmative language.
  - **"Define field names and descriptions in the same language as your file. Language mismatches can significantly reduce accuracy."**
  - Use arrays of objects for repeated items, not strings that ask for JSON.
  - Set extract/generate/classify explicitly.
  - "Prioritize refining field descriptions before adding labeled training examples."
  — [Microsoft Learn: CU best practices](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/concepts/best-practices)
- Per-field confidence thresholds (illustrative): critical fields ≥0.90 (TotalAmount, ContractTerminationDate), important fields ≥0.80 (VendorName, InvoiceNumber), non-critical fields ≥0.70 (Comments). "Determine thresholds experimentally for each use case" — [Microsoft Learn: CU best practices](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/concepts/best-practices)
- Microsoft claims CU "employs multiple AI models to analyze and cross-validate information simultaneously". This is a marketing claim with no published numbers — [Microsoft Learn: CU overview](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/overview)
- Model churn: the CU preview→GA migration guide says analyzers must name default completion and embedding models. The GPT-4.1 family retires in October 2026, and Microsoft recommends gpt-5.2 — [Microsoft Learn: CU migration preview to GA](https://learn.microsoft.com/en-au/Azure/ai-services/content-understanding/how-to/migration-preview-to-ga)

### Inferences
- Microsoft and Google give the same order of operations: description first, then a few corrected examples, then (DI only) full fine-tuning. Each step is gated by before/after evaluation on a representative set.
- The CU preview "distill labeled samples into the analyzer" feature shows vendors moving from runtime few-shot (token cost, data retention) toward compiled adaptation. For an in-house vLLM stack, the analogue is to start with few-shot in the prompt for new fields, then fold those examples into the next LoRA/SFT round.
- The "same language as the document" guidance applies directly if the team's documents are Vietnamese. Field descriptions should probably be written in the document's language, or at least A/B-tested that way.
- Base-model retirement (GPT-4.1 in CU, expiring DI custom models) means a vendor-hosted generative pipeline still needs a regression test set to re-validate fields whenever the underlying model changes. A self-hosted pipeline with pinned Gemma/Qwen weights avoids forced upgrades.

### Gaps
- I could not confirm the retirement date or status of the Document Intelligence "custom generative" model. The redirect suggests it was removed or merged into CU, but I found no explicit notice.
- Microsoft publishes no accuracy numbers for CU zero-shot vs. with labeled samples, and nothing on how calibrated its confidence is.

## Q4. Other vendors and APIs: LlamaExtract, Reducto, Unstructured, Extend, Instabase, Rossum, NuMind, Mistral, OpenAI/Gemini

### Takeaway
None of the developer-API vendors (LlamaExtract, Reducto, Unstructured, Mistral, Extend) fine-tunes a model per customer as the default path. You add a field by editing a JSON schema and its descriptions. Accuracy is then defended with per-field citations and bounding boxes, a confidence score (often split into parse vs. extract), evaluation sets, and a human-review queue. The enterprise IDP vendors (Rossum, Instabase, NuMind) do offer customer-specific learning from a few corrected documents. Their accuracy numbers are mostly vendor claims.

The best measured data point is LlamaIndex's own published threshold curve on ExtractBench. With no confidence filter, precision was 94.69%. Raising the filter to reach 97% precision sent about 27% of values to review, and recall fell to 66.48%.

### Cited Findings
**LlamaExtract (LlamaIndex / LlamaParse Extract)** (docs undated; Extract v2 default, `configuration.version` e.g. `2.5`)
- The docs guarantee schema-compliant output. v2 tiers are Agentic Plus (highest quality; "Max effort" option), Agentic, Cost Effective, and Turbo. v1 had FAST, MULTIMODAL, and PREMIUM. The only accuracy statement is qualitative: Turbo "scores above Cost Effective on accuracy" on their extraction benchmark. The page does not mention per-customer fine-tuning — [LlamaExtract getting started](https://developers.llamaindex.ai/python/cloud/llamaextract/getting_started)
- Citations: for each leaf field you get page, `matching_text` (verbatim source), `bounding_boxes`, and page dimensions. Citations attach at the leaf sub-field level, not per array item. Turbo returns text-only citations (no boxes). "Advanced citations" are on by default from Agentic 2.5 — [LlamaExtract metadata extensions](https://developers.llamaindex.ai/python/cloud/llamaextract/features/extensions); [LlamaExtract response format](https://developers.llamaindex.ai/llamaparse/extract/guides/response-format/)
- Confidence: enable it with `confidence_scores: true`. You get `confidence` (combined; threshold on this one), `parsing_confidence`, and `extraction_confidence`. The docs say the scores are **uncalibrated**, that long text fields score lower, and that citations and confidence slow extraction noticeably. Both are off by default, and neither is available in spreadsheet mode — [LlamaExtract metadata extensions](https://developers.llamaindex.ai/python/cloud/llamaextract/features/extensions)
- **Measured threshold trade-off.** LlamaIndex blog, 2026-09-16, ExtractBench (370 documents, 841,895 expected fields), configuration "LlamaParse Agentic Plus":
  - Cutoff 0 (no filter): precision 94.69%, recall 88.67%.
  - Cutoff 0.7717: precision 97.0%, recall 66.48%. 559,666 values accepted correctly, 17,309 accepted incorrectly, 211,401 sent to review (about 26.8% of 788,376 returned values; my calculation).
  - Recommended method: start with a low cutoff and raise it until accepted values meet the precision target. Validate the cutoff on a separate representative set and re-run periodically as the document mix drifts. "Missing fields need separate checks."
  — [LlamaIndex: What makes an extraction confidence score useful](https://www.llamaindex.ai/blog/what-makes-an-extraction-confidence-score-useful)

**Reducto** (docs undated; "v3" referenced)
- Schema-driven Extract. The SDK docs call field descriptions "critical for accurate extraction". To reduce run-to-run variance: use enums, make descriptions more specific, and reword descriptions "to match how the value appears in the document" — [Reducto Python SDK: extract](https://docs.reducto.ai/sdk/python/extract.md); [Reducto extract response format](https://docs.reducto.ai/extract/response-format)
- Each citation carries a categorical `confidence` ("high"/"low") and, by default in v3, a `granular_confidence` with numeric 0–1 parse and extract scores. A low parse score means OCR or layout trouble. "Low extract_confidence suggests the schema description may need refinement" — [Reducto citations](https://docs.reducto.ai/citations); [Reducto extract response format](https://docs.reducto.ai/extract/response-format)
- Citation constraints:
  - Turning citations on disables chunking; the document is processed as one unit.
  - Citations can be empty when the value was inferred or couldn't be localized.
  - Citations aren't supported when `agent_in_the_loop` is enabled.
  — [Reducto extract citations config](https://docs.reducto.ai/configs/extract/citations)

**Unstructured**
- An Extract node sits in a workflow before the destination. Schemas use "the JSON format used by OpenAI Structured Outputs", with a visual schema builder. Extraction runs with an LLM or regex. Maximum schema nesting depth is 10. A "Schema-Only Output" toggle controls the output shape — [Unstructured: structured data extractor](https://docs.unstructured.io/ui/structured-data-extractor/using-the-structured-data-extractor); [Unstructured blog: Introducing Extract](https://unstructured.io/blog/introducing-extract)

**Extend** (all vendor-authored)
- Review Agent (blog 2026-01-13):
  - It is a second LLM pass that checks each field against the source document and schema.
  - It returns `reviewAgentScore` 1–5 per field, array item, and property. "≤3 usually warrants review."
  - Breaking a user-defined "critical" rule forces the score to 1.
  - Routing uses `minReviewAgentScore` and `numFieldsFlaggedForReview`.
  - Extend argues OCR confidence and logprobs "measure the wrong thing" (a model can be confidently wrong when it picks the wrong one of two similar values). **The post gives no measured precision or recall for the agent.**
  — [Extend: Introducing Review Agent](https://www.extend.ai/resources/introducing-review-agent-re-gaining-confidence-in-confidence-scores); [Extend docs: Review agent](https://docs.extend.ai/extraction/review-agent)
- Evaluation sets are attached to a processor and pair files with expected output. They can be run via API or in Studio — [Extend docs: Evaluation overview](https://docs.extend.ai/evaluation/overview)
- Schema versioning (blog published 2026-04-13): "draft, test, publish, and pin extraction configs to specific workflow versions", instant rollback, and regression runs against evaluation sets before a schema change deploys. Corrections from the review UI flow back into evaluation sets. The "Composer" agent tries prompt and schema variants against eval sets in the background. Extend's "99%+ accuracy" claim is not independently verified — [Extend: Schema versioning tools](https://www.extend.ai/resources/schema-versioning-tools-document-ai-pipelines); [Extend: Self-improving document processing systems](https://www.extend.ai/resources/self-improving-document-processing-systems)

**Instabase AI Hub**
- Validation rules come in two kinds. Confidence rules use confidence scores. Custom rules use validation prompts or functions. Failing rules route documents to human review. Confidence types are field, classification, and OCR. Validation rules "are also factored into accuracy metrics" — [Instabase docs: Validating documents](https://docs.instabase.com/automate/validating-documents)
- Instabase notes that LLMs "do not produce confidence scores natively". It sells a Human Review Suite with reviewer groups, assignment, and escalation. It publishes no accuracy figures — [Instabase blog: Overcoming the limitations of LLMs](https://instabase.com/blog/overcoming-the-limitations-of-llms-data-validation-and-human-review); [Instabase whitepaper (2025-07)](https://instabase.com/wp-content/uploads/2025/07/Why-Instabase_-The-Future-of-Document-Understanding.pdf)

**Rossum Aurora** (all vendor claims)
- "Instant learning" (few-shot) adapts to new formats and custom fields from a handful of corrected documents.
- Rossum claims "10x fewer documents" to reach target accuracy, against an older baseline that needed "up to 5000 documents".
- One early adopter (Adyen) is quoted at 92.6% accuracy after 20 documents.
- Rossum claims an "average 37.6% error reduction" and "zero hallucinated values". It also claims 276 languages, 30 with handwriting.
- The baseline is Rossum's own previous model generation.
— [Rossum: Aurora 1.5 and Copilot](https://rossum.ai/blog/rossum-aurora-1-5-and-copilot/); [Rossum: Aurora](https://rossum.ai/blog/rossum-aurora/); [Rossum press release](https://rossum.ai/company/newsroom/press-release-rossum-aurora/)

**NuMind (NuExtract)**
- NuExtract 3 takes the document, a JSON template whose structure matches the output, and optional in-context-learning (ICL) examples. Docs recommend non-reasoning mode by default and reasoning mode only for hard cases — [Hugging Face: numind/NuExtract3](https://huggingface.co/numind/NuExtract3)
- Earlier guidance: NuExtract-tiny "is intended to be fine-tuned on a specific task (at least 30 examples needed)". This is a mirrored model card for an old model — [HF: NuExtract-tiny (mirror)](https://huggingface.co/philipp-zettl/NuExtract-tiny). The NuExtract 2 card advises temperature at or near 0 — [GitHub: numindai/nuextract](https://github.com/numindai/nuextract)
- In 2024, NuMind's CEO said users can fine-tune NuExtract on input→output pairs and that "we are working on a platform to make this fine-tuning easier" — [Hacker News thread, 2024](https://news.ycombinator.com/item?id=40834089)

**Mistral Document AI (OCR + annotations)**
- `document_annotation_format` returns a whole-document structured annotation. `bbox_annotation_format` annotates figure and chart bounding boxes that OCR detects. Both are JSON schemas, with Pydantic/Zod helpers in the SDKs — [Mistral docs: Annotations](https://docs.mistral.ai/studio/document-processing/annotations); [Mistral cookbook: Extract data via annotations](https://docs.mistral.ai/cookbooks/mistral-ocr-data_extraction)
- The cookbook says document annotations are limited to 8 pages, so long documents should be split. bbox annotations don't have that limit — [Mistral cookbook](https://docs.mistral.ai/cookbooks/mistral-ocr-data_extraction)
- A user reports that Azure-hosted "mistral-document-ai-2505" sometimes returns `null` for the annotation while the markdown OCR succeeds. This is an anecdote about silent failures — [Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/5767943/mistral-doc-ai-2505-gives-null-in-the-data-annotat)

**OpenAI / Gemini structured outputs as an IDP base**
- OpenAI `response_format: {type: "json_schema"}` (Structured Outputs) is supported on gpt-4o-2024-08-06 and later snapshots — [OpenAI: Structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs); [OpenAI: Introducing Structured Outputs (2024-08)](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- A developer forum post reports a 500-enum-value cap per schema, and other threads report opaque "invalid schema" errors for large or deep schemas. These are community reports, not official limits — [OpenAI community: 500 enums](https://community.openai.com/t/500-enums-limitation-in-structured-output/974416); [OpenAI community: depth and properties](https://community.openai.com/t/measuring-maximum-depth-and-object-properties-in-structured-outputs/918388)
- Google's Document AI custom extractor is itself built on Gemini 2.5/3/3.5 processor versions (see Q2). Google's managed IDP layer adds versioning, auto-labeling, evaluation, and fine-tuning on top of Gemini — [Google: Custom extractor with generative AI](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)

### Inferences
- Two vendors split confidence into "parse" and "extract" (LlamaExtract `parsing_confidence`/`extraction_confidence`; Reducto parse/extract granular confidence). This tells the user what to fix: OCR or layout errors vs. schema-description errors. An in-house OCR-text→JSON pipeline can copy the split with OCR token confidence plus a model-side score.
- Vendors say plainly that LLM confidence is uncalibrated (LlamaExtract docs) or unreliable as logprobs (Extend). The measured LlamaIndex curve shows how expensive a high precision target is: going from about 94.7% to 97% precision cost about 27% of values going to human review. A 95% field exact-match target with automated pass-through will need per-field calibrated thresholds tuned on held-out data, not a single global cutoff.
- Citations are often weakest exactly where extraction is hardest: they are dropped for inferred values (Reducto), turned off in agentic modes, or limited to text on fast tiers (LlamaExtract Turbo). A team that wants grounding for every field should prefer the "Extract" (copy from the document) method over "Generate/Derive" for regulated fields.

### Gaps
- I couldn't find or verify a NuMind claim that customized NuExtract cut errors from 70% to 3% on Japanese insurance documents (it appeared only in a search summary without a confirmed primary page). I also couldn't confirm whether nuextract.ai currently offers per-customer fine-tuning.
- Mistral documents no per-field confidence score for annotations (none found).
- I found no Gemini API documentation fetched on schema limits, and no official OpenAI property-count or nesting limits (the official "Supported schemas" text wasn't retrieved).
- No vendor in this group publishes a calibration metric (ECE/AUROC) for its confidence, except AWS's statement that BDA optimizes ECE, with no number given.
- I didn't verify whether LlamaExtract, Reducto, or Unstructured offer per-customer fine-tuning on enterprise plans. Their public docs describe only schema and prompt configuration.

## Q5. Common production patterns for adding or changing fields

### Takeaway
Across vendors, the pattern converges:
1. **Schema with rich per-field descriptions** in the document's language, with explicit extract/classify/generate method and enums.
2. **Zero-shot first**, measured on a labeled test set with per-field F1 and exact match.
3. **A few corrected examples**: 1 per template for Azure CU, 5–10 for Google few-shot, up to 10 for AWS BDA, 5+5 per query for Textract adapters. These are bootstrapped by auto-labeling with the current model and corrected by humans.
4. **Fine-tune only if still short**: Google 10–50+ documents, Azure DI neural 5+.
5. **Immutable versions** of schema and model, promoted after regression runs.
6. **Per-field confidence thresholds plus a review queue**, with corrections flowing back into eval and training sets.

### Cited Findings
- Descriptions before data: "Prioritize refining field descriptions before adding labeled training examples" — [Azure CU best practices](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/concepts/best-practices). Google: descriptive names "strongly affect accuracy", and descriptions should carry location and pattern hints — [Google ce-with-genai](https://docs.cloud.google.com/document-ai/docs/ce-with-genai). Reducto: low extract confidence means "the schema description may need refinement" — [Reducto response format](https://docs.reducto.ai/extract/response-format). AWS BDA optimizes the descriptions themselves from ground truth — [AWS What's New, Dec 2025](https://aws.amazon.com/about-aws/whats-new/2025/12/bedrock-data-automation-optimization-document-blueprints)
- Auto-label then correct:
  - Textract: auto-label with pretrained Queries, then edit — [AWS](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)
  - Google: auto-labeling from label names and earlier annotations, human "Mark as Labeled" — [Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
  - Azure CU: "Auto label" prepopulates, humans correct, and the field is tagged "corrected" — [Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement)
  - Extend: review UI corrections feed evaluation sets — [Extend](https://www.extend.ai/resources/schema-versioning-tools-document-ai-pipelines)
- Versioning:
  - Google processor versions snapshot the schema, and description edits don't change existing versions — [Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
  - Textract adapters have versions — [AWS](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md)
  - Extend has draft/publish/pin/rollback — [Extend](https://www.extend.ai/resources/schema-versioning-tools-document-ai-pipelines)
  - Azure DI custom models expire with their API version — [Azure DI](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0)
- Regression and evaluation sets:
  - Google: per-field and per-document F1/precision/recall, exact vs fuzzy — [Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai)
  - BDA: F1, exact match rate, confidence — [AWS BDA](https://docs.aws.amazon.com/bedrock/latest/userguide/bda-optimize-blueprint-info.html)
  - Azure: "Evaluate before and after training… so you can confirm training improved results instead of just changing them" — [Azure CU best practices](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/concepts/best-practices)
  - Extend: regression tests on evaluation sets before deploy — [Extend](https://www.extend.ai/resources/schema-versioning-tools-document-ai-pipelines)
- Confidence thresholds and human review:
  - Azure: critical ≥0.90, important ≥0.80, non-critical ≥0.70 (illustrative) — [Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/concepts/best-practices)
  - AWS example: <70% goes to review — [AWS blog](https://aws.amazon.com/blogs/machine-learning/process-multi-page-documents-with-human-review-using-amazon-bedrock-data-automation-and-amazon-sagemaker-ai/)
  - Extend: ≤3 of 5 goes to review — [Extend](https://www.extend.ai/resources/introducing-review-agent-re-gaining-confidence-in-confidence-scores)
  - LlamaIndex: raise the cutoff until the precision target holds, then validate on a separate set — [LlamaIndex](https://www.llamaindex.ai/blog/what-makes-an-extraction-confidence-score-useful)
- Rule-based validation as a second guard: Instabase confidence rules plus custom prompt or function rules — [Instabase](https://docs.instabase.com/automate/validating-documents). Extend "critical" rules force a score of 1 — [Extend docs](https://docs.extend.ai/extraction/review-agent)
- Coverage of layouts: "add a representative sample for each template or format" — [Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement). "sample data should contain layout variations and image variations that are representative" — [AWS](https://docs.aws.amazon.com/textract/latest/dg/best-practices-adapters.md)
- Example labeling volumes for a new field (per vendor docs):
  - Textract adapters: ≥5 train + ≥5 test documents, ≥5 samples per query
  - Google: 0 / 5–10 / 10–50+ documents by mode
  - Azure CU: ≥1 per template
  - Azure DI neural: ≥5
  - AWS BDA: up to 10 assets
  - NuExtract-tiny: ≥30 examples
  - Sources: [AWS](https://docs.aws.amazon.com/textract/latest/dg/textract-using-adapters.md), [Google](https://docs.cloud.google.com/document-ai/docs/ce-with-genai), [Azure CU](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/analyzer-improvement), [Azure DI](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-model?view=doc-intel-4.0.0), [AWS BDA](https://aws.amazon.com/about-aws/whats-new/2025/12/bedrock-data-automation-optimization-document-blueprints), [NuExtract-tiny](https://huggingface.co/philipp-zettl/NuExtract-tiny)

### Inferences
- A concrete in-house version for the "5→7 fields" case (my synthesis, not any single vendor's procedure):
  1. Bump the schema version and add the two fields with descriptions, aliases, format, and location hints in the document's language.
  2. Run the current fine-tuned model with the new schema in the prompt (zero-shot), plus 3–5 in-context examples that show the new fields.
  3. Have the model auto-label a batch, then human-correct about 20–50 documents (the vendor ranges above) for both a held-out test split and a training split.
  4. Report per-field exact match, so the old 5 fields are checked for regression separately from the 2 new ones.
  5. If the new fields miss target, fold the corrected examples into the next LoRA/SFT round.
  6. Promote the new model and schema version only if the old fields don't regress.
  7. Route new-field values below a per-field calibrated threshold to review until enough corrections accumulate.
- Any model fine-tuned on a fixed 5-field output format may have lost the ability to follow a changed schema in the prompt. NuExtract's design (template as input, plus ICL examples) and the vendors' schema-in-prompt designs suggest training on many varied schemas, field subsets, and descriptions, so that schema changes stay a prompt-side operation. This is an inference, not a vendor statement.

### Gaps
- No vendor publishes how many labeled documents a new field typically needs to reach a given exact-match level. All guidance is in document-count ranges without accuracy curves.
- I found no vendor documentation describing "active learning" in the strict sense (uncertainty sampling to choose which documents to label). What vendors describe is human correction of low-confidence production outputs flowing back into the training or eval set.

## Q6. Published benchmarks and case studies: zero-shot schema extraction vs. fine-tuned

### Takeaway
No public benchmark compares vendor zero-shot schema extraction against fine-tuned extraction on the same documents. The available evidence points the same way, though:
- On the IDP Leaderboard v1.5, small open models used zero-shot fall far behind frontier models (Qwen3.5-4B 74.5, Gemma-4-E4B-it 55.0, Gemini 3.1 Pro 89.6 on the IDP column).
- In an ICDM 2026 paper, LoRA fine-tuning moves Qwen3.5-4B from 0.758 to 0.966 F1 on a text extraction task, close to an 8B production model.
- In a Spanish invoice study, few-shot prompting alone raised F1 from 82.92% to 96.11%.

### Cited Findings
- **IDP Leaderboard (Nanonets), v1.5, last-updated date not shown:**
  - Covers OCR, tables, KIE, and VQA. "The overall score is the mean of all benchmark scores."
  - Top scores (overall / IDP column): Nanonets OCR-3 85.9 overall; GPT-5.4 83.5 / 84.4; Gemini-3-Pro 82.8 / 81.8; Gemini 3.1 Pro 89.6 on IDP.
  - Small open-weight families (overall / IDP): Qwen3.5-9B 76.7 / 76.2; Qwen3.5-4B 72.5 / 74.5; Qwen3.5-2B 62.6 / 67.1; Qwen3.5-0.8B 57.8 / 61.2; Gemma-4-E4B-it 53.9 / 55.0; Gemma-4-E2B-it 41.9 / 44.1.
  - The page doesn't say whether the "IDP" column equals KIE.
  — [IDP Leaderboard](https://benchmarking.nanonets.com/)
- The IDP Core benchmark's KIE component uses about 2,000 invoices, receipts, forms, and handwritten documents, scored with exact-match and fuzzy-match metrics — [IDP Core Bench](https://benchmarking.nanonets.com/benchmarks/idp). One aggregator lists Gemini-3-Flash first on IDP KIE at 91.10. This is secondary and not reconciled with the leaderboard page — [benchmarklist.com: IDP Leaderboard](https://benchmarklist.com/benchmarks/idp_leaderboard/). The original 2025 launch found Gemini 2.5 Flash best at that time — [Nanonets: IDP Leaderboard 1.5 blog](https://nanonets.com/blog/idp-leaderboard-1-5/)
- **"How Small Can You Go? LoRA Fine-Tuning 270M–8B Models for Merchant Information Extraction in Financial Transactions"** (arXiv 2606.08051, v1 2026-06-06, v4 2026-09-08; accepted at IEEE ICDM 2026). This is **text** extraction from transaction strings, not document images.
  - Models: Gemma 3 (270M–4B), Qwen 3.5 (0.8B–4B), Aya 3.35B, LLaMA 3.1-8B.
  - Best zero-shot (Qwen 3.5 4B, JSON-only prompt): 0.758 F1 and 0.473 strict record-level exact match.
  - Fine-tuned: 96.60% F1 and 91.67% strict record-level exact match.
  - Production LLaMA 3.1-8B reference: 96.95% F1. Qwen 3.5 0.8B fine-tuned: 94.75% F1.
  - Qwen 3.5 4B's inverse-throughput estimate is 3.8× lower than the rank-8 8B model's.
  - Across 14 Databricks serving endpoints, mean F1 changed by −0.0081 from local evaluation. Aya fell 2.7–5.1 points.
  — [arXiv 2606.08051 (abstract)](https://arxiv.org/abs/2606.08051); zero-shot numbers from [arXiv PDF](https://arxiv.org/pdf/2606.08051)
- **Spanish electricity invoices, prompting only** (arXiv 2604.25927): Mistral-small zero-shot F1 82.92% vs. cross-validated few-shot 96.11% — [arXiv 2604.25927](https://arxiv.org/pdf/2604.25927)
- **LlamaIndex ExtractBench** (vendor-run, measured, 2026-09-16): a frontier-model agentic pipeline reaches 94.69% precision and 88.67% recall with no confidence filtering. That is roughly the ceiling of zero-shot schema extraction on a hard, diverse benchmark — [LlamaIndex blog](https://www.llamaindex.ai/blog/what-makes-an-extraction-confidence-score-useful)
- **Vendor-claimed few-shot gains:** Rossum claims 10× fewer documents, a 37.6% average error reduction, and Adyen at 92.6% after 20 documents. None is independently measured — [Rossum](https://rossum.ai/blog/rossum-aurora-1-5-and-copilot/)
- AWS states that fine-tuning small base models "can deliver similar results as fine-tuning large base models" for document-to-JSON. The article body (and its numbers) could not be retrieved — [AWS ML blog: Fine-tune VLMs for multipage document-to-JSON](https://aws.amazon.com/blogs/machine-learning/fine-tune-vlms-for-multipage-document-to-json-with-sagemaker-ai-and-swift/)

### Inferences
- For a team targeting 95% field exact match with Gemma 4 and Qwen3.5 small VLMs, the evidence says zero-shot on a brand-new field is a bootstrap, not a production answer.
  - Small models score 55–76 zero-shot on the IDP Leaderboard.
  - Strict record exact match for zero-shot Qwen3.5-4B was 0.473 in the ICDM paper.
  - Fine-tuning closes most of the gap (0.917 record exact match).
  - Few-shot prompting can add about 13 F1 points on invoices.
  - A realistic plan is: zero-shot plus few-shot to bootstrap labels for the new field, then a short LoRA/SFT refresh, gated by per-field regression tests.
- The record-level vs. field-level gap in the ICDM paper (96.6% F1 vs. 91.7% strict record exact match) shows how a 95% target behaves at different granularities. A 95% field-level target implies a much lower document-level exact match when documents have many fields. The team should state which level the 95% applies to.

### Gaps
- I found no study that runs zero-shot schema extraction, few-shot, and fine-tuning on the same document-image KIE dataset with the same metric for small VLMs (Gemma 4 or Qwen3.5 VL).
- I did not retrieve per-dataset KIE scores for Gemma 4 or Qwen3.5 on the IDP Leaderboard (only the overall and IDP columns), or the leaderboard's GitHub methodology page.
- I did not retrieve results from OmniDocBench, RealKIE (arXiv 2403.20101), or DocILE for this question. OmniDocBench is mainly a parsing/OCR benchmark rather than schema extraction.
- The VirtusLab write-up on SFT for small Qwen3-VL invoice extraction was in search results, but I did not retrieve its zero-shot baseline numbers ([VirtusLab](https://virtuslab.com/blog/ai/sft-scaling-small-vision-language-models/)).
