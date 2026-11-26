# Data formats and contracts

Every track reads or writes these files. They are implemented in `src/llm_opt_lab/` and
covered by `tests/`. Changing one is a core change: open a PR and ask the lead to review.

| Contract | Written by | Read by | Code |
|----------|-----------|---------|------|
| [Dataset records](#1-dataset-records) | Lead (data tasks) | everyone | `data.py` |
| [Doc types](#2-doc-types) | Lead (data tasks) | everyone | `doc_types.py` |
| [Prompt construction](#3-prompt-construction) | — | training, HF inference, vLLM, calibration | `prompting.py`, `field_requests.py` |
| [Predictions](#4-predictions) | every inference path | `scripts/evaluate.py` | `predictions.py` |
| [Scoring](#5-scoring) | — | `scripts/evaluate.py` | `parsing.py`, `postprocess.py`, `metrics.py` |
| [Metrics JSON](#6-metrics-json) | every stage | result tables (T-L07), README | `results.py` |
| [Run paths](#7-run-paths) | everyone | everyone | `paths.py` |
| [Configs and model sources](#8-configs-and-model-sources) | everyone | every script | `config.py`, `paths.py` |

## 1. Dataset records

A dataset root (`data.dir` in a config: `${LLMOPT_DATA_DIR}/public/cord` for the public
dev set, `${LLMOPT_DATA_DIR}/private` for company data) looks like:

```text
<data root>/
├── all.jsonl              # every record, before splitting (input of split_dataset.py)
├── train.jsonl  validation.jsonl  test.jsonl  calibration.jsonl
├── benchmark.jsonl        # subset of test for serving benchmarks (make_benchmark.py)
├── split_manifest.json    # seed, ratios, ids per split
└── images/...             # image paths in records are relative to the root
```

One JSON object per line, **one record per (document, input type)**. The image of a
document and its spatial text are two records sharing a `group_id`:

```json
{
  "id": "receipt-001-img",
  "doc_type": "receipt",
  "input_type": "image",
  "group_id": "receipt-001",
  "image": "images/receipt-001.png",
  "target": {
    "store": {"name": "Cửa hàng An", "address": "12 Lê Lợi, Quận 1"},
    "date": "2026-01-05",
    "items": [{"name": "Cà phê sữa", "quantity": 2, "price": 40000}],
    "total": 65000
  },
  "meta": {"source": "internal:batch-07"}
}
```

| Field | Required | Meaning |
|-------|:--------:|---------|
| `id` | yes | Unique across the whole dataset |
| `doc_type` | yes | Folder name under `doc_types/` |
| `input_type` | yes | `image` or `spatial_text` |
| `group_id` | no | Same value for records of the same source document; the split keeps a group together. Defaults to `id` |
| `image` | if `input_type` is `image` | Path relative to the data root |
| `spatial_text` | if `input_type` is `spatial_text` | The spatial text, inline |
| `target` | yes | Ground truth JSON, nested, following the doc type schema |
| `label_source` | no | `human`, `teacher` (written by a model, e.g. the production Qwen3.6, not checked) or `teacher_reviewed` (written by a model, then checked by a person) |
| `meta` | no | Anything else (source, annotator, flags). Never used by training or scoring |

**What the target says about each field.** A field is *annotated* when its key is present:

| In the target | Meaning | Training | Scoring |
|---------------|---------|----------|---------|
| `"tax": "2,727"` | the value, as printed | trained on | scored |
| `"tax": null` (`[]` for a list) | checked: not on the document | trained on (the answer is null) | scored |
| key missing | not annotated (e.g. the field was added to the registry later) | never asked | skipped when asked |

So a record annotated before a field existed stays usable: it is never asked for that field.
Every entry of a list annotates the same keys. A target holds **printed facts only**, never a
`derive` field (§2): business rules compute those after extraction.

Unknown fields are rejected. `scripts/data/validate.py` also checks every target against
its doc type schema, that image files exist, that no target holds a `derive` field, and that
list entries annotate the same keys; `--reviewed-only` also requires a `human` or
`teacher_reviewed` label (test splits, T-L03). How to write `target` is in the annotation
guideline (T-L02).

## 2. Doc types

Each document type has a field registry and instructions:

```text
doc_types/<name>/
├── instructions.md    # what the document is and doc-type rules; never lists fields
└── schema.json        # the field registry: JSON Schema of the target plus what the model sees
```

The schema uses a subset of JSON Schema: `object` with `properties`, `array` with
`items`, and leaves. A leaf may set:

| Key | Meaning |
|-----|---------|
| `description` | What the field is and where it is printed; shown to the model. Write it so a new annotator could find the value |
| `examples` | 1–3 typical values as printed; shown to the model |
| `x-field-type` | Normalizer used in scoring and production (default `text`; also `date`, `number`, filled in by T-L05). Unknown types are an error |
| `x-mode` | `extract` (default): printed on the document, asked of the model. `derive`: computed by business rules after extraction (T-L05), never asked, never in a target |

The root may list `x-decoys`: `[{"name", "description"}]` fields that can never appear on
this doc type (a container number on a receipt). Training and evaluation ask for them to
teach and measure answering `null`. Keeping real JSON Schema means vLLM structured outputs
can use the same file (T-C09). `doc_types/cord_receipt/` is the public dev doc type (12
fields); `doc_types/cord_receipt_full/` has every CORD field, for the field-holdout
experiment E20.

**Raw data.** The model returns values exactly as printed (`"5 Pallet"`, `"45,500"`).
Two layers run after it, in code, in production as in scoring:

1. Normalization (`postprocess.py`, T-L05): one value, a fixed mapping that does not depend
   on the customer: `"45,500"` → `45500`, a date → ISO 8601, `"Pallet"` → a unit code.
2. Business rules (T-L05), per customer and versioned: anything that combines facts or
   depends on who asks, e.g. "3 Pallet, 5 Carton" → 3, 5 or 8 units. These are `derive` fields.

Neither is ever taught to the model, so a rule or code-list change needs no new labels and
no retraining.

## 3. Prompt construction

`build_messages(sample, doc_type, data_dir, request=..., include_target=...)` returns:

1. `system`: `render_system_prompt(doc_type, request)`: the instructions, the JSON structure
   of the requested fields, one line per field (path, description, examples), and the
   shared output rules (copy as printed, `null` / `[]` when absent, JSON only)
2. `user`: the document only, as one image part (absolute path) or one text part (spatial text)
3. `assistant` (training only): `serialize_target(project(target, request.paths))`: the
   target restricted to the requested fields, in request order; decoys are `null`

A **request** (`field_requests.py`) is the list of fields the prompt asks for:

| Where | Request | Config |
|-------|---------|--------|
| HF inference, vLLM, evaluation | Fixed per doc type: every extract field (`full`), all but the holdout (`seen`), or the holdout only (`held_out`), plus the first N decoys | `fields.inference`, `fields.decoys` |
| Training (SFT, GRPO) | Every annotated field minus the holdout, in registry order; with `fields.sampling`, a random subset per draw (size uniform in [`min_fields`, all]; `full_rate` of draws keep every field), shuffled, sometimes with decoys | `fields.holdout`, `fields.sampling` |
| Calibration | The inference request, or sampled as in training | `quantization.calibration_requests` |

The inference request depends only on the doc type and the config, so its system prompt is
byte-identical across requests and vLLM prefix caching can share it. Sampled training
requests teach the model to read the field list, so adding or removing a field later is a
registry change instead of a new output format (decided 2026-11-26; E20 measures it).
SFT seeds each draw with (seed, sample id, draw number), so runs are reproducible.

`to_openai_messages` converts this to the OpenAI chat format for vLLM (images become
base64 data URLs). Training (`hf/sft_data.py`), HF inference, quantization calibration
and vLLM all go through these two functions, so the model sees the same input everywhere.
In training, only the answer tokens (and the end-of-turn after them) have labels.

## 4. Predictions

`artifacts/predictions/<run_id>/<split>.jsonl`, one line per sample:

```json
{"id": "receipt-001-img", "raw_output": "{\"store\": ...}", "error": null,
 "fields": ["store.name", "items[].price", "total"],
 "latency_ms": 812.4, "ttft_ms": 95.1, "prompt_tokens": 24117, "completion_tokens": 186}
```

- `raw_output` is the untouched model text. Parsing happens in evaluation, not in the predictor.
- `fields` is the request the prompt asked for (decoys included). Evaluation scores these,
  so re-scoring after a registry change still scores what was asked. When absent, the
  config's inference request is used.
- `error` is set when the request itself failed (OOM, timeout); the sample then counts as unparsed.
- Every sample of the evaluated split needs a line. A missing id is an error, not a lower score.
- Timing and token fields are optional; fill them when the backend reports them.

## 5. Scoring

For each sample: restrict the target to the requested fields → parse `raw_output` →
normalize prediction and target with the same rules → flatten both to leaf paths → compare.

- **Requested fields only:** the gold answer is the target projected on the request
  (decoys are `null`). A requested field the record does not annotate (§1) is skipped: not
  scored, and a predicted value for it is not an extra field.
- **Parse:** strips a `<think>` block and a Markdown code fence, then takes the outermost
  `{...}`. Anything that is not a JSON object is unparsed.
- **Normalize:** NFC, collapse whitespace, trim, casefold, empty strings → `null`, numbers
  → strings. Then the field-type rule from the schema (`date`, `number`, ... see T-L05).
- **Flatten:** `{"items": [{"price": 1}]}` → `{"items[0].price": "1"}`. Empty objects and
  lists become a `null` leaf. List items are compared **by position** (T-L06 reviews this).
- **Field correct:** the normalized prediction at the same path equals the target. A
  missing path counts as `null`, so a `null` target field is correct when the prediction
  omits it.
- **Extra fields:** non-null predicted leaves whose path is not in the target. They make
  the sample fail exact match but are not in the field-accuracy denominator.
- **Unparsed sample:** every target field is wrong, NED 1.0.

| Metric | Definition |
|--------|-----------|
| Exact match | Share of samples with every target field correct and no extra field |
| Field-level accuracy | Correct target fields / all target fields, over the whole split |
| NED | Mean over target fields of Levenshtein distance / length of the longer string |
| Parse rate | Share of samples whose output parsed to a JSON object |
| Key presence | Share of target leaf paths present in the output: a dropped or renamed key, or a missing list entry, lowers it even when the expected value is null |

Reported overall, by input type, by doc type, and per `(doc type, field)` in
`artifacts/eval/<run_id>/<split>/fields.json`, lowest accuracy first.

**Field groups.** When a request holds held-out fields (`fields.holdout`) or decoys, the
fields are also summarized as `seen`, `held_out` and `decoy`: field accuracy, NED, key
presence, **missed value rate** (expected value, predicted null) and **invented value
rate** (expected null, predicted a value; for decoys, the hallucination rate). The
field-holdout experiment E20 reads its verdict from these groups.

## 6. Metrics JSON

`results/metrics/<run_id>.json` (committed). Each stage (`train`, `merge`, `quantize`,
`predict_hf`, `evaluate`, `benchmark`) calls
`record_metrics(path, experiment_id=..., run_name=..., stage=..., updates={...})`: values
are merged into the file, `None` never overwrites, and the stage's provenance is stamped
under `stages`:

```json
{
  "experiment_id": "E1",
  "run_name": "gemma4_e2b_lora_r16",
  "split": "test",
  "config": "configs/reference/gemma4_e2b_lora.yaml",
  "seed": 42,
  "max_seq_length": 32768,
  "stages": {
    "train":    {"timestamp": "...", "git_commit": "abc1234", "git_dirty": false,
                 "hostname": "gpu-03", "cuda_visible_devices": "1", "gpu_name": "NVIDIA L40", "config": "..."},
    "evaluate": {"...": "..."}
  },
  "n_samples": 400, "n_fields": 9120,
  "exact_match": 0.96, "field_accuracy": 0.991, "ned_mean": 0.004, "parse_rate": 1.0,
  "key_presence": 0.998, "meets_target": true,
  "by_input_type": {"image": {"n": 200, "n_fields": 4560, "exact_match": 0.95, "...": "..."}},
  "by_doc_type":   {"invoice": {"...": "..."}},
  "by_field_group": {"seen": {"n_fields": 8800, "field_accuracy": 0.993, "key_presence": 0.999,
                              "missed_value_rate": 0.004, "invented_value_rate": 0.002, "...": "..."},
                     "held_out": {"...": "..."}, "decoy": {"...": "..."}},
  "train_peak_vram_gb": 41.2, "train_time_s": 15120, "train_steps_per_s": 0.11,
  "trainable_params": 14680064, "n_train_samples": 8000, "n_dropped_over_length": 3,
  "merge_max_abs_logit_diff": 0.03, "merge_identical_output_rate": 1.0,
  "hf_latency_ms_p50": 5300, "hf_latency_ms_p95": 9100, "hf_tokens_per_s": 38.2,
  "infer_peak_vram_gb": 14.8, "kv_cache_gb": null,
  "ttft_ms_p50": null, "ttft_ms_p95": null, "tpot_ms_p50": null,
  "latency_ms_p50": null, "latency_ms_p95": null, "tokens_per_s": null, "requests_per_s": null,
  "by_concurrency": {"60": {"n_requests": 300, "n_errors": 0, "latency_ms_p95": null,
                            "kv_cache_usage_max": null, "prefix_cache_hit_rate": null, "...": null}},
  "checkpoint_size_gb": null, "adapter_size_mb": 56.1,
  "stability_notes": ""
}
```

- `hf_*` come from `predict_hf.py` (one request at a time) and are not comparable with
  vLLM serving numbers. Top-level `ttft_*`, `latency_*`, `tokens_per_s`, `requests_per_s`
  come from `benchmark_api.py` at concurrency 1; the sweep goes in `by_concurrency`.
- `train`, `merge`, `quantize` and `benchmark` write `<run_id>.json`. `predict_hf` and
  `evaluate` write the file of the split they ran on: `<run_id>.json` for test,
  `<run_id>__validation.json` for validation, so tuning runs keep their scores apart from
  the final test score.
- Unknown fields are rejected; add new ones to `RunMetrics` in a PR.

## 7. Run paths

`run_id = <experiment_id>_<run_name>`. `RunPaths.for_config(config)` gives:

| Path | Committed | Content |
|------|:---------:|---------|
| `results/metrics/<run_id>.json` | yes | Metrics JSON |
| `results/logs/<run_id>/` | yes | `train_log.jsonl`, `benchmark_c<N>.jsonl` (timings only), `nvidia-smi` snapshots |
| `artifacts/predictions/<run_id>/<split>.jsonl` | no | Predictions (private data) |
| `artifacts/eval/<run_id>/<split>/` | no | `samples.jsonl` per-sample scoring, `fields.json` |
| `artifacts/checkpoints/<run_id>/` | no | Intermediate trainer checkpoints |
| `artifacts/adapters/<run_id>/` | no | Final LoRA adapter, processor, copy of the config |
| `artifacts/merged/<run_id>/` | no | Merged checkpoint |
| `artifacts/quantized/<run_id>/` | no | Quantized checkpoint |

`artifacts/` can be moved with `LLMOPT_ARTIFACTS_DIR`.

## 8. Configs and model sources

A config may start with `extends: <path>` (relative to the file). Mappings are merged
recursively; lists and scalars replace the parent's value. Experiments extend the
reference config and override only the variable they test:

```yaml
# configs/gemma4_e2b/qlora_nf4.yaml (E2)
extends: ../reference/gemma4_e2b_lora.yaml
experiment_id: E2
run_name: gemma4_e2b_qlora_r16
method: {name: qlora, quantization: nf4, double_quant: true}
```

The `fields:` section chooses the requests of §3 (defaults shown):

```yaml
fields:
  holdout: {}          # doc type -> fields or decoys training never asks for (E20)
  inference: full      # full | seen | held_out: what inference and evaluation ask for
  decoys: 0            # decoys added to inference requests (the doc type's first N)
  sampling: null       # null: one fixed training request; {} for the defaults below
  # sampling: {full_rate: 0.4, min_fields: 1, decoy_rate: 0.3, max_decoys: 2, shuffle: true}
```

`quantization.source`, `serving.source` and `predict_hf.py --source` name what to load:

| Source | Loads |
|--------|-------|
| `base` | `model.id` |
| `adapter` | `model.id` + this run's adapter (HF only) |
| `merged`, `quantized` | this run's merged / quantized checkpoint |
| `merged:<run_id>`, `quantized:<run_id>`, `adapter:<run_id>` | another run's artifact, e.g. E4 quantizes `merged:<E3 run id>` |
| anything else | a local path or hub id, as is |

`rl.init_adapter` takes `adapter:<run_id>` only: an RL run continues that SFT adapter, so
its own adapter is again relative to `model.id` and every source above works for it.

To evaluate one adapter under several requests (E20's conditions), give each request its
own config with its own `run_name` and load the trained run's adapter:
`predict_hf.py --config <condition config> --source adapter:<trained run id>`.

A vLLM server serves the model under the run id, so start it (`serve_vllm.py`) and query it
(`predict_vllm.py`, `benchmark_api.py`) with the same config.
