# Data formats and contracts

Every track reads or writes these files. They are implemented in `src/llm_opt_lab/` and
covered by `tests/`. Changing one is a core change: open a PR and ask the lead to review.

| Contract | Written by | Read by | Code |
|----------|-----------|---------|------|
| [Dataset records](#1-dataset-records) | Lead (data tasks) | everyone | `data.py` |
| [Doc types](#2-doc-types) | Lead (data tasks) | everyone | `doc_types.py` |
| [Prompt construction](#3-prompt-construction) | — | training, HF inference, vLLM, calibration | `prompting.py` |
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
| `meta` | no | Anything else (source, annotator, flags). Never used by training or scoring |

Unknown fields are rejected. `scripts/data/validate.py` also checks every target against
its doc type schema and that image files exist. How to write `target` is in the
annotation guideline (T-L02).

## 2. Doc types

Each document type has its own system prompt and output schema:

```text
doc_types/<name>/
├── system_prompt.md   # sent verbatim as the system message (leading/trailing whitespace stripped)
└── schema.json        # JSON Schema of the target
```

The schema uses a subset of JSON Schema: `object` with `properties`, `array` with
`items`, and leaves. A leaf may set `"x-field-type"` to choose its normalizer (default
`text`; today also `date` and `number`, filled in by T-L05). Unknown field types are an
error at evaluation time. Keeping real JSON Schema means vLLM structured outputs can use
the same file (T-C09). `doc_types/cord_receipt/` is the public dev doc type.

## 3. Prompt construction

`build_messages(sample, doc_type, data_dir, include_target=...)` returns:

1. `system`: the doc type's system prompt, byte-identical across requests so vLLM prefix
   caching can share it
2. `user`: the document only, as one image part (absolute path) or one text part (spatial text)
3. `assistant` (training only): `serialize_target(target)`, i.e. `json.dumps(target, ensure_ascii=False)`

`to_openai_messages` converts this to the OpenAI chat format for vLLM (images become
base64 data URLs). Training (`hf/sft_data.py`), HF inference, quantization calibration
and vLLM all go through these two functions, so the model sees the same input everywhere.
In training, only the answer tokens (and the end-of-turn after them) have labels.

## 4. Predictions

`artifacts/predictions/<run_id>/<split>.jsonl`, one line per sample:

```json
{"id": "receipt-001-img", "raw_output": "{\"store\": ...}", "error": null,
 "latency_ms": 812.4, "ttft_ms": 95.1, "prompt_tokens": 24117, "completion_tokens": 186}
```

- `raw_output` is the untouched model text. Parsing happens in evaluation, not in the predictor.
- `error` is set when the request itself failed (OOM, timeout); the sample then counts as unparsed.
- Every sample of the evaluated split needs a line. A missing id is an error, not a lower score.
- Timing and token fields are optional; fill them when the backend reports them.

## 5. Scoring

For each sample: parse `raw_output` → normalize prediction and target with the same
rules → flatten both to leaf paths → compare.

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

Reported overall, by input type, by doc type, and per `(doc type, field)` in
`artifacts/eval/<run_id>/<split>/fields.json`, lowest accuracy first.

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
  "meets_target": true,
  "by_input_type": {"image": {"n": 200, "n_fields": 4560, "exact_match": 0.95, "...": "..."}},
  "by_doc_type":   {"invoice": {"...": "..."}},
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

A vLLM server serves the model under the run id, so start it (`serve_vllm.py`) and query it
(`predict_vllm.py`, `benchmark_api.py`) with the same config.
