# JSON schema reference — CLINC150

Field-by-field schema for the files this benchmark produces, for downstream tools.

## File layout

```
results/<quant>/
├── inference.jsonl   # raw per-item rows (one JSON object per line)
└── summary.json      # aggregate metrics
```

`<quant>` ∈ `bartowski-q8_0`, `bartowski-iq4_xs`, `bartowski-iq2_m`.

## `inference.jsonl` — one row per item

```json
{
  "id": 1138,
  "text": "pay $175 on my visa",
  "true_intent": "pay_bill",
  "true_is_oos": false,
  "prediction": "pay_bill",
  "pred_is_oos": false,
  "correct": true,
  "raw_output": "pay_bill",
  "thinking": "*   Task: Intent classification ... *   Label: `pay_bill`.",
  "metrics": {
    "input_tokens": 689,
    "output_tokens": 914,
    "thinking_chars": 3160,
    "cached_tokens": 672,
    "latency_s": 9.83,
    "finish_reason": "stop"
  }
}
```

| Field         | Notes                                                                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`          | Index into the `clinc_oos/plus/test` split. Stable across runs; use to match items across quants.                                                            |
| `text`        | The utterance sent to the model.                                                                                                                             |
| `true_intent` | Gold label — one of the 151 `ClassLabel` names (incl. `oos`).                                                                                                |
| `true_is_oos` | `true_intent == "oos"`.                                                                                                                                      |
| `prediction`  | Label parsed from the model's final answer. `""` if the request failed all retries or the answer was unparseable (counts as wrong; retried on the next run). |
| `pred_is_oos` | `prediction == "oos"`.                                                                                                                                       |
| `correct`     | `prediction == true_intent` (exact match). Predicting `oos` for an in-scope item, or an intent for an OOS item, is wrong.                                    |
| `raw_output`  | The raw final `content` before label matching. Normally identical to `prediction`.                                                                           |
| `thinking`    | Full `reasoning_content` (the chain-of-thought). Empty if the answer was truncated before the model closed its reasoning.                                    |
| `metrics`     | See below.                                                                                                                                                   |

### `metrics`

| Field            | Definition                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `input_tokens`   | Prompt tokens (server-side count, Gemma tokenizer).                                                                |
| `output_tokens`  | Generated tokens — includes the reasoning trace.                                                                   |
| `thinking_chars` | Character length of the stored `thinking` text.                                                                    |
| `cached_tokens`  | Prompt tokens served from llama.cpp's prefix cache.                                                                |
| `latency_s`      | Client-side wall-clock for the request.                                                                            |
| `finish_reason`  | `stop` (model finished) or `length` (hit `max_tokens` — a truncated reasoning trace, usually an empty prediction). |

## `summary.json`

```json
{
  "model_id": "bartowski-q8_0",
  "dataset": "clinc_oos/plus/test",
  "n_total": 5500,
  "n_in_scope": 4500,
  "n_oos": 1000,
  "n_errors": 0,
  "n_truncated": 0,
  "in_scope_accuracy": 0.0,
  "overall_accuracy": 0.0,
  "oos_recall": 0.0,
  "oos_precision": 0.0,
  "oos_f1": 0.0,
  "macro_f1": 0.0,
  "oos_confusion": { "tp": 0, "fn": 0, "fp_in_scope_called_oos": 0 },
  "throughput": {
    "n_latency_samples": 5500,
    "latency_s_mean": 0.0,
    "latency_s_median": 0.0,
    "input_tokens_mean": 0.0,
    "output_tokens_mean": 0.0,
    "output_tokens_max": 0,
    "thinking_tokens_mean": 0.0,
    "thinking_tokens_median": 0.0,
    "thinking_tokens_max": 0,
    "cached_tokens_mean": 0.0,
    "decode_tps_per_stream_median": 0.0,
    "decode_tps_per_stream_mean": 0.0,
    "gen_tps_aggregate_est": 0.0,
    "concurrency_assumed": 4
  },
  "per_intent": {
    "pay_bill": { "n": 30, "correct": 28, "accuracy": 0.933 },
    "...": {}
  }
}
```

| Field               | Definition                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `n_errors`          | Rows with an empty prediction (failed/unparseable).                                               |
| `n_truncated`       | Rows with `finish_reason == "length"`.                                                            |
| `in_scope_accuracy` | Correct ÷ 4,500 in-scope items.                                                                   |
| `overall_accuracy`  | Correct ÷ all `n_total`.                                                                          |
| `oos_recall`        | True-OOS predicted `oos` ÷ 1,000.                                                                 |
| `oos_precision`     | True-OOS predicted `oos` ÷ all predicted `oos`.                                                   |
| `oos_confusion`     | `tp` = OOS called OOS; `fn` = OOS missed; `fp_in_scope_called_oos` = in-scope wrongly called OOS. |
| `macro_f1`          | Unweighted mean per-label F1 over all 151 labels.                                                 |
| `per_intent`        | `{label: {n, correct, accuracy}}` for every gold label present.                                   |

### `throughput`

| Field                                       | Definition                                                                                                                                                                                       |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `latency_s_mean` / `_median`                | Per-request wall-clock latency.                                                                                                                                                                  |
| `input_tokens_mean`                         | Mean prompt tokens (~688; ~97% prefix-cached).                                                                                                                                                   |
| `output_tokens_mean` / `_max`               | Mean / max generated tokens (reasoning trace + final answer).                                                                                                                                    |
| `thinking_tokens_mean` / `_median` / `_max` | Reasoning tokens per request, derived as `output_tokens − answer_tokens`. The answer is a short label (~3 tokens), so this ≈ `output_tokens`. llama.cpp returns no native reasoning-token split. |
| `cached_tokens_mean`                        | Mean prompt tokens served from the prefix cache.                                                                                                                                                 |
| `decode_tps_per_stream_median` / `_mean`    | Per-request generation speed = `output_tokens / latency_s` (latency is ~all decode).                                                                                                             |
| `gen_tps_aggregate_est`                     | `concurrency × Σoutput_tokens / Σlatency_s` — aggregate generation throughput (assumes all streams stay busy; an upper-ish estimate).                                                            |
| `concurrency_assumed`                       | Client concurrency used for the aggregate estimate (the run uses 4, matching `-np 4`).                                                                                                           |

Rows are deduplicated by `id` (last non-empty prediction wins) before scoring, so
resumed/retried runs score cleanly.
