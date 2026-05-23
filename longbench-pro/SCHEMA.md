# JSON schema reference

Field-by-field schema for every JSON/JSONL file in this benchmark, intended for downstream agents and tools.

## File layout

```
longbench-pro/results/<quant>/
├── inference.jsonl                 # raw per-item rows (one JSON object per line)
├── summary_official.json           # Pro schema (matches main.py --only_eval output)
├── summary_derived.json            # our extended analytics (perfect counts, throughput, etc.)
└── slices/<slice_name>.json        # filtered subset summaries (e.g. english_le32k)
```

Naming convention for `<quant>`:

- `q8_k_xl`, `q4_k_m`, `iq2_m` — **Unsloth UD** quants (dynamic imatrix)
- `bartowski-q8_0`, `bartowski-iq4_xs`, `bartowski-iq2_m` — **Bartowski** quants (standard imatrix)

## `inference.jsonl` — one row per dataset item

```json
{
  "bon_idx": 1,
  "id": "401c9ee31d21dabf734bc2f48d13a4ebe30368041a84cdb460c964f9228120c3",
  "language": "English" | "Chinese",
  "token_length": "8k" | "16k" | "32k" | "64k" | "128k" | "256k",
  "primary_task": "T1. Retrieval & Ranking",
  "secondary_task": "T1.1 Global Cohesive Retrieval",
  "contextual_requirement": "Full" | "Partial",
  "question_nonthinking": "Please rearrange ...",
  "question_thinking": "Please reason ... <think>",
  "answer": ["1970", "2015"],
  "difficulty": "Easy" | "Moderate" | "Hard" | "Extreme",
  "context": "<first 512 chars of original context, truncated for repo size>",
  "prediction": "[Answer]\n1970\n2015",
  "thinking": "<reasoning content the model produced before the final answer>",
  "metrics": {
    "input_tokens": 67678,
    "output_tokens": 8290,
    "ttft_s": 13.85,
    "total_time_s": 81.0,
    "decode_time_s": 67.15,
    "prefill_tps": 4885.0,
    "decode_tps": 123.5
  }
}
```

### Field semantics

| Field                             | Notes                                                                                                                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                              | Stable across runs (matches the upstream Pro dataset id). Use this to match items across quants.                                                                                                  |
| `bon_idx`                         | Best-of-N iteration index. We ran `--bon_num 1`, so always `1`.                                                                                                                                   |
| `language`                        | One of `English`, `Chinese`.                                                                                                                                                                      |
| `token_length`                    | Bucketed by **Qwen tokenizer** (not the model's tokenizer).                                                                                                                                       |
| `primary_task` / `secondary_task` | 11 primary categories, 25 secondary tasks.                                                                                                                                                        |
| `contextual_requirement`          | `Full` = entire context needed; `Partial` = only a span.                                                                                                                                          |
| `question_nonthinking`            | The prompt we sent (we ran in non-thinking mode for all runs).                                                                                                                                    |
| `answer`                          | **List of strings**, not a single string. The scoring metric for each task type interprets this list differently.                                                                                 |
| `difficulty`                      | Pro's curated rating; `Extreme` = hardest.                                                                                                                                                        |
| `context`                         | **Truncated to 512 chars** in our committed JSONLs to keep repo size down. The model received the full context at inference time. Reconstruct from upstream Pro dataset if you need full context. |
| `prediction`                      | Raw model output (post-thinking, the `content` part of the stream). Empty string = inference failed all retries.                                                                                  |
| `thinking`                        | Raw `reasoning_content` from the stream. Pro's official scorer does NOT use this; it's preserved for analysis.                                                                                    |
| `metrics`                         | Our additions — not in the official Pro pipeline. See breakdown below.                                                                                                                            |

### `metrics` sub-object

All values measured client-side via OpenAI streaming. Times in seconds, throughputs in tokens-per-second.

| Field           | Definition                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_tokens`  | From the `usage` field in the final stream chunk (server-side count, Gemma tokenizer).                                                                |
| `output_tokens` | Total tokens generated, includes both `content` and `reasoning_content`.                                                                              |
| `ttft_s`        | Wall-clock from request start to first non-empty `content` or `reasoning_content` delta.                                                              |
| `total_time_s`  | Wall-clock from request start to end-of-stream.                                                                                                       |
| `decode_time_s` | `total_time_s − ttft_s`.                                                                                                                              |
| `prefill_tps`   | `input_tokens / ttft_s`. Reflects server prefill throughput.                                                                                          |
| `decode_tps`    | `output_tokens / decode_time_s`. **Note:** under `n_proc=4` contention this is per-stream, not aggregate. Multiply by ~4 for total decode throughput. |

Throughput fields may be `null` if the request failed or returned no tokens.

### How predictions are scored

The Pro scorer looks for `[Answer]` (English) or `[答案]` (Chinese) markers in the prediction and extracts only what follows the **last** occurrence. If no marker is present, the entire prediction text is treated as the answer (usually scores 0 on non-trivial tasks).

Per-task metric assignments (from Pro's `modules/utils.py`):

| Secondary task prefix   | Metric                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| T1.\*                   | `NDCG` (ranking quality)                                                                      |
| T2.\*, T6.3             | `Pairwise_Accuracy` (correct relative ordering)                                               |
| T3._, T11._             | `Accuracy` (exact-match on the first answer item, normalized)                                 |
| T4.\*                   | `Summary` = 0.5 × `Max_Rouge_L` + 0.5 × `Max_Semantic_Similarity` (Qwen3-Embedding-8B cosine) |
| T5._, T6.2, T7._, T9.\* | `F1_Score` (token-set F1 against answer list)                                                 |
| T6.1, T8._, T10._       | `SubEM` (sub-string exact-match: fraction of answer items appearing in the prediction)        |

All metric implementations are in [`scripts/_pro_metrics.py`](scripts/_pro_metrics.py) (vendored verbatim from Pro's `modules/utils.py`).

---

## `summary_official.json` — Pro schema (matches `main.py --only_eval` output)

This is the official Pro reporting format. Other agents writing leaderboard tooling should consume this file.

```json
{
  "date": "2026-05-22",
  "total_questions_num": 1500,
  "inference_iterations": 1,
  "total_samples_num": 1500,
  "fail_samples_num": 107,
  "inference_inconsistent_samples_num": 0,
  "average_overall_metric": 0.6091,
  "inference_iteration_1_overall_metric": 0.6091,

  "average_token_length_metric": {
    "8k": 0.731, "16k": 0.667, "32k": 0.652,
    "64k": 0.593, "128k": 0.553, "256k": 0.459
  },
  "average_contextual_requirement_metric": {
    "Full": 0.526, "Partial": 0.664
  },
  "average_difficulty_metric": {
    "Easy": 0.796, "Moderate": 0.654, "Hard": 0.553, "Extreme": 0.336
  },
  "average_primary_task_metric": {
    "T1. Retrieval & Ranking": 0.789,
    "T2. Sequencing & Structure Reconstruction": 0.784,
    "T3. Evidence-Grounded QA": 0.542,
    "T4. Summarization & Synthesis": 0.535,
    "T5. Attribution & Citation Alignment": 0.758,
    "T6. Aggregation & Clustering": 0.503,
    "T7. Consistency & Compliance Checking": 0.445,
    "T8. Structured & Numeric Reasoning": 0.660,
    "T9. Version & Code Diff Analysis": 0.773,
    "T10. Rule Induction & In-Context Learning": 0.596,
    "T11. Dialogue Memory & Long-Horizon Tracking": 0.425
  },
  "average_language_metric": {
    "Chinese": 0.625, "English": 0.593
  },

  "BoN-1": {
    "overall_metric": 0.6091,
    "token_length": { ... }, "contextual_requirement": { ... },
    "difficulty": { ... }, "primary_task": { ... }, "language": { ... }
  },
  "pass@1": 0.401,

  "_caveats": {
    "T4_summarization_metric": "Full official Summary metric used: 0.5 * Max_Rouge_L + 0.5 * Max_Semantic_Similarity (Qwen3-Embedding-8B)."
  }
}
```

### Field semantics

| Field                                | Notes                                                                                                                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `date`                               | When the summary was generated (not when the run finished).                                                                                                           |
| `total_questions_num`                | Number of unique question IDs. With `bon_num=1` this equals `total_samples_num`.                                                                                      |
| `total_samples_num`                  | Number of inference rows (= `total_questions_num × bon_num`).                                                                                                         |
| `fail_samples_num`                   | Rows with empty `prediction`. These count as score 0.0 in every aggregate.                                                                                            |
| `inference_inconsistent_samples_num` | Number of questions where the BoN iteration count didn't match `bon_num`. Should always be 0 for `bon_num=1`.                                                         |
| `average_overall_metric`             | Mean of `metric` across all rows (zeros included for failed predictions). **Headline number.**                                                                        |
| `average_<dimension>_metric`         | Mean grouped by dimension. Sort keys preserve canonical order (length: 8k→256k, difficulty: Easy→Extreme, etc.). `null` if no rows of that value.                     |
| `BoN-i`                              | Best-of-N for the first `i` iterations. With `bon_num=1`, `BoN-1` equals the average results.                                                                         |
| `pass@i`                             | Fraction of items where at least one of the first `i` BoN iterations was "perfect" — `metric == 1.0` for non-T4 tasks, `metric > 0.65` for T4 tasks (Pro convention). |
| `_caveats`                           | Our addition (not in upstream Pro) — currently notes whether T4 used the full Summary metric.                                                                         |

---

## `summary_derived.json` — our extended analytics

Strictly a superset of what's in `summary_official.json`. Includes:

- Per-secondary-task breakdowns (25 tasks, not just 11 primaries).
- Per-metric-type breakdowns (NDCG / F1_Score / Accuracy / SubEM / Pairwise_Accuracy / Summary).
- Perfect / zero / partial counts at every level.
- Full throughput stats: `input_tokens`, `output_tokens`, `ttft_s`, `total_time_s`, `prefill_tps`, `decode_tps` (p10 / p25 / median / p75 / p90 / mean / min / max).
- Total token volume processed (across all rows with metrics).

Use this for deeper analysis. The `summary_official.json` is the file to ship to leaderboard tooling.

---

## `slices/<name>.json` — filtered subset summaries

Generated by [`scripts/compute_slice.py`](scripts/compute_slice.py). Same schema as `summary_derived.json`'s rich breakdowns but only over the filtered subset.

Currently shipped:

- `slices/english_le32k.json` — English-only, ≤32K tokens (375 items: 125 each at 8K/16K/32K).

Generate your own with:

```bash
python scripts/compute_slice.py results/q8_k_xl/inference.jsonl \
  --language English --max-length 32k \
  --save-json results/q8_k_xl/slices/english_le32k.json
```

Filter args: `--language`, `--difficulty`, `--token-length` (repeatable), `--min-length`, `--max-length`, `--primary-task` (e.g. `T4` for all T4.\* subtasks), `--secondary-task`, `--contextual-requirement`.

---

## Common queries (Python)

### Compare two quants on the same items

```python
import json, statistics
q8 = {r['id']: r for r in (json.loads(l) for l in open('results/q8_k_xl/inference.jsonl'))}
q4 = {r['id']: r for r in (json.loads(l) for l in open('results/q4_k_m/inference.jsonl'))}
shared = set(q8) & set(q4)
# Re-score via scripts._pro_metrics for apples-to-apples; or use the inference rows' metric field
# if already scored elsewhere.
```

### Filter by task and dimension

```python
import json
rows = [json.loads(l) for l in open('results/q8_k_xl/inference.jsonl')]
t1_long = [r for r in rows if r['primary_task'].startswith('T1.') and r['token_length'] in ('128k', '256k')]
print(f"{len(t1_long)} T1 long-context items")
```

### Pull a single quant's headline numbers

```python
import json
s = json.load(open('results/q8_k_xl/summary_official.json'))
print(f"overall: {s['average_overall_metric']:.3f}")
print(f"pass@1:  {s['pass@1']:.3f}")
print(f"256K:    {s['average_token_length_metric']['256k']:.3f}")
print(f"T4 Summary: {s['average_primary_task_metric']['T4. Summarization & Synthesis']:.3f}")
```

### Score a fresh inference.jsonl

```python
import sys
sys.path.insert(0, 'longbench-pro/scripts')
from build_summary import score_one  # uses _pro_metrics internally
score = score_one(
    secondary_task="T1.1 Global Cohesive Retrieval",
    answer=["1970", "2015"],
    prediction="[Answer]\n1970\n2015",
    is_zh=False,
)
```

### Build the cross-quant dashboard

```bash
python longbench-pro/scripts/build_dashboard.py
# writes dashboard.html and dashboard.csv at the repo root
```

`dashboard.csv` is in long format — one row per (quant × dimension × value):

```
quant_dir,quant,size_gb,total_samples,fail_samples,dimension,value,score
q8_k_xl,Q8_K_XL (Unsloth UD),27.0,1500,107,overall,all,0.6091
q8_k_xl,Q8_K_XL (Unsloth UD),27.0,1500,107,primary_task,T1. Retrieval & Ranking,0.789
q8_k_xl,Q8_K_XL (Unsloth UD),27.0,1500,107,token_length,8k,0.731
...
```

Convenient for pivoting in Excel/Sheets: rows = quants, columns = dimension values, values = score.
