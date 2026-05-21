# Modifications to caskcsg/longcontext/LongBench-Pro

The runs in this folder used the [official Pro inference harness](https://github.com/caskcsg/longcontext/tree/main/LongBench-Pro) with two small modifications. Both are additive — predictions and scoring are unchanged from the upstream methodology.

## 1. Streaming + per-request metrics — `modules/model_manager.py`

**What changed:** The `ModelManagerOpenAI.query()` method was rewritten to use OpenAI streaming with `stream_options={"include_usage": true}` instead of the original non-streaming call. The decorator and base class signatures were updated to return a 3-tuple `(answer, thinking, metrics)` instead of `(answer, thinking)`.

**Why:** Capture per-request throughput data without any extra server round-trips. Specifically:

- `input_tokens`, `output_tokens` — from the final `usage` chunk in the stream
- `ttft_s` — wall-clock from request start to the first non-empty `content` or `reasoning_content` delta
- `total_time_s` — wall-clock from request start to end-of-stream
- `decode_time_s` — `total_time_s − ttft_s`
- `prefill_tps` — `input_tokens / ttft_s`
- `decode_tps` — `output_tokens / decode_time_s`

The streaming loop accumulates both `content` and `reasoning_content` deltas separately so thinking models (Gemma 4) work correctly — the official Pro harness only read `message.content` from the final response object, which returns empty content for thinking models since the actual answer comes through `reasoning_content` first and `content` after.

**Failure mode preserved:** the decorator's retry loop still wraps the call, and a fully failed request returns `("", None, {})` — `process_single_item` writes that to disk so the cache layer can identify and retry it on the next launch (the `data_loader` filters rows with empty predictions).

## 2. Persist metrics on each row — `modules/inference.py`

**What changed:** `InferenceEngine.process_single_item()` was updated to unpack the 3-tuple from `model_manager.query()` and add `item['metrics'] = metrics` before writing the row.

Also swapped `import torch.multiprocessing as mp` → `import multiprocessing as mp`. The `mp.Process`/`mp.Lock` API is identical for our purposes, and dropping the torch import removes a multi-GB CUDA dependency from the inference-only path (we are not running the model locally — we hit an OpenAI-compatible llama-server).

## Files

Diff against `caskcsg/longcontext/LongBench-Pro` (as of the date the runs in this folder were performed):

- `modules/model_manager.py`: ~50 lines changed in `model_query_decorator` (return signature) and `ModelManagerOpenAI.query` (streaming + metric capture)
- `modules/inference.py`: 1 import swap + 1 line added to write `metrics` on each row

The exact patches are available on request; both are mechanical and isolated.

## Compatibility with `--only_eval`

The official evaluation pipeline reads each row's `prediction`, `secondary_task`, `answer`, `language`, and `bon_idx`. None of those fields were touched. The extra `metrics` and `thinking` fields are ignored by the official evaluator and don't affect scoring.

Our `scripts/build_summary.py` in this repo can also generate the official summary JSON without running the Pro Evaluator (useful when the 8 GB Qwen3-Embedding-8B model isn't available — see the T4 Summarization caveat in the top-level [README](../README.md)).
