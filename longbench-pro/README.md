# LongBench Pro — Gemma 4 26B-A4B-it benchmarks

[LongBench Pro](https://arxiv.org/abs/2601.02872) is a bilingual long-context benchmark (1,500 samples, 11 primary tasks, 25 secondary tasks, contexts 8K–256K tokens by Qwen tokenizer). This folder contains complete inference runs of Gemma 4 26B-A4B-it across multiple quantizations, with the same prompts, sampling parameters, and runtime config.

## Headline results

> Scores are from the official Pro schema (mean across all items processed, with empty/failed predictions counted as 0). T4 Summarization uses the full official metric `0.5 × ROUGE-L + 0.5 × embedding_cosine` (Qwen3-Embedding-8B).

| Quant                                |                                     Overall | pass@1 | Empty rate | T1 Retrieval | T9 Code Diff | T11 Dialogue Memory | T4 Summary | 256K context |      Coverage |
| ------------------------------------ | ------------------------------------------: | -----: | ---------: | -----------: | -----------: | ------------------: | ---------: | -----------: | ------------: |
| **Q8_K_XL**                          |                                   **0.609** |  0.401 |       7.1% |        0.789 |        0.773 |               0.425 |      0.535 |        0.459 |     1500/1500 |
| Q4_K_M (partial)                     |                                       0.592 |  0.389 |       8.4% |        0.725 |        0.730 |               0.470 |      0.539 |        0.411 | **1000/1500** |
| IQ2_M (planned, English ≤64K subset) | _scheduled to auto-launch when Q4 finishes_ |        |            |              |              |                     |            |              |               |

**Q4_K_M is still running** — at 1000/1500 (66.7%). For an apples-to-apples comparison at the current timestep, the **same-id matched subset** (Q4 1000 scored items vs Q8 1000 same-id items) gives:

|              |    Q4 |    Q8 |          Δ |
| ------------ | ----: | ----: | ---------: |
| Overall mean | 0.592 | 0.610 | **−0.018** |
| Perfect rate | 38.0% | 40.0% |    −2.0 pp |
| 256K context | 0.411 | 0.412 |     −0.001 |

At N=1000 Q4 sits ~1.8 percentage points below Q8 overall — a small but consistent gap (was −0.007 at N=504, so it's grown slightly with more data). Per-length breakdown shows Q4 underperformance concentrated in the 8K–64K mid-range (Δ ≈ −0.03 to −0.04), while at the extreme 256K bucket both quants struggle equally (Δ ≈ 0). The earlier "Q4 is indistinguishable from Q8" finding at small N was somewhat optimistic; the converged answer is "Q4 is ~3% worse than Q8" — still a strong showing for half the disk and VRAM, but not literally free.

## Subset analyses

The full headline number averages over a brutal distribution (8K–256K context, four difficulty tiers including "Extreme"). For a sense of how the model performs on more typical workloads — English text at moderate context — slice the dataset to **English only, ≤32K tokens** (375 items, evenly split across 8K/16K/32K):

| Slice                     |       n |      Mean |    pass@1 | Perfect |
| ------------------------- | ------: | --------: | --------: | ------: |
| Full Q8 (all 1500)        |    1500 |     0.609 |     0.401 |       — |
| **Q8 English-only, ≤32K** | **375** | **0.681** | **0.472** |   46.1% |

The +0.072 lift over the full benchmark comes almost entirely from removing long-context items — the model is actually slightly _better_ in Chinese on the full set (0.625 Chinese vs 0.593 English), so Chinese is not the drag. The killer for the headline number is long context.

Sliced by primary task:

| Task                    |   n |                             Mean |
| ----------------------- | --: | -------------------------------: |
| T5 Citation Alignment   |  30 |                        **0.856** |
| T9 Code Diff            |  30 |                            0.854 |
| T2 Sequencing           |  30 |                            0.809 |
| T1 Retrieval            |  30 |                            0.799 |
| T8 Structured Reasoning |  45 |                            0.789 |
| T10 Rule Induction      |  30 |                            0.680 |
| T6 Aggregation          |  45 |                            0.637 |
| T7 Consistency          |  45 |                            0.612 |
| T4 Summarization        |  30 |                            0.553 |
| T3 Evidence QA          |  30 |                            0.500 |
| T11 Dialogue Memory     |  30 | **0.400** ← persistent weak spot |

Sliced by length (still monotonic decay within the 8K–32K range):

- 8K: 0.731
- 16K: 0.683
- 32K: 0.628

Suggests an effective context length closer to ~16K than the marketed 256K for dense reasoning tasks. Full slice JSON: [`results/q8_k_xl/slices/english_le32k.json`](results/q8_k_xl/slices/english_le32k.json).

You can generate other slices with [`scripts/compute_slice.py`](scripts/compute_slice.py):

```bash
# English-only, easy difficulty
python scripts/compute_slice.py results/q8_k_xl/inference.jsonl --language English --difficulty Easy

# T4 Summarization tasks only, all languages, 8K and 16K
python scripts/compute_slice.py results/q8_k_xl/inference.jsonl --primary-task T4 --token-length 8k --token-length 16k
```

## Run methodology

All runs use the same setup so comparison is apples-to-apples:

```bash
python main.py --only_infer \
  --model_manager openai \
  --model_name <ALIAS> \
  --tokenizer_path model/Tokenizers/qwen \
  --context_max_length 170000 \
  --url http://127.0.0.1:8080/v1 \
  --api_key dummy \
  --temperature 1.0 \
  --max_new_tokens 32768 \
  --timeout 1800 \
  --max_tries 5 \
  --time_sleep 0.0 \
  --bon_num 1 \
  --n_proc 4
```

### llama.cpp / llama-server flags (per quant)

```
--ctx-size 1048576    # 4 × 262K → each parallel slot gets full 262K context
-np 4                 # 4 parallel server slots
--temp 1.0            # Gemma 4 recommended
--top-p 0.95          # Gemma 4 recommended
--top-k 64            # Gemma 4 recommended
-ub 4096              # large ubatch for prefill throughput
--jinja
--spec-type ngram-cache    # cheap n-gram speculative decoding (~10-15% decode gain on quote-heavy tasks)
```

### Why `context_max_length=170000` (Qwen tokens)?

Pro's pipeline tokenizes prompts with the Qwen tokenizer (bundled in-repo) and truncates to fit the model's context window. The server uses the Gemma tokenizer, which produces **up to 1.3× more tokens** than Qwen for some content (especially Chinese — Gemma's vocab is more English-biased). 170K Qwen tokens × 1.3 ≈ 221K Gemma tokens, well under the 262144 server `--ctx-size`, leaving 32K headroom for the output budget.

### Why `n_proc=4`?

Allows 4 concurrent OpenAI client requests, each routed to a server slot. Aggregate throughput improvement over `n_proc=1` is ~38% for this workload — much less than the naive 4× because decode is memory-bandwidth-bound on MoE-A4B at long context. See [the bandwidth analysis below](#why-batching-only-helps-modestly).

## Why batching only helps modestly

Gemma 4 26B-A4B is an MoE model: 26B total parameters, only ~4B active per token. At 262K context, our measurements show:

|                               |                       batch=1 |                                        batch=4 |
| ----------------------------- | ----------------------------: | ---------------------------------------------: |
| Active weight reads per token | ~4 GB (only 4B of 26B active) | ~7 GB (MoE expert variance bloats this 1.5–2×) |
| KV cache reads per token      |                        ~10 GB |                 ~40 GB (4 streams, no sharing) |
| **Total per step**            |                     **14 GB** |                                      **47 GB** |
| Observed throughput           |                       111 t/s |              32 t/s/stream → 128 t/s aggregate |

KV cache reads (~40 GB at batch=4) dominate over weight reads (~7 GB), so 4 parallel streams just spread the same ~1.5 TB/s of effective memory bandwidth across 4 channels. The ~10–15% real-world speedup comes mostly from prefill being compute-bound (not bandwidth-bound) and parallelizing better.

Decode TPS by input size bucket on Q8_K_XL (per-stream, under 4-way contention) is essentially flat:

| Input size | Prefill TPS | Decode TPS |
| ---------- | ----------: | ---------: |
| <10K       |       1,507 |       31.2 |
| 10–50K     |       2,870 |       29.9 |
| 50–100K    |       3,979 |       28.9 |
| 100–200K   |       3,774 |       27.7 |
| >200K      |       3,551 |       27.1 |

## File layout

```
longbench-pro/
├── README.md              # this file
├── results/
│   └── q8_k_xl/
│       ├── inference.jsonl          # 1,500 per-item rows with prediction, thinking, metrics
│       ├── summary_derived.json     # comprehensive analytics (perfect counts, throughput stats, cross-cuts)
│       └── summary_official.json    # Pro's official schema (matches main.py --only_eval output)
├── scripts/
│   └── build_summary.py             # regenerate the two summary JSONs from any inference.jsonl
└── patches/
    └── longbench-pro-modifications.md   # what we changed in caskcsg/longcontext to add streaming + metrics
```

## Inference JSONL row schema

```json
{
  "bon_idx": 1,
  "id": "...",
  "context": "<first 512 chars of original context>",
  "language": "English" | "Chinese",
  "token_length": "8k" | "16k" | "32k" | "64k" | "128k" | "256k",
  "primary_task": "T1. Retrieval & Ranking",
  "secondary_task": "T1.1 Global Cohesive Retrieval",
  "contextual_requirement": "Full" | "Partial",
  "question_nonthinking": "...",
  "question_thinking": "...",
  "answer": ["expected", "tokens"],
  "difficulty": "Easy" | "Moderate" | "Hard" | "Extreme",
  "prediction": "<final model answer text>",
  "thinking": "<reasoning content the model produced before answering>",
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

The `metrics` object is captured per request via OpenAI streaming with `stream_options={"include_usage": true}`. Empty `prediction` indicates the request failed all retries (a small fraction of rows during a 502 burst at server-load time on Q8_K_XL; future runs use `--max_tries 5`).

## Regenerating the summary JSONs

```bash
python scripts/build_summary.py results/q8_k_xl/inference.jsonl
# writes:
#   results/q8_k_xl/inference_summary_derived.json
#   results/q8_k_xl/inference_summary_official.json
```

## Citations

- LongBench v2 / Pro authors and dataset: [caskcsg/LongBench-Pro](https://huggingface.co/datasets/caskcsg/LongBench-Pro)
- Inference harness: [caskcsg/longcontext/LongBench-Pro](https://github.com/caskcsg/longcontext/tree/main/LongBench-Pro) (with our modifications, see `patches/`)
- Model: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- llama.cpp: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- llama-swap: [mostlygeek/llama-swap](https://github.com/mostlygeek/llama-swap)
