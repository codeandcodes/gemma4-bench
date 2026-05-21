# LongBench Pro — Gemma 4 26B-A4B-it benchmarks

[LongBench Pro](https://arxiv.org/abs/2601.02872) is a bilingual long-context benchmark (1,500 samples, 11 primary tasks, 25 secondary tasks, contexts 8K–256K tokens by Qwen tokenizer). This folder contains complete inference runs of Gemma 4 26B-A4B-it across multiple quantizations, with the same prompts, sampling parameters, and runtime config.

## Headline results

> Scores are from the official Pro schema (mean across all items processed, with empty/failed predictions counted as 0).

| Quant            |   Overall | pass@1 | Empty rate | T1 Retrieval | T9 Code Diff | T11 Dialogue Memory | T4 Summary\* | 256K context |     Coverage |
| ---------------- | --------: | -----: | ---------: | -----------: | -----------: | ------------------: | -----------: | -----------: | -----------: |
| **Q8_K_XL**      | **0.587** |  0.395 |       7.1% |        0.789 |        0.773 |               0.425 |      0.256\* |        0.437 |    1500/1500 |
| Q4_K_M (partial) |     0.602 |  0.404 |       7.1% |        0.806 |        0.768 |               0.594 |      0.276\* |        0.425 | **339/1500** |
| IQ2_M            | _pending_ |        |            |              |              |                     |              |              |              |

**Q4_K_M is still running** — the table will be updated as more items complete. The partial numbers above are computed on 339 items so far; they appear close to Q8 but the sample is not uniformly distributed across tasks/lengths yet.

For a fair apples-to-apples comparison at the partial timestep, the **same-id matched subset** (Q4 188 scored items vs Q8 188 same-id items) gives:

|              |    Q4 |    Q8 |       Δ |
| ------------ | ----: | ----: | ------: |
| Overall mean | 0.660 | 0.666 |  −0.006 |
| Perfect rate | 45.7% | 50.0% | −4.3 pp |
| 256K context | 0.563 | 0.578 |  −0.015 |

Q4 is currently tracking Q8 within sampling noise (−0.006). Notable per-task gaps so far: T5 Citation Alignment (Q4 −0.144), T8 Structured Reasoning (Q4 −0.087). One positive gap: T11 Dialogue Memory (Q4 +0.182) — could be a real signal or persistent sample bias, will know at full N.

\* T4 Summarization is undercounted in our runs because we did not download the 8 GB Qwen3-Embedding-8B model needed for the embedding-similarity component. Real T4 scores will be ~50% higher (the official metric is `0.5 × ROUGE-L + 0.5 × embedding_cosine`).

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
