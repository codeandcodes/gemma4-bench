# CLINC150 — intent classification

Zero-shot intent classification of **Gemma 4 26B-A4B-it** Bartowski quants on the
[CLINC150](https://aclanthology.org/D19-1131/) `plus` benchmark (the OOS variant),
served by llama.cpp via llama-swap on the same RTX Pro 6000 as the rest of this repo.

## Results

Full test set (5,500 items per quant), thinking mode, with truncated-reasoning items
recovered at an 8,192-token budget.

| Quant      | In-scope acc | OOS recall | OOS F1 | Overall | Macro-F1 | Think tok/req | Aggregate tok/s |
| ---------- | ------------ | ---------- | ------ | ------- | -------- | ------------- | --------------- |
| **Q8_0**   | 91.8%        | 84.4%      | 86.3%  | 90.5%   | 90.4%    | 941           | 260             |
| **IQ4_XS** | 91.5%        | 85.5%      | 86.5%  | 90.4%   | 90.2%    | 1,045         | 306             |
| **IQ2_M**  | 91.6%        | 76.4%      | 82.9%  | 88.8%   | 89.3%    | 989           | 334             |

**In-scope intent accuracy is essentially quant-independent** — 91.5–91.8% across all
three. Even 2-bit holds up for routing in-domain utterances; whatever quantization
degrades, it is not the core 150-way classification.

**Out-of-scope detection is what low-bit costs you.** OOS recall holds at ~84–85% for
Q8_0/IQ4_XS but falls to 76.4% at IQ2_M — an ~9pt drop that pulls IQ2_M's overall
accuracy down to 88.8%. The 2-bit model more often forces a confident in-scope label
instead of abstaining; its OOS _precision_ actually rises (it says `oos` less often, but
is right when it does).

**IQ4_XS is the sweet spot.** It matches Q8_0 on every accuracy metric (and edges it on
OOS recall) while running ~18% faster (306 vs 260 tok/s aggregate). For this task there
is little reason to pay for Q8_0 over IQ4_XS.

**IQ2_M shows broader instruction-following decay** beyond the OOS miss: its final
answers average ~37 tokens instead of ~3 (it wraps the label in prose rather than
emitting it cleanly), and 19 items loop past even an 8,192-token budget (vs 1 each for
the higher quants).

Each classification costs ~940–1,045 thinking tokens, and throughput scales inversely
with precision (260 → 306 → 334 tok/s aggregate — smaller quant, faster decode).

## Task

|         |                                                                               |
| ------- | ----------------------------------------------------------------------------- |
| Dataset | `clinc_oos`, config `plus`, `test` split (HuggingFace)                        |
| Items   | **5,500** = 4,500 in-scope (150 intents × 30) + 1,000 out-of-scope            |
| Labels  | **151** — the 150 intents plus `oos`                                          |
| Prompt  | single user turn: fixed instructions + the 150-label list, then the utterance |

The static instruction+label-list prefix is identical on every request and placed
first, so llama.cpp prefix-caches it (≈672 of ~686 prompt tokens reused per request).

## Method: thinking mode

Gemma 4 is a **thinking model** — by default it reasons (returned in
`reasoning_content`) before emitting the final answer in `content`. These runs use
that native mode:

- **Thinking enabled** (server default). The model reasons (~900 output tokens on
  average; the long tail reaches ~2,900), then states the label.
- **Free-form generation, then parse.** No output grammar — a grammar that forces a
  label at token 0 muzzles the model's reasoning and collapses accuracy to ~6%
  (see "Why no grammar" below). The label is matched out of the final `content`,
  which is reliably exactly the label string (0 fallback parses needed in
  validation).
- **Sampling = server defaults** (temp 1.0 / top-p 0.95 / top-k 64 from the
  llama-swap config); no per-request override. Implies mild run-to-run variance.
- **`max_tokens = 4096`** so reasoning + answer fit with truncation near zero.
- **Concurrency 4** to match the server's `-np 4`.

### Why no grammar

Grammar-constrained decoding (`root ::= "label1" | … | "oos"`) was the original plan
and works mechanically — but on a thinking model it forces an answer at token
position 0, before any reasoning, where the model has no competent distribution. It
collapses onto generic labels (`thank_you`, `order_status`) and scores ~5–6% even at
greedy. The harness still supports it (`--no-think` + the default grammar) for the
direct-answer methodology, which is what the LongBench-Pro runs in this repo use.

## Models (llama-swap ids)

- `Gemma-4-26B-A4B-IT-bartowski-Q8_0`
- `Gemma-4-26B-A4B-IT-bartowski-IQ4_XS`
- `Gemma-4-26B-A4B-IT-bartowski-IQ2_M`

## Running

```bash
# all three quants, sequentially, scoring each as it finishes (resumable)
bash scripts/run_thinking_all.sh

# one quant
python3 scripts/run_clinc.py --model Gemma-4-26B-A4B-IT-bartowski-Q8_0 \
    --out-dir results/bartowski-q8_0 --no-grammar --max-tokens 4096 --concurrency 4
python3 scripts/score_clinc.py --in-dir results/bartowski-q8_0

# fast smoke test (stratified sample incl. OOS)
python3 scripts/run_clinc.py --model <id> --out-dir /tmp/smoke --sample 30 \
    --no-grammar --max-tokens 4096
```

Re-running skips items that already have a prediction, so a crash, an interrupt, or a
higher-`--max-tokens` cleanup pass resumes cleanly.

## Results layout

```
results/<quant>/
├── inference.jsonl   # one row per item (prediction, reasoning trace, metrics)
└── summary.json      # accuracy / OOS / macro-F1 / throughput / per-intent
```

`<quant>` ∈ `bartowski-q8_0`, `bartowski-iq4_xs`, `bartowski-iq2_m`. See
[`SCHEMA.md`](./SCHEMA.md) for field-by-field definitions.

## Live dashboard

[`dashboard.html`](./dashboard.html) is a self-contained cross-quant comparison
(accuracy + throughput tables, per-quant progress). Open it directly in a browser;
it auto-refreshes every 60s.

```bash
bash scripts/publish_loop.sh        # re-score + rebuild dashboard.html every 120s
```

The publisher reads the append-only `inference.jsonl` files (safe to run alongside a
live benchmark) and exits after a final rebuild once the run finishes.

## Metrics

- **in-scope accuracy** — over the 4,500 in-scope items; predicting `oos` for an
  in-scope utterance counts as wrong.
- **OOS recall / precision / F1** — over the 1,000 OOS items, as a binary
  oos-vs-not detection problem. The headline CLINC150 pairing is _(in-scope
  accuracy, OOS recall)_.
- **overall accuracy** — exact-label accuracy over all 5,500.
- **macro-F1** — unweighted mean per-label F1 across all 151 labels.
- **per-intent accuracy**, plus throughput (latency, token counts) and the count of
  truncated reasoning traces.
