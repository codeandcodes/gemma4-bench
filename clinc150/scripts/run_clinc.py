#!/usr/bin/env python3
"""CLINC150 intent-classification benchmark runner.

Hits an OpenAI-compatible llama.cpp server (via llama-swap) and classifies each
CLINC150 `plus`/test utterance into exactly one of the 151 labels (150 in-scope
intents + `oos`) using grammar-constrained decoding.

Design notes
------------
* Output is constrained by a GBNF grammar (`root ::= "label1" | ... | "oos"`),
  so every prediction is guaranteed to be a valid label and the model's thinking
  preamble is bypassed (the label is emitted immediately in `content`).
* Sampling uses the server's *configured* defaults (temp/top-p/top-k from the
  llama-swap config) — we deliberately do NOT send temperature/top_p/top_k.
* The static instruction+label-list prefix comes first and is identical for every
  request, so llama.cpp prefix-caches it (huge speedup after the first request).
* Resumable: rows are appended to inference.jsonl as they complete. Re-running
  skips any id that already has a non-empty prediction; errored rows are retried.

Usage
-----
  python3 run_clinc.py --model Gemma-4-26B-A4B-IT-bartowski-Q8_0 \
      --out-dir ../results/bartowski-q8_0
  python3 run_clinc.py --model ... --out-dir ... --sample 20   # stratified smoke test
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI

DATASET = "clinc_oos"
CONFIG = "plus"
SPLIT = "test"


def build_instructions(in_scope_labels):
    """The static, cacheable prompt prefix (everything before the utterance)."""
    return (
        "You are an intent classifier for a task-oriented dialog system. "
        "Read the user's utterance and classify it into exactly one of the "
        "following intent labels:\n\n"
        + ", ".join(in_scope_labels)
        + "\n\nIf the utterance does not clearly match any of these intents, "
        "respond with: oos\n\n"
        "Respond with only the single intent label and nothing else."
    )


def build_grammar(all_labels):
    """GBNF grammar allowing exactly one of the label strings."""
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')
    return "root ::= " + " | ".join('"%s"' % esc(n) for n in all_labels)


def _norm(s):
    return s.strip().strip("`\"'*.:\n ").lower()


def match_label(content, label_set, label_norm):
    """Extract a valid label from free-form (thinking-mode) output.

    The model is instructed to answer with only the label, so its final
    `content` is normally exactly the label. Falls back to last-line, then a
    word-boundary search (latest, then longest match wins). Returns '' if
    nothing matches (counts as wrong, and is retried on the next run).
    """
    if not content:
        return ""
    s = content.strip()
    if _norm(s) in label_norm:
        return label_norm[_norm(s)]
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if lines and _norm(lines[-1]) in label_norm:
        return label_norm[_norm(lines[-1])]
    low = s.lower()
    best = None
    for lab in label_set:
        for mo in re.finditer(r"(?<![a-z0-9_])" + re.escape(lab) + r"(?![a-z0-9_])", low):
            cand = (mo.start(), len(lab), lab)
            if best is None or cand[:2] > best[:2]:
                best = cand
    return best[2] if best else ""


def load_done_ids(path):
    """Ids that already have a non-empty prediction (so we can resume/retry)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prediction"):
                done.add(row["id"])
    return done


def classify(client, model, instructions, grammar, text, max_tokens,
             label_set, label_norm, think_off=False, retries=3):
    """Return (prediction, raw_output, thinking, metrics, error).

    Gemma 4 is a thinking model: by default it reasons (in `reasoning_content`)
    before answering in `content`.

      * grammar is not None -> constrained decoding; `content` IS a valid label.
        Pair with think_off=True, else the position-0 constraint muzzles the
        model and yields garbage.
      * grammar is None     -> free-form; the model thinks, then the label is
        matched out of the final `content`. This is the thinking-mode path.

    `think_off=True` sends chat_template_kwargs={"enable_thinking": false}.
    prediction == '' on hard failure or unparseable output (retried next run).
    """
    content = instructions + "\n\nUtterance: " + text + "\nIntent:"
    extra_body = {}
    if grammar is not None:
        extra_body["grammar"] = grammar
    if think_off:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    last_err = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            dt = time.time() - t0
            ch = r.choices[0]
            raw = (ch.message.content or "").strip()
            thinking = getattr(ch.message, "reasoning_content", None) or ""
            pred = match_label(raw, label_set, label_norm)
            u = r.usage
            cached = None
            if u and getattr(u, "prompt_tokens_details", None):
                cached = u.prompt_tokens_details.cached_tokens
            metrics = {
                "input_tokens": u.prompt_tokens if u else None,
                "output_tokens": u.completion_tokens if u else None,
                "thinking_chars": len(thinking),
                "cached_tokens": cached,
                "latency_s": round(dt, 3),
                "finish_reason": ch.finish_reason,
            }
            return pred, raw, thinking, metrics, None
        except Exception as e:  # noqa: BLE001 - record and retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return "", "", "", {"latency_s": None}, str(last_err)[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="llama-swap model id")
    ap.add_argument("--out-dir", required=True, help="results dir for this quant")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--api-key", default="sk-nokey")
    ap.add_argument("--concurrency", type=int, default=4, help="match server -np")
    ap.add_argument("--no-think", action="store_true",
                    help="disable the model's thinking mode (required for the "
                         "grammar-constrained direct-answer method on Gemma 4)")
    ap.add_argument("--no-grammar", action="store_true",
                    help="free-form generation; parse the label out of the final "
                         "answer (use for thinking mode, where a grammar would "
                         "muzzle the model's reasoning)")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="cap per request; use a large value (e.g. 3072) in "
                         "thinking mode so reasoning + answer fit")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N items")
    ap.add_argument("--sample", type=int, default=None,
                    help="run N evenly-spaced items (stratified smoke test)")
    ap.add_argument("--progress-every", type=int, default=250)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "inference.jsonl")

    print(f"[load] {DATASET}/{CONFIG} {SPLIT}", flush=True)
    ds = load_dataset(DATASET, CONFIG)[SPLIT]
    names = ds.features["intent"].names           # 151, includes 'oos'
    in_scope = [n for n in names if n != "oos"]
    instructions = build_instructions(in_scope)
    grammar = None if args.no_grammar else build_grammar(names)
    label_norm = {_norm(n): n for n in names}

    # Build the full item list: (id, text, true_intent, true_is_oos)
    intents = ds["intent"]
    texts = ds["text"]
    items = [
        {"id": i, "text": texts[i], "true_intent": names[intents[i]],
         "true_is_oos": names[intents[i]] == "oos"}
        for i in range(len(ds))
    ]

    if args.sample:
        n = min(args.sample, len(items))
        idxs = sorted({round(k * (len(items) - 1) / max(1, n - 1)) for k in range(n)})
        items = [items[i] for i in idxs]
    elif args.limit:
        items = items[: args.limit]

    done = load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    mode = ("grammar" if grammar is not None else "free-form") + \
           ("/no-think" if args.no_think else "/think")
    print(f"[plan] model={args.model} | mode={mode} max_tokens={args.max_tokens} | "
          f"total={len(items)} done={len(done)} todo={len(todo)} | "
          f"concurrency={args.concurrency}", flush=True)
    if not todo:
        print("[done] nothing to do", flush=True)
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=args.timeout, max_retries=0)

    n_done = 0
    n_correct = 0
    n_err = 0
    t_start = time.time()
    with open(out_path, "a") as fout, \
            ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(classify, client, args.model, instructions, grammar,
                      it["text"], args.max_tokens, names, label_norm,
                      args.no_think): it
            for it in todo
        }
        for fut in as_completed(futs):
            it = futs[fut]
            pred, raw, thinking, metrics, err = fut.result()
            row = {
                "id": it["id"],
                "text": it["text"],
                "true_intent": it["true_intent"],
                "true_is_oos": it["true_is_oos"],
                "prediction": pred,
                "pred_is_oos": pred == "oos",
                "correct": pred == it["true_intent"],
                "raw_output": raw,
                "thinking": thinking,
                "metrics": metrics,
            }
            if err:
                row["error"] = err
                n_err += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            n_correct += int(row["correct"])
            if n_done % args.progress_every == 0 or n_done == len(todo):
                rate = n_done / (time.time() - t_start)
                print(f"[{n_done}/{len(todo)}] acc={n_correct/n_done:.3f} "
                      f"err={n_err} {rate:.1f} it/s", flush=True)

    dt = time.time() - t_start
    print(f"[finished] {n_done} items in {dt:.1f}s "
          f"({n_done/dt:.1f} it/s) | running_acc={n_correct/max(1,n_done):.3f} "
          f"| errors={n_err}", flush=True)
    print(f"[out] {out_path}", flush=True)


if __name__ == "__main__":
    main()
