#!/usr/bin/env python3
"""Score a CLINC150 inference.jsonl into summary.json.

Metrics (the standard CLINC150 headline pair + extras):
  * in_scope_accuracy   - accuracy over the 4,500 in-scope items. Predicting `oos`
                          for an in-scope utterance counts as wrong.
  * oos_recall / precision / f1 - over the 1,000 out-of-scope items (binary
                          oos-vs-not view).
  * overall_accuracy    - exact-label accuracy over all items.
  * macro_f1            - unweighted mean per-label F1 across all 151 labels.
  * per_intent          - {label: {n, correct, accuracy}}.
  * throughput          - mean/median latency and token counts.

Rows are deduplicated by id (last non-empty prediction wins), so resumed/retried
runs score cleanly.

Usage:
  python3 score_clinc.py --in-dir ../results/bartowski-q8_0 [--model-id bartowski-q8_0]
"""
import argparse
import json
import os
import statistics
from collections import defaultdict


def load_rows(path):
    by_id = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # tolerate a partial trailing line while the file is being appended
                continue
            if "id" not in row:
                continue
            prev = by_id.get(row["id"])
            # last write wins, but never let an empty prediction clobber a real one
            if prev is not None and not row.get("prediction") and prev.get("prediction"):
                continue
            by_id[row["id"]] = row
    return list(by_id.values())


def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def score(rows, concurrency=4):
    n_total = len(rows)
    in_scope = [r for r in rows if not r["true_is_oos"]]
    oos = [r for r in rows if r["true_is_oos"]]

    in_scope_correct = sum(r["correct"] for r in in_scope)
    overall_correct = sum(r["correct"] for r in rows)

    # OOS as a binary detection problem
    tp = sum(1 for r in oos if r["pred_is_oos"])              # true oos called oos
    fn = len(oos) - tp                                        # true oos missed
    fp = sum(1 for r in in_scope if r["pred_is_oos"])         # in-scope called oos
    oos_precision = tp / (tp + fp) if (tp + fp) else 0.0
    oos_recall = tp / len(oos) if oos else 0.0

    # Per-label tallies for accuracy + macro-F1 (multiclass, all 151 labels)
    n_by_label = defaultdict(int)       # support (true count)
    correct_by_label = defaultdict(int)
    pred_by_label = defaultdict(int)    # times predicted
    tp_by_label = defaultdict(int)      # correct predictions of this label
    for r in rows:
        t, p = r["true_intent"], r["prediction"]
        n_by_label[t] += 1
        pred_by_label[p] += 1
        if r["correct"]:
            correct_by_label[t] += 1
            tp_by_label[t] += 1

    labels = sorted(set(n_by_label) | set(pred_by_label))
    per_label_f1 = {}
    for lab in labels:
        prec = tp_by_label[lab] / pred_by_label[lab] if pred_by_label[lab] else 0.0
        rec = tp_by_label[lab] / n_by_label[lab] if n_by_label[lab] else 0.0
        per_label_f1[lab] = f1(prec, rec)
    macro_f1 = statistics.mean(per_label_f1.values()) if per_label_f1 else 0.0

    per_intent = {
        lab: {
            "n": n_by_label[lab],
            "correct": correct_by_label[lab],
            "accuracy": (correct_by_label[lab] / n_by_label[lab]) if n_by_label[lab] else None,
        }
        for lab in sorted(n_by_label)
    }

    # Throughput / latency
    lat = [r["metrics"]["latency_s"] for r in rows
           if r.get("metrics", {}).get("latency_s") is not None]
    in_tok = [r["metrics"]["input_tokens"] for r in rows
              if r.get("metrics", {}).get("input_tokens") is not None]
    out_tok = [r["metrics"]["output_tokens"] for r in rows
               if r.get("metrics", {}).get("output_tokens") is not None]
    cached = [r["metrics"]["cached_tokens"] for r in rows
              if r.get("metrics", {}).get("cached_tokens") is not None]
    # Thinking tokens ≈ output tokens minus the tiny final answer (a label,
    # ~1 token per ~4 chars). llama.cpp doesn't return a reasoning-token split,
    # so we derive it; the answer is a few tokens, so this ≈ output_tokens.
    think_tok = [max(0, r["metrics"]["output_tokens"]
                     - max(1, round(len(r.get("raw_output") or "") / 4)))
                 for r in rows
                 if r.get("metrics", {}).get("output_tokens") is not None]

    # Per-request decode throughput = output tokens / wall latency. Latency is
    # almost entirely decode here (prompt is ~97% prefix-cached), so this is a
    # good per-stream generation-TPS estimate.
    pairs = [(r["metrics"]["output_tokens"], r["metrics"]["latency_s"])
             for r in rows
             if r.get("metrics", {}).get("output_tokens") is not None
             and r.get("metrics", {}).get("latency_s")]
    per_stream = [o / l for o, l in pairs if l > 0]
    sum_out = sum(o for o, _ in pairs)
    sum_lat = sum(l for _, l in pairs)
    # Aggregate ≈ concurrency × (total output tokens / total stream-busy seconds);
    # an upper-ish estimate that assumes all `concurrency` streams stay busy.
    gen_tps_aggregate = round(concurrency * sum_out / sum_lat, 1) if sum_lat else None

    throughput = {
        "n_latency_samples": len(lat),
        "latency_s_mean": round(statistics.mean(lat), 4) if lat else None,
        "latency_s_median": round(statistics.median(lat), 4) if lat else None,
        "input_tokens_mean": round(statistics.mean(in_tok), 1) if in_tok else None,
        "output_tokens_mean": round(statistics.mean(out_tok), 2) if out_tok else None,
        "output_tokens_max": max(out_tok) if out_tok else None,
        "thinking_tokens_mean": round(statistics.mean(think_tok), 1) if think_tok else None,
        "thinking_tokens_median": round(statistics.median(think_tok), 1) if think_tok else None,
        "thinking_tokens_max": max(think_tok) if think_tok else None,
        "cached_tokens_mean": round(statistics.mean(cached), 1) if cached else None,
        "decode_tps_per_stream_median": round(statistics.median(per_stream), 1) if per_stream else None,
        "decode_tps_per_stream_mean": round(statistics.mean(per_stream), 1) if per_stream else None,
        "gen_tps_aggregate_est": gen_tps_aggregate,
        "concurrency_assumed": concurrency,
    }

    n_errors = sum(1 for r in rows if not r.get("prediction"))
    n_truncated = sum(1 for r in rows
                      if r.get("metrics", {}).get("finish_reason") == "length")

    return {
        "n_total": n_total,
        "n_in_scope": len(in_scope),
        "n_oos": len(oos),
        "n_errors": n_errors,
        "n_truncated": n_truncated,
        "in_scope_accuracy": round(in_scope_correct / len(in_scope), 4) if in_scope else None,
        "overall_accuracy": round(overall_correct / n_total, 4) if n_total else None,
        "oos_recall": round(oos_recall, 4),
        "oos_precision": round(oos_precision, 4),
        "oos_f1": round(f1(oos_precision, oos_recall), 4),
        "macro_f1": round(macro_f1, 4),
        "oos_confusion": {"tp": tp, "fn": fn, "fp_in_scope_called_oos": fp},
        "throughput": throughput,
        "per_intent": per_intent,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="results dir with inference.jsonl")
    ap.add_argument("--model-id", default=None, help="label for this run")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="client concurrency the run used (for the aggregate-TPS estimate)")
    args = ap.parse_args()

    in_path = os.path.join(args.in_dir, "inference.jsonl")
    rows = load_rows(in_path)
    summary = score(rows, concurrency=args.concurrency)
    summary = {"model_id": args.model_id or os.path.basename(args.in_dir.rstrip("/")),
               "dataset": "clinc_oos/plus/test", **summary}

    out_path = os.path.join(args.in_dir, "summary.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)               # atomic: safe under concurrent scorers

    print(f"== {summary['model_id']} ==")
    print(f"  items:          {summary['n_total']} "
          f"(in-scope {summary['n_in_scope']}, oos {summary['n_oos']}, "
          f"errors {summary['n_errors']}, truncated {summary['n_truncated']})")
    print(f"  in-scope acc:   {summary['in_scope_accuracy']}")
    print(f"  overall acc:    {summary['overall_accuracy']}")
    print(f"  oos recall:     {summary['oos_recall']}  "
          f"precision: {summary['oos_precision']}  f1: {summary['oos_f1']}")
    print(f"  macro-F1:       {summary['macro_f1']}")
    tp = summary["throughput"]
    print(f"  latency (s):    mean {tp['latency_s_mean']}  "
          f"median {tp['latency_s_median']}")
    print(f"  throughput:     {tp['decode_tps_per_stream_median']} tok/s/stream  "
          f"~{tp['gen_tps_aggregate_est']} tok/s aggregate "
          f"(@{tp['concurrency_assumed']} streams)")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
