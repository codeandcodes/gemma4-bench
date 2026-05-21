#!/usr/bin/env python3
"""
Compute a sliced view of LongBench-Pro inference results, filtered by any
combination of language / difficulty / token_length / primary_task /
secondary_task / contextual_requirement.

Uses the same official metric functions as build_summary.py / Pro's
modules/evaluation.py. T4 Summarization uses the full embedding-based
metric if model/Qwen3-Embedding-8B is available, else ROUGE-only.

Usage examples:
  python scripts/compute_slice.py results/q8_k_xl/inference.jsonl --language English
  python scripts/compute_slice.py results/q8_k_xl/inference.jsonl --language English --max-length 32k
  python scripts/compute_slice.py results/q8_k_xl/inference.jsonl --primary-task T4 --difficulty Easy
"""
import sys, os, json, argparse, statistics, re
from collections import defaultdict

# Re-use the scoring logic from build_summary.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from build_summary import TASK_METRIC, score_one  # noqa: E402

LENGTH_ORDER = ["8k", "16k", "32k", "64k", "128k", "256k"]

PRIMARY_NAMES = {
    1: "Retrieval & Ranking",
    2: "Sequencing & Structure",
    3: "Evidence-Grounded QA",
    4: "Summarization & Synthesis",
    5: "Citation Alignment",
    6: "Aggregation & Clustering",
    7: "Consistency & Compliance",
    8: "Structured & Numeric Reasoning",
    9: "Version & Code Diff",
    10: "Rule Induction & ICL",
    11: "Dialogue Memory & Long-Horizon Tracking",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inference_jsonl")
    p.add_argument("--language", choices=["English", "Chinese"], action="append",
                   help="restrict to one or more languages (default: all)")
    p.add_argument("--difficulty", choices=["Easy", "Moderate", "Hard", "Extreme"], action="append",
                   help="restrict to one or more difficulty levels")
    p.add_argument("--token-length", choices=LENGTH_ORDER, action="append",
                   help="explicit token-length bucket(s) to include")
    p.add_argument("--max-length", choices=LENGTH_ORDER,
                   help="include only buckets up to and including this length (e.g. 32k)")
    p.add_argument("--min-length", choices=LENGTH_ORDER,
                   help="include only buckets at or above this length")
    p.add_argument("--primary-task", action="append",
                   help="restrict to one or more primary task prefixes (e.g. 'T1' or 'T4')")
    p.add_argument("--secondary-task", action="append",
                   help="restrict to specific secondary tasks (full string)")
    p.add_argument("--contextual-requirement", choices=["Full", "Partial"], action="append",
                   help="restrict to one or more contextual requirements")
    p.add_argument("--save-json",
                   help="optional path to save the summary as JSON")
    return p.parse_args()


def filter_rows(rows, args):
    out = []
    length_set = None
    if args.token_length:
        length_set = set(args.token_length)
    elif args.max_length or args.min_length:
        lo = LENGTH_ORDER.index(args.min_length) if args.min_length else 0
        hi = LENGTH_ORDER.index(args.max_length) if args.max_length else len(LENGTH_ORDER) - 1
        length_set = set(LENGTH_ORDER[lo:hi + 1])

    for r in rows:
        if args.language and r["language"] not in args.language:
            continue
        if args.difficulty and r["difficulty"] not in args.difficulty:
            continue
        if length_set and r["token_length"] not in length_set:
            continue
        if args.primary_task:
            prefix_ok = any(r["primary_task"].startswith(p + ".") or r["primary_task"].startswith(p + " ")
                            for p in args.primary_task)
            if not prefix_ok:
                continue
        if args.secondary_task and r["secondary_task"] not in args.secondary_task:
            continue
        if args.contextual_requirement and r["contextual_requirement"] not in args.contextual_requirement:
            continue
        out.append(r)
    return out


def compute_summary(rows):
    scored = []
    for r in rows:
        m = score_one(r["secondary_task"], r["answer"],
                      r.get("prediction") or "",
                      r["language"] == "Chinese")
        if m is None:
            m = 0.0
        r["metric"] = m
        scored.append(r)

    if not scored:
        return {"n": 0}

    all_s = [r["metric"] for r in scored]

    def pass_at_1(items):
        ok = 0
        for r in items:
            if "T4" in r["primary_task"]:
                if r["metric"] > 0.65:
                    ok += 1
            else:
                if r["metric"] == 1.0:
                    ok += 1
        return ok / len(items)

    summary = {
        "n": len(scored),
        "overall": {
            "mean": statistics.mean(all_s),
            "median": statistics.median(all_s),
            "perfect": sum(1 for s in all_s if s == 1.0),
            "zero": sum(1 for s in all_s if s == 0.0),
            "partial": sum(1 for s in all_s if 0 < s < 1),
            "pass_at_1": pass_at_1(scored),
        },
        "empty_rows": sum(1 for r in scored if not (r.get("prediction") or "").strip()),
        "by_primary_task": {},
        "by_language": {},
        "by_difficulty": {},
        "by_token_length": {},
    }

    by_p = defaultdict(list)
    for r in scored:
        n = int(re.match(r"T(\d+)", r["primary_task"]).group(1))
        by_p[n].append(r["metric"])
    for n, vals in sorted(by_p.items()):
        summary["by_primary_task"][f"T{n} {PRIMARY_NAMES[n]}"] = {
            "n": len(vals), "mean": statistics.mean(vals),
            "perfect": sum(1 for v in vals if v == 1.0),
        }

    for dim, dim_name in [("language", "by_language"), ("difficulty", "by_difficulty"),
                           ("token_length", "by_token_length")]:
        groups = defaultdict(list)
        for r in scored:
            groups[r[dim]].append(r["metric"])
        order = (["Chinese", "English"] if dim == "language"
                 else (["Easy", "Moderate", "Hard", "Extreme"] if dim == "difficulty"
                       else LENGTH_ORDER))
        for k in order:
            if k in groups:
                summary[dim_name][k] = {
                    "n": len(groups[k]), "mean": statistics.mean(groups[k]),
                }
    return summary


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(args.inference_jsonl)]
    filtered = filter_rows(rows, args)
    summary = compute_summary(filtered)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nsaved to: {args.save_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
