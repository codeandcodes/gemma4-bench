#!/usr/bin/env python3
"""
Build summary JSONs from a LongBench-Pro inference JSONL.

Produces two files alongside the inference JSONL:
  - {prefix}_summary_derived.json   - our comprehensive analytics (perfect counts,
                                       throughput stats, language/length cross-cuts)
  - {prefix}_summary_official.json  - Pro's official schema (matches main.py --only_eval),
                                       with T4 Summarization scored via ROUGE-only
                                       (Qwen3-Embedding-8B not available).

Usage:
  python scripts/build_summary.py output/<MODEL>/<prefix>_inference_1-of-1.jsonl
"""
import sys, os, json, re, statistics
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import (
    NDCG, Pairwise_Accuracy, Accuracy, F1_Score, SubEM,
    Summary, Summary_Max_Rouge_L,
)

# Embedding model (loaded lazily, only if Summary tasks are scored)
EMBEDDING_MODEL_PATH = os.environ.get(
    "LONGBENCH_PRO_EMBEDDING_PATH", "model/Qwen3-Embedding-8B"
)
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the SentenceTransformer for Summary scoring.

    Returns the model if available at EMBEDDING_MODEL_PATH, else None
    (in which case Summary falls back to ROUGE-L only).
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model if _embedding_model is not False else None
    if not os.path.isdir(EMBEDDING_MODEL_PATH):
        _embedding_model = False
        return None
    try:
        from sentence_transformers import SentenceTransformer
        # Default to CPU to avoid competing with the inference llama-server GPU.
        # Override with LONGBENCH_PRO_EMBEDDING_DEVICE=cuda when the GPU is idle.
        device = os.environ.get("LONGBENCH_PRO_EMBEDDING_DEVICE", "cpu")
        sys.stderr.write(f"[info] loading {EMBEDDING_MODEL_PATH} on {device}...\n")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            tokenizer_kwargs={"padding_side": "left"},
            device=device,
        )
        sys.stderr.write(f"[info] embedding model loaded.\n")
        return _embedding_model
    except Exception as e:
        sys.stderr.write(f"[warn] failed to load embedding model at "
                         f"{EMBEDDING_MODEL_PATH}: {e}; falling back to ROUGE-only\n")
        _embedding_model = False
        return None

TASK_METRIC = {
    "T1.1 Global Cohesive Retrieval": "NDCG",
    "T1.2 Key-Snippet Retrieval": "NDCG",
    "T2.1 Global Timeline Reconstruction": "Pairwise_Accuracy",
    "T2.2 Local Causal Chain Sorting": "Pairwise_Accuracy",
    "T3.1 Multi-Doc Integration QA": "Accuracy",
    "T3.2 Single-Hop Fact QA": "Accuracy",
    "T4.1 Global-Coverage Constrained Summary": "Summary",
    "T4.2 Query-Focused Summary": "Summary",
    "T5.1 Full-Sentence Citation Alignment": "F1_Score",
    "T5.2 Key-Statement Citation Alignment": "F1_Score",
    "T6.1 Large-Scale Document Clustering": "SubEM",
    "T6.2 Targeted Subset Cluster Identification": "F1_Score",
    "T6.3 Global Frequency Analysis": "Pairwise_Accuracy",
    "T7.1 Global Conflict & Inconsistency Localization": "F1_Score",
    "T7.2 Targeted Rule or Condition Violation Detection": "F1_Score",
    "T7.3 Comprehensive Error & Anomaly Sweep": "F1_Score",
    "T8.1 Structured Multi-Source Consistency Verification": "SubEM",
    "T8.2 Single-Source Targeted Aggregation": "SubEM",
    "T8.3 Long-Context Procedural State Tracking": "SubEM",
    "T9.1 Dependency-Aware Multi-Version Impact Analysis": "F1_Score",
    "T9.2 Localized Interface Change Detection": "F1_Score",
    "T10.1 Large-Scale In-Context Rule Induction": "SubEM",
    "T10.2 Targeted Example-Based Rule Induction": "SubEM",
    "T11.1 Long-Range Entity & Commitment Tracking": "Accuracy",
    "T11.2 Short-Range Reference Resolution & State Query": "Accuracy",
}

EVAL_DIMENSIONS = {
    "token_length": ["8k", "16k", "32k", "64k", "128k", "256k"],
    "contextual_requirement": ["Full", "Partial"],
    "difficulty": ["Easy", "Moderate", "Hard", "Extreme"],
    "primary_task": [
        "T1. Retrieval & Ranking",
        "T2. Sequencing & Structure Reconstruction",
        "T3. Evidence-Grounded QA",
        "T4. Summarization & Synthesis",
        "T5. Attribution & Citation Alignment",
        "T6. Aggregation & Clustering",
        "T7. Consistency & Compliance Checking",
        "T8. Structured & Numeric Reasoning",
        "T9. Version & Code Diff Analysis",
        "T10. Rule Induction & In-Context Learning",
        "T11. Dialogue Memory & Long-Horizon Tracking",
    ],
    "language": ["Chinese", "English"],
}


def score_one(secondary_task, answer, prediction, is_zh):
    """Return metric in [0,1] or 0.0 for empty/erroring predictions."""
    if not prediction or not prediction.strip():
        return 0.0
    metric_name = TASK_METRIC[secondary_task]
    try:
        if metric_name == "NDCG":
            return NDCG(answer, prediction)
        if metric_name == "Pairwise_Accuracy":
            return Pairwise_Accuracy(answer, prediction)
        if metric_name == "Accuracy":
            return Accuracy(answer, prediction)
        if metric_name == "F1_Score":
            return F1_Score(answer, prediction)
        if metric_name == "SubEM":
            return SubEM(answer, prediction)
        if metric_name == "Summary":
            # Official Summary = 0.5*ROUGE-L + 0.5*embedding_cosine.
            # If the embedding model is available, use the full metric; otherwise
            # fall back to ROUGE-L only (which under-counts T4).
            emb = _get_embedding_model()
            if emb is not None:
                return Summary(emb, answer, prediction, is_zh)
            return Summary_Max_Rouge_L(answer, prediction, is_zh)
    except Exception:
        return 0.0
    return 0.0


def score_all(rows):
    """Attach 'metric' field to every row in-place. Returns (scored_count, failed_count)."""
    failed = 0
    for r in rows:
        try:
            r["metric"] = score_one(
                r["secondary_task"], r["answer"], r.get("prediction") or "",
                r["language"] == "Chinese",
            )
        except Exception:
            r["metric"] = 0.0
            failed += 1
    return len(rows), failed


def build_official_summary(rows, bon_num, inference_samples_num, fail_samples_num):
    """Produce Pro's official metric_summary schema (matches modules/evaluation.py)."""
    data = rows

    # average overall (group by id, mean across bon iterations)
    by_id = defaultdict(list)
    for r in data:
        by_id[r["id"]].append(r)
    average_overall_results = []
    inconsistent = 0
    for items in by_id.values():
        if len(items) != bon_num:
            inconsistent += 1
        tmp = items[0].copy()
        tmp["metric"] = sum(it["metric"] for it in items) / len(items)
        average_overall_results.append(tmp)

    def overall_mean(items):
        return sum(it["metric"] for it in items) / len(items) if items else 0.0

    def dim_mean(items, dim, sort_keys):
        groups = defaultdict(list)
        for it in items:
            groups[it[dim]].append(it["metric"])
        out = {}
        for k in sort_keys:
            if k in groups:
                out[k] = sum(groups[k]) / len(groups[k])
            else:
                out[k] = None
        return out

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_questions_num": len(average_overall_results),
        "inference_iterations": bon_num,
        "total_samples_num": len(data),
        "fail_samples_num": fail_samples_num,
        "inference_inconsistent_samples_num": inconsistent,
        "average_overall_metric": overall_mean(average_overall_results),
    }

    # per inference iteration
    for i in range(1, bon_num + 1):
        sub = [r for r in data if r.get("bon_idx") == i]
        summary[f"inference_iteration_{i}_overall_metric"] = overall_mean(sub)

    # per dimension (average across bon iterations)
    for dim, sort_keys in EVAL_DIMENSIONS.items():
        summary[f"average_{dim}_metric"] = dim_mean(average_overall_results, dim, sort_keys)

    # BoN-i / pass@i
    def best_of_n(data, n):
        best = {}
        for r in data:
            if r.get("bon_idx", 1) > n:
                continue
            rid = r["id"]
            if rid not in best or r["metric"] > best[rid]["metric"]:
                best[rid] = r
        return list(best.values())

    def pass_at_n(results):
        if not results:
            return 0.0
        passed = 0
        for r in results:
            if "T4" in r["primary_task"]:
                # Official threshold uses ROUGE+embedding combined; with rouge-only we
                # cap closer to 0, which underreports pass@k for T4.
                if r["metric"] > 0.65:
                    passed += 1
            else:
                if r["metric"] == 1.0:
                    passed += 1
        return passed / len(results)

    for i in range(1, bon_num + 1):
        bon_results = best_of_n(data, i)
        bon_dict = {"overall_metric": overall_mean(bon_results)}
        for dim, sort_keys in EVAL_DIMENSIONS.items():
            bon_dict[dim] = dim_mean(bon_results, dim, sort_keys)
        summary[f"BoN-{i}"] = bon_dict
        summary[f"pass@{i}"] = pass_at_n(bon_results)

    using_embeddings = _get_embedding_model() is not None
    summary["_caveats"] = {
        "T4_summarization_metric": (
            "Full official Summary metric used: 0.5 * Max_Rouge_L + 0.5 * "
            "Max_Semantic_Similarity (Qwen3-Embedding-8B)."
            if using_embeddings else
            "Official Summary metric = 0.5 * Max_Rouge_L + 0.5 * "
            "Max_Semantic_Similarity (Qwen3-Embedding-8B). Embedding model was not "
            "available; T4 scores here use ROUGE-L only, undercounting T4 by "
            "roughly the embedding component."
        ),
    }
    return summary


def build_derived_summary(rows):
    """Comprehensive analytics, richer than the official schema."""
    primary_name = {
        1: "Retrieval & Ranking", 2: "Sequencing & Structure",
        3: "Evidence-Grounded QA", 4: "Summarization & Synthesis*",
        5: "Citation Alignment", 6: "Aggregation & Clustering",
        7: "Consistency & Compliance", 8: "Structured & Numeric Reasoning",
        9: "Version & Code Diff", 10: "Rule Induction & ICL",
        11: "Dialogue Memory & Long-Horizon Tracking",
    }

    def stats(scores):
        if not scores:
            return {"n": 0}
        return {
            "n": len(scores),
            "mean": sum(scores) / len(scores),
            "median": statistics.median(scores),
            "perfect": sum(1 for s in scores if s == 1.0),
            "zero": sum(1 for s in scores if s == 0.0),
            "partial": sum(1 for s in scores if 0 < s < 1),
        }

    scored = [r for r in rows if (r.get("prediction") or "").strip()]
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "totals": {
            "n_total_rows": len(rows),
            "n_scored": len(scored),
            "n_empty": len(rows) - len(scored),
            "empty_rate": (len(rows) - len(scored)) / len(rows) if rows else 0,
        },
        "overall": stats([r["metric"] for r in rows]),
        "overall_scored_only": stats([r["metric"] for r in scored]),
        "by_primary_task": {},
        "by_secondary_task": {},
        "by_metric_type": {},
        "by_language": {},
        "by_difficulty": {},
        "by_token_length": {},
        "by_contextual_requirement": {},
        "throughput": {},
    }

    # By primary task
    for r in rows:
        m = re.match(r"T(\d+)", r["primary_task"])
        key = f"T{m.group(1)} {primary_name[int(m.group(1))]}"
        summary["by_primary_task"].setdefault(key, []).append(r["metric"])
    for k, v in summary["by_primary_task"].items():
        summary["by_primary_task"][k] = stats(v)

    # By secondary task
    for r in rows:
        summary["by_secondary_task"].setdefault(r["secondary_task"], []).append(r["metric"])
    for k, v in summary["by_secondary_task"].items():
        summary["by_secondary_task"][k] = stats(v)

    # By metric type
    for r in rows:
        summary["by_metric_type"].setdefault(TASK_METRIC[r["secondary_task"]], []).append(r["metric"])
    for k, v in summary["by_metric_type"].items():
        summary["by_metric_type"][k] = stats(v)

    # By language / difficulty / token_length / contextual_requirement
    for dim in ("language", "difficulty", "token_length", "contextual_requirement"):
        for r in rows:
            summary[f"by_{dim}"].setdefault(r[dim], []).append(r["metric"])
        for k, v in summary[f"by_{dim}"].items():
            summary[f"by_{dim}"][k] = stats(v)

    # Throughput
    m_rows = [r for r in rows if r.get("metrics", {}).get("input_tokens")]
    if m_rows:
        in_tok = [r["metrics"]["input_tokens"] for r in m_rows]
        out_tok = [r["metrics"]["output_tokens"] for r in m_rows
                   if r["metrics"].get("output_tokens")]
        prefill = [r["metrics"]["prefill_tps"] for r in m_rows
                   if r["metrics"].get("prefill_tps")]
        decode = [r["metrics"]["decode_tps"] for r in m_rows
                  if r["metrics"].get("decode_tps") and r["metrics"]["decode_tps"] > 0]
        ttft = [r["metrics"]["ttft_s"] for r in m_rows
                if r["metrics"].get("ttft_s")]
        total = [r["metrics"]["total_time_s"] for r in m_rows
                 if r["metrics"].get("total_time_s")]

        def quant(vs):
            s = sorted(vs)

            def at(p):
                return s[min(int(len(s) * p / 100), len(s) - 1)]

            return {
                "n": len(vs), "min": s[0], "max": s[-1],
                "p10": at(10), "p25": at(25),
                "median": s[len(s) // 2], "p75": at(75), "p90": at(90),
                "mean": sum(vs) / len(vs),
            }

        summary["throughput"] = {
            "totals": {
                "rows_with_metrics": len(m_rows),
                "total_input_tokens": sum(in_tok),
                "total_output_tokens": sum(out_tok),
            },
            "per_stream_input_tokens": quant(in_tok),
            "per_stream_output_tokens": quant(out_tok),
            "per_stream_ttft_s": quant(ttft),
            "per_stream_total_s": quant(total),
            "per_stream_prefill_tps": quant(prefill),
            "per_stream_decode_tps": quant(decode),
        }

    using_embeddings = _get_embedding_model() is not None
    summary["_caveats"] = {
        "T4_summarization": (
            "T4 Summarization uses the full official metric: "
            "Summary = 0.5 * ROUGE-L + 0.5 * embedding_cosine (Qwen3-Embedding-8B)."
            if using_embeddings else
            "T4 Summarization uses ROUGE-L only; embedding-based similarity "
            "(Qwen3-Embedding-8B) was not available. Real T4 scores will be "
            "higher when embedding component is computed (Summary = 0.5*ROUGE + "
            "0.5*embedding)."
        ),
        "n_proc": (
            "Throughput numbers are per-stream under n_proc=4 contention. "
            "Aggregate throughput is roughly 4x per-stream during the parallel "
            "phase. decode_tps is bandwidth-pegged at ~28-31 t/s per stream due "
            "to MoE-A4B + long-context KV cache reads dominating bandwidth."
        ),
    }
    return summary


def main():
    if len(sys.argv) < 2:
        print("usage: build_summary.py <inference_jsonl> [bon_num=1]")
        sys.exit(1)

    in_path = sys.argv[1]
    bon_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    rows = [json.loads(l) for l in open(in_path)]
    inference_samples_num = len(rows)
    fail_samples_num = sum(1 for r in rows if not (r.get("prediction") or "").strip())

    # add bon_idx if missing (older runs may not have it)
    for r in rows:
        r.setdefault("bon_idx", 1)

    score_all(rows)

    derived = build_derived_summary(rows)
    official = build_official_summary(rows, bon_num, inference_samples_num, fail_samples_num)

    base = os.path.dirname(in_path)
    base_name = os.path.basename(in_path).replace("_inference_1-of-1.jsonl", "")
    derived_path = os.path.join(base, f"{base_name}_summary_derived.json")
    official_path = os.path.join(base, f"{base_name}_summary_official.json")

    with open(derived_path, "w") as f:
        json.dump(derived, f, indent=2, ensure_ascii=False)
    with open(official_path, "w") as f:
        json.dump(official, f, indent=2, ensure_ascii=False)

    print(f"derived  -> {derived_path}")
    print(f"official -> {official_path}")
    print()
    print(f"OVERALL:                {official['average_overall_metric']:.4f}")
    print(f"pass@1:                 {official['pass@1']:.4f}")
    print(f"empty rows:             {fail_samples_num} / {len(rows)} "
          f"({100*fail_samples_num/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
