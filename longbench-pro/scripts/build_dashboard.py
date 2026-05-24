#!/usr/bin/env python3
"""
Build dashboard outputs from all summary_official.json files under results/*/.
Produces two artifacts:

  - dashboard.html: single self-contained HTML file with all comparison tables.
                    No external CSS/JS dependencies; openable from disk.
  - dashboard.csv:  long-format CSV (one row per quant × dimension × value),
                    convenient for pivoting in Excel/Sheets.

Usage:
  python scripts/build_dashboard.py
  # writes ../dashboard.html and ../dashboard.csv at the repo root
"""
import sys, os, json, argparse, csv, datetime, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_HTML = os.path.join(REPO_ROOT, "..", "dashboard.html")
DEFAULT_CSV = os.path.join(REPO_ROOT, "..", "dashboard.csv")

# Make _pro_metrics importable so we can re-score on-the-fly subset cuts
# of full-benchmark runs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_summary import score_one  # noqa: E402


# Display-friendly aliases for the quant directory names.
QUANT_DISPLAY = {
    "q8_k_xl": "Q8_K_XL (Unsloth UD)",
    "q4_k_m": "Q4_K_M (Unsloth UD)",
    "iq2_m": "IQ2_M (Unsloth UD)",
    "bartowski-q8_0": "Q8_0 (Bartowski)",
    "bartowski-iq4_xs": "IQ4_XS (Bartowski)",
    "bartowski-iq2_m": "IQ2_M (Bartowski)",
}
QUANT_ORDER = ["q8_k_xl", "q4_k_m", "iq2_m",
               "bartowski-q8_0", "bartowski-iq4_xs", "bartowski-iq2_m"]
QUANT_SIZE_GB = {
    "q8_k_xl": 27.0, "q4_k_m": 16.0, "iq2_m": 9.0,
    "bartowski-q8_0": 26.9, "bartowski-iq4_xs": 14.2, "bartowski-iq2_m": 10.7,
}

PRIMARY_TASKS = [
    ("T1. Retrieval & Ranking", "T1 Retrieval"),
    ("T2. Sequencing & Structure Reconstruction", "T2 Sequencing"),
    ("T3. Evidence-Grounded QA", "T3 Evidence QA"),
    ("T4. Summarization & Synthesis", "T4 Summary"),
    ("T5. Attribution & Citation Alignment", "T5 Citation"),
    ("T6. Aggregation & Clustering", "T6 Aggregation"),
    ("T7. Consistency & Compliance Checking", "T7 Consistency"),
    ("T8. Structured & Numeric Reasoning", "T8 Struct Reason"),
    ("T9. Version & Code Diff Analysis", "T9 Code Diff"),
    ("T10. Rule Induction & In-Context Learning", "T10 Rule Induction"),
    ("T11. Dialogue Memory & Long-Horizon Tracking", "T11 Dialogue Memory"),
]
LENGTHS = ["8k", "16k", "32k", "64k", "128k", "256k"]
DIFFICULTIES = ["Easy", "Moderate", "Hard", "Extreme"]
LANGUAGES = ["Chinese", "English"]
CTX_REQS = ["Full", "Partial"]


def quant_display(d: str) -> str:
    return QUANT_DISPLAY.get(d, d)


def fmt(v, digits=3, dash="—"):
    if v is None:
        return dash
    return f"{v:.{digits}f}"


def fmt_pct(v, dash="—"):
    if v is None:
        return dash
    return f"{100*v:.1f}%"


def load_summary(quant_dir: str) -> Optional[Dict]:
    p = os.path.join(quant_dir, "summary_official.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p))


def load_slice(quant_dir: str, slice_name: str) -> Optional[Dict]:
    p = os.path.join(quant_dir, "slices", f"{slice_name}.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p))


def load_inference(quant_dir: str) -> Optional[List[Dict]]:
    """Read all rows from inference.jsonl."""
    p = os.path.join(quant_dir, "inference.jsonl")
    if not os.path.isfile(p):
        return None
    return [json.loads(l) for l in open(p)]


def compute_subset_official(rows: List[Dict]) -> Dict:
    """Re-score and aggregate a list of inference rows into the same shape as
    summary_official.json. Used to compute subset cuts of full-benchmark runs
    so they slot into the subset dashboard alongside the natively-subset runs.
    """
    scored = []
    for r in rows:
        m = score_one(r["secondary_task"], r["answer"],
                      r.get("prediction") or "",
                      r["language"] == "Chinese")
        r2 = dict(r)
        r2["metric"] = m if m is not None else 0.0
        scored.append(r2)

    def mean_or_none(xs):
        return sum(xs) / len(xs) if xs else None

    def group(dim, sort_keys):
        groups = defaultdict(list)
        for r in scored:
            groups[r[dim]].append(r["metric"])
        return {k: mean_or_none(groups[k]) for k in sort_keys if k in groups}

    def group_primary():
        groups = defaultdict(list)
        for r in scored:
            groups[r["primary_task"]].append(r["metric"])
        return {k: mean_or_none(v) for k, v in groups.items()}

    all_metric = [r["metric"] for r in scored]
    empty = sum(1 for r in scored if not (r.get("prediction") or "").strip())
    # pass@1 using the Pro convention (T4 threshold 0.65, others exact-1.0)
    passed = sum(
        1 for r in scored
        if ("T4" in r["primary_task"] and r["metric"] > 0.65)
        or ("T4" not in r["primary_task"] and r["metric"] == 1.0)
    )
    return {
        "total_samples_num": len(scored),
        "total_questions_num": len(scored),
        "fail_samples_num": empty,
        "average_overall_metric": mean_or_none(all_metric) or 0.0,
        "pass@1": passed / len(scored) if scored else 0.0,
        "average_token_length_metric": group("token_length", LENGTHS),
        "average_contextual_requirement_metric": group("contextual_requirement", CTX_REQS),
        "average_difficulty_metric": group("difficulty", DIFFICULTIES),
        "average_language_metric": group("language", LANGUAGES),
        "average_primary_task_metric": group_primary(),
    }


def collect_quants(results_dir: str) -> List[Tuple[str, Dict]]:
    found: Dict[str, Dict] = {}
    for d in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, d)
        if not os.path.isdir(full):
            continue
        s = load_summary(full)
        if s:
            found[d] = s
    out = []
    for name in QUANT_ORDER:
        if name in found:
            out.append((name, found.pop(name)))
    for name in sorted(found):
        out.append((name, found[name]))
    return out


def color_for(score: Optional[float]) -> str:
    """Return a heatmap-friendly background hue for a [0,1] score."""
    if score is None:
        return "background:#f6f6f6;color:#999;"
    # Green for high, red for low; use lightness so text stays readable.
    hue = max(0, min(120, int(score * 120)))
    return f"background:hsl({hue},65%,87%);"


# ---------- CSV (long format) ----------

def csv_rows(quants: List[Tuple[str, Dict]]) -> List[Dict]:
    rows: List[Dict] = []
    for d, s in quants:
        common = {
            "quant_dir": d,
            "quant": quant_display(d),
            "size_gb": QUANT_SIZE_GB.get(d, ""),
            "total_samples": s.get("total_samples_num"),
            "fail_samples": s.get("fail_samples_num"),
        }
        # Overall
        rows.append({**common, "dimension": "overall", "value": "all",
                     "score": s["average_overall_metric"]})
        rows.append({**common, "dimension": "pass_at_1", "value": "all",
                     "score": s.get("pass@1")})

        for k, _ in PRIMARY_TASKS:
            rows.append({**common, "dimension": "primary_task",
                         "value": k, "score": s["average_primary_task_metric"].get(k)})
        for k in LENGTHS:
            rows.append({**common, "dimension": "token_length",
                         "value": k, "score": s["average_token_length_metric"].get(k)})
        for k in DIFFICULTIES:
            rows.append({**common, "dimension": "difficulty",
                         "value": k, "score": s["average_difficulty_metric"].get(k)})
        for k in LANGUAGES:
            rows.append({**common, "dimension": "language",
                         "value": k, "score": s["average_language_metric"].get(k)})
        for k in CTX_REQS:
            rows.append({**common, "dimension": "contextual_requirement",
                         "value": k, "score": s["average_contextual_requirement_metric"].get(k)})
    return rows


def write_csv(quants, path: str) -> None:
    rows = csv_rows(quants)
    fields = ["quant_dir", "quant", "size_gb", "total_samples", "fail_samples",
              "dimension", "value", "score"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- HTML ----------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 24px auto; padding: 0 16px; color: #222; }
h1 { margin-bottom: 4px; }
h2 { margin-top: 36px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
.subtitle { color: #666; margin-bottom: 16px; font-size: 14px; }
.note { background: #f7f7f9; border-left: 3px solid #ccc; padding: 8px 12px;
        font-size: 13px; color: #444; margin: 12px 0; }
table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
th, td { border: 1px solid #e6e6e6; padding: 6px 10px; text-align: right;
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #fafafa; font-weight: 600; }
tbody tr:nth-child(even) td { background-color: #fafafa; }
td.metric { font-variant-numeric: tabular-nums; }
td.dash { color: #aaa; }
.note code { font-family: monospace; background: #efefef; padding: 1px 4px;
             border-radius: 3px; }
"""


def build_html(quants: List[Tuple[str, Dict]], results_dir: str) -> str:
    quant_dirs = [d for d, _ in quants]
    quant_summaries = {d: s for d, s in quants}
    full_runs = [(d, s) for d, s in quants if s["total_samples_num"] >= 1000]
    sub_runs = [(d, s) for d, s in quants if s["total_samples_num"] < 1000]
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # For the subset section, extend `sub_runs` with subset-cuts of full-benchmark
    # quants so all quants appear in the apples-to-apples 500-item view.
    sub_runs_extended = []
    subset_ids = None
    if sub_runs:
        # Use the first subset run's IDs as the canonical 500-item subset.
        sample_dir = os.path.join(results_dir, sub_runs[0][0])
        sample_rows = load_inference(sample_dir) or []
        subset_ids = {r["id"] for r in sample_rows}
        # Insert subset cuts of full-run quants into the ordering, keeping
        # the same display order as QUANT_ORDER.
        sub_by_dir = dict(sub_runs)
        full_by_dir = dict(full_runs)
        seen = set()
        for d in QUANT_ORDER:
            if d in sub_by_dir:
                sub_runs_extended.append((d, sub_by_dir[d]))
                seen.add(d)
            elif d in full_by_dir and subset_ids:
                rows = load_inference(os.path.join(results_dir, d)) or []
                cut = [r for r in rows if r["id"] in subset_ids]
                if cut:
                    sub_runs_extended.append((d, compute_subset_official(cut)))
                    seen.add(d)
        # Append any unknowns from sub_by_dir at the end (for safety).
        for d, s in sub_runs:
            if d not in seen:
                sub_runs_extended.append((d, s))

    def cell(v, digits=3):
        if v is None:
            return f'<td class="dash metric">—</td>'
        return f'<td class="metric" style="{color_for(v)}">{fmt(v, digits)}</td>'

    def table_one_row_per_quant(rows: List[List[str]]) -> str:
        # rows[0] is header
        out = ["<table><thead><tr>"]
        out += [f"<th>{h}</th>" for h in rows[0]]
        out.append("</tr></thead><tbody>")
        for row in rows[1:]:
            out.append("<tr>" + "".join(row) + "</tr>")
        out.append("</tbody></table>")
        return "\n".join(out)

    out = []
    out.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    out.append("<title>LongBench Pro — Gemma 4 26B-A4B-it dashboard</title>")
    out.append(f"<style>{CSS}</style></head><body>")

    out.append("<h1>LongBench Pro — Gemma 4 26B-A4B-it dashboard</h1>")
    out.append(f'<div class="subtitle">Generated {generated}. All scores use the full official Pro '
               'metric suite (Summary = 0.5×ROUGE-L + 0.5×embedding_cosine, Qwen3-Embedding-8B). '
               'Empty predictions count as 0.0 in every aggregate.</div>')
    out.append('<div class="note">Cell colors: red ≈ 0, green ≈ 1. CSV at '
               '<code>dashboard.csv</code>; per-quant JSON at '
               '<code>longbench-pro/results/&lt;quant&gt;/summary_official.json</code>.</div>')

    # Headline table
    out.append("<h2>Headline</h2>")
    out.append('<div class="subtitle">Overall + per-primary-task. Note that '
               'sample sizes differ (some runs are on the full 1500-item benchmark, '
               'some on the 500-item English ≤64K subset).</div>')

    header = ["Quant", "Disk", "Samples", "Overall", "pass@1", "Empty rate"]
    header += [short for _, short in PRIMARY_TASKS]
    body_rows = []
    for d, s in quants:
        size = QUANT_SIZE_GB.get(d)
        size_str = f"{size:.1f} GB" if size else "—"
        pt = s["average_primary_task_metric"]
        empty_rate = s["fail_samples_num"] / max(s["total_samples_num"], 1)
        row = [f"<td><b>{quant_display(d)}</b></td>"]
        row.append(f"<td>{size_str}</td>")
        row.append(f"<td>{s['total_samples_num']}</td>")
        row.append(cell(s["average_overall_metric"]))
        row.append(cell(s.get("pass@1")))
        row.append(f'<td class="metric">{fmt_pct(empty_rate)}</td>')
        for k, _ in PRIMARY_TASKS:
            row.append(cell(pt.get(k)))
        body_rows.append(row)
    out.append(table_one_row_per_quant([header] + body_rows))

    def cross_cut_table(runs, dim_key: str, dim_values: List[str], dim_label: str) -> str:
        header = [dim_label] + [quant_display(d) for d, _ in runs]
        body = []
        for val in dim_values:
            row = [f"<td>{val}</td>"]
            for _, s in runs:
                row.append(cell(s[dim_key].get(val)))
            body.append(row)
        return table_one_row_per_quant([header] + body)

    if full_runs:
        out.append("<h2>Full-benchmark runs (1500 items each)</h2>")
        out.append("<h3>By token length</h3>")
        out.append(cross_cut_table(full_runs, "average_token_length_metric", LENGTHS, "Length"))
        out.append("<h3>By difficulty</h3>")
        out.append(cross_cut_table(full_runs, "average_difficulty_metric", DIFFICULTIES, "Difficulty"))
        out.append("<h3>By language</h3>")
        out.append(cross_cut_table(full_runs, "average_language_metric", LANGUAGES, "Language"))
        out.append("<h3>By contextual requirement</h3>")
        out.append(cross_cut_table(full_runs, "average_contextual_requirement_metric", CTX_REQS, "Ctx req"))

    if sub_runs_extended:
        out.append("<h2>Subset runs (English ≤64K, 500 items each)</h2>")
        out.append('<div class="subtitle">All quants on the same 500 item IDs '
                   '(English text, ≤64K Qwen tokens). For quants whose full run '
                   'covered all 1500 items (Unsloth Q8, Q4), the values below are '
                   'computed on-the-fly by re-scoring their inference rows '
                   'restricted to this same subset.</div>')

        # Headline table for the subset
        sub_header = ["Quant", "Disk", "Mean", "pass@1", "Empty rate"]
        sub_header += [short for _, short in PRIMARY_TASKS]
        sub_body = []
        for d, s in sub_runs_extended:
            size = QUANT_SIZE_GB.get(d)
            size_str = f"{size:.1f} GB" if size else "—"
            pt = s["average_primary_task_metric"]
            empty_rate = s["fail_samples_num"] / max(s["total_samples_num"], 1)
            r = [f"<td><b>{quant_display(d)}</b></td>",
                 f"<td>{size_str}</td>",
                 cell(s["average_overall_metric"]),
                 cell(s.get("pass@1")),
                 f'<td class="metric">{fmt_pct(empty_rate)}</td>']
            for k, _ in PRIMARY_TASKS:
                r.append(cell(pt.get(k)))
            sub_body.append(r)
        out.append(table_one_row_per_quant([sub_header] + sub_body))

        out.append("<h3>By token length (subset only covers 8K–64K)</h3>")
        out.append(cross_cut_table(sub_runs_extended, "average_token_length_metric", LENGTHS[:4], "Length"))
        out.append("<h3>By difficulty</h3>")
        out.append(cross_cut_table(sub_runs_extended, "average_difficulty_metric", DIFFICULTIES, "Difficulty"))

    # Slices
    slice_rows = []
    header = ["Quant", "n", "Mean", "Perfect", "Zero", "pass@1"]
    for d, _ in quants:
        slc = load_slice(os.path.join(results_dir, d), "english_le32k")
        if not slc or "overall" not in slc:
            continue
        ov = slc["overall"]
        n = slc.get("n", 0)
        row = [f"<td><b>{quant_display(d)}</b></td>",
               f"<td>{n}</td>",
               cell(ov.get("mean")),
               f'<td class="metric">{ov.get("perfect", 0)}/{n}</td>',
               f'<td class="metric">{ov.get("zero", 0)}/{n}</td>',
               cell(ov.get("pass_at_1"))]
        slice_rows.append(row)
    if slice_rows:
        out.append("<h2>English-only ≤32K slice</h2>")
        out.append('<div class="subtitle">From '
                   '<code>results/&lt;quant&gt;/slices/english_le32k.json</code>. '
                   '375 items (125 each at 8K/16K/32K, English only). '
                   'A cleaner subset for typical workloads (no Chinese, no long-context).</div>')
        out.append(table_one_row_per_quant([header] + slice_rows))

    out.append("<h2>How to consume programmatically</h2>")
    out.append('<div class="subtitle">See <code>longbench-pro/SCHEMA.md</code> for the '
               'JSON schema of each result file and example Python snippets.</div>')

    out.append("</body></html>")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--html-out", default=DEFAULT_HTML)
    p.add_argument("--csv-out", default=DEFAULT_CSV)
    args = p.parse_args()

    quants = collect_quants(args.results_dir)
    if not quants:
        print("no summary_official.json files found; nothing to build")
        return

    html_path = os.path.abspath(args.html_out)
    with open(html_path, "w") as f:
        f.write(build_html(quants, args.results_dir))
    print(f"wrote {html_path}")

    csv_path = os.path.abspath(args.csv_out)
    write_csv(quants, csv_path)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
