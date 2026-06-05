#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from the CLINC150 per-quant summaries.

Reads results/<quant>/summary.json (+ inference.jsonl row counts for live
progress) and writes results-relative dashboard.html. Safe to run repeatedly
while the benchmark is in progress; quants without a summary yet show as pending.
"""
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # clinc150/
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "dashboard.html")
TARGET = 5500                                     # full clinc_oos/plus test split

# (dir, display name) in the order they are run
QUANTS = [
    ("bartowski-q8_0", "Q8_0"),
    ("bartowski-iq4_xs", "IQ4_XS"),
    ("bartowski-iq2_m", "IQ2_M"),
]


def load_summary(d):
    p = os.path.join(RESULTS, d, "summary.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def count_rows(d):
    p = os.path.join(RESULTS, d, "inference.jsonl")
    try:
        with open(p) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def num(x, d=0):
    return f"{x:,.{d}f}" if isinstance(x, (int, float)) else "—"


def secs(x):
    return f"{x:.1f}s" if isinstance(x, (int, float)) else "—"


def status_of(done):
    if done <= 0:
        return ("pending", "⋯")
    if done >= TARGET:
        return ("done", "✓")
    return ("running", "▶")


def build():
    rows = []
    for d, name in QUANTS:
        s = load_summary(d)
        done = count_rows(d)
        rows.append({"dir": d, "name": name, "summary": s, "done": done})

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def acc_cells(r):
        s = r["summary"] or {}
        tp = s.get("throughput") or {}
        cls, icon = status_of(r["done"])
        prog = min(100, round(100 * r["done"] / TARGET))
        oos_seen = (s.get("n_oos") or 0) > 0
        oos = lambda k: pct(s.get(k)) if oos_seen else '<span class="muted">—</span>'
        return f"""
        <tr>
          <td class="name">{r['name']} <span class="badge {cls}">{icon} {cls}</span></td>
          <td class="prog">
            <div class="bar"><div class="fill" style="width:{prog}%"></div></div>
            <span class="small">{r['done']:,} / {TARGET:,} ({prog}%)</span>
          </td>
          <td class="big">{pct(s.get('in_scope_accuracy'))}</td>
          <td>{oos('oos_recall')}</td>
          <td>{oos('oos_precision')}</td>
          <td>{oos('oos_f1')}</td>
          <td>{pct(s.get('overall_accuracy'))}</td>
          <td>{pct(s.get('macro_f1'))}</td>
          <td class="{'warn' if (s.get('n_errors') or 0) else ''}">{num(s.get('n_errors'))}</td>
          <td class="{'warn' if (s.get('n_truncated') or 0) else ''}">{num(s.get('n_truncated'))}</td>
        </tr>"""

    def perf_cells(r):
        s = r["summary"] or {}
        tp = s.get("throughput") or {}
        return f"""
        <tr>
          <td class="name">{r['name']}</td>
          <td class="big">{num(tp.get('decode_tps_per_stream_median'), 1)}</td>
          <td>{num(tp.get('gen_tps_aggregate_est'), 0)}</td>
          <td>{num(tp.get('input_tokens_mean'), 0)}</td>
          <td>{num(tp.get('cached_tokens_mean'), 0)}</td>
          <td>{num(tp.get('output_tokens_mean'), 0)}</td>
          <td class="big">{num(tp.get('thinking_tokens_mean'), 0)}</td>
          <td>{num(tp.get('output_tokens_max'))}</td>
          <td>{secs(tp.get('latency_s_mean'))}</td>
          <td>{secs(tp.get('latency_s_median'))}</td>
        </tr>"""

    def findings_html():
        s = {r["name"]: (r["summary"] or {}) for r in rows}
        need = ("Q8_0", "IQ4_XS", "IQ2_M")
        if not all(s.get(n) and (s[n].get("n_total") or 0) >= TARGET
                   and s[n].get("oos_recall") is not None for n in need):
            return ""              # only show once all three runs are complete
        isc = [s[n]["in_scope_accuracy"] for n in need]
        oos = {n: s[n]["oos_recall"] for n in need}
        tps = {n: (s[n].get("throughput") or {}).get("gen_tps_aggregate_est") or 0
               for n in need}
        drop = (max(oos["Q8_0"], oos["IQ4_XS"]) - oos["IQ2_M"]) * 100
        speedup = (tps["IQ4_XS"] / tps["Q8_0"] - 1) * 100 if tps["Q8_0"] else 0
        bullets = [
            f"<b>In-scope intent accuracy is nearly quant-independent</b> — only "
            f"{min(isc) * 100:.1f}–{max(isc) * 100:.1f}% across Q8_0 / IQ4_XS / IQ2_M. "
            f"Even 2-bit holds for routing in-domain utterances.",
            f"<b>Out-of-scope detection is what low-bit costs you</b> — OOS recall "
            f"{oos['Q8_0'] * 100:.0f}% / {oos['IQ4_XS'] * 100:.0f}% / {oos['IQ2_M'] * 100:.0f}%; "
            f"IQ2_M drops ~{drop:.0f}pt (it abstains less), pulling its overall accuracy down.",
            f"<b>IQ4_XS is the sweet spot</b> — matches Q8_0 accuracy, edges it on OOS recall, "
            f"and runs ~{speedup:.0f}% faster ({tps['IQ4_XS']:.0f} vs {tps['Q8_0']:.0f} tok/s aggregate).",
            f"<b>Throughput scales inversely with precision</b> — "
            f"{tps['Q8_0']:.0f} → {tps['IQ4_XS']:.0f} → {tps['IQ2_M']:.0f} tok/s aggregate "
            f"(smaller quant = faster decode).",
        ]
        lis = "".join(f"<li>{b}</li>" for b in bullets)
        return f'  <h2>Key findings</h2>\n  <div class="card findings"><ul>{lis}</ul></div>\n'

    acc_rows = "".join(acc_cells(r) for r in rows)
    perf_rows = "".join(perf_cells(r) for r in rows)
    findings = findings_html()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLINC150 — Gemma 4 quant comparison</title>
<style>
  :root {{ --bg:#0f1115; --card:#171a21; --line:#262b36; --fg:#e6e9ef; --mut:#8b93a7;
           --accent:#6ea8fe; --good:#3fb950; --warn:#e3b341; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:32px 20px 60px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:var(--mut); margin:0 0 2px; }}
  .upd {{ color:var(--mut); font-size:12px; margin:0 0 24px; }}
  h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em;
        color:var(--mut); margin:28px 0 10px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; }}
  th, td {{ padding:10px 12px; text-align:right; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ color:var(--mut); font-weight:600; font-size:12px; text-align:right;
       background:#13161c; position:sticky; top:0; }}
  th:first-child, td:first-child {{ text-align:left; }}
  tr:last-child td {{ border-bottom:none; }}
  td.name {{ font-weight:600; }}
  td.big {{ font-weight:700; color:var(--accent); }}
  td.warn {{ color:var(--warn); font-weight:700; }}
  .muted {{ color:var(--mut); }}
  .small {{ color:var(--mut); font-size:11px; }}
  .prog {{ min-width:160px; }}
  .bar {{ height:6px; background:#0c0e12; border-radius:4px; overflow:hidden; margin-bottom:3px; }}
  .fill {{ height:100%; background:linear-gradient(90deg,#3a6fd6,#6ea8fe); }}
  .badge {{ font-size:10px; padding:1px 7px; border-radius:20px; text-transform:uppercase;
           letter-spacing:.04em; margin-left:6px; vertical-align:middle; }}
  .badge.running {{ background:#173a2a; color:var(--good); }}
  .badge.done {{ background:#16233c; color:var(--accent); }}
  .badge.pending {{ background:#23252c; color:var(--mut); }}
  .note {{ color:var(--mut); font-size:12px; margin-top:18px; }}
  .findings ul {{ margin:0; padding:14px 22px 16px 38px; }}
  .findings li {{ margin:7px 0; }}
  .findings b {{ color:var(--fg); }}
  code {{ background:#0c0e12; padding:1px 5px; border-radius:4px; }}
</style></head>
<body><div class="wrap">
  <h1>CLINC150 — Gemma 4 26B-A4B quant comparison</h1>
  <p class="sub">Dataset <code>clinc_oos/plus/test</code> ({TARGET:,} items: 4,500 in-scope + 1,000 OOS)
     · thinking mode, free-form + label parse · sampling = model defaults · llama.cpp via llama-swap</p>
  <p class="upd">Updated {now}</p>
{findings}
  <h2>Accuracy</h2>
  <div class="card"><table>
    <thead><tr>
      <th>Quant</th><th>Progress</th><th>In-scope acc</th><th>OOS recall</th>
      <th>OOS prec</th><th>OOS F1</th><th>Overall acc</th><th>Macro-F1</th>
      <th>Err</th><th>Trunc</th>
    </tr></thead><tbody>{acc_rows}
    </tbody></table></div>

  <h2>Throughput &amp; tokens</h2>
  <div class="card"><table>
    <thead><tr>
      <th>Quant</th><th>tok/s / stream</th><th>tok/s aggregate</th><th>in tok</th>
      <th>cached</th><th>out tok</th><th>think tok</th><th>max out</th><th>lat mean</th><th>lat med</th>
    </tr></thead><tbody>{perf_rows}
    </tbody></table></div>

  <p class="note">
    In-scope accuracy is the headline metric (correct ÷ 4,500; predicting <code>oos</code>
    for an in-scope item is wrong). The test set is ordered in-scope first, OOS last, so
    <b>OOS metrics stay blank until a run passes item 4,500</b> and macro-F1 is artificially
    low mid-run (most labels unseen). Throughput: <code>tok/s / stream</code> = output tokens ÷
    request latency; <code>tok/s aggregate</code> ≈ that × concurrency (assumes all streams busy).
    Auto-refreshes every 60s.
  </p>
  <script>setTimeout(function(){{location.reload();}}, 60000);</script>
</div></body></html>"""


def main():
    html = build()
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write(html)
    os.replace(tmp, OUT)                     # atomic: readers never see a half-written page
    print(f"[build_html] wrote {OUT}")


if __name__ == "__main__":
    main()
