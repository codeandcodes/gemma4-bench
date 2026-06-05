#!/usr/bin/env bash
# Live publisher: every INTERVAL seconds, re-score each quant and rebuild
# dashboard.html while the benchmark runs. Reads the append-only inference.jsonl
# files (the scorer tolerates a partial trailing line), so it is safe to run
# alongside run_thinking_all.sh. Writes are atomic. Exits after a final rebuild
# once the benchmark driver is no longer running.
#
# Usage: bash scripts/publish_loop.sh [interval_seconds]   (default 120)
set -uo pipefail
cd "$(dirname "$0")/.."                 # -> clinc150/
INTERVAL="${1:-120}"

publish() {
  for d in results/*/; do
    [ -f "${d}inference.jsonl" ] || continue
    python3 scripts/score_clinc.py --in-dir "$d" --concurrency 4 >/dev/null 2>&1 || true
  done
  python3 scripts/build_html.py >/dev/null 2>&1 || true
}

echo "[publish] starting; interval=${INTERVAL}s -> dashboard.html"
while true; do
  publish
  echo "[publish] $(date '+%F %T') refreshed"
  if ! pgrep -f run_thinking_all.sh >/dev/null 2>&1; then
    echo "[publish] $(date '+%F %T') benchmark driver gone — final rebuild and exit"
    publish
    break
  fi
  sleep "$INTERVAL"
done
