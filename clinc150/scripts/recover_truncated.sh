#!/usr/bin/env bash
# Recover truncated items. The items that hit the 4096-token cap have empty
# predictions, so run_clinc's resume logic re-processes exactly those (and only
# those) when pointed at the same out-dir. Re-run them with a larger 8192-token
# budget, then rescore. Sequential per quant via llama-swap. Idempotent: any item
# that still truncates at 8192 simply stays empty and can be retried again.
set -uo pipefail
cd "$(dirname "$0")/.."                 # -> clinc150/
CAP=8192

run() {
  echo "=== [$(date '+%F %T')] RECOVER $2 (cap $CAP) ==="
  python3 scripts/run_clinc.py --model "$1" --out-dir "results/$2" \
      --no-grammar --max-tokens "$CAP" --concurrency 4 --progress-every 25
  python3 scripts/score_clinc.py --in-dir "results/$2" --model-id "$2"
}

run "Gemma-4-26B-A4B-IT-bartowski-Q8_0"   "bartowski-q8_0"
run "Gemma-4-26B-A4B-IT-bartowski-IQ4_XS" "bartowski-iq4_xs"
run "Gemma-4-26B-A4B-IT-bartowski-IQ2_M"  "bartowski-iq2_m"
python3 scripts/build_html.py
echo "=== [$(date '+%F %T')] RECOVERY DONE ==="
