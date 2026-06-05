#!/usr/bin/env bash
# Run the CLINC150 thinking-mode benchmark across all three Bartowski quants,
# sequentially (one model loaded at a time via llama-swap), scoring each as it
# finishes. Resumable: re-running skips items that already have a prediction, so
# a crash/restart (or a higher --max-tokens cleanup pass) picks up where it left
# off.
#
# Method: thinking enabled (model default), free-form generation, label parsed
# out of the final answer. Sampling = server defaults (no temperature override).
set -uo pipefail
cd "$(dirname "$0")/.."            # -> clinc150/
SCRIPTS=scripts
BASE=results
MAXTOK=4096

run() {
  local model="$1" dir="$2"
  echo "=== [$(date '+%F %T')] RUN $model -> $dir ==="
  python3 "$SCRIPTS/run_clinc.py" --model "$model" --out-dir "$BASE/$dir" \
      --no-grammar --max-tokens "$MAXTOK" --concurrency 4 --progress-every 100
  echo "=== [$(date '+%F %T')] SCORE $dir ==="
  python3 "$SCRIPTS/score_clinc.py" --in-dir "$BASE/$dir" --model-id "$dir"
}

run "Gemma-4-26B-A4B-IT-bartowski-Q8_0"   "bartowski-q8_0"
run "Gemma-4-26B-A4B-IT-bartowski-IQ4_XS" "bartowski-iq4_xs"
run "Gemma-4-26B-A4B-IT-bartowski-IQ2_M"  "bartowski-iq2_m"
echo "=== [$(date '+%F %T')] ALL DONE ==="
