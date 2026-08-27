#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P251_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P251_OUTPUT_ROOT:-$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125}"
LOG_DIR="$PROJECT/logs/p25_1_p23_mode_paired_grpo"
[[ -f "$OUT/GATE_dev_COMPLETE" ]] || { echo "ERROR: dev gate is incomplete" >&2; exit 2; }
decision=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$OUT/gate/dev/comparison.json")
[[ "$decision" == "PROMOTE_FULL_EVAL" ]] || { echo "ERROR: dev gate decision is $decision" >&2; exit 3; }
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p251-final-base --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --output="$LOG_DIR/final-baseline-%j.log" \
  --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR",P251_GATE_SPLIT=final,P251_EVAL_METHOD=baseline "$SCRIPT_DIR/run_eval.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p251-final-rl --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --output="$LOG_DIR/final-rl-%j.log" \
  --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR",P251_GATE_SPLIT=final,P251_EVAL_METHOD=rl "$SCRIPT_DIR/run_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p251-final-cmp --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$baseline:$rl_eval" --output="$LOG_DIR/final-compare-%j.log" \
  --export=ALL,P251_SCRIPT_DIR="$SCRIPT_DIR",P251_GATE_SPLIT=final "$SCRIPT_DIR/run_compare.sh")
printf 'final_baseline_job=%s\nfinal_rl_job=%s\nfinal_compare_job=%s\n' "$baseline" "$rl_eval" "$compare"
