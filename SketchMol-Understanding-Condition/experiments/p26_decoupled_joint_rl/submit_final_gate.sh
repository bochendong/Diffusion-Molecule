#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P26_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
LOG_DIR="$PROJECT/logs/p26_decoupled_joint_rl"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
OUT="$PROJECT/outputs/p26_decoupled_joint_rl/seed_26001"
DEV_COMPARISON="$OUT/gate/dev/comparison.json"
for path in "$DEV_COMPARISON" "$P251/data/gates/final.jsonl" "$OUT/TRAIN_COMPLETE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P26 final-gate prerequisite: $path" >&2; exit 2; }
done
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("full_table_evaluation_authorized") else 2)' "$DEV_COMPARISON" \
  || { echo "ERROR: P26 dev gate did not authorize the final gate" >&2; exit 2; }
mkdir -p "$LOG_DIR"
baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p26-final-base --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --output="$LOG_DIR/final-baseline-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR",P26_GATE_SPLIT=final,P26_EVAL_TAG=baseline,P26_EVAL_ADAPTER="$P23/model/stage1_v2/adapter",P26_EVAL_REQUIRED="$P23/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$SCRIPT_DIR/run_gate_eval.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p26-final-rl --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --output="$LOG_DIR/final-rl-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR",P26_GATE_SPLIT=final,P26_EVAL_TAG=p26,P26_EVAL_ADAPTER="$OUT/model/p26/adapter",P26_EVAL_REQUIRED="$OUT/TRAIN_COMPLETE" \
  "$SCRIPT_DIR/run_gate_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p26-final-cmp --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline:$rl_eval" \
  --output="$LOG_DIR/final-compare-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$SCRIPT_DIR",P26_GATE_SPLIT=final "$SCRIPT_DIR/run_compare.sh")
printf 'final_baseline_job=%s\nfinal_rl_job=%s\nfinal_compare_job=%s\n' \
  "$baseline" "$rl_eval" "$compare"
