#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P26_DIR="$PROJECT/experiments/p26_decoupled_joint_rl"
INPUT_ADAPTER="${P28_INPUT_ADAPTER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/checkpoint-11000}"
OUT="${P28_OUTPUT_ROOT:-$PROJECT/outputs/p28_p24_rl_method_sweep/checkpoint_11000_seed_28001}"
RL_ADAPTER="$OUT/model/editing_protected_pcgrad/checkpoint-020/adapter"
DEV_COMPARISON="$OUT/gate/dev/comparison_editing_protected_pcgrad.json"
LOG_DIR="$PROJECT/logs/p28_p24_rl_method_sweep"
mkdir -p "$LOG_DIR"

python - "$DEV_COMPARISON" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result.get("decision") != "PROMOTE_FULL_EVAL":
    raise SystemExit("ERROR: P28 protected RL did not pass the dev gate")
PY

baseline=$(sbatch --parsable --account=def-hup-ab --job-name=p28-fin-base --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --output="$LOG_DIR/final-baseline-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_GATE_SPLIT=final,P26_EVAL_TAG=p24_ckpt11000,P26_EVAL_ADAPTER="$INPUT_ADAPTER",P26_EVAL_REQUIRED="$INPUT_ADAPTER/adapter_model.safetensors" \
  "$P26_DIR/run_gate_eval.sh")
rl_eval=$(sbatch --parsable --account=def-hup-ab --job-name=p28-fin-edit --time=00:45:00 \
  --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 \
  --output="$LOG_DIR/final-protected-%j.log" \
  --export=ALL,P26_SCRIPT_DIR="$P26_DIR",P26_OUTPUT_ROOT="$OUT",P26_GATE_SPLIT=final,P26_EVAL_TAG=editing_protected_pcgrad,P26_EVAL_ADAPTER="$RL_ADAPTER",P26_EVAL_REQUIRED="$OUT/TRAIN_editing_protected_pcgrad_COMPLETE" \
  "$P26_DIR/run_gate_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p28-fin-cmp --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$baseline:$rl_eval" \
  --output="$LOG_DIR/final-compare-%j.log" --wrap \
  "/home/bdong/.venvs/molscribe_overlay/bin/python '$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py' --baseline '$OUT/gate/final/p24_ckpt11000/summary.json' --rl '$OUT/gate/final/editing_protected_pcgrad/summary.json' --output '$OUT/gate/final/comparison.json'")

printf 'final_baseline_job=%s\nfinal_rl_job=%s\nfinal_compare_job=%s\n' \
  "$baseline" "$rl_eval" "$compare"
