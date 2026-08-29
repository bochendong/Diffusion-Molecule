#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P303_SCRIPT_DIR:?P303_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P26="$PROJECT/experiments/p26_decoupled_joint_rl"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
OUT="${P303_OUTPUT_ROOT:-$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301}"
TAG="${P303_EDIT_TAG:?P303_EDIT_TAG must be exported}"
case "$TAG" in
  baseline) ADAPTER="$P24/alignment_refresh/model/adapter"; REQUIRED="$P24/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE" ;;
  refined) ADAPTER="$OUT/model/refined/adapter"; REQUIRED="$OUT/TRAIN_COMPLETE" ;;
  *) echo "ERROR: unsupported P303_EDIT_TAG=$TAG" >&2; exit 2 ;;
esac
export P26_SCRIPT_DIR="$P26" P26_OUTPUT_ROOT="$OUT/edit_eval" P26_GATE_SPLIT=final
export P26_EVAL_TAG="$TAG" P26_EVAL_ADAPTER="$ADAPTER" P26_EVAL_REQUIRED="$REQUIRED"
exec "$P26/run_gate_eval.sh"
