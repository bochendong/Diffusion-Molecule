#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P27_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P26_DIR="$PROJECT/experiments/p26_decoupled_joint_rl"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P24_CHECKPOINT="${P27_INPUT_ADAPTER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/checkpoint-7500}"
OUT="${P27_OUTPUT_ROOT:-$PROJECT/outputs/p27_p24_checkpoint_joint_rl/checkpoint_7500_seed_26001}"

export P26_SCRIPT_DIR="$P26_DIR"
export P26_OUTPUT_ROOT="$OUT"
export P26_TRAIN_JSONL="${P27_TRAIN_JSONL:-$P23/data/train.sft.jsonl}"
export P26_INPUT_ADAPTER="$P24_CHECKPOINT"
export P26_POLICY_OUTPUT_DIR="$OUT/model/p24_ckpt7500_rl"
export P26_TRAIN_SEED=26001
exec "$P26_DIR/run_train.sh"
