#!/usr/bin/env bash
# Submit the first constraint-factorized pilot: fixed n=20, joint-success and worst-property reward.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${UMTP_JOINT_BOTTLENECK_SEED:-7}"
export UMTP_RL_PILOT_SEED="$SEED"
export UMTP_RL_PILOT_OUTPUT_ROOT="${UMTP_JOINT_BOTTLENECK_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/umtp_joint_bottleneck_pilot_v1/seed_${SEED}}"
export UMTP_RL_PILOT_REWARD_AGGREGATION=joint_bottleneck
export UMTP_RL_PILOT_REWARD_JOINT_BONUS_WEIGHT="${UMTP_JOINT_BONUS_WEIGHT:-2.0}"
export UMTP_RL_PILOT_REWARD_BOTTLENECK_WEIGHT="${UMTP_BOTTLENECK_WEIGHT:-0.5}"
export UMTP_RL_PILOT_TRAIN_ROWS_PER_GROUP="${UMTP_JOINT_BOTTLENECK_TRAIN_ROWS_PER_GROUP:-32}"
export UMTP_RL_PILOT_VALIDATION_ROWS_PER_GROUP="${UMTP_JOINT_BOTTLENECK_VALIDATION_ROWS_PER_GROUP:-4}"
export UMTP_RL_PILOT_EVAL_CANDIDATES="${UMTP_JOINT_BOTTLENECK_EVAL_CANDIDATES:-20}"
export UMTP_RL_PILOT_ROLLOUTS="${UMTP_JOINT_BOTTLENECK_ROLLOUTS:-8}"
export UMTP_RL_PILOT_SLURM_JOB_NAME="umtp-joint-bottleneck-s${SEED}"
export UMTP_RL_PILOT_SLURM_MAIL_USER="${UMTP_RL_PILOT_SLURM_MAIL_USER:-dongbochen1218@gmail.com}"
export SUCC_LOG_DIR="${SUCC_LOG_DIR:-SketchMol-Understanding-Condition/logs/umtp_joint_bottleneck_pilot_v1}"

bash "$SCRIPT_DIR/submit_umtp_v1_rl_pilot.sh"
