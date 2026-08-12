#!/usr/bin/env bash
# Submit the minimal three-stage paper replay smoke DAG on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_PAPER_REPLAY_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_paper_replay_smoke_v10/seed_1713}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_paper_replay_smoke_v10/seed_1713}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
GPU_ACCOUNT="${SUCC_UCA_GPU_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_PAPER_REPLAY_ROOT="$RUN_ROOT"

submit() {
  local output
  output="$(sbatch --parsable "$@")"
  printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

prepare_job="$(submit \
  --account="$CPU_ACCOUNT" \
  --job-name=uca-replay-v10-prep \
  --time=00:30:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,FAIL \
  --output="$LOG_DIR/%x-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_paper_replay_smoke_v10.sh' prepare")"

rank_job="$(submit \
  --account="$GPU_ACCOUNT" \
  --job-name=uca-replay-v10-rank \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --time=01:00:00 --cpus-per-task=4 --mem=32G \
  --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/%x-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_paper_replay_smoke_v10.sh' rank")"

score_job="$(submit \
  --account="$CPU_ACCOUNT" \
  --job-name=uca-replay-v10-score \
  --dependency="afterok:$rank_job" --kill-on-invalid-dep=yes \
  --time=01:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/%x-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_paper_replay_smoke_v10.sh' score")"

cat <<EOF
paper_replay_smoke_v10_submitted
prepare_job=$prepare_job
rank_job=$rank_job
score_job=$score_job
de_novo_conditions=60
table1_conditions=100
candidate_budget=20
requested_accelerator_hours=1.0
output=$RUN_ROOT/gate/summary.json
EOF
