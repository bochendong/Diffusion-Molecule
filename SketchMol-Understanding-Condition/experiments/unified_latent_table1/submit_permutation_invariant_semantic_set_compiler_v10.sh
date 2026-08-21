#!/usr/bin/env bash
# Submit one bounded V10 GPU execution and a separate CPU science gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_V10_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/permutation_invariant_semantic_set_compiler_v10}"
MAIL_USER="${SUCC_V10_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_V10_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_V10_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

execution_job_id="$(sbatch --parsable \
  --job-name=uca-setcomp-v10-exec \
  --account="$ACCOUNT" --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-setcomp-v10-exec-%j.log" \
  --wrap="SUCC_V10_STAGE=execute bash '$SCRIPT_DIR/run_permutation_invariant_semantic_set_compiler_v10.sh'")"

gate_job_id="$(sbatch --parsable \
  --job-name=uca-setcomp-v10-scigate \
  --account="$ACCOUNT" --time=00:05:00 --cpus-per-task=1 --mem=2G \
  --dependency="afterok:$execution_job_id" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-setcomp-v10-scigate-%j.log" \
  --wrap="SUCC_V10_STAGE=gate bash '$SCRIPT_DIR/run_permutation_invariant_semantic_set_compiler_v10.sh'")"

echo "execution_job_id=$execution_job_id"
echo "gate_job_id=$gate_job_id"
echo "execution_contract=engineering_success_exits_zero"
echo "science_gate_contract=scientific_stop_is_a_complete_artifact"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/permutation_invariant_semantic_set_compiler_v10/seed_2151/gate/gate_summary.json"
