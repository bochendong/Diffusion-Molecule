#!/usr/bin/env bash
# Resume v14a from its completed adapter, with memory-safe LLM scoring and CPU shards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_FEEDBACK_REPAIR_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_feedback_repair_v14a/seed_1716}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_feedback_repair_v14a/seed_1716_resume}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
GPU_ACCOUNT="${SUCC_UCA_GPU_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LLM_SHARD_COUNT="${SUCC_UCA_FEEDBACK_REPAIR_LLM_SHARD_COUNT:-2}"
DET_SHARD_COUNT="${SUCC_UCA_FEEDBACK_REPAIR_DET_SHARD_COUNT:-8}"
SCORE_BATCH_SIZE="${SUCC_UCA_FEEDBACK_SCORE_BATCH_SIZE:-4}"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
[[ -s "$RUN_ROOT/controller/model/adapter/adapter_model.safetensors" \
  && -s "$RUN_ROOT/controller/model/training_summary.json" \
  && -s "$RUN_ROOT/controller/validation/baseline/summary.json" \
  && -s "$RUN_ROOT/controller/validation/candidate/summary.json" ]] || {
  echo "ERROR: completed v14a training artifacts are unavailable" >&2; exit 2;
}
[[ "$LLM_SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid LLM shard count" >&2; exit 2; }
[[ "$DET_SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid deterministic shard count" >&2; exit 2; }
(( LLM_SHARD_COUNT * 5 <= 12 )) || { echo "ERROR: LLM signal exceeds 6 requested MIG-hours" >&2; exit 2; }
[[ "$SCORE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid score batch size" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$RUN_ROOT/llm/shards" "$RUN_ROOT/deterministic/shards"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_FEEDBACK_REPAIR_ROOT="$RUN_ROOT"

submit() {
  local output
  output="$(sbatch --parsable "$@")"
  printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

llm_array_end="$((LLM_SHARD_COUNT - 1))"
det_array_end="$((DET_SHARD_COUNT - 1))"
gpu_array="$(submit --account="$GPU_ACCOUNT" --job-name=uca-feedback-v14a-llm-resume \
  --array="0-${llm_array_end}%${LLM_SHARD_COUNT}" \
  --time=02:30:00 --cpus-per-task=4 --mem=32G --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL,SUCC_UCA_FEEDBACK_REPAIR_SHARD_COUNT="$LLM_SHARD_COUNT",SUCC_UCA_FEEDBACK_SCORE_BATCH_SIZE="$SCORE_BATCH_SIZE" \
  --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' gpu_generate")"
det_array="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-det-resume \
  --array="0-${det_array_end}%${DET_SHARD_COUNT}" \
  --time=01:00:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL,SUCC_UCA_FEEDBACK_REPAIR_SHARD_COUNT="$DET_SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' deterministic")"
llm_merge="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-llm-merge \
  --dependency="afterok:$gpu_array" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL,SUCC_UCA_FEEDBACK_REPAIR_SHARD_COUNT="$LLM_SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' merge_llm")"
det_merge="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-det-merge \
  --dependency="afterok:$det_array" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL,SUCC_UCA_FEEDBACK_REPAIR_SHARD_COUNT="$DET_SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' merge_deterministic")"
llm_oracle="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-llo-resume \
  --dependency="afterok:$llm_merge" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' oracle_llm")"
det_oracle="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-deo-resume \
  --dependency="afterok:$det_merge" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' oracle_deterministic")"
gate="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-gate-resume \
  --dependency="afterok:$llm_oracle:$det_oracle" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' gate")"

cat <<EOF
feedback_repair_v14a_resume_submitted
llm_gpu_array_job=$gpu_array
deterministic_array_job=$det_array
llm_merge_job=$llm_merge
deterministic_merge_job=$det_merge
llm_oracle_job=$llm_oracle
deterministic_oracle_job=$det_oracle
gate_job=$gate
reused_adapter=$RUN_ROOT/controller/model/adapter
score_batch_size=$SCORE_BATCH_SIZE
llm_shard_count=$LLM_SHARD_COUNT
deterministic_shard_count=$DET_SHARD_COUNT
signal_conditions=200
generation_attempts_per_condition=20
output_selection=none
evaluation_target_access=false
requested_accelerator_hours=5.0
output=$RUN_ROOT/gate/summary.json
EOF
