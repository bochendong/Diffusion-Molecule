#!/usr/bin/env bash
# Submit two paired de novo jobs and one CPU collector that also consumes P1 edit consistency.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${P2_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p2_validity_edit_repair_seed7}"
TWO_P_BASE="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
OOD_BASE="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_mixed_condition"
LOG_DIR="${P2_LOG_DIR:-$PROJECT_DIR/logs/p2_validity_edit_repair}"
GPU_REQUEST="${P2_GPU_REQUEST:-h100:1}"
EDIT_DEPENDENCY_JOB="${P2_EDIT_DEPENDENCY_JOB:-}"

mkdir -p "$OUTPUT_ROOT/data" "$LOG_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_p2_eval_subsets.py" \
  --two-p-seven-p-csv "$TWO_P_BASE/denovo_2p7p_eval_rows.csv" \
  --ood-csv "$OOD_BASE/denovo_ood_eval_rows.csv" \
  --output-dir "$OUTPUT_ROOT/data" \
  --per-property-count "${P2_PER_PROPERTY_COUNT:-64}" \
  --per-ood-bucket "${P2_PER_OOD_BUCKET:-100}" \
  --seed "${P2_EVAL_SEED:-20260823}"

gpu_jobs=()
for benchmark in two_p_to_seven_p ood; do
  job_id="$(sbatch --parsable --account="${P2_GPU_ACCOUNT:-def-hup-ab_gpu}" \
    --job-name="p2-valid-${benchmark}" --time="${P2_GPU_TIME:-01:15:00}" \
    --mem="${P2_GPU_MEM:-32G}" --cpus-per-task="${P2_GPU_CPUS:-4}" --gres="gpu:$GPU_REQUEST" \
    --output="$LOG_DIR/p2-valid-${benchmark}-%j.log" \
    --export="ALL,P2_BENCHMARK=$benchmark,P2_OUTPUT_ROOT=$OUTPUT_ROOT" \
    --wrap="bash '$SCRIPT_DIR/run_p2_denovo_validity_benchmark.sh'")"
  job_id="${job_id%%;*}"
  gpu_jobs+=("$job_id")
  echo "$benchmark=$job_id"
done

dependencies=("${gpu_jobs[@]}")
if [[ -n "$EDIT_DEPENDENCY_JOB" ]]; then
  dependencies+=("$EDIT_DEPENDENCY_JOB")
elif [[ ! -s "$PROJECT_DIR/outputs/p1_source_consistency_validity_pilot_v1/seed_7/p1_source_consistency_validity_metrics.csv" ]]; then
  echo "ERROR: set P2_EDIT_DEPENDENCY_JOB when the P1 edit-consistency metrics do not exist." >&2
  exit 2
fi
dependency="$(IFS=:; echo "${dependencies[*]}")"
final_job="$(sbatch --parsable --account="${P2_CPU_ACCOUNT:-def-hup-ab}" \
  --job-name=p2-repair-final --time=00:30:00 --mem=8G --cpus-per-task=2 \
  --dependency="afterok:$dependency" --output="$LOG_DIR/p2-repair-final-%j.log" \
  --export="ALL,P2_OUTPUT_ROOT=$OUTPUT_ROOT" \
  --wrap="bash '$SCRIPT_DIR/finalize_p2_validity_edit_repair.sh'")"
final_job="${final_job%%;*}"

echo "final=$final_job"
echo "dependency=$dependency"
echo "report=$OUTPUT_ROOT/final/p2_report.md"
