#!/usr/bin/env bash
# Submit a short 6p/7p n=20 kill test while the full P1 pools remain queued.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${SUCC_P1_FAST_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p1_fast_hard_6p7p_seed7}"
BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
LOG_DIR="${SUCC_P1_LOG_DIR:-$PROJECT_DIR/logs/p1_property_program_group_rl}"
GPU_REQUEST="${SUCC_P1_FAST_GPU:-h100:1}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_p1_fast_hard_subset.py" \
  --input-csv "$BASE_DIR/denovo_2p7p_eval_rows.csv" \
  --output-csv "$OUTPUT_ROOT/eval_6p7p_128_each.csv" \
  --manifest "$OUTPUT_ROOT/subset_manifest.json" --conditions-per-count 128 --selection-seed 20260823

job_ids=()
for arm in sft group_rl; do
  job_id="$(sbatch --parsable --account="${SUCC_P1_GPU_ACCOUNT:-def-hup-ab_gpu}" \
    --job-name="p1-fast-$arm" --time="${SUCC_P1_FAST_TIME:-01:00:00}" \
    --mem="${SUCC_P1_MEM:-24G}" --cpus-per-task="${SUCC_P1_CPUS:-4}" --gres="gpu:$GPU_REQUEST" \
    --output="$LOG_DIR/p1-fast-$arm-%j.log" --mail-user="${SUCC_P1_MAIL_USER:-dongbochen1218@gmail.com}" \
    --mail-type=END,FAIL --export="ALL,P1_FAST_ARM=$arm,SUCC_P1_FAST_OUTPUT_ROOT=$OUTPUT_ROOT" \
    --wrap="bash '$SCRIPT_DIR/run_p1_fast_hard_pool.sh'")"
  job_id="${job_id%%;*}"
  job_ids+=("$job_id")
  echo "$arm=$job_id"
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
final_job="$(sbatch --parsable --account="${SUCC_P1_CPU_ACCOUNT:-def-hup-ab}" \
  --job-name=p1-fast-final --time=00:30:00 --mem=8G --cpus-per-task=2 \
  --dependency="afterok:$dependency" --output="$LOG_DIR/p1-fast-final-%j.log" \
  --mail-user="${SUCC_P1_MAIL_USER:-dongbochen1218@gmail.com}" --mail-type=END,FAIL \
  --export="ALL,SUCC_P1_FAST_OUTPUT_ROOT=$OUTPUT_ROOT" \
  --wrap="bash '$SCRIPT_DIR/finalize_p1_fast_hard_gate.sh'")"
final_job="${final_job%%;*}"
echo "final=$final_job"
echo "dependency=$dependency"
echo "report=$OUTPUT_ROOT/final/report.md"
