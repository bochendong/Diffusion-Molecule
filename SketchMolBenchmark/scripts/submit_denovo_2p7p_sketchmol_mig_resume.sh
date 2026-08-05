#!/usr/bin/env bash
# Resume the official SketchMol de novo 2p-7p @40 benchmark on H100 MIGs.
#
# Existing completed shards are preserved. The remaining contiguous shard
# range is split across 20 GB and 40 GB MIG arrays, then one CPU evaluator
# joins all 600 shards and sends the final notification.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ACCOUNT="${SKETCHMOL_SLURM_ACCOUNT:-def-hup-ab}"
MAIL_USER="${SKETCHMOL_MAIL_USER:-dongbochen1218@gmail.com}"
LOG_DIR="${SKETCHMOL_LOG_DIR:-SketchMolBenchmark/logs}"
OUTPUT_DIR="${SKETCHMOL_DENOVO_OUTPUT_DIR:-SketchMolBenchmark/outputs/sketchmol_denovo_2p7p_at40_v1}"
EVAL_CSV="${SKETCHMOL_DENOVO_EVAL_CSV:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv}"
SHARD_COUNT="${SKETCHMOL_DENOVO_SHARD_COUNT:-600}"
START_SHARD="${SKETCHMOL_DENOVO_START_SHARD:-271}"
SPLIT_SHARD="${SKETCHMOL_DENOVO_MIG_SPLIT_SHARD:-435}"
END_SHARD="${SKETCHMOL_DENOVO_END_SHARD:-599}"
ARRAY_CONCURRENCY="${SKETCHMOL_DENOVO_ARRAY_CONCURRENCY:-48}"
GPU_TIME="${SKETCHMOL_DENOVO_GPU_TIME:-03:00:00}"

mkdir -p "$LOG_DIR"

if (( START_SHARD < 0 || START_SHARD > SPLIT_SHARD || SPLIT_SHARD > END_SHARD + 1 )); then
  echo "ERROR: invalid shard ranges: start=$START_SHARD split=$SPLIT_SHARD end=$END_SHARD" >&2
  exit 2
fi
if (( END_SHARD >= SHARD_COUNT )); then
  echo "ERROR: end shard $END_SHARD exceeds shard count $SHARD_COUNT" >&2
  exit 2
fi

common_exports=(
  SKETCHMOL_DENOVO_EVAL_CSV="$EVAL_CSV"
  SKETCHMOL_DENOVO_OUTPUT_DIR="$OUTPUT_DIR"
  SKETCHMOL_DENOVO_SHARD_COUNT="$SHARD_COUNT"
  SKETCHMOL_DENOVO_CONDITIONAL_COUNT=40
  SKETCHMOL_DENOVO_RESUME=1
  SKETCHMOL_MOLSCRIBE_PYTHON_BIN=/home/bdong/.venvs/molscribe_torch251/bin/python
  SKETCHMOL_MOLSCRIBE_BATCH_SIZE=8
  SKETCHMOL_MOLSCRIBE_BACKEND=custom
  SKETCHMOL_ONMT_OVERLAY=/scratch/bdong/python_overlays/onmt220
  SUCC_ONMT_OVERLAY=/scratch/bdong/python_overlays/onmt220
  SUCC_PREFER_PIP_MOLSCRIBE=1
  SUCC_MOLSCRIBE_PIP_OVERLAY=/scratch/bdong/python_overlays/molscribe111
)

export_csv="ALL"
for pair in "${common_exports[@]}"; do
  export_csv+=",$pair"
done

begin_id="$(
  sbatch --parsable \
    --account="$ACCOUNT" \
    --job-name=sketchmol-2p7p-mig-begin \
    --time=00:05:00 \
    --mem=1G \
    --cpus-per-task=1 \
    --mail-user="$MAIL_USER" \
    --mail-type=BEGIN,END,FAIL \
    --output="$LOG_DIR/sketchmol-2p7p-mig-begin-%j.log" \
    --wrap="echo 'SketchMol de novo 2p-7p @40 MIG resume submitted'"
)"

job_ids=()
if (( START_SHARD < SPLIT_SHARD )); then
  end_20=$((SPLIT_SHARD - 1))
  id_20="$(
    sbatch --parsable \
      --account="$ACCOUNT" \
      --job-name=sketchmol-2p7p-20g \
      --array="${START_SHARD}-${end_20}%${ARRAY_CONCURRENCY}" \
      --time="$GPU_TIME" \
      --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --mail-user="$MAIL_USER" \
      --mail-type=FAIL \
      --output="$LOG_DIR/sketchmol-2p7p-20g-%A_%a.log" \
      --export="${export_csv},SKETCHMOL_DENOVO_SAMPLE_BATCH_SIZE=5" \
      --wrap="SKETCHMOL_DENOVO_SHARD_INDEX=\$SLURM_ARRAY_TASK_ID bash 'SketchMolBenchmark/scripts/run_denovo_2p7p_sketchmol_shard.sh'"
  )"
  job_ids+=("$id_20")
  echo "Submitted 20GB MIG array: $id_20 shards=$START_SHARD-$end_20"
fi

if (( SPLIT_SHARD <= END_SHARD )); then
  id_40="$(
    sbatch --parsable \
      --account="$ACCOUNT" \
      --job-name=sketchmol-2p7p-40g \
      --array="${SPLIT_SHARD}-${END_SHARD}%${ARRAY_CONCURRENCY}" \
      --time="$GPU_TIME" \
      --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 \
      --cpus-per-task=4 \
      --mem=32G \
      --mail-user="$MAIL_USER" \
      --mail-type=FAIL \
      --output="$LOG_DIR/sketchmol-2p7p-40g-%A_%a.log" \
      --export="${export_csv},SKETCHMOL_DENOVO_SAMPLE_BATCH_SIZE=20" \
      --wrap="SKETCHMOL_DENOVO_SHARD_INDEX=\$SLURM_ARRAY_TASK_ID bash 'SketchMolBenchmark/scripts/run_denovo_2p7p_sketchmol_shard.sh'"
  )"
  job_ids+=("$id_40")
  echo "Submitted 40GB MIG array: $id_40 shards=$SPLIT_SHARD-$END_SHARD"
fi

if (( ${#job_ids[@]} == 0 )); then
  echo "ERROR: no GPU arrays were submitted" >&2
  exit 2
fi

dependency="afterok:$(IFS=:; echo "${job_ids[*]}")"
eval_id="$(
  sbatch --parsable \
    --account="$ACCOUNT" \
    --job-name=sketchmol-2p7p-eval \
    --time=01:00:00 \
    --mem=16G \
    --cpus-per-task=2 \
    --dependency="$dependency" \
    --mail-user="$MAIL_USER" \
    --mail-type=BEGIN,END,FAIL \
    --output="$LOG_DIR/sketchmol-2p7p-eval-%j.log" \
    --export="$export_csv" \
    --wrap="bash 'SketchMolBenchmark/scripts/run_denovo_2p7p_sketchmol_eval.sh'"
)"

echo "BEGIN_JOB=$begin_id"
echo "GPU_ARRAYS=${job_ids[*]}"
echo "EVAL_JOB=$eval_id dependency=$dependency"
echo "OUTPUT_DIR=$OUTPUT_DIR"
