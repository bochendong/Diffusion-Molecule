#!/usr/bin/env bash
# Submit the SketchMol de novo 2p-7p OCR benchmark as a wave chain on Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

SKETCHMOL_DENOVO_EVAL_CSV="${SKETCHMOL_DENOVO_EVAL_CSV:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv}"
SKETCHMOL_DENOVO_OUTPUT_DIR="${SKETCHMOL_DENOVO_OUTPUT_DIR:-SketchMolBenchmark/outputs/sketchmol_denovo_2p7p_at40_v1}"
SKETCHMOL_DENOVO_SHARD_COUNT="${SKETCHMOL_DENOVO_SHARD_COUNT:-600}"
SKETCHMOL_DENOVO_START_SHARD="${SKETCHMOL_DENOVO_START_SHARD:-0}"
SKETCHMOL_DENOVO_SUBMIT_MODE="${SKETCHMOL_DENOVO_SUBMIT_MODE:-waves}"
SKETCHMOL_DENOVO_WAVE_SHARD_COUNT="${SKETCHMOL_DENOVO_WAVE_SHARD_COUNT:-22}"
SKETCHMOL_DENOVO_WAVE_HOURS="${SKETCHMOL_DENOVO_WAVE_HOURS:-11:45:00}"
SKETCHMOL_DENOVO_SUBMIT_EVAL="${SKETCHMOL_DENOVO_SUBMIT_EVAL:-1}"
SKETCHMOL_DENOVO_SLURM_JOB_NAME="${SKETCHMOL_DENOVO_SLURM_JOB_NAME:-sketchmol-denovo-2p7p}"
SKETCHMOL_LOG_DIR="${SKETCHMOL_LOG_DIR:-SketchMolBenchmark/logs}"
SKETCHMOL_SLURM_ACCOUNT="${SKETCHMOL_SLURM_ACCOUNT:-def-hup-ab}"
SKETCHMOL_SLURM_MEM="${SKETCHMOL_SLURM_MEM:-24G}"
SKETCHMOL_SLURM_CPUS="${SKETCHMOL_SLURM_CPUS:-4}"
SKETCHMOL_SLURM_GPUS="${SKETCHMOL_SLURM_GPUS:-h100}"

mkdir -p "$SKETCHMOL_LOG_DIR"

if (( SKETCHMOL_DENOVO_START_SHARD < 0 || SKETCHMOL_DENOVO_START_SHARD >= SKETCHMOL_DENOVO_SHARD_COUNT )); then
  echo "ERROR: invalid SKETCHMOL_DENOVO_START_SHARD=$SKETCHMOL_DENOVO_START_SHARD" >&2
  exit 2
fi

remaining=$((SKETCHMOL_DENOVO_SHARD_COUNT - SKETCHMOL_DENOVO_START_SHARD))
wave_count=$(((remaining + SKETCHMOL_DENOVO_WAVE_SHARD_COUNT - 1) / SKETCHMOL_DENOVO_WAVE_SHARD_COUNT))

echo "Submitting SketchMol de novo 2p-7p benchmark"
echo "  eval_csv=$SKETCHMOL_DENOVO_EVAL_CSV"
echo "  output_dir=$SKETCHMOL_DENOVO_OUTPUT_DIR"
echo "  shard_count=$SKETCHMOL_DENOVO_SHARD_COUNT"
echo "  start_shard=$SKETCHMOL_DENOVO_START_SHARD"
echo "  submit_mode=$SKETCHMOL_DENOVO_SUBMIT_MODE"
echo "  wave_shard_count=$SKETCHMOL_DENOVO_WAVE_SHARD_COUNT"
echo "  wave_hours=$SKETCHMOL_DENOVO_WAVE_HOURS"
echo "  wave_count=$wave_count"
echo "  submit_eval=$SKETCHMOL_DENOVO_SUBMIT_EVAL"

if [[ "$SKETCHMOL_DENOVO_SUBMIT_MODE" != "waves" ]]; then
  echo "ERROR: only waves submit mode is supported (got: $SKETCHMOL_DENOVO_SUBMIT_MODE)" >&2
  exit 2
fi

prev_job_id=""
last_gpu_job_id=""

for ((wave_idx = 0; wave_idx < wave_count; wave_idx++)); do
  serial_start=$((SKETCHMOL_DENOVO_START_SHARD + wave_idx * SKETCHMOL_DENOVO_WAVE_SHARD_COUNT))
  serial_end=$((serial_start + SKETCHMOL_DENOVO_WAVE_SHARD_COUNT - 1))
  if (( serial_end >= SKETCHMOL_DENOVO_SHARD_COUNT )); then
    serial_end=$((SKETCHMOL_DENOVO_SHARD_COUNT - 1))
  fi

  wave_num=$((wave_idx + 1))
  job_name="${SKETCHMOL_DENOVO_SLURM_JOB_NAME}-w${wave_num}"
  sbatch_args=(
    --parsable
    --account="$SKETCHMOL_SLURM_ACCOUNT"
    --job-name="$job_name"
    --time="$SKETCHMOL_DENOVO_WAVE_HOURS"
    --mem="$SKETCHMOL_SLURM_MEM"
    --cpus-per-task="$SKETCHMOL_SLURM_CPUS"
    --gpus="$SKETCHMOL_SLURM_GPUS"
    --output="$SKETCHMOL_LOG_DIR/${job_name}-%j.log"
    --export=ALL,SKETCHMOL_DENOVO_SERIAL_START="$serial_start",SKETCHMOL_DENOVO_SERIAL_END="$serial_end"
    --wrap="bash 'SketchMolBenchmark/scripts/run_denovo_2p7p_sketchmol_shard_serial.sh'"
  )
  if [[ -n "$prev_job_id" ]]; then
    sbatch_args+=(--dependency="afterok:${prev_job_id}")
  fi

  job_id="$(sbatch "${sbatch_args[@]}")"
  echo "Submitted wave ${wave_num}/${wave_count}: job=${job_id} shards=${serial_start}..${serial_end}"
  prev_job_id="$job_id"
  last_gpu_job_id="$job_id"
done

if [[ "$SKETCHMOL_DENOVO_SUBMIT_EVAL" == "1" && -n "$last_gpu_job_id" ]]; then
  eval_job_name="${SKETCHMOL_DENOVO_SLURM_JOB_NAME}-eval"
  eval_job_id="$(
    sbatch --parsable \
      --account="$SKETCHMOL_SLURM_ACCOUNT" \
      --job-name="$eval_job_name" \
      --time="01:00:00" \
      --mem="16G" \
      --cpus-per-task="2" \
      --dependency="afterok:${last_gpu_job_id}" \
      --output="$SKETCHMOL_LOG_DIR/${eval_job_name}-%j.log" \
      --export=ALL \
      --wrap="bash 'SketchMolBenchmark/scripts/run_denovo_2p7p_sketchmol_eval.sh'"
  )"
  echo "Submitted eval job: $eval_job_id (depends on $last_gpu_job_id)"
fi
