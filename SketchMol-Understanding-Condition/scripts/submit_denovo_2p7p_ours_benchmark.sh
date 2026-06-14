#!/usr/bin/env bash
# Submit the trained SUCC UniVideo de novo 2p-7p benchmark as a Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_DENOVO_OUTPUT_DIR="${SUCC_DENOVO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_v1}"
export SUCC_DENOVO_MODEL_OUTPUT_DIR="${SUCC_DENOVO_MODEL_OUTPUT_DIR:-${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v3_attack}}"
export SUCC_DENOVO_RESUME_CHECKPOINT="${SUCC_DENOVO_RESUME_CHECKPOINT:-${SUCC_RESUME_CHECKPOINT:-$SUCC_DENOVO_MODEL_OUTPUT_DIR/univideo_molecule/univideo_molecule_generation.pt}}"
export SUCC_DENOVO_MOLECULE_DB_CSV="${SUCC_DENOVO_MOLECULE_DB_CSV:-$SMMED_DEFAULT_MOLECULE_DB}"
export SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT="${SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT:-1000}"
export SUCC_DENOVO_MATERIALIZED_METHODS="${SUCC_DENOVO_MATERIALIZED_METHODS:-latent_nearest}"
export SUCC_DENOVO_PROPERTY_RERANK_CANDIDATES="${SUCC_DENOVO_PROPERTY_RERANK_CANDIDATES:-4096}"
export SUCC_DENOVO_PROPERTY_RERANK_WEIGHT="${SUCC_DENOVO_PROPERTY_RERANK_WEIGHT:-10}"
export SUCC_DENOVO_STRICT_RERANK_WEIGHT="${SUCC_DENOVO_STRICT_RERANK_WEIGHT:-100}"
export SUCC_DENOVO_LATENT_RERANK_WEIGHT="${SUCC_DENOVO_LATENT_RERANK_WEIGHT:-1}"
export SUCC_DENOVO_RUN_FEATURE_EXPORT="${SUCC_DENOVO_RUN_FEATURE_EXPORT:-auto}"
export SUCC_DENOVO_USE_CONDITION_FEATURES="${SUCC_DENOVO_USE_CONDITION_FEATURES:-1}"
export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"

ACCOUNT="${SUCC_DENOVO_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
TIME="${SUCC_DENOVO_SLURM_TIME:-${SUCC_SLURM_TIME:-08:00:00}}"
MEM="${SUCC_DENOVO_SLURM_MEM:-${SUCC_SLURM_MEM:-80G}}"
CPUS="${SUCC_DENOVO_SLURM_CPUS:-${SUCC_SLURM_CPUS:-8}}"
JOB_NAME="${SUCC_DENOVO_SLURM_JOB_NAME:-succ-denovo-2p7p-ours}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
PARTITION="${SUCC_DENOVO_SLURM_PARTITION:-${SUCC_SLURM_PARTITION:-}}"
GPU_PROFILE="${SUCC_DENOVO_GPU_PROFILE:-${SUCC_GPU_PROFILE:-h100_40gb_mig}}"
DEPENDENCY="${SUCC_DENOVO_SLURM_DEPENDENCY:-}"

if [[ ! -x "$SUCC_PYTHON_BIN" ]]; then
  echo "ERROR: SUCC_PYTHON_BIN is not executable: $SUCC_PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$SUCC_DENOVO_MOLECULE_DB_CSV" ]]; then
  echo "ERROR: missing molecule database: $SUCC_DENOVO_MOLECULE_DB_CSV" >&2
  exit 2
fi
if [[ ! -f "$SUCC_DENOVO_RESUME_CHECKPOINT" && -z "$DEPENDENCY" ]]; then
  echo "ERROR: missing trained SUCC checkpoint: $SUCC_DENOVO_RESUME_CHECKPOINT" >&2
  exit 2
fi

if [[ -n "${SUCC_DENOVO_SLURM_GPUS:-}" ]]; then
  GPU_CANDIDATES=("$SUCC_DENOVO_SLURM_GPUS")
elif [[ "$GPU_PROFILE" == "none" || "$GPU_PROFILE" == "0" ]]; then
  GPU_CANDIDATES=("")
elif [[ "$GPU_PROFILE" == "h100_10gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_1g.10gb:1")
elif [[ "$GPU_PROFILE" == "h100_20gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_40gb_mig" ]]; then
  GPU_CANDIDATES=("nvidia_h100_80gb_hbm3_3g.40gb:1" "h100:1" "a100:1" "nvidia_h100_80gb_hbm3_2g.20gb:1")
elif [[ "$GPU_PROFILE" == "h100_full" ]]; then
  GPU_CANDIDATES=("h100:1")
elif [[ "$GPU_PROFILE" == "a100" ]]; then
  GPU_CANDIDATES=("a100:1")
elif [[ "$GPU_PROFILE" == "t4" ]]; then
  GPU_CANDIDATES=("t4:1")
else
  GPU_CANDIDATES=("$GPU_PROFILE")
fi

mkdir -p "$LOG_DIR"

echo "Submitting SUCC UniVideo de novo 2p-7p benchmark"
echo "  account=$ACCOUNT"
echo "  time=$TIME"
echo "  mem=$MEM"
echo "  cpus=$CPUS"
echo "  python=$SUCC_PYTHON_BIN"
echo "  output_dir=$SUCC_DENOVO_OUTPUT_DIR"
echo "  model_output_dir=$SUCC_DENOVO_MODEL_OUTPUT_DIR"
echo "  resume_checkpoint=$SUCC_DENOVO_RESUME_CHECKPOINT"
echo "  molecule_db=$SUCC_DENOVO_MOLECULE_DB_CSV"
echo "  rows_per_property_count=$SUCC_DENOVO_ROWS_PER_PROPERTY_COUNT"
echo "  latent_backend=$SUCC_LATENT_BACKEND"
echo "  methods=$SUCC_DENOVO_MATERIALIZED_METHODS"
echo "  property_rerank_candidates=$SUCC_DENOVO_PROPERTY_RERANK_CANDIDATES"
echo "  property_rerank_weight=$SUCC_DENOVO_PROPERTY_RERANK_WEIGHT"
echo "  strict_rerank_weight=$SUCC_DENOVO_STRICT_RERANK_WEIGHT"
echo "  latent_rerank_weight=$SUCC_DENOVO_LATENT_RERANK_WEIGHT"
echo "  gpu_candidates=${GPU_CANDIDATES[*]:-none}"
if [[ -n "$DEPENDENCY" ]]; then
  echo "  dependency=$DEPENDENCY"
fi

SBATCH_ARGS=(
  --account="$ACCOUNT"
  --job-name="$JOB_NAME"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
if [[ -n "$PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [[ -n "$DEPENDENCY" ]]; then
  SBATCH_ARGS+=(--dependency="$DEPENDENCY")
fi

job_id=""
for GPU_REQUEST in "${GPU_CANDIDATES[@]}"; do
  if [[ -n "$GPU_REQUEST" ]]; then
    echo "Trying sbatch with --gpus=$GPU_REQUEST"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --gpus="$GPU_REQUEST" --wrap="bash '$PROJECT_DIR/scripts/run_denovo_2p7p_ours_benchmark.sh'")"; then
      continue
    fi
  else
    echo "Trying sbatch without GPU request"
    if ! output="$(sbatch "${SBATCH_ARGS[@]}" --wrap="bash '$PROJECT_DIR/scripts/run_denovo_2p7p_ours_benchmark.sh'")"; then
      continue
    fi
  fi
  echo "$output"
  job_id="$(echo "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  if [[ -n "$job_id" ]]; then
    break
  fi
done

if [[ -z "$job_id" ]]; then
  echo "ERROR: failed to submit SUCC de novo benchmark." >&2
  exit 1
fi

echo
echo "SUCC de novo 2p-7p benchmark submitted."
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/${JOB_NAME}-${job_id}.log"
echo "  benchmark_report=$SUCC_DENOVO_OUTPUT_DIR/benchmark_ours/benchmark_report.md"
