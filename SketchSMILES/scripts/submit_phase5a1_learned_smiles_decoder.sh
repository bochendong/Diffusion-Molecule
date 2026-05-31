#!/usr/bin/env bash
# Submit Phase 5A-1 oracle-conditioned learned SMILES decoder to Slurm.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACCOUNT="${SKETCHSMILES_SLURM_ACCOUNT:-def-hup-ab}"
TIME="${SKETCHSMILES_SLURM_TIME:-04:00:00}"
MEM="${SKETCHSMILES_SLURM_MEM:-32G}"
CPUS="${SKETCHSMILES_SLURM_CPUS:-4}"
GPU_REQUEST="${SKETCHSMILES_SLURM_GPUS:-gpu:1}"
JOB_NAME="${SKETCHSMILES_SLURM_JOB_NAME:-sketchsmiles-5a1}"
LOG_DIR="${SKETCHSMILES_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export SKETCHSMILES_MODULES="${SKETCHSMILES_MODULES:-gcc rdkit/2025.09.4}"
export SKETCHSMILES_PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
export SKETCHSMILES_PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
export SKETCHSMILES_RUN_NAME="${SKETCHSMILES_RUN_NAME:-phase5a1_learned_smiles_decoder_seed${SKETCHSMILES_SEED:-7}}"
export SKETCHSMILES_EPOCHS="${SKETCHSMILES_EPOCHS:-20}"
export SKETCHSMILES_BATCH_SIZE="${SKETCHSMILES_BATCH_SIZE:-128}"
export SKETCHSMILES_DEVICE="${SKETCHSMILES_DEVICE:-auto}"

sbatch \
  --account="$ACCOUNT" \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  --gres="$GPU_REQUEST" \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash scripts/run_phase5a1_learned_smiles_decoder.sh"
