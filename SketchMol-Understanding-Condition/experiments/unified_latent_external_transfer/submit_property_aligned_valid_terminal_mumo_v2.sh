#!/usr/bin/env bash
# Submit the final property-aligned B-series route gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIL_USER="${SUCC_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT_CPU="${SUCC_CPU_ACCOUNT:-def-hup-ab_cpu}"
ACCOUNT_GPU="${SUCC_GPU_ACCOUNT:-def-hup-ab}"
LOG_DIR="${SUCC_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/property_aligned_valid_terminal_mumo_v2_vocab_expanded}"
RUN_SCRIPT="$SCRIPT_DIR/run_property_aligned_valid_terminal_mumo_v2.sh"
mkdir -p "$LOG_DIR"

prepare_id="$(sbatch --parsable \
  --account="$ACCOUNT_CPU" --job-name=uca-bmumo-v2x-prepare \
  --time=01:30:00 --cpus-per-task=16 --mem=64G \
  --output="$LOG_DIR/uca-bmumo-v2x-prepare-%j.log" \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL \
  --wrap="bash '$RUN_SCRIPT' prepare")"
prepare_id="${prepare_id%%;*}"

train_id=""
used_gpu=""
for gpu in nvidia_h100_80gb_hbm3_2g.20gb:1 nvidia_h100_80gb_hbm3_3g.40gb:1; do
  if output="$(sbatch --parsable \
    --account="$ACCOUNT_GPU" --job-name=uca-bmumo-v2x-train \
    --dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes \
    --time=04:00:00 --cpus-per-task=8 --mem=64G --gres="gpu:$gpu" \
    --output="$LOG_DIR/uca-bmumo-v2x-train-%j.log" \
    --mail-user="$MAIL_USER" --mail-type=FAIL \
    --wrap="bash '$RUN_SCRIPT' trainfreeze")"; then
    train_id="${output%%;*}"
    used_gpu="$gpu"
    break
  fi
done
[[ -n "$train_id" ]] || { echo "ERROR: unable to submit H100 MIG training" >&2; exit 1; }

oracle_id="$(sbatch --parsable \
  --account="$ACCOUNT_CPU" --job-name=uca-bmumo-v2x-oracle \
  --dependency="afterok:$train_id" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --output="$LOG_DIR/uca-bmumo-v2x-oracle-%j.log" \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --wrap="bash '$RUN_SCRIPT' oracle")"
oracle_id="${oracle_id%%;*}"

gate_id="$(sbatch --parsable \
  --account="$ACCOUNT_CPU" --job-name=uca-bmumo-v2x-scigate \
  --dependency="afterok:$oracle_id" --kill-on-invalid-dep=yes \
  --time=00:15:00 --cpus-per-task=2 --mem=8G \
  --output="$LOG_DIR/uca-bmumo-v2x-scigate-%j.log" \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --wrap="bash '$RUN_SCRIPT' gate")"
gate_id="${gate_id%%;*}"

printf 'prepare=%s\ntrainfreeze=%s\noracle=%s\nscience_gate=%s\ngpu=%s\n' \
  "$prepare_id" "$train_id" "$oracle_id" "$gate_id" "$used_gpu"
printf 'summary=/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/property_aligned_valid_terminal_mumo_v2_vocab_expanded/seed_2211/gate/summary.json\n'
