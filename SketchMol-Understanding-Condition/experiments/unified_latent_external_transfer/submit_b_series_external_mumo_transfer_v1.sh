#!/usr/bin/env bash
# Submit prepare -> frozen B generation -> official oracle/eval -> science gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIL_USER="${SUCC_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT_CPU="${SUCC_CPU_ACCOUNT:-def-hup-ab_cpu}"
ACCOUNT_GPU="${SUCC_GPU_ACCOUNT:-def-hup-ab}"
LOG_DIR="${SUCC_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/b_series_external_mumo_transfer_v1}"
RUN_SCRIPT="$SCRIPT_DIR/run_b_series_external_mumo_transfer_v1.sh"
mkdir -p "$LOG_DIR"

prepare_id="$(sbatch --parsable \
  --account="$ACCOUNT_CPU" --job-name=uca-bext-mumo-prepare \
  --time=00:45:00 --cpus-per-task=4 --mem=16G \
  --output="$LOG_DIR/uca-bext-mumo-prepare-%j.log" \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL \
  --wrap="bash '$RUN_SCRIPT' prepare")"
prepare_id="${prepare_id%%;*}"

freeze_id=""
used_gpu=""
for gpu in nvidia_h100_80gb_hbm3_1g.10gb:1 nvidia_h100_80gb_hbm3_2g.20gb:1; do
  if output="$(sbatch --parsable \
    --account="$ACCOUNT_GPU" --job-name=uca-bext-mumo-freeze \
    --dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes \
    --time=02:00:00 --cpus-per-task=4 --mem=20G --gres="gpu:$gpu" \
    --output="$LOG_DIR/uca-bext-mumo-freeze-%j.log" \
    --mail-user="$MAIL_USER" --mail-type=FAIL \
    --wrap="bash '$RUN_SCRIPT' freeze")"; then
    freeze_id="${output%%;*}"
    used_gpu="$gpu"
    break
  fi
done
if [[ -z "$freeze_id" ]]; then
  echo "ERROR: unable to submit frozen B generation" >&2
  exit 1
fi

oracle_id="$(sbatch --parsable \
  --account="$ACCOUNT_CPU" --job-name=uca-bext-mumo-oracle \
  --dependency="afterok:$freeze_id" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --output="$LOG_DIR/uca-bext-mumo-oracle-%j.log" \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --wrap="bash '$RUN_SCRIPT' oracle")"
oracle_id="${oracle_id%%;*}"

gate_id="$(sbatch --parsable \
  --account="$ACCOUNT_CPU" --job-name=uca-bext-mumo-scigate \
  --dependency="afterok:$oracle_id" --kill-on-invalid-dep=yes \
  --time=00:15:00 --cpus-per-task=2 --mem=8G \
  --output="$LOG_DIR/uca-bext-mumo-scigate-%j.log" \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --wrap="bash '$RUN_SCRIPT' gate")"
gate_id="${gate_id%%;*}"

printf 'prepare=%s\nfreeze=%s\noracle=%s\nscience_gate=%s\ngpu=%s\n' \
  "$prepare_id" "$freeze_id" "$oracle_id" "$gate_id" "$used_gpu"
printf 'summary=/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/b_series_external_mumo_transfer_v1/seed_2191/gate/summary.json\n'
