#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_ARGS=(
  --account="${P16_ACCOUNT:-def-hup-ab}"
  --job-name=p16-direct-7b
  --time="${P16_TIME:-00:45:00}"
  --cpus-per-task="${P16_CPUS:-4}"
  --mem="${P16_MEM:-40G}"
  --gres="${P16_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
  --output="${P16_LOG:-$SCRIPT_DIR/p16-%j.log}"
  --export="ALL,P16_SCRIPT_DIR=$SCRIPT_DIR"
)
sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_p16_pilot.sh"
