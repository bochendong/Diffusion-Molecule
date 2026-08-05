#!/usr/bin/env bash
# Run multiple de novo 2p-7p shards serially on one GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Match SketchMolBenchmark README / resume_real_sketchmol_ocr: MolScribe needs cluster OpenCV.
SKETCHMOL_MODULES="${SKETCHMOL_MODULES:-gcc opencv/4.13.0 rdkit/2024.09.6}"
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck source=/dev/null
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
fi
if command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_MODULES
fi

SKETCHMOL_DENOVO_SHARD_COUNT="${SKETCHMOL_DENOVO_SHARD_COUNT:-600}"
SKETCHMOL_DENOVO_SERIAL_START="${SKETCHMOL_DENOVO_SERIAL_START:-0}"
SKETCHMOL_DENOVO_SERIAL_END="${SKETCHMOL_DENOVO_SERIAL_END:-$((SKETCHMOL_DENOVO_SHARD_COUNT - 1))}"
SKETCHMOL_DENOVO_RESUME="${SKETCHMOL_DENOVO_RESUME:-1}"

if (( SKETCHMOL_DENOVO_SERIAL_START < 0 || SKETCHMOL_DENOVO_SERIAL_END >= SKETCHMOL_DENOVO_SHARD_COUNT )); then
  echo "ERROR: invalid serial shard range ${SKETCHMOL_DENOVO_SERIAL_START}..${SKETCHMOL_DENOVO_SERIAL_END}" >&2
  exit 2
fi

wave_size=$((SKETCHMOL_DENOVO_SERIAL_END - SKETCHMOL_DENOVO_SERIAL_START + 1))
echo "Serial shard run on one GPU"
echo "  shard_count=$SKETCHMOL_DENOVO_SHARD_COUNT"
echo "  shard_range=${SKETCHMOL_DENOVO_SERIAL_START}..${SKETCHMOL_DENOVO_SERIAL_END} (${wave_size} shards)"
echo "  resume=$SKETCHMOL_DENOVO_RESUME"
echo "  modules=$SKETCHMOL_MODULES"

for shard_index in $(seq "$SKETCHMOL_DENOVO_SERIAL_START" "$SKETCHMOL_DENOVO_SERIAL_END"); do
  echo
  echo "================================================================"
  echo "Serial shard ${shard_index}/${SKETCHMOL_DENOVO_SERIAL_END}  ($(date -Iseconds))"
  echo "================================================================"
  SKETCHMOL_DENOVO_SHARD_INDEX="$shard_index" \
    bash "$SCRIPT_DIR/run_denovo_2p7p_sketchmol_shard.sh"
done

echo "Serial shard run complete: ${SKETCHMOL_DENOVO_SERIAL_START}..${SKETCHMOL_DENOVO_SERIAL_END}"
