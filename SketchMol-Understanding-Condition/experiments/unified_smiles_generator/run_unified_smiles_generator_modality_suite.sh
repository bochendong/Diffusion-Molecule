#!/usr/bin/env bash
# Run one modality end-to-end: SFT -> group-RL -> sample/beam benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

MODALITY="${1:-}"
if [[ -z "$MODALITY" ]]; then
  echo "Usage: $0 <with_image|no_image>" >&2
  exit 2
fi

case "$MODALITY" in
  with_image) VARIANT="full" ;;
  no_image) VARIANT="text_only" ;;
  *)
    echo "ERROR: unsupported modality=$MODALITY" >&2
    exit 2
    ;;
esac

SUCC_UNIFIED_SUITE_MODALITIES="$MODALITY" \
SUCC_UNIFIED_SUITE_DECODING_MODES="${SUCC_UNIFIED_SUITE_DECODING_MODES:-sample,beam}" \
SUCC_UNIFIED_SUITE_RUN_FEATURE_EXPORT=0 \
SUCC_UNIFIED_SUITE_RUN_TRAIN="${SUCC_UNIFIED_SUITE_RUN_TRAIN:-1}" \
SUCC_UNIFIED_SUITE_RUN_RL="${SUCC_UNIFIED_SUITE_RUN_RL:-1}" \
SUCC_UNIFIED_SUITE_RUN_BENCHMARK="${SUCC_UNIFIED_SUITE_RUN_BENCHMARK:-1}" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_experiment_suite.sh"
