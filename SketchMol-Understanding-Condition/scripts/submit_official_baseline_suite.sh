#!/usr/bin/env bash
# Submit or dry-run the paper-facing official/paper-faithful baseline suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

BENCHMARKS="${BENCHMARKS:-moledit_table1,mumo,cmumo,sketchmol_denovo}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --benchmarks)
      BENCHMARKS="${2:-}"
      shift 2
      ;;
    DRY_RUN=*)
      DRY_RUN="${1#DRY_RUN=}"
      shift
      ;;
    BENCHMARKS=*)
      BENCHMARKS="${1#BENCHMARKS=}"
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Official/paper-faithful baseline suite"
echo "  benchmarks=$BENCHMARKS"
echo "  dry_run=$DRY_RUN"
echo "  registry=${DM_BENCHMARK_REGISTRY:-SketchMol-Understanding-Condition/configs/official_benchmark_registry.json}"

run_or_print() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'would run:'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  "$@"
}

IFS=',' read -r -a benchmark_array <<< "$BENCHMARKS"
for benchmark in "${benchmark_array[@]}"; do
  benchmark_trimmed="${benchmark#"${benchmark%%[![:space:]]*}"}"
  benchmark_trimmed="${benchmark_trimmed%"${benchmark_trimmed##*[![:space:]]}"}"
  [[ -z "$benchmark_trimmed" ]] && continue
  case "$benchmark_trimmed" in
    moledit_table1|MolEditRL|moleditrl)
      run_or_print \
        "MolEditRL Table1 paper-faithful baseline" \
        bash "$PROJECT_DIR/scripts/submit_moleditrl_table1_paper_faithful.sh"
      ;;
    mumo|GeLLMO|gellmo)
      run_or_print \
        "GeLLMO MuMO official baseline" \
        bash "$PROJECT_DIR/scripts/submit_external_gellmo_official_suite.sh"
      ;;
    cmumo|c-mumo|GeLLMO-C|gellmoc)
      run_or_print \
        "GeLLMO-C C-MuMO official baseline" \
        bash "$PROJECT_DIR/scripts/submit_external_gellmoc_official_suite.sh"
      ;;
    sketchmol_denovo|SketchMol|sketchmol)
      if [[ "$DRY_RUN" == "1" ]]; then
        echo
        echo "=== SketchMol de novo official baseline ==="
        echo "would run: bash SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh"
        echo "requires on server: SKETCHMOL_CKPT, SKETCHMOL_MOLSCRIBE_MODEL, SKETCHMOL_PYTHON_BIN, SKETCHMOL_MOLSCRIBE_PYTHON_BIN"
      else
        bash SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh
      fi
      ;;
    *)
      echo "ERROR: unknown benchmark in BENCHMARKS: $benchmark_trimmed" >&2
      exit 2
      ;;
  esac
done

echo
echo "Official/paper-faithful baseline suite dispatch complete."
