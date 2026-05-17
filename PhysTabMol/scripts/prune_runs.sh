#!/bin/bash
set -euo pipefail

# Remove obsolete PhysTabMol run directories while preserving paper-facing runs.
#
# Default mode is a dry run:
#   bash scripts/prune_runs.sh
#
# Actually delete:
#   bash scripts/prune_runs.sh --apply
#
# Keep extra exact run basenames:
#   bash scripts/prune_runs.sh --keep 20260514_062738_instruction_freeform_fragment_v1 --apply
#
# Useful env overrides:
#   PHYSTABMOL_BEST_STRUCTURE_RUN=20260512_235957_sketchmol_comparable_structure_v1
#   PHYSTABMOL_KEEP_RUNS="run_a run_b"
#   PHYSTABMOL_KEEP_LEGACY_INSTRUCTION=1

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"

RUNS_DIR="${PHYSTABMOL_RUNS_DIR:-runs}"
APPLY=0
KEEP_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --dry-run)
      APPLY=0
      shift
      ;;
    --keep)
      if [[ $# -lt 2 ]]; then
        echo "--keep requires a run basename or runs/<basename>" >&2
        exit 2
      fi
      KEEP_EXTRA+=("$(basename "$2")")
      shift 2
      ;;
    --runs-dir)
      if [[ $# -lt 2 ]]; then
        echo "--runs-dir requires a directory" >&2
        exit 2
      fi
      RUNS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,32p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "No runs directory found: $RUNS_DIR"
  exit 0
fi

BEST_STRUCTURE_RUN="${PHYSTABMOL_BEST_STRUCTURE_RUN:-}"
if [[ -z "$BEST_STRUCTURE_RUN" ]]; then
  if [[ -d "$RUNS_DIR/20260512_235957_sketchmol_comparable_structure_v1" ]]; then
    BEST_STRUCTURE_RUN="20260512_235957_sketchmol_comparable_structure_v1"
  else
    BEST_STRUCTURE_RUN="$(
      find "$RUNS_DIR" -maxdepth 1 -mindepth 1 -type d -name '*sketchmol_comparable_structure_v1' -print \
        | sed 's#^.*/##' \
        | sort \
        | tail -1
    )"
  fi
fi

KEEP_EXACT=()
if [[ -n "$BEST_STRUCTURE_RUN" ]]; then
  KEEP_EXACT+=("$BEST_STRUCTURE_RUN")
fi
for run_name in ${PHYSTABMOL_KEEP_RUNS:-}; do
  KEEP_EXACT+=("$(basename "$run_name")")
done
if [[ "${PHYSTABMOL_KEEP_LEGACY_INSTRUCTION:-0}" == "1" ]]; then
  KEEP_EXACT+=("20260514_062738_instruction_freeform_fragment_v1")
fi
if [[ ${#KEEP_EXTRA[@]} -gt 0 ]]; then
  KEEP_EXACT+=("${KEEP_EXTRA[@]}")
fi

should_keep() {
  local name="$1"
  local keep
  for keep in "${KEEP_EXACT[@]}"; do
    [[ "$name" == "$keep" ]] && return 0
  done

  case "$name" in
    *instruction_ablation_*|*instruction_generalization_*|*instruction_verified_*)
      return 0
      ;;
  esac

  return 1
}

KEEP_LIST=()
DELETE_LIST=()
while IFS= read -r run_path; do
  name="$(basename "$run_path")"
  if should_keep "$name"; then
    KEEP_LIST+=("$run_path")
  else
    DELETE_LIST+=("$run_path")
  fi
done < <(find "$RUNS_DIR" -maxdepth 1 -mindepth 1 -type d -print | sort)

cat <<EOF
PhysTabMol run pruning
  root=$(pwd)
  runs_dir=$RUNS_DIR
  mode=$([[ "$APPLY" == "1" ]] && echo "APPLY" || echo "DRY-RUN")
  best_structure_run=${BEST_STRUCTURE_RUN:-none}
  keep_count=${#KEEP_LIST[@]}
  delete_count=${#DELETE_LIST[@]}
EOF

echo
echo "Keeping:"
if [[ ${#KEEP_LIST[@]} -eq 0 ]]; then
  echo "  (none)"
else
  printf '  %s\n' "${KEEP_LIST[@]}"
fi

echo
echo "Deleting:"
if [[ ${#DELETE_LIST[@]} -eq 0 ]]; then
  echo "  (none)"
else
  printf '  %s\n' "${DELETE_LIST[@]}"
fi

if [[ "$APPLY" != "1" ]]; then
  echo
  echo "Dry run only. Re-run with --apply to delete the listed directories."
  exit 0
fi

if [[ ${#DELETE_LIST[@]} -eq 0 ]]; then
  echo "Nothing to delete."
  exit 0
fi

for run_path in "${DELETE_LIST[@]}"; do
  case "$run_path" in
    "$RUNS_DIR"/*)
      rm -rf -- "$run_path"
      ;;
    *)
      echo "Refusing to delete suspicious path: $run_path" >&2
      exit 3
      ;;
  esac
done

echo "Deleted ${#DELETE_LIST[@]} obsolete run directories."
