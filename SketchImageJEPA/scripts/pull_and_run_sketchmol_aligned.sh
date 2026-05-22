#!/usr/bin/env bash
# Pull the latest repository code and run the SketchMol-aligned
# SketchImage-JEPA experiment.
#
# Typical server usage from anywhere inside this repo:
#   bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh
#
# Useful overrides:
#   SKETCHIMAGE_DATASET_CSV=/path/to/tasks.csv \
#   SKETCHIMAGE_RUN_NAME=sketchmol_aligned_server_v1 \
#   bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"

cd "$REPO_ROOT"

REMOTE="${SKETCHIMAGE_GIT_REMOTE:-origin}"
BRANCH="${SKETCHIMAGE_GIT_BRANCH:-main}"
SKIP_PULL="${SKETCHIMAGE_SKIP_PULL:-0}"

echo "SketchImage-JEPA pull-and-run"
echo "  repo_root=$REPO_ROOT"
echo "  project_dir=$PROJECT_DIR"
echo "  remote=$REMOTE"
echo "  branch=$BRANCH"

if [[ "$SKIP_PULL" != "1" ]]; then
  if [[ -n "$(git status --porcelain)" && "${SKETCHIMAGE_ALLOW_DIRTY_PULL:-0}" != "1" ]]; then
    cat <<EOF
Working tree is not clean, so I will not pull automatically.

Commit/stash your local changes, or run with:
  SKETCHIMAGE_ALLOW_DIRTY_PULL=1 bash SketchImageJEPA/scripts/pull_and_run_sketchmol_aligned.sh

Current changes:
EOF
    git status --short
    exit 2
  fi

  echo
  echo "[1/2] Pulling latest code"
  git fetch "$REMOTE" "$BRANCH"
  git pull --rebase "$REMOTE" "$BRANCH"
else
  echo
  echo "[1/2] Skipping git pull because SKETCHIMAGE_SKIP_PULL=1"
fi

echo
echo "[2/2] Running SketchMol-aligned experiment"
cd "$PROJECT_DIR"
bash scripts/run_sketchmol_aligned.sh
