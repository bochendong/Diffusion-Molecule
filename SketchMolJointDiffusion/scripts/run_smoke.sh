#!/usr/bin/env bash
# Run lightweight tests for SketchMolJointDiffusion.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_JOINT_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchMolTokenDiffusion:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolJointDiffusion smoke"
echo "  python=$PYTHON_BIN"

"$PYTHON_BIN" -m unittest discover -s SketchMolJointDiffusion/tests -p 'test_*.py'
