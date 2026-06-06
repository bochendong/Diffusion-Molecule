#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
CONFIG="${SDEA_CONFIG:-$PROJECT_DIR/configs/large.yaml}"
PYTHON_BIN="${SDEA_PYTHON_BIN:-python3}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_large_pipeline.py" --config "$CONFIG" --prepare-only "${@}"

