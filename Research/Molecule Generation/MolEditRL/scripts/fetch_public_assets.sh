#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
RAW_DIR="$ROOT_DIR/data/raw"

ARXIV_ID="2505.20131"
DATASET_FILE="MolEdit-Instruct_3034459.txt"
DATASET_URL="https://huggingface.co/datasets/FanSiLeC/MolEdit-Instruct/resolve/main/${DATASET_FILE}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$PAPER_DIR" "$RAW_DIR"

echo "MolEditRL public asset fetch"
echo "root: $ROOT_DIR"
echo "paper: https://arxiv.org/abs/${ARXIV_ID}"
echo "dataset: $DATASET_URL"
echo "download dataset: ${MOLEDITRL_DOWNLOAD_DATASET:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ -s "$output" ]]; then
    echo "exists: $output"
    return
  fi
  echo "download: $url"
  curl -L --fail --continue-at - --output "$output" "$url"
}

download_if_missing "https://arxiv.org/pdf/${ARXIV_ID}" "$PAPER_DIR/arxiv-${ARXIV_ID}.pdf"
download_if_missing "https://arxiv.org/e-print/${ARXIV_ID}" "$PAPER_DIR/arxiv-${ARXIV_ID}-source.tar.gz"

if [[ "${MOLEDITRL_DOWNLOAD_DATASET:-0}" == "1" ]]; then
  download_if_missing "$DATASET_URL" "$RAW_DIR/$DATASET_FILE"
else
  echo "skip dataset download; set MOLEDITRL_DOWNLOAD_DATASET=1 to fetch the 768 MB public dataset"
fi

