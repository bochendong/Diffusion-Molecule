#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${P24_PUBCHEM_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/raw/pubchem/current_sdf}"
mkdir -p "$RAW_ROOT"
for start in 000000001 000500001 001000001 001500001 002000001 002500001 003000001 003500001; do
  end=$((10#$start + 499999))
  end_padded=$(printf '%09d' "$end")
  name="Compound_${start}_${end_padded}.sdf.gz"
  url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/$name"
  echo "[p24-download] $name"
  curl --fail --location --retry 8 --retry-delay 10 --continue-at - --output "$RAW_ROOT/$name" "$url"
  gzip -t "$RAW_ROOT/$name"
done
touch "$RAW_ROOT/DOWNLOAD_COMPLETE"

