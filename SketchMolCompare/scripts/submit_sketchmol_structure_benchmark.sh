#!/usr/bin/env bash
# Deprecated: this used to submit a PhysTabMol proxy benchmark. The real
# SketchMol baseline lives under Research/Molecule Generation/SketchMol and is
# submitted via SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh.

set -euo pipefail

cat <<'EOF' >&2
ERROR: This script is deprecated because it submits the older PhysTabMol proxy.

For the real SketchMol benchmark, use:

  SKETCHMOL_CKPT=/absolute/path/to/sketchmol/model.ckpt \
  SKETCHMOL_MOLSCRIBE_MODEL=/absolute/path/to/swin_base_char_aux_200k.pth \
  bash SketchMolBenchmark/scripts/submit_real_sketchmol_ocr.sh

Replace both /absolute/path/to/... examples with real checkpoint files on the
cluster filesystem. The submit script validates those files before sbatch.

If you already have a SketchMol image_path.csv after MolScribe OCR, use:

  SKETCHMOL_BENCHMARK_SOURCE_CSV=/path/to/image_path.csv \
  bash SketchMolBenchmark/scripts/materialize_current.sh
EOF
exit 2
