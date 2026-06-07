#!/usr/bin/env bash
# Shared SketchMol vendored MolScribe setup for the molscribe_overlay venv.
#
# SketchMol ships a patched molscribe under evaluate/molscribe. Importing the
# upstream pip package instead breaks graph decoding. Always prepend the vendored
# evaluate directory before running OCR.
#
# molscribe_overlay reuses phystabmol's bundled OpenNMT, which is too old for
# graph->SMILES decoding. Prepend the onmt220 overlay first; job 15544986
# (paper_repro_mw400_real_official_ocr) succeeded only with this layout.

SKETCHMOL_ROOT="${SKETCHMOL_ROOT:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
MOLSCRIBE_WORKDIR="${SUCC_MOLSCRIBE_WORKDIR:-${SKETCHMOL_MOLSCRIBE_WORKDIR:-${MOLSCRIBE_WORKDIR:-$SKETCHMOL_ROOT/evaluate}}}"
ONMT_OVERLAY="${SUCC_ONMT_OVERLAY:-${SKETCHMOL_ONMT_OVERLAY:-/scratch/bdong/python_overlays/onmt220}}"

prepend_molscribe_pythonpath() {
  if [[ -n "$ONMT_OVERLAY" && -d "$ONMT_OVERLAY" ]]; then
    export PYTHONPATH="$ONMT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
  elif [[ -n "$ONMT_OVERLAY" ]]; then
    echo "WARNING: ONMT overlay not found: $ONMT_OVERLAY" >&2
    echo "         MolScribe graph decoding may return empty SMILES." >&2
  fi

  if [[ -z "$MOLSCRIBE_WORKDIR" ]]; then
    return
  fi
  if [[ -d "$MOLSCRIBE_WORKDIR/molscribe" ]]; then
    export PYTHONPATH="$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  elif [[ -d "$MOLSCRIBE_WORKDIR/evaluate/molscribe" ]]; then
    export PYTHONPATH="$MOLSCRIBE_WORKDIR/evaluate:$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  else
    export PYTHONPATH="$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

check_molscribe_import() {
  local python_bin="${PYTHON_BIN:-${SKETCHMOL_MOLSCRIBE_PYTHON_BIN:-python3}}"
  ONMT_OVERLAY="$ONMT_OVERLAY" MOLSCRIBE_WORKDIR="$MOLSCRIBE_WORKDIR" "$python_bin" - <<'PY'
import os
import sys

onmt_overlay = os.environ.get("ONMT_OVERLAY", "")
molscribe_workdir = os.environ.get("MOLSCRIBE_WORKDIR", "")

try:
    from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg  # noqa: F401
    from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights  # noqa: F401
    import molscribe
    import onmt
    from molscribe import MolScribe  # noqa: F401
except Exception as exc:
    print("ERROR: MolScribe/timm/onmt compatibility check failed:", file=sys.stderr)
    print(f"  {exc}", file=sys.stderr)
    print("Hint: use molscribe_overlay venv, prepend onmt220 overlay, and set", file=sys.stderr)
    print("      SUCC_MOLSCRIBE_WORKDIR=Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate", file=sys.stderr)
    sys.exit(2)

molscribe_path = molscribe.__file__ or ""
if "SketchMol" not in molscribe_path and "evaluate/molscribe" not in molscribe_path:
    print(
        "ERROR: molscribe is not imported from SketchMol evaluate/:",
        molscribe_path,
        file=sys.stderr,
    )
    sys.exit(2)

onmt_path = onmt.__file__ or ""
if onmt_overlay:
    if onmt_overlay not in onmt_path:
        print("ERROR: onmt is not imported from the expected overlay:", file=sys.stderr)
        print(f"  onmt={onmt_path}", file=sys.stderr)
        print(f"  expected overlay prefix={onmt_overlay}", file=sys.stderr)
        print("Hint: source molscribe_env.sh or export SUCC_ONMT_OVERLAY before OCR.", file=sys.stderr)
        sys.exit(2)
else:
    if "python_overlays/onmt220" not in onmt_path:
        print(
            "WARNING: onmt is not from onmt220 overlay:",
            onmt_path,
            file=sys.stderr,
        )

print(f"molscribe import ok ({molscribe_path})")
print(f"onmt import ok ({onmt_path})")
if molscribe_workdir:
    print(f"molscribe_workdir={molscribe_workdir}")
if onmt_overlay:
    print(f"onmt_overlay={onmt_overlay}")
PY
}
