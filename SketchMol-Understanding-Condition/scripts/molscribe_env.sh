#!/usr/bin/env bash
# SUCC-only MolScribe runtime env.
#
# OCR fixes live under SketchMol-Understanding-Condition/ only. Do not patch
# Research/.../evaluate/molscribe or SketchMolBenchmark scripts; prepend
# PYTHONPATH here (and in run_molscribe_ocr.py) at runtime instead.

SKETCHMOL_ROOT="${SKETCHMOL_ROOT:-${SKETCHMOL_REPO:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}}"
MOLSCRIBE_WORKDIR="${SUCC_MOLSCRIBE_WORKDIR:-${SKETCHMOL_MOLSCRIBE_WORKDIR:-${MOLSCRIBE_WORKDIR:-$SKETCHMOL_ROOT/evaluate}}}"
ONMT_OVERLAY="${SUCC_ONMT_OVERLAY:-${SKETCHMOL_ONMT_OVERLAY:-/scratch/bdong/python_overlays/onmt220}}"

_resolve_existing_dir() {
  local candidate="$1"
  if [[ -z "$candidate" ]]; then
    return 1
  fi
  if [[ -d "$candidate" ]]; then
    (cd "$candidate" && pwd)
    return 0
  fi
  if [[ -n "${REPO_ROOT:-}" && -d "$REPO_ROOT/$candidate" ]]; then
    (cd "$REPO_ROOT/$candidate" && pwd)
    return 0
  fi
  if [[ -n "${REPO_DIR:-}" && -d "$REPO_DIR/$candidate" ]]; then
    (cd "$REPO_DIR/$candidate" && pwd)
    return 0
  fi
  return 1
}

resolve_molscribe_eval_dir() {
  local root
  if root="$(_resolve_existing_dir "$MOLSCRIBE_WORKDIR")"; then
    if [[ -d "$root/molscribe" ]]; then
      printf '%s\n' "$root"
      return 0
    fi
    if [[ -d "$root/evaluate/molscribe" ]]; then
      (cd "$root/evaluate" && pwd)
      return 0
    fi
  fi

  if root="$(_resolve_existing_dir "$SKETCHMOL_ROOT")"; then
    if [[ -d "$root/evaluate/molscribe" ]]; then
      (cd "$root/evaluate" && pwd)
      return 0
    fi
  fi

  if [[ -d molscribe && -f predict_csv.py ]]; then
    pwd
    return 0
  fi

  return 1
}

_prepend_pythonpath_entries() {
  local prefix=""
  local entry
  for entry in "$@"; do
    if [[ -z "$entry" ]]; then
      continue
    fi
    if [[ -z "$prefix" ]]; then
      prefix="$entry"
    else
      prefix="$prefix:$entry"
    fi
  done
  if [[ -n "$prefix" ]]; then
    export PYTHONPATH="$prefix${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

prepend_molscribe_pythonpath() {
  local eval_dir=""
  local sketchmol_repo=""
  if eval_dir="$(resolve_molscribe_eval_dir)"; then
    sketchmol_repo="$(cd "$eval_dir/.." && pwd)"
  fi

  local overlay_dir=""
  if [[ -n "$ONMT_OVERLAY" && -d "$ONMT_OVERLAY" ]]; then
    overlay_dir="$(cd "$ONMT_OVERLAY" && pwd)"
  elif [[ -n "$ONMT_OVERLAY" ]]; then
    echo "WARNING: ONMT overlay not found: $ONMT_OVERLAY" >&2
    echo "         MolScribe graph decoding may return empty SMILES." >&2
  fi

  if [[ -n "$eval_dir" ]]; then
    _prepend_pythonpath_entries "$overlay_dir" "$eval_dir" "$sketchmol_repo"
  elif [[ -n "$MOLSCRIBE_WORKDIR" ]]; then
    _prepend_pythonpath_entries "$overlay_dir" "$MOLSCRIBE_WORKDIR"
  else
    _prepend_pythonpath_entries "$overlay_dir"
  fi
}

run_official_molscribe_predict_csv() {
  local python_bin="$1"
  local model_path="$2"
  local image_csv="$3"
  local batch_size="$4"
  local project_dir
  project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

  local eval_dir
  if ! eval_dir="$(resolve_molscribe_eval_dir)"; then
    echo "ERROR: could not resolve SketchMol evaluate/ directory for MolScribe." >&2
    echo "       Set SUCC_MOLSCRIBE_WORKDIR=Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate" >&2
    return 2
  fi

  if [[ ! -f "$eval_dir/predict_csv.py" ]]; then
    echo "ERROR: official SketchMol predict_csv.py not found in $eval_dir" >&2
    return 2
  fi

  if [[ ! -f "$image_csv" ]]; then
    echo "ERROR: MolScribe image CSV not found: $image_csv" >&2
    return 2
  fi
  image_csv="$(cd "$(dirname "$image_csv")" && pwd)/$(basename "$image_csv")"
  if [[ -f "$model_path" ]]; then
    model_path="$(cd "$(dirname "$model_path")" && pwd)/$(basename "$model_path")"
  fi

  local sketchmol_repo
  sketchmol_repo="$(cd "$eval_dir/.." && pwd)"
  local overlay_dir=""
  if [[ -n "$ONMT_OVERLAY" && -d "$ONMT_OVERLAY" ]]; then
    overlay_dir="$(cd "$ONMT_OVERLAY" && pwd)"
  fi

  local pythonpath_prefix="$eval_dir:$sketchmol_repo"
  if [[ -n "$overlay_dir" ]]; then
    pythonpath_prefix="$overlay_dir:$pythonpath_prefix"
  fi

  local wrapper_script="$project_dir/scripts/molscribe_official_predict_csv.py"
  if [[ ! -f "$wrapper_script" ]]; then
    echo "ERROR: molscribe_official_predict_csv.py wrapper not found: $wrapper_script" >&2
    return 2
  fi

  echo "Running SketchMol predict_csv.py (onmt220 mask patch)"
  echo "  wrapper=$wrapper_script"
  echo "  evaluate_dir=$eval_dir"
  echo "  image_csv=$image_csv"
  echo "  batch_size=$batch_size"

  local had_errexit=0
  case "$-" in
    *e*) had_errexit=1 ;;
  esac

  pushd "$eval_dir" >/dev/null
  set +e
  PYTHONPATH="$pythonpath_prefix:$project_dir${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" "$wrapper_script" \
      --model_path "$model_path" \
      --image_path "$image_csv" \
      -n "$batch_size"
  local status=$?
  if [[ "$had_errexit" == "1" ]]; then
    set -e
  else
    set +e
  fi
  popd >/dev/null
  return "$status"
}

check_molscribe_import() {
  local python_bin="${PYTHON_BIN:-${SKETCHMOL_MOLSCRIBE_PYTHON_BIN:-python3}}"
  ONMT_OVERLAY="$ONMT_OVERLAY" MOLSCRIBE_WORKDIR="$MOLSCRIBE_WORKDIR" "$python_bin" - <<'PY'
import importlib
import os
import sys

onmt_overlay = os.environ.get("ONMT_OVERLAY", "")
molscribe_workdir = os.environ.get("MOLSCRIBE_WORKDIR", "")

try:
    from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg  # noqa: F401
    from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights  # noqa: F401
    import molscribe
    import onmt
    importlib.import_module("onmt.modules.multi_headed_attn")
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
    expected_overlay = os.path.realpath(onmt_overlay)
    actual_onmt = os.path.realpath(onmt_path) if onmt_path else ""
    if not actual_onmt.startswith(expected_overlay + os.sep):
        print("ERROR: onmt is not imported from the expected overlay:", file=sys.stderr)
        print(f"  onmt={onmt_path}", file=sys.stderr)
        print(f"  expected overlay prefix={onmt_overlay}", file=sys.stderr)
        print("Hint: source molscribe_env.sh or export SUCC_ONMT_OVERLAY before OCR.", file=sys.stderr)
        sys.exit(2)
    mixed_modules = []
    for name, module in sorted(sys.modules.items()):
        if name != "onmt" and not name.startswith("onmt."):
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        real_path = os.path.realpath(path)
        if not real_path.startswith(expected_overlay + os.sep):
            mixed_modules.append((name, path))
    if mixed_modules:
        print("ERROR: mixed onmt modules detected outside the expected overlay:", file=sys.stderr)
        for name, path in mixed_modules[:10]:
            print(f"  {name}: {path}", file=sys.stderr)
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
