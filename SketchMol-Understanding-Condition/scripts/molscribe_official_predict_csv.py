#!/usr/bin/env python3
"""Run SketchMol evaluate/predict_csv.py with onmt220 attention-mask compatibility."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.molscribe_onmt_compat import (  # noqa: E402
    apply_onmt_attention_mask_patch,
)


def main() -> None:
    apply_onmt_attention_mask_patch()

    eval_dir = Path.cwd()
    script = eval_dir / "predict_csv.py"
    if not script.is_file():
        raise FileNotFoundError(f"predict_csv.py not found in cwd: {eval_dir}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
