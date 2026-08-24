#!/usr/bin/env python3
"""Reuse the P8.1.7 null-source-exact full-SMILES entrypoint."""
from pathlib import Path
import runpy

target = Path(__file__).resolve().parents[1] / "p8_1_7_source_clamped_policy" / "source_clamped_entrypoint.py"
runpy.run_path(str(target), run_name="__main__")
