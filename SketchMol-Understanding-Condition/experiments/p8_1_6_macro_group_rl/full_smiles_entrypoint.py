#!/usr/bin/env python3
"""Reuse the audited P8.1.4 direct-compatible full-SMILES entrypoint."""
from pathlib import Path
import runpy

target = Path(__file__).resolve().parents[1] / "p8_1_4_full_smiles_multitask" / "full_smiles_entrypoint.py"
runpy.run_path(str(target), run_name="__main__")
