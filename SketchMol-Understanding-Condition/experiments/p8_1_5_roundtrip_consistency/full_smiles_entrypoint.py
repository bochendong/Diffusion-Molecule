#!/usr/bin/env python3
"""P8.1.5 one-decoder entrypoint with a P1-exact empty-source path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for path in (PROJECT_DIR, UNIFIED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import unified_smiles_generator as core  # noqa: E402
from sketchmol_understanding_condition import direct_condition_tokens as direct  # noqa: E402


def p1_exact_denovo_source_condition(
    row, store, condition_dim, *, max_source_tokens, condition_layout="unified"
):
    if str(condition_layout) != "unified":
        return _ORIGINAL(
            row,
            store,
            condition_dim,
            max_source_tokens=max_source_tokens,
            condition_layout=condition_layout,
        )
    base = store.get(row)
    if base is None:
        base = direct.fallback_condition_features(row, condition_dim)
    program = direct.property_program_tokens(row, condition_dim)
    if core.task_mode_for_row(row) != core.EDIT_MODE:
        # This is deliberately the exact P1 condition layout.  It keeps the
        # source-free path protected while the same decoder learns editing.
        return np.concatenate([base, program], axis=0).astype(np.float32)
    edit_token = core.mode_condition_token(core.EDIT_MODE, condition_dim)
    source = core.source_smiles_condition_tokens(
        row, condition_dim, max_source_tokens=max_source_tokens
    )
    return np.concatenate([base, edit_token, source, program], axis=0).astype(np.float32)


_ORIGINAL = core.condition_array_for_row
core.condition_array_for_row = p1_exact_denovo_source_condition

if __name__ == "__main__":
    raise SystemExit(core.main())
