#!/usr/bin/env python3
"""Thin P8.1.4 entrypoint: direct-compatible goal tokens plus explicit task token."""

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


_original = core.condition_array_for_row


def direct_goal_with_task_token(row, store, condition_dim, *, max_source_tokens, condition_layout="unified"):
    if str(condition_layout) != "unified":
        return _original(
            row,
            store,
            condition_dim,
            max_source_tokens=max_source_tokens,
            condition_layout=condition_layout,
        )
    base = store.get(row)
    if base is None:
        base = direct.fallback_condition_features(row, condition_dim)
    mode = core.task_mode_for_row(row)
    task_token = core.mode_condition_token(mode, condition_dim)
    program = direct.property_program_tokens(row, condition_dim)
    pieces = [base, task_token]
    if mode == core.EDIT_MODE:
        pieces.append(core.source_smiles_condition_tokens(row, condition_dim, max_source_tokens=max_source_tokens))
    pieces.append(program)
    return np.concatenate(pieces, axis=0).astype(np.float32)


core.condition_array_for_row = direct_goal_with_task_token

if __name__ == "__main__":
    raise SystemExit(core.main())
