#!/usr/bin/env python3
"""Evaluate P32.2 through the unchanged P32.1 residual-routing contract."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P321_DIR = SCRIPT_DIR.parent / "p32_1_verifier_routed_residual_rl"
if str(P321_DIR) not in sys.path:
    sys.path.insert(0, str(P321_DIR))
import evaluate_residual_checkpoint as evaluator  # noqa: E402


evaluator.protocol.PROTOCOL = "p32_2_multistep_terminal_return_rl_v1"


if __name__ == "__main__":
    raise SystemExit(evaluator.main())
