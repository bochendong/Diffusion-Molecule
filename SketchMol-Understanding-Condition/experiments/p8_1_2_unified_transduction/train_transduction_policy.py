#!/usr/bin/env python3
"""Train the existing single UMTP decoder with the P8.1.2 vocabulary."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
sys.path.insert(0, str(UNIFIED_DIR))

import selfies  # noqa: E402
import umtp_graph_action_policy as policy  # noqa: E402


BASE_VOCABULARY = policy.action_vocabulary


def transduction_vocabulary(*, max_site_index: int = 127) -> list[str]:
    del max_site_index
    tokens = ["<TRANSDUCE>", "<INSERT>", "<INSERT_END>", "<STOP>"]
    tokens.extend(f"<KEEP_{count}>" for count in range(1, 189))
    tokens.extend(f"<DELETE_{count}>" for count in range(1, 189))
    tokens.extend(sorted(selfies.get_semantic_robust_alphabet()))
    # Retaining legacy rows makes expansion checkpoint-safe; P8.1.2 targets
    # and its constrained sampler expose only the transduction sublanguage.
    return list(dict.fromkeys([*BASE_VOCABULARY(max_site_index=127), *tokens]))


policy.action_vocabulary = transduction_vocabulary


if __name__ == "__main__":
    raise SystemExit(policy.main())
