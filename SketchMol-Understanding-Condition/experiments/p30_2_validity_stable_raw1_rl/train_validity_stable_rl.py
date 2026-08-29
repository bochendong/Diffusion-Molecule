#!/usr/bin/env python3
"""Run P30 balanced shared-policy RL with preregistered validity weights."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P30_DIR = SCRIPT_DIR.parent / "p30_balanced_shared_policy_rl"
P26_DIR = SCRIPT_DIR.parent / "p26_decoupled_joint_rl"
for path in (P30_DIR, P26_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_balanced_shared_rl as p30  # noqa: E402


def main() -> int:
    p30.p26.CHANNEL_WEIGHTS["de_novo"]["validity"] = 1.50
    p30.p26.CHANNEL_WEIGHTS["de_novo"]["canonical"] = 0.25
    return p30.main()


if __name__ == "__main__":
    raise SystemExit(main())

