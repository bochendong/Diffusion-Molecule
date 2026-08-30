#!/usr/bin/env python3
"""Apply the preregistered P32.2 dual-mode trajectory-RL gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P321_DIR = SCRIPT_DIR.parent / "p32_1_verifier_routed_residual_rl"
if str(P321_DIR) not in sys.path:
    sys.path.insert(0, str(P321_DIR))
import collect_residual_gate as base  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    results = {
        step: json.loads((args.eval_root / f"step-{step:03d}" / "result.json").read_text())
        for step in (0, 10, 20, 30)
    }
    direct = {
        mode: {
            metric: float(results[0]["direct"][mode][metric])
            for metric in base.METRICS
        }
        for mode in base.MODES
    }
    step0 = base.metrics(results[0])
    candidates = {}
    passed_steps = []
    for step in (10, 20, 30):
        current = base.metrics(results[step])
        rescues = {
            mode: int(results[step]["diagnostics"][mode]["strict_rescues"])
            for mode in base.MODES
        }
        checks, passed = base.assess(current, direct, step0, rescues)
        candidates[str(step)] = {"metrics": current, "checks": checks, "passed": passed}
        if passed:
            passed_steps.append(step)
    selected = None
    if passed_steps:
        selected = max(
            passed_steps,
            key=lambda step: min(
                candidates[str(step)]["checks"][mode]["strict_delta_vs_direct"]
                for mode in base.MODES
            ),
        )
    result = {
        "protocol": "p32_2_multistep_terminal_return_rl_v1",
        "direct": direct,
        "checkpoint_0": step0,
        "candidates": candidates,
        "selected_checkpoint": selected,
        "decision": "PROMOTE_MULTISTEP_TRAJECTORY_RL" if selected is not None else "STOP_MULTISTEP_TRAJECTORY_RL",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "COLLECT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
