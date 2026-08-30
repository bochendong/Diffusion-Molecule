#!/usr/bin/env python3
"""Apply the preregistered P32 dual-mode promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


MODES = ("de_novo", "edit")


def checkpoint_metrics(result: Mapping[str, object]) -> dict[str, dict[str, float]]:
    policy = result["policy"]
    return {
        mode: {
            metric: float(policy[mode][metric])
            for metric in ("strict_macro", "relaxed_macro", "valid_macro")
        }
        for mode in MODES
    }


def assess(candidate, direct, step0):
    checks = {
        mode: {
            "strict_delta_vs_direct": candidate[mode]["strict_macro"] - direct[mode]["strict_macro"],
            "strict_delta_vs_step0": candidate[mode]["strict_macro"] - step0[mode]["strict_macro"],
            "valid_delta_vs_direct": candidate[mode]["valid_macro"] - direct[mode]["valid_macro"],
        }
        for mode in MODES
    }
    passed = all(
        values["strict_delta_vs_direct"] > 0.0
        and values["strict_delta_vs_step0"] > 0.0
        and values["valid_delta_vs_direct"] >= -0.01 - 1e-12
        for values in checks.values()
    )
    return checks, passed


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
            for metric in ("strict_macro", "relaxed_macro", "valid_macro")
        }
        for mode in MODES
    }
    step0 = checkpoint_metrics(results[0])
    candidates = {}
    passed_steps = []
    for step in (10, 20, 30):
        metrics = checkpoint_metrics(results[step])
        checks, passed = assess(metrics, direct, step0)
        candidates[str(step)] = {"metrics": metrics, "checks": checks, "passed": passed}
        if passed:
            passed_steps.append(step)
    selected = None
    if passed_steps:
        selected = max(
            passed_steps,
            key=lambda step: min(
                candidates[str(step)]["checks"][mode]["strict_delta_vs_direct"]
                for mode in MODES
            ),
        )
    result = {
        "protocol": "p32_unified_graph_repair_rl_v1",
        "direct": direct,
        "checkpoint_0": step0,
        "candidates": candidates,
        "selected_checkpoint": selected,
        "decision": "SCALE_SHARED_GRAPH_RL" if selected is not None else "STOP_SHARED_GRAPH_RL_PILOT",
        "promotion_rule": {
            "strict_above_direct_in_both_modes": True,
            "strict_above_checkpoint_0_in_both_modes": True,
            "validity_drop_max": 0.01,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "COLLECT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
