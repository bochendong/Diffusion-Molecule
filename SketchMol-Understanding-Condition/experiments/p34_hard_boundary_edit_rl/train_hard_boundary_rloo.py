#!/usr/bin/env python3
"""Run online sequence RLOO with P34's hard source-feasibility boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P31_DIR = SCRIPT_DIR.parent / "p31_1_frontier_online_rloo"
if str(P31_DIR) not in sys.path:
    sys.path.insert(0, str(P31_DIR))
import train_online_rloo as p31  # noqa: E402
from hard_boundary_reward import hard_boundary_reward  # noqa: E402


PROTOCOL = "p34_hard_boundary_edit_online_rloo_v1"


def option_value(argv: Sequence[str], option: str) -> str:
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing required option {option}") from error


def relabel_states(output_dir: Path) -> None:
    for state_path in output_dir.glob("checkpoint-*/state.json"):
        state = json.loads(state_path.read_text())
        state.update(
            {
                "protocol": PROTOCOL,
                "initialization": "p29_editing_specialist",
                "reward": "hard_source_boundary_strict_dominant_v1",
                "source_feasibility_threshold": 0.65,
            }
        )
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if "--mode" in forwarded:
        raise ValueError("P34 fixes --mode edit; do not pass --mode")
    output_dir = Path(option_value(forwarded, "--output-dir"))
    p31.scalar_reward = hard_boundary_reward
    result = p31.main(["--mode", "edit", *forwarded])
    relabel_states(output_dir)
    (output_dir / "P34_PROTOCOL.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "initialization": "p29_editing_specialist",
                "construction_policy": "frozen and unchanged",
                "rl_algorithm": "online sequence RLOO with fresh rollouts",
                "hard_boundary": "valid, non-copy, source similarity >= 0.65",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
