#!/usr/bin/env python3
"""Evaluate P32.3 with verifier-confirmed strict success as an absorbing state."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P321_DIR = SCRIPT_DIR.parent / "p32_1_verifier_routed_residual_rl"
if str(P321_DIR) not in sys.path:
    sys.path.insert(0, str(P321_DIR))
import evaluate_residual_checkpoint as evaluator  # noqa: E402


PROTOCOL = "p32_3_strict_absorbing_exploration_rl_v1"


def strict_absorbing_greedy_rollout(
    model,
    tokenizer,
    record,
    *,
    max_steps,
    max_actions,
    site_limit,
    max_length,
    score_batch_size,
):
    protocol = evaluator.protocol
    if protocol.hard_accept_direct(record):
        smiles = protocol.direct_smiles(record)
        return smiles, [{
            "step": -1,
            "kind": "hard_accept_direct",
            "smiles": smiles,
            "strict": True,
        }]
    current = protocol.initial_smiles(record)
    history = []
    trace = []
    for step_index in range(int(max_steps)):
        bundle = protocol.support_bundle(
            model,
            tokenizer,
            record,
            current_smiles=current,
            history=history,
            step_index=step_index,
            max_steps=max_steps,
            max_actions=max_actions,
            site_limit=site_limit,
            max_length=max_length,
            score_batch_size=score_batch_size,
        )
        if bundle is None:
            break
        selected = max(
            range(len(bundle.support.action_scores)),
            key=lambda index: float(bundle.support.action_scores[index]),
        )
        action = bundle.actions[selected]
        feedback = bundle.support.feedback[selected]
        current = action.next_smiles
        row = {
            "step": step_index,
            "action": action.payload,
            "kind": action.kind,
            "smiles": current,
            "score": float(bundle.support.action_scores[selected]),
            "reward": feedback.reward,
            "strict": feedback.strict_success,
            "absorbing_strict": bool(feedback.strict_success),
        }
        trace.append(row)
        history.append({"tool_call": action.payload, "result_smiles": current, "observation": row})
        if action.terminal or feedback.strict_success:
            break
    return current, trace


evaluator.protocol.PROTOCOL = PROTOCOL
evaluator.protocol.greedy_rollout = strict_absorbing_greedy_rollout


if __name__ == "__main__":
    raise SystemExit(evaluator.main())
