#!/usr/bin/env python3
"""Audit target-blind two-step GraphEditDSL support before P32 RL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import graph_repair_protocol as protocol  # noqa: E402


def oracle_search(record, *, steps: int, max_actions: int, site_limit: int, beam_size: int):
    direct_details = dict(record["direct_details"])
    direct_smiles = str(record.get("direct_smiles", "") or "")
    baseline_reward = protocol.score_smiles(record, direct_smiles).reward if direct_smiles else -1.0
    best = {
        "smiles": direct_smiles,
        "reward": baseline_reward,
        "strict": bool(direct_details.get("strict")),
        "valid": bool(direct_details.get("valid")),
        "depth": 0,
    }
    beams = [(protocol.initial_smiles(record), baseline_reward)]
    seen = {protocol.initial_smiles(record), direct_smiles}
    evaluated = 0
    for depth in range(1, int(steps) + 1):
        candidates = []
        for current, _parent_reward in beams:
            for action in protocol.executable_actions(
                record, current, step_index=depth - 1,
                max_actions=max_actions, site_limit=site_limit,
            ):
                if action.next_smiles in seen and action.kind != "stop":
                    continue
                seen.add(action.next_smiles)
                feedback = protocol.score_smiles(record, action.next_smiles)
                evaluated += 1
                item = {
                    "smiles": action.next_smiles,
                    "reward": feedback.reward,
                    "strict": feedback.strict_success,
                    "valid": feedback.valid,
                    "depth": depth,
                }
                candidates.append(item)
                if (item["strict"], item["reward"]) > (best["strict"], best["reward"]):
                    best = item
        candidates.sort(key=lambda item: (bool(item["strict"]), float(item["reward"])), reverse=True)
        beams = [(str(item["smiles"]), float(item["reward"])) for item in candidates[:beam_size]]
        if not beams:
            break
    return {
        "example_id": record.get("example_id", ""),
        "task_mode": record["task_mode"],
        "bucket": record["bucket"],
        "baseline_strict": bool(direct_details.get("strict")),
        "baseline_valid": bool(direct_details.get("valid")),
        "baseline_reward": baseline_reward,
        "oracle_strict": bool(best["strict"]),
        "oracle_valid": bool(best["valid"]),
        "oracle_reward": float(best["reward"]),
        "oracle_depth": int(best["depth"]),
        "strict_opportunity": bool(not direct_details.get("strict") and best["strict"]),
        "reward_improved": float(best["reward"]) > baseline_reward + 1e-9,
        "actions_evaluated": evaluated,
    }


def summarize(rows: Sequence[Mapping[str, object]]):
    return {
        "rows": len(rows),
        "baseline_strict_rate": protocol.mean(bool(row["baseline_strict"]) for row in rows),
        "oracle_strict_rate": protocol.mean(bool(row["oracle_strict"]) for row in rows),
        "strict_opportunity_rate": protocol.mean(bool(row["strict_opportunity"]) for row in rows),
        "reward_improvement_rate": protocol.mean(bool(row["reward_improved"]) for row in rows),
        "mean_actions_evaluated": protocol.mean(float(row["actions_evaluated"]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--min-opportunity-rate", type=float, default=0.03)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    records = protocol.read_jsonl(args.gate_jsonl)
    rows = []
    for index, record in enumerate(records):
        rows.append(oracle_search(
            record, steps=args.steps, max_actions=args.max_actions,
            site_limit=args.site_limit, beam_size=args.beam_size,
        ))
        if (index + 1) % 10 == 0:
            print(f"[p32-support] {index + 1}/{len(records)}", flush=True)
    by_mode = {
        mode: summarize([row for row in rows if row["task_mode"] == mode])
        for mode in ("de_novo", "edit")
    }
    gates = {
        mode: float(by_mode[mode]["strict_opportunity_rate"]) >= args.min_opportunity_rate
        for mode in by_mode
    }
    result = {
        "protocol": protocol.PROTOCOL,
        "audit": "two_step_target_blind_graph_action_support",
        "by_mode": by_mode,
        "gates": gates,
        "decision": "RUN_SHARED_RL" if all(gates.values()) else "STOP_NO_SHARED_ACTION_SUPPORT",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_jsonl(args.output_dir / "rows.jsonl", rows)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if args.require_pass and not all(gates.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
