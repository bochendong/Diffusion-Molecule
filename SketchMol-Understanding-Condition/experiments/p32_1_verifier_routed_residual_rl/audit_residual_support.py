#!/usr/bin/env python3
"""Audit whether failed direct proposals have target-blind graph-action rescues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import residual_protocol as protocol  # noqa: E402


def oracle_search(record, *, steps: int, max_actions: int, site_limit: int, beam_size: int):
    baseline = protocol.direct_feedback(record)
    if baseline.strict_success:
        return {
            "example_id": record["example_id"],
            "task_mode": record["task_mode"],
            "bucket": record["bucket"],
            "baseline_strict": True,
            "oracle_strict": True,
            "strict_rescue": False,
            "actions_evaluated": 0,
        }
    best = baseline
    beams = [(protocol.initial_smiles(record), baseline.reward)]
    seen = {protocol.initial_smiles(record)}
    evaluated = 0
    for depth in range(int(steps)):
        candidates = []
        for current, _reward in beams:
            for action in protocol.executable_actions(
                record,
                current,
                step_index=depth,
                max_actions=max_actions,
                site_limit=site_limit,
            ):
                if action.next_smiles in seen and action.kind != "stop":
                    continue
                seen.add(action.next_smiles)
                feedback = protocol.score_smiles(record, action.next_smiles)
                evaluated += 1
                candidates.append((action.next_smiles, feedback))
                if (feedback.strict_success, feedback.reward) > (best.strict_success, best.reward):
                    best = feedback
        candidates.sort(
            key=lambda item: (item[1].strict_success, item[1].reward), reverse=True
        )
        beams = [(smiles, feedback.reward) for smiles, feedback in candidates[:beam_size]]
        if not beams:
            break
    return {
        "example_id": record["example_id"],
        "task_mode": record["task_mode"],
        "bucket": record["bucket"],
        "baseline_strict": False,
        "oracle_strict": best.strict_success,
        "strict_rescue": best.strict_success,
        "actions_evaluated": evaluated,
    }


def summarize(rows):
    failed = [row for row in rows if not row["baseline_strict"]]
    rescues = sum(bool(row["strict_rescue"]) for row in rows)
    return {
        "rows": len(rows),
        "failed_direct": len(failed),
        "strict_rescues": rescues,
        "strict_rescue_rate_all": rescues / max(len(rows), 1),
        "strict_rescue_rate_failed": rescues / max(len(failed), 1),
        "oracle_strict_rate_with_hard_route": sum(bool(row["oracle_strict"]) for row in rows) / max(len(rows), 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--minimum-rescues", type=int, default=1)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    records = protocol.read_jsonl(args.gate_jsonl)
    rows = []
    for index, record in enumerate(records):
        rows.append(oracle_search(
            record,
            steps=args.steps,
            max_actions=args.max_actions,
            site_limit=args.site_limit,
            beam_size=args.beam_size,
        ))
        if (index + 1) % 10 == 0:
            print(f"[p32.1-support] {index + 1}/{len(records)}", flush=True)
    by_mode = {
        mode: summarize([row for row in rows if row["task_mode"] == mode])
        for mode in ("de_novo", "edit")
    }
    gates = {
        mode: int(summary["strict_rescues"]) >= int(args.minimum_rescues)
        for mode, summary in by_mode.items()
    }
    result = {
        "protocol": protocol.PROTOCOL,
        "by_mode": by_mode,
        "gates": gates,
        "decision": "RUN_RESIDUAL_RL" if all(gates.values()) else "STOP_NO_RESIDUAL_SUPPORT",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_jsonl(args.output_dir / "rows.jsonl", rows)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if args.require_pass and not all(gates.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
