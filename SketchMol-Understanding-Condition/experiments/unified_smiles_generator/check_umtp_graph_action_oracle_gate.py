#!/usr/bin/env python3
"""Gate instruction-aligned GraphEditDSL training on oracle reachability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--required-task", default="GSK3B:increase")
    parser.add_argument("--min-fully-evaluable-rate", type=float, default=0.95)
    parser.add_argument("--min-strict-reachability", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task_rows = dict(manifest.get("instruction_oracle_by_task", {}))
    required = dict(task_rows.get(str(args.required_task), {}))
    fully_evaluable = float(required.get("fully_evaluable_rate", 0.0) or 0.0)
    reachability = float(required.get("strict_reachability", 0.0) or 0.0)
    passes = (
        bool(required)
        and fully_evaluable >= float(args.min_fully_evaluable_rate)
        and reachability >= float(args.min_strict_reachability)
    )
    result = {
        "protocol": "umtp_graph_action_instruction_oracle_gate_v2",
        "decision": "go" if passes else "stop",
        "required_task": str(args.required_task),
        "criteria": {
            "min_fully_evaluable_rate": float(args.min_fully_evaluable_rate),
            "min_strict_reachability": float(args.min_strict_reachability),
        },
        "observed": required,
        "manifest": str(args.manifest),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passes else 3


if __name__ == "__main__":
    raise SystemExit(main())
