#!/usr/bin/env python3
"""Frontier vs singleton next-event gates on Table1 n=20 real5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REAL_TASK_KEYS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "MW:increase",
    "RB:decrease",
    "SA:decrease",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--frontier-summary", required=True, type=Path)
    parser.add_argument("--canonical-summary", required=True, type=Path)
    parser.add_argument("--random-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def acc(summary: dict, key: str) -> float | None:
    value = summary.get(key)
    if value in ("", None):
        return None
    return float(value)


def task_acc(summary: dict, task_key: str) -> float | None:
    by_task = dict(summary.get("by_task") or {})
    row = by_task.get(task_key) or {}
    value = row.get("acc_all_0_65")
    if value in ("", None):
        return None
    return float(value)


def pack(summary: dict) -> dict[str, float | None]:
    return {
        "gsk3b_any20_t0_65": acc(summary, "gsk3b_any20_t0_65"),
        "real5_any20_t0_65": acc(summary, "real5_any20_t0_65"),
        "rb_any20_t0_65": acc(summary, "rb_any20_t0_65"),
        "validity": acc(summary, "validity"),
        "by_task_acc_0_65": {key: task_acc(summary, key) for key in REAL_TASK_KEYS},
    }


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    frontier = json.loads(args.frontier_summary.read_text(encoding="utf-8"))
    canonical = json.loads(args.canonical_summary.read_text(encoding="utf-8"))
    random_singleton = json.loads(args.random_summary.read_text(encoding="utf-8"))
    gates = dict(prereg["gates"])
    frontier_real5 = acc(frontier, "real5_any20_t0_65")
    canonical_real5 = acc(canonical, "real5_any20_t0_65")
    random_real5 = acc(random_singleton, "real5_any20_t0_65")
    margin = float(gates["real5_margin"])
    slack = float(gates["task_slack"])
    task_ok = 0
    task_rows = {}
    for key in REAL_TASK_KEYS:
        f_acc = task_acc(frontier, key)
        c_acc = task_acc(canonical, key)
        r_acc = task_acc(random_singleton, key)
        task_rows[key] = {"frontier": f_acc, "canonical": c_acc, "random_singleton": r_acc}
        if f_acc is None or c_acc is None or r_acc is None:
            continue
        if f_acc + slack >= min(c_acc, r_acc):
            task_ok += 1
    checks = {
        "frontier_gt_canonical_real5": (
            frontier_real5 is not None
            and canonical_real5 is not None
            and frontier_real5 > canonical_real5 + margin
        ),
        "frontier_gt_random_real5": (
            frontier_real5 is not None
            and random_real5 is not None
            and frontier_real5 > random_real5 + margin
        ),
        "majority_tasks_not_worse": task_ok >= int(gates["tasks_not_worse"]),
        "validity_canonical": (acc(canonical, "validity") or 0.0) >= float(gates["validity"]),
        "validity_random": (acc(random_singleton, "validity") or 0.0) >= float(gates["validity"]),
    }
    passed = all(checks.values())
    payload = {
        "protocol": prereg["protocol"],
        "decision": "keep_frontier_objective" if passed else "frontier_objective_dies",
        "checks": checks,
        "tasks_not_worse": task_ok,
        "by_task": task_rows,
        "frontier": pack(frontier),
        "canonical": pack(canonical),
        "random_singleton": pack(random_singleton),
        "claim": prereg["claim_if_pass"] if passed else prereg["claim_if_fail"],
        "second_seed_if_pass": prereg.get("second_seed_if_pass"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
