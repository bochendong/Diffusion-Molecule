#!/usr/bin/env python3
"""Compare D4 oracle-hard vs matched-random vs locked B41. Diagnostic only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--oracle-summary", required=True, type=Path)
    parser.add_argument("--random-summary", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def acc(summary: dict, key: str) -> float | None:
    value = summary.get(key)
    if value in ("", None):
        return None
    return float(value)


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle_summary.read_text(encoding="utf-8"))
    random_summary = json.loads(args.random_summary.read_text(encoding="utf-8"))
    b41 = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    margin = float(prereg["gates"]["oracle_beats_b41_gsk3b"])
    oracle_gsk = acc(oracle, "gsk3b_any20_t0_65")
    random_gsk = acc(random_summary, "gsk3b_any20_t0_65")
    b41_gsk = acc(b41, "gsk3b_any20_t0_65")
    oracle_real5 = acc(oracle, "real5_any20_t0_65")
    random_real5 = acc(random_summary, "real5_any20_t0_65")
    b41_real5 = acc(b41, "real5_any20_t0_65")
    headroom = (
        oracle_gsk is not None
        and b41_gsk is not None
        and oracle_gsk >= b41_gsk + margin
        and random_gsk is not None
        and oracle_gsk > random_gsk
    )
    payload = {
        "protocol": prereg["protocol"],
        "not_ours": True,
        "decision": "go_train_learned_mask" if headroom else "no_localization_headroom",
        "headroom": headroom,
        "oracle_hard": {
            "gsk3b_any20_t0_65": oracle_gsk,
            "real5_any20_t0_65": oracle_real5,
            "rb_any20_t0_65": acc(oracle, "rb_any20_t0_65"),
            "official_gsk3b_any20_t0_65": acc(oracle, "official_gsk3b_any20_t0_65"),
            "validity": acc(oracle, "validity"),
        },
        "random_matched": {
            "gsk3b_any20_t0_65": random_gsk,
            "real5_any20_t0_65": random_real5,
            "rb_any20_t0_65": acc(random_summary, "rb_any20_t0_65"),
            "official_gsk3b_any20_t0_65": acc(random_summary, "official_gsk3b_any20_t0_65"),
            "validity": acc(random_summary, "validity"),
        },
        "b41": {
            "gsk3b_any20_t0_65": b41_gsk,
            "real5_any20_t0_65": b41_real5,
            "rb_any20_t0_65": acc(b41, "rb_any20_t0_65"),
            "official_gsk3b_any20_t0_65": acc(b41, "official_gsk3b_any20_t0_65"),
        },
        "next": (
            "Train (x,p)→atom/bond P(EDIT). Hard learned vs matched random. Soft only if hard helps."
            if headroom
            else "Do not train a planner. Localization is not the bottleneck; do not add Δ or operations."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
