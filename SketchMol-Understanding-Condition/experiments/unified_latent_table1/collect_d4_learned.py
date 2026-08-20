#!/usr/bin/env python3
"""D4b localization checks: learned vs matched-count random vs shuffled learned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--learned-summary", required=True, type=Path)
    parser.add_argument("--random-summary", required=True, type=Path)
    parser.add_argument("--shuffled-summary", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def acc(summary: dict, key: str) -> float | None:
    value = summary.get(key)
    if value in ("", None):
        return None
    return float(value)


def pack(summary: dict) -> dict[str, float | None]:
    return {
        "gsk3b_any20_t0_65": acc(summary, "gsk3b_any20_t0_65"),
        "real5_any20_t0_65": acc(summary, "real5_any20_t0_65"),
        "rb_any20_t0_65": acc(summary, "rb_any20_t0_65"),
        "official_gsk3b_any20_t0_65": acc(summary, "official_gsk3b_any20_t0_65"),
        "validity": acc(summary, "validity"),
    }


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    learned = json.loads(args.learned_summary.read_text(encoding="utf-8"))
    random_summary = json.loads(args.random_summary.read_text(encoding="utf-8"))
    shuffled = json.loads(args.shuffled_summary.read_text(encoding="utf-8"))
    b41 = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    learned_real5 = acc(learned, "real5_any20_t0_65")
    random_real5 = acc(random_summary, "real5_any20_t0_65")
    shuffled_real5 = acc(shuffled, "real5_any20_t0_65")
    learned_rb = acc(learned, "rb_any20_t0_65")
    rb_floor = float(prereg["gates"]["rb_not_collapsed"])
    checks = {
        "learned_gt_random_real5": (
            learned_real5 is not None and random_real5 is not None and learned_real5 > random_real5
        ),
        "rb_not_collapsed": learned_rb is not None and learned_rb >= rb_floor,
        "learned_gt_shuffled_real5": (
            learned_real5 is not None and shuffled_real5 is not None and learned_real5 > shuffled_real5
        ),
    }
    passed = all(checks.values())
    payload = {
        "protocol": prereg["protocol"],
        "not_ours": True,
        "decision": "go_localization_learned" if passed else "stop_not_localization",
        "checks": checks,
        "learned_hard": pack(learned),
        "random_matched": pack(random_summary),
        "shuffled_learned": pack(shuffled),
        "b41": pack(b41),
        "claim": (
            "learn where editing should remain possible"
            if passed
            else "learned region did not beat budget/shuffle controls"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
