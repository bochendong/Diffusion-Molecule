#!/usr/bin/env python3
"""D5a gates: property eta vs constant eta vs frozen B41. Diagnostic only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--property-summary", required=True, type=Path)
    parser.add_argument("--constant-summary", required=True, type=Path)
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
    property_summary = json.loads(args.property_summary.read_text(encoding="utf-8"))
    constant_summary = json.loads(args.constant_summary.read_text(encoding="utf-8"))
    b41 = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    gates = dict(prereg["gates"])
    property_gsk = acc(property_summary, "gsk3b_any20_t0_65")
    property_real5 = acc(property_summary, "real5_any20_t0_65")
    property_rb = acc(property_summary, "rb_any20_t0_65")
    constant_real5 = acc(constant_summary, "real5_any20_t0_65")
    b41_gsk = acc(b41, "gsk3b_any20_t0_65")
    b41_real5 = acc(b41, "real5_any20_t0_65")
    checks = {
        "gsk3b_gt_b41_plus_margin": (
            property_gsk is not None
            and b41_gsk is not None
            and property_gsk >= b41_gsk + float(gates["gsk3b_margin"])
        ),
        "real5_gt_b41": (
            property_real5 is not None
            and b41_real5 is not None
            and property_real5 > b41_real5 + float(gates["real5_margin"])
        ),
        "rb_not_dropped": property_rb is not None and property_rb >= float(gates["rb_floor"]),
        "property_gt_constant_real5": (
            property_real5 is not None
            and constant_real5 is not None
            and property_real5 > constant_real5
        ),
        "validity": (acc(property_summary, "validity") or 0.0) >= float(gates["validity"]),
    }
    passed = all(checks.values())
    payload = {
        "protocol": prereg["protocol"],
        "not_ours": True,
        "not_a_method_contribution": True,
        "decision": "go_magnitude_signal" if passed else "stop_no_magnitude_signal",
        "checks": checks,
        "property_alpha": pack(property_summary),
        "constant_eta": pack(constant_summary),
        "b41": pack(b41),
        "claim": (
            "property-conditioned preservation strength has signal"
            if passed
            else "global soft eta did not beat B41; do not upgrade to trust region"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
