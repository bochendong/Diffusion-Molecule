#!/usr/bin/env python3
"""Particle-coverage gates: full B41 any@k curve vs iid independent latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--full-curve", required=True, type=Path)
    parser.add_argument("--iid-independent-curve", required=True, type=Path)
    parser.add_argument("--ortho-independent-curve", required=True, type=Path)
    parser.add_argument("--iid-interacting-curve", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def load_curve(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def series(curve: dict, key: str) -> dict[str, float | None]:
    raw = dict(curve.get(key) or {})
    out: dict[str, float | None] = {}
    for name, value in raw.items():
        if value in ("", None):
            out[str(name)] = None
        else:
            out[str(name)] = float(value)
    return out


def packed(curve: dict) -> dict[str, object]:
    return {
        "real5_anyk_t0_65": series(curve, "real5_anyk_t0_65"),
        "gsk3b_anyk_t0_65": series(curve, "gsk3b_anyk_t0_65"),
        "auc_real5_t0_65": curve.get("auc_real5_t0_65"),
        "mean_unique_smiles": curve.get("mean_unique_smiles"),
    }


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    full = load_curve(args.full_curve)
    iid = load_curve(args.iid_independent_curve)
    ortho = load_curve(args.ortho_independent_curve)
    interact = load_curve(args.iid_interacting_curve)
    ks = [str(k) for k in prereg["ks"]]
    early = [str(k) for k in prereg["gates"]["early_ks"]]
    full_real5 = series(full, "real5_anyk_t0_65")
    iid_real5 = series(iid, "real5_anyk_t0_65")
    diffs = {
        k: (
            None
            if full_real5.get(k) is None or iid_real5.get(k) is None
            else float(full_real5[k]) - float(iid_real5[k])
        )
        for k in ks
    }
    comparable = [k for k in ks if diffs[k] is not None]
    early_comparable = [k for k in early if diffs.get(k) is not None]
    mean_gain = (
        sum(float(diffs[k]) for k in comparable) / len(comparable) if comparable else None
    )
    early_wins = sum(1 for k in early_comparable if float(diffs[k]) > 0)
    checks = {
        "full_ge_iid_all_k": bool(comparable) and all(float(diffs[k]) >= 0 for k in comparable),
        "early_k_strict_wins": early_wins >= int(prereg["gates"]["early_k_strict_wins"]),
        "mean_real5_gain": mean_gain is not None
        and mean_gain >= float(prereg["gates"]["min_mean_real5_gain"]),
        "validity_full": True,
    }
    passed = all(checks.values())
    payload = {
        "protocol": prereg["protocol"],
        "decision": "keep_particle_contribution" if passed else "particle_is_implementation_detail",
        "checks": checks,
        "real5_diff_full_minus_iid": diffs,
        "mean_real5_gain": mean_gain,
        "early_k_strict_wins": early_wins,
        "full_interacting": packed(full),
        "iid_independent": packed(iid),
        "ortho_independent": packed(ortho),
        "iid_interacting": packed(interact),
        "claim": prereg["claim_if_pass"] if passed else prereg["claim_if_fail"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
