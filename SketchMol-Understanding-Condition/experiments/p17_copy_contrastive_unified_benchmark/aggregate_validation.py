#!/usr/bin/env python3
"""Apply the frozen relaxed P17 gate and retain OOD as a separate diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def compare(base: dict, optimized: dict) -> dict:
    result = {}
    for mode in ("de_novo", "edit"):
        left, right = base["metrics"][mode], optimized["metrics"][mode]
        result[mode] = {
            "p16": left,
            "p17": right,
            "chosen_token_nll_delta_p17_minus_p16": right["chosen_token_nll"] - left["chosen_token_nll"],
            "greedy_validity_delta_p17_minus_p16": right["greedy"]["valid_rate"] - left["greedy"]["valid_rate"],
            "greedy_noncopy_delta_p17_minus_p16": (
                right["greedy"].get("noncopy_rate", 0.0) - left["greedy"].get("noncopy_rate", 0.0)
                if mode == "edit" else None
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p16-id", required=True, type=Path)
    parser.add_argument("--p17-id", required=True, type=Path)
    parser.add_argument("--p16-ood", required=True, type=Path)
    parser.add_argument("--p17-ood", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base_id, opt_id = load(args.p16_id), load(args.p17_id)
    base_ood, opt_ood = load(args.p16_ood), load(args.p17_ood)
    id_cmp, ood_cmp = compare(base_id, opt_id), compare(base_ood, opt_ood)
    failures = []
    denovo = id_cmp["de_novo"]
    edit = id_cmp["edit"]
    if denovo["p17"]["greedy"]["valid_rate"] < 0.80:
        failures.append("ID de-novo P17 greedy validity < 0.80")
    if denovo["greedy_validity_delta_p17_minus_p16"] < -0.10:
        failures.append("ID de-novo greedy validity delta < -0.10")
    if edit["p17"]["greedy"]["valid_rate"] < 0.60:
        failures.append("ID edit P17 greedy validity < 0.60")
    if edit["greedy_validity_delta_p17_minus_p16"] < -0.15:
        failures.append("ID edit greedy validity delta < -0.15")
    if edit["p17"]["greedy"].get("noncopy_rate", 0.0) < 0.35:
        failures.append("ID edit P17 greedy noncopy < 0.35")
    if edit["greedy_noncopy_delta_p17_minus_p16"] < 0.0:
        failures.append("ID edit greedy noncopy delta < 0.00")
    for mode in ("de_novo", "edit"):
        if id_cmp[mode]["p17"]["any_at_3"]["valid_rate"] < 0.90:
            failures.append(f"ID {mode} P17 Any@3 validity < 0.90")
    manifest = load(args.manifest)
    if manifest["id_view"]["source_overlap"] or manifest["id_view"]["target_overlap"]:
        failures.append("ID development leakage")
    if manifest["ood_view"]["exact_condition_overlap"] or manifest["ood_view"]["source_overlap"] or manifest["ood_view"]["target_overlap"]:
        failures.append("OOD development leakage")
    payload = {
        "protocol": "p17_expanded_relaxed_gate_v1",
        "decision": "freeze_and_pilot" if not failures else "freeze_negative_diagnostic_pilot",
        "gate_passed": not failures,
        "gate_view": "ID-condition source-isolated only",
        "gate_failures": failures,
        "id_comparison": id_cmp,
        "ood_diagnostic_comparison": ood_cmp,
        "ood_strict_family_rows": manifest["ood_view"]["strict_family_ood_rows"],
        "ood_exact_condition_fallback_rows": manifest["rows"]["ood_dev_total"] - manifest["ood_view"]["strict_family_ood_rows"],
        "checkpoint_frozen_before_pilot": True,
        "benchmark_tuning_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
