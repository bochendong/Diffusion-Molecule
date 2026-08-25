#!/usr/bin/env python3
"""Compare frozen P16/P17/P18 on the exact paired P17 ID and OOD views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def compact(payload: dict, mode: str) -> dict:
    block = payload["metrics"][mode]
    out = {
        "chosen_token_nll": block["chosen_token_nll"],
        "greedy_validity": block["greedy"]["valid_rate"],
        "any_at_3_validity": block["any_at_3"]["valid_rate"],
    }
    if mode == "edit":
        out.update({
            "greedy_noncopy": block["greedy"]["noncopy_rate"],
            "greedy_source_similarity": block["greedy"].get("mean_source_similarity"),
            "any_at_3_noncopy": block["any_at_3"].get("noncopy_rate"),
            "chosen_preferred_to_copy": block.get("chosen_preferred_to_copy_rate"),
            "mean_chosen_minus_copy_nll": block.get("mean_chosen_minus_copy_nll"),
        })
    return out


def view(p16: dict, p17: dict, p18: dict) -> dict:
    result = {}
    for mode in ("de_novo", "edit"):
        blocks = {"p16": compact(p16, mode), "p17": compact(p17, mode), "p18": compact(p18, mode)}
        keys = sorted(set(blocks["p18"]) & set(blocks["p16"]) & set(blocks["p17"]))
        blocks["delta_p18_minus_p16"] = {key: blocks["p18"][key] - blocks["p16"][key] for key in keys if blocks["p18"][key] is not None}
        blocks["delta_p18_minus_p17"] = {key: blocks["p18"][key] - blocks["p17"][key] for key in keys if blocks["p18"][key] is not None}
        result[mode] = blocks
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ("p16", "p17", "p18"):
        for split in ("id", "ood"):
            parser.add_argument(f"--{model}-{split}", required=True, type=Path)
    parser.add_argument("--p17-manifest", required=True, type=Path)
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {model: {split: load(getattr(args, f"{model}_{split}")) for split in ("id", "ood")} for model in ("p16", "p17", "p18")}
    id_cmp = view(inputs["p16"]["id"], inputs["p17"]["id"], inputs["p18"]["id"])
    ood_cmp = view(inputs["p16"]["ood"], inputs["p17"]["ood"], inputs["p18"]["ood"])
    p16_edit = id_cmp["edit"]["p16"]
    p18_edit = id_cmp["edit"]["p18"]
    p18_denovo = id_cmp["de_novo"]["p18"]
    checks = {
        "de_novo_greedy_validity_ge_0.90": p18_denovo["greedy_validity"] >= 0.90,
        "edit_greedy_validity_ge_0.875": p18_edit["greedy_validity"] >= 0.875,
        "edit_greedy_validity_delta_vs_p16_ge_minus_0.10": p18_edit["greedy_validity"] - p16_edit["greedy_validity"] >= -0.10,
        "edit_greedy_noncopy_ge_0.70": p18_edit["greedy_noncopy"] >= 0.70,
        "de_novo_any_at_3_validity_ge_0.9375": p18_denovo["any_at_3_validity"] >= 0.9375,
        "edit_any_at_3_validity_ge_0.9375": p18_edit["any_at_3_validity"] >= 0.9375,
    }
    manifest = load(args.p17_manifest)
    input_audit = load(args.input_audit)
    checks["locked_input_hashes_match"] = bool(input_audit["all_hashes_match"])
    checks["id_source_target_isolated"] = not manifest["id_view"]["source_overlap"] and not manifest["id_view"]["target_overlap"]
    checks["ood_condition_source_target_isolated"] = (
        not manifest["ood_view"]["exact_condition_overlap"]
        and not manifest["ood_view"]["source_overlap"]
        and not manifest["ood_view"]["target_overlap"]
    )
    payload = {
        "protocol": "p18_paired_operational_gate_v1",
        "decision": "freeze_and_pilot" if all(checks.values()) else "freeze_negative_diagnostic_pilot",
        "gate_passed": all(checks.values()),
        "gate_checks": checks,
        "gate_failures": [key for key, passed in checks.items() if not passed],
        "id_comparison": id_cmp,
        "strict_family_ood_comparison": ood_cmp,
        "ood_strict_family_rows": manifest["ood_view"]["strict_family_ood_rows"],
        "checkpoint_frozen_before_pilot": True,
        "benchmark_tuning_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
