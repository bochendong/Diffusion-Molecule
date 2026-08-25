#!/usr/bin/env python3
"""Compare unified mixed SFT with matched single-mode P16 LoRA controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", required=True, type=Path)
    parser.add_argument("--denovo", required=True, type=Path)
    parser.add_argument("--edit", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    mixed, singles = load(args.mixed), {"de_novo": load(args.denovo), "edit": load(args.edit)}
    comparisons = {}
    gate_failures: list[str] = []
    for mode in ("de_novo", "edit"):
        unified = mixed["metrics"][mode]
        single = singles[mode]["metrics"][mode]
        nll_delta = unified["assistant_token_nll"] - single["assistant_token_nll"]
        valid_delta = unified["greedy"]["valid_rate"] - single["greedy"]["valid_rate"]
        comparisons[mode] = {
            "mixed_minus_single_token_nll": nll_delta,
            "mixed_minus_single_greedy_validity": valid_delta,
            "mixed": unified,
            "matched_single_mode": single,
        }
        if nll_delta > 0.50:
            gate_failures.append(f"{mode}: token-NLL negative transfer {nll_delta:.4f} > 0.50")
        if valid_delta < -0.10:
            gate_failures.append(f"{mode}: greedy-validity negative transfer {valid_delta:.4f} < -0.10")
        if unified["greedy"]["valid_rate"] < 0.50:
            gate_failures.append(f"{mode}: mixed greedy validity < 0.50")
        any_key = f"any_at_{mixed['fixed_k']}"
        if unified[any_key]["valid_rate"] < 0.75:
            gate_failures.append(f"{mode}: mixed Any@{mixed['fixed_k']} validity < 0.75")
    edit_noncopy = mixed["metrics"]["edit"]["greedy"].get("noncopy_rate", 0.0)
    if edit_noncopy < 0.50:
        gate_failures.append("edit: mixed greedy noncopy < 0.50")
    manifest = load(args.manifest)
    if manifest["condition_hash_overlap"] or manifest["source_hash_overlap"]:
        gate_failures.append("split isolation failure")
    payload = {
        "protocol": "p16_mixed_vs_single_negative_transfer_gate_v1",
        "decision": "advance_to_larger_train_only_sft" if not gate_failures else "stop_negative_gate",
        "gate_failures": gate_failures,
        "comparisons": comparisons,
        "canonical_exact_is_diagnostic_not_selection": True,
        "sft_gate_passed": not gate_failures,
        "preference_training_started": False,
        "no_static_pool": True,
        "no_property_reranking": True,
        "official_test_access": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
