#!/usr/bin/env python3
"""Merge the four preregistered common-LLM latent-operator signal arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL = "train_only_common_llm_latent_operator_signal_v1"
ARMS = ("property_mlp", "base_frozen", "sft_frozen", "sft_lora")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--valid-terminal-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_contract_failures(arm: str, summary: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if summary.get("protocol") != PROTOCOL:
        failures.append(f"{arm}:protocol")
    if summary.get("arm") != arm:
        failures.append(f"{arm}:arm")
    if summary.get("decision") != "await_cross_arm_llm_latent_operator_gate":
        failures.append(f"{arm}:decision")
    metrics = dict(summary.get("metrics", {}))
    manifest = dict(summary.get("manifest", {}))
    if int(metrics.get("attempted_per_condition", -1)) != 20:
        failures.append(f"{arm}:attempts")
    if int(metrics.get("candidate_rows", -1)) != 4700:
        failures.append(f"{arm}:candidate_rows")
    if int(metrics.get("conditions", -1)) != 235:
        failures.append(f"{arm}:conditions")
    required_false = (
        "common_llm_emits_text_or_actions",
        "molecular_candidate_ranking",
        "oracle_selection",
        "retry_or_resampling",
        "posthoc_molecule_repair",
        "generation_target_access",
        "generation_property_oracle_access",
        "b26_heldout_access",
        "b33_fresh_source_access",
        "moledit_table1_benchmark_access",
        "official_test_access",
    )
    for name in required_false:
        if manifest.get(name) is not False:
            failures.append(f"{arm}:{name}")
    if manifest.get("frozen_before_target_or_property_evaluation") is not True:
        failures.append(f"{arm}:freeze_before_evaluation")
    if manifest.get("frozen_b41_checkpoint") is not True:
        failures.append(f"{arm}:frozen_b41_checkpoint")
    if manifest.get("b41_training") is not False:
        failures.append(f"{arm}:b41_training")
    if manifest.get("controller_fit_property_counts") != [2]:
        failures.append(f"{arm}:fit_property_counts")
    if manifest.get("composition_ood_property_counts") != [3]:
        failures.append(f"{arm}:ood_property_counts")
    for name in ("validity", "strict_any20", "mean_source_tanimoto", "max_horizon_hit_rate"):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{arm}:{name}")
            continue
        if not math.isfinite(value):
            failures.append(f"{arm}:{name}_nonfinite")
    return failures


def compact_metrics(summary: Mapping[str, object]) -> dict[str, float]:
    metrics = dict(summary["metrics"])
    diagnostic = dict(metrics["by_property_count_diagnostic"])
    three = dict(diagnostic["3"])
    return {
        "validity": float(metrics["validity"]),
        "property_any20": float(metrics["property_any20"]),
        "strict_any20": float(metrics["strict_any20"]),
        "mean_source_tanimoto": float(metrics["mean_source_tanimoto"]),
        "mean_unique_valid": float(metrics["mean_unique_valid"]),
        "max_horizon_hit_rate": float(metrics["max_horizon_hit_rate"]),
        "three_property_validity": float(three["validity"]),
        "three_property_strict_any20": float(three["strict_any20"]),
        "three_property_horizon_hit_rate": float(three["max_horizon_hit_rate"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_json(args.protocol_manifest)
    if preregistration.get("protocol") != PROTOCOL:
        raise ValueError("LLM latent-operator merge protocol drift")
    if tuple(preregistration.get("arms", ())) != ARMS:
        raise ValueError("LLM latent-operator merge arm drift")
    baseline_summary = read_json(args.valid_terminal_summary)
    baseline_metrics = dict(baseline_summary.get("metrics", {}))
    expected_baseline = dict(preregistration["valid_terminal_baseline"])
    drift = {
        key: {"expected": expected, "actual": baseline_metrics.get(key)}
        for key, expected in expected_baseline.items()
        if not math.isclose(
            float(expected),
            float(baseline_metrics.get(key, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if drift:
        raise ValueError(f"LLM latent-operator merge baseline drift: {drift}")

    summaries: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for arm in ARMS:
        path = args.arm_root / arm / "summary.json"
        if not path.is_file():
            failures.append(f"{arm}:missing_summary")
            continue
        summary = read_json(path)
        summaries[arm] = summary
        failures.extend(arm_contract_failures(arm, summary))

    table = {arm: compact_metrics(summary) for arm, summary in summaries.items()}
    gate_checks: dict[str, dict[str, object]] = {}
    selected_arm: str | None = None
    if not failures and set(summaries) == set(ARMS):
        sft_arms = ("sft_frozen", "sft_lora")
        selected_arm = max(
            sft_arms,
            key=lambda arm: (
                table[arm]["validity"],
                table[arm]["strict_any20"],
                -table[arm]["max_horizon_hit_rate"],
            ),
        )
        selected = table[selected_arm]
        comparators = (table["property_mlp"], table["base_frozen"])
        comparator_validity = max(row["validity"] for row in comparators)
        comparator_horizon = min(row["max_horizon_hit_rate"] for row in comparators)
        gates = dict(preregistration["signal_gates"])
        gate_checks = {
            "validity_gain_vs_valid_terminal": {
                "value": selected["validity"] - float(baseline_metrics["validity"]),
                "threshold": gates["validity_gain_vs_valid_terminal"],
            },
            "horizon_reduction_vs_valid_terminal": {
                "value": float(baseline_metrics["max_horizon_hit_rate"])
                - selected["max_horizon_hit_rate"],
                "threshold": gates["horizon_reduction_vs_valid_terminal"],
            },
            "strict_delta_vs_valid_terminal": {
                "value": selected["strict_any20"]
                - float(baseline_metrics["strict_any20"]),
                "threshold": gates["strict_delta_vs_valid_terminal"],
            },
            "mean_source_tanimoto": {
                "value": selected["mean_source_tanimoto"],
                "threshold": gates["mean_source_tanimoto"],
            },
            "three_property_strict_delta": {
                "value": selected["three_property_strict_any20"]
                - float(preregistration["valid_terminal_three_property_strict_any20"]),
                "threshold": gates["three_property_strict_delta"],
            },
            "llm_validity_gain_vs_non_sft_comparators": {
                "value": selected["validity"] - comparator_validity,
                "threshold": gates["llm_gain_vs_non_sft_comparators"],
                "alternative": "horizon",
            },
            "llm_horizon_reduction_vs_non_sft_comparators": {
                "value": comparator_horizon - selected["max_horizon_hit_rate"],
                "threshold": gates["llm_gain_vs_non_sft_comparators"],
                "alternative": "validity",
            },
        }
        for name, check in gate_checks.items():
            if name.startswith("llm_"):
                continue
            if float(check["value"]) < float(check["threshold"]):
                failures.append(name)
        llm_better = (
            float(gate_checks["llm_validity_gain_vs_non_sft_comparators"]["value"])
            >= float(gates["llm_gain_vs_non_sft_comparators"])
            or float(gate_checks["llm_horizon_reduction_vs_non_sft_comparators"]["value"])
            >= float(gates["llm_gain_vs_non_sft_comparators"])
        )
        if not llm_better:
            failures.append("llm_gain_vs_non_sft_comparators")

    passed = not failures and selected_arm is not None
    result = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_common_llm_latent_operator_to_full_train_only_dev"
            if passed
            else "stop_common_llm_latent_operator_signal_without_gate_changes"
        ),
        "selected_arm": selected_arm,
        "valid_terminal_baseline": expected_baseline,
        "arms": table,
        "signal_gate": {
            "passed": passed,
            "checks": gate_checks,
            "failures": failures,
        },
        "contract": {
            "exact_raw_attempts_per_condition": 20,
            "controller_fit_property_counts": [2],
            "composition_ood_property_counts": [3],
            "common_llm_emits_text_or_actions": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "generation_target_access": False,
            "generation_property_oracle_access": False,
            "official_test_access": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
