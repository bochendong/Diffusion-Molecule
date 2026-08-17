#!/usr/bin/env python3
"""Merge the property and Common-LLM transport-adapter arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL = "train_only_language_grounded_graph_latent_flow_v1"
ARMS = ("property_memory", "common_llm_memory")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--valid-terminal-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_failures(arm: str, summary: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if summary.get("protocol") != PROTOCOL:
        failures.append(f"{arm}:protocol")
    if summary.get("arm") != arm:
        failures.append(f"{arm}:arm")
    if summary.get("decision") != "await_cross_arm_language_grounded_flow_gate":
        failures.append(f"{arm}:decision")
    metrics = dict(summary.get("metrics", {}))
    manifest = dict(summary.get("manifest", {}))
    if int(metrics.get("attempted_per_condition", -1)) != 20:
        failures.append(f"{arm}:attempts")
    if int(metrics.get("candidate_rows", -1)) != 4700:
        failures.append(f"{arm}:candidate_rows")
    if int(metrics.get("conditions", -1)) != 235:
        failures.append(f"{arm}:conditions")
    required_true = (
        "frozen_b41_checkpoint",
        "state_dependent_transport_adapter",
        "paired_flow_matching_supervision",
        "terminal_reachability_is_auxiliary_training_loss",
        "frozen_before_target_or_property_evaluation",
    )
    required_false = (
        "b41_training",
        "inference_classifier_gradient_guidance",
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
    for name in required_true:
        if manifest.get(name) is not True:
            failures.append(f"{arm}:{name}")
    for name in required_false:
        if manifest.get(name) is not False:
            failures.append(f"{arm}:{name}")
    if manifest.get("fit_property_counts") != [2]:
        failures.append(f"{arm}:fit_property_counts")
    for name in (
        "validity",
        "strict_any20",
        "mean_source_tanimoto",
        "max_horizon_hit_rate",
    ):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{arm}:{name}")
            continue
        if not math.isfinite(value):
            failures.append(f"{arm}:{name}_nonfinite")
    validation = dict(summary.get("fit_validation", {}))
    for name in ("flow_loss", "base_flow_loss", "relative_flow_mse_reduction"):
        try:
            value = float(validation[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{arm}:fit_validation:{name}")
            continue
        if not math.isfinite(value):
            failures.append(f"{arm}:fit_validation:{name}_nonfinite")
    return failures


def compact(summary: Mapping[str, object]) -> dict[str, float]:
    metrics = dict(summary["metrics"])
    validation = dict(summary["fit_validation"])
    three = dict(dict(metrics["by_property_count_diagnostic"])["3"])
    return {
        "fit_flow_loss": float(validation["flow_loss"]),
        "fit_base_flow_loss": float(validation["base_flow_loss"]),
        "relative_flow_mse_reduction": float(
            validation["relative_flow_mse_reduction"]
        ),
        "terminal_state_auc_diagnostic": float(
            validation["terminal_state_auc_diagnostic"]
        ),
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
        raise ValueError("Language-grounded merge protocol drift")
    if tuple(preregistration.get("arms", ())) != ARMS:
        raise ValueError("Language-grounded merge arm drift")
    baseline_summary = read_json(args.valid_terminal_summary)
    baseline = dict(baseline_summary.get("metrics", {}))
    expected_baseline = dict(preregistration["valid_terminal_baseline"])
    failures: list[str] = []
    for name, expected in expected_baseline.items():
        actual = float(baseline.get(name, math.nan))
        if not math.isclose(float(expected), actual, rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"baseline:{name}")
    summaries: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        path = args.arm_root / arm / "summary.json"
        if not path.is_file():
            failures.append(f"{arm}:missing_summary")
            continue
        summaries[arm] = read_json(path)
        failures.extend(contract_failures(arm, summaries[arm]))
    table = {arm: compact(summary) for arm, summary in summaries.items()}
    checks: dict[str, dict[str, object]] = {}
    if set(summaries) == set(ARMS) and not failures:
        llm = table["common_llm_memory"]
        prop = table["property_memory"]
        gates = dict(preregistration["signal_gates"])
        checks = {
            "relative_flow_mse_reduction": {
                "value": llm["relative_flow_mse_reduction"],
                "threshold": gates["relative_flow_mse_reduction"],
            },
            "validity_delta_vs_valid_terminal": {
                "value": llm["validity"] - float(baseline["validity"]),
                "threshold": gates["validity_delta_vs_valid_terminal"],
            },
            "horizon_reduction_vs_valid_terminal": {
                "value": float(baseline["max_horizon_hit_rate"])
                - llm["max_horizon_hit_rate"],
                "threshold": gates["horizon_reduction_vs_valid_terminal"],
            },
            "strict_gain_vs_valid_terminal": {
                "value": llm["strict_any20"] - float(baseline["strict_any20"]),
                "threshold": gates["strict_gain_vs_valid_terminal"],
            },
            "mean_source_tanimoto": {
                "value": llm["mean_source_tanimoto"],
                "threshold": gates["mean_source_tanimoto"],
            },
            "three_property_strict_delta": {
                "value": llm["three_property_strict_any20"]
                - float(preregistration["valid_terminal_three_property_strict_any20"]),
                "threshold": gates["three_property_strict_delta"],
            },
            "llm_strict_gain_vs_property_memory": {
                "value": llm["strict_any20"] - prop["strict_any20"],
                "threshold": gates["llm_strict_gain_vs_property_memory"],
            },
        }
        for name, check in checks.items():
            if float(check["value"]) < float(check["threshold"]):
                failures.append(name)
    passed = not failures and set(summaries) == set(ARMS)
    result = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_language_grounded_graph_latent_flow_to_fresh_confirmation"
            if passed
            else "stop_language_grounded_graph_latent_flow_without_gate_changes"
        ),
        "selected_arm": "common_llm_memory" if passed else None,
        "valid_terminal_baseline": expected_baseline,
        "arms": table,
        "signal_gate": {"passed": passed, "checks": checks, "failures": failures},
        "contract": {
            "exact_raw_attempts_per_condition": 20,
            "fit_property_counts": [2],
            "composition_diagnostic_property_counts": [3],
            "state_dependent_transport_adapter": True,
            "paired_flow_matching_supervision": True,
            "inference_classifier_gradient_guidance": False,
            "common_llm_emits_text_or_actions": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "generation_target_access": False,
            "generation_property_oracle_access": False,
            "official_test_access": False,
            "development_is_reused_method_development_split": True,
            "development_is_formal_fresh_ood": False
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
