#!/usr/bin/env python3
"""Merge the two preregistered state-viability critic arms."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL = "train_only_common_llm_state_viability_guidance_v1"
ARMS = ("property_memory", "common_llm_memory")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--prepare-summary", type=Path, required=True)
    parser.add_argument("--valid-terminal-summary", type=Path, required=True)
    parser.add_argument("--valid-terminal-candidates", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def baseline_three_property_validity(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["property_count"]) == 3]
    if not rows:
        raise ValueError("Valid-terminal baseline lacks three-property candidate rows")
    return sum(as_bool(row["valid"]) for row in rows) / len(rows)


def contract_failures(arm: str, summary: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if summary.get("protocol") != PROTOCOL:
        failures.append(f"{arm}:protocol")
    if summary.get("arm") != arm:
        failures.append(f"{arm}:arm")
    if summary.get("decision") != "await_cross_arm_state_viability_gate":
        failures.append(f"{arm}:decision")
    metrics = dict(summary.get("metrics", {}))
    manifest = dict(summary.get("manifest", {}))
    if int(metrics.get("attempted_per_condition", -1)) != 20:
        failures.append(f"{arm}:attempts")
    if int(metrics.get("candidate_rows", -1)) != 4700:
        failures.append(f"{arm}:candidate_rows")
    if int(metrics.get("conditions", -1)) != 235:
        failures.append(f"{arm}:conditions")
    if dict(summary.get("critic_gate", {})).get("passed") is not True:
        failures.extend(
            f"{arm}:critic:{name}"
            for name in dict(summary.get("critic_gate", {})).get("failures", ["missing"])
        )
    required_true = (
        "frozen_b41_checkpoint",
        "current_latent_queries_constraint_memory_each_flow_step",
        "terminal_reachability_gradient_guides_latent_vector_field",
        "frozen_before_target_or_property_evaluation",
    )
    required_false = (
        "b41_training",
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
    for name in ("validity", "strict_any20", "mean_source_tanimoto", "max_horizon_hit_rate"):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{arm}:{name}")
            continue
        if not math.isfinite(value):
            failures.append(f"{arm}:{name}_nonfinite")
    return failures


def compact(summary: Mapping[str, object]) -> dict[str, float]:
    metrics = dict(summary["metrics"])
    three = dict(dict(metrics["by_property_count_diagnostic"])["3"])
    critic = dict(summary["critic_metrics"])
    return {
        "critic_state_auc": float(critic["state_auc"]),
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
    prereg = read_json(args.protocol_manifest)
    if prereg.get("protocol") != PROTOCOL or tuple(prereg.get("arms", ())) != ARMS:
        raise ValueError("State-viability merge protocol drift")
    prepare = read_json(args.prepare_summary)
    failures: list[str] = []
    if prepare.get("decision") != "trajectory_dataset_frozen_for_parallel_critics":
        failures.append("prepare:decision")
    baseline_summary = read_json(args.valid_terminal_summary)
    baseline = dict(baseline_summary.get("metrics", {}))
    expected_baseline = dict(prereg["valid_terminal_baseline"])
    for name, expected in expected_baseline.items():
        actual = float(baseline.get(name, math.nan))
        if not math.isclose(float(expected), actual, rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"baseline:{name}")
    baseline_three = baseline_three_property_validity(args.valid_terminal_candidates)
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
        gates = dict(prereg["signal_gates"])
        checks = {
            "validity_gain_vs_valid_terminal": {
                "value": llm["validity"] - float(baseline["validity"]),
                "threshold": gates["validity_gain_vs_valid_terminal"],
            },
            "horizon_reduction_vs_valid_terminal": {
                "value": float(baseline["max_horizon_hit_rate"])
                - llm["max_horizon_hit_rate"],
                "threshold": gates["horizon_reduction_vs_valid_terminal"],
            },
            "strict_delta_vs_valid_terminal": {
                "value": llm["strict_any20"] - float(baseline["strict_any20"]),
                "threshold": gates["strict_delta_vs_valid_terminal"],
            },
            "mean_source_tanimoto": {
                "value": llm["mean_source_tanimoto"],
                "threshold": gates["mean_source_tanimoto"],
            },
            "three_property_validity_gain": {
                "value": llm["three_property_validity"] - baseline_three,
                "threshold": gates["three_property_validity_gain"],
            },
            "llm_validity_gain_vs_property_memory": {
                "value": llm["validity"] - prop["validity"],
                "threshold": gates["llm_gain_vs_property_memory"],
                "alternative": "horizon",
            },
            "llm_horizon_reduction_vs_property_memory": {
                "value": prop["max_horizon_hit_rate"] - llm["max_horizon_hit_rate"],
                "threshold": gates["llm_gain_vs_property_memory"],
                "alternative": "validity",
            },
        }
        for name, check in checks.items():
            if name.startswith("llm_"):
                continue
            if float(check["value"]) < float(check["threshold"]):
                failures.append(name)
        if not (
            float(checks["llm_validity_gain_vs_property_memory"]["value"])
            >= float(gates["llm_gain_vs_property_memory"])
            or float(checks["llm_horizon_reduction_vs_property_memory"]["value"])
            >= float(gates["llm_gain_vs_property_memory"])
        ):
            failures.append("llm_gain_vs_property_memory")
    passed = not failures and set(summaries) == set(ARMS)
    result = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_state_dependent_common_llm_latent_guidance"
            if passed
            else "stop_state_viability_guidance_without_gate_changes"
        ),
        "selected_arm": "common_llm_memory" if passed else None,
        "valid_terminal_baseline": {**expected_baseline, "three_property_validity": baseline_three},
        "arms": table,
        "signal_gate": {"passed": passed, "checks": checks, "failures": failures},
        "contract": {
            "exact_raw_attempts_per_condition": 20,
            "fit_property_counts": [2],
            "composition_diagnostic_property_counts": [3],
            "common_llm_emits_text_or_actions": False,
            "current_latent_queries_constraint_memory_each_flow_step": True,
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
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
