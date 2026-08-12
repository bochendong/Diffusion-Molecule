#!/usr/bin/env python3
"""Gate the bounded common-LLM residual planner against frozen MuMO dev."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--oracle-summary", required=True, type=Path)
    parser.add_argument("--baseline-gate", required=True, type=Path)
    parser.add_argument("--baseline-format-summary", required=True, type=Path)
    parser.add_argument("--candidate-format-summary", required=True, type=Path)
    parser.add_argument("--baseline-preference-summary", required=True, type=Path)
    parser.add_argument("--candidate-preference-summary", required=True, type=Path)
    parser.add_argument("--preference-manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-sr", type=float, default=0.65)
    parser.add_argument("--min-ood-sr", type=float, default=0.60)
    parser.add_argument("--min-preference-accuracy", type=float, default=0.55)
    parser.add_argument("--max-overall-forgetting-drop", type=float, default=0.02)
    parser.add_argument("--max-origin-forgetting-drop", type=float, default=0.05)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing mapping: {label}")
    return value


def read_rates(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    overall = next(row for row in rows if row["external_suite"] == "all")
    counts = {split: 0 for split in ("ind", "ood")}
    successes = {split: 0.0 for split in ("ind", "ood")}
    for row in rows:
        if row["external_suite"] != "mumo":
            continue
        split = row["external_task_split"]
        count = int(row["input_groups"])
        counts[split] += count
        successes[split] += count * float(row["success_rate"])
    return {
        "conditions": int(overall["input_groups"]),
        "success_rate": float(overall["success_rate"]),
        "ind_success_rate": successes["ind"] / max(counts["ind"], 1),
        "ood_success_rate": successes["ood"] / max(counts["ood"], 1),
        "validity": float(overall["validity"]),
        "official_oracle_coverage": float(overall["official_evaluable_rate"]),
    }


def anti_forgetting_comparison(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    overall_allowance: float,
    origin_allowance: float,
) -> tuple[dict[str, object], dict[str, bool]]:
    before_groups = as_mapping(before.get("groups"), "baseline.groups")
    after_groups = as_mapping(after.get("groups"), "candidate.groups")
    if set(before_groups) != set(after_groups):
        raise ValueError("Anti-forgetting group sets differ")
    comparisons = {}
    checks = {}
    for name in sorted(before_groups):
        old = as_mapping(before_groups[name], f"baseline.groups.{name}")
        new = as_mapping(after_groups[name], f"candidate.groups.{name}")
        if int(old["rows"]) != int(new["rows"]):
            raise ValueError(f"Anti-forgetting row count changed for {name}")
        allowance = overall_allowance if name == "all" else origin_allowance
        record: dict[str, object] = {"rows": int(new["rows"])}
        for metric in ("json_parse_rate", "action_type_rate"):
            baseline = float(old[metric])
            candidate = float(new[metric])
            delta = candidate - baseline
            record[f"baseline_{metric}"] = baseline
            record[f"candidate_{metric}"] = candidate
            record[f"{metric}_delta"] = delta
            checks[f"{name}_{metric}_preserved"] = delta >= -allowance
        comparisons[name] = record
    return comparisons, checks


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_json(args.candidate_manifest)
    oracle = load_json(args.oracle_summary)
    baseline = load_json(args.baseline_gate)
    preference = load_json(args.preference_manifest)
    training = load_json(args.training_summary)
    baseline_pref = load_json(args.baseline_preference_summary)
    candidate_pref = load_json(args.candidate_preference_summary)
    metrics = read_rates(args.summary_csv)
    format_comparisons, format_checks = anti_forgetting_comparison(
        load_json(args.baseline_format_summary),
        load_json(args.candidate_format_summary),
        overall_allowance=float(args.max_overall_forgetting_drop),
        origin_allowance=float(args.max_origin_forgetting_drop),
    )

    baseline_metrics = {
        key: float(baseline[key])
        for key in ("success_rate", "ind_success_rate", "ood_success_rate", "validity")
    }
    missing = {
        key: int(value)
        for key, value in dict(oracle.get("missing_counts", {})).items()
        if int(value) > 0
    }
    support_checks = {
        "minimum_overall_signal": float(metrics["success_rate"]) >= float(args.min_sr),
        "minimum_ood_signal": float(metrics["ood_success_rate"]) >= float(args.min_ood_sr),
        "overall_non_decrease": float(metrics["success_rate"]) + 1e-12 >= baseline_metrics["success_rate"],
        "ind_non_decrease": float(metrics["ind_success_rate"]) + 1e-12 >= baseline_metrics["ind_success_rate"],
        "ood_non_decrease": float(metrics["ood_success_rate"]) + 1e-12 >= baseline_metrics["ood_success_rate"],
        "validity_complete": float(metrics["validity"]) >= 0.95,
        "official_oracle_complete": float(metrics["official_oracle_coverage"]) == 1.0 and not missing,
    }
    preference_checks = {
        "residual_preference_signal": float(candidate_pref["ranking_accuracy"])
        >= float(args.min_preference_accuracy),
        "residual_preference_non_decrease": float(candidate_pref["ranking_accuracy"]) + 1e-12
        >= float(baseline_pref["ranking_accuracy"]),
        "residual_preference_finite": int(candidate_pref["nonfinite_scores"]) == 0,
    }
    protocol_checks = {
        "baseline_gate_passed": baseline.get("passed") is True,
        "fixed_candidate_budget": int(manifest.get("candidate_budget", 0)) == 20,
        "exact_candidate_rows": int(manifest.get("output_rows", 0))
        == 20 * int(manifest.get("conditions", 0))
        == 20 * int(metrics["conditions"]),
        "evaluation_target_hidden": manifest.get("evaluation_target_access") is False,
        "evaluation_oracle_hidden": manifest.get("evaluation_oracle_access") is False,
        "official_test_hidden": manifest.get("official_test_content_access") is False,
        "preference_prompt_target_hidden": preference.get("prompt_target_access") is False,
        "preference_source_disjoint": int(preference.get("source_group_overlap", -1)) == 0,
        "adapter_finite": int(training.get("adapter_nonfinite_parameters", -1)) == 0,
        "bounded_residual_policy": manifest.get("selection_policy")
        == "deterministic_prefix_plus_bounded_common_llm_residual",
    }
    checks = {**support_checks, **preference_checks, **format_checks, **protocol_checks}
    passed = all(checks.values())
    result = {
        "protocol": "common_llm_mumo_residual_planner_v9_gate_v1",
        "passed": passed,
        "decision": "advance" if passed else "stop",
        "next_transition": "common_llm_7b_residual" if passed else "STOP",
        "candidate_budget": 20,
        "baseline": baseline_metrics,
        "candidate": metrics,
        "gains": {
            key: float(metrics[key]) - baseline_metrics[key]
            for key in ("success_rate", "ind_success_rate", "ood_success_rate", "validity")
        },
        "residual_preference": {
            "baseline_ranking_accuracy": float(baseline_pref["ranking_accuracy"]),
            "candidate_ranking_accuracy": float(candidate_pref["ranking_accuracy"]),
            "candidate_mean_margin": float(candidate_pref["mean_log_probability_margin"]),
        },
        "anti_forgetting": {
            "passed": all(format_checks.values()),
            "comparisons": format_comparisons,
            "max_overall_drop": float(args.max_overall_forgetting_drop),
            "max_origin_drop": float(args.max_origin_forgetting_drop),
        },
        "oracle_missing_counts": missing,
        "changed_conditions": int(manifest.get("changed_conditions", 0)),
        "evaluation_target_access": False,
        "checks": checks,
        "failures": [name for name, value in checks.items() if not value],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Common-LLM MuMO residual planner v9 gate",
        "",
        f"Decision: **{'ADVANCE' if passed else 'STOP'}**",
        "",
        "| Split | Deterministic | 1.5B residual | Gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("IND", "ind_success_rate"),
        ("OOD", "ood_success_rate"),
        ("All", "success_rate"),
    ):
        lines.append(
            f"| {label} | {baseline_metrics[key]:.1%} | {float(metrics[key]):.1%} | "
            f"{float(metrics[key]) - baseline_metrics[key]:+.1%} |"
        )
    lines.extend(["", "Checks:"])
    lines.extend(f"- {name}: `{'pass' if value else 'fail'}`" for name, value in checks.items())
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
