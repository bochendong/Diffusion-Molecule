#!/usr/bin/env python3
"""Gate target-hidden exact-20 direct repair trajectories against frozen v8 dev."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-manifest", required=True, type=Path)
    parser.add_argument("--plan-manifest", required=True, type=Path)
    parser.add_argument("--controller-data-manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--oracle-summary", required=True, type=Path)
    parser.add_argument("--baseline-gate", required=True, type=Path)
    parser.add_argument("--baseline-format-summary", required=True, type=Path)
    parser.add_argument("--candidate-format-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-sr", type=float, default=0.65)
    parser.add_argument("--min-ood-sr", type=float, default=0.60)
    parser.add_argument("--max-overall-drop", type=float, default=0.03)
    parser.add_argument("--max-ood-drop", type=float, default=0.03)
    parser.add_argument("--min-plan-parse-rate", type=float, default=0.95)
    parser.add_argument("--max-overall-forgetting-drop", type=float, default=0.02)
    parser.add_argument("--max-origin-forgetting-drop", type=float, default=0.05)
    parser.add_argument("--require-transactional", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


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
        "strict_success_rate": float(overall["strict_success_rate"]),
        "official_oracle_coverage": float(overall["official_evaluable_rate"]),
    }


def anti_forgetting_checks(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    overall_allowance: float,
    origin_allowance: float,
) -> tuple[dict[str, object], dict[str, bool]]:
    before_groups = dict(before.get("groups", {}))
    after_groups = dict(after.get("groups", {}))
    if set(before_groups) != set(after_groups):
        raise ValueError("Anti-forgetting group sets differ")
    comparisons = {}
    checks = {}
    for name in sorted(before_groups):
        old = dict(before_groups[name])
        new = dict(after_groups[name])
        if int(old["rows"]) != int(new["rows"]):
            raise ValueError(f"Anti-forgetting row count changed for {name}")
        allowance = overall_allowance if name == "all" else origin_allowance
        record = {"rows": int(new["rows"])}
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
    trajectories = load_json(args.trajectory_manifest)
    plans = load_json(args.plan_manifest)
    controller_data = load_json(args.controller_data_manifest)
    training = load_json(args.training_summary)
    oracle = load_json(args.oracle_summary)
    baseline = load_json(args.baseline_gate)
    metrics = read_rates(args.summary_csv)
    baseline_metrics = {
        key: float(baseline[key])
        for key in ("success_rate", "ind_success_rate", "ood_success_rate", "validity")
    }
    missing = {
        key: int(value)
        for key, value in dict(oracle.get("missing_counts", {})).items()
        if int(value) > 0
    }
    format_comparisons, format_checks = anti_forgetting_checks(
        load_json(args.baseline_format_summary),
        load_json(args.candidate_format_summary),
        overall_allowance=float(args.max_overall_forgetting_drop),
        origin_allowance=float(args.max_origin_forgetting_drop),
    )
    support_checks = {
        "minimum_overall_signal": float(metrics["success_rate"]) >= float(args.min_sr),
        "minimum_ood_signal": float(metrics["ood_success_rate"]) >= float(args.min_ood_sr),
        "within_overall_baseline_tolerance": float(metrics["success_rate"])
        >= baseline_metrics["success_rate"] - float(args.max_overall_drop),
        "within_ood_baseline_tolerance": float(metrics["ood_success_rate"])
        >= baseline_metrics["ood_success_rate"] - float(args.max_ood_drop),
        "validity_complete": float(metrics["validity"]) >= 0.95,
        "official_oracle_complete": float(metrics["official_oracle_coverage"]) == 1.0
        and not missing,
    }
    protocol_checks = {
        "exact_twenty_direct_attempts": int(trajectories.get("attempts_per_condition", 0)) == 20
        and int(trajectories.get("output_rows", 0))
        == 20 * int(trajectories.get("conditions", 0))
        == 20 * int(metrics["conditions"]),
        "no_molecular_candidate_pool": trajectories.get("internal_molecular_candidate_pool")
        is False,
        "no_output_selection": trajectories.get("output_selection") == "none",
        "no_rank_column": trajectories.get("output_rows_have_rank") is False,
        "no_selected_flag": trajectories.get("output_rows_have_selected_flag") is False,
        "target_and_oracle_hidden": trajectories.get("evaluation_target_access") is False
        and trajectories.get("evaluation_oracle_access") is False,
        "official_test_hidden": trajectories.get("official_test_content_access") is False,
        "verifier_feedback_after_each_edit": trajectories.get(
            "train_verifier_observation_after_each_edit"
        )
        is True,
        "controller_prompt_target_hidden": controller_data.get("prompt_target_access") is False,
        "controller_train_validation_disjoint": int(controller_data.get("source_group_overlap", -1))
        == 0,
        "controller_plan_parse_rate": float(plans.get("controller_parse_rate", 0.0))
        >= float(args.min_plan_parse_rate),
        "controller_plans_target_hidden": plans.get("evaluation_target_access") is False
        and plans.get("evaluation_oracle_access") is False,
        "adapter_finite": int(training.get("adapter_nonfinite_parameters", -1)) == 0,
    }
    if args.require_transactional:
        protocol_checks["verifier_feedback_controls_commit_or_rollback"] = (
            trajectories.get("train_verifier_transactional_acceptance") is True
            and isinstance(trajectories.get("transaction_policy"), Mapping)
        )
    checks = {**support_checks, **protocol_checks, **format_checks}
    passed = all(checks.values())
    result = {
        "protocol": (
            "common_llm_transactional_constraint_repair_v13_gate_v1"
            if args.require_transactional
            else "common_llm_direct_constraint_repair_v12_gate_v1"
        ),
        "passed": passed,
        "decision": "advance" if passed else "stop",
        "next_transition": "direct_repair_scale_signal" if passed else "STOP",
        "method": str(
            trajectories.get("method", "common_llm_direct_constraint_repair_v12")
        ),
        "output_selection": "none",
        "candidate_budget": 20,
        "baseline": baseline_metrics,
        "candidate": metrics,
        "gains": {
            key: float(metrics[key]) - baseline_metrics[key]
            for key in ("success_rate", "ind_success_rate", "ood_success_rate", "validity")
        },
        "controller_plan_parse_rate": float(plans.get("controller_parse_rate", 0.0)),
        "trajectory_diagnostics": {
            key: trajectories.get(key)
            for key in (
                "mean_unique_candidates_per_condition",
                "min_unique_candidates_per_condition",
                "repeated_attempt_rows",
                "noop_attempt_rows",
                "trajectory_trace_rate",
                "mean_steps_per_attempt",
                "mean_proposals_per_attempt",
                "committed_edits_total",
                "transaction_rollbacks_total",
            )
        },
        "anti_forgetting": {
            "passed": all(format_checks.values()),
            "comparisons": format_comparisons,
        },
        "oracle_missing_counts": missing,
        "checks": checks,
        "failures": [name for name, value in checks.items() if not value],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {str(trajectories.get('method', 'Common-LLM direct constraint repair'))}",
        "",
        f"Decision: **{'ADVANCE' if passed else 'STOP'}**",
        "",
        "No candidate ranking or output selection is used; each condition launches exactly 20 trajectories.",
        "",
        "| Split | v8 ranked baseline | direct trajectories | Gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (("IND", "ind_success_rate"), ("OOD", "ood_success_rate"), ("All", "success_rate")):
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
