#!/usr/bin/env python3
"""Evaluate the bounded P1 6p/7p n=20 low-budget kill test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_p1_sampling_scaling import (
    build_condition_table,
    build_paired_deltas,
    condition_key,
    load_candidate_groups,
    read_rows,
    summarize_condition_table,
    write_csv,
)


BUDGETS = (1, 4, 8, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--sft-candidates", required=True, type=Path)
    parser.add_argument("--group-rl-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_rows = read_rows(args.eval_csv)
    eval_keys = {condition_key(row) for row in eval_rows}
    groups = {
        "sft": load_candidate_groups(args.sft_candidates, required_count=20),
        "group_rl": load_candidate_groups(args.group_rl_candidates, required_count=20),
    }
    for model, values in groups.items():
        if set(values) != eval_keys:
            raise RuntimeError(f"{model} condition mismatch: expected {len(eval_keys)}, found {len(values)}")
    condition_rows = []
    for model in ("sft", "group_rl"):
        condition_rows.extend(build_condition_table("two_p_to_seven_p", model, eval_rows, groups[model], BUDGETS))
    lookup = {("two_p_to_seven_p", condition_key(row)): row for row in eval_rows}
    summary = summarize_condition_table(condition_rows, lookup)
    deltas = build_paired_deltas(condition_rows, lookup, resamples=args.bootstrap_resamples, seed=args.seed)
    delta_index = {
        (str(row["group_type"]), str(row["group"]), int(row["candidate_budget"])): row for row in deltas
    }
    point_checks = {}
    for count in ("6", "7"):
        for budget in (8, 20):
            row = delta_index[("property_count", count, budget)]
            for metric in ("raw_success_fraction", "empirical_prefix_pass_at_k"):
                point_checks[f"{count}p_k{budget}_{metric}_positive"] = float(row[f"delta_{metric}"]) > 0.0
    confidence_checks = {}
    for budget in (8, 20):
        row = delta_index[("overall", "all", budget)]
        confidence_checks[f"overall_k{budget}_raw_ci_positive"] = float(
            row["delta_raw_success_fraction_ci95_low"]
        ) > 0.0
    row = delta_index[("overall", "all", 8)]
    confidence_checks["overall_k8_pass_ci_positive"] = float(
        row["delta_empirical_prefix_pass_at_k_ci95_low"]
    ) > 0.0
    if all(point_checks.values()) and all(confidence_checks.values()):
        verdict = "fast_strong_signal"
    elif all(point_checks.values()):
        verdict = "fast_promising_signal"
    else:
        verdict = "fast_mixed_or_negative"
    gate = {
        "protocol": "p1_fast_hard_6p7p_k20_kill_test_v1",
        "claim_scope": "interim_kill_test_not_final_preregistered_p1_result",
        "conditions": len(eval_rows),
        "candidate_budget": 20,
        "verdict": verdict,
        "point_checks": point_checks,
        "confidence_checks": confidence_checks,
    }
    lines = [
        "# P1 fast 6p/7p low-budget kill test",
        "",
        f"Verdict: **{verdict}**. This is an interim n=20 kill test, not the final preregistered n=256 result.",
        "",
        "| stratum | k | delta raw | 95% CI | delta pass@k | 95% CI |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for budget in (8, 20):
        for group_type, group in (("overall", "all"), ("property_count", "6"), ("property_count", "7")):
            row = delta_index[(group_type, group, budget)]
            label = "overall" if group_type == "overall" else f"{group}p"
            lines.append(
                f"| {label} | {budget} | {float(row['delta_raw_success_fraction']):+.4f} | "
                f"[{float(row['delta_raw_success_fraction_ci95_low']):+.4f}, {float(row['delta_raw_success_fraction_ci95_high']):+.4f}] | "
                f"{float(row['delta_empirical_prefix_pass_at_k']):+.4f} | "
                f"[{float(row['delta_empirical_prefix_pass_at_k_ci95_low']):+.4f}, {float(row['delta_empirical_prefix_pass_at_k_ci95_high']):+.4f}] |"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_metrics.csv", condition_rows)
    write_csv(args.output_dir / "scaling_summary.csv", summary)
    write_csv(args.output_dir / "paired_deltas.csv", deltas)
    (args.output_dir / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
