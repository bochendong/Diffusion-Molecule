#!/usr/bin/env python3
"""Combine v6 support and anti-forgetting checks into one terminal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-summary", required=True, type=Path)
    parser.add_argument("--baseline-support-summary", required=True, type=Path)
    parser.add_argument("--baseline-format-summary", required=True, type=Path)
    parser.add_argument("--candidate-format-summary", required=True, type=Path)
    parser.add_argument("--preference-manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-support-rate", type=float, default=0.46)
    parser.add_argument("--min-support-gain", type=float, default=0.06)
    parser.add_argument("--max-overall-forgetting-drop", type=float, default=0.02)
    parser.add_argument("--max-origin-forgetting-drop", type=float, default=0.05)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing mapping: {label}")
    return value


def rate(group: Mapping[str, object], key: str) -> float:
    return float(group[key])


def support_scope(summary: Mapping[str, object], split: str) -> Mapping[str, object]:
    support = mapping(summary.get("support"), label="support")
    if split == "all":
        return mapping(support.get("all"), label="support.all")
    return mapping(
        mapping(support.get("by_split"), label="support.by_split").get(split),
        label=f"support.by_split.{split}",
    )


def format_groups(summary: Mapping[str, object]) -> Mapping[str, object]:
    return mapping(summary.get("groups"), label="format.groups")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = read_json(args.support_summary)
    baseline = read_json(args.baseline_support_summary)
    baseline_format = read_json(args.baseline_format_summary)
    candidate_format = read_json(args.candidate_format_summary)
    preference = read_json(args.preference_manifest)
    training = read_json(args.training_summary)

    candidate_all = support_scope(candidate, "all")
    baseline_all = support_scope(baseline, "all")
    comparisons = {}
    for split in ("all", "ind", "ood"):
        candidate_metrics = support_scope(candidate, split)
        baseline_metrics = support_scope(baseline, split)
        comparisons[split] = {
            "conditions": int(candidate_metrics["conditions"]),
            "baseline_property_any_rate": rate(baseline_metrics, "property_any_rate"),
            "candidate_property_any_rate": rate(candidate_metrics, "property_any_rate"),
            "property_gain": rate(candidate_metrics, "property_any_rate")
            - rate(baseline_metrics, "property_any_rate"),
            "baseline_strict_any_rate": rate(baseline_metrics, "strict_any_rate"),
            "candidate_strict_any_rate": rate(candidate_metrics, "strict_any_rate"),
            "strict_gain": rate(candidate_metrics, "strict_any_rate")
            - rate(baseline_metrics, "strict_any_rate"),
        }

    support_checks = {
        "property_at_least_signal_threshold": rate(candidate_all, "property_any_rate")
        >= float(args.min_support_rate),
        "strict_at_least_signal_threshold": rate(candidate_all, "strict_any_rate")
        >= float(args.min_support_rate),
        "property_gain": comparisons["all"]["property_gain"] + 1e-12 >= float(args.min_support_gain),
        "strict_gain": comparisons["all"]["strict_gain"] + 1e-12 >= float(args.min_support_gain),
        "ind_strict_non_decrease": comparisons["ind"]["strict_gain"] >= 0.0,
        "ood_strict_non_decrease": comparisons["ood"]["strict_gain"] >= 0.0,
        "validity_complete": rate(candidate_all, "valid_any_rate") == 1.0,
        "full_oracle_complete": rate(candidate_all, "full_oracle_condition_rate") == 1.0,
    }

    baseline_groups = format_groups(baseline_format)
    candidate_groups = format_groups(candidate_format)
    if set(baseline_groups) != set(candidate_groups):
        raise ValueError("Baseline and v6 format evaluation groups differ")
    format_comparisons = {}
    format_checks = {}
    for name in sorted(baseline_groups):
        before = mapping(baseline_groups[name], label=f"baseline.groups.{name}")
        after = mapping(candidate_groups[name], label=f"candidate.groups.{name}")
        if int(before["rows"]) != int(after["rows"]):
            raise ValueError(f"Format evaluation row count changed for {name}")
        record = {"rows": int(after["rows"])}
        for metric in ("json_parse_rate", "action_type_rate"):
            record[f"baseline_{metric}"] = rate(before, metric)
            record[f"candidate_{metric}"] = rate(after, metric)
            record[f"{metric}_delta"] = rate(after, metric) - rate(before, metric)
        format_comparisons[name] = record
        allowance = (
            float(args.max_overall_forgetting_drop)
            if name == "all"
            else float(args.max_origin_forgetting_drop)
        )
        format_checks[f"{name}_json_parse_preserved"] = record["json_parse_rate_delta"] >= -allowance
        format_checks[f"{name}_action_type_preserved"] = record["action_type_rate_delta"] >= -allowance

    candidate_builder = mapping(candidate.get("candidate_builder"), label="candidate_builder")
    protocol_checks = {
        "fixed_candidate_budget": int(candidate.get("final_oracle_candidate_budget", 0)) == 20,
        "evaluation_target_hidden": candidate_builder.get("evaluation_target_access") is False,
        "prompt_target_hidden": preference.get("prompt_target_access") is False,
        "preference_source_disjoint": int(preference.get("source_group_overlap", -1)) == 0,
        "split_source_disjoint": int(mapping(candidate.get("split_audit"), label="split_audit").get("source_overlap", -1)) == 0,
        "adapter_finite": int(training.get("adapter_nonfinite_parameters", -1)) == 0,
    }
    checks = {**support_checks, **format_checks, **protocol_checks}
    decision = "advance" if all(checks.values()) else "stop"
    result = {
        "protocol": "hierarchical_common_agent_retrieved_delta_planner_v6",
        "data_role": "train_only_heldout",
        "decision": decision,
        "proposal_budget": 0,
        "final_oracle_candidate_budget": 20,
        "split_audit": candidate["split_audit"],
        "candidate_builder": candidate_builder,
        "support": candidate["support"],
        "baseline_support": baseline["support"],
        "planner_comparison": comparisons,
        "anti_forgetting": {
            "passed": all(format_checks.values()),
            "comparisons": format_comparisons,
            "max_overall_drop": float(args.max_overall_forgetting_drop),
            "max_origin_drop": float(args.max_origin_forgetting_drop),
        },
        "training": {
            "preference_protocol": preference.get("protocol"),
            "prompt_target_access": preference.get("prompt_target_access"),
            "training_target_role": preference.get("training_target_role"),
            "source_group_overlap": preference.get("source_group_overlap"),
            "preference_train_pairs": preference.get("train_pairs"),
            "adapter_nonfinite_parameters": training.get("adapter_nonfinite_parameters"),
        },
        "thresholds": {
            "min_support_rate": float(args.min_support_rate),
            "min_support_gain": float(args.min_support_gain),
        },
        "checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Common-LLM RetrievedDelta planner v6 gate",
        "",
        f"Decision: **{decision.upper()}**",
        "",
        "| Split | Baseline strict@20 | V6 strict@20 | Gain | V6 property@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("ind", "ood", "all"):
        item = comparisons[split]
        lines.append(
            f"| {split} | {item['baseline_strict_any_rate']:.1%} | "
            f"{item['candidate_strict_any_rate']:.1%} | {item['strict_gain']:+.1%} | "
            f"{item['candidate_property_any_rate']:.1%} |"
        )
    lines.extend(["", "Checks:"])
    for name, passed in checks.items():
        lines.append(f"- {name}: `{'pass' if passed else 'fail'}`")
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
