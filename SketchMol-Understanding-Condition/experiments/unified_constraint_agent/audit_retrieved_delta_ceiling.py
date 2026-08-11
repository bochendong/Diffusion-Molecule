#!/usr/bin/env python3
"""Audit property and strict support in an oracle-blind diagnostic prefix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-detail-csv", required=True, type=Path)
    parser.add_argument("--candidate-manifest-json", required=True, type=Path)
    parser.add_argument("--baseline-support-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-property-ceiling", type=float, default=0.70)
    parser.add_argument("--target-split-property-ceiling", type=float, default=0.70)
    parser.add_argument("--target-strict-ceiling", type=float, default=0.70)
    parser.add_argument("--target-split-strict-ceiling", type=float, default=None)
    parser.add_argument("--expected-conditions", type=int, default=50)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def condition_id(row: Mapping[str, object]) -> str:
    value = str(row.get("condition_id", "") or "").strip()
    if not value:
        raise ValueError("Official detail row is missing condition_id")
    return value


def split_value(row: Mapping[str, object]) -> str:
    return str(row.get("external_task_split", "") or row.get("split", "") or "all").strip().lower()


def task_value(row: Mapping[str, object]) -> str:
    return str(
        row.get("external_task_key", "")
        or row.get("external_task_id", "")
        or row.get("condition_properties", "")
        or "unknown"
    ).strip()


def summarize(groups: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, object]:
    denominator = max(len(groups), 1)
    candidate_rows = sum(len(group) for group in groups)
    counts = [len(group) for group in groups]
    return {
        "conditions": len(groups),
        "candidate_rows": candidate_rows,
        "min_candidates": min(counts) if counts else 0,
        "mean_candidates": sum(counts) / denominator,
        "max_candidates": max(counts) if counts else 0,
        "property_any_rate": sum(
            any(truthy(row.get("external_official_success")) for row in group) for group in groups
        )
        / denominator,
        "strict_any_rate": sum(
            any(truthy(row.get("external_strict_success")) for row in group) for group in groups
        )
        / denominator,
        "valid_candidate_rate": sum(
            truthy(row.get("external_valid")) for group in groups for row in group
        )
        / max(candidate_rows, 1),
        "full_oracle_candidate_rate": sum(
            truthy(row.get("external_full_property_coverage")) for group in groups for row in group
        )
        / max(candidate_rows, 1),
        "full_oracle_condition_rate": sum(
            all(truthy(row.get("external_full_property_coverage")) for row in group)
            for group in groups
        )
        / denominator,
    }


def baseline_rate(payload: Mapping[str, object], name: str) -> float:
    support = payload.get("support", {})
    if not isinstance(support, Mapping):
        return 0.0
    all_metrics = support.get("all", {})
    if not isinstance(all_metrics, Mapping):
        return 0.0
    return float(all_metrics.get(name, 0.0) or 0.0)


def render_report(summary: Mapping[str, object]) -> str:
    all_metrics = summary["support_ceiling"]["all"]
    comparison = summary["comparison_to_v5"]
    lines = [
        "# RetrievedDelta Support-Ceiling Audit",
        "",
        f"Decision: **{str(summary['decision']).upper()}**",
        "",
        "This is an oracle-blind, post-selection support diagnostic over at most 96 pre-ranked candidates. "
        "It is not a paper-facing n=96 result; the benchmark contract remains n=20.",
        "",
        "| Scope | Conditions | Mean k | Property any@k | Strict any@k | Full oracle |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| all | {all_metrics['conditions']} | {all_metrics['mean_candidates']:.1f} | "
            f"{all_metrics['property_any_rate']:.1%} | {all_metrics['strict_any_rate']:.1%} | "
            f"{all_metrics['full_oracle_condition_rate']:.1%} |"
        ),
    ]
    for name, metrics in summary["support_ceiling"]["by_split"].items():
        lines.append(
            f"| {name} | {metrics['conditions']} | {metrics['mean_candidates']:.1f} | "
            f"{metrics['property_any_rate']:.1%} | {metrics['strict_any_rate']:.1%} | "
            f"{metrics['full_oracle_condition_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"V5 strict any@20 was {comparison['baseline_strict_any_rate']:.1%}; the diagnostic ceiling gain is "
            f"{comparison['strict_ceiling_gain']:+.1%}.",
            "",
            "The declared all/split property and strict ceilings must all pass before a verifier-observed n=20 "
            "planner is trained. This diagnostic does not itself count as the final result.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.candidate_manifest_json.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_support_summary.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("evaluation_target_access") is not False:
        raise ValueError("Candidate manifest must declare evaluation_target_access=false")
    if manifest.get("diagnostic_only") is not True:
        raise ValueError("Candidate manifest must declare diagnostic_only=true")
    if int(manifest.get("paper_facing_candidate_budget", 0) or 0) != 20:
        raise ValueError("Candidate manifest must preserve the paper-facing n=20 contract")

    rows = read_rows(args.official_detail_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[condition_id(row)].append(row)
    if len(grouped) != int(args.expected_conditions):
        raise ValueError(f"Detail has {len(grouped)} conditions; expected {args.expected_conditions}")
    groups = list(grouped.values())
    by_split: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    by_task: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in groups:
        by_split[split_value(group[0])].append(group)
        by_task[task_value(group[0])].append(group)

    support = {
        "all": summarize(groups),
        "by_split": {name: summarize(value) for name, value in sorted(by_split.items())},
        "by_task": {name: summarize(value) for name, value in sorted(by_task.items())},
    }
    all_metrics = support["all"]
    min_split_strict = min(
        (float(value["strict_any_rate"]) for value in support["by_split"].values()),
        default=0.0,
    )
    min_split_property = min(
        (float(value["property_any_rate"]) for value in support["by_split"].values()),
        default=0.0,
    )
    complete = (
        float(all_metrics["full_oracle_candidate_rate"]) == 1.0
        and float(all_metrics["full_oracle_condition_rate"]) == 1.0
    )
    property_target = float(args.target_property_ceiling)
    split_property_target = float(args.target_split_property_ceiling)
    strict_target = float(args.target_strict_ceiling)
    split_strict_target = (
        strict_target
        if args.target_split_strict_ceiling is None
        else float(args.target_split_strict_ceiling)
    )
    if not complete:
        decision = "needs_attention"
    elif (
        float(all_metrics["property_any_rate"]) >= property_target
        and min_split_property >= split_property_target
        and float(all_metrics["strict_any_rate"]) >= strict_target
        and min_split_strict >= split_strict_target
    ):
        decision = "support_sufficient_for_n20_planner"
    else:
        decision = "generator_expansion_required"

    baseline_property = baseline_rate(baseline, "property_any_rate")
    baseline_strict = baseline_rate(baseline, "strict_any_rate")
    summary = {
        "protocol": "retrieved_delta_support_ceiling_audit_v1",
        "data_role": "train_only_heldout_diagnostic",
        "decision": decision,
        "evaluation_target_access": False,
        "oracle_used_for_selection": False,
        "paper_facing_candidate_budget": 20,
        "diagnostic_candidate_limit": int(manifest["diagnostic_candidate_limit"]),
        "targets": {
            "property_ceiling": property_target,
            "split_property_ceiling": split_property_target,
            "strict_ceiling": strict_target,
            "split_strict_ceiling": split_strict_target,
        },
        "candidate_manifest": manifest,
        "support_ceiling": support,
        "comparison_to_v5": {
            "baseline_property_any_rate": baseline_property,
            "baseline_strict_any_rate": baseline_strict,
            "property_ceiling_gain": float(all_metrics["property_any_rate"]) - baseline_property,
            "strict_ceiling_gain": float(all_metrics["strict_any_rate"]) - baseline_strict,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
