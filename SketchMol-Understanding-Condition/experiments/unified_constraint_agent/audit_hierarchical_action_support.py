#!/usr/bin/env python3
"""Gate a train-only hierarchical proposal-plus-edit pool at fixed final n=20."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposer-train-csv", required=True, type=Path)
    parser.add_argument("--audit-rows-csv", required=True, type=Path)
    parser.add_argument("--official-detail-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--min-property-any-rate", type=float, default=0.20)
    parser.add_argument("--min-strict-any-rate", type=float, default=0.05)
    parser.add_argument("--min-full-oracle-condition-rate", type=float, default=1.0)
    parser.add_argument("--protocol", default="hierarchical_common_agent_action_support_v4")
    parser.add_argument("--proposal-budget", type=int, default=1)
    parser.add_argument("--method-label", default="raw-1 proposal plus GraphEditDSL")
    parser.add_argument("--candidate-manifest-json", type=Path, default=None)
    parser.add_argument("--validate-splits-only", action="store_true")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def source_key(row: Mapping[str, object]) -> tuple[str, ...]:
    """Identify the underlying paired example, not the regenerated condition id."""
    source_index = str(row.get("external_source_row_index", "") or "").strip()
    task = str(
        row.get("external_task_id", "")
        or row.get("external_task_key", "")
        or row.get("benchmark_task", "")
        or ""
    ).strip()
    if source_index:
        # The same source row can be projected into multiple benchmark tasks;
        # keep all of those projections on one side of the split.
        return ("external", source_index)
    for key in ("pair_id", "sample_id", "variant_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return (key, task, value)
    return (
        "molecule_pair",
        task,
        str(row.get("source_smiles", "") or "").strip(),
        str(row.get("target_smiles", "") or "").strip(),
    )


def validate_disjoint_rows(
    proposer_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    proposer_keys = {source_key(row) for row in proposer_rows}
    audit_keys = {source_key(row) for row in audit_rows}
    overlap = proposer_keys & audit_keys
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"Proposer-train/audit source overlap: {len(overlap)} examples; preview={preview}")
    return {
        "proposer_train_rows": len(proposer_rows),
        "proposer_train_unique_sources": len(proposer_keys),
        "audit_rows": len(audit_rows),
        "audit_unique_sources": len(audit_keys),
        "source_overlap": 0,
    }


def condition_id(row: Mapping[str, object]) -> str:
    value = str(row.get("condition_id", "") or "").strip()
    if not value:
        raise ValueError("Official detail row is missing condition_id")
    return value


def split_value(row: Mapping[str, object]) -> str:
    return str(row.get("external_task_split", "") or row.get("split", "") or "all").strip().lower()


def property_success(row: Mapping[str, object]) -> bool:
    return truthy(row.get("external_official_success"))


def strict_success(row: Mapping[str, object]) -> bool:
    return truthy(row.get("external_strict_success"))


def valid_candidate(row: Mapping[str, object]) -> bool:
    value = str(row.get("external_valid", "") or "").strip()
    if value:
        return truthy(value)
    return bool(str(row.get("generated_smiles", "") or "").strip())


def summarize_groups(groups: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, object]:
    denominator = max(len(groups), 1)
    candidate_count = sum(len(group) for group in groups)
    return {
        "conditions": len(groups),
        "property_any_rate": sum(any(property_success(row) for row in group) for group in groups) / denominator,
        "strict_any_rate": sum(any(strict_success(row) for row in group) for group in groups) / denominator,
        "valid_any_rate": sum(any(valid_candidate(row) for row in group) for group in groups) / denominator,
        "mean_valid_candidates": sum(sum(valid_candidate(row) for row in group) for group in groups) / denominator,
        "full_oracle_candidate_rate": (
            sum(truthy(row.get("external_full_property_coverage")) for group in groups for row in group)
            / max(candidate_count, 1)
        ),
        "full_oracle_condition_rate": (
            sum(all(truthy(row.get("external_full_property_coverage")) for row in group) for group in groups)
            / denominator
        ),
        "direct_root_in_prefix_rate": sum(
            any(str(row.get("graph_edit_candidate_source", "") or "") == "direct_model" for row in group)
            for group in groups
        )
        / denominator,
    }


def summarize_official_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate_budget: int,
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[condition_id(row)].append(row)
    incomplete = {key: len(group) for key, group in grouped.items() if len(group) != int(candidate_budget)}
    if incomplete:
        raise ValueError(f"Fixed n={candidate_budget} pool is incomplete: {list(incomplete.items())[:5]}")
    groups = list(grouped.values())
    by_split: dict[str, list[Sequence[Mapping[str, object]]]] = defaultdict(list)
    for group in groups:
        by_split[split_value(group[0])].append(group)
    source_counts = Counter(
        str(row.get("graph_edit_candidate_source", "") or "unknown")
        for row in rows
    )
    return {
        "candidate_budget": int(candidate_budget),
        "candidate_rows": len(rows),
        "complete_conditions": len(groups),
        "all": summarize_groups(groups),
        "by_split": {
            name: summarize_groups(split_groups)
            for name, split_groups in sorted(by_split.items())
        },
        "candidate_source_counts": dict(sorted(source_counts.items())),
    }


def render_report(summary: Mapping[str, object]) -> str:
    support = summary["support"]
    all_metrics = support["all"]
    lines = [
        "# Hierarchical Common-Agent Action-Support Gate",
        "",
        f"Decision: **{str(summary['decision']).upper()}**",
        "",
        "This is a train-only support diagnostic, not a formal benchmark result. "
        f"The candidate tool is {summary['method_label']}; "
        "the official property stack evaluates exactly 20 final molecules per condition. Internal enumeration "
        "is recorded as internal search compute and is not counted as additional oracle candidates.",
        "",
        "| Scope | Conditions | Property any@20 | Strict any@20 | Full-oracle groups | Valid candidates | Direct root retained |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| all | {all_metrics['conditions']} | {all_metrics['property_any_rate']:.1%} | "
            f"{all_metrics['strict_any_rate']:.1%} | {all_metrics['full_oracle_condition_rate']:.1%} | "
            f"{all_metrics['mean_valid_candidates']:.1f} | "
            f"{all_metrics['direct_root_in_prefix_rate']:.1%} |"
        ),
    ]
    for name, metrics in support["by_split"].items():
        lines.append(
            f"| {name} | {metrics['conditions']} | {metrics['property_any_rate']:.1%} | "
            f"{metrics['strict_any_rate']:.1%} | {metrics['full_oracle_condition_rate']:.1%} | "
            f"{metrics['mean_valid_candidates']:.1f} | "
            f"{metrics['direct_root_in_prefix_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Advance requires complete n=20 groups, property any@20 at or above "
            f"{summary['thresholds']['min_property_any_rate']:.1%}, and strict any@20 at or above "
            f"{summary['thresholds']['min_strict_any_rate']:.1%}, with full-oracle groups at or above "
            f"{summary['thresholds']['min_full_oracle_condition_rate']:.1%}.",
            "",
            "If the gate stops, the next change belongs in the proposal/action-support tool. If it advances, "
            "the next experiment trains the common LLM on complete proposal-plus-edit tool plans.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split_audit = validate_disjoint_rows(
        read_rows(args.proposer_train_csv),
        read_rows(args.audit_rows_csv),
    )
    if args.validate_splits_only:
        print(json.dumps(split_audit, indent=2, sort_keys=True))
        return 0
    if args.official_detail_csv is None or args.output_dir is None:
        raise ValueError("--official-detail-csv and --output-dir are required unless --validate-splits-only is set")
    support = summarize_official_rows(
        read_rows(args.official_detail_csv),
        candidate_budget=int(args.candidate_budget),
    )
    all_metrics = support["all"]
    advance = (
        float(all_metrics["property_any_rate"]) >= float(args.min_property_any_rate)
        and float(all_metrics["strict_any_rate"]) >= float(args.min_strict_any_rate)
        and float(all_metrics["full_oracle_condition_rate"])
        >= float(args.min_full_oracle_condition_rate)
    )
    summary = {
        "protocol": str(args.protocol),
        "data_role": "train_only_heldout",
        "method_label": str(args.method_label),
        "decision": "advance" if advance else "stop",
        "proposal_budget": int(args.proposal_budget),
        "final_oracle_candidate_budget": int(args.candidate_budget),
        "split_audit": split_audit,
        "thresholds": {
            "min_property_any_rate": float(args.min_property_any_rate),
            "min_strict_any_rate": float(args.min_strict_any_rate),
            "min_full_oracle_condition_rate": float(args.min_full_oracle_condition_rate),
        },
        "support": support,
    }
    if args.candidate_manifest_json is not None:
        candidate_manifest = json.loads(args.candidate_manifest_json.read_text(encoding="utf-8"))
        if not isinstance(candidate_manifest, dict):
            raise ValueError("Candidate manifest must contain one JSON object")
        summary["candidate_builder"] = candidate_manifest
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
