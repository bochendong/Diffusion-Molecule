#!/usr/bin/env python3
"""Build aligned MuMO/C-MuMO comparison and failure-diagnosis reports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SUMMARY_NAME = "external_multiproperty_summary.csv"
DETAIL_NAME = "external_multiproperty_detail.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run label and eval dir in the form LABEL=path/to/eval_dir. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-column", default="condition_id")
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--failure-csv", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [parse_run(value) for value in args.run]
    comparison_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for label, eval_dir in runs:
        summary_path = eval_dir / SUMMARY_NAME
        detail_path = eval_dir / DETAIL_NAME
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary CSV for {label}: {summary_path}")
        if not detail_path.is_file():
            raise FileNotFoundError(f"Missing detail CSV for {label}: {detail_path}")
        for row in read_rows(summary_path):
            comparison_rows.append(make_comparison_row(label, eval_dir, row))
        detail_rows = read_rows(detail_path)
        failure_rows.extend(make_failure_rows(label, eval_dir, detail_rows, group_column=str(args.group_column)))

    comparison_csv = args.comparison_csv or args.output_dir / "external_multiproperty_aligned_comparison.csv"
    failure_csv = args.failure_csv or args.output_dir / "external_multiproperty_failure_breakdown.csv"
    report_md = args.report_md or args.output_dir / "external_multiproperty_aligned_report.md"
    write_rows(comparison_csv, comparison_rows)
    write_rows(failure_csv, failure_rows)
    report_md.write_text(render_report(comparison_rows, failure_rows, runs), encoding="utf-8")
    print(f"comparison_csv={comparison_csv}")
    print(f"failure_csv={failure_csv}")
    print(f"report_md={report_md}")
    return 0


def parse_run(value: str) -> tuple[str, Path]:
    label, sep, path = str(value or "").partition("=")
    if not sep or not label.strip() or not path.strip():
        raise ValueError(f"--run must be LABEL=eval_dir, got: {value!r}")
    return label.strip(), Path(path.strip())


def make_comparison_row(label: str, eval_dir: Path, row: Mapping[str, str]) -> dict[str, object]:
    return {
        "run_label": label,
        "eval_dir": str(eval_dir),
        "external_suite": row.get("external_suite", ""),
        "external_task_split": row.get("external_task_split", ""),
        "external_task_id": row.get("external_task_id", ""),
        "input_groups": row.get("input_groups", ""),
        "candidate_rows": row.get("candidate_rows", ""),
        "validity": row.get("validity", ""),
        "source_available_rate": row.get("source_available_rate", ""),
        "SR": row.get("success_rate", ""),
        "Sim_success": row.get("similarity", ""),
        "RI_success": row.get("relative_improvement", ""),
        "Sim_ge_threshold": row.get("source_similarity_success_rate", ""),
        "property_success_rate": row.get("all_property_success_rate", ""),
        "strict_success_rate": row.get("strict_success_rate", ""),
        "official_evaluable_rate": row.get("official_evaluable_rate", ""),
        "eval_prop_frac": row.get("mean_evaluated_property_fraction", ""),
        "missing_oracle_properties": row.get("missing_oracle_properties", ""),
        "status": row.get("success_rate_status", ""),
    }


def make_failure_rows(
    label: str,
    eval_dir: Path,
    rows: list[dict[str, str]],
    *,
    group_column: str,
) -> list[dict[str, object]]:
    grouped_by_scope: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_by_scope[scope_key(row)].append((index, row))
        grouped_by_scope[("all", "all", "all")].append((index, row))

    out = []
    for (suite, split, task_id), indexed_rows in sorted(grouped_by_scope.items()):
        groups = group_candidate_rows(indexed_rows, group_column=group_column)
        diagnoses = [diagnose_group(items) for items in groups.values()]
        counter = Counter(item["failure_bucket"] for item in diagnoses)
        input_groups = len(diagnoses)
        candidate_rows = sum(len(items) for items in groups.values())
        out.append(
            {
                "run_label": label,
                "eval_dir": str(eval_dir),
                "external_suite": suite,
                "external_task_split": split,
                "external_task_id": task_id,
                "input_groups": input_groups,
                "candidate_rows": candidate_rows,
                "valid_any_rate": format_rate(count_bool(diagnoses, "valid_any"), input_groups),
                "official_evaluable_rate": format_rate(count_bool(diagnoses, "official_evaluable_any"), input_groups),
                "official_success_rate": format_rate(count_bool(diagnoses, "official_success_any"), input_groups),
                "similarity_success_rate": format_rate(count_bool(diagnoses, "similarity_success_any"), input_groups),
                "strict_success_rate": format_rate(count_bool(diagnoses, "strict_success_any"), input_groups),
                "invalid_groups": counter.get("invalid", 0),
                "missing_oracle_groups": counter.get("missing_oracle", 0),
                "property_failed_groups": counter.get("property_failed", 0),
                "property_pass_similarity_failed_groups": counter.get("property_pass_similarity_failed", 0),
                "similarity_pass_property_failed_groups": counter.get("similarity_pass_property_failed", 0),
                "property_and_similarity_failed_groups": counter.get("property_and_similarity_failed", 0),
                "official_success_groups": counter.get("official_success", 0),
                "official_success_similarity_failed_groups": count_bool(
                    diagnoses,
                    "official_success_similarity_failed",
                ),
                "strict_gap_after_official_success_groups": count_bool(
                    diagnoses,
                    "strict_gap_after_official_success",
                ),
            }
        )
    return out


def scope_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("external_suite") or "unknown"),
        str(row.get("external_task_split") or "unknown"),
        str(row.get("external_task_id") or row.get("external_task_key") or "unknown"),
    )


def group_candidate_rows(
    indexed_rows: list[tuple[int, dict[str, str]]],
    *,
    group_column: str,
) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in indexed_rows:
        key = str(
            row.get(group_column)
            or row.get("condition_id")
            or row.get("sample_id")
            or row.get("variant_id")
            or row.get("pair_id")
            or f"row_{index:08d}"
        )
        out[key].append(row)
    return out


def diagnose_group(items: Sequence[Mapping[str, str]]) -> dict[str, object]:
    valid_any = any(truthy(item.get("external_valid")) for item in items)
    official_evaluable_any = any(
        truthy(item.get("external_valid")) and truthy(item.get("external_full_property_coverage")) for item in items
    )
    property_success_any = any(truthy(item.get("external_all_property_success")) for item in items)
    similarity_success_any = any(truthy(item.get("external_source_similarity_success")) for item in items)
    official_success_any = any(truthy(item.get("external_official_success")) for item in items)
    strict_success_any = any(truthy(item.get("external_strict_success")) for item in items)

    if official_success_any:
        bucket = "official_success"
    elif not valid_any:
        bucket = "invalid"
    elif not official_evaluable_any:
        bucket = "missing_oracle"
    elif property_success_any and not similarity_success_any:
        bucket = "property_pass_similarity_failed"
    elif similarity_success_any and not property_success_any:
        bucket = "similarity_pass_property_failed"
    elif not property_success_any and not similarity_success_any:
        bucket = "property_and_similarity_failed"
    else:
        bucket = "property_failed"

    return {
        "valid_any": valid_any,
        "official_evaluable_any": official_evaluable_any,
        "property_success_any": property_success_any,
        "similarity_success_any": similarity_success_any,
        "official_success_any": official_success_any,
        "strict_success_any": strict_success_any,
        "official_success_similarity_failed": official_success_any and not similarity_success_any,
        "strict_gap_after_official_success": official_success_any and not strict_success_any,
        "failure_bucket": bucket,
    }


def render_report(
    comparison_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
    runs: Sequence[tuple[str, Path]],
) -> str:
    lines = [
        "# External Multi-property Aligned Report",
        "",
        "Runs:",
    ]
    for label, eval_dir in runs:
        lines.append(f"- {label}: `{eval_dir}`")
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Run | Suite | Split | Inputs | Candidates | SR | Sim(success) | RI(success) | Sim>=0.4 | Strict | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        if row.get("external_suite") != "all":
            continue
        lines.append(
            "| {run_label} | {external_suite} | {external_task_split} | {input_groups} | "
            "{candidate_rows} | {SR} | {Sim_success} | {RI_success} | {Sim_ge_threshold} | "
            "{strict_success_rate} | {status} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Task Breakdown",
            "",
            "| Run | Suite | Split | Task | Inputs | SR | Sim(success) | RI(success) | Sim>=0.4 | Strict | Missing oracle |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        if row.get("external_suite") == "all":
            continue
        lines.append(
            "| {run_label} | {external_suite} | {external_task_split} | {external_task_id} | "
            "{input_groups} | {SR} | {Sim_success} | {RI_success} | {Sim_ge_threshold} | "
            "{strict_success_rate} | {missing_oracle_properties} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Failure Diagnosis",
            "",
            "| Run | Suite | Split | Task | Inputs | Invalid | Missing oracle | Prop fail | Prop pass / Sim fail | Sim pass / Prop fail | Both fail | Official hit but Sim fail |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in failure_rows:
        if row.get("external_suite") != "all":
            continue
        lines.append(format_failure_row(row))
    for row in failure_rows:
        if row.get("external_suite") == "all":
            continue
        lines.append(format_failure_row(row))
    lines.extend(
        [
            "",
            "`SR` is the external-paper style candidate-level success rate. `Strict` adds the source-similarity threshold.",
            "`Official hit but Sim fail` is the main warning bucket when official SR looks good but source preservation is weak.",
            "",
        ]
    )
    return "\n".join(lines)


def format_failure_row(row: Mapping[str, object]) -> str:
    return (
        "| {run_label} | {external_suite} | {external_task_split} | {external_task_id} | "
        "{input_groups} | {invalid_groups} | {missing_oracle_groups} | {property_failed_groups} | "
        "{property_pass_similarity_failed_groups} | {similarity_pass_property_failed_groups} | "
        "{property_and_similarity_failed_groups} | {official_success_similarity_failed_groups} |"
    ).format(**row)


def count_bool(rows: Sequence[Mapping[str, object]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def format_rate(count: int, total: int) -> str:
    if total <= 0:
        return ""
    text = f"{count / total:.6f}"
    return text.rstrip("0").rstrip(".")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
