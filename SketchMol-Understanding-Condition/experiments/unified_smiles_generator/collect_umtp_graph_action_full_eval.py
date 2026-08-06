#!/usr/bin/env python3
"""Collect full-test Table1 metrics for the protected GraphEditDSL policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_umtp_v1_rl_pilot as paired  # noqa: E402


SELECTIONS = ("raw", "finalizer")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8,20,64,256")
    return parser.parse_args(argv)


def parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if not values or 1 not in values:
        raise ValueError("Full evaluation budgets must include n=1")
    return values


def summary_path(root: Path, budget: int, selection: str) -> Path:
    return root / "moledit_table1" / f"n{budget}" / selection / "metrics" / "moledit_table_summary.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect(root: Path, budgets: Sequence[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aggregate = []
    per_task = []
    for budget in budgets:
        for selection in SELECTIONS:
            path = summary_path(root, budget, selection)
            metrics = paired.table1_summary(path)
            aggregate.append({"budget": int(budget), "selection": selection, **metrics})
            for row in read_rows(path):
                per_task.append(
                    {
                        "budget": int(budget),
                        "selection": selection,
                        "task": row.get("task", row.get("task_key", "")),
                        "task_key": row.get("task_key", ""),
                        "n": row.get("n", ""),
                        "validity": row.get("Validity", ""),
                        "acc_all_0_65": row.get("Acc_all(0.65)", ""),
                        "acc_all_0_15": row.get("Acc_all(0.15)", ""),
                    }
                )
    return aggregate, per_task


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    parsed = paired.parse_float(value)
    return "" if not math.isfinite(parsed) else f"{100.0 * parsed:.1f}%"


def write_report(
    path: Path,
    aggregate: Sequence[Mapping[str, object]],
    per_task: Sequence[Mapping[str, object]],
    candidate_summary: Mapping[str, object],
    budgets: Sequence[int],
) -> None:
    lines = [
        "# Protected GraphEditDSL Full Table1 Evaluation",
        "",
        "The checkpoint is unchanged from the protected pilot. Candidate selection follows the official source-relative `instruction_tasks` predicate.",
        "",
        "## Aggregate",
        "",
        "| n | Selection | Validity | Acc_all@0.65 | Acc_all@0.15 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['budget']} | {row['selection']} | {pct(row.get('validity'))} | "
            f"{pct(row.get('acc_all_0_65'))} | {pct(row.get('acc_all_0_15'))} |"
        )
    largest = max(budgets)
    lines.extend(
        [
            "",
            f"## Per-task n={largest} finalizer",
            "",
            "| Task | n | Validity | Acc_all@0.65 | Acc_all@0.15 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_task:
        if int(row["budget"]) == largest and row["selection"] == "finalizer":
            lines.append(
                f"| {row['task']} | {row['n']} | {pct(row.get('validity'))} | "
                f"{pct(row.get('acc_all_0_65'))} | {pct(row.get('acc_all_0_15'))} |"
            )
    lines.extend(
        [
            "",
            "## Candidate pool",
            "",
            f"- Evaluated rows: {candidate_summary.get('eval_rows', '')}",
            f"- Rows with executable candidates: {candidate_summary.get('rows_with_candidates', '')}",
            f"- Mean executable candidates: {candidate_summary.get('mean_executable_candidates', '')}",
            f"- Candidate rows written: {candidate_summary.get('candidate_rows', '')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(str(args.budgets))
    aggregate, per_task = collect(args.eval_root, budgets)
    candidate_summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    aggregate_csv = args.output_prefix.with_name(args.output_prefix.name + "_aggregate.csv")
    per_task_csv = args.output_prefix.with_name(args.output_prefix.name + "_per_task.csv")
    summary_json = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    report_md = args.output_prefix.with_name(args.output_prefix.name + "_report.md")
    write_csv(aggregate_csv, aggregate)
    write_csv(per_task_csv, per_task)
    payload = {
        "protocol": "protected_graph_edit_dsl_full_table1",
        "budgets": list(budgets),
        "candidate_csv": str(args.candidate_csv),
        "candidate_summary": candidate_summary,
        "aggregate": aggregate,
        "per_task": per_task,
        "aggregate_csv": str(aggregate_csv),
        "per_task_csv": str(per_task_csv),
        "report": str(report_md),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_md, aggregate, per_task, candidate_summary, budgets)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
