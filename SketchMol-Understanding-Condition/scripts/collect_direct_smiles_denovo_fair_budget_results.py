#!/usr/bin/env python3
"""Collect direct-SMILES fair-budget de novo summaries into one report."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class FairVariant:
    slug: str
    label: str
    benchmark: str
    budget: int
    summary_relpath: str


DEFAULT_VARIANTS = (
    FairVariant("2p7p_n1", "2p-7p Ours@1", "2p-7p", 1, "2p7p_n1/benchmark_direct_smiles_group_rl_n1/benchmark_summary.csv"),
    FairVariant("2p7p_n40", "2p-7p Ours@40", "2p-7p", 40, "2p7p_n40/benchmark_direct_smiles_group_rl_n40/benchmark_summary.csv"),
    FairVariant("ood_n1", "OOD Ours@1", "OOD", 1, "ood_n1/benchmark_direct_smiles_group_rl_n1/benchmark_summary.csv"),
    FairVariant("ood_n40", "OOD Ours@40", "OOD", 40, "ood_n40/benchmark_direct_smiles_group_rl_n40/benchmark_summary.csv"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite_root = args.suite_root.resolve()
    report_path = args.report_path or suite_root / "fair_budget_report.md"
    csv_path = args.csv_path or suite_root / "fair_budget_summary.csv"
    rows = [collect_variant(suite_root, variant) for variant in DEFAULT_VARIANTS]
    write_rows(csv_path, rows)
    report_path.write_text(render_report(suite_root, rows), encoding="utf-8")
    print(json.dumps({"suite_root": str(suite_root), "report_path": str(report_path), "csv_path": str(csv_path)}, indent=2))
    return 0


def collect_variant(suite_root: Path, variant: FairVariant) -> dict[str, object]:
    summary_path = suite_root / variant.summary_relpath
    row: dict[str, object] = {
        "variant": variant.slug,
        "label": variant.label,
        "benchmark": variant.benchmark,
        "budget": variant.budget,
        "status": "missing",
        "summary_csv": str(summary_path),
        "overall_strict": "",
        "validity": "",
        "strict_in_valid": "",
        "unique_valid_smiles": "",
        "uniqueness_in_valid": "",
    }
    for count in range(2, 8):
        row[f"strict_{count}p"] = ""
    if not summary_path.exists():
        return row
    rows = read_rows(summary_path)
    all_row = select_summary_row(rows, "all")
    if all_row is None:
        row["status"] = "malformed"
        return row
    row["status"] = "complete"
    row["overall_strict"] = all_row.get("strict_success_rate", "")
    row["validity"] = all_row.get("validity", "")
    row["strict_in_valid"] = all_row.get("success_rate_strict_in_valid_mols", "")
    row["unique_valid_smiles"] = all_row.get("unique_valid_smiles", "")
    row["uniqueness_in_valid"] = all_row.get("uniqueness_in_valid_mols", "")
    for count in range(2, 8):
        count_row = select_summary_row(rows, str(count))
        if count_row is not None:
            row[f"strict_{count}p"] = count_row.get("strict_success_rate", "")
    return row


def select_summary_row(rows: Sequence[dict[str, str]], property_count: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("method", "")) != "direct_smiles_mllm":
            continue
        if str(row.get("property_count", "")) == property_count:
            return row
    for row in rows:
        if str(row.get("property_count", "")) == property_count:
            return row
    return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "label",
        "benchmark",
        "budget",
        "status",
        "summary_csv",
        "overall_strict",
        "validity",
        "strict_in_valid",
        "unique_valid_smiles",
        "uniqueness_in_valid",
    ]
    fieldnames.extend(f"strict_{count}p" for count in range(2, 8))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(suite_root: Path, rows: Sequence[dict[str, object]]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Direct SMILES De Novo Fair-Budget Suite",
        "",
        f"- suite root: `{suite_root}`",
        f"- generated at: `{generated_at}`",
        "- purpose: compare Ours@1 and Ours@40 against SketchMol-like candidate budgets before reporting Ours@256.",
        "",
        "| Variant | Status | Strict | Validity | Strict-in-valid | Unique valid | 2p | 3p | 4p | 5p | 6p | 7p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["status"]),
                    fmt(row.get("overall_strict")),
                    fmt(row.get("validity")),
                    fmt(row.get("strict_in_valid")),
                    fmt(row.get("unique_valid_smiles"), integer=True),
                    fmt(row.get("strict_2p")),
                    fmt(row.get("strict_3p")),
                    fmt(row.get("strict_4p")),
                    fmt(row.get("strict_5p")),
                    fmt(row.get("strict_6p")),
                    fmt(row.get("strict_7p")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Summary CSV Paths", "", "| Variant | Summary CSV |", "| --- | --- |"])
    for row in rows:
        lines.append(f"| {row['label']} | `{row['summary_csv']}` |")
    lines.append("")
    return "\n".join(lines)


def fmt(value: object, *, integer: bool = False) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if integer:
        return str(int(number))
    return f"{number:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
