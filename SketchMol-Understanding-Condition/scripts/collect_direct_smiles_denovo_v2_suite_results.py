#!/usr/bin/env python3
"""Collect direct-SMILES v2 main-pipeline suite benchmark summaries into one table."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class SuiteVariant:
    slug: str
    label: str
    summary_relpath: str


VARIANTS = (
    SuiteVariant(
        "2p7p_default_n128",
        "2p7p v2 default n=128",
        "2p7p_default_n128/benchmark_direct_smiles/benchmark_summary.csv",
    ),
    SuiteVariant(
        "2p7p_conservative_n128",
        "2p7p v2 conservative n=128",
        "2p7p_conservative_n128/benchmark_direct_smiles/benchmark_summary.csv",
    ),
    SuiteVariant("ood_default_n128", "OOD v2 default n=128", "ood_default_n128/benchmark_direct_smiles/benchmark_summary.csv"),
    SuiteVariant(
        "ood_conservative_n128",
        "OOD v2 conservative n=128",
        "ood_conservative_n128/benchmark_direct_smiles/benchmark_summary.csv",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    report_path = args.report_path or (suite_root / "suite_report.md")
    csv_path = args.csv_path or (suite_root / "suite_summary.csv")

    results = [collect_variant(suite_root, variant) for variant in VARIANTS]
    write_csv(csv_path, results)
    write_report(report_path, suite_root, results)

    payload = {
        "suite_root": str(suite_root),
        "report_path": str(report_path),
        "csv_path": str(csv_path),
        "variants": len(results),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def collect_variant(suite_root: Path, variant: SuiteVariant) -> dict[str, object]:
    summary_path = suite_root / variant.summary_relpath
    row: dict[str, object] = {
        "variant": variant.slug,
        "label": variant.label,
        "summary_csv": str(summary_path),
        "status": "missing",
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_summary_row(rows: list[dict[str, str]], property_count: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("method", "")) != "direct_smiles_mllm":
            continue
        if str(row.get("property_count", "")) == property_count:
            return row
    for row in rows:
        if str(row.get("property_count", "")) == property_count:
            return row
    return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = [
        "variant",
        "label",
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


def write_report(path: Path, suite_root: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# SUCC Direct SMILES De Novo v2 Main Pipeline Suite",
        "",
        f"- suite root: `{suite_root}`",
        f"- generated at: `{generated_at}`",
        "",
        "## Main table",
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
    lines.extend(
        [
            "",
            "## Paths",
            "",
            "| Variant | Summary CSV |",
            "| --- | --- |",
        ]
    )
    for row in rows:
        lines.append(f"| {row['label']} | `{row['summary_csv']}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    main()
