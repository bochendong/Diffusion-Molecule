#!/usr/bin/env python3
"""Audit source/candidate SMILES overlap with an external-property oracle CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402


DEFAULT_COLUMNS = ("source_smiles", "generated_smiles", "target_smiles")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", action="append", default=[], type=Path)
    parser.add_argument("--prediction-csv", action="append", default=[], type=Path)
    parser.add_argument("--oracle-properties-csv", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--sample-size", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    oracle_keys = read_smiles_column(args.oracle_properties_csv, candidates=("smiles", "SMILES", "canonical_smiles"))
    columns = tuple(item.strip() for item in str(args.columns).split(",") if item.strip())
    input_paths = [*args.rows_csv, *args.prediction_csv]
    if not input_paths:
        raise ValueError("At least one --rows-csv or --prediction-csv is required")
    audits = [
        audit_csv(path, columns=columns, oracle_keys=oracle_keys, sample_size=int(args.sample_size))
        for path in input_paths
    ]
    combined = combine_audits(audits, oracle_keys=oracle_keys, sample_size=int(args.sample_size))
    report = {
        "oracle_properties_csv": str(args.oracle_properties_csv),
        "oracle_unique_smiles": len(oracle_keys),
        "inputs": audits,
        "combined": combined,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = render_report(report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text, encoding="utf-8")
    print(text)
    return 0


def audit_csv(
    path: Path,
    *,
    columns: Sequence[str],
    oracle_keys: set[str],
    sample_size: int,
) -> dict[str, object]:
    rows = read_rows(path)
    column_stats = {}
    for column in columns:
        smiles = collect_column_smiles(rows, column)
        missing = sorted(smiles - oracle_keys)
        column_stats[column] = {
            "unique_smiles": len(smiles),
            "oracle_overlap": len(smiles & oracle_keys),
            "oracle_coverage": coverage(len(smiles & oracle_keys), len(smiles)),
            "missing_sample": missing[:sample_size],
        }
    return {
        "path": str(path),
        "rows": len(rows),
        "columns": column_stats,
    }


def combine_audits(
    audits: Sequence[Mapping[str, object]],
    *,
    oracle_keys: set[str],
    sample_size: int,
) -> dict[str, object]:
    by_column: dict[str, set[str]] = {}
    for audit in audits:
        rows = read_rows(Path(str(audit["path"])))
        for column in audit["columns"]:  # type: ignore[index]
            by_column.setdefault(str(column), set()).update(collect_column_smiles(rows, str(column)))
    out = {}
    for column, smiles in sorted(by_column.items()):
        missing = sorted(smiles - oracle_keys)
        out[column] = {
            "unique_smiles": len(smiles),
            "oracle_overlap": len(smiles & oracle_keys),
            "oracle_coverage": coverage(len(smiles & oracle_keys), len(smiles)),
            "missing_sample": missing[:sample_size],
        }
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_smiles_column(path: Path, *, candidates: Sequence[str]) -> set[str]:
    rows = read_rows(path)
    out = set()
    for row in rows:
        for column in candidates:
            value = canonical_or_blank(row.get(column))
            if value:
                out.add(value)
                break
    return out


def collect_column_smiles(rows: Sequence[Mapping[str, str]], column: str) -> set[str]:
    out = set()
    for row in rows:
        value = canonical_or_blank(row.get(column))
        if value:
            out.add(value)
    return out


def canonical_or_blank(value: object) -> str:
    try:
        return canonical_smiles(str(value or "").strip()) or ""
    except RuntimeError:
        return str(value or "").strip()


def coverage(overlap: int, total: int) -> float:
    return float(overlap) / max(int(total), 1)


def render_report(report: Mapping[str, object]) -> str:
    lines = [
        "# External Oracle Coverage Audit",
        "",
        f"- oracle_properties_csv: `{report['oracle_properties_csv']}`",
        f"- oracle_unique_smiles: `{report['oracle_unique_smiles']}`",
        "",
        "## Combined",
        "",
        "| Column | Unique SMILES | Oracle overlap | Coverage | Missing sample |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    combined = report["combined"]  # type: ignore[index]
    for column, stats in combined.items():  # type: ignore[union-attr]
        lines.append(
            "| {column} | {unique_smiles} | {oracle_overlap} | {oracle_coverage:.3f} | {missing_sample} |".format(
                column=column,
                unique_smiles=stats["unique_smiles"],
                oracle_overlap=stats["oracle_overlap"],
                oracle_coverage=float(stats["oracle_coverage"]),
                missing_sample=", ".join(stats["missing_sample"][:5]),  # type: ignore[index]
            )
        )
    lines.extend(["", "## Inputs", ""])
    for audit in report["inputs"]:  # type: ignore[index]
        lines.append(f"### `{audit['path']}`")  # type: ignore[index]
        lines.append("")
        lines.append(f"- rows: `{audit['rows']}`")  # type: ignore[index]
        lines.append("")
        lines.append("| Column | Unique SMILES | Oracle overlap | Coverage | Missing sample |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for column, stats in audit["columns"].items():  # type: ignore[index, union-attr]
            lines.append(
                "| {column} | {unique_smiles} | {oracle_overlap} | {oracle_coverage:.3f} | {missing_sample} |".format(
                    column=column,
                    unique_smiles=stats["unique_smiles"],
                    oracle_overlap=stats["oracle_overlap"],
                    oracle_coverage=float(stats["oracle_coverage"]),
                    missing_sample=", ".join(stats["missing_sample"][:5]),  # type: ignore[index]
                )
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
