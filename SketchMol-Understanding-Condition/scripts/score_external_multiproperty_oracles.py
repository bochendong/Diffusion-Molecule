#!/usr/bin/env python3
"""Build property-oracle CSVs for MuMO/C-MuMO official evaluation.

This script collects unique SMILES from prediction/row CSVs, merges any
precomputed ADMET oracle CSVs, fills locally computable properties, and writes a
single `smiles,property...` CSV consumable by
`evaluate_external_multiproperty_predictions.py`.

It intentionally reports missing coverage instead of silently treating absent
ADMET predictors as failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties  # noqa: E402


PROPERTY_ALIASES = {
    "ames": "mutagenicity",
    "mutagen": "mutagenicity",
    "mutagenicity": "mutagenicity",
    "bbbp": "bbbp",
    "bbb_martins": "bbbp",
    "hia": "hia",
    "hia_hou": "hia",
    "qed": "qed",
    "plogp": "plogp",
    "penalized_logp": "plogp",
    "logp": "logp",
    "drd2": "drd2",
    "carc": "carc",
    "carcinogenicity": "carc",
    "erg": "erg",
    "herg": "erg",
    "liver": "liver",
    "dili": "liver",
    "ampa": "ampa",
    "pampa": "ampa",
    "sas": "sas",
    "sa": "sas",
}
DEFAULT_PROPERTIES = (
    "ampa",
    "bbbp",
    "carc",
    "drd2",
    "erg",
    "hia",
    "liver",
    "mutagenicity",
    "plogp",
    "qed",
    "sas",
)
TDC_ORACLE_NAMES = {
    "drd2": "DRD2",
    "qed": "QED",
    "sas": "SA",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", default=[], type=Path, help="Prediction/row CSV to scan.")
    parser.add_argument("--merge-properties-csv", action="append", default=[], type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument(
        "--smiles-columns",
        default="generated_smiles,source_smiles,target_smiles,smiles,SMILES,canonical_smiles",
    )
    parser.add_argument("--required-properties", default=",".join(DEFAULT_PROPERTIES))
    parser.add_argument("--tdc-oracles", choices=("auto", "never"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input_csv:
        raise ValueError("At least one --input-csv is required")
    smiles_columns = [item.strip() for item in str(args.smiles_columns).split(",") if item.strip()]
    required_props = [canonical_prop(item) for item in str(args.required_properties).split(",") if item.strip()]
    smiles = collect_smiles(args.input_csv, smiles_columns=smiles_columns)
    merged = merge_property_csvs(args.merge_properties_csv)
    rows = []
    missing_counter = Counter()
    for smi in sorted(smiles):
        props = dict(merged.get(smi, {}))
        props.update({key: value for key, value in local_properties(smi).items() if key not in props})
        if str(args.tdc_oracles) == "auto":
            for prop in required_props:
                if prop not in props:
                    value = tdc_property(smi, prop)
                    if value is not None:
                        props[prop] = value
        row = {"smiles": smi}
        for prop in required_props:
            value = props.get(prop)
            if value is None:
                missing_counter[prop] += 1
                row[prop] = ""
            else:
                row[prop] = format_float(value, digits=8)
        rows.append(row)
    write_rows(args.output_csv, rows, ["smiles", *required_props])
    report = {
        "input_csvs": [str(path) for path in args.input_csv],
        "merge_properties_csvs": [str(path) for path in args.merge_properties_csv],
        "output_csv": str(args.output_csv),
        "unique_smiles": len(rows),
        "required_properties": required_props,
        "missing_counts": {prop: int(missing_counter[prop]) for prop in required_props},
        "coverage": {
            prop: 1.0 - (float(missing_counter[prop]) / max(len(rows), 1))
            for prop in required_props
        },
    }
    report_json = args.report_json or args.output_csv.with_suffix(".summary.json")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md = args.report_md or args.output_csv.with_suffix(".report.md")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(render_report(report), encoding="utf-8")
    print(render_report(report))
    return 0


def collect_smiles(paths: Sequence[Path], *, smiles_columns: Sequence[str]) -> set[str]:
    out = set()
    for path in paths:
        for row in read_rows(path):
            for column in smiles_columns:
                smi = canonical_or_blank(row.get(column))
                if smi:
                    out.add(smi)
    return out


def merge_property_csvs(paths: Sequence[Path]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in paths:
        if not path or not path.exists():
            continue
        for row in read_rows(path):
            smi = canonical_or_blank(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles"))
            if not smi:
                continue
            props = out.setdefault(smi, {})
            for key, value in row.items():
                prop = canonical_prop(key)
                if prop not in DEFAULT_PROPERTIES and prop != "logp":
                    continue
                parsed = parse_float(value)
                if parsed is not None:
                    props[prop] = parsed
    return out


def local_properties(smiles: str) -> dict[str, float]:
    try:
        props = molecular_properties(smiles) or {}
    except RuntimeError:
        props = {}
    out = {}
    if "QED" in props:
        out["qed"] = float(props["QED"])
    if "LogP" in props:
        out["logp"] = float(props["LogP"])
        out["plogp"] = float(props["LogP"]) - float(props.get("SA", 0.0))
    if "SA" in props:
        out["sas"] = float(props["SA"])
    return out


_TDC_CACHE: dict[str, object | None] = {}


def tdc_property(smiles: str, prop: str) -> float | None:
    prop = canonical_prop(prop)
    oracle_name = TDC_ORACLE_NAMES.get(prop)
    if not oracle_name:
        return None
    oracle = tdc_oracle(prop, oracle_name)
    if oracle is None:
        return None
    try:
        return float(oracle(smiles))  # type: ignore[operator]
    except Exception:
        return None


def tdc_oracle(prop: str, oracle_name: str):
    if prop in _TDC_CACHE:
        return _TDC_CACHE[prop]
    ensure_rdkit_six_compat()
    try:
        from tdc import Oracle

        _TDC_CACHE[prop] = Oracle(name=oracle_name)
    except Exception:
        _TDC_CACHE[prop] = None
    return _TDC_CACHE[prop]


def ensure_rdkit_six_compat() -> None:
    if "rdkit.six" in sys.modules:
        return
    try:
        from rdkit.six import iteritems  # noqa: F401
    except ModuleNotFoundError:
        six_mod = types.ModuleType("rdkit.six")
        six_mod.iteritems = dict.items
        sys.modules["rdkit.six"] = six_mod


def render_report(report: Mapping[str, object]) -> str:
    lines = [
        "# External Multi-property Oracle Coverage",
        "",
        f"- output_csv: `{report['output_csv']}`",
        f"- unique_smiles: `{report['unique_smiles']}`",
        "",
        "| Property | Coverage | Missing |",
        "| --- | ---: | ---: |",
    ]
    coverage = report["coverage"]  # type: ignore[index]
    missing = report["missing_counts"]  # type: ignore[index]
    for prop in report["required_properties"]:  # type: ignore[index]
        lines.append(f"| {prop} | {float(coverage[prop]):.3f} | {missing[prop]} |")  # type: ignore[index]
    lines.append("")
    lines.append(
        "Missing ADMET columns must be supplied by an external predictor CSV before official SR can be claimed."
    )
    return "\n".join(lines)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def canonical_or_blank(value: object) -> str:
    try:
        return canonical_smiles(str(value or "").strip()) or ""
    except RuntimeError:
        return str(value or "").strip()


def canonical_prop(value: object) -> str:
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return PROPERTY_ALIASES.get(key, key)


def parse_float(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def format_float(value: float, *, digits: int = 6) -> str:
    if not math.isfinite(float(value)):
        return ""
    text = f"{float(value):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


if __name__ == "__main__":
    raise SystemExit(main())
