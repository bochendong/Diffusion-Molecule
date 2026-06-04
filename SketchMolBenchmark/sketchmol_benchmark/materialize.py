"""Materialize real SketchMol + OCR outputs into benchmark artifacts.

The expected input is the `image_path.csv` written by the original SketchMol
sampling scripts after it has been processed by SketchMol's MolScribe wrapper:

    evaluate/predict_csv.py --model_path <molscribe.pth> --image_path image_path.csv

That CSV should contain at least `image_path` and `SMILES`. When RDKit is
available, this module also computes validity and lightweight property-match
metrics for rows that include SketchMol condition columns such as
`logp_setting`, `QED_setting`, `MW_setting`, `TPSA_setting`, `HBD_setting`,
`HBA_setting`, and `rotatable_setting`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence


PROPERTY_COLUMNS = {
    "LogP": (("logp_setting", "LogP_setting"), ("logp_None", "logp_setting_None", "LogP_None")),
    "QED": (("QED_setting",), ("QED_None", "QED_setting_None")),
    "MW": (("MolWt_setting", "MW_setting"), ("MolWt_None", "MW_setting_None", "MW_None")),
    "TPSA": (("TPSA_setting",), ("TPSA_None", "TPSA_setting_None")),
    "HBD": (("HBD_setting",), ("HBD_None", "HBD_setting_None")),
    "HBA": (("HBA_setting",), ("HBA_None", "HBA_setting_None")),
    "RB": (("rotatable_setting", "RB_setting"), ("rotatable_None", "rotatable_setting_None", "RB_None")),
}

PROPERTY_TOLERANCES = {
    "LogP": 0.5,
    "QED": 0.1,
    "MW": 50.0,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}

SUMMARY_COLUMNS = [
    "benchmark_task",
    "benchmark_label",
    "n",
    "image_path_exists_fraction",
    "ocr_smiles_present_rate",
    "predicted_smiles_present_rate",
    "molscribe_score_mean",
    "validity",
    "success_rate_in_valid_mols",
    "success_rate_strict_in_valid_mols",
    "success_rate_sketchmol_tolerance_in_valid_mols",
    "LogP_mae",
    "QED_mae",
    "MW_mae",
    "TPSA_mae",
    "HBD_mae",
    "HBA_mae",
    "RB_mae",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: Sequence[Mapping[str, object]], path: Path, preferred: Sequence[str] | None = None) -> None:
    seen = set()
    fieldnames: List[str] = []
    for field in preferred or []:
        if any(field in row for row in rows):
            fieldnames.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_value(row.get(field, "")) for field in fieldnames})


def to_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def clean_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def truthy_flag(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def smiles_present(smiles: object) -> bool:
    text = str(smiles or "").strip()
    return bool(text) and text.lower() not in {"none", "nan", "invalid"}


def load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

        return {
            "Chem": Chem,
            "Crippen": Crippen,
            "Descriptors": Descriptors,
            "Lipinski": Lipinski,
            "QED": QED,
            "rdMolDescriptors": rdMolDescriptors,
        }
    except Exception:
        return None


def calc_properties(mol, rdkit_modules: Mapping[str, object]) -> Dict[str, float]:
    return {
        "LogP": float(rdkit_modules["Crippen"].MolLogP(mol)),
        "QED": float(rdkit_modules["QED"].qed(mol)),
        "MW": float(rdkit_modules["Descriptors"].MolWt(mol)),
        "TPSA": float(rdkit_modules["rdMolDescriptors"].CalcTPSA(mol)),
        "HBD": float(rdkit_modules["Lipinski"].NumHDonors(mol)),
        "HBA": float(rdkit_modules["Lipinski"].NumHAcceptors(mol)),
        "RB": float(rdkit_modules["Lipinski"].NumRotatableBonds(mol)),
    }


def first_float(row: Mapping[str, str], columns: Iterable[str]) -> float:
    for column in columns:
        value = to_float(row.get(column))
        if not math.isnan(value):
            return value
    return math.nan


def first_flag(row: Mapping[str, str], columns: Iterable[str]) -> bool | None:
    for column in columns:
        if column not in row:
            continue
        flag = truthy_flag(row.get(column))
        if flag is not None:
            return flag
    return None


def active_target(row: Mapping[str, str], prop: str) -> float:
    value_cols, flag_cols = PROPERTY_COLUMNS[prop]
    value = first_float(row, value_cols)
    if math.isnan(value):
        return math.nan

    flag = first_flag(row, flag_cols)
    if flag is True:
        return math.nan
    return value


def summarise_rows(
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    source_csv: Path,
    *,
    smiles_column: str = "SMILES",
    image_column: str = "image_path",
    benchmark_task: str = "sketchmol_plus_ocr",
    present_rate_field: str = "ocr_smiles_present_rate",
    present_detail_field: str = "ocr_smiles_present",
) -> Dict[str, object]:
    rdkit_modules = load_rdkit()
    detailed_rows: List[Dict[str, object]] = []
    valid_count = 0
    present_count = 0
    image_exists_count = 0
    molscribe_scores: List[float] = []
    property_success_flags: List[bool] = []
    mae_values: Dict[str, List[float]] = {prop: [] for prop in PROPERTY_COLUMNS}

    for index, row in enumerate(rows):
        image_path = str(row.get(image_column, "")).strip()
        image_exists = bool(image_path) and Path(image_path).exists()
        image_exists_count += int(image_exists)

        smiles = row.get(smiles_column, "")
        present = smiles_present(smiles)
        present_count += int(present)

        score = to_float(row.get("molscribe_score"))
        if not math.isnan(score):
            molscribe_scores.append(score)

        valid = False
        properties: Dict[str, float] = {}
        if present and rdkit_modules is not None:
            mol = rdkit_modules["Chem"].MolFromSmiles(str(smiles))
            if mol is not None:
                valid = True
                valid_count += 1
                properties = calc_properties(mol, rdkit_modules)
        elif present and rdkit_modules is None:
            valid = False

        row_successes: List[bool] = []
        for prop in PROPERTY_COLUMNS:
            target = active_target(row, prop)
            if math.isnan(target) or prop not in properties:
                continue
            error = abs(properties[prop] - target)
            mae_values[prop].append(error)
            row_successes.append(error <= PROPERTY_TOLERANCES[prop])
        if valid and row_successes:
            property_success_flags.append(all(row_successes))

        detail = dict(row)
        detail.update(
            {
                "row_index": index,
                "image_path_exists": image_exists,
                present_detail_field: present,
                "valid": valid,
            }
        )
        for prop, value in properties.items():
            detail[f"actual_{prop}"] = value
        detailed_rows.append(detail)

    n = len(rows)
    success_rate = (
        sum(1 for flag in property_success_flags if flag) / len(property_success_flags)
        if property_success_flags
        else math.nan
    )

    summary: Dict[str, object] = {
        "benchmark_task": benchmark_task,
        "benchmark_label": "overall",
        "n": n,
        "image_path_exists_fraction": image_exists_count / n if n else math.nan,
        present_rate_field: present_count / n if n else math.nan,
        "molscribe_score_mean": mean(molscribe_scores) if molscribe_scores else math.nan,
        "validity": valid_count / n if n and rdkit_modules is not None else math.nan,
        "success_rate_in_valid_mols": success_rate,
        "success_rate_strict_in_valid_mols": success_rate,
        "success_rate_sketchmol_tolerance_in_valid_mols": success_rate,
        "rdkit_available": rdkit_modules is not None,
        "source_path": str(source_csv),
    }
    for prop, values in mae_values.items():
        if values:
            summary[f"{prop}_mae"] = mean(values)

    write_csv(detailed_rows, output_dir / "benchmark_decoded.csv")
    return summary


def render_report(summary: Mapping[str, object], metrics: Mapping[str, object]) -> str:
    present_label = metrics.get("present_label", "OCR SMILES present")
    present_field = metrics.get("present_rate_field", "ocr_smiles_present_rate")
    description = metrics.get(
        "report_description",
        "This artifact represents the real SketchMol pathway: diffusion image generation followed by MolScribe/OCR structure recovery.",
    )
    return "\n".join(
        [
            f"# {metrics.get('report_title', 'Real SketchMol + OCR Benchmark')}",
            "",
            f"Benchmark name: `{metrics.get('benchmark_name', '')}`",
            f"Source SketchMol repo: `{metrics.get('sketchmol_repo', '')}`",
            f"Source CSV: `{metrics.get('source_csv', '')}`",
            "",
            f"| n | image exists | {present_label} | validity | property success | MolScribe score |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {n} | {img} | {ocr} | {valid} | {succ} | {score} |".format(
                n=format_value(summary.get("n", "")),
                img=format_value(summary.get("image_path_exists_fraction", "")),
                ocr=format_value(summary.get(str(present_field), "")),
                valid=format_value(summary.get("validity", "")),
                succ=format_value(summary.get("success_rate_in_valid_mols", "")),
                score=format_value(summary.get("molscribe_score_mean", "")),
            ),
            "",
            str(description),
            "",
        ]
    )


def format_value(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) >= 100:
            return f"{numeric:.0f}"
        return f"{numeric:.4f}"
    return str(value)


def materialize(
    *,
    source_csv: Path,
    output_dir: Path,
    benchmark_name: str,
    sketchmol_repo: Path | None = None,
    run_log: Path | None = None,
    benchmark_kind: str = "real_sketchmol_plus_ocr",
    benchmark_task: str = "sketchmol_plus_ocr",
    smiles_column: str = "SMILES",
    image_column: str = "image_path",
    present_rate_field: str = "ocr_smiles_present_rate",
    present_detail_field: str = "ocr_smiles_present",
    source_copy_name: str = "source_image_path_ocr.csv",
    report_title: str = "Real SketchMol + OCR Benchmark",
    report_description: str = "This artifact represents the real SketchMol pathway: diffusion image generation followed by MolScribe/OCR structure recovery.",
    present_label: str = "OCR SMILES present",
) -> Dict[str, object]:
    if not source_csv.exists():
        raise FileNotFoundError(f"missing SketchMol OCR CSV: {source_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(source_csv)
    summary = summarise_rows(
        rows,
        output_dir,
        source_csv,
        smiles_column=smiles_column,
        image_column=image_column,
        benchmark_task=benchmark_task,
        present_rate_field=present_rate_field,
        present_detail_field=present_detail_field,
    )
    summary["benchmark_label"] = benchmark_name

    summary_csv = output_dir / "benchmark_summary.csv"
    write_csv([summary], summary_csv, preferred=SUMMARY_COLUMNS)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps({k: clean_value(v) for k, v in summary.items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied_source = output_dir / source_copy_name
    shutil.copy2(source_csv, copied_source)
    copied_log = None
    if run_log and run_log.exists():
        copied_log = output_dir / "source_run.log"
        shutil.copy2(run_log, copied_log)

    metrics = {
        "benchmark_kind": benchmark_kind,
        "benchmark_name": benchmark_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "sketchmol_repo": str(sketchmol_repo or ""),
        "source_csv": str(source_csv),
        "smiles_column": smiles_column,
        "image_column": image_column,
        "present_rate_field": present_rate_field,
        "present_label": present_label,
        "report_title": report_title,
        "report_description": report_description,
        "rows": len(rows),
        "summary": {k: clean_value(v) for k, v in summary.items()},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "benchmark_report.md").write_text(render_report(summary, metrics), encoding="utf-8")

    manifest = {
        "benchmark_kind": benchmark_kind,
        "benchmark_name": benchmark_name,
        "sketchmol_repo": str(sketchmol_repo or ""),
        "smiles_column": smiles_column,
        "image_column": image_column,
        "sources": {
            "source_csv": str(source_csv),
            "run_log": str(run_log) if run_log else "",
        },
        "copied": {
            "source_csv": str(copied_source),
            "run_log": str(copied_log) if copied_log else "",
        },
        "outputs": {
            "benchmark_summary": str(summary_csv),
            "benchmark_decoded": str(output_dir / "benchmark_decoded.csv"),
            "metrics": str(output_dir / "metrics.json"),
            "report": str(output_dir / "benchmark_report.md"),
        },
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path, help="SketchMol image_path.csv after MolScribe OCR, or a direct prediction CSV with compatible condition columns")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-name", default="real_sketchmol_plus_ocr")
    parser.add_argument("--sketchmol-repo", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--benchmark-kind", default="real_sketchmol_plus_ocr")
    parser.add_argument("--benchmark-task", default="sketchmol_plus_ocr")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--present-rate-field", default="ocr_smiles_present_rate")
    parser.add_argument("--present-detail-field", default="ocr_smiles_present")
    parser.add_argument("--source-copy-name", default="source_image_path_ocr.csv")
    parser.add_argument("--report-title", default="Real SketchMol + OCR Benchmark")
    parser.add_argument(
        "--report-description",
        default="This artifact represents the real SketchMol pathway: diffusion image generation followed by MolScribe/OCR structure recovery.",
    )
    parser.add_argument("--present-label", default="OCR SMILES present")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = materialize(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        benchmark_name=args.benchmark_name,
        sketchmol_repo=args.sketchmol_repo,
        run_log=args.run_log,
        benchmark_kind=args.benchmark_kind,
        benchmark_task=args.benchmark_task,
        smiles_column=args.smiles_column,
        image_column=args.image_column,
        present_rate_field=args.present_rate_field,
        present_detail_field=args.present_detail_field,
        source_copy_name=args.source_copy_name,
        report_title=args.report_title,
        report_description=args.report_description,
        present_label=args.present_label,
    )
    print(json.dumps({k: clean_value(v) for k, v in metrics.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
