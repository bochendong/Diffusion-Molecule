#!/usr/bin/env python3
"""Run ADMET-AI predictions and export SUCC external-oracle property CSV.

GeLLMO / GeLLMO-C official evaluation uses ADMET-AI for ADMET oracle properties
(BBBP, HIA, mutagenicity, hERG, DILI, PAMPA, carcinogenicity). See:
- https://github.com/ninglab/GeLLMO (process-output.ipynb -> generate_props)
- https://github.com/ninglab/GeLLMO-C

This script maps ADMET-AI column names to the lowercase property keys consumed by
`score_external_multiproperty_oracles.py` / `evaluate_external_multiproperty_predictions.py`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402


ADMET_AI_COLUMN_MAP = {
    "BBB_Martins": "bbbp",
    "bbb_martins": "bbbp",
    "HIA_Hou": "hia",
    "hia_hou": "hia",
    "AMES": "mutagenicity",
    "ames": "mutagenicity",
    "hERG": "erg",
    "herg": "erg",
    "DILI": "liver",
    "dili": "liver",
    "PAMPA_NCATS": "ampa",
    "pampa_ncats": "ampa",
    "Carcinogens_Lagunin": "carc",
    "carcinogens_lagunin": "carc",
}
DEFAULT_OUTPUT_PROPERTIES = (
    "ampa",
    "bbbp",
    "carc",
    "erg",
    "hia",
    "liver",
    "mutagenicity",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-smiles-csv", type=Path, required=True, help="CSV with a smiles column.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--properties",
        default=",".join(DEFAULT_OUTPUT_PROPERTIES),
        help="Comma-separated SUCC property keys to export.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-percentiles", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    smiles = collect_smiles(args.input_smiles_csv, args.smiles_column)
    if not smiles:
        raise ValueError(f"No SMILES found in {args.input_smiles_csv}")

    try:
        from admet_ai import ADMETModel
    except ImportError as exc:
        raise ImportError(
            "admet-ai is required for ADMET oracle prediction. "
            "Install with: pip install admet-ai"
        ) from exc

    requested = [item.strip().lower() for item in str(args.properties).split(",") if item.strip()]
    reverse_map = {canonical: raw for raw, canonical in ADMET_AI_COLUMN_MAP.items()}
    admet_columns = []
    for prop in requested:
        if prop in reverse_map:
            admet_columns.append(reverse_map[prop])
        elif prop.title() in ADMET_AI_COLUMN_MAP:
            admet_columns.append(prop.title())

    model = ADMETModel()
    rows: list[dict[str, str]] = []
    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        preds = model.predict(smiles=batch)
        if hasattr(preds, "iterrows"):
            iterator = ((idx, row) for idx, row in preds.iterrows())
        else:
            iterator = ((batch[0], preds),)

        for smi, row in iterator:
            canonical = canonical_or_blank(smi)
            if not canonical:
                continue
            out = {"smiles": canonical}
            source = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            for admet_name, succ_name in ADMET_AI_COLUMN_MAP.items():
                if succ_name not in requested:
                    continue
                value = source.get(admet_name)
                if value is None:
                    value = source.get(admet_name.lower())
                if value is None:
                    continue
                out[succ_name] = format_float(value)
                if args.include_percentiles:
                    percentile_key = f"{admet_name}_drugbank_approved_percentile"
                    percentile = source.get(percentile_key)
                    if percentile is not None:
                        out[f"{succ_name}_percentile"] = format_float(percentile)
            rows.append(out)

    write_rows(args.output_csv, rows, ["smiles", *requested])
    print(f"wrote {args.output_csv} unique_smiles={len(rows)} properties={','.join(requested)}")
    return 0


def collect_smiles(path: Path, smiles_column: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    columns = [smiles_column, "SMILES", "generated_smiles", "source_smiles", "target_smiles", "canonical_smiles"]
    for row in read_rows(path):
        for column in columns:
            smi = canonical_or_blank(row.get(column))
            if smi and smi not in seen:
                seen.add(smi)
                ordered.append(smi)
                break
    return ordered


def canonical_or_blank(value: object) -> str:
    try:
        return canonical_smiles(str(value or "").strip()) or ""
    except RuntimeError:
        return str(value or "").strip()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    text = f"{float(value):.8f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


if __name__ == "__main__":
    raise SystemExit(main())
