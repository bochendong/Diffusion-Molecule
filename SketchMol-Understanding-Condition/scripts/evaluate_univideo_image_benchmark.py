#!/usr/bin/env python3
"""Evaluate UniVideo generated molecule images after MolScribe OCR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import (  # noqa: E402
    canonical_smiles,
    molecular_properties,
    morgan_tanimoto,
    scaffold_smiles,
)


PROPERTY_COLUMNS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")
SKETCHMOL_STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}
SKETCHMOL_REFERENCE_MULTI_PROPERTY = {
    2: 0.804,
    3: 0.768,
    4: 0.736,
    5: 0.716,
    6: 0.678,
    7: 0.685,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", default="univideo_image_diffusion")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--source-tanimoto-thresholds", default="0.4,0.6,0.8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.image_csv)
    thresholds = _parse_float_list(args.source_tanimoto_thresholds)
    decoded_rows = [_decode_row(row, method=args.method, smiles_column=args.smiles_column) for row in rows]
    summary_rows = _summarize(decoded_rows, thresholds=thresholds)

    _write_rows(args.output_dir / "benchmark_decoded.csv", decoded_rows)
    _write_rows(args.output_dir / "benchmark_summary.csv", summary_rows)
    _write_report(args.output_dir / "benchmark_report.md", summary_rows, args, thresholds)
    payload = {
        "image_csv": str(args.image_csv),
        "output_dir": str(args.output_dir),
        "method": args.method,
        "rows": len(decoded_rows),
        "source_tanimoto_thresholds": thresholds,
        "decoded_csv": str(args.output_dir / "benchmark_decoded.csv"),
        "summary_csv": str(args.output_dir / "benchmark_summary.csv"),
        "report": str(args.output_dir / "benchmark_report.md"),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _decode_row(row: Mapping[str, str], *, method: str, smiles_column: str) -> dict[str, object]:
    raw_smiles = row.get(smiles_column) or row.get("generated_smiles") or ""
    generated = _safe_canonical(raw_smiles) or ""
    valid = bool(generated)
    selected_props = _selected_props(row)
    generated_props = _normalized_properties(generated) if valid else {}
    source_smiles = row.get("source_smiles", "")
    target_smiles = row.get("target_smiles", "")
    source_scaffold = row.get("source_scaffold", "") or _safe_scaffold(source_smiles)
    generated_scaffold = _safe_scaffold(generated) if valid else ""
    target_scaffold = row.get("target_scaffold", "") or _safe_scaffold(target_smiles)

    property_successes = []
    errors: dict[str, float] = {}
    for prop in selected_props:
        target = _to_float(row.get(f"target_{prop}"))
        actual = _to_float(generated_props.get(prop))
        error = abs(actual - target) if not math.isnan(actual) and not math.isnan(target) else math.nan
        errors[prop] = error
        property_successes.append((not math.isnan(error)) and error <= SKETCHMOL_STRICT_TOLERANCE[prop])
    strict_success = valid and bool(property_successes) and all(property_successes)
    scaffold_match_source = valid and bool(source_scaffold and generated_scaffold and source_scaffold == generated_scaffold)
    scaffold_match_target = valid and bool(target_scaffold and generated_scaffold and target_scaffold == generated_scaffold)
    source_tanimoto = _safe_tanimoto(source_smiles, generated) if valid else math.nan
    target_tanimoto = _safe_tanimoto(target_smiles, generated) if valid else math.nan
    exact_target_match = valid and bool((_safe_canonical(target_smiles) or target_smiles) == generated)
    source_identity = valid and bool((_safe_canonical(source_smiles) or source_smiles) == generated)

    out: dict[str, object] = dict(row)
    out.update(
        {
            "method": row.get("method", "") or method,
            "ocr_smiles": raw_smiles,
            "generated_smiles": generated,
            "valid": valid,
            "property_count": _property_count(row, selected_props),
            "condition_properties": ",".join(selected_props),
            "source_scaffold": source_scaffold,
            "target_scaffold": target_scaffold,
            "generated_scaffold": generated_scaffold,
            "source_scaffold_match": scaffold_match_source,
            "target_scaffold_match": scaffold_match_target,
            "strict_success": strict_success,
            "joint_success_source_scaffold": strict_success and scaffold_match_source,
            "joint_success_target_scaffold": strict_success and scaffold_match_target,
            "source_tanimoto": "" if math.isnan(source_tanimoto) else source_tanimoto,
            "target_tanimoto": "" if math.isnan(target_tanimoto) else target_tanimoto,
            "exact_target_match": exact_target_match,
            "source_identity": source_identity,
            "image_path_exists": bool(row.get("image_path") and Path(row.get("image_path", "")).exists()),
            "ocr_smiles_present": valid,
        }
    )
    for prop in PROPERTY_COLUMNS:
        actual = generated_props.get(prop, math.nan)
        if prop in selected_props:
            out[f"{prop}_actual"] = "" if math.isnan(_to_float(actual)) else actual
            out[f"{prop}_abs_error"] = "" if math.isnan(errors.get(prop, math.nan)) else errors[prop]
            out[f"{prop}_success"] = bool(errors.get(prop, math.nan) <= SKETCHMOL_STRICT_TOLERANCE[prop])
        else:
            out[f"{prop}_actual"] = ""
            out[f"{prop}_abs_error"] = ""
            out[f"{prop}_success"] = ""
    return out


def _summarize(rows: list[dict[str, object]], *, thresholds: list[float]) -> list[dict[str, object]]:
    out = []
    methods = sorted({str(row.get("method", "")) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if str(row.get("method", "")) == method]
        for property_count in range(2, 8):
            selected = [row for row in method_rows if int(row.get("property_count", 0) or 0) == property_count]
            if selected:
                out.append(_summary_row(method, selected, property_count, thresholds=thresholds))
        if method_rows:
            out.append(_summary_row(method, method_rows, "all", thresholds=thresholds))
    return out


def _summary_row(method: str, rows: list[dict[str, object]], property_count: int | str, *, thresholds: list[float]) -> dict[str, object]:
    valid_rows = [row for row in rows if _to_bool(row.get("valid"))]
    strict_count = sum(1 for row in rows if _to_bool(row.get("strict_success")))
    valid_strict_count = sum(1 for row in valid_rows if _to_bool(row.get("strict_success")))
    generated = [str(row.get("generated_smiles", "")) for row in valid_rows if row.get("generated_smiles")]
    unique_generated = sorted(set(generated))
    source_tanimotos = [_to_float(row.get("source_tanimoto")) for row in rows if row.get("source_tanimoto") != ""]
    source_tanimotos = [value for value in source_tanimotos if not math.isnan(value)]
    target_tanimotos = [_to_float(row.get("target_tanimoto")) for row in rows if row.get("target_tanimoto") != ""]
    target_tanimotos = [value for value in target_tanimotos if not math.isnan(value)]
    reference = SKETCHMOL_REFERENCE_MULTI_PROPERTY.get(property_count, "") if isinstance(property_count, int) else ""
    decode_sources = Counter(str(row.get("molscribe_decode_source", "") or "unknown") for row in rows)
    summary: dict[str, object] = {
        "family": "univideo_image_to_structure",
        "method": method,
        "benchmark_task": "multi_property_image_to_structure",
        "benchmark_label": f"{property_count}_properties" if isinstance(property_count, int) else "all",
        "property_count": property_count,
        "n": len(rows),
        "image_path_exists_fraction": _fraction(_to_bool(row.get("image_path_exists")) for row in rows),
        "ocr_smiles_present_rate": _fraction(_to_bool(row.get("ocr_smiles_present")) for row in rows),
        "validity": len(valid_rows) / len(rows) if rows else 0.0,
        "strict_success_rate": strict_count / len(rows) if rows else 0.0,
        "success_rate_strict_in_valid_mols": valid_strict_count / len(valid_rows) if valid_rows else 0.0,
        "source_scaffold_match_rate": _fraction(_to_bool(row.get("source_scaffold_match")) for row in rows),
        "target_scaffold_match_rate": _fraction(_to_bool(row.get("target_scaffold_match")) for row in rows),
        "joint_success_source_scaffold_rate": _fraction(
            _to_bool(row.get("joint_success_source_scaffold")) for row in rows
        ),
        "mean_source_tanimoto": _mean(source_tanimotos),
        "median_source_tanimoto": _median(source_tanimotos),
        "mean_target_tanimoto": _mean(target_tanimotos),
        "unique_valid_smiles": len(unique_generated),
        "uniqueness_in_valid_mols": len(unique_generated) / len(generated) if generated else "",
        "exact_target_match_rate": _fraction(_to_bool(row.get("exact_target_match")) for row in rows),
        "source_identity_rate": _fraction(_to_bool(row.get("source_identity")) for row in rows),
        "molscribe_decode_source_counts": ";".join(
            f"{key}:{value}" for key, value in sorted(decode_sources.items())
        ),
        "sketchmol_reference_strict": reference,
        "strict_margin_vs_sketchmol": (
            (strict_count / len(rows)) - float(reference)
            if rows and isinstance(reference, float)
            else ""
        ),
    }
    for threshold in thresholds:
        suffix = _threshold_suffix(threshold)
        summary[f"source_tanimoto_ge_{suffix}_rate"] = (
            _fraction(value >= threshold for value in source_tanimotos) if source_tanimotos else ""
        )
        summary[f"strict_success_at_source_tanimoto_ge_{suffix}"] = (
            _fraction(
                _to_bool(row.get("strict_success")) and _to_float(row.get("source_tanimoto")) >= threshold
                for row in rows
            )
            if rows
            else ""
        )
    for prop in PROPERTY_COLUMNS:
        errors = [_to_float(row.get(f"{prop}_abs_error")) for row in rows if row.get(f"{prop}_abs_error") != ""]
        errors = [value for value in errors if not math.isnan(value)]
        if errors:
            summary[f"{prop}_mae"] = _mean(errors)
    return summary


def _write_report(path: Path, summary_rows: list[dict[str, object]], args: argparse.Namespace, thresholds: list[float]) -> None:
    rows_by_key = {(row["method"], row["property_count"]): row for row in summary_rows}
    methods = sorted({str(row.get("method", "")) for row in summary_rows})
    lines = [
        "# UniVideo Image-to-Structure Benchmark",
        "",
        "This report evaluates generated molecule images after MolScribe OCR and RDKit validation.",
        "",
        f"- image CSV: `{args.image_csv}`",
        f"- method label: `{args.method}`",
        f"- source Tanimoto thresholds: `{','.join(str(value) for value in thresholds)}`",
        "",
        "## Main 2p-7p Table",
        "",
        "`strict` is computed over all generated rows; invalid/OCR-failed molecules count as failures.",
        "",
        "| method | validity all | 2p strict | 3p strict | 4p strict | 5p strict | 6p strict | 7p strict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        all_row = rows_by_key.get((method, "all"))
        values = [_fmt(all_row.get("validity")) if all_row else ""]
        for count in range(2, 8):
            row = rows_by_key.get((method, count))
            values.append(_fmt(row.get("strict_success_rate")) if row else "")
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines.append("| SketchMol structured reference |  | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |")
    lines.extend(
        [
            "",
            "## Source-Similarity-Constrained Success",
            "",
            "`strict@Tanimoto>=t` means strict property success and Morgan Tanimoto(source, generated) >= t.",
            "",
            "| method | mean source Tani | median source Tani | "
            + " | ".join(f"strict@{threshold:g}" for threshold in thresholds)
            + " |",
            "| --- | ---: | ---: | " + " | ".join("---:" for _ in thresholds) + " |",
        ]
    )
    for method in methods:
        all_row = rows_by_key.get((method, "all"))
        if not all_row:
            continue
        values = [_fmt(all_row.get("mean_source_tanimoto")), _fmt(all_row.get("median_source_tanimoto"))]
        for threshold in thresholds:
            values.append(_fmt(all_row.get(f"strict_success_at_source_tanimoto_ge_{_threshold_suffix(threshold)}")))
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| method | OCR present | valid | source scaffold | unique valid | exact target | source identity | decode source |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for method in methods:
        all_row = rows_by_key.get((method, "all"))
        if not all_row:
            continue
        lines.append(
            f"| {method} | "
            f"{_fmt(all_row.get('ocr_smiles_present_rate'))} | "
            f"{_fmt(all_row.get('validity'))} | "
            f"{_fmt(all_row.get('source_scaffold_match_rate'))} | "
            f"{_fmt(all_row.get('uniqueness_in_valid_mols'))} | "
            f"{_fmt(all_row.get('exact_target_match_rate'))} | "
            f"{_fmt(all_row.get('source_identity_rate'))} | "
            f"{all_row.get('molscribe_decode_source_counts', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_props(row: Mapping[str, object]) -> list[str]:
    selected = [item.strip() for item in str(row.get("condition_properties", "")).split(",") if item.strip()]
    selected = [prop for prop in selected if prop in PROPERTY_COLUMNS]
    if selected:
        return selected
    return [prop for prop in PROPERTY_COLUMNS if _to_bool(row.get(f"{prop}_active"))]


def _property_count(row: Mapping[str, object], selected_props: list[str]) -> int:
    value = _to_float(row.get("property_count"))
    return int(value) if not math.isnan(value) and value > 0 else len(selected_props)


def _normalized_properties(smiles: str) -> dict[str, float]:
    try:
        props = molecular_properties(smiles)
    except RuntimeError as exc:
        raise RuntimeError("RDKit is required for image-to-structure benchmark evaluation") from exc
    if props is None:
        return {}
    return {
        "MW": float(props.get("MolWt", math.nan)),
        "LogP": float(props.get("LogP", math.nan)),
        "QED": float(props.get("QED", math.nan)),
        "TPSA": float(props.get("TPSA", math.nan)),
        "HBD": float(props.get("HBD", math.nan)),
        "HBA": float(props.get("HBA", math.nan)),
        "RB": float(props.get("rotatable", math.nan)),
    }


def _safe_canonical(smiles: object) -> str | None:
    text = str(smiles or "").strip()
    if not text or text.lower() in {"none", "nan", "invalid"}:
        return None
    try:
        return canonical_smiles(text)
    except RuntimeError as exc:
        raise RuntimeError("RDKit is required for image-to-structure benchmark evaluation") from exc


def _safe_scaffold(smiles: str) -> str:
    try:
        return scaffold_smiles(smiles) or ""
    except RuntimeError:
        return ""


def _safe_tanimoto(left: str, right: str) -> float:
    try:
        value = morgan_tanimoto(left, right)
    except RuntimeError as exc:
        raise RuntimeError("RDKit is required for image-to-structure benchmark evaluation") from exc
    return float(value) if value is not None else math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def _parse_float_list(text: str) -> list[float]:
    out = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def _threshold_suffix(value: float) -> str:
    return str(value).replace(".", "_")


def _to_float(value: object) -> float:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return math.nan


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value if value is not None else "").strip().lower() in {"1", "true", "yes", "y"}


def _fraction(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for item in items if item) / len(items) if items else 0.0


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _median(values: list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.3f}"


if __name__ == "__main__":
    main()
