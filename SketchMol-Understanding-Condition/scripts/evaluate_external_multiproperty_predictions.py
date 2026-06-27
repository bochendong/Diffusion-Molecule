#!/usr/bin/env python3
"""Evaluate source-conditioned external multi-property predictions.

The evaluator supports two modes:

1. Local/proxy scoring for properties SUCC can compute directly (QED, LogP,
   SA when RDKit's SA scorer is installed, and a simple pLogP proxy).
2. External-property scoring from a generated properties CSV, keyed by SMILES,
   for ADMET/oracle properties such as BBBP, HIA, AMES/mutagenicity, hERG, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties, morgan_tanimoto  # noqa: E402


DEFAULT_DIRECTION = {
    "ampa": "increase",
    "bbbp": "increase",
    "drd2": "increase",
    "hia": "increase",
    "plogp": "increase",
    "qed": "increase",
    "carc": "decrease",
    "erg": "decrease",
    "liver": "decrease",
    "mutagenicity": "decrease",
}
DEFAULT_THRESHOLDS = {
    "ampa": 0.1,
    "bbbp": 0.1,
    "carc": 0.1,
    "drd2": 0.2,
    "erg": 0.2,
    "hia": 0.1,
    "liver": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-properties-csv", type=Path, default=None)
    parser.add_argument("--source-properties-csv", type=Path, default=None)
    parser.add_argument("--smiles-column", default="generated_smiles")
    parser.add_argument("--source-smiles-column", default="source_smiles")
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--report-title", default="External Multi-property Benchmark")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_rows(args.prediction_csv)
    generated_props = read_property_lookup(args.generated_properties_csv)
    source_props_lookup = read_property_lookup(args.source_properties_csv)
    detail_rows = [
        evaluate_row(
            row,
            generated_props=generated_props,
            source_props_lookup=source_props_lookup,
            smiles_column=args.smiles_column,
            source_smiles_column=args.source_smiles_column,
            min_source_tanimoto=float(args.min_source_tanimoto),
        )
        for row in predictions
    ]
    summary_rows = summarize(detail_rows)
    write_rows(args.output_dir / "external_multiproperty_detail.csv", detail_rows)
    write_rows(args.output_dir / "external_multiproperty_summary.csv", summary_rows)
    report = render_report(args.report_title, args, summary_rows, detail_rows)
    (args.output_dir / "external_multiproperty_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def evaluate_row(
    row: Mapping[str, str],
    *,
    generated_props: Mapping[str, Mapping[str, float]],
    source_props_lookup: Mapping[str, Mapping[str, float]],
    smiles_column: str,
    source_smiles_column: str,
    min_source_tanimoto: float,
) -> dict[str, object]:
    generated = str(row.get(smiles_column, "") or "").strip()
    source = str(row.get(source_smiles_column, "") or "").strip()
    canonical = safe_canonical(generated)
    valid = bool(canonical)
    tanimoto = safe_tanimoto(source, canonical) if valid and source else math.nan
    task_props = parse_list(row.get("external_task_properties") or row.get("condition_properties"))
    directions = parse_json_dict(row.get("external_property_directions_json"), DEFAULT_DIRECTION)
    thresholds = parse_json_dict(row.get("external_property_thresholds_json"), DEFAULT_THRESHOLDS)
    source_scores = merged_properties(source, row, source_props_lookup)
    generated_scores = generated_properties(canonical, generated_props) if canonical else {}
    per_prop_success: dict[str, bool | None] = {}
    missing_props = []
    for prop in task_props:
        prop = canonical_prop(prop)
        source_value = source_scores.get(prop)
        generated_value = generated_scores.get(prop)
        if source_value is None:
            source_value = parse_float(row.get(f"external_source_{prop}"))
        target_value = parse_float(row.get(f"external_target_{prop}"))
        if generated_value is None or (source_value is None and target_value is None):
            per_prop_success[prop] = None
            missing_props.append(prop)
            continue
        direction = str(directions.get(prop, DEFAULT_DIRECTION.get(prop, "increase"))).lower()
        threshold = float(thresholds.get(prop, DEFAULT_THRESHOLDS.get(prop, 0.0)))
        if target_value is not None:
            success = generated_value >= target_value if direction == "increase" else generated_value <= target_value
        else:
            delta = generated_value - float(source_value)
            success = delta >= threshold if direction == "increase" else -delta >= threshold
        per_prop_success[prop] = bool(success)
    evaluated = [value for value in per_prop_success.values() if value is not None]
    all_property_success = bool(evaluated) and all(evaluated) and not missing_props
    source_similarity_success = bool(valid) and (math.isnan(tanimoto) or tanimoto >= min_source_tanimoto)
    strict_success = bool(valid) and source_similarity_success and all_property_success
    return {
        **dict(row),
        "external_generated_canonical_smiles": canonical,
        "external_valid": "True" if valid else "False",
        "external_source_tanimoto": "" if math.isnan(tanimoto) else format_float(tanimoto, digits=6),
        "external_source_similarity_success": "True" if source_similarity_success else "False",
        "external_property_success_json": json.dumps(per_prop_success, sort_keys=True),
        "external_missing_generated_oracle_properties": ",".join(missing_props),
        "external_evaluated_property_fraction": format_float(len(evaluated) / max(len(task_props), 1), digits=6),
        "external_all_property_success": "True" if all_property_success else "False",
        "external_strict_success": "True" if strict_success else "False",
    }


def merged_properties(
    source_smiles: str,
    row: Mapping[str, str],
    source_props_lookup: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    props = {}
    canonical = safe_canonical(source_smiles)
    if canonical and canonical in source_props_lookup:
        props.update(source_props_lookup[canonical])
    for key, value in row.items():
        if key.startswith("external_source_"):
            prop = canonical_prop(key.removeprefix("external_source_"))
            parsed = parse_float(value)
            if parsed is not None:
                props[prop] = parsed
    if canonical:
        props.update({key: val for key, val in local_properties(canonical).items() if key not in props})
    return props


def generated_properties(smiles: str, generated_props: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    props = {}
    if smiles in generated_props:
        props.update(generated_props[smiles])
    props.update({key: val for key, val in local_properties(smiles).items() if key not in props})
    return props


def local_properties(smiles: str) -> dict[str, float]:
    try:
        props = molecular_properties(smiles) or {}
    except RuntimeError:
        return {}
    out = {}
    if "QED" in props:
        out["qed"] = float(props["QED"])
    if "LogP" in props:
        out["logp"] = float(props["LogP"])
    if "LogP" in props:
        sa = float(props.get("SA", 0.0))
        out["plogp"] = float(props["LogP"]) - sa
    if "SA" in props:
        out["sas"] = float(props["SA"])
    return out


def read_property_lookup(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    rows = read_rows(path)
    lookup: dict[str, dict[str, float]] = {}
    for row in rows:
        smiles = str(row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles") or "").strip()
        canonical = safe_canonical(smiles)
        if not canonical:
            continue
        props: dict[str, float] = {}
        for key, value in row.items():
            prop = canonical_prop(key)
            if prop not in DEFAULT_DIRECTION and prop not in {"logp", "sas"}:
                continue
            parsed = parse_float(value)
            if parsed is not None:
                props[prop] = parsed
        lookup[canonical] = props
    return lookup


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("external_suite") or "unknown"),
                str(row.get("external_task_split") or "unknown"),
                str(row.get("external_task_id") or str(row.get("external_task_key") or "unknown")),
            )
        ].append(row)
    out = []
    for (suite, split, task_id), items in sorted(groups.items()):
        n = len(items)
        valid = mean_bool(items, "external_valid")
        sim = mean_bool(items, "external_source_similarity_success")
        prop = mean_bool(items, "external_all_property_success")
        strict = mean_bool(items, "external_strict_success")
        missing_counter = Counter()
        evaluated_fraction = []
        for item in items:
            missing_counter.update(parse_list(item.get("external_missing_generated_oracle_properties")))
            value = parse_float(item.get("external_evaluated_property_fraction"))
            if value is not None:
                evaluated_fraction.append(value)
        out.append(
            {
                "external_suite": suite,
                "external_task_split": split,
                "external_task_id": task_id,
                "rows": n,
                "validity": format_float(valid, digits=6),
                "source_similarity_success_rate": format_float(sim, digits=6),
                "all_property_success_rate": format_float(prop, digits=6),
                "strict_success_rate": format_float(strict, digits=6),
                "mean_evaluated_property_fraction": format_float(
                    sum(evaluated_fraction) / max(len(evaluated_fraction), 1),
                    digits=6,
                ),
                "missing_oracle_properties": ",".join(sorted(missing_counter)),
            }
        )
    if rows:
        out.append(
            {
                "external_suite": "all",
                "external_task_split": "all",
                "external_task_id": "all",
                "rows": len(rows),
                "validity": format_float(mean_bool(rows, "external_valid"), digits=6),
                "source_similarity_success_rate": format_float(
                    mean_bool(rows, "external_source_similarity_success"),
                    digits=6,
                ),
                "all_property_success_rate": format_float(mean_bool(rows, "external_all_property_success"), digits=6),
                "strict_success_rate": format_float(mean_bool(rows, "external_strict_success"), digits=6),
                "mean_evaluated_property_fraction": format_float(
                    sum(parse_float(row.get("external_evaluated_property_fraction")) or 0.0 for row in rows)
                    / max(len(rows), 1),
                    digits=6,
                ),
                "missing_oracle_properties": ",".join(
                    sorted({prop for row in rows for prop in parse_list(row.get("external_missing_generated_oracle_properties"))})
                ),
            }
        )
    return out


def render_report(
    title: str,
    args: argparse.Namespace,
    summary_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- prediction_csv: `{args.prediction_csv}`",
        f"- generated_properties_csv: `{args.generated_properties_csv or 'none'}`",
        f"- min_source_tanimoto: `{args.min_source_tanimoto}`",
        f"- rows: `{len(detail_rows)}`",
        "",
        "| Suite | Split | Task | Rows | Valid | Sim | Prop all | Strict | Eval prop frac | Missing oracle props |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {external_suite} | {external_task_split} | {external_task_id} | {rows} | "
            "{validity} | {source_similarity_success_rate} | {all_property_success_rate} | "
            "{strict_success_rate} | {mean_evaluated_property_fraction} | {missing_oracle_properties} |".format(**row)
        )
    lines.append("")
    lines.append("Rows with missing oracle properties are diagnostic-only until a generated-properties CSV is supplied.")
    lines.append("")
    return "\n".join(lines)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def safe_canonical(smiles: str) -> str:
    try:
        return canonical_smiles(smiles) or ""
    except RuntimeError:
        return str(smiles or "").strip()


def safe_tanimoto(left: str, right: str) -> float:
    try:
        value = morgan_tanimoto(left, right)
    except RuntimeError:
        return math.nan
    return math.nan if value is None else float(value)


def parse_json_dict(value: object, fallback: Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {canonical_prop(key): val for key, val in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return dict(fallback)
        if isinstance(parsed, Mapping):
            return {canonical_prop(key): val for key, val in parsed.items()}
    return dict(fallback)


def parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [canonical_prop(item) for item in value if str(item).strip()]
    return [canonical_prop(item) for item in str(value or "").replace("|", ",").split(",") if item.strip()]


def canonical_prop(value: object) -> str:
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return PROPERTY_ALIASES.get(key, key)


def parse_float(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def mean_bool(rows: list[dict[str, object]], key: str) -> float:
    return sum(1 for row in rows if truthy(row.get(key))) / max(len(rows), 1)


def format_float(value: float, *, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return ""
    text = f"{float(value):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


if __name__ == "__main__":
    raise SystemExit(main())
