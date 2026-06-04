"""Collect SketchSMILES and real SketchMol+OCR metrics into one report.

The module is intentionally stdlib-only so it can run on a login node without
extra Python packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


SKETCHSMILES_FIELDS = [
    "eval_pairs",
    "train_pairs",
    "top1_exact_match_fraction",
    "topk_exact_match_fraction",
    "top1_target_tanimoto",
    "mean_best_tanimoto",
    "top1_scaffold_match_fraction",
    "top1_valid_fraction",
    "paired_output_success_fraction",
    "image_exact_match_fraction",
    "image_mse_mean",
    "fingerprint_bits",
    "fingerprint_loss_weight",
    "mean_predicted_target_fingerprint_tanimoto",
    "top1_condition_tanimoto",
    "mean_best_condition_tanimoto",
    "generated_image_fraction",
    "mean_candidate_count",
    "tokenization",
    "latent_dim",
    "clip_loss_weight",
    "final_train_clip_loss",
    "decode_length_mode",
    "train_decode_length",
    "diffusion_steps",
    "final_train_loss",
    "final_train_image_loss",
    "image_loss_weight",
    "smiles_render_image_compared_fraction",
    "smiles_render_image_exact_match_fraction",
    "smiles_render_image_mse_mean",
    "randomized_smiles_per_molecule",
    "randomized_smiles_max_attempts",
]

SKETCHMOL_WEIGHTED_FIELDS = [
    "image_path_exists_fraction",
    "ocr_smiles_present_rate",
    "predicted_smiles_present_rate",
    "molscribe_score_mean",
    "success_rate_in_valid_mols",
    "success_rate_strict_in_valid_mols",
    "success_rate_sketchmol_tolerance_in_valid_mols",
    "validity",
    "uniqueness",
    "novelty",
    "druglike_rate",
    "mean_pairwise_tanimoto",
    "LogP_mae",
    "QED_mae",
    "MW_mae",
    "TPSA_mae",
    "HBD_mae",
    "HBA_mae",
    "RB_mae",
]

PREFERRED_COLUMNS = [
    "family",
    "run_name",
    "phase",
    "benchmark_task",
    "benchmark_label",
    "source_path",
    "n",
    "eval_pairs",
    "train_pairs",
    "top1_exact_match_fraction",
    "topk_exact_match_fraction",
    "top1_target_tanimoto",
    "mean_best_tanimoto",
    "top1_scaffold_match_fraction",
    "top1_valid_fraction",
    "paired_output_success_fraction",
    "image_exact_match_fraction",
    "image_path_exists_fraction",
    "ocr_smiles_present_rate",
    "predicted_smiles_present_rate",
    "molscribe_score_mean",
    "mean_predicted_target_fingerprint_tanimoto",
    "top1_condition_tanimoto",
    "mean_best_condition_tanimoto",
    "tokenization",
    "latent_dim",
    "clip_loss_weight",
    "final_train_clip_loss",
    "decode_length_mode",
    "train_decode_length",
    "diffusion_steps",
    "final_train_loss",
    "final_train_image_loss",
    "image_loss_weight",
    "smiles_render_image_compared_fraction",
    "smiles_render_image_exact_match_fraction",
    "smiles_render_image_mse_mean",
    "randomized_smiles_per_molecule",
    "randomized_smiles_max_attempts",
    "success_rate_in_valid_mols",
    "success_rate_strict_in_valid_mols",
    "success_rate_sketchmol_tolerance_in_valid_mols",
    "validity",
    "uniqueness",
    "novelty",
    "druglike_rate",
    "mean_pairwise_tanimoto",
]


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
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


def _clean_value(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _derive_run_name(path: Path) -> str:
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    if path.name == "metrics.json":
        return path.parent.name
    return path.stem


def _derive_sketchmol_run_name(path: Path) -> str:
    manifest_path = path.parent / "source_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            benchmark_name = manifest.get("benchmark_name")
            if benchmark_name:
                return str(benchmark_name)
        except (OSError, json.JSONDecodeError):
            pass
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def _derive_sketchmol_family(path: Path) -> str:
    manifest_path = path.parent / "source_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
            benchmark_kind = str(manifest.get("benchmark_kind", "")).strip()
            if benchmark_kind == "real_sketchmol_plus_ocr":
                return "real_sketchmol_plus_ocr"
            if benchmark_kind:
                return benchmark_kind
        except (OSError, json.JSONDecodeError):
            pass
    return "sketchmol_aligned"


def _weighted_mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = _to_float(row.get("n"))
        value = _to_float(row.get(field))
        if math.isnan(weight) or weight <= 0 or math.isnan(value):
            continue
        numerator += weight * value
        denominator += weight
    if denominator == 0:
        return math.nan
    return numerator / denominator


def _sum_n(rows: Sequence[Mapping[str, str]]) -> float:
    total = 0.0
    for row in rows:
        value = _to_float(row.get("n"))
        if not math.isnan(value):
            total += value
    return total


def collect_sketchsmiles_run(run_dir: Path) -> Dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing SketchSMILES metrics: {metrics_path}")
    metrics = _read_json(metrics_path)
    phase = str(metrics.get("phase", ""))
    row: Dict[str, object] = {
        "family": _derive_metrics_family(phase),
        "run_name": run_dir.name,
        "phase": phase,
        "benchmark_task": _derive_metrics_task(phase),
        "benchmark_label": metrics.get("model_type", "") or metrics.get("tokenization", ""),
        "source_path": str(metrics_path),
    }
    for field in SKETCHSMILES_FIELDS:
        if field in metrics:
            row[field] = metrics[field]
    for field in [
        "tokenization",
        "decoding",
        "beam_size",
        "rerank_mode",
        "model_type",
        "image_size",
        "samples_per_condition",
        "randomized_smiles_per_molecule",
    ]:
        if field in metrics:
            row[field] = metrics[field]
    return row


def _derive_metrics_family(phase: str) -> str:
    if phase.startswith("sketchmol_token_diffusion"):
        return "token_diffusion_ocr_free"
    if phase.startswith("sketchmol_joint_diffusion"):
        return "joint_diffusion_ocr_free"
    return "sketchsmiles_ocr_free"


def _derive_metrics_task(phase: str) -> str:
    if phase.startswith("sketchmol_token_diffusion"):
        return "condition_to_smiles_token_diffusion"
    if phase.startswith("sketchmol_joint_diffusion"):
        return "condition_to_image_and_smiles_diffusion"
    return "image_or_latent_to_smiles"


def _aggregate_sketchmol_rows(
    rows: Sequence[Mapping[str, str]],
    path: Path,
    benchmark_task: str,
    benchmark_label: str = "",
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "family": _derive_sketchmol_family(path),
        "run_name": _derive_sketchmol_run_name(path),
        "phase": "sketchmol_benchmark",
        "benchmark_task": benchmark_task,
        "benchmark_label": benchmark_label,
        "source_path": str(path),
        "n": _sum_n(rows),
    }
    for field in SKETCHMOL_WEIGHTED_FIELDS:
        value = _weighted_mean(rows, field)
        if not math.isnan(value):
            out[field] = value
    return out


def collect_sketchmol_summary(summary_csv: Path) -> List[Dict[str, object]]:
    if not summary_csv.exists():
        raise FileNotFoundError(f"missing SketchMol summary CSV: {summary_csv}")
    rows = _read_csv(summary_csv)
    grouped: MutableMapping[str, List[Mapping[str, str]]] = {}
    grouped_by_label: MutableMapping[str, List[Mapping[str, str]]] = {}
    for row in rows:
        task = row.get("benchmark_task", "") or "unknown"
        label = row.get("benchmark_label", "") or ""
        grouped.setdefault(task, []).append(row)
        grouped_by_label.setdefault(f"{task}:{label}", []).append(row)

    out: List[Dict[str, object]] = [
        _aggregate_sketchmol_rows(rows, summary_csv, "overall", "all")
    ]
    for task in sorted(grouped):
        out.append(_aggregate_sketchmol_rows(grouped[task], summary_csv, task, "all"))
    for key in sorted(grouped_by_label):
        task, label = key.split(":", 1)
        if not label:
            continue
        out.append(_aggregate_sketchmol_rows(grouped_by_label[key], summary_csv, task, label))
    return out


def collect_rows(
    sketchsmiles_runs: Iterable[Path],
    sketchmol_summaries: Iterable[Path],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in sketchsmiles_runs:
        rows.append(collect_sketchsmiles_run(run_dir))
    for summary_csv in sketchmol_summaries:
        rows.extend(collect_sketchmol_summary(summary_csv))
    return rows


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    seen = set()
    fields: List[str] = []
    for field in PREFERRED_COLUMNS:
        if any(field in row for row in rows):
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def write_outputs(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    csv_path = output_dir / "comparison_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _clean_value(row.get(field, "")) for field in fieldnames})

    json_path = output_dir / "comparison_rows.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([{k: _clean_value(v) for k, v in row.items()} for row in rows], handle, indent=2, sort_keys=True)
        handle.write("\n")

    report_path = output_dir / "comparison_report.md"
    report_path.write_text(render_report(rows), encoding="utf-8")


def _fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        if abs(float(value)) >= 100:
            return f"{float(value):.0f}"
        return f"{float(value):.4f}"
    return str(value)


def render_report(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# SketchMolCompare Report",
        "",
        "This report is generated from finished run artifacts. It does not rerun training or evaluation.",
        "",
        "| family | run | task | label | n/eval | top1 exact | topk exact | top1 tanimoto | mean best tanimoto | sketchmol success | SMILES present | validity | MolScribe score |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        n_or_eval = row.get("eval_pairs", row.get("n", ""))
        lines.append(
            "| {family} | {run} | {task} | {label} | {n_eval} | {top1_exact} | {topk_exact} | {top1_tani} | {best_tani} | {success} | {ocr} | {validity} | {molscribe} |".format(
                family=_fmt(row.get("family", "")),
                run=_fmt(row.get("run_name", "")),
                task=_fmt(row.get("benchmark_task", "")),
                label=_fmt(row.get("benchmark_label", "")),
                n_eval=_fmt(n_or_eval),
                top1_exact=_fmt(row.get("top1_exact_match_fraction", "")),
                topk_exact=_fmt(row.get("topk_exact_match_fraction", "")),
                top1_tani=_fmt(row.get("top1_target_tanimoto", "")),
                best_tani=_fmt(row.get("mean_best_tanimoto", "")),
                success=_fmt(row.get("success_rate_in_valid_mols", "")),
                ocr=_fmt(row.get("ocr_smiles_present_rate", row.get("predicted_smiles_present_rate", ""))),
                validity=_fmt(row.get("top1_valid_fraction", row.get("validity", ""))),
                molscribe=_fmt(row.get("molscribe_score_mean", "")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sketchsmiles-run", action="append", default=[], type=Path)
    parser.add_argument("--sketchmol-summary", action="append", default=[], type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect_rows(args.sketchsmiles_run, args.sketchmol_summary)
    if not rows:
        raise SystemExit("no input rows collected; pass at least one existing run or summary")
    write_outputs(rows, args.output_dir)
    print(json.dumps([{k: _clean_value(v) for k, v in row.items()} for row in rows], indent=2, sort_keys=True))
    print()
    print(f"wrote={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
