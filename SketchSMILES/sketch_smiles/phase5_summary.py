"""Summarize SketchSMILES run metrics into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "run_name",
    "phase",
    "eval_pairs",
    "train_pairs",
    "model_type",
    "tokenization",
    "decoding",
    "beam_size",
    "rerank_mode",
    "randomized_smiles_per_molecule",
    "randomized_smiles_max_attempts",
    "top1_exact_match_fraction",
    "topk_exact_match_fraction",
    "top1_target_tanimoto",
    "mean_best_tanimoto",
    "top1_scaffold_match_fraction",
    "top1_valid_fraction",
    "paired_output_success_fraction",
    "image_exact_match_fraction",
    "fingerprint_bits",
    "fingerprint_loss_weight",
    "mean_predicted_target_fingerprint_tanimoto",
    "top1_condition_tanimoto",
    "mean_best_condition_tanimoto",
    "mean_candidate_count",
    "final_train_token_loss",
    "output_dir",
]


def summarize_runs(run_dirs: list[str | Path], output_dir: str | Path = "outputs/summary/phase5") -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        metrics_path = run_path / "metrics.json"
        if not metrics_path.exists():
            rows.append({"run_name": run_path.name, "output_dir": str(run_path), "missing_metrics": True})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {field: metrics.get(field, "") for field in SUMMARY_FIELDS}
        row["run_name"] = run_path.name
        row["output_dir"] = str(run_path)
        row["missing_metrics"] = False
        rows.append(row)

    csv_path = output_path / "phase5_summary.csv"
    json_path = output_path / "phase5_summary.json"
    fieldnames = SUMMARY_FIELDS + ["missing_metrics"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SketchSMILES Phase 5 run metrics.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-dir", default="outputs/summary/phase5")
    args = parser.parse_args()
    rows = summarize_runs(run_dirs=args.run_dirs, output_dir=args.output_dir)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
