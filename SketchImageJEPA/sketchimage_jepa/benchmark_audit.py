"""Audit whether a benchmark split can be solved by train-set shortcuts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .chem import morgan_fingerprint_bits, scaffold_key, tanimoto
from .dataset import load_examples_csv
from .schema import BenchmarkExample


THRESHOLDS = (0.50, 0.60, 0.70, 0.80)


def audit_split(
    train_examples: Iterable[BenchmarkExample],
    eval_examples: Iterable[BenchmarkExample],
    predictions_csv: str | Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_examples = list(train_examples)
    eval_examples = list(eval_examples)
    train_targets = _unique_smiles(example.target_smiles for example in train_examples)
    train_sources = _unique_smiles(example.source_smiles for example in train_examples if example.source_smiles)
    train_target_scaffolds = {scaffold_key(smiles) for smiles in train_targets}
    train_source_scaffolds = {scaffold_key(smiles) for smiles in train_sources}
    target_index = _SimilarityIndex(train_targets)
    source_index = _SimilarityIndex(train_sources)
    top1_by_task = _read_top1_predictions(predictions_csv) if predictions_csv else {}

    rows: list[dict[str, object]] = []
    for example in eval_examples:
        target_scaffold = scaffold_key(example.target_smiles)
        source_scaffold = scaffold_key(example.source_smiles)
        best_target_smiles, best_target_sim = target_index.nearest(example.target_smiles)
        best_source_smiles, best_source_sim = source_index.nearest(example.target_smiles)
        source_target_sim = tanimoto(example.source_smiles, example.target_smiles) if example.source_smiles else 0.0
        top1 = top1_by_task.get(example.task_id, {})
        row = {
            "task_id": example.task_id,
            "task_type": example.task_type.value,
            "source_smiles": example.source_smiles or "",
            "target_smiles": example.target_smiles,
            "target_scaffold": target_scaffold,
            "source_scaffold": source_scaffold,
            "target_scaffold_in_train_targets": bool(target_scaffold and target_scaffold in train_target_scaffolds),
            "source_scaffold_in_train_sources": bool(source_scaffold and source_scaffold in train_source_scaffolds),
            "source_target_tanimoto": source_target_sim,
            "nearest_train_target_smiles": best_target_smiles,
            "nearest_train_target_tanimoto": best_target_sim,
            "nearest_train_source_smiles": best_source_smiles,
            "nearest_train_source_tanimoto": best_source_sim,
            "top1_candidate_smiles": top1.get("candidate_smiles", ""),
            "top1_candidate_origin": top1.get("origin", ""),
            "top1_target_tanimoto": _float(top1.get("target_tanimoto")),
            "top1_is_nearest_train_target": bool(top1.get("candidate_smiles") and top1.get("candidate_smiles") == best_target_smiles),
        }
        for threshold in THRESHOLDS:
            suffix = _threshold_suffix(threshold)
            row[f"nearest_train_target_ge_{suffix}"] = best_target_sim >= threshold
            row[f"source_target_ge_{suffix}"] = source_target_sim >= threshold
        rows.append(row)

    return rows, summarize_audit_rows(rows)


def summarize_audit_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["task_type"])].append(row)
        by_type["overall"].append(row)

    summary: list[dict[str, object]] = []
    for task_type in sorted(by_type, key=lambda value: (value != "overall", value)):
        group = by_type[task_type]
        record: dict[str, object] = {
            "task_type": task_type,
            "n": len(group),
            "target_scaffold_overlap_rate": _mean(_bool(row["target_scaffold_in_train_targets"]) for row in group),
            "source_scaffold_overlap_rate": _mean(_bool(row["source_scaffold_in_train_sources"]) for row in group),
            "mean_source_target_tanimoto": _mean(float(row["source_target_tanimoto"]) for row in group),
            "mean_nearest_train_target_tanimoto": _mean(float(row["nearest_train_target_tanimoto"]) for row in group),
            "mean_nearest_train_source_tanimoto": _mean(float(row["nearest_train_source_tanimoto"]) for row in group),
            "top1_is_nearest_train_target_rate": _mean(_bool(row["top1_is_nearest_train_target"]) for row in group),
        }
        for threshold in THRESHOLDS:
            suffix = _threshold_suffix(threshold)
            record[f"nearest_train_target_ge_{suffix}_rate"] = _mean(_bool(row[f"nearest_train_target_ge_{suffix}"]) for row in group)
            record[f"source_target_ge_{suffix}_rate"] = _mean(_bool(row[f"source_target_ge_{suffix}"]) for row in group)
        summary.append(record)
    return summary


def write_audit(rows: list[dict[str, object]], summary: list[dict[str, object]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "benchmark_audit_rows.csv", rows)
    _write_csv(out_dir / "benchmark_audit_summary.csv", summary)
    (out_dir / "benchmark_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/eval shortcut difficulty for SketchImage-JEPA benchmarks.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing train_examples.csv/eval_examples.csv.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    train_csv = Path(args.train_csv) if args.train_csv else (run_dir / "train_examples.csv" if run_dir else None)
    eval_csv = Path(args.eval_csv) if args.eval_csv else (run_dir / "eval_examples.csv" if run_dir else None)
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else (run_dir / "predictions.csv" if run_dir and (run_dir / "predictions.csv").exists() else None)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir if run_dir else Path("outputs/audit"))
    if train_csv is None or eval_csv is None:
        raise SystemExit("Provide --run-dir or both --train-csv and --eval-csv.")

    rows, summary = audit_split(load_examples_csv(train_csv), load_examples_csv(eval_csv), predictions_csv=predictions_csv)
    write_audit(rows, summary, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


class _SimilarityIndex:
    def __init__(self, smiles_values: list[str]):
        self.smiles = smiles_values
        self.bits = [_fingerprint_bits(smiles) for smiles in self.smiles]
        usable = [bits for bits in self.bits if bits is not None]
        self.matrix = np.stack(usable).astype(np.float32) if len(usable) == len(self.bits) and usable else None
        self.sums = self.matrix.sum(axis=1) if self.matrix is not None else None

    def nearest(self, query_smiles: str | None) -> tuple[str, float]:
        if not query_smiles or not self.smiles:
            return "", 0.0
        query = _fingerprint_bits(query_smiles)
        if self.matrix is not None and self.sums is not None and query is not None:
            query_sum = float(query.sum())
            intersection = self.matrix @ query
            denom = np.maximum(self.sums + query_sum - intersection, 1e-8)
            sims = intersection / denom
            idx = int(np.argmax(sims))
            return self.smiles[idx], float(sims[idx])
        scored = [(tanimoto(query_smiles, smiles), smiles) for smiles in self.smiles]
        sim, smiles = max(scored, key=lambda item: item[0])
        return smiles, float(sim)


def _fingerprint_bits(smiles: str | None) -> np.ndarray | None:
    bits = morgan_fingerprint_bits(smiles, n_bits=2048)
    return np.asarray(bits, dtype=np.float32) if bits is not None else None


def _unique_smiles(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _read_top1_predictions(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None or not Path(path).exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("rank", "999999"))) == 1:
                out[row["task_id"]] = row
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Iterable[float | bool]) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _threshold_suffix(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
