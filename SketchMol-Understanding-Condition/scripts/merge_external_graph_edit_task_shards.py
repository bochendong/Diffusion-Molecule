#!/usr/bin/env python3
"""Merge per-task GraphEditDSL shard outputs into one benchmark CSV."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_external_agentic_revise_predictions as revise  # noqa: E402
import build_external_graph_edit_agent_predictions as graph_edit  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--shard-dir", action="append", required=True, type=Path)
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--candidate-output-csv", required=True, type=Path)
    parser.add_argument("--plan-jsonl", type=Path, default=None)
    parser.add_argument(
        "--prediction-name",
        default="graph_edit_agent_predictions.csv",
        help="Prediction CSV filename inside each shard's benchmark_graph_edit_agent directory.",
    )
    parser.add_argument(
        "--candidate-name",
        default="graph_edit_agent_candidate_predictions.csv",
        help="Candidate CSV filename inside each shard's benchmark_graph_edit_agent directory.",
    )
    parser.add_argument(
        "--plan-name",
        default="graph_edit_plans.jsonl",
        help="Plan JSONL filename inside each shard directory.",
    )
    return parser.parse_args(argv)


def shard_prediction_path(shard_dir: Path, filename: str) -> Path:
    direct = shard_dir / "benchmark_graph_edit_agent" / filename
    if direct.exists():
        return direct
    checkpoint = shard_dir / "benchmark_graph_edit_agent" / "checkpoints" / "predictions.partial.csv"
    if filename.endswith("predictions.csv") and checkpoint.exists():
        return checkpoint
    return direct


def shard_candidate_path(shard_dir: Path, filename: str) -> Path:
    direct = shard_dir / "benchmark_graph_edit_agent" / filename
    if direct.exists():
        return direct
    checkpoint = shard_dir / "benchmark_graph_edit_agent" / "checkpoints" / "candidates.partial.csv"
    if filename.endswith("candidate_predictions.csv") and checkpoint.exists():
        return checkpoint
    return direct


def load_predictions_by_key(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"missing shard prediction CSV: {path}")
    return {revise.row_key(row): dict(row) for row in revise.read_rows(path) if revise.row_key(row)}


def load_candidates_grouped(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"missing shard candidate CSV: {path}")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in revise.read_rows(path):
        key = revise.row_key(row)
        if not key:
            continue
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = revise.read_rows(args.rows_csv)
    predictions_by_key: dict[str, dict[str, object]] = {}
    candidates_by_key: dict[str, list[dict[str, object]]] = {}
    shard_report = []

    for shard_dir in args.shard_dir:
        prediction_path = shard_prediction_path(shard_dir, args.prediction_name)
        candidate_path = shard_candidate_path(shard_dir, args.candidate_name)
        shard_predictions = load_predictions_by_key(prediction_path)
        shard_candidates = load_candidates_grouped(candidate_path)
        predictions_by_key.update(shard_predictions)
        for key, values in shard_candidates.items():
            candidates_by_key.setdefault(key, []).extend(values)
        shard_report.append(
            {
                "shard_dir": str(shard_dir),
                "prediction_csv": str(prediction_path),
                "candidate_csv": str(candidate_path),
                "prediction_rows": len(shard_predictions),
                "candidate_rows": sum(len(values) for values in shard_candidates.values()),
            }
        )

    ordered_predictions: list[dict[str, object]] = []
    ordered_candidates: list[dict[str, object]] = []
    missing_keys: list[str] = []
    for row in rows:
        key = revise.row_key(row)
        if not key:
            continue
        prediction = predictions_by_key.get(key)
        if prediction is None:
            missing_keys.append(key)
            continue
        ordered_predictions.append(prediction)
        ordered_candidates.extend(candidates_by_key.get(key, []))

    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        raise RuntimeError(
            f"missing {len(missing_keys)} prediction rows after merge; first keys: {preview}"
        )

    graph_edit.finalize_graph_edit_outputs(
        argparse.Namespace(
            prediction_csv=args.prediction_csv,
            candidate_output_csv=args.candidate_output_csv,
            plan_jsonl=None,
        ),
        output_rows=ordered_predictions,
        candidate_rows=ordered_candidates,
        plan_records=[],
    )
    if args.plan_jsonl is not None:
        args.plan_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.plan_jsonl.open("wb") as handle:
            for shard_dir in args.shard_dir:
                plan_path = shard_dir / args.plan_name
                if not plan_path.exists():
                    continue
                with plan_path.open("rb") as src:
                    shutil.copyfileobj(src, handle)

    summary = {
        "rows_csv": str(args.rows_csv),
        "prediction_csv": str(args.prediction_csv),
        "candidate_output_csv": str(args.candidate_output_csv),
        "plan_jsonl": str(args.plan_jsonl) if args.plan_jsonl else "",
        "merged_rows": len(ordered_predictions),
        "merged_candidate_rows": len(ordered_candidates),
        "shards": shard_report,
    }
    args.prediction_csv.with_name("graph_edit_merge.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
