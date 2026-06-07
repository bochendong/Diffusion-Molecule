#!/usr/bin/env python
"""Merge sharded multi-property benchmark outputs into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_multiproperty_retrieval import _read_rows, _summarize, _write_report, _write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-tanimoto-thresholds", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_csvs = sorted(args.shards_dir.glob("shard_*_of_*/benchmark_decoded.csv"))
    if not shard_csvs:
        raise SystemExit(f"No shard decoded CSVs found under {args.shards_dir}")

    decoded_rows = []
    shard_metrics = []
    for csv_path in shard_csvs:
        decoded_rows.extend(_read_rows(csv_path))
        metrics_path = csv_path.parent / "metrics.json"
        if metrics_path.exists():
            shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not shard_metrics:
        raise SystemExit(f"No shard metrics.json files found under {args.shards_dir}")

    first_metrics = shard_metrics[0]
    source_tanimoto_thresholds = args.source_tanimoto_thresholds
    if source_tanimoto_thresholds is None:
        source_tanimoto_thresholds = ",".join(
            str(value) for value in first_metrics.get("source_tanimoto_thresholds", [])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _summarize(
        decoded_rows,
        source_tanimoto_thresholds=[
            float(value)
            for value in str(source_tanimoto_thresholds or "").split(",")
            if str(value).strip()
        ],
    )
    _write_rows(args.output_dir / "benchmark_decoded.csv", decoded_rows)
    _write_rows(args.output_dir / "benchmark_summary.csv", summary_rows)
    _write_report(args.output_dir / "benchmark_report.md", summary_rows, _report_args(first_metrics, source_tanimoto_thresholds))

    payload = {
        "decoded_csv": str(args.output_dir / "benchmark_decoded.csv"),
        "output_dir": str(args.output_dir),
        "report": str(args.output_dir / "benchmark_report.md"),
        "shard_count": len(shard_csvs),
        "shards_dir": str(args.shards_dir),
        "source_tanimoto_thresholds": source_tanimoto_thresholds,
        "summary_csv": str(args.output_dir / "benchmark_summary.csv"),
        "total_decoded_rows": len(decoded_rows),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _report_args(metrics: dict[str, object], source_tanimoto_thresholds: str) -> argparse.Namespace:
    return argparse.Namespace(
        allow_eval_target_candidates=not bool(metrics.get("eval_target_candidates_excluded", True)),
        candidate_molecule_db_csv=metrics.get("candidate_molecule_db_csv"),
        condition_feature_array=metrics.get("condition_feature_array"),
        condition_feature_variant=metrics.get("condition_feature_variant"),
        condition_features_dir=metrics.get("condition_features_dir"),
        condition_rows_csv=metrics.get("condition_rows_csv"),
        edit_latent_delta_weight=float(metrics.get("edit_latent_delta_weight", 0.35)),
        edit_latent_dir=metrics.get("edit_latent_dir"),
        edit_latent_direction_weight=float(metrics.get("edit_latent_direction_weight", 0.1)),
        edit_latent_fingerprint_weight=float(metrics.get("edit_latent_fingerprint_weight", 0.0)),
        edit_latent_property_weight=float(metrics.get("edit_latent_property_weight", 1.0)),
        edit_latent_source_similarity_rerank_candidates=int(
            metrics.get("edit_latent_source_similarity_rerank_candidates", 0)
        ),
        edit_latent_source_similarity_weight=float(metrics.get("edit_latent_source_similarity_weight", 0.25)),
        eval_split="eval",
        max_eval_per_property_count=metrics.get("max_eval_per_property_count"),
        restrict_eval_to_edit_latent_index=bool(metrics.get("restrict_eval_to_edit_latent_index", False)),
        scaffold_fallback_mode=metrics.get("scaffold_fallback_mode", "global"),
        source_tanimoto_thresholds=source_tanimoto_thresholds,
    )


if __name__ == "__main__":
    main()
