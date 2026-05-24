"""Post-hoc reranking diagnostics for existing SketchImage-JEPA predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .chem import scaffold_key, tanimoto
from .property_guidance import parse_property_targets, property_match_score
from .report import summarize_prediction_rows


def rerank_predictions_csv(
    predictions_csv: str | Path,
    out_dir: str | Path,
    base_weights: list[float],
    source_weights: list[float],
    property_weights: list[float],
    scaffold_weights: list[float],
    top_configs: int = 20,
) -> list[dict[str, object]]:
    predictions_csv = Path(predictions_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(predictions_csv)
    enriched = _enrich_rows(rows)
    records = []
    for base_weight in base_weights:
        for source_weight in source_weights:
            for property_weight in property_weights:
                for scaffold_weight in scaffold_weights:
                    reranked = _rerank_rows(
                        enriched,
                        base_weight=base_weight,
                        source_weight=source_weight,
                        property_weight=property_weight,
                        scaffold_weight=scaffold_weight,
                    )
                    summary = summarize_prediction_rows(reranked)
                    overall = next(row for row in summary if row["task_type"] == "overall")
                    records.append(
                        {
                            "base_weight": base_weight,
                            "source_weight": source_weight,
                            "property_weight": property_weight,
                            "scaffold_weight": scaffold_weight,
                            "top1_target_tanimoto": overall["top1_target_tanimoto"],
                            "mean_best_tanimoto": overall["mean_best_tanimoto"],
                            "topk_target_hit": overall["topk_target_hit"],
                            "top1_scaffold_match": overall["top1_scaffold_match"],
                            "top1_property_success": overall.get("top1_property_success", 0.0),
                            "topk_property_success": overall.get("topk_property_success", 0.0),
                        }
                    )
    records.sort(
        key=lambda row: (
            float(row["top1_target_tanimoto"]),
            float(row["top1_property_success"]),
            float(row["top1_scaffold_match"]),
        ),
        reverse=True,
    )
    _write_csv(out_dir / "rerank_sweep_summary.csv", records)
    (out_dir / "rerank_sweep_summary.json").write_text(json.dumps(records[:top_configs], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if records:
        best = records[0]
        best_rows = _rerank_rows(
            enriched,
            base_weight=float(best["base_weight"]),
            source_weight=float(best["source_weight"]),
            property_weight=float(best["property_weight"]),
            scaffold_weight=float(best["scaffold_weight"]),
        )
        _write_csv(out_dir / "best_reranked_predictions.csv", best_rows)
        best_summary = summarize_prediction_rows(best_rows)
        _write_csv(out_dir / "best_task_type_summary.csv", best_summary)
        (out_dir / "best_task_type_summary.json").write_text(json.dumps(best_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep non-oracle reranking weights over an existing predictions.csv.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--base-weights", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--source-weights", default="0,0.15,0.35,0.55")
    parser.add_argument("--property-weights", default="0,0.10,0.25,0.40")
    parser.add_argument("--scaffold-weights", default="0,0.10,0.20,0.30")
    parser.add_argument("--top-configs", type=int, default=20)
    args = parser.parse_args()
    predictions = Path(args.predictions)
    out_dir = Path(args.out_dir) if args.out_dir else predictions.parent / "rerank_diagnostics"
    records = rerank_predictions_csv(
        predictions,
        out_dir=out_dir,
        base_weights=_floats(args.base_weights),
        source_weights=_floats(args.source_weights),
        property_weights=_floats(args.property_weights),
        scaffold_weights=_floats(args.scaffold_weights),
        top_configs=args.top_configs,
    )
    print(json.dumps(records[: args.top_configs], indent=2, sort_keys=True))
    print(f"wrote={out_dir}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    out = []
    for task_rows in by_task.values():
        scores = [_float(row.get("score")) for row in task_rows]
        score_min = min(scores) if scores else 0.0
        score_max = max(scores) if scores else 0.0
        denom = max(score_max - score_min, 1e-8)
        for row in task_rows:
            updated = dict(row)
            candidate = updated.get("candidate_smiles", "")
            source = updated.get("source_smiles", "")
            targets = parse_property_targets(updated.get("instruction", ""))
            desc_score = 0.0
            if targets:
                from .chem import molecular_descriptors

                desc_score = property_match_score(molecular_descriptors(candidate).descriptors, targets)
            updated["_base_score"] = f"{(_float(updated.get('score')) - score_min) / denom:.8f}"
            updated["_source_score"] = f"{tanimoto(source, candidate) if source else 0.0:.8f}"
            source_scaffold = scaffold_key(source)
            candidate_scaffold = scaffold_key(candidate)
            updated["_scaffold_score"] = "1.00000000" if source_scaffold and source_scaffold == candidate_scaffold else "0.00000000"
            updated["_property_score"] = f"{desc_score:.8f}"
            out.append(updated)
    return out


def _rerank_rows(
    rows: list[dict[str, str]],
    base_weight: float,
    source_weight: float,
    property_weight: float,
    scaffold_weight: float,
) -> list[dict[str, str]]:
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    reranked = []
    for task_rows in by_task.values():
        rescored = []
        for row in task_rows:
            score = (
                base_weight * _float(row.get("_base_score"))
                + source_weight * _float(row.get("_source_score"))
                + property_weight * _float(row.get("_property_score"))
                + scaffold_weight * _float(row.get("_scaffold_score"))
            )
            updated = {key: value for key, value in row.items() if not key.startswith("_")}
            updated["rerank_score"] = f"{score:.8f}"
            rescored.append((score, -int(float(row.get("rank", 999999))), updated))
        rescored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (_, _, row) in enumerate(rescored, start=1):
            updated = dict(row)
            updated["rank"] = str(rank)
            reranked.append(updated)
    return reranked


def _floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
