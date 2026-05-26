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
    property_delta_weights: list[float] | None = None,
    top_configs: int = 20,
) -> list[dict[str, object]]:
    predictions_csv = Path(predictions_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(predictions_csv)
    enriched = _enrich_rows(rows)
    property_delta_weights = property_delta_weights or [0.0]
    records = []
    for base_weight in base_weights:
        for source_weight in source_weights:
            for property_weight in property_weights:
                for scaffold_weight in scaffold_weights:
                    for property_delta_weight in property_delta_weights:
                        reranked = _rerank_rows(
                            enriched,
                            base_weight=base_weight,
                            source_weight=source_weight,
                            property_weight=property_weight,
                            scaffold_weight=scaffold_weight,
                            property_delta_weight=property_delta_weight,
                        )
                        summary = summarize_prediction_rows(reranked)
                        overall = next(row for row in summary if row["task_type"] == "overall")
                        records.append(
                            {
                                "base_weight": base_weight,
                                "source_weight": source_weight,
                                "property_weight": property_weight,
                                "scaffold_weight": scaffold_weight,
                                "property_delta_weight": property_delta_weight,
                                "top1_target_tanimoto": overall["top1_target_tanimoto"],
                                "mean_best_tanimoto": overall["mean_best_tanimoto"],
                                "topk_target_hit": overall["topk_target_hit"],
                                "top1_scaffold_match": overall["top1_scaffold_match"],
                                "top1_property_success": overall.get("top1_property_success", 0.0),
                                "topk_property_success": overall.get("topk_property_success", 0.0),
                                "top1_property_delta_mae": overall.get("top1_property_delta_mae", 0.0),
                                "mean_best_property_delta_mae": overall.get("mean_best_property_delta_mae", 0.0),
                                "top1_property_delta_success": overall.get("top1_property_delta_success", 0.0),
                                "topk_property_delta_success": overall.get("topk_property_delta_success", 0.0),
                            }
                        )
    records.sort(
        key=lambda row: (
            float(row["top1_target_tanimoto"]),
            float(row["top1_property_delta_success"]),
            float(row["top1_property_success"]),
            float(row["top1_scaffold_match"]),
        ),
        reverse=True,
    )
    _write_csv(out_dir / "rerank_sweep_summary.csv", records)
    (out_dir / "rerank_sweep_summary.json").write_text(json.dumps(records[:top_configs], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if records:
        objective_rows = _best_by_objective(records)
        _write_csv(out_dir / "best_by_objective.csv", objective_rows)
        (out_dir / "best_by_objective.json").write_text(json.dumps(objective_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for row in objective_rows:
            objective = str(row["objective"])
            objective_rows_reranked = _rerank_rows(
                enriched,
                base_weight=float(row["base_weight"]),
                source_weight=float(row["source_weight"]),
                property_weight=float(row["property_weight"]),
                scaffold_weight=float(row["scaffold_weight"]),
                property_delta_weight=float(row["property_delta_weight"]),
            )
            _write_csv(out_dir / f"best_{objective}_reranked_predictions.csv", objective_rows_reranked)
            objective_summary = summarize_prediction_rows(objective_rows_reranked)
            _write_csv(out_dir / f"best_{objective}_task_type_summary.csv", objective_summary)
            (out_dir / f"best_{objective}_task_type_summary.json").write_text(
                json.dumps(objective_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        best = records[0]
        best_rows = _rerank_rows(
            enriched,
            base_weight=float(best["base_weight"]),
            source_weight=float(best["source_weight"]),
            property_weight=float(best["property_weight"]),
            scaffold_weight=float(best["scaffold_weight"]),
            property_delta_weight=float(best["property_delta_weight"]),
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
    parser.add_argument("--base-weights", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--source-weights", default="0,0.15,0.35,0.55")
    parser.add_argument("--property-weights", default="0,0.10,0.25,0.40")
    parser.add_argument("--scaffold-weights", default="0,0.10,0.20,0.30")
    parser.add_argument("--property-delta-weights", default="0,0.25,0.50,0.75,1.0,2.0,4.0,8.0")
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
        property_delta_weights=_floats(args.property_delta_weights),
        top_configs=args.top_configs,
    )
    print(json.dumps(records[: args.top_configs], indent=2, sort_keys=True))
    print(f"best_by_objective={out_dir / 'best_by_objective.csv'}")
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
            delta_mae = _float(updated.get("property_delta_mae"))
            updated["_property_delta_score"] = f"{max(0.0, 1.0 - delta_mae):.8f}" if updated.get("property_delta_mae") else "0.00000000"
            out.append(updated)
    return out


def _rerank_rows(
    rows: list[dict[str, str]],
    base_weight: float,
    source_weight: float,
    property_weight: float,
    scaffold_weight: float,
    property_delta_weight: float = 0.0,
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
                + property_delta_weight * _float(row.get("_property_delta_score"))
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


def _best_by_objective(records: list[dict[str, object]]) -> list[dict[str, object]]:
    objectives = {
        "target": lambda row: (
            _float(row.get("top1_target_tanimoto")),
            _float(row.get("top1_property_delta_success")),
            _float(row.get("top1_property_success")),
        ),
        "property_delta": lambda row: (
            _float(row.get("top1_property_delta_success")),
            -_float(row.get("top1_property_delta_mae")),
            _float(row.get("top1_target_tanimoto")),
        ),
        "property": lambda row: (
            _float(row.get("top1_property_success")),
            _float(row.get("top1_property_delta_success")),
            _float(row.get("top1_target_tanimoto")),
        ),
        "balanced": lambda row: (
            _float(row.get("top1_target_tanimoto"))
            + 0.25 * _float(row.get("top1_property_delta_success"))
            + 0.25 * _float(row.get("top1_property_success")),
            _float(row.get("top1_target_tanimoto")),
        ),
    }
    out = []
    for name, key_fn in objectives.items():
        best = max(records, key=key_fn)
        row = {"objective": name}
        row.update(best)
        out.append(row)
    return out


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
