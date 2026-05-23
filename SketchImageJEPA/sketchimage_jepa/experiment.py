"""Runnable SketchImage-JEPA smoke experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .dataset import load_examples_csv, split_examples, toy_examples, write_examples_csv
from .decoder import RetrievalDecoder
from .features import matrix_from_examples
from .image_context import attach_rendered_image_context
from .jepa import JEPAConfig, SketchImageJEPAPredictor
from .report import summarize_predictions_csv
from .sketchmol_reference import SKETCHMOL_REFERENCE
from .verifier import score_candidates, summarize_scores


def run_experiment(
    output_dir: str | Path = "outputs/smoke",
    dataset_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    eval_csv: str | Path | None = None,
    feature_dim: int = 96,
    latent_dim: int = 48,
    top_k: int = 5,
    ridge: float = 1e-3,
    limit: int | None = None,
    train_fraction: float = 0.75,
    seed: int = 7,
    render_image_context: bool = False,
    preset: str | None = None,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples, eval_examples = _load_splits(dataset_csv=dataset_csv, train_csv=train_csv, eval_csv=eval_csv, limit=limit, train_fraction=train_fraction, seed=seed)
    if render_image_context:
        train_examples, train_image_meta = attach_rendered_image_context(train_examples, output_dir / "rendered_context" / "train")
        eval_examples, eval_image_meta = attach_rendered_image_context(eval_examples, output_dir / "rendered_context" / "eval")
    else:
        train_image_meta = {"rdkit_available": None, "rendered_images": 0, "masked_images": 0, "skipped_images": len(train_examples)}
        eval_image_meta = {"rdkit_available": None, "rendered_images": 0, "masked_images": 0, "skipped_images": len(eval_examples)}

    train_conditions, train_targets, train_sources = matrix_from_examples(train_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    eval_conditions, _, eval_sources = matrix_from_examples(eval_examples, feature_dim=feature_dim, latent_dim=latent_dim)
    model = SketchImageJEPAPredictor(JEPAConfig(feature_dim=feature_dim, latent_dim=latent_dim, ridge=ridge)).fit(train_conditions, train_targets, train_sources)
    pred_latents = model.predict(eval_conditions, eval_sources)
    decoder = RetrievalDecoder([example.target_smiles for example in train_examples], train_targets)
    decoded = decoder.decode(pred_latents, [example.source_smiles for example in eval_examples], top_k=top_k)
    scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
    metrics = summarize_scores(scores_by_task)
    metrics.update({"train_tasks": float(len(train_examples)), "eval_tasks": float(len(eval_examples))})
    run_config = {
        "dataset_csv": str(dataset_csv) if dataset_csv else None,
        "train_csv": str(train_csv) if train_csv else None,
        "eval_csv": str(eval_csv) if eval_csv else None,
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "top_k": top_k,
        "ridge": ridge,
        "limit": limit,
        "train_fraction": train_fraction,
        "seed": seed,
        "render_image_context": render_image_context,
        "preset": preset,
        "train_image_context": train_image_meta,
        "eval_image_context": eval_image_meta,
        "model_history": model.history,
    }
    if preset == "sketchmol_aligned":
        run_config["sketchmol_reference"] = SKETCHMOL_REFERENCE

    model.save(output_dir / "model")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_examples_csv(output_dir / "train_examples.csv", train_examples)
    write_examples_csv(output_dir / "eval_examples.csv", eval_examples)
    predictions_path = output_dir / "predictions.csv"
    _write_predictions(predictions_path, eval_examples, scores_by_task)
    summarize_predictions_csv(predictions_path, out_dir=output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SketchImage-JEPA smoke benchmark.")
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--feature-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-image-context", action="store_true")
    parser.add_argument("--preset", choices=["smoke", "sketchmol_aligned"], default=None)
    args = parser.parse_args()
    metrics = run_experiment(
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        top_k=args.top_k,
        ridge=args.ridge,
        limit=args.limit,
        train_fraction=args.train_fraction,
        seed=args.seed,
        render_image_context=args.render_image_context,
        preset=args.preset,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_splits(
    dataset_csv: str | Path | None,
    train_csv: str | Path | None,
    eval_csv: str | Path | None,
    limit: int | None,
    train_fraction: float,
    seed: int,
) -> tuple[list, list]:
    if train_csv or eval_csv:
        if not train_csv or not eval_csv:
            raise ValueError("Provide both --train-csv and --eval-csv, or provide neither.")
        train_examples = load_examples_csv(train_csv)
        eval_examples = load_examples_csv(eval_csv)
        if limit is not None:
            train_examples = train_examples[:limit]
            eval_examples = eval_examples[:limit]
    else:
        examples = load_examples_csv(dataset_csv) if dataset_csv else toy_examples()
        if limit is not None:
            examples = examples[:limit]
        train_examples, eval_examples = split_examples(examples, train_fraction=train_fraction, seed=seed)
    if not train_examples:
        raise ValueError("No training examples were provided.")
    if not eval_examples:
        raise ValueError("No evaluation examples were provided.")
    return train_examples, eval_examples


def _write_predictions(path: Path, examples, scores_by_task) -> None:
    fieldnames = [
        "task_id",
        "task_type",
        "instruction",
        "source_smiles",
        "target_smiles",
        "rank",
        "candidate_smiles",
        "origin",
        "valid",
        "target_tanimoto",
        "scaffold_match",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example, scores in zip(examples, scores_by_task):
            for score in scores:
                row = {
                    "task_id": example.task_id,
                    "task_type": example.task_type.value,
                    "instruction": example.instruction,
                    "source_smiles": example.source_smiles or "",
                    "target_smiles": example.target_smiles,
                    "rank": score.rank,
                    "candidate_smiles": score.smiles,
                    "origin": score.origin,
                    "valid": score.valid,
                    "target_tanimoto": f"{score.target_tanimoto:.6f}",
                    "scaffold_match": score.scaffold_match,
                    "score": f"{score.score:.6f}",
                }
                writer.writerow(row)


if __name__ == "__main__":
    main()
