"""Runnable SketchImage-JEPA smoke experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .dataset import load_examples_csv, split_examples, toy_examples, write_examples_csv
from .decoder import RetrievalDecoder
from .features import MOLECULE_LATENT_VERSION, matrix_from_examples
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
    backend: str = "ridge",
    torch_hidden_dim: int = 1024,
    torch_epochs: int = 20,
    torch_batch_size: int = 128,
    torch_lr: float = 1e-3,
    torch_weight_decay: float = 1e-4,
    torch_diffusion_steps: int = 16,
    torch_train_noise: float = 0.35,
    torch_direct_loss_weight: float = 1.0,
    torch_cosine_loss_weight: float = 1.0,
    torch_positive_loss_weight: float = 8.0,
    torch_contrastive_loss_weight: float = 0.25,
    torch_contrastive_temperature: float = 0.10,
    torch_device: str = "auto",
    de_novo_latent_rerank_weight: float = 0.05,
    source_rerank_weight: float = 0.35,
    property_rerank_weight: float = 0.25,
    scaffold_rerank_bonus: float = 0.15,
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
    model = _build_model(
        backend=backend,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        ridge=ridge,
        torch_hidden_dim=torch_hidden_dim,
        torch_epochs=torch_epochs,
        torch_batch_size=torch_batch_size,
        torch_lr=torch_lr,
        torch_weight_decay=torch_weight_decay,
        torch_diffusion_steps=torch_diffusion_steps,
        torch_train_noise=torch_train_noise,
        torch_direct_loss_weight=torch_direct_loss_weight,
        torch_cosine_loss_weight=torch_cosine_loss_weight,
        torch_positive_loss_weight=torch_positive_loss_weight,
        torch_contrastive_loss_weight=torch_contrastive_loss_weight,
        torch_contrastive_temperature=torch_contrastive_temperature,
        torch_device=torch_device,
        seed=seed,
    ).fit(train_conditions, train_targets, train_sources)
    pred_latents = model.predict(eval_conditions, eval_sources)
    decoder = RetrievalDecoder(
        [example.target_smiles for example in train_examples],
        train_targets,
        de_novo_latent_rerank_weight=de_novo_latent_rerank_weight,
        source_rerank_weight=source_rerank_weight,
        property_rerank_weight=property_rerank_weight,
        scaffold_rerank_bonus=scaffold_rerank_bonus,
    )
    decoded = decoder.decode(
        pred_latents,
        [example.source_smiles for example in eval_examples],
        top_k=top_k,
        examples=eval_examples,
        source_latents=eval_sources,
    )
    scores_by_task = [score_candidates(example, candidates) for example, candidates in zip(eval_examples, decoded)]
    metrics = summarize_scores(scores_by_task)
    metrics.update({"train_tasks": float(len(train_examples)), "eval_tasks": float(len(eval_examples))})
    run_config = {
        "dataset_csv": str(dataset_csv) if dataset_csv else None,
        "train_csv": str(train_csv) if train_csv else None,
        "eval_csv": str(eval_csv) if eval_csv else None,
        "feature_dim": feature_dim,
        "latent_dim": latent_dim,
        "molecule_latent_version": MOLECULE_LATENT_VERSION,
        "top_k": top_k,
        "ridge": ridge,
        "limit": limit,
        "train_fraction": train_fraction,
        "seed": seed,
        "render_image_context": render_image_context,
        "preset": preset,
        "backend": backend,
        "torch_hidden_dim": torch_hidden_dim if backend == "torch_denoiser" else None,
        "torch_epochs": torch_epochs if backend == "torch_denoiser" else None,
        "torch_batch_size": torch_batch_size if backend == "torch_denoiser" else None,
        "torch_lr": torch_lr if backend == "torch_denoiser" else None,
        "torch_weight_decay": torch_weight_decay if backend == "torch_denoiser" else None,
        "torch_diffusion_steps": torch_diffusion_steps if backend == "torch_denoiser" else None,
        "torch_train_noise": torch_train_noise if backend == "torch_denoiser" else None,
        "torch_direct_loss_weight": torch_direct_loss_weight if backend == "torch_denoiser" else None,
        "torch_cosine_loss_weight": torch_cosine_loss_weight if backend == "torch_denoiser" else None,
        "torch_positive_loss_weight": torch_positive_loss_weight if backend == "torch_denoiser" else None,
        "torch_contrastive_loss_weight": torch_contrastive_loss_weight if backend == "torch_denoiser" else None,
        "torch_contrastive_temperature": torch_contrastive_temperature if backend == "torch_denoiser" else None,
        "torch_device": getattr(model, "device_name", torch_device) if backend == "torch_denoiser" else None,
        "de_novo_latent_rerank_weight": de_novo_latent_rerank_weight,
        "source_rerank_weight": source_rerank_weight,
        "property_rerank_weight": property_rerank_weight,
        "scaffold_rerank_bonus": scaffold_rerank_bonus,
        "train_image_context": train_image_meta,
        "eval_image_context": eval_image_meta,
        "model_history": model.history,
        "decoder": {
            "de_novo": "property_guided_retrieval",
            "source_conditioned": "task_guided_retrieval",
            "source_policy": "exclude_source_from_ranked_candidates",
            "ranking": "model_plus_source_property_scaffold_no_target_oracle",
        },
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
    parser.add_argument("--backend", choices=["ridge", "torch_denoiser"], default="ridge")
    parser.add_argument("--torch-hidden-dim", type=int, default=1024)
    parser.add_argument("--torch-epochs", type=int, default=20)
    parser.add_argument("--torch-batch-size", type=int, default=128)
    parser.add_argument("--torch-lr", type=float, default=1e-3)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument("--torch-diffusion-steps", type=int, default=16)
    parser.add_argument("--torch-train-noise", type=float, default=0.35)
    parser.add_argument("--torch-direct-loss-weight", type=float, default=1.0)
    parser.add_argument("--torch-cosine-loss-weight", type=float, default=1.0)
    parser.add_argument("--torch-positive-loss-weight", type=float, default=8.0)
    parser.add_argument("--torch-contrastive-loss-weight", type=float, default=0.25)
    parser.add_argument("--torch-contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--de-novo-latent-rerank-weight", type=float, default=0.05)
    parser.add_argument("--source-rerank-weight", type=float, default=0.35)
    parser.add_argument("--property-rerank-weight", type=float, default=0.25)
    parser.add_argument("--scaffold-rerank-bonus", type=float, default=0.15)
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
        backend=args.backend,
        torch_hidden_dim=args.torch_hidden_dim,
        torch_epochs=args.torch_epochs,
        torch_batch_size=args.torch_batch_size,
        torch_lr=args.torch_lr,
        torch_weight_decay=args.torch_weight_decay,
        torch_diffusion_steps=args.torch_diffusion_steps,
        torch_train_noise=args.torch_train_noise,
        torch_direct_loss_weight=args.torch_direct_loss_weight,
        torch_cosine_loss_weight=args.torch_cosine_loss_weight,
        torch_positive_loss_weight=args.torch_positive_loss_weight,
        torch_contrastive_loss_weight=args.torch_contrastive_loss_weight,
        torch_contrastive_temperature=args.torch_contrastive_temperature,
        torch_device=args.torch_device,
        de_novo_latent_rerank_weight=args.de_novo_latent_rerank_weight,
        source_rerank_weight=args.source_rerank_weight,
        property_rerank_weight=args.property_rerank_weight,
        scaffold_rerank_bonus=args.scaffold_rerank_bonus,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _build_model(
    backend: str,
    feature_dim: int,
    latent_dim: int,
    ridge: float,
    torch_hidden_dim: int,
    torch_epochs: int,
    torch_batch_size: int,
    torch_lr: float,
    torch_weight_decay: float,
    torch_diffusion_steps: int,
    torch_train_noise: float,
    torch_direct_loss_weight: float,
    torch_cosine_loss_weight: float,
    torch_positive_loss_weight: float,
    torch_contrastive_loss_weight: float,
    torch_contrastive_temperature: float,
    torch_device: str,
    seed: int,
):
    if backend == "ridge":
        return SketchImageJEPAPredictor(JEPAConfig(feature_dim=feature_dim, latent_dim=latent_dim, ridge=ridge))
    if backend == "torch_denoiser":
        from .torch_denoiser import TorchDenoiserConfig, TorchLatentDenoiser

        return TorchLatentDenoiser(
            TorchDenoiserConfig(
                feature_dim=feature_dim,
                latent_dim=latent_dim,
                hidden_dim=torch_hidden_dim,
                epochs=torch_epochs,
                batch_size=torch_batch_size,
                lr=torch_lr,
                weight_decay=torch_weight_decay,
                diffusion_steps=torch_diffusion_steps,
                train_noise=torch_train_noise,
                direct_loss_weight=torch_direct_loss_weight,
                cosine_loss_weight=torch_cosine_loss_weight,
                positive_loss_weight=torch_positive_loss_weight,
                contrastive_loss_weight=torch_contrastive_loss_weight,
                contrastive_temperature=torch_contrastive_temperature,
                device=torch_device,
                seed=seed,
            )
        )
    raise ValueError(f"Unsupported backend: {backend}")


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
        "property_mae",
        "property_success",
        "mw_abs_error",
        "logp_abs_error",
        "qed_abs_error",
        "tpsa_abs_error",
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
                    "property_mae": f"{score.property_mae:.6f}",
                    "property_success": score.property_success,
                    "mw_abs_error": f"{score.property_errors.get('MW', 0.0):.6f}",
                    "logp_abs_error": f"{score.property_errors.get('LogP', 0.0):.6f}",
                    "qed_abs_error": f"{score.property_errors.get('QED', 0.0):.6f}",
                    "tpsa_abs_error": f"{score.property_errors.get('TPSA', 0.0):.6f}",
                }
                writer.writerow(row)


if __name__ == "__main__":
    main()
