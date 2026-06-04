"""Training entry point for Latent Edit Trajectory Attention."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .data import (
    SketchMolOptPairConfig,
    SketchMolOptPairDataset,
    SketchMolTrajectoryConfig,
    SketchMolTrajectoryDataset,
    SyntheticTrajectoryConfig,
    SyntheticTrajectoryDataset,
    collate_trajectory_batch,
    move_batch_to_device,
)
from .models import (
    CurrentStateDiffusionEditor,
    TrajectoryConditionedDiffusionEditor,
    TrajectoryDiffusionConfig,
    _load_torch,
    add_diffusion_noise,
)


MODEL_KINDS = {"history", "current_only", "no_reward_history", "shuffled_history"}


def _resolve_device(device: str) -> str:
    torch = _load_torch()
    value = device.strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch = _load_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on local env
        torch.cuda.manual_seed_all(seed)


def _prepare_batch_for_model_kind(batch: dict[str, Any], model_kind: str) -> dict[str, Any]:
    torch = _load_torch()
    if model_kind == "no_reward_history":
        batch = dict(batch)
        batch["property_delta"] = torch.zeros_like(batch["property_delta"])
    elif model_kind == "shuffled_history":
        batch = dict(batch)
        z_history = batch["z_history"].clone()
        property_delta = batch["property_delta"].clone()
        edit_type_ids = batch["edit_type_ids"].clone()
        history_mask = batch["history_mask"]
        for row_idx in range(z_history.shape[0]):
            length = int(history_mask[row_idx].long().sum().item())
            if length <= 1:
                continue
            order = torch.randperm(length, device=z_history.device)
            z_history[row_idx, :length] = z_history[row_idx, :length][order]
            property_delta[row_idx, :length] = property_delta[row_idx, :length][order]
            edit_type_ids[row_idx, :length] = edit_type_ids[row_idx, :length][order]
        batch["z_history"] = z_history
        batch["property_delta"] = property_delta
        batch["edit_type_ids"] = edit_type_ids
    return batch


def _train_editor(
    dataset: Any,
    model_config: TrajectoryDiffusionConfig,
    output_path: Path,
    phase: str,
    run_config: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    model_kind: str = "history",
) -> dict[str, Any]:
    torch = _load_torch()
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"Unsupported model_kind={model_kind!r}; expected one of {sorted(MODEL_KINDS)}.")
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    collate_fn = collate_trajectory_batch if getattr(dataset, "requires_padding", False) else None
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model_factory = CurrentStateDiffusionEditor if model_kind == "current_only" else TrajectoryConditionedDiffusionEditor
    model = model_factory(model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            batch = move_batch_to_device(batch, resolved_device)
            batch = _prepare_batch_for_model_kind(batch, model_kind)
            clean_next_z = batch["next_z"]
            noise_step = torch.randint(
                low=1,
                high=model_config.diffusion_steps + 1,
                size=(clean_next_z.shape[0],),
                device=clean_next_z.device,
            )
            noisy_next_z, noise = add_diffusion_noise(clean_next_z, noise_step, model_config.diffusion_steps)
            loss = model.denoising_loss(
                noisy_next_z=noisy_next_z,
                noise=noise,
                noise_step=noise_step,
                z_history=batch["z_history"],
                property_delta=batch["property_delta"],
                edit_type_ids=batch["edit_type_ids"],
                history_mask=batch["history_mask"],
                target=batch["target"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size_actual = int(clean_next_z.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size_actual
            seen += batch_size_actual

        epoch_loss = total_loss / max(1, seen)
        history.append({"epoch": float(epoch), "loss": float(epoch_loss)})

    metrics = {
        "phase": phase,
        "model_kind": model_kind,
        "examples": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": resolved_device,
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "loss_decreased": bool(history and history[-1]["loss"] <= history[0]["loss"]),
    }
    run_config["training"] = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": resolved_device,
    }
    run_config["model_kind"] = model_kind
    (output_path / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_synthetic_training(
    output_dir: str | Path = "outputs/runs/synthetic_trajectory_attention_seed7",
    seed: int = 7,
    examples: int = 256,
    history_length: int = 6,
    latent_dim: int = 128,
    property_dim: int = 4,
    target_dim: int = 4,
    edit_type_count: int = 16,
    hidden_dim: int = 256,
    transformer_layers: int = 4,
    attention_heads: int = 8,
    diffusion_steps: int = 100,
    max_history: int = 16,
    dropout: float = 0.1,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "auto",
    model_kind: str = "history",
) -> dict[str, Any]:
    """Train the trajectory-conditioned editor on synthetic latent trajectories."""

    _set_seeds(seed)
    output_path = Path(output_dir)
    model_config = TrajectoryDiffusionConfig(
        latent_dim=latent_dim,
        property_dim=property_dim,
        target_dim=target_dim,
        edit_type_count=edit_type_count,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        diffusion_steps=diffusion_steps,
        max_history=max_history,
        dropout=dropout,
    )
    data_config = SyntheticTrajectoryConfig(
        examples=examples,
        history_length=history_length,
        seed=seed,
    )
    dataset = SyntheticTrajectoryDataset(model_config=model_config, data_config=data_config)
    run_config = {
        "model": model_config.__dict__,
        "data": data_config.__dict__,
    }
    metrics = _train_editor(
        dataset=dataset,
        model_config=model_config,
        output_path=output_path,
        phase="latent_edit_trajectory_attention_synthetic",
        run_config=run_config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        model_kind=model_kind,
    )
    metrics["history_length"] = history_length
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_sketchmol_opt_training(
    output_dir: str | Path = "outputs/runs/sketchmol_opt_pairs_seed7",
    opt_examples_dir: str = SketchMolOptPairConfig.opt_examples_dir,
    seed: int = 7,
    max_examples: int | None = None,
    fingerprint_radius: int = 2,
    latent_dim: int = 256,
    property_dim: int = 4,
    target_dim: int = 4,
    edit_type_count: int = 16,
    hidden_dim: int = 256,
    transformer_layers: int = 4,
    attention_heads: int = 8,
    diffusion_steps: int = 100,
    max_history: int = 4,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "auto",
    model_kind: str = "history",
) -> dict[str, Any]:
    """Train on the original SketchMol before/after optimization examples."""

    _set_seeds(seed)
    output_path = Path(output_dir)
    model_config = TrajectoryDiffusionConfig(
        latent_dim=latent_dim,
        property_dim=property_dim,
        target_dim=target_dim,
        edit_type_count=edit_type_count,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        diffusion_steps=diffusion_steps,
        max_history=max_history,
        dropout=dropout,
    )
    data_config = SketchMolOptPairConfig(
        opt_examples_dir=opt_examples_dir,
        fingerprint_radius=fingerprint_radius,
        max_examples=max_examples,
    )
    dataset = SketchMolOptPairDataset(model_config=model_config, data_config=data_config)
    run_config = {
        "model": model_config.__dict__,
        "data": data_config.__dict__,
        "sketchmol_reference": {
            "source": "Research/Molecule Generation/SketchMol/SketchMol-v1-main/opt_examples",
            "format": "Before_opt_smiles -> After_opt_smiles optimization pairs",
        },
    }
    phase_suffix = model_kind
    return _train_editor(
        dataset=dataset,
        model_config=model_config,
        output_path=output_path,
        phase=f"latent_edit_trajectory_attention_sketchmol_opt_pairs_{phase_suffix}",
        run_config=run_config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        model_kind=model_kind,
    )


def run_sketchmol_trajectory_training(
    output_dir: str | Path = "outputs/runs/sketchmol_trajectory_history_seed7",
    trajectory_path: str = SketchMolTrajectoryConfig.trajectory_path,
    seed: int = 7,
    max_examples: int | None = None,
    min_history: int = 1,
    fingerprint_radius: int = 2,
    latent_dim: int = 256,
    property_dim: int = 5,
    target_dim: int = 4,
    edit_type_count: int = 16,
    hidden_dim: int = 256,
    transformer_layers: int = 4,
    attention_heads: int = 8,
    diffusion_steps: int = 100,
    max_history: int = 8,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "auto",
    model_kind: str = "history",
) -> dict[str, Any]:
    """Train on multi-step SketchMol trajectory logs."""

    _set_seeds(seed)
    output_path = Path(output_dir)
    model_config = TrajectoryDiffusionConfig(
        latent_dim=latent_dim,
        property_dim=property_dim,
        target_dim=target_dim,
        edit_type_count=edit_type_count,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        diffusion_steps=diffusion_steps,
        max_history=max_history,
        dropout=dropout,
    )
    data_config = SketchMolTrajectoryConfig(
        trajectory_path=trajectory_path,
        fingerprint_radius=fingerprint_radius,
        max_examples=max_examples,
        min_history=min_history,
    )
    dataset = SketchMolTrajectoryDataset(model_config=model_config, data_config=data_config)
    dataset.requires_padding = True
    run_config = {
        "model": model_config.__dict__,
        "data": data_config.__dict__,
        "sketchmol_reference": {
            "format": "trajectory JSONL with trajectory_id, step, SMILES, properties, reward",
            "source": trajectory_path,
        },
    }
    return _train_editor(
        dataset=dataset,
        model_config=model_config,
        output_path=output_path,
        phase=f"latent_edit_trajectory_attention_sketchmol_trajectory_{model_kind}",
        run_config=run_config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        model_kind=model_kind,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a history-conditioned latent diffusion editor.")
    parser.add_argument("--dataset", choices=["synthetic", "sketchmol_opt", "sketchmol_trajectory"], default="synthetic")
    parser.add_argument("--model-kind", choices=sorted(MODEL_KINDS), default="history")
    parser.add_argument("--output-dir", default="outputs/runs/synthetic_trajectory_attention_seed7")
    parser.add_argument("--opt-examples-dir", default=SketchMolOptPairConfig.opt_examples_dir)
    parser.add_argument("--trajectory-path", default=SketchMolTrajectoryConfig.trajectory_path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--property-dim", type=int, default=4)
    parser.add_argument("--target-dim", type=int, default=4)
    parser.add_argument("--edit-type-count", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--max-history", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dataset == "sketchmol_trajectory":
        metrics = run_sketchmol_trajectory_training(
            output_dir=args.output_dir,
            trajectory_path=args.trajectory_path,
            seed=args.seed,
            max_examples=args.max_examples,
            min_history=args.min_history,
            fingerprint_radius=args.fingerprint_radius,
            latent_dim=args.latent_dim,
            property_dim=args.property_dim,
            target_dim=args.target_dim,
            edit_type_count=args.edit_type_count,
            hidden_dim=args.hidden_dim,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            diffusion_steps=args.diffusion_steps,
            max_history=args.max_history,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            model_kind=args.model_kind,
        )
    elif args.dataset == "sketchmol_opt":
        metrics = run_sketchmol_opt_training(
            output_dir=args.output_dir,
            opt_examples_dir=args.opt_examples_dir,
            seed=args.seed,
            max_examples=args.max_examples,
            fingerprint_radius=args.fingerprint_radius,
            latent_dim=args.latent_dim,
            property_dim=args.property_dim,
            target_dim=args.target_dim,
            edit_type_count=args.edit_type_count,
            hidden_dim=args.hidden_dim,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            diffusion_steps=args.diffusion_steps,
            max_history=args.max_history,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            model_kind=args.model_kind,
        )
    else:
        metrics = run_synthetic_training(
            output_dir=args.output_dir,
            seed=args.seed,
            examples=args.examples,
            history_length=args.history_length,
            latent_dim=args.latent_dim,
            property_dim=args.property_dim,
            target_dim=args.target_dim,
            edit_type_count=args.edit_type_count,
            hidden_dim=args.hidden_dim,
            transformer_layers=args.transformer_layers,
            attention_heads=args.attention_heads,
            diffusion_steps=args.diffusion_steps,
            max_history=args.max_history,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            model_kind=args.model_kind,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

