"""Train and evaluate an instruction-guided tabular diffusion edit planner."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .decoder import decode_table_row
from .diffusion import TabularDiffusion
from .instruction_evaluate import evaluate_instruction_candidates
from .instruction_features import (
    INSTRUCTION_SPEC_FEATURE_END,
    INSTRUCTION_SPEC_FEATURE_START,
    condition_from_source_and_spec,
    target_table_from_smiles,
)
from .instruction_multimodal import MULTIMODAL_CONTEXT_MODES, multimodal_context_from_row, multimodal_feature_names
from .instruction_source_decoder import SourceAwareCandidateIndex, SourceAwareDecoderConfig, decode_source_aware
from .io import make_run_dir, save_json, save_text, set_seed
from .schema import TABLE_COLUMNS


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    save_json(vars(args), run_dir / "config.json")

    print(f"Loading dataset {args.dataset}...", flush=True)
    dataset = pd.read_csv(args.dataset)
    if args.limit:
        dataset = dataset.head(args.limit).reset_index(drop=True)
    train_df = dataset[dataset.get("split", "train") == "train"].reset_index(drop=True) if "split" in dataset else dataset
    eval_df = dataset[dataset["split"].isin(args.eval_splits)].reset_index(drop=True) if "split" in dataset else dataset
    if args.eval_limit:
        eval_df = eval_df.head(args.eval_limit).reset_index(drop=True)
    if train_df.empty or eval_df.empty:
        raise ValueError("Instruction dataset must contain non-empty train and eval splits.")

    print(
        f"Building feature arrays: train={len(train_df)} eval={len(eval_df)} "
        f"(RDKit per row; may take several minutes)...",
        flush=True,
    )
    train_x, train_y = _build_arrays(
        train_df,
        multimodal_context=args.multimodal_context,
        allow_target_reference=args.allow_target_reference,
        disable_instruction_features=args.disable_instruction_features,
    )
    eval_x, _ = _build_arrays(
        eval_df,
        multimodal_context=args.multimodal_context,
        allow_target_reference=args.allow_target_reference,
        disable_instruction_features=args.disable_instruction_features,
    )
    train_df.to_csv(run_dir / "tables" / "instruction_train.csv", index=False)
    eval_df.to_csv(run_dir / "tables" / "instruction_eval.csv", index=False)

    fit_name = "latent VAE diffusion" if args.latent_vae else "diffusion"
    print(f"Fitting {fit_name} (backend may be sklearn MLP on ~{len(train_df) * args.noise_repeats} synthetic steps)...", flush=True)
    diffusion, backend = _fit_diffusion(args, train_y, train_x)
    print(f"Fit done ({backend}).", flush=True)
    model_path = run_dir / "models" / ("instruction_diffusion.pt" if backend.startswith("torch") else "instruction_diffusion.pkl")
    diffusion.save(model_path)
    source_index = None
    if args.source_aware_decoder:
        print("Building source-aware retrieval index from the train split...", flush=True)
        source_index = SourceAwareCandidateIndex.from_dataframe(train_df)
        print(f"Source-aware index size={len(source_index.candidates)} molecules.", flush=True)

    n_gen = len(eval_df) * args.samples_per_instruction
    print(
        f"Sampling {len(eval_df)} instructions x {args.samples_per_instruction} = {n_gen} plans "
        f"({args.timesteps} denoise steps each)...",
        flush=True,
    )
    generated_df = _sample_edit_plans(diffusion, eval_x, eval_df, args.samples_per_instruction)
    generated_df.to_csv(run_dir / "tables" / "generated_edit_plans.csv", index=False)
    print(f"Decoding {len(generated_df)} rows to SMILES (RDKit; often the slowest step)...", flush=True)
    decoded_df = _decode_and_attach(
        generated_df,
        eval_df,
        top_k=args.decode_top_k,
        dynamic_decoder=args.dynamic_decoder,
        source_index=source_index,
        source_aware_config=SourceAwareDecoderConfig(
            pool_size=args.source_aware_pool_size,
            plan_neighbors=args.source_aware_plan_neighbors,
            source_neighbors=args.source_aware_source_neighbors,
            reference_neighbors=args.source_aware_reference_neighbors,
            verify_candidates=args.source_aware_verify_candidates,
        ),
    )
    decoded_df.to_csv(run_dir / "tables" / "decoded_instruction_candidates.csv", index=False)

    train_smiles = set(train_df["source_smiles"].astype(str)) | set(train_df["target_smiles"].astype(str))
    metrics, detailed = evaluate_instruction_candidates(decoded_df, train_smiles=train_smiles)
    detailed.to_csv(run_dir / "tables" / "verified_instruction_candidates.csv", index=False)
    metrics.update(
        {
            "backend": backend,
            "train_size": float(len(train_df)),
            "eval_size": float(len(eval_df)),
            "condition_dim": float(train_x.shape[1]),
            "multimodal_feature_dim": float(len(multimodal_feature_names(args.multimodal_context))),
            "source_aware_decoder": bool(args.source_aware_decoder),
            "latent_vae": bool(args.latent_vae),
            "vae_latent_dim": float(args.vae_latent_dim) if args.latent_vae else 0.0,
            "run_dir": str(run_dir),
        }
    )
    save_json(metrics, run_dir / "metrics.json")
    save_json(_environment(), run_dir / "environment.json")
    save_text(_summary(metrics, args), run_dir / "summary.txt")
    print(_summary(metrics, args))
    print(f"run_dir={run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruction-guided molecular editing with tabular diffusion.")
    parser.add_argument("--dataset", default="data/instruction_editing.csv")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="instruction_editing")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--eval-limit", type=int, default=2000)
    parser.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    parser.add_argument("--latent-vae", action="store_true", help="Train a table/molecule VAE and run diffusion in its latent space.")
    parser.add_argument("--samples-per-instruction", type=int, default=8)
    parser.add_argument("--decode-top-k", type=int, default=2)
    parser.add_argument("--dynamic-decoder", action="store_true")
    parser.add_argument("--disable-source-aware-decoder", dest="source_aware_decoder", action="store_false")
    parser.set_defaults(source_aware_decoder=True)
    parser.add_argument("--source-aware-pool-size", type=int, default=256)
    parser.add_argument("--source-aware-plan-neighbors", type=int, default=128)
    parser.add_argument("--source-aware-source-neighbors", type=int, default=128)
    parser.add_argument("--source-aware-reference-neighbors", type=int, default=64)
    parser.add_argument("--source-aware-verify-candidates", type=int, default=192)
    parser.add_argument(
        "--disable-instruction-features",
        action="store_true",
        help="Ablation: keep source/target hints but zero goal/edit/constraint features.",
    )
    parser.add_argument("--multimodal-context", choices=MULTIMODAL_CONTEXT_MODES, default="none")
    parser.add_argument(
        "--allow-target-reference",
        action="store_true",
        help="Use target_smiles as reference_smiles when a dataset lacks reference_smiles. This is an oracle-reference setting.",
    )
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--noise-repeats", type=int, default=8)
    parser.add_argument("--torch-epochs", type=int, default=80)
    parser.add_argument("--torch-batch-size", type=int, default=1024)
    parser.add_argument("--torch-hidden-dim", type=int, default=1024)
    parser.add_argument("--torch-layers", type=int, default=6)
    parser.add_argument("--torch-lr", type=float, default=2e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vae-latent-dim", type=int, default=16)
    parser.add_argument("--vae-hidden-dim", type=int, default=512)
    parser.add_argument("--vae-layers", type=int, default=3)
    parser.add_argument("--vae-epochs", type=int, default=60)
    parser.add_argument("--vae-batch-size", type=int, default=1024)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=1e-3)
    parser.add_argument("--source-anchor-weight", type=float, default=0.35)
    parser.add_argument("--source-count-anchor-weight", type=float, default=0.15)
    parser.add_argument("--source-anchor-neighbors", type=int, default=32)
    parser.add_argument("--sklearn-hidden", type=int, nargs=2, default=(160, 160))
    parser.add_argument("--target-anchor", type=float, default=1.0)
    parser.add_argument("--anchor-neighbors", type=int, default=128)
    parser.add_argument("--count-anchor-weight", type=float, default=0.8)
    return parser.parse_args()


def _build_arrays(
    df: pd.DataFrame,
    multimodal_context: str = "none",
    allow_target_reference: bool = False,
    disable_instruction_features: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    conditions = []
    targets = []
    for _, row in df.iterrows():
        condition = condition_from_source_and_spec(row["source_smiles"], row["instruction_spec_json"])
        if disable_instruction_features:
            condition[INSTRUCTION_SPEC_FEATURE_START:INSTRUCTION_SPEC_FEATURE_END] = 0.0
        multimodal = multimodal_context_from_row(
            row,
            mode=multimodal_context,
            allow_target_reference=allow_target_reference,
        )
        if len(multimodal):
            condition = np.concatenate([condition, multimodal])
        conditions.append(condition)
        targets.append(target_table_from_smiles(row["target_smiles"]))
    return np.asarray(conditions, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _fit_diffusion(args, train_y: np.ndarray, train_x: np.ndarray):
    target_start = len(TABLE_COLUMNS)
    if args.latent_vae:
        if args.backend not in {"auto", "torch"}:
            raise ValueError("--latent-vae requires --backend auto or --backend torch.")
        try:
            from .torch_latent_vae import TORCH_AVAILABLE, TorchLatentVAEDiffusion

            if not TORCH_AVAILABLE:
                raise RuntimeError("Requested --latent-vae but PyTorch is not installed.")
            model = TorchLatentVAEDiffusion(
                timesteps=args.timesteps,
                noise_repeats=args.noise_repeats,
                hidden_dim=args.torch_hidden_dim,
                layers=args.torch_layers,
                epochs=args.torch_epochs,
                batch_size=args.torch_batch_size,
                lr=args.torch_lr,
                seed=args.seed,
                device=args.device,
                target_condition_start=target_start,
                target_anchor=args.target_anchor,
                anchor_neighbors=args.anchor_neighbors,
                count_anchor_weight=args.count_anchor_weight,
                vae_latent_dim=args.vae_latent_dim,
                vae_hidden_dim=args.vae_hidden_dim,
                vae_layers=args.vae_layers,
                vae_epochs=args.vae_epochs,
                vae_batch_size=args.vae_batch_size,
                vae_lr=args.vae_lr,
                vae_beta=args.vae_beta,
                source_anchor_weight=args.source_anchor_weight,
                source_count_anchor_weight=args.source_count_anchor_weight,
                source_anchor_neighbors=args.source_anchor_neighbors,
            ).fit(train_y, train_x)
            return model, "torch_latent_vae"
        except Exception:
            raise
    if args.backend in {"auto", "torch"}:
        try:
            from .torch_diffusion import TORCH_AVAILABLE, TorchTabularDiffusion

            if TORCH_AVAILABLE:
                model = TorchTabularDiffusion(
                    timesteps=args.timesteps,
                    noise_repeats=args.noise_repeats,
                    hidden_dim=args.torch_hidden_dim,
                    layers=args.torch_layers,
                    epochs=args.torch_epochs,
                    batch_size=args.torch_batch_size,
                    lr=args.torch_lr,
                    seed=args.seed,
                    device=args.device,
                    target_condition_start=target_start,
                    target_anchor=args.target_anchor,
                    anchor_neighbors=args.anchor_neighbors,
                    count_anchor_weight=args.count_anchor_weight,
                ).fit(train_y, train_x)
                return model, "torch"
            if args.backend == "torch":
                raise RuntimeError("Requested --backend torch but PyTorch is not installed.")
        except Exception:
            if args.backend == "torch":
                raise

    model = TabularDiffusion(
        timesteps=args.timesteps,
        noise_repeats=args.noise_repeats,
        hidden=tuple(args.sklearn_hidden),
        seed=args.seed,
        target_condition_start=target_start,
        target_anchor=args.target_anchor,
        anchor_neighbors=args.anchor_neighbors,
        count_anchor_weight=args.count_anchor_weight,
    ).fit(train_y, train_x)
    return model, "sklearn"


def _sample_edit_plans(diffusion, conditions: np.ndarray, eval_df: pd.DataFrame, samples_per_instruction: int) -> pd.DataFrame:
    rows = []
    for instruction_idx, condition in enumerate(conditions):
        for sample_idx, row in enumerate(diffusion.sample(condition[None, :], n=samples_per_instruction)):
            out = {
                "instruction_id": instruction_idx,
                "pair_id": eval_df.iloc[instruction_idx].get("pair_id", instruction_idx),
                "sample_idx": sample_idx,
            }
            out.update(row)
            rows.append(out)
    return pd.DataFrame(rows)


def _decode_and_attach(
    generated_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    top_k: int,
    dynamic_decoder: bool = False,
    source_index: SourceAwareCandidateIndex | None = None,
    source_aware_config: SourceAwareDecoderConfig | None = None,
) -> pd.DataFrame:
    rows = []
    for _, source in generated_df.iterrows():
        instruction_idx = int(source["instruction_id"])
        instruction = eval_df.iloc[instruction_idx]
        table_row = {col: float(source[col]) for col in TABLE_COLUMNS}
        decode_seed = instruction_idx * 1009 + int(source["sample_idx"])
        candidates = []
        if source_index is not None:
            candidates.extend(
                decode_source_aware(
                    table_row,
                    instruction,
                    source_index,
                    top_k=top_k,
                    seed=decode_seed,
                    config=source_aware_config,
                )
            )
        if len(candidates) < top_k:
            candidates.extend(
                decode_table_row(
                    table_row,
                    top_k=top_k - len(candidates),
                    seed=decode_seed,
                    include_dynamic=dynamic_decoder,
                )
            )
        for rank, candidate in enumerate(_dedupe_candidates(candidates)[:top_k], start=1):
            out = {
                "instruction_id": instruction_idx,
                "pair_id": source["pair_id"],
                "sample_idx": int(source["sample_idx"]),
                "rank": rank,
                "source_smiles": instruction["source_smiles"],
                "target_smiles": instruction["target_smiles"],
                "candidate_smiles": candidate.smiles,
                "instruction_text": instruction["instruction_text"],
                "instruction_spec_json": instruction["instruction_spec_json"],
                "decoder_score": candidate.score,
                "decoder_valid": candidate.valid,
                "candidate_source": candidate.source,
            }
            out.update({f"plan_{key}": value for key, value in table_row.items()})
            out.update({f"actual_{key}": value for key, value in candidate.descriptors.items() if isinstance(value, (int, float))})
            rows.append(out)
    return pd.DataFrame(rows)


def _dedupe_candidates(candidates):
    out = []
    seen = set()
    for candidate in candidates:
        if candidate.smiles in seen:
            continue
        out.append(candidate)
        seen.add(candidate.smiles)
    return out


def _summary(metrics: dict[str, float], args: argparse.Namespace) -> str:
    keys = [
        "backend",
        "train_size",
        "eval_size",
        "n",
        "validity",
        "goal_success_rate",
        "constraint_success_rate",
        "edit_success_rate",
        "overall_instruction_success_rate",
        "overall_success_at_1_by_instruction",
        "overall_success_at_5_by_instruction",
        "overall_success_at_10_by_instruction",
        "sketchmol_local_edit_success_rate",
        "local_edit_success_at_5_by_instruction",
        "similarity_to_source",
        "target_similarity",
        "druglike_rate",
        "novelty",
    ]
    lines = ["PhysTabMol instruction-editing experiment complete", f"dataset={args.dataset}"]
    lines.append(f"multimodal_context={args.multimodal_context}")
    lines.append(f"source_aware_decoder={args.source_aware_decoder}")
    lines.append(f"latent_vae={args.latent_vae}")
    if args.latent_vae:
        lines.append(f"vae_latent_dim={args.vae_latent_dim}")
    for key in keys:
        if key in metrics:
            value = metrics[key]
            lines.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return "\n".join(lines) + "\n"


def _environment() -> dict[str, object]:
    env = {"python": sys.version, "platform": platform.platform()}
    try:
        import rdkit

        env["rdkit"] = rdkit.__version__
    except Exception:
        env["rdkit"] = None
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    except Exception:
        env["torch"] = None
        env["cuda_available"] = False
    return env


if __name__ == "__main__":
    main()
