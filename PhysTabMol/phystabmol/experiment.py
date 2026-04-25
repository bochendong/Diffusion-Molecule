"""Server-oriented PhysTabMol training and evaluation entrypoint."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .contrastive import ContrastiveAligner
from .context import INTENT_DELTAS, InContextConditioner
from .dataset import arrays_from_dataframe, load_experiment_dataframe, train_test_split_df
from .decoder import decode_table_row
from .diffusion import TabularDiffusion
from .evaluate import evaluate_decoded_table
from .features import IMAGE_FEATURE_COLUMNS, extract_image_features
from .io import make_run_dir, save_json, save_text, set_seed
from .schema import TARGET_COLUMNS


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    save_json(vars(args), run_dir / "config.json")

    df = load_experiment_dataframe(
        data_path=args.data,
        smiles_column=args.smiles_column,
        image_column=args.image_column,
        limit=args.limit,
    )
    train_df, test_df = train_test_split_df(df, test_fraction=args.test_fraction, seed=args.seed)
    train_df.to_csv(run_dir / "tables" / "train_table.csv", index=False)
    test_df.to_csv(run_dir / "tables" / "test_table.csv", index=False)

    train_image_x, _, train_condition_x, train_table_y = arrays_from_dataframe(train_df)
    test_image_x, _, test_condition_x, _ = arrays_from_dataframe(test_df if not test_df.empty else train_df)

    aligner = ContrastiveAligner(
        embedding_dim=args.embedding_dim,
        temperature=args.contrastive_temperature,
        lr=args.contrastive_lr,
        epochs=args.contrastive_epochs,
        seed=args.seed,
    ).fit(train_image_x, train_table_y)
    aligner.save(run_dir / "models" / "contrastive_aligner.pkl")

    train_condition = _with_alignment(train_condition_x, aligner.transform_image(train_image_x))
    test_condition = _with_alignment(test_condition_x, aligner.transform_image(test_image_x))
    if args.reference_smiles or args.reference_image or args.intent != "default":
        test_condition = _build_in_context_conditions(test_df if not test_df.empty else train_df, aligner, args)

    diffusion, backend = _fit_diffusion(args, train_table_y, train_condition)
    model_path = run_dir / "models" / ("diffusion.pt" if backend == "torch" else "diffusion.pkl")
    diffusion.save(model_path)

    generated_df = _sample_tables(diffusion, test_condition, args.samples_per_condition)
    generated_df.to_csv(run_dir / "tables" / "generated_table_rows.csv", index=False)

    decoded_df = _decode_generated(generated_df, top_k=args.decode_top_k)
    decoded_df.to_csv(run_dir / "tables" / "decoded_candidates.csv", index=False)

    metrics = evaluate_decoded_table(decoded_df, train_smiles=train_df["smiles"].tolist())
    metrics.update(
        {
            "backend": backend,
            "train_size": float(len(train_df)),
            "test_size": float(len(test_df)),
            "contrastive_train_retrieval_accuracy": aligner.retrieval_accuracy(train_image_x, train_table_y),
            "run_dir": str(run_dir),
        }
    )
    save_json(metrics, run_dir / "metrics.json")
    save_text(_summary(metrics, args), run_dir / "summary.txt")
    save_json(_environment(), run_dir / "environment.json")

    print(_summary(metrics, args))
    print(f"run_dir={run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PhysTabMol on a server.")
    parser.add_argument("--data", type=str, default=None, help="CSV dataset. If omitted, uses built-in smoke-test molecules.")
    parser.add_argument("--smiles-column", type=str, default="smiles")
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default="phystabmol")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    parser.add_argument("--samples-per-condition", type=int, default=16)
    parser.add_argument("--decode-top-k", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--contrastive-epochs", type=int, default=600)
    parser.add_argument("--contrastive-lr", type=float, default=0.04)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--noise-repeats", type=int, default=16)
    parser.add_argument("--torch-epochs", type=int, default=100)
    parser.add_argument("--torch-batch-size", type=int, default=512)
    parser.add_argument("--torch-hidden-dim", type=int, default=384)
    parser.add_argument("--torch-layers", type=int, default=4)
    parser.add_argument("--torch-lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sklearn-hidden", type=int, nargs=2, default=(160, 160))
    parser.add_argument("--reference-image", type=str, default=None)
    parser.add_argument("--reference-smiles", type=str, default=None)
    parser.add_argument("--intent", choices=sorted(INTENT_DELTAS), default="default")
    return parser.parse_args()


def _fit_diffusion(args, train_table_y: np.ndarray, train_condition: np.ndarray):
    backend = args.backend
    if backend in {"auto", "torch"}:
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
                ).fit(train_table_y, train_condition)
                return model, "torch"
            if backend == "torch":
                raise RuntimeError("Requested --backend torch but PyTorch is not available.")
        except Exception:
            if backend == "torch":
                raise

    model = TabularDiffusion(
        timesteps=args.timesteps,
        noise_repeats=args.noise_repeats,
        hidden=tuple(args.sklearn_hidden),
        seed=args.seed,
    ).fit(train_table_y, train_condition)
    return model, "sklearn"


def _sample_tables(diffusion, conditions: np.ndarray, samples_per_condition: int) -> pd.DataFrame:
    rows = []
    for condition_idx, condition in enumerate(conditions):
        samples = diffusion.sample(condition[None, :], n=samples_per_condition)
        for sample_idx, row in enumerate(samples):
            out = {"condition_idx": condition_idx, "sample_idx": sample_idx}
            out.update(row)
            rows.append(out)
    return pd.DataFrame(rows)


def _decode_generated(generated_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    table_cols = [col for col in generated_df.columns if col not in {"condition_idx", "sample_idx"}]
    for _, source in generated_df.iterrows():
        row = {col: float(source[col]) for col in table_cols}
        for rank, candidate in enumerate(decode_table_row(row, top_k=top_k), start=1):
            out = {
                "condition_idx": int(source["condition_idx"]),
                "sample_idx": int(source["sample_idx"]),
                "rank": rank,
                "smiles": candidate.smiles,
                "decoder_score": candidate.score,
                "valid": candidate.valid,
            }
            out.update({f"target_{k}": v for k, v in row.items()})
            out.update({f"actual_{k}": v for k, v in candidate.descriptors.items() if isinstance(v, (int, float))})
            rows.append(out)
    return pd.DataFrame(rows)


def _build_in_context_conditions(df: pd.DataFrame, aligner: ContrastiveAligner, args) -> np.ndarray:
    conditioner = InContextConditioner()
    conditions = []
    reference_features = extract_image_features(args.reference_image) if args.reference_image else None
    for _, row in df.iterrows():
        query_features = {col: float(row[col]) for col in IMAGE_FEATURE_COLUMNS}
        default_targets = {col: float(row[col]) for col in TARGET_COLUMNS}
        base, _ = conditioner.build(
            query_image_features=query_features,
            default_targets=default_targets,
            reference_image_features=reference_features,
            reference_smiles=args.reference_smiles,
            intent=args.intent,
        )
        image_row = base[:, : len(IMAGE_FEATURE_COLUMNS)]
        conditions.append(_with_alignment(base, aligner.transform_image(image_row))[0])
    return np.asarray(conditions, dtype=float)


def _with_alignment(condition_x: np.ndarray, image_embed: np.ndarray) -> np.ndarray:
    return np.concatenate([condition_x, image_embed], axis=1)


def _summary(metrics: dict, args: argparse.Namespace) -> str:
    keys = [
        "backend",
        "train_size",
        "test_size",
        "n",
        "validity",
        "uniqueness",
        "novelty",
        "druglike_rate",
        "mean_pairwise_tanimoto",
        "MW_mae",
        "LogP_mae",
        "QED_mae",
        "TPSA_mae",
    ]
    lines = ["PhysTabMol experiment complete", f"intent={args.intent}", f"reference_smiles={args.reference_smiles}"]
    for key in keys:
        if key in metrics:
            value = metrics[key]
            lines.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return "\n".join(lines) + "\n"


def _environment() -> dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        env["torch"] = None
        env["cuda_available"] = False
    try:
        import rdkit

        env["rdkit"] = rdkit.__version__
    except Exception:
        env["rdkit"] = None
    return env


if __name__ == "__main__":
    main()

