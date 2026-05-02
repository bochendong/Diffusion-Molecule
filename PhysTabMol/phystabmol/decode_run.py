"""Decode and evaluate an existing PhysTabMol run.

This is useful when a Slurm job finishes table generation but hits the time
limit during molecular decoding/evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluate import evaluate_decoded_table
from .experiment import _decode_generated
from .geometry3d import add_3d_metrics
from .io import save_json, save_text


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tables_dir = run_dir / "tables"
    generated_path = tables_dir / "generated_table_rows.csv"
    train_path = tables_dir / "train_table.csv"
    if not generated_path.exists():
        raise FileNotFoundError(f"Missing generated rows: {generated_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train table: {train_path}")

    generated_df = pd.read_csv(generated_path)
    generated_df = _select_generated_rows(
        generated_df,
        max_conditions=args.max_conditions,
        samples_per_condition=args.samples_per_condition,
        max_rows=args.max_rows,
    )
    decoded_df = _decode_generated(
        generated_df,
        top_k=args.decode_top_k,
        dynamic_decoder=args.dynamic_decoder,
    )
    if args.enable_3d:
        sdf_path = tables_dir / "decoded_candidates_3d.sdf" if args.save_3d_sdf else None
        decoded_df = add_3d_metrics(decoded_df, sdf_path=sdf_path, max_sdf=args.max_3d_sdf)

    output_path = tables_dir / args.output_name
    decoded_df.to_csv(output_path, index=False)

    train_df = pd.read_csv(train_path)
    metrics = evaluate_decoded_table(decoded_df, train_smiles=train_df["smiles"].tolist())
    metrics.update(
        {
            "postprocess_run_dir": str(run_dir),
            "postprocess_generated_rows": float(len(generated_df)),
            "postprocess_decode_top_k": float(args.decode_top_k),
            "postprocess_dynamic_decoder": bool(args.dynamic_decoder),
        }
    )
    if args.enable_3d and "3d_embed_success" in decoded_df:
        metrics["3d_embed_success_rate"] = float(decoded_df["3d_embed_success"].mean())
    save_json(metrics, run_dir / "metrics.json")
    save_text(_summary(metrics, output_path), run_dir / "summary.txt")
    save_json(vars(args), run_dir / "postprocess_config.json")

    print(_summary(metrics, output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode/evaluate an existing PhysTabMol generated table.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-conditions", type=int, default=5000, help="Use 0 to decode all conditions.")
    parser.add_argument("--samples-per-condition", type=int, default=8, help="Use 0 to keep all samples.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional hard cap after condition/sample filtering.")
    parser.add_argument("--decode-top-k", type=int, default=2)
    parser.add_argument("--dynamic-decoder", action="store_true")
    parser.add_argument("--output-name", default="decoded_candidates.csv")
    parser.add_argument("--enable-3d", action="store_true")
    parser.add_argument("--save-3d-sdf", action="store_true")
    parser.add_argument("--max-3d-sdf", type=int, default=500)
    return parser.parse_args()


def _select_generated_rows(
    generated_df: pd.DataFrame,
    max_conditions: int,
    samples_per_condition: int,
    max_rows: int,
) -> pd.DataFrame:
    out = generated_df
    if max_conditions and "condition_idx" in out:
        keep_conditions = sorted(out["condition_idx"].unique())[:max_conditions]
        out = out[out["condition_idx"].isin(keep_conditions)]
    if samples_per_condition and "sample_idx" in out:
        out = out[out["sample_idx"] < samples_per_condition]
    if max_rows and len(out) > max_rows:
        out = out.head(max_rows)
    return out.reset_index(drop=True)


def _summary(metrics: dict, output_path: Path) -> str:
    keys = [
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
    lines = ["PhysTabMol postprocess complete", f"decoded_path={output_path}"]
    for key in keys:
        if key in metrics:
            value = metrics[key]
            lines.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
