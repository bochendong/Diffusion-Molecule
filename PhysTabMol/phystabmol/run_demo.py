"""End-to-end PhysTabMol proof-of-concept run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .contrastive import ContrastiveAligner
from .context import INTENT_DELTAS, InContextConditioner, features_from_image_or_default
from .data import STARTER_SMILES, build_demo_dataframe, split_arrays
from .decoder import decode_table_row
from .diffusion import TabularDiffusion
from .evaluate import evaluate_smiles
from .features import IMAGE_FEATURE_COLUMNS, extract_image_features
from .schema import TARGET_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PhysTabMol proof-of-concept pipeline.")
    parser.add_argument("--image", type=str, default=None, help="Optional image used as the generation condition.")
    parser.add_argument("--reference-image", type=str, default=None, help="Optional UniVideo-style reference image.")
    parser.add_argument("--reference-smiles", type=str, default=None, help="Optional UniVideo-style reference molecule.")
    parser.add_argument(
        "--intent",
        type=str,
        default="default",
        choices=sorted(INTENT_DELTAS),
        help="Lightweight text instruction for in-context molecular editing.",
    )
    parser.add_argument("--samples", type=int, default=8, help="Number of tabular diffusion samples to decode.")
    parser.add_argument("--out", type=str, default="outputs/phystabmol_demo.csv", help="CSV path for decoded candidates.")
    args = parser.parse_args()

    df = build_demo_dataframe()
    image_x, target_x, condition_x, table_y = split_arrays(df)

    aligner = ContrastiveAligner(embedding_dim=8, epochs=300).fit(image_x, table_y)
    image_embed = aligner.transform_image(image_x)
    condition_with_alignment = _with_alignment(condition_x, image_embed)

    diffusion = TabularDiffusion(timesteps=35, noise_repeats=18).fit(table_y, condition_with_alignment)

    condition, readable_target = _condition_from_input(
        args.image,
        df,
        aligner,
        reference_image=args.reference_image,
        reference_smiles=args.reference_smiles,
        intent=args.intent,
    )
    generated_rows = diffusion.sample(condition, n=args.samples)

    decoded_rows = []
    for sample_idx, row in enumerate(generated_rows):
        for rank, candidate in enumerate(decode_table_row(row, top_k=3), start=1):
            out = {
                "sample_idx": sample_idx,
                "rank": rank,
                "smiles": candidate.smiles,
                "decoder_score": candidate.score,
                "valid": candidate.valid,
            }
            out.update({f"target_{k}": v for k, v in row.items()})
            out.update({f"actual_{k}": v for k, v in candidate.descriptors.items() if isinstance(v, (int, float))})
            decoded_rows.append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decoded_rows).to_csv(out_path, index=False)

    smiles = [row["smiles"] for row in decoded_rows]
    target = readable_target or {col: generated_rows[0][col] for col in TARGET_COLUMNS}
    metrics = evaluate_smiles(smiles, train_smiles=STARTER_SMILES, target=target)
    print("PhysTabMol demo complete")
    print(f"contrastive_retrieval_accuracy={aligner.retrieval_accuracy(image_x, table_y):.3f}")
    print(f"in_context_intent={args.intent}")
    if args.reference_image:
        print(f"reference_image={args.reference_image}")
    if args.reference_smiles:
        print(f"reference_smiles={args.reference_smiles}")
    print(f"decoded_candidates={len(decoded_rows)}")
    print(f"output_csv={out_path}")
    for key, value in metrics.items():
        print(f"{key}={value:.3f}")


def _with_alignment(condition_x, image_embed):
    import numpy as np

    return np.concatenate([condition_x, image_embed], axis=1)


def _condition_from_input(
    image_path: str | None,
    df,
    aligner,
    reference_image: str | None = None,
    reference_smiles: str | None = None,
    intent: str = "default",
):
    default_image_source = df.iloc[0]
    query_features = features_from_image_or_default(image_path, default_image_source)
    reference_features = extract_image_features(reference_image) if reference_image else None
    default_targets = {col: float(df[col].median()) for col in TARGET_COLUMNS}
    conditioner = InContextConditioner()
    base, targets = conditioner.build(
        query_image_features=query_features,
        default_targets=default_targets,
        reference_image_features=reference_features,
        reference_smiles=reference_smiles,
        intent=intent,
    )
    image_row = base[:, : len(IMAGE_FEATURE_COLUMNS)]
    return _with_alignment(base, aligner.transform_image(image_row)), targets


if __name__ == "__main__":
    main()
