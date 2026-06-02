"""Route B: shared diffusion backbone with SMILES and image outputs."""

from __future__ import annotations

import argparse
import json
from typing import Any

from sketchmol_token_diffusion.masked_token_diffusion import build_arg_parser, run_masked_token_diffusion


def run_joint_diffusion(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("image_loss_weight", 1.0)
    kwargs.setdefault("image_foreground_weight", 8.0)
    kwargs.setdefault("route_name", "sketchmol_joint_diffusion_image_smiles")
    return run_masked_token_diffusion(**kwargs)


def main() -> None:
    parser = build_arg_parser("Run Route B joint image+SMILES masked diffusion.")
    parser.set_defaults(
        image_loss_weight=1.0,
        image_foreground_weight=8.0,
        route_name="sketchmol_joint_diffusion_image_smiles",
    )
    args = parser.parse_args()
    metrics = run_joint_diffusion(
        pair_dir=args.pair_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        seed=args.seed,
        limit=args.limit,
        fingerprint_bits=args.fingerprint_bits,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        min_mask_prob=args.min_mask_prob,
        max_mask_prob=args.max_mask_prob,
        samples_per_condition=args.samples_per_condition,
        temperature=args.temperature,
        sample_top_k=args.sample_top_k,
        rerank_mode=args.rerank_mode,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        condition_tokens=args.condition_tokens,
        dropout=args.dropout,
        tokenization=args.tokenization,
        latent_dim=args.latent_dim,
        image_loss_weight=args.image_loss_weight,
        image_foreground_weight=args.image_foreground_weight,
        clip_loss_weight=args.clip_loss_weight,
        clip_temperature=args.clip_temperature,
        decode_length_mode=args.decode_length_mode,
        min_decode_tokens=args.min_decode_tokens,
        image_size=args.image_size,
        sample_count=args.sample_count,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_thumb_size=args.contact_thumb_size,
        device=args.device,
        route_name=args.route_name,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
