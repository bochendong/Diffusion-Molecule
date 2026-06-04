"""End-to-end image-to-structure baseline for SketchMol-style sketches.

This module intentionally avoids MolScribe/OCR. It uses paired molecular sketch
images as input and trains a decoder to emit a machine-readable structure
directly, then renders the prediction with RDKit for closed-loop checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sketch_smiles.phase5c_image_smiles_decoder import run_image_conditioned_smiles_decoder


PHASE = "sketchmol_end2end_image_to_structure"


def run_image_to_structure(
    pair_dir: str | Path,
    output_dir: str | Path = "SketchMolEnd2End/outputs/runs/image_to_structure_seed7",
    train_fraction: float = 0.8,
    seed: int = 7,
    limit: int | None = None,
    max_length: int = 128,
    hidden_dim: int = 384,
    embedding_dim: int = 96,
    encoder_channels: int = 64,
    image_token_grid: int = 4,
    fingerprint_bits: int = 0,
    fingerprint_loss_weight: float = 0.0,
    rerank_mode: str = "beam",
    transformer_layers: int = 4,
    attention_heads: int = 8,
    dropout: float = 0.1,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    samples_per_condition: int = 8,
    temperature: float = 0.9,
    sample_top_k: int = 16,
    tokenization: str = "smiles_token",
    decoding: str = "beam",
    beam_size: int = 8,
    length_penalty: float = 0.0,
    image_size: int = 128,
    sample_count: int = 64,
    contact_sheet_cols: int = 8,
    contact_thumb_size: int = 144,
    device: str = "auto",
) -> dict[str, Any]:
    """Train/evaluate the no-OCR image-to-structure baseline."""

    output_path = Path(output_dir)
    metrics = run_image_conditioned_smiles_decoder(
        pair_dir=pair_dir,
        output_dir=output_path,
        train_fraction=train_fraction,
        seed=seed,
        limit=limit,
        max_length=max_length,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        encoder_channels=encoder_channels,
        image_token_grid=image_token_grid,
        fingerprint_bits=fingerprint_bits,
        fingerprint_loss_weight=fingerprint_loss_weight,
        rerank_mode=rerank_mode,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
        tokenization=tokenization,
        decoding=decoding,
        beam_size=beam_size,
        length_penalty=length_penalty,
        image_size=image_size,
        sample_count=sample_count,
        contact_sheet_cols=contact_sheet_cols,
        contact_thumb_size=contact_thumb_size,
        device=device,
    )
    metrics["phase"] = PHASE
    metrics["ocr_used"] = False
    metrics["structure_target"] = "canonical_smiles"
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "phase": PHASE,
        "ocr_used": False,
        "research_question": (
            "Can a molecular sketch image be mapped directly to a machine-readable molecule "
            "without a MolScribe/OCR post-processing stage?"
        ),
        "pair_dir": str(pair_dir),
        "output_dir": str(output_path),
        "tokenization": tokenization,
        "decoding": decoding,
        "closed_loop_check": "prediction -> RDKit render -> image metrics against input sketch",
    }
    (output_path / "end2end_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-OCR SketchMol image-to-structure baseline.")
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", default="SketchMolEnd2End/outputs/runs/image_to_structure_seed7")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--encoder-channels", type=int, default=64)
    parser.add_argument("--image-token-grid", type=int, default=4)
    parser.add_argument("--fingerprint-bits", type=int, default=0)
    parser.add_argument("--fingerprint-loss-weight", type=float, default=0.0)
    parser.add_argument("--rerank-mode", default="beam")
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--tokenization", default="smiles_token", choices=["char", "smiles", "smiles_token"])
    parser.add_argument("--decoding", default="beam", choices=["sample", "beam"])
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=144)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = run_image_to_structure(
        pair_dir=args.pair_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        seed=args.seed,
        limit=args.limit,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        encoder_channels=args.encoder_channels,
        image_token_grid=args.image_token_grid,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_loss_weight=args.fingerprint_loss_weight,
        rerank_mode=args.rerank_mode,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        samples_per_condition=args.samples_per_condition,
        temperature=args.temperature,
        sample_top_k=args.sample_top_k,
        tokenization=args.tokenization,
        decoding=args.decoding,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        image_size=args.image_size,
        sample_count=args.sample_count,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_thumb_size=args.contact_thumb_size,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
