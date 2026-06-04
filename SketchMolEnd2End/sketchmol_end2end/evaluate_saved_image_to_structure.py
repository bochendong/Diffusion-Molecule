"""Eval-only candidate expansion for a saved SketchMolEnd2End model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sketch_smiles.audit_pairs import _load_pillow, _load_rdkit
from sketch_smiles.phase5a0_oracle_baseline import _sample_rows, _write_oracle_contact_sheet, _write_rows
from sketch_smiles.phase5a1_learned_smiles_decoder import _load_numpy, _load_torch, _load_torch_checkpoint, _make_fingerprint_fn, _set_rdkit_error_logging, _set_seeds
from sketch_smiles.phase5c_image_smiles_decoder import (
    ImageConditionedSmilesTransformer,
    _evaluate_image_smiles_model,
    _normalize_decoding,
    _normalize_image_rerank_mode,
    _normalize_tokenization,
    _prepare_image_examples,
    _read_rows,
    _resolve_device,
    _summarize_image_smiles_decoder,
)


def evaluate_saved_image_to_structure(
    run_dir: str | Path,
    output_dir: str | Path,
    pair_dir: str | Path | None = None,
    decoding: str = "beam",
    beam_size: int = 32,
    length_penalty: float = 0.0,
    rerank_mode: str = "beam",
    samples_per_condition: int = 32,
    temperature: float = 0.9,
    sample_top_k: int = 16,
    image_size: int | None = None,
    sample_count: int = 64,
    contact_sheet_cols: int = 8,
    contact_thumb_size: int = 144,
    eval_offset: int = 0,
    eval_limit: int | None = None,
    seed: int = 7,
    device: str = "auto",
) -> dict[str, Any]:
    """Load a trained image-to-structure model and regenerate eval candidates."""

    source_dir = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = source_dir / "model.pt"
    vocab_path = source_dir / "vocab.json"
    train_pairs_path = source_dir / "train_pairs.csv"
    eval_pairs_path = source_dir / "eval_pairs.csv"
    history_path = source_dir / "train_history.json"
    config_path = source_dir / "run_config.json"

    for required_path in (model_path, vocab_path, train_pairs_path, eval_pairs_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing saved run artifact: {required_path}")

    rdkit = _load_rdkit()
    pillow = _load_pillow()
    if not rdkit:
        raise RuntimeError("RDKit is required for saved SketchMolEnd2End evaluation.")
    if not pillow:
        raise RuntimeError("Pillow is required for saved SketchMolEnd2End evaluation.")
    _set_rdkit_error_logging(enabled=False)

    torch = _load_torch()
    np = _load_numpy()
    _set_seeds(seed, torch=torch, np=np)
    resolved_device = _resolve_device(device, torch)

    checkpoint = _load_torch_checkpoint(model_path, torch=torch, map_location=resolved_device)
    checkpoint_config = checkpoint.get("config", {})
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    stoi = {str(token): int(index) for token, index in vocab["stoi"].items()}
    itos = [str(token) for token in vocab["itos"]]

    pair_path = Path(pair_dir or run_config.get("pair_dir") or checkpoint_config.get("pair_dir") or "")
    if not pair_path:
        raise ValueError("pair_dir must be provided or present in run_config.json.")

    fingerprint_bits = int(checkpoint_config.get("fingerprint_bits", run_config.get("fingerprint_bits", 0)))
    hidden_dim = int(checkpoint_config.get("hidden_dim", run_config.get("hidden_dim", 384)))
    embedding_dim = int(checkpoint_config.get("embedding_dim", run_config.get("embedding_dim", 96)))
    encoder_channels = int(checkpoint_config.get("encoder_channels", run_config.get("encoder_channels", 64)))
    image_token_grid = int(checkpoint_config.get("image_token_grid", run_config.get("image_token_grid", 4)))
    max_length = int(checkpoint_config.get("max_length", run_config.get("max_length", 128)))
    transformer_layers = int(checkpoint_config.get("transformer_layers", run_config.get("transformer_layers", 4)))
    attention_heads = int(checkpoint_config.get("attention_heads", run_config.get("attention_heads", 8)))
    dropout = float(checkpoint_config.get("dropout", run_config.get("dropout", 0.1)))
    image_size = int(image_size or checkpoint_config.get("image_size", run_config.get("image_size", 128)))
    train_fraction = float(run_config.get("train_fraction", 0.8))
    tokenization = _normalize_tokenization(str(run_config.get("tokenization", "smiles_token")))
    decoding = _normalize_decoding(decoding)
    rerank_mode = _normalize_image_rerank_mode(rerank_mode)

    fingerprint_fn = _make_fingerprint_fn(rdkit, np=np, fingerprint_bits=fingerprint_bits) if fingerprint_bits > 0 else None
    if rerank_mode == "predicted_fingerprint" and fingerprint_fn is None:
        raise ValueError("predicted_fingerprint reranking requires a saved model with fingerprint_bits > 0.")

    train_rows = _read_rows(train_pairs_path)
    eval_rows = _read_rows(eval_pairs_path)
    if eval_offset > 0:
        eval_rows = eval_rows[int(eval_offset) :]
    if eval_limit is not None:
        eval_rows = eval_rows[: int(eval_limit)]
    _write_rows(output_path / "train_pairs.csv", train_rows)
    _write_rows(output_path / "eval_pairs.csv", eval_rows)
    (output_path / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_examples = _prepare_image_examples(
        rows=train_rows,
        pair_dir=pair_path,
        stoi=stoi,
        max_length=max_length,
        np=np,
        pillow=pillow,
        image_size=image_size,
        tokenization=tokenization,
        fingerprint_fn=fingerprint_fn,
        allow_unknown=True,
    )
    eval_examples = _prepare_image_examples(
        rows=eval_rows,
        pair_dir=pair_path,
        stoi=stoi,
        max_length=max_length,
        np=np,
        pillow=pillow,
        image_size=image_size,
        tokenization=tokenization,
        fingerprint_fn=fingerprint_fn,
        allow_unknown=True,
    )
    if not eval_examples:
        raise RuntimeError("No eval examples available for saved SketchMolEnd2End evaluation.")

    model = ImageConditionedSmilesTransformer(
        vocab_size=len(itos),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_channels=encoder_channels,
        image_token_grid=image_token_grid,
        fingerprint_bits=fingerprint_bits,
        pad_idx=int(checkpoint_config.get("pad_idx", stoi["<pad>"])),
        max_length=max_length,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        dropout=dropout,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])

    generated_image_dir = output_path / "generated_images"
    resized_target_image_dir = output_path / "target_images_resized"
    generated_image_dir.mkdir(parents=True, exist_ok=True)
    resized_target_image_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = _evaluate_image_smiles_model(
        model=model,
        eval_examples=eval_examples,
        stoi=stoi,
        itos=itos,
        generated_image_dir=generated_image_dir,
        resized_target_image_dir=resized_target_image_dir,
        rdkit=rdkit,
        pillow=pillow,
        torch=torch,
        device=resolved_device,
        max_length=max_length,
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
        decoding=decoding,
        beam_size=beam_size,
        length_penalty=length_penalty,
        rerank_mode=rerank_mode,
        fingerprint_fn=fingerprint_fn,
        np=np,
        image_size=image_size,
    )

    predictions_path = output_path / "predictions.csv"
    _write_rows(predictions_path, prediction_rows)
    sample_rows = _sample_rows(prediction_rows, sample_count=sample_count, seed=seed)
    sample_predictions_path = output_path / "sample_predictions.csv"
    _write_rows(sample_predictions_path, sample_rows)
    contact_sheet_path = _write_oracle_contact_sheet(
        sample_rows=sample_rows,
        pillow=pillow,
        cols=contact_sheet_cols,
        thumb_size=contact_thumb_size,
        output_path=output_path / "sample_contact_sheet.png",
    )
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    metrics = _summarize_image_smiles_decoder(
        prediction_rows=prediction_rows,
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_examples=train_examples,
        eval_examples=eval_examples,
        history=history,
        pair_dir=pair_path,
        output_dir=output_path,
        predictions_path=predictions_path,
        sample_predictions_path=sample_predictions_path,
        contact_sheet_path=contact_sheet_path,
        model_path=model_path,
        vocab_path=vocab_path,
        train_fraction=train_fraction,
        seed=seed,
        max_length=max_length,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        encoder_channels=encoder_channels,
        image_token_grid=image_token_grid,
        fingerprint_bits=fingerprint_bits,
        fingerprint_loss_weight=float(run_config.get("fingerprint_loss_weight", 0.0)),
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        dropout=dropout,
        epochs=int(run_config.get("epochs", 0)),
        batch_size=int(run_config.get("batch_size", 0)),
        learning_rate=float(run_config.get("learning_rate", 0.0)),
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
        tokenization=tokenization,
        decoding=decoding,
        beam_size=beam_size,
        length_penalty=length_penalty,
        rerank_mode=rerank_mode,
        image_size=image_size,
        device=str(resolved_device),
    )
    metrics["phase"] = "sketchmol_end2end_saved_image_to_structure_eval"
    metrics["eval_only"] = True
    metrics["source_run_dir"] = str(source_dir)
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "sketchmol_end2end_saved_image_to_structure_eval",
                "research_question": "Does expanding saved image-to-SMILES candidates improve the render-select upper bound?",
                "eval_only": True,
                "source_run_dir": str(source_dir),
                "pair_dir": str(pair_path),
                "output_dir": str(output_path),
                "model": str(model_path),
                "vocab": str(vocab_path),
                "decoding": decoding,
                "beam_size": beam_size,
                "length_penalty": length_penalty,
                "rerank_mode": rerank_mode,
                "samples_per_condition": samples_per_condition,
                "temperature": temperature,
                "sample_top_k": sample_top_k,
                "image_size": image_size,
                "eval_offset": eval_offset,
                "eval_limit": eval_limit,
                "seed": seed,
                "device": str(resolved_device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved SketchMolEnd2End image-to-structure model.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-dir", default=None)
    parser.add_argument("--decoding", default="beam", choices=["beam", "sample"])
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--rerank-mode", default="beam")
    parser.add_argument("--samples-per-condition", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=144)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = evaluate_saved_image_to_structure(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        pair_dir=args.pair_dir,
        decoding=args.decoding,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        rerank_mode=args.rerank_mode,
        samples_per_condition=args.samples_per_condition,
        temperature=args.temperature,
        sample_top_k=args.sample_top_k,
        image_size=args.image_size,
        sample_count=args.sample_count,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_thumb_size=args.contact_thumb_size,
        eval_offset=args.eval_offset,
        eval_limit=args.eval_limit,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
