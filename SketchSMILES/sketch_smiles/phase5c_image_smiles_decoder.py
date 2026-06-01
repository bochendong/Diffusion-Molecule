"""Phase 5C image-conditioned SMILES decoder.

This experiment removes the oracle molecular fingerprint condition from Phase
5A and conditions the SMILES decoder directly on the paired molecular sketch.
The top generated SMILES is rendered back to an image so the predicted string
and visual output can be checked together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from .audit_pairs import _load_pillow, _load_rdkit, _resolve_image_path
from .phase5a0_oracle_baseline import _fraction, _image_pair_metrics, _render_smiles, _sample_rows, _split_rows, _write_oracle_contact_sheet, _write_rows
from .phase5a1_learned_smiles_decoder import (
    BOS,
    EOS,
    PAD,
    _beam_search_smiles,
    _build_vocab,
    _canonical_candidate_list,
    _encode_tokens,
    _load_numpy,
    _load_torch,
    _normalize_decoding,
    _normalize_tokenization,
    _sample_smiles,
    _scaffold_match,
    _set_rdkit_error_logging,
    _set_seeds,
    _tanimoto,
    _tokenize_smiles,
)
from .phase5b_joint_decoder import _load_ink_tensor, _save_ink_image


def run_image_conditioned_smiles_decoder(
    pair_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase5c_image_smiles_decoder",
    train_fraction: float = 0.8,
    seed: int = 7,
    limit: int | None = None,
    max_length: int = 128,
    hidden_dim: int = 384,
    embedding_dim: int = 96,
    encoder_channels: int = 64,
    image_token_grid: int = 4,
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
    pair_dir = Path(pair_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pair_dir / "pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing paired manifest: {pairs_path}")

    rdkit = _load_rdkit()
    pillow = _load_pillow()
    if not rdkit:
        raise RuntimeError("RDKit is required for Phase 5C rendering and molecular metrics.")
    if not pillow:
        raise RuntimeError("Pillow is required for Phase 5C image-conditioned examples.")
    _set_rdkit_error_logging(enabled=False)

    torch = _load_torch()
    np = _load_numpy()
    _set_seeds(seed, torch=torch, np=np)

    rows = _read_rows(pairs_path)
    if limit is not None:
        rows = rows[: int(limit)]
    rows = [row for row in rows if row.get("valid", "True") == "True" and (row.get("canonical_smiles") or row.get("input_smiles"))]
    train_rows, eval_rows = _split_rows(rows, train_fraction=train_fraction, seed=seed)
    _write_rows(output_dir / "train_pairs.csv", train_rows)
    _write_rows(output_dir / "eval_pairs.csv", eval_rows)

    tokenization = _normalize_tokenization(tokenization)
    decoding = _normalize_decoding(decoding)
    stoi, itos = _build_vocab(train_rows, tokenization=tokenization)
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps({"stoi": stoi, "itos": itos}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_examples = _prepare_image_examples(
        rows=train_rows,
        pair_dir=pair_dir,
        stoi=stoi,
        max_length=max_length,
        np=np,
        pillow=pillow,
        image_size=image_size,
        tokenization=tokenization,
    )
    eval_examples = _prepare_image_examples(
        rows=eval_rows,
        pair_dir=pair_dir,
        stoi=stoi,
        max_length=max_length,
        np=np,
        pillow=pillow,
        image_size=image_size,
        tokenization=tokenization,
        allow_unknown=True,
    )
    if not train_examples:
        raise RuntimeError("No train examples available for Phase 5C.")
    if not eval_examples:
        raise RuntimeError("No eval examples available for Phase 5C.")

    resolved_device = _resolve_device(device, torch)
    model = ImageConditionedSmilesTransformer(
        vocab_size=len(itos),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_channels=encoder_channels,
        image_token_grid=image_token_grid,
        pad_idx=stoi[PAD],
        max_length=max_length,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = _train_image_smiles_model(
        model=model,
        examples=train_examples,
        optimizer=optimizer,
        torch=torch,
        np=np,
        device=resolved_device,
        batch_size=batch_size,
        epochs=epochs,
        pad_idx=stoi[PAD],
        seed=seed,
    )

    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": len(itos),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "encoder_channels": encoder_channels,
                "image_token_grid": image_token_grid,
                "pad_idx": stoi[PAD],
                "max_length": max_length,
                "transformer_layers": transformer_layers,
                "attention_heads": attention_heads,
                "dropout": dropout,
                "image_size": image_size,
            },
        },
        model_path,
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generated_image_dir = output_dir / "generated_images"
    resized_target_image_dir = output_dir / "target_images_resized"
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
        image_size=image_size,
    )

    predictions_path = output_dir / "predictions.csv"
    _write_rows(predictions_path, prediction_rows)
    sample_rows = _sample_rows(prediction_rows, sample_count=sample_count, seed=seed)
    sample_predictions_path = output_dir / "sample_predictions.csv"
    _write_rows(sample_predictions_path, sample_rows)
    contact_sheet_path = _write_oracle_contact_sheet(
        sample_rows=sample_rows,
        pillow=pillow,
        cols=contact_sheet_cols,
        thumb_size=contact_thumb_size,
        output_path=output_dir / "sample_contact_sheet.png",
    )

    metrics = _summarize_image_smiles_decoder(
        prediction_rows=prediction_rows,
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_examples=train_examples,
        eval_examples=eval_examples,
        history=history,
        pair_dir=pair_dir,
        output_dir=output_dir,
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
        device=str(resolved_device),
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "phase5c_image_conditioned_smiles_decoder",
                "research_question": "Can a molecular sketch image condition a direct SMILES decoder well enough to bypass image-to-OCR post-processing?",
                "pair_dir": str(pair_dir),
                "pairs_csv": str(pairs_path),
                "output_dir": str(output_dir),
                "train_fraction": train_fraction,
                "seed": seed,
                "limit": limit,
                "max_length": max_length,
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "encoder_channels": encoder_channels,
                "image_token_grid": image_token_grid,
                "transformer_layers": transformer_layers,
                "attention_heads": attention_heads,
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "samples_per_condition": samples_per_condition,
                "temperature": temperature,
                "sample_top_k": sample_top_k,
                "tokenization": tokenization,
                "decoding": decoding,
                "beam_size": beam_size,
                "length_penalty": length_penalty,
                "image_size": image_size,
                "device": str(resolved_device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


class ImageConditionedSmilesTransformer:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _load_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                embedding_dim: int,
                hidden_dim: int,
                encoder_channels: int,
                image_token_grid: int,
                pad_idx: int,
                max_length: int,
                transformer_layers: int,
                attention_heads: int,
                dropout: float,
            ) -> None:
                super().__init__()
                if hidden_dim % attention_heads != 0:
                    raise ValueError(f"hidden_dim={hidden_dim} must be divisible by attention_heads={attention_heads}.")
                self.pad_idx = pad_idx
                self.max_length = max_length
                c1 = int(encoder_channels)
                c2 = max(c1 * 2, hidden_dim // 2)
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(1, c1, kernel_size=5, stride=2, padding=2),
                    nn.GELU(),
                    nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(c2, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((int(image_token_grid), int(image_token_grid))),
                )
                self.image_norm = nn.LayerNorm(hidden_dim)
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
                self.input_projection = nn.Linear(embedding_dim, hidden_dim)
                self.position = nn.Embedding(max_length, hidden_dim)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=attention_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)
                self.out = nn.Linear(hidden_dim, vocab_size)

            def _memory(self, images: Any) -> Any:
                encoded = self.image_encoder(images)
                tokens = encoded.flatten(2).transpose(1, 2)
                return self.image_norm(tokens)

            def forward(self, images: Any, input_ids: Any) -> Any:
                batch, length = input_ids.shape
                if length > self.max_length:
                    input_ids = input_ids[:, -self.max_length :]
                    length = self.max_length
                positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
                embeddings = self.input_projection(self.embedding(input_ids)) + self.position(positions)
                causal_mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=input_ids.device), diagonal=1)
                padding_mask = input_ids.eq(self.pad_idx)
                decoded = self.decoder(
                    tgt=embeddings,
                    memory=self._memory(images),
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=padding_mask,
                )
                return self.out(decoded)

            def decode_prefix_logits(self, feature_tensor: Any, ids: list[int]) -> Any:
                prefix = ids[-self.max_length :]
                input_ids = torch.tensor([prefix], dtype=torch.long, device=feature_tensor.device)
                return self.forward(feature_tensor, input_ids)[:, -1, :]

        return _Model(*args, **kwargs)


def _train_image_smiles_model(
    model: Any,
    examples: list[dict[str, Any]],
    optimizer: Any,
    torch: Any,
    np: Any,
    device: Any,
    batch_size: int,
    epochs: int,
    pad_idx: int,
    seed: int,
) -> list[dict[str, float]]:
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    history: list[dict[str, float]] = []
    rng = random.Random(seed)
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for start in range(0, len(order), int(batch_size)):
            batch = [examples[idx] for idx in order[start : start + int(batch_size)]]
            images = torch.as_tensor(np.stack([row["image"] for row in batch]), dtype=torch.float32, device=device)
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            target_ids = torch.tensor([row["target_ids"] for row in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, input_ids)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            token_count = int((target_ids != pad_idx).sum().item())
            total_loss += float(loss.item()) * max(1, token_count)
            total_tokens += token_count
        mean_loss = total_loss / max(1, total_tokens)
        history.append({"epoch": float(epoch), "train_token_loss": float(mean_loss), "train_token_ppl": float(math.exp(min(20.0, mean_loss)))})
        print(f"  epoch={epoch} train_token_loss={mean_loss:.4f} train_token_ppl={math.exp(min(20.0, mean_loss)):.3f}", flush=True)
    return history


def _evaluate_image_smiles_model(
    model: Any,
    eval_examples: list[dict[str, Any]],
    stoi: dict[str, int],
    itos: list[str],
    generated_image_dir: Path,
    resized_target_image_dir: Path,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
    torch: Any,
    device: Any,
    max_length: int,
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
    decoding: str,
    beam_size: int,
    length_penalty: float,
    image_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in eval_examples:
            if decoding == "beam":
                raw_samples = _beam_search_smiles(
                    model=model,
                    feature=example["image"],
                    stoi=stoi,
                    itos=itos,
                    torch=torch,
                    device=device,
                    max_length=max_length,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                raw_samples = _sample_smiles(
                    model=model,
                    feature=example["image"],
                    stoi=stoi,
                    itos=itos,
                    torch=torch,
                    device=device,
                    max_length=max_length,
                    samples_per_condition=samples_per_condition,
                    temperature=temperature,
                    sample_top_k=sample_top_k,
                )
            candidate_smiles = _canonical_candidate_list(raw_samples, rdkit)
            target_smiles = example["smiles"]
            top1_smiles = candidate_smiles[0] if candidate_smiles else ""
            target_mol = rdkit["Chem"].MolFromSmiles(target_smiles)
            top1_mol = rdkit["Chem"].MolFromSmiles(top1_smiles) if top1_smiles else None
            top1_tanimoto = _tanimoto(target_mol, top1_mol, rdkit)
            best_tanimoto = max((_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidate_smiles), default=0.0)
            generated_image_path = generated_image_dir / f"{example['pair_id']}.png"
            target_image_path = resized_target_image_dir / f"{example['pair_id']}.png"
            _save_ink_image(example["image"], target_image_path, pillow)
            render_error = _render_smiles(top1_smiles, generated_image_path, image_size=image_size, rdkit=rdkit)
            image_metrics = _image_pair_metrics(target_image_path, generated_image_path, pillow)
            scaffold_match = _scaffold_match(target_smiles, top1_smiles, rdkit)
            rows.append(
                {
                    "pair_id": example["pair_id"],
                    "target_smiles": target_smiles,
                    "generated_smiles": top1_smiles,
                    "raw_samples": "|".join(raw_samples),
                    "canonical_candidates": "|".join(candidate_smiles),
                    "candidate_count": float(len(candidate_smiles)),
                    "top1_valid": bool(top1_smiles),
                    "top1_exact_match": bool(top1_smiles == target_smiles),
                    "topk_exact_match": bool(target_smiles in candidate_smiles),
                    "top1_target_tanimoto": float(top1_tanimoto),
                    "mean_best_tanimoto": float(best_tanimoto),
                    "top1_scaffold_match": bool(scaffold_match),
                    "source_image_path": example["image_path"],
                    "target_image_path": str(target_image_path),
                    "generated_image_path": str(generated_image_path),
                    "generated_image_exists": generated_image_path.exists(),
                    "render_error": render_error,
                    "paired_output_success": bool(top1_smiles and generated_image_path.exists()),
                    **image_metrics,
                }
            )
    return rows


def _prepare_image_examples(
    rows: list[dict[str, str]],
    pair_dir: Path,
    stoi: dict[str, int],
    max_length: int,
    np: Any,
    pillow: dict[str, Any],
    image_size: int,
    tokenization: str,
    allow_unknown: bool = False,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        smiles = row.get("canonical_smiles") or row.get("input_smiles", "")
        if not smiles:
            continue
        tokens = _tokenize_smiles(smiles, tokenization=tokenization)
        if len(tokens) + 1 > max_length:
            continue
        if not allow_unknown and any(token not in stoi for token in tokens):
            continue
        image_path = _resolve_image_path(row.get("image_path", ""), pair_dir)
        image = _load_ink_tensor(image_path, pillow=pillow, image_size=image_size, np=np)
        if image is None:
            continue
        input_ids, target_ids = _encode_tokens(tokens, stoi=stoi, max_length=max_length)
        examples.append(
            {
                "pair_id": row.get("pair_id", ""),
                "smiles": smiles,
                "image_path": str(image_path) if image_path else "",
                "image": image,
                "input_ids": input_ids,
                "target_ids": target_ids,
            }
        )
    return examples


def _summarize_image_smiles_decoder(
    prediction_rows: list[dict[str, Any]],
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    history: list[dict[str, float]],
    pair_dir: Path,
    output_dir: Path,
    predictions_path: Path,
    sample_predictions_path: Path,
    contact_sheet_path: str,
    model_path: Path,
    vocab_path: Path,
    train_fraction: float,
    seed: int,
    max_length: int,
    hidden_dim: int,
    embedding_dim: int,
    encoder_channels: int,
    image_token_grid: int,
    transformer_layers: int,
    attention_heads: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
    tokenization: str,
    decoding: str,
    beam_size: int,
    length_penalty: float,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    total = len(prediction_rows)
    compared = [row for row in prediction_rows if row["image_compared"]]
    image_mse_values = [float(row["image_mse"]) for row in compared if row["image_mse"] != ""]
    return {
        "phase": "phase5c_image_conditioned_smiles_decoder",
        "pair_dir": str(pair_dir),
        "output_dir": str(output_dir),
        "train_fraction": float(train_fraction),
        "seed": float(seed),
        "max_length": float(max_length),
        "hidden_dim": float(hidden_dim),
        "embedding_dim": float(embedding_dim),
        "encoder_channels": float(encoder_channels),
        "image_token_grid": float(image_token_grid),
        "transformer_layers": float(transformer_layers),
        "attention_heads": float(attention_heads),
        "dropout": float(dropout),
        "epochs": float(epochs),
        "batch_size": float(batch_size),
        "learning_rate": float(learning_rate),
        "samples_per_condition": float(samples_per_condition),
        "temperature": float(temperature),
        "sample_top_k": float(sample_top_k),
        "tokenization": tokenization,
        "decoding": decoding,
        "beam_size": float(beam_size),
        "length_penalty": float(length_penalty),
        "image_size": float(image_size),
        "device": device,
        "pairs": float(len(train_rows) + len(eval_rows)),
        "train_pairs": float(len(train_rows)),
        "eval_pairs": float(len(eval_rows)),
        "train_examples": float(len(train_examples)),
        "eval_examples": float(len(eval_examples)),
        "final_train_token_loss": float(history[-1]["train_token_loss"]) if history else 0.0,
        "final_train_token_ppl": float(history[-1]["train_token_ppl"]) if history else 0.0,
        "top1_valid": float(_count(prediction_rows, "top1_valid")),
        "top1_valid_fraction": _fraction(_count(prediction_rows, "top1_valid"), total),
        "top1_exact_matches": float(_count(prediction_rows, "top1_exact_match")),
        "top1_exact_match_fraction": _fraction(_count(prediction_rows, "top1_exact_match"), total),
        "topk_exact_matches": float(_count(prediction_rows, "topk_exact_match")),
        "topk_exact_match_fraction": _fraction(_count(prediction_rows, "topk_exact_match"), total),
        "top1_scaffold_matches": float(_count(prediction_rows, "top1_scaffold_match")),
        "top1_scaffold_match_fraction": _fraction(_count(prediction_rows, "top1_scaffold_match"), total),
        "top1_target_tanimoto": _mean_float(prediction_rows, "top1_target_tanimoto"),
        "mean_best_tanimoto": _mean_float(prediction_rows, "mean_best_tanimoto"),
        "mean_candidate_count": _mean_float(prediction_rows, "candidate_count"),
        "generated_images": float(_count(prediction_rows, "generated_image_exists")),
        "generated_image_fraction": _fraction(_count(prediction_rows, "generated_image_exists"), total),
        "image_compared": float(len(compared)),
        "image_compared_fraction": _fraction(len(compared), total),
        "image_exact_matches": float(_count(prediction_rows, "image_exact_match")),
        "image_exact_match_fraction": _fraction(_count(prediction_rows, "image_exact_match"), len(compared)),
        "image_mse_mean": float(sum(image_mse_values) / len(image_mse_values)) if image_mse_values else 0.0,
        "image_mse_max": float(max(image_mse_values)) if image_mse_values else 0.0,
        "paired_output_success": float(_count(prediction_rows, "paired_output_success")),
        "paired_output_success_fraction": _fraction(_count(prediction_rows, "paired_output_success"), total),
        "predictions": str(predictions_path),
        "sample_predictions": str(sample_predictions_path),
        "sample_contact_sheet": contact_sheet_path,
        "model": str(model_path),
        "vocab": str(vocab_path),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return float(sum(values) / len(values)) if values else 0.0


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5C image-conditioned SMILES decoder.")
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase5c_image_smiles_decoder")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--encoder-channels", type=int, default=64)
    parser.add_argument("--image-token-grid", type=int, default=4)
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
    args = parser.parse_args()

    metrics = run_image_conditioned_smiles_decoder(
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
