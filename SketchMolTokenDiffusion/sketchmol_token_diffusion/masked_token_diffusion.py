"""Masked token diffusion for direct SMILES generation.

The model is conditioned on a molecular fingerprint, corrupts canonical SMILES
tokens with mask noise, and learns to denoise the full sequence. When
``image_loss_weight`` is positive, the same denoising backbone also predicts a
sketch image, which is used by the joint image+SMILES route.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from sketch_smiles.audit_pairs import _load_pillow, _load_rdkit, _resolve_image_path
from sketch_smiles.phase5a0_oracle_baseline import (
    _fraction,
    _image_pair_metrics,
    _render_smiles,
    _sample_rows,
    _split_rows,
    _write_oracle_contact_sheet,
    _write_rows,
)
from sketch_smiles.phase5a1_learned_smiles_decoder import (
    BOS,
    EOS,
    PAD,
    _build_vocab,
    _canonical_candidate_list,
    _fingerprint_tanimoto,
    _load_numpy,
    _load_torch,
    _make_fingerprint_fn,
    _normalize_rerank_mode,
    _normalize_tokenization as _normalize_smiles_tokenization,
    _prepare_examples,
    _read_rows,
    _resolve_device,
    _scaffold_match,
    _set_rdkit_error_logging,
    _set_seeds,
    _tanimoto,
)
from sketch_smiles.phase5b_joint_decoder import _load_ink_tensor, _prefix_metrics, _save_ink_image


MASK = "<mask>"
SELFIES_TOKENIZATION = "selfies"
DECODE_LENGTH_MODES = {"free", "train_median", "oracle"}


def _normalize_generation_tokenization(tokenization: str) -> str:
    value = tokenization.strip().lower().replace("-", "_")
    if value in {"selfies", "selfie"}:
        return SELFIES_TOKENIZATION
    return _normalize_smiles_tokenization(value)


def _load_selfies() -> Any:
    try:
        import selfies

        return selfies
    except Exception as exc:
        raise RuntimeError("SELFIES tokenization requires the `selfies` Python package.") from exc


def _tokens_for_generation(smiles: str, tokenization: str) -> list[str]:
    if tokenization != SELFIES_TOKENIZATION:
        from sketch_smiles.phase5a1_learned_smiles_decoder import _tokenize_smiles

        return _tokenize_smiles(smiles, tokenization=tokenization)
    selfies = _load_selfies()
    try:
        encoded = selfies.encoder(smiles)
        return list(selfies.split_selfies(encoded))
    except Exception:
        return []


def _decode_generation_tokens(tokens: list[str], tokenization: str) -> str:
    if tokenization != SELFIES_TOKENIZATION:
        return "".join(tokens)
    selfies = _load_selfies()
    try:
        return selfies.decoder("".join(tokens))
    except Exception:
        return ""


def _build_generation_vocab(rows: list[dict[str, str]], tokenization: str) -> tuple[dict[str, int], list[str]]:
    if tokenization != SELFIES_TOKENIZATION:
        return _build_vocab(rows, tokenization=tokenization)
    tokens = sorted(
        {
            token
            for row in rows
            for token in _tokens_for_generation(row.get("canonical_smiles") or row.get("input_smiles", ""), tokenization=tokenization)
        }
    )
    itos = [PAD, BOS, EOS] + tokens
    return {token: idx for idx, token in enumerate(itos)}, itos


def _encode_generation_tokens(tokens: list[str], stoi: dict[str, int], max_length: int) -> tuple[list[int], list[int]]:
    input_tokens = [BOS] + tokens
    target_tokens = tokens + [EOS]
    input_ids = [stoi.get(token, stoi[PAD]) for token in input_tokens][:max_length]
    target_ids = [stoi.get(token, stoi[PAD]) for token in target_tokens][:max_length]
    while len(input_ids) < max_length:
        input_ids.append(stoi[PAD])
    while len(target_ids) < max_length:
        target_ids.append(stoi[PAD])
    return input_ids, target_ids


def _prepare_generation_examples(
    rows: list[dict[str, str]],
    stoi: dict[str, int],
    fingerprint_fn: Any,
    max_length: int,
    np: Any,
    tokenization: str,
    pad_idx: int,
    allow_unknown: bool = False,
) -> list[dict[str, Any]]:
    if tokenization != SELFIES_TOKENIZATION:
        examples = _prepare_examples(
            rows,
            stoi,
            fingerprint_fn,
            max_length=max_length,
            np=np,
            tokenization=tokenization,
            allow_unknown=allow_unknown,
        )
        return _attach_target_lengths(examples, pad_idx=pad_idx)

    examples: list[dict[str, Any]] = []
    for row in rows:
        smiles = row.get("canonical_smiles") or row.get("input_smiles", "")
        condition_smiles = row.get("condition_smiles") or smiles
        if not smiles:
            continue
        tokens = _tokens_for_generation(smiles, tokenization=tokenization)
        if not tokens or len(tokens) + 1 > max_length:
            continue
        if not allow_unknown and any(token not in stoi for token in tokens):
            continue
        if allow_unknown and any(token not in stoi for token in tokens):
            continue
        feature = fingerprint_fn(condition_smiles)
        if feature is None:
            continue
        input_ids, target_ids = _encode_generation_tokens(tokens, stoi=stoi, max_length=max_length)
        examples.append(
            {
                "pair_id": row.get("pair_id", ""),
                "smiles": smiles,
                "condition_smiles": condition_smiles,
                "image_path": row.get("image_path", ""),
                "feature": np.asarray(feature, dtype=np.float32),
                "input_ids": input_ids,
                "target_ids": target_ids,
                "target_length": sum(1 for value in target_ids if value != pad_idx),
            }
        )
    return examples


def _attach_target_lengths(examples: list[dict[str, Any]], pad_idx: int) -> list[dict[str, Any]]:
    for example in examples:
        example["target_length"] = sum(1 for value in example.get("target_ids", []) if value != pad_idx)
    return examples


def _median_target_length(examples: list[dict[str, Any]]) -> int:
    lengths = sorted(int(example.get("target_length", 0)) for example in examples if int(example.get("target_length", 0)) > 0)
    if not lengths:
        return 0
    return int(lengths[len(lengths) // 2])


def _normalize_decode_length_mode(mode: str) -> str:
    value = mode.strip().lower().replace("-", "_")
    aliases = {"none": "free", "median": "train_median", "target": "oracle", "oracle_target": "oracle"}
    value = aliases.get(value, value)
    if value not in DECODE_LENGTH_MODES:
        raise ValueError(f"Unsupported decode_length_mode {mode!r}; expected one of {sorted(DECODE_LENGTH_MODES)}.")
    return value


def run_masked_token_diffusion(
    pair_dir: str | Path,
    output_dir: str | Path = "outputs/runs/masked_token_diffusion",
    train_fraction: float = 0.8,
    seed: int = 7,
    limit: int | None = None,
    fingerprint_bits: int = 2048,
    max_length: int = 128,
    hidden_dim: int = 384,
    embedding_dim: int = 96,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    diffusion_steps: int = 16,
    min_mask_prob: float = 0.15,
    max_mask_prob: float = 0.95,
    samples_per_condition: int = 8,
    temperature: float = 0.9,
    sample_top_k: int = 16,
    rerank_mode: str = "condition_fingerprint",
    transformer_layers: int = 4,
    attention_heads: int = 8,
    condition_tokens: int = 8,
    dropout: float = 0.1,
    tokenization: str = "smiles_token",
    latent_dim: int = 128,
    image_loss_weight: float = 0.0,
    image_foreground_weight: float = 8.0,
    clip_loss_weight: float = 0.0,
    clip_temperature: float = 0.07,
    decode_length_mode: str = "free",
    min_decode_tokens: int = 1,
    image_size: int = 128,
    sample_count: int = 64,
    contact_sheet_cols: int = 8,
    contact_thumb_size: int = 144,
    device: str = "auto",
    route_name: str = "sketchmol_token_diffusion_masked_smiles",
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
        raise RuntimeError("RDKit is required for token diffusion fingerprints and validation.")
    if not pillow:
        raise RuntimeError("Pillow is required for rendering and image consistency metrics.")
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

    tokenization = _normalize_generation_tokenization(tokenization)
    rerank_mode = _normalize_rerank_mode(rerank_mode)
    decode_length_mode = _normalize_decode_length_mode(decode_length_mode)
    clip_loss_weight = float(clip_loss_weight)
    image_loss_weight = float(image_loss_weight)
    if clip_loss_weight > 0 and image_loss_weight <= 0:
        raise ValueError("clip_loss_weight requires image_loss_weight > 0 so the image branch is trained.")
    stoi, itos = _ensure_mask_token(*_build_generation_vocab(train_rows, tokenization=tokenization))
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps({"stoi": stoi, "itos": itos}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fingerprint_fn = _make_fingerprint_fn(rdkit, np=np, fingerprint_bits=fingerprint_bits)
    train_examples = _prepare_generation_examples(
        train_rows,
        stoi,
        fingerprint_fn,
        max_length=max_length,
        np=np,
        tokenization=tokenization,
        pad_idx=stoi[PAD],
    )
    eval_examples = _prepare_generation_examples(
        eval_rows,
        stoi,
        fingerprint_fn,
        max_length=max_length,
        np=np,
        tokenization=tokenization,
        pad_idx=stoi[PAD],
        allow_unknown=True,
    )
    if image_loss_weight > 0 or clip_loss_weight > 0:
        train_examples = _attach_images(train_examples, pair_dir=pair_dir, pillow=pillow, np=np, image_size=image_size)
        eval_examples = _attach_images(eval_examples, pair_dir=pair_dir, pillow=pillow, np=np, image_size=image_size)
    if not train_examples:
        raise RuntimeError("No train examples available for masked token diffusion.")
    if not eval_examples:
        raise RuntimeError("No eval examples available for masked token diffusion.")

    resolved_device = _resolve_device(device, torch)
    model = MaskedTokenDiffusionTransformer(
        vocab_size=len(itos),
        feature_dim=fingerprint_bits,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_idx=stoi[PAD],
        mask_idx=stoi[MASK],
        max_length=max_length,
        diffusion_steps=diffusion_steps,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        condition_tokens=condition_tokens,
        dropout=dropout,
        latent_dim=latent_dim,
        image_size=image_size if image_loss_weight > 0 else 0,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = _train_diffusion_model(
        model=model,
        examples=train_examples,
        optimizer=optimizer,
        torch=torch,
        np=np,
        device=resolved_device,
        batch_size=batch_size,
        epochs=epochs,
        pad_idx=stoi[PAD],
        mask_idx=stoi[MASK],
        diffusion_steps=diffusion_steps,
        min_mask_prob=min_mask_prob,
        max_mask_prob=max_mask_prob,
        image_loss_weight=image_loss_weight,
        image_foreground_weight=image_foreground_weight,
        clip_loss_weight=clip_loss_weight,
        clip_temperature=clip_temperature,
        seed=seed,
    )

    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "feature_dim": fingerprint_bits,
                "vocab_size": len(itos),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "pad_idx": stoi[PAD],
                "mask_idx": stoi[MASK],
                "max_length": max_length,
                "diffusion_steps": diffusion_steps,
                "transformer_layers": transformer_layers,
                "attention_heads": attention_heads,
                "condition_tokens": condition_tokens,
                "dropout": dropout,
                "latent_dim": latent_dim,
                "image_size": image_size if image_loss_weight > 0 else 0,
            },
        },
        model_path,
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    prediction_rows = _evaluate_diffusion_model(
        model=model,
        eval_examples=eval_examples,
        stoi=stoi,
        itos=itos,
        pair_dir=pair_dir,
        output_dir=output_dir,
        rdkit=rdkit,
        pillow=pillow,
        torch=torch,
        np=np,
        device=resolved_device,
        max_length=max_length,
        diffusion_steps=diffusion_steps,
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
        rerank_mode=rerank_mode,
        fingerprint_fn=fingerprint_fn,
        image_size=image_size,
        has_image_head=image_loss_weight > 0,
        tokenization=tokenization,
        decode_length_mode=decode_length_mode,
        train_decode_length=_median_target_length(train_examples),
        min_decode_tokens=min_decode_tokens,
    )

    predictions_path = output_dir / "predictions.csv"
    _write_rows(predictions_path, prediction_rows)
    sample_prediction_rows = _sample_rows(prediction_rows, sample_count=sample_count, seed=seed)
    sample_predictions_path = output_dir / "sample_predictions.csv"
    _write_rows(sample_predictions_path, sample_prediction_rows)
    contact_sheet_path = _write_oracle_contact_sheet(
        sample_rows=sample_prediction_rows,
        pillow=pillow,
        cols=contact_sheet_cols,
        thumb_size=contact_thumb_size,
        output_path=output_dir / "sample_contact_sheet.png",
    )

    metrics = _summarize_diffusion(
        route_name=route_name,
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
        fingerprint_bits=fingerprint_bits,
        max_length=max_length,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        diffusion_steps=diffusion_steps,
        min_mask_prob=min_mask_prob,
        max_mask_prob=max_mask_prob,
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
        rerank_mode=rerank_mode,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        condition_tokens=condition_tokens,
        dropout=dropout,
        tokenization=tokenization,
        latent_dim=latent_dim,
        image_loss_weight=image_loss_weight,
        image_foreground_weight=image_foreground_weight,
        clip_loss_weight=clip_loss_weight,
        clip_temperature=clip_temperature,
        decode_length_mode=decode_length_mode,
        train_decode_length=_median_target_length(train_examples),
        min_decode_tokens=min_decode_tokens,
        image_size=image_size,
        device=str(resolved_device),
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phase": route_name,
                "research_question": (
                    "Can a diffusion model directly generate molecular structure tokens, rather than molecular images, "
                    "while preserving SketchMol-style multi-candidate sampling and verification?"
                ),
                "pair_dir": str(pair_dir),
                "pairs_csv": str(pairs_path),
                "output_dir": str(output_dir),
                "train_fraction": train_fraction,
                "seed": seed,
                "limit": limit,
                "fingerprint_bits": fingerprint_bits,
                "max_length": max_length,
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "diffusion_steps": diffusion_steps,
                "min_mask_prob": min_mask_prob,
                "max_mask_prob": max_mask_prob,
                "samples_per_condition": samples_per_condition,
                "temperature": temperature,
                "sample_top_k": sample_top_k,
                "rerank_mode": rerank_mode,
                "transformer_layers": transformer_layers,
                "attention_heads": attention_heads,
                "condition_tokens": condition_tokens,
                "dropout": dropout,
                "tokenization": tokenization,
                "latent_dim": latent_dim,
                "image_loss_weight": image_loss_weight,
                "image_foreground_weight": image_foreground_weight,
                "clip_loss_weight": clip_loss_weight,
                "clip_temperature": clip_temperature,
                "decode_length_mode": decode_length_mode,
                "train_decode_length": _median_target_length(train_examples),
                "min_decode_tokens": min_decode_tokens,
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


class MaskedTokenDiffusionTransformer:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _load_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                feature_dim: int,
                embedding_dim: int,
                hidden_dim: int,
                pad_idx: int,
                mask_idx: int,
                max_length: int,
                diffusion_steps: int,
                transformer_layers: int,
                attention_heads: int,
                condition_tokens: int,
                dropout: float,
                latent_dim: int,
                image_size: int,
            ) -> None:
                super().__init__()
                if hidden_dim % attention_heads != 0:
                    raise ValueError(f"hidden_dim={hidden_dim} must be divisible by attention_heads={attention_heads}.")
                self.pad_idx = int(pad_idx)
                self.mask_idx = int(mask_idx)
                self.max_length = int(max_length)
                self.diffusion_steps = int(diffusion_steps)
                self.hidden_dim = int(hidden_dim)
                self.condition_tokens = int(condition_tokens)
                self.latent_dim = int(latent_dim)
                self.image_size = int(image_size)
                self.condition_encoder = nn.Sequential(
                    nn.Linear(feature_dim, self.latent_dim),
                    nn.LayerNorm(self.latent_dim),
                    nn.Tanh(),
                )
                self.latent_to_condition = nn.Linear(self.latent_dim, hidden_dim * condition_tokens)
                self.condition_alignment = nn.Linear(self.latent_dim, self.latent_dim)
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
                self.input_projection = nn.Linear(embedding_dim, hidden_dim)
                self.position = nn.Embedding(max_length + condition_tokens, hidden_dim)
                self.time_embedding = nn.Embedding(max(2, diffusion_steps + 1), hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=attention_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
                self.out = nn.Linear(hidden_dim, vocab_size)
                self.text_alignment = nn.Linear(hidden_dim, self.latent_dim)
                self.image_head = (
                    nn.Sequential(
                        nn.Linear(self.latent_dim, hidden_dim * 2),
                        nn.GELU(),
                        nn.Linear(hidden_dim * 2, image_size * image_size),
                        nn.Sigmoid(),
                    )
                    if image_size > 0
                    else None
                )
                self.image_encoder = (
                    nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
                        nn.GELU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                        nn.GELU(),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                        nn.GELU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(128, self.latent_dim),
                    )
                    if image_size > 0
                    else None
                )

            def encode_latent(self, features: Any) -> Any:
                return self.condition_encoder(features)

            def encode_context(self, features: Any, noisy_ids: Any, timesteps: Any) -> tuple[Any, Any]:
                batch, length = noisy_ids.shape
                if length > self.max_length:
                    noisy_ids = noisy_ids[:, : self.max_length]
                    length = self.max_length
                latent = self.encode_latent(features)
                cond = self.latent_to_condition(latent).view(batch, self.condition_tokens, self.hidden_dim)
                token_hidden = self.input_projection(self.embedding(noisy_ids))
                positions = torch.arange(self.condition_tokens + length, device=noisy_ids.device).unsqueeze(0).expand(batch, -1)
                token_positions = self.position(positions)[:, self.condition_tokens :, :]
                t = timesteps.clamp(0, self.diffusion_steps).long()
                time_hidden = self.time_embedding(t).unsqueeze(1)
                token_hidden = token_hidden + token_positions + time_hidden
                full_hidden = torch.cat([cond + self.position(positions)[:, : self.condition_tokens, :], token_hidden], dim=1)
                token_padding = noisy_ids.eq(self.pad_idx)
                cond_padding = torch.zeros((batch, self.condition_tokens), dtype=torch.bool, device=noisy_ids.device)
                encoded = self.encoder(full_hidden, src_key_padding_mask=torch.cat([cond_padding, token_padding], dim=1))
                return latent, encoded

            def forward(self, features: Any, noisy_ids: Any, timesteps: Any) -> tuple[Any, Any | None]:
                latent, encoded = self.encode_context(features, noisy_ids, timesteps)
                token_encoded = encoded[:, self.condition_tokens :, :]
                logits = self.out(token_encoded)
                image = None
                if self.image_head is not None:
                    batch = features.shape[0]
                    image = self.image_head(latent).view(batch, 1, self.image_size, self.image_size)
                return logits, image

            def alignment_latents(self, features: Any, noisy_ids: Any, timesteps: Any, images: Any) -> tuple[Any, Any, Any]:
                if self.image_encoder is None:
                    raise RuntimeError("alignment_latents requires image_size > 0.")
                latent, encoded = self.encode_context(features, noisy_ids, timesteps)
                text_latent = self.text_alignment(encoded.mean(dim=1))
                image_latent = self.image_encoder(images)
                condition_latent = self.condition_alignment(latent)
                return text_latent, image_latent, condition_latent

            def generate_image(self, features: Any) -> Any | None:
                if self.image_head is None:
                    return None
                batch = features.shape[0]
                noisy = torch.full((batch, self.max_length), self.mask_idx, dtype=torch.long, device=features.device)
                timesteps = torch.full((batch,), max(1, self.diffusion_steps // 2), dtype=torch.long, device=features.device)
                _logits, image = self.forward(features, noisy, timesteps)
                return image

        return _Model(*args, **kwargs)


def _train_diffusion_model(
    model: Any,
    examples: list[dict[str, Any]],
    optimizer: Any,
    torch: Any,
    np: Any,
    device: Any,
    batch_size: int,
    epochs: int,
    pad_idx: int,
    mask_idx: int,
    diffusion_steps: int,
    min_mask_prob: float,
    max_mask_prob: float,
    image_loss_weight: float,
    image_foreground_weight: float,
    clip_loss_weight: float,
    clip_temperature: float,
    seed: int,
) -> list[dict[str, float]]:
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")
    history: list[dict[str, float]] = []
    rng = random.Random(seed)
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        total_token_loss = 0.0
        total_image_loss = 0.0
        total_clip_loss = 0.0
        total_tokens = 0
        total_batches = 0
        for start in range(0, len(order), int(batch_size)):
            batch = [examples[idx] for idx in order[start : start + int(batch_size)]]
            features = torch.as_tensor(np.stack([row["feature"] for row in batch]), dtype=torch.float32, device=device)
            target_ids = torch.tensor([row["target_ids"] for row in batch], dtype=torch.long, device=device)
            timesteps = torch.randint(1, int(diffusion_steps) + 1, (len(batch),), dtype=torch.long, device=device)
            noisy_ids, loss_mask = _corrupt_tokens(
                target_ids=target_ids,
                timesteps=timesteps,
                torch=torch,
                pad_idx=pad_idx,
                mask_idx=mask_idx,
                diffusion_steps=diffusion_steps,
                min_mask_prob=min_mask_prob,
                max_mask_prob=max_mask_prob,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, image = model(features, noisy_ids, timesteps)
            raw_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)).view_as(target_ids)
            token_loss = (raw_loss * loss_mask.float()).sum() / loss_mask.float().sum().clamp_min(1.0)
            loss = token_loss
            image_loss_value = 0.0
            clip_loss_value = 0.0
            target_images = None
            if image_loss_weight > 0 and image is not None and batch and "image" in batch[0]:
                target_images = torch.as_tensor(np.stack([row["image"] for row in batch]), dtype=torch.float32, device=device)
                weights = 1.0 + float(image_foreground_weight) * target_images
                image_loss = (((image - target_images) ** 2) * weights).mean()
                image_loss_value = float(image_loss.item())
                loss = loss + float(image_loss_weight) * image_loss
            if clip_loss_weight > 0 and batch and "image" in batch[0]:
                if target_images is None:
                    target_images = torch.as_tensor(np.stack([row["image"] for row in batch]), dtype=torch.float32, device=device)
                text_latent, image_latent, condition_latent = model.alignment_latents(features, noisy_ids, timesteps, target_images)
                clip_loss = 0.5 * (
                    _contrastive_alignment_loss(text_latent, image_latent, torch=torch, temperature=clip_temperature)
                    + _contrastive_alignment_loss(condition_latent, image_latent, torch=torch, temperature=clip_temperature)
                )
                clip_loss_value = float(clip_loss.item())
                loss = loss + float(clip_loss_weight) * clip_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            token_count = int(loss_mask.sum().item())
            total_loss += float(loss.item()) * max(1, token_count)
            total_token_loss += float(token_loss.item()) * max(1, token_count)
            total_image_loss += image_loss_value
            total_clip_loss += clip_loss_value
            total_tokens += token_count
            total_batches += 1
        mean_loss = total_loss / max(1, total_tokens)
        mean_token_loss = total_token_loss / max(1, total_tokens)
        mean_image_loss = total_image_loss / max(1, total_batches)
        mean_clip_loss = total_clip_loss / max(1, total_batches)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(mean_loss),
                "train_token_loss": float(mean_token_loss),
                "train_token_ppl": float(math.exp(min(20.0, mean_token_loss))),
                "train_image_loss": float(mean_image_loss),
                "train_clip_loss": float(mean_clip_loss),
            }
        )
        image_suffix = f" train_image_loss={mean_image_loss:.5f}" if image_loss_weight > 0 else ""
        clip_suffix = f" train_clip_loss={mean_clip_loss:.4f}" if clip_loss_weight > 0 else ""
        print(
            f"  epoch={epoch} train_token_loss={mean_token_loss:.4f} "
            f"train_token_ppl={math.exp(min(20.0, mean_token_loss)):.3f}{image_suffix}{clip_suffix}",
            flush=True,
        )
    return history


def _contrastive_alignment_loss(left: Any, right: Any, torch: Any, temperature: float) -> Any:
    left = torch.nn.functional.normalize(left, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1)
    logits = left @ right.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(left.shape[0], device=left.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels)
    )


def _corrupt_tokens(
    target_ids: Any,
    timesteps: Any,
    torch: Any,
    pad_idx: int,
    mask_idx: int,
    diffusion_steps: int,
    min_mask_prob: float,
    max_mask_prob: float,
) -> tuple[Any, Any]:
    valid = target_ids.ne(pad_idx)
    probs = float(min_mask_prob) + (float(max_mask_prob) - float(min_mask_prob)) * (timesteps.float() / float(max(1, diffusion_steps)))
    mask = torch.rand_like(target_ids.float()).lt(probs.unsqueeze(1)) & valid
    for row_idx in range(mask.shape[0]):
        if valid[row_idx].any() and not mask[row_idx].any():
            positions = torch.nonzero(valid[row_idx], as_tuple=False).flatten()
            chosen = int(positions[torch.randint(0, positions.numel(), (1,), device=positions.device)].item())
            mask[row_idx, chosen] = True
    noisy = target_ids.clone()
    noisy[mask] = int(mask_idx)
    return noisy, mask


def _evaluate_diffusion_model(
    model: Any,
    eval_examples: list[dict[str, Any]],
    stoi: dict[str, int],
    itos: list[str],
    pair_dir: Path,
    output_dir: Path,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
    torch: Any,
    np: Any,
    device: Any,
    max_length: int,
    diffusion_steps: int,
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
    rerank_mode: str,
    fingerprint_fn: Any,
    image_size: int,
    has_image_head: bool,
    tokenization: str,
    decode_length_mode: str,
    train_decode_length: int,
    min_decode_tokens: int,
) -> list[dict[str, Any]]:
    generated_image_dir = output_dir / ("joint_images" if has_image_head else "generated_images")
    rendered_smiles_image_dir = output_dir / "rendered_smiles_images"
    resized_target_image_dir = output_dir / "target_images_resized"
    generated_image_dir.mkdir(parents=True, exist_ok=True)
    rendered_smiles_image_dir.mkdir(parents=True, exist_ok=True)
    resized_target_image_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in eval_examples:
            raw_samples = [
                _sample_diffusion_smiles(
                    model=model,
                    feature=example["feature"],
                    stoi=stoi,
                    itos=itos,
                    torch=torch,
                    device=device,
                    max_length=max_length,
                    diffusion_steps=diffusion_steps,
                    temperature=temperature,
                    sample_top_k=sample_top_k,
                    tokenization=tokenization,
                    pad_idx=stoi[PAD],
                    eos_idx=stoi[EOS],
                    decode_length_limit=_decode_length_limit_for_example(
                        example,
                        mode=decode_length_mode,
                        train_decode_length=train_decode_length,
                    ),
                    min_decode_tokens=min_decode_tokens,
                )
                for _ in range(int(samples_per_condition))
            ]
            beam_candidate_smiles = _canonical_candidate_list(raw_samples, rdkit)
            candidate_smiles, condition_scores = _rerank_candidate_smiles(
                beam_candidate_smiles,
                condition_feature=example["feature"],
                fingerprint_fn=fingerprint_fn,
                np=np,
                rerank_mode=rerank_mode,
            )
            pair_id = example["pair_id"]
            target_smiles = example["smiles"]
            top1_smiles = candidate_smiles[0] if candidate_smiles else ""
            top1_valid = bool(top1_smiles)
            target_mol = rdkit["Chem"].MolFromSmiles(target_smiles)
            top1_mol = rdkit["Chem"].MolFromSmiles(top1_smiles) if top1_smiles else None
            top1_tanimoto = _tanimoto(target_mol, top1_mol, rdkit)
            best_tanimoto = max((_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidate_smiles), default=0.0)
            scaffold_match = _scaffold_match(target_smiles, top1_smiles, rdkit)

            if has_image_head:
                feature_tensor = torch.as_tensor(example["feature"], dtype=torch.float32, device=device).unsqueeze(0)
                predicted_ink = model.generate_image(feature_tensor)[0].detach().cpu().numpy()
                generated_image_path = generated_image_dir / f"{pair_id}.png"
                _save_ink_image(predicted_ink, generated_image_path, pillow)
                target_image_path = resized_target_image_dir / f"{pair_id}.png"
                if "image" in example:
                    _save_ink_image(example["image"], target_image_path, pillow)
            else:
                generated_image_path = generated_image_dir / f"{pair_id}.png"
                _render_smiles(top1_smiles, generated_image_path, image_size=image_size, rdkit=rdkit)
                target_image_path = _resolve_image_path(example["image_path"], pair_dir)

            rendered_smiles_image_path = rendered_smiles_image_dir / f"{pair_id}.png"
            render_error = _render_smiles(top1_smiles, rendered_smiles_image_path, image_size=image_size, rdkit=rdkit)
            image_metrics = _image_pair_metrics(target_image_path, generated_image_path, pillow)
            consistency_metrics = (
                _prefix_metrics(_image_pair_metrics(rendered_smiles_image_path, generated_image_path, pillow), "smiles_render")
                if has_image_head
                else {}
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "target_smiles": target_smiles,
                    "generated_smiles": top1_smiles,
                    "raw_samples": "|".join(raw_samples),
                    "beam_canonical_candidates": "|".join(beam_candidate_smiles),
                    "canonical_candidates": "|".join(candidate_smiles),
                    "candidate_condition_tanimotos": "|".join(f"{condition_scores.get(smiles, 0.0):.6f}" for smiles in candidate_smiles),
                    "candidate_count": float(len(candidate_smiles)),
                    "decode_length_mode": decode_length_mode,
                    "decode_length_limit": float(
                        _decode_length_limit_for_example(
                            example,
                            mode=decode_length_mode,
                            train_decode_length=train_decode_length,
                        )
                        or 0
                    ),
                    "top1_valid": top1_valid,
                    "top1_exact_match": bool(top1_smiles == target_smiles),
                    "topk_exact_match": bool(target_smiles in candidate_smiles),
                    "top1_target_tanimoto": float(top1_tanimoto),
                    "mean_best_tanimoto": float(best_tanimoto),
                    "top1_condition_tanimoto": float(condition_scores.get(top1_smiles, 0.0)) if top1_smiles else 0.0,
                    "mean_best_condition_tanimoto": float(max(condition_scores.values(), default=0.0)),
                    "top1_scaffold_match": bool(scaffold_match),
                    "target_image_path": str(target_image_path) if target_image_path else "",
                    "generated_image_path": str(generated_image_path),
                    "rendered_smiles_image_path": str(rendered_smiles_image_path),
                    "generated_image_exists": generated_image_path.exists(),
                    "rendered_smiles_image_exists": rendered_smiles_image_path.exists(),
                    "render_error": render_error,
                    "paired_output_success": bool(top1_valid and generated_image_path.exists()),
                    **image_metrics,
                    **consistency_metrics,
                }
            )
    return rows


def _sample_diffusion_smiles(
    model: Any,
    feature: Any,
    stoi: dict[str, int],
    itos: list[str],
    torch: Any,
    device: Any,
    max_length: int,
    diffusion_steps: int,
    temperature: float,
    sample_top_k: int,
    tokenization: str,
    pad_idx: int,
    eos_idx: int,
    decode_length_limit: int | None,
    min_decode_tokens: int,
) -> str:
    feature_tensor = torch.as_tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
    ids = torch.full((1, int(max_length)), stoi[MASK], dtype=torch.long, device=device)
    fixed = torch.zeros_like(ids, dtype=torch.bool)
    eos_position = None
    if decode_length_limit is not None and int(decode_length_limit) > 0:
        limit = max(1, min(int(max_length), int(decode_length_limit)))
        eos_position = max(0, limit - 1)
        ids[0, eos_position] = int(eos_idx)
        fixed[0, eos_position] = True
        if eos_position + 1 < int(max_length):
            ids[0, eos_position + 1 :] = int(pad_idx)
            fixed[0, eos_position + 1 :] = True
    blocked = {stoi[PAD], stoi[BOS], stoi[MASK]}
    for step in range(int(diffusion_steps), 0, -1):
        timestep = torch.tensor([step], dtype=torch.long, device=device)
        logits, _image = model(feature_tensor, ids, timestep)
        logits = logits[0] / max(float(temperature), 1e-6)
        for blocked_id in blocked:
            logits[:, blocked_id] = -1e9
        _apply_eos_constraints(logits, eos_idx=eos_idx, eos_position=eos_position, min_decode_tokens=min_decode_tokens)
        sampled, confidence = _sample_positions(logits, torch=torch, top_k=sample_top_k)
        mask_positions = torch.nonzero(~fixed[0], as_tuple=False).flatten()
        if mask_positions.numel() == 0:
            break
        reveal_count = max(1, math.ceil(mask_positions.numel() / max(1, step)))
        ranked = mask_positions[torch.argsort(confidence[mask_positions], descending=True)[:reveal_count]]
        ids[0, ranked] = sampled[ranked]
        fixed[0, ranked] = True
    if (~fixed).any():
        timestep = torch.tensor([0], dtype=torch.long, device=device)
        logits, _image = model(feature_tensor, ids, timestep)
        logits = logits[0] / max(float(temperature), 1e-6)
        for blocked_id in blocked:
            logits[:, blocked_id] = -1e9
        _apply_eos_constraints(logits, eos_idx=eos_idx, eos_position=eos_position, min_decode_tokens=min_decode_tokens)
        sampled, _confidence = _sample_positions(logits, torch=torch, top_k=sample_top_k)
        remaining = torch.nonzero(~fixed[0], as_tuple=False).flatten()
        if remaining.numel() > 0:
            ids[0, remaining] = sampled[remaining]
    if eos_position is not None:
        ids[0, eos_position] = int(eos_idx)
    return _decode_ids_to_smiles(ids[0].tolist(), itos, tokenization=tokenization)


def _apply_eos_constraints(logits: Any, eos_idx: int, eos_position: int | None, min_decode_tokens: int) -> None:
    if eos_position is not None:
        if eos_position > 0:
            logits[:eos_position, int(eos_idx)] = -1e9
        if eos_position + 1 < logits.shape[0]:
            logits[eos_position + 1 :, int(eos_idx)] = -1e9
        return
    min_tokens = max(0, int(min_decode_tokens))
    if min_tokens > 0:
        logits[:min(min_tokens, logits.shape[0]), int(eos_idx)] = -1e9


def _sample_positions(logits: Any, torch: Any, top_k: int) -> tuple[Any, Any]:
    if top_k and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        values, indices = torch.topk(logits, int(top_k), dim=-1)
        probs = torch.softmax(values, dim=-1)
        picked = torch.multinomial(probs, 1).squeeze(1)
        sampled = indices.gather(1, picked.unsqueeze(1)).squeeze(1)
        confidence = probs.gather(1, picked.unsqueeze(1)).squeeze(1)
        return sampled, confidence
    probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, 1).squeeze(1)
    confidence = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
    return sampled, confidence


def _decode_ids(ids: Iterable[int], itos: list[str]) -> str:
    tokens: list[str] = []
    for idx in ids:
        token = itos[int(idx)]
        if token == EOS:
            break
        if token in {PAD, BOS, MASK}:
            continue
        tokens.append(token)
    return "".join(tokens)


def _decode_ids_to_smiles(ids: Iterable[int], itos: list[str], tokenization: str) -> str:
    tokens: list[str] = []
    for idx in ids:
        token = itos[int(idx)]
        if token == EOS:
            break
        if token in {PAD, BOS, MASK}:
            continue
        tokens.append(token)
    return _decode_generation_tokens(tokens, tokenization=tokenization)


def _decode_length_limit_for_example(example: dict[str, Any], mode: str, train_decode_length: int) -> int | None:
    if mode == "oracle":
        return int(example.get("target_length", 0)) or None
    if mode == "train_median":
        return int(train_decode_length) or None
    return None


def _rerank_candidate_smiles(
    candidate_smiles: list[str],
    condition_feature: Any,
    fingerprint_fn: Any,
    np: Any,
    rerank_mode: str,
) -> tuple[list[str], dict[str, float]]:
    if not candidate_smiles:
        return [], {}
    if rerank_mode == "beam":
        return candidate_smiles, {smiles: 0.0 for smiles in candidate_smiles}
    scores = {smiles: _fingerprint_tanimoto(condition_feature, fingerprint_fn(smiles), np=np) for smiles in candidate_smiles}
    ranked = sorted(enumerate(candidate_smiles), key=lambda item: (scores[item[1]], -item[0]), reverse=True)
    return [smiles for _index, smiles in ranked], scores


def _attach_images(
    examples: list[dict[str, Any]],
    pair_dir: Path,
    pillow: dict[str, Any],
    np: Any,
    image_size: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for example in examples:
        image_path = _resolve_image_path(example.get("image_path", ""), pair_dir)
        image = _load_ink_tensor(image_path, pillow=pillow, image_size=image_size, np=np)
        if image is None:
            continue
        copy = dict(example)
        copy["image"] = image
        copy["image_path"] = str(image_path) if image_path else ""
        out.append(copy)
    return out


def _ensure_mask_token(stoi: dict[str, int], itos: list[str]) -> tuple[dict[str, int], list[str]]:
    if MASK not in stoi:
        stoi = dict(stoi)
        itos = list(itos)
        stoi[MASK] = len(itos)
        itos.append(MASK)
    return stoi, itos


def _summarize_diffusion(
    route_name: str,
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
    fingerprint_bits: int,
    max_length: int,
    hidden_dim: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    diffusion_steps: int,
    min_mask_prob: float,
    max_mask_prob: float,
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
    rerank_mode: str,
    transformer_layers: int,
    attention_heads: int,
    condition_tokens: int,
    dropout: float,
    tokenization: str,
    latent_dim: int,
    image_loss_weight: float,
    image_foreground_weight: float,
    clip_loss_weight: float,
    clip_temperature: float,
    decode_length_mode: str,
    train_decode_length: int,
    min_decode_tokens: int,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    total = len(prediction_rows)
    compared = [row for row in prediction_rows if row.get("image_compared")]
    consistency_compared = [row for row in prediction_rows if row.get("smiles_render_image_compared")]
    image_mse_values = [float(row["image_mse"]) for row in compared if row.get("image_mse") != ""]
    consistency_mse_values = [float(row["smiles_render_image_mse"]) for row in consistency_compared if row.get("smiles_render_image_mse") != ""]
    return {
        "phase": route_name,
        "pair_dir": str(pair_dir),
        "output_dir": str(output_dir),
        "train_fraction": float(train_fraction),
        "seed": float(seed),
        "fingerprint_bits": float(fingerprint_bits),
        "max_length": float(max_length),
        "hidden_dim": float(hidden_dim),
        "embedding_dim": float(embedding_dim),
        "epochs": float(epochs),
        "batch_size": float(batch_size),
        "learning_rate": float(learning_rate),
        "diffusion_steps": float(diffusion_steps),
        "min_mask_prob": float(min_mask_prob),
        "max_mask_prob": float(max_mask_prob),
        "samples_per_condition": float(samples_per_condition),
        "temperature": float(temperature),
        "sample_top_k": float(sample_top_k),
        "rerank_mode": rerank_mode,
        "transformer_layers": float(transformer_layers),
        "attention_heads": float(attention_heads),
        "condition_tokens": float(condition_tokens),
        "dropout": float(dropout),
        "tokenization": tokenization,
        "latent_dim": float(latent_dim),
        "image_loss_weight": float(image_loss_weight),
        "image_foreground_weight": float(image_foreground_weight),
        "clip_loss_weight": float(clip_loss_weight),
        "clip_temperature": float(clip_temperature),
        "decode_length_mode": decode_length_mode,
        "train_decode_length": float(train_decode_length),
        "min_decode_tokens": float(min_decode_tokens),
        "image_size": float(image_size),
        "device": device,
        "pairs": float(len(train_rows) + len(eval_rows)),
        "train_pairs": float(len(train_rows)),
        "eval_pairs": float(len(eval_rows)),
        "train_examples": float(len(train_examples)),
        "eval_examples": float(len(eval_examples)),
        "final_train_loss": float(history[-1]["train_loss"]) if history else 0.0,
        "final_train_token_loss": float(history[-1]["train_token_loss"]) if history else 0.0,
        "final_train_token_ppl": float(history[-1]["train_token_ppl"]) if history else 0.0,
        "final_train_image_loss": float(history[-1]["train_image_loss"]) if history else 0.0,
        "final_train_clip_loss": float(history[-1].get("train_clip_loss", 0.0)) if history else 0.0,
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
        "top1_condition_tanimoto": _mean_float(prediction_rows, "top1_condition_tanimoto"),
        "mean_best_condition_tanimoto": _mean_float(prediction_rows, "mean_best_condition_tanimoto"),
        "mean_candidate_count": _mean_float(prediction_rows, "candidate_count"),
        "generated_images": float(_count(prediction_rows, "generated_image_exists")),
        "generated_image_fraction": _fraction(_count(prediction_rows, "generated_image_exists"), total),
        "rendered_smiles_images": float(_count(prediction_rows, "rendered_smiles_image_exists")),
        "rendered_smiles_image_fraction": _fraction(_count(prediction_rows, "rendered_smiles_image_exists"), total),
        "image_compared": float(len(compared)),
        "image_compared_fraction": _fraction(len(compared), total),
        "image_exact_matches": float(_count(prediction_rows, "image_exact_match")),
        "image_exact_match_fraction": _fraction(_count(prediction_rows, "image_exact_match"), len(compared)),
        "image_mse_mean": float(sum(image_mse_values) / len(image_mse_values)) if image_mse_values else 0.0,
        "image_mse_max": float(max(image_mse_values)) if image_mse_values else 0.0,
        "smiles_render_image_compared": float(len(consistency_compared)),
        "smiles_render_image_compared_fraction": _fraction(len(consistency_compared), total),
        "smiles_render_image_exact_matches": float(_count(prediction_rows, "smiles_render_image_exact_match")),
        "smiles_render_image_exact_match_fraction": _fraction(_count(prediction_rows, "smiles_render_image_exact_match"), len(consistency_compared)),
        "smiles_render_image_mse_mean": float(sum(consistency_mse_values) / len(consistency_mse_values)) if consistency_mse_values else 0.0,
        "smiles_render_image_mse_max": float(max(consistency_mse_values)) if consistency_mse_values else 0.0,
        "paired_output_success": float(_count(prediction_rows, "paired_output_success")),
        "paired_output_success_fraction": _fraction(_count(prediction_rows, "paired_output_success"), total),
        "predictions": str(predictions_path),
        "sample_predictions": str(sample_predictions_path),
        "sample_contact_sheet": contact_sheet_path,
        "model": str(model_path),
        "vocab": str(vocab_path),
    }


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return float(sum(values) / len(values)) if values else 0.0


def build_arg_parser(description: str = "Run masked token diffusion for direct SMILES generation.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/masked_token_diffusion")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--min-mask-prob", type=float, default=0.15)
    parser.add_argument("--max-mask-prob", type=float, default=0.95)
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--rerank-mode", default="condition_fingerprint", choices=["beam", "none", "condition", "fingerprint", "condition_fingerprint", "condition_tanimoto"])
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--condition-tokens", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tokenization", default="smiles_token", choices=["char", "smiles", "smiles_token", "selfies"])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--image-loss-weight", type=float, default=0.0)
    parser.add_argument("--image-foreground-weight", type=float, default=8.0)
    parser.add_argument("--clip-loss-weight", type=float, default=0.0)
    parser.add_argument("--clip-temperature", type=float, default=0.07)
    parser.add_argument("--decode-length-mode", default="free", choices=sorted(DECODE_LENGTH_MODES))
    parser.add_argument("--min-decode-tokens", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=144)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--route-name", default="sketchmol_token_diffusion_masked_smiles")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    metrics = run_masked_token_diffusion(
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
