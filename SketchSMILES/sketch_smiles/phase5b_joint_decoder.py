"""Phase 5B joint SMILES/sketch decoder.

This experiment turns the paired-output idea into a real two-head model:
a shared molecular latent feeds both a SMILES decoder and a learned sketch-image
decoder. The image head predicts an ink map, which is saved as a white-background
molecular sketch for consistency checks.
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
from .phase5a0_oracle_baseline import (
    _canonicalize,
    _fraction,
    _image_pair_metrics,
    _render_smiles,
    _sample_rows,
    _split_rows,
    _write_rows,
)
from .phase5a1_learned_smiles_decoder import (
    BOS,
    EOS,
    PAD,
    _beam_search_smiles,
    _build_vocab,
    _canonical_candidate_list,
    _load_numpy,
    _load_torch,
    _make_fingerprint_fn,
    _normalize_decoding,
    _normalize_tokenization,
    _sample_smiles,
    _scaffold_match,
    _set_rdkit_error_logging,
    _set_seeds,
    _tanimoto,
    _tokenize_smiles,
)


def run_joint_paired_decoder(
    pair_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase5b_joint_decoder",
    train_fraction: float = 0.8,
    seed: int = 7,
    limit: int | None = None,
    fingerprint_bits: int = 2048,
    max_length: int = 128,
    hidden_dim: int = 512,
    latent_dim: int = 512,
    embedding_dim: int = 128,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    smiles_loss_weight: float = 1.0,
    image_loss_weight: float = 1.0,
    image_foreground_weight: float = 8.0,
    samples_per_condition: int = 8,
    temperature: float = 0.9,
    sample_top_k: int = 16,
    tokenization: str = "smiles_token",
    decoding: str = "beam",
    beam_size: int = 8,
    length_penalty: float = 0.0,
    image_size: int = 128,
    sample_count: int = 64,
    contact_sheet_cols: int = 4,
    contact_thumb_size: int = 128,
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
        raise RuntimeError("RDKit is required for Phase 5B fingerprints, rendering, and molecular metrics.")
    if not pillow:
        raise RuntimeError("Pillow is required for Phase 5B learned image targets and consistency metrics.")
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

    fingerprint_fn = _make_fingerprint_fn(rdkit, np=np, fingerprint_bits=fingerprint_bits)
    train_examples = _prepare_joint_examples(
        rows=train_rows,
        pair_dir=pair_dir,
        stoi=stoi,
        fingerprint_fn=fingerprint_fn,
        max_length=max_length,
        np=np,
        pillow=pillow,
        image_size=image_size,
        tokenization=tokenization,
    )
    eval_examples = _prepare_joint_examples(
        rows=eval_rows,
        pair_dir=pair_dir,
        stoi=stoi,
        fingerprint_fn=fingerprint_fn,
        max_length=max_length,
        np=np,
        pillow=pillow,
        image_size=image_size,
        tokenization=tokenization,
        allow_unknown=True,
    )
    if not train_examples:
        raise RuntimeError("No train examples available for Phase 5B.")
    if not eval_examples:
        raise RuntimeError("No eval examples available for Phase 5B.")

    resolved_device = _resolve_device(device, torch)
    model = JointSmilesSketchDecoder(
        vocab_size=len(itos),
        feature_dim=fingerprint_bits,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        pad_idx=stoi[PAD],
        image_size=image_size,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = _train_joint_model(
        model=model,
        examples=train_examples,
        optimizer=optimizer,
        torch=torch,
        np=np,
        device=resolved_device,
        batch_size=batch_size,
        epochs=epochs,
        pad_idx=stoi[PAD],
        smiles_loss_weight=smiles_loss_weight,
        image_loss_weight=image_loss_weight,
        image_foreground_weight=image_foreground_weight,
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
                "latent_dim": latent_dim,
                "pad_idx": stoi[PAD],
                "image_size": image_size,
            },
        },
        model_path,
    )

    learned_image_dir = output_dir / "joint_images"
    rendered_smiles_image_dir = output_dir / "rendered_smiles_images"
    resized_target_image_dir = output_dir / "target_images_resized"
    learned_image_dir.mkdir(parents=True, exist_ok=True)
    rendered_smiles_image_dir.mkdir(parents=True, exist_ok=True)
    resized_target_image_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = _evaluate_joint_model(
        model=model,
        eval_examples=eval_examples,
        stoi=stoi,
        itos=itos,
        learned_image_dir=learned_image_dir,
        rendered_smiles_image_dir=rendered_smiles_image_dir,
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
    contact_sheet_path = _write_joint_contact_sheet(
        sample_rows=sample_rows,
        pillow=pillow,
        cols=contact_sheet_cols,
        thumb_size=contact_thumb_size,
        output_path=output_dir / "sample_contact_sheet.png",
    )

    metrics = _summarize_joint_decoder(
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
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        smiles_loss_weight=smiles_loss_weight,
        image_loss_weight=image_loss_weight,
        image_foreground_weight=image_foreground_weight,
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
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "phase5b_shared_latent_smiles_sketch_decoder",
                "research_question": "Can one shared molecular latent jointly emit a machine-readable SMILES string and a visually consistent molecular sketch, avoiding a generated-image-to-OCR bottleneck?",
                "pair_dir": str(pair_dir),
                "pairs_csv": str(pairs_path),
                "output_dir": str(output_dir),
                "train_fraction": train_fraction,
                "seed": seed,
                "limit": limit,
                "fingerprint_bits": fingerprint_bits,
                "max_length": max_length,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "embedding_dim": embedding_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "smiles_loss_weight": smiles_loss_weight,
                "image_loss_weight": image_loss_weight,
                "image_foreground_weight": image_foreground_weight,
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


class JointSmilesSketchDecoder:
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
                latent_dim: int,
                pad_idx: int,
                image_size: int,
            ) -> None:
                super().__init__()
                if image_size % 16 != 0:
                    raise ValueError(f"image_size={image_size} must be divisible by 16 for the sketch decoder.")
                self.pad_idx = pad_idx
                self.image_size = image_size
                self.base_size = max(4, image_size // 16)
                base_channels = max(32, min(128, hidden_dim // 4))
                self.feature_encoder = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                )
                self.smiles_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh())
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
                self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
                self.out = nn.Linear(hidden_dim, vocab_size)
                self.image_seed = nn.Sequential(
                    nn.Linear(latent_dim, base_channels * self.base_size * self.base_size),
                    nn.GELU(),
                )
                self.image_decoder = nn.Sequential(
                    nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(base_channels // 8, 1, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid(),
                )

            def encode(self, features: Any) -> Any:
                return self.feature_encoder(features)

            def forward(self, features: Any, input_ids: Any) -> tuple[Any, Any]:
                latent = self.encode(features)
                h0 = self.smiles_hidden(latent).unsqueeze(0)
                embeddings = self.embedding(input_ids)
                output, _ = self.gru(embeddings, h0)
                token_logits = self.out(output)
                image = self.decode_image_from_latent(latent)
                return token_logits, image

            def decode_image_from_latent(self, latent: Any) -> Any:
                seed = self.image_seed(latent).view(latent.shape[0], -1, self.base_size, self.base_size)
                return self.image_decoder(seed)

            def generate_image(self, features: Any) -> Any:
                return self.decode_image_from_latent(self.encode(features))

            def decode_prefix_logits(self, feature_tensor: Any, ids: list[int]) -> Any:
                input_ids = torch.tensor([ids], dtype=torch.long, device=feature_tensor.device)
                logits, _image = self.forward(feature_tensor, input_ids)
                return logits[:, -1, :]

        return _Model(*args, **kwargs)


def _train_joint_model(
    model: Any,
    examples: list[dict[str, Any]],
    optimizer: Any,
    torch: Any,
    np: Any,
    device: Any,
    batch_size: int,
    epochs: int,
    pad_idx: int,
    smiles_loss_weight: float,
    image_loss_weight: float,
    image_foreground_weight: float,
    seed: int,
) -> list[dict[str, float]]:
    token_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    history: list[dict[str, float]] = []
    rng = random.Random(seed)
    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        model.train()
        total_token_loss = 0.0
        total_image_loss = 0.0
        total_loss = 0.0
        total_tokens = 0
        total_examples = 0
        for start in range(0, len(order), int(batch_size)):
            batch = [examples[idx] for idx in order[start : start + int(batch_size)]]
            features = torch.as_tensor(np.stack([row["feature"] for row in batch]), dtype=torch.float32, device=device)
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            target_ids = torch.tensor([row["target_ids"] for row in batch], dtype=torch.long, device=device)
            target_images = torch.as_tensor(np.stack([row["image"] for row in batch]), dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_images = model(features, input_ids)
            token_loss = token_loss_fn(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            image_loss = _weighted_image_mse(predicted_images, target_images, image_foreground_weight=image_foreground_weight)
            loss = float(smiles_loss_weight) * token_loss + float(image_loss_weight) * image_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            token_count = int((target_ids != pad_idx).sum().item())
            batch_size_actual = len(batch)
            total_token_loss += float(token_loss.item()) * max(1, token_count)
            total_image_loss += float(image_loss.item()) * batch_size_actual
            total_loss += float(loss.item()) * batch_size_actual
            total_tokens += token_count
            total_examples += batch_size_actual
        mean_token_loss = total_token_loss / max(1, total_tokens)
        mean_image_loss = total_image_loss / max(1, total_examples)
        mean_loss = total_loss / max(1, total_examples)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(mean_loss),
                "train_token_loss": float(mean_token_loss),
                "train_token_ppl": float(math.exp(min(20.0, mean_token_loss))),
                "train_image_loss": float(mean_image_loss),
            }
        )
        print(
            f"  epoch={epoch} train_loss={mean_loss:.4f} train_token_loss={mean_token_loss:.4f} "
            f"train_token_ppl={math.exp(min(20.0, mean_token_loss)):.3f} train_image_loss={mean_image_loss:.5f}",
            flush=True,
        )
    return history


def _evaluate_joint_model(
    model: Any,
    eval_examples: list[dict[str, Any]],
    stoi: dict[str, int],
    itos: list[str],
    learned_image_dir: Path,
    rendered_smiles_image_dir: Path,
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
                    feature=example["feature"],
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
                    feature=example["feature"],
                    stoi=stoi,
                    itos=itos,
                    torch=torch,
                    device=device,
                    max_length=max_length,
                    samples_per_condition=samples_per_condition,
                    temperature=temperature,
                    sample_top_k=sample_top_k,
                )
            feature_tensor = torch.as_tensor(example["feature"], dtype=torch.float32, device=device).unsqueeze(0)
            predicted_ink = model.generate_image(feature_tensor)[0].detach().cpu().numpy()

            pair_id = example["pair_id"]
            target_smiles = example["smiles"]
            candidate_smiles = _canonical_candidate_list(raw_samples, rdkit)
            top1_smiles = candidate_smiles[0] if candidate_smiles else ""
            top1_valid = bool(top1_smiles)

            learned_image_path = learned_image_dir / f"{pair_id}.png"
            rendered_smiles_image_path = rendered_smiles_image_dir / f"{pair_id}.png"
            resized_target_image_path = resized_target_image_dir / f"{pair_id}.png"
            _save_ink_image(predicted_ink, learned_image_path, pillow)
            _save_ink_image(example["image"], resized_target_image_path, pillow)
            render_error = _render_smiles(top1_smiles, rendered_smiles_image_path, image_size=image_size, rdkit=rdkit)

            target_mol = rdkit["Chem"].MolFromSmiles(target_smiles)
            top1_mol = rdkit["Chem"].MolFromSmiles(top1_smiles) if top1_smiles else None
            top1_tanimoto = _tanimoto(target_mol, top1_mol, rdkit)
            best_tanimoto = max((_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidate_smiles), default=0.0)
            target_image_metrics = _image_pair_metrics(resized_target_image_path, learned_image_path, pillow)
            consistency_metrics = _prefix_metrics(_image_pair_metrics(rendered_smiles_image_path, learned_image_path, pillow), "smiles_render")
            scaffold_match = _scaffold_match(target_smiles, top1_smiles, rdkit)
            rendered_smiles_image_exists = rendered_smiles_image_path.exists()
            learned_image_exists = learned_image_path.exists()
            rows.append(
                {
                    "pair_id": pair_id,
                    "target_smiles": target_smiles,
                    "generated_smiles": top1_smiles,
                    "raw_samples": "|".join(raw_samples),
                    "canonical_candidates": "|".join(candidate_smiles),
                    "candidate_count": float(len(candidate_smiles)),
                    "top1_valid": top1_valid,
                    "top1_exact_match": bool(top1_smiles == target_smiles),
                    "topk_exact_match": bool(target_smiles in candidate_smiles),
                    "top1_target_tanimoto": float(top1_tanimoto),
                    "mean_best_tanimoto": float(best_tanimoto),
                    "top1_scaffold_match": bool(scaffold_match),
                    "target_image_path": str(resized_target_image_path),
                    "source_target_image_path": example["image_path"],
                    "generated_image_path": str(learned_image_path),
                    "rendered_smiles_image_path": str(rendered_smiles_image_path),
                    "generated_image_exists": learned_image_exists,
                    "rendered_smiles_image_exists": rendered_smiles_image_exists,
                    "render_error": render_error,
                    "paired_output_success": bool(top1_valid and learned_image_exists and rendered_smiles_image_exists and consistency_metrics["smiles_render_image_compared"]),
                    **target_image_metrics,
                    **consistency_metrics,
                }
            )
    return rows


def _prepare_joint_examples(
    rows: list[dict[str, str]],
    pair_dir: Path,
    stoi: dict[str, int],
    fingerprint_fn: Any,
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
        feature = fingerprint_fn(smiles)
        if feature is None:
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
                "feature": np.asarray(feature, dtype=np.float32),
                "image": image,
                "input_ids": input_ids,
                "target_ids": target_ids,
            }
        )
    return examples


def _encode_tokens(tokens: list[str], stoi: dict[str, int], max_length: int) -> tuple[list[int], list[int]]:
    input_tokens = [BOS] + tokens
    target_tokens = tokens + [EOS]
    input_ids = [stoi.get(token, stoi[PAD]) for token in input_tokens][:max_length]
    target_ids = [stoi.get(token, stoi[PAD]) for token in target_tokens][:max_length]
    while len(input_ids) < max_length:
        input_ids.append(stoi[PAD])
    while len(target_ids) < max_length:
        target_ids.append(stoi[PAD])
    return input_ids, target_ids


def _load_ink_tensor(image_path: Path | None, pillow: dict[str, Any], image_size: int, np: Any) -> Any | None:
    if image_path is None or not image_path.exists():
        return None
    try:
        with pillow["Image"].open(image_path) as image:
            image = image.convert("L")
            image = image.resize((int(image_size), int(image_size)), _resampling_lanczos(pillow))
            arr = np.asarray(image, dtype=np.float32) / 255.0
        ink = 1.0 - arr
        return ink.reshape(1, int(image_size), int(image_size)).astype(np.float32)
    except Exception:
        return None


def _save_ink_image(ink: Any, output_path: Path, pillow: dict[str, Any]) -> None:
    import numpy as np

    arr = np.asarray(ink, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(1.0 - arr, 0.0, 1.0)
    image_arr = (arr * 255.0).round().astype("uint8")
    image = pillow["Image"].fromarray(image_arr, mode="L").convert("RGB")
    image.save(output_path)


def _weighted_image_mse(predicted: Any, target: Any, image_foreground_weight: float) -> Any:
    weights = 1.0 + float(image_foreground_weight) * target
    return (((predicted - target) ** 2) * weights).mean()


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _resampling_lanczos(pillow: dict[str, Any]) -> Any:
    image_cls = pillow["Image"]
    try:
        return image_cls.Resampling.LANCZOS
    except AttributeError:
        return image_cls.LANCZOS


def _summarize_joint_decoder(
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
    latent_dim: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    smiles_loss_weight: float,
    image_loss_weight: float,
    image_foreground_weight: float,
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
    consistency_compared = [row for row in prediction_rows if row["smiles_render_image_compared"]]
    image_mse_values = [float(row["image_mse"]) for row in compared if row["image_mse"] != ""]
    consistency_mse_values = [float(row["smiles_render_image_mse"]) for row in consistency_compared if row["smiles_render_image_mse"] != ""]
    return {
        "phase": "phase5b_shared_latent_smiles_sketch_decoder",
        "pair_dir": str(pair_dir),
        "output_dir": str(output_dir),
        "train_fraction": float(train_fraction),
        "seed": float(seed),
        "fingerprint_bits": float(fingerprint_bits),
        "max_length": float(max_length),
        "hidden_dim": float(hidden_dim),
        "latent_dim": float(latent_dim),
        "embedding_dim": float(embedding_dim),
        "epochs": float(epochs),
        "batch_size": float(batch_size),
        "learning_rate": float(learning_rate),
        "smiles_loss_weight": float(smiles_loss_weight),
        "image_loss_weight": float(image_loss_weight),
        "image_foreground_weight": float(image_foreground_weight),
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
        "final_train_loss": float(history[-1]["train_loss"]) if history else 0.0,
        "final_train_token_loss": float(history[-1]["train_token_loss"]) if history else 0.0,
        "final_train_token_ppl": float(history[-1]["train_token_ppl"]) if history else 0.0,
        "final_train_image_loss": float(history[-1]["train_image_loss"]) if history else 0.0,
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


def _write_joint_contact_sheet(
    sample_rows: list[dict[str, Any]],
    pillow: dict[str, Any],
    cols: int,
    thumb_size: int,
    output_path: Path,
) -> str:
    if not sample_rows:
        return ""
    image_cls = pillow["Image"]
    draw_cls = pillow["ImageDraw"]
    cols = max(1, int(cols))
    rows = int(math.ceil(len(sample_rows) / cols))
    label_height = 52
    cell_w = int(thumb_size * 3)
    cell_h = int(thumb_size + label_height)
    sheet = image_cls.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = draw_cls.Draw(sheet)
    for idx, row in enumerate(sample_rows):
        x0 = (idx % cols) * cell_w
        y0 = (idx // cols) * cell_h
        _paste_thumb(image_cls, sheet, row.get("target_image_path", ""), x0, y0, thumb_size)
        _paste_thumb(image_cls, sheet, row.get("generated_image_path", ""), x0 + thumb_size, y0, thumb_size)
        _paste_thumb(image_cls, sheet, row.get("rendered_smiles_image_path", ""), x0 + 2 * thumb_size, y0, thumb_size)
        label = f"{row.get('pair_id', '')} tan={float(row.get('top1_target_tanimoto', 0.0)):.2f} img={row.get('image_mse', '')}"[:64]
        draw.text((x0 + 4, y0 + thumb_size + 4), "target | learned | rendered-smiles", fill=(0, 0, 0))
        draw.text((x0 + 4, y0 + thumb_size + 22), label, fill=(0, 0, 0))
    sheet.save(output_path)
    return str(output_path)


def _paste_thumb(image_cls: Any, sheet: Any, image_path: str, x0: int, y0: int, thumb_size: int) -> None:
    try:
        with image_cls.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_size, thumb_size))
            x = x0 + (thumb_size - image.width) // 2
            y = y0 + (thumb_size - image.height) // 2
            sheet.paste(image, (x, y))
    except Exception:
        return


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
    parser = argparse.ArgumentParser(description="Run Phase 5B shared-latent joint SMILES/sketch decoder.")
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase5b_joint_decoder")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--smiles-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-foreground-weight", type=float, default=8.0)
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--tokenization", default="smiles_token", choices=["char", "smiles", "smiles_token"])
    parser.add_argument("--decoding", default="beam", choices=["sample", "beam"])
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--contact-thumb-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metrics = run_joint_paired_decoder(
        pair_dir=args.pair_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        seed=args.seed,
        limit=args.limit,
        fingerprint_bits=args.fingerprint_bits,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        smiles_loss_weight=args.smiles_loss_weight,
        image_loss_weight=args.image_loss_weight,
        image_foreground_weight=args.image_foreground_weight,
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
