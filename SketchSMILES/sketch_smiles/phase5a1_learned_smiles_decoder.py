"""Phase 5A-1 oracle-conditioned learned SMILES decoder.

This is the first learned paired-output baseline for SketchSMILES. The model is
conditioned on an oracle molecular fingerprint, generates SMILES directly, and
renders the top prediction into a molecular sketch for consistency evaluation.
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
    _write_oracle_contact_sheet,
    _write_rows,
)

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"


def run_learned_smiles_decoder(
    pair_dir: str | Path,
    output_dir: str | Path = "outputs/runs/phase5a1_learned_smiles_decoder",
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
    samples_per_condition: int = 8,
    temperature: float = 0.9,
    sample_top_k: int = 16,
    image_size: int = 256,
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
        raise RuntimeError("RDKit is required for Phase 5A-1 oracle molecular fingerprints and rendering.")
    if not pillow:
        raise RuntimeError("Pillow is required for Phase 5A-1 image consistency metrics.")
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

    stoi, itos = _build_vocab(train_rows)
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps({"stoi": stoi, "itos": itos}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fingerprint_fn = _make_fingerprint_fn(rdkit, np=np, fingerprint_bits=fingerprint_bits)
    train_examples = _prepare_examples(train_rows, stoi, fingerprint_fn, max_length=max_length, np=np)
    eval_examples = _prepare_examples(eval_rows, stoi, fingerprint_fn, max_length=max_length, np=np, allow_unknown=True)
    if not train_examples:
        raise RuntimeError("No train examples available for Phase 5A-1.")
    if not eval_examples:
        raise RuntimeError("No eval examples available for Phase 5A-1.")

    resolved_device = _resolve_device(device, torch)
    model = ConditionalSmilesGRU(
        vocab_size=len(itos),
        feature_dim=fingerprint_bits,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_idx=stoi[PAD],
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = _train_model(
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
                "feature_dim": fingerprint_bits,
                "vocab_size": len(itos),
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "pad_idx": stoi[PAD],
            },
        },
        model_path,
    )

    generated_image_dir = output_dir / "generated_images"
    generated_image_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = _evaluate_model(
        model=model,
        eval_examples=eval_examples,
        stoi=stoi,
        itos=itos,
        pair_dir=pair_dir,
        generated_image_dir=generated_image_dir,
        rdkit=rdkit,
        pillow=pillow,
        torch=torch,
        device=resolved_device,
        max_length=max_length,
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
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

    metrics = _summarize_learned_decoder(
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
        samples_per_condition=samples_per_condition,
        temperature=temperature,
        sample_top_k=sample_top_k,
        image_size=image_size,
        device=str(resolved_device),
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "phase": "phase5a1_oracle_conditioned_learned_smiles_decoder",
                "research_question": "Can a learned decoder emit machine-readable SMILES directly from an oracle molecular condition while retaining paired sketch consistency through rendering?",
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
                "samples_per_condition": samples_per_condition,
                "temperature": temperature,
                "sample_top_k": sample_top_k,
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


class ConditionalSmilesGRU:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _load_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self, vocab_size: int, feature_dim: int, embedding_dim: int, hidden_dim: int, pad_idx: int) -> None:
                super().__init__()
                self.pad_idx = pad_idx
                self.cond = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
                self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
                self.out = nn.Linear(hidden_dim, vocab_size)

            def forward(self, features: Any, input_ids: Any) -> Any:
                h0 = self.cond(features).unsqueeze(0)
                embeddings = self.embedding(input_ids)
                output, _ = self.gru(embeddings, h0)
                return self.out(output)

        return _Model(*args, **kwargs)


def _train_model(
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
            features = torch.as_tensor(np.stack([row["feature"] for row in batch]), dtype=torch.float32, device=device)
            input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
            target_ids = torch.tensor([row["target_ids"] for row in batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, input_ids)
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


def _evaluate_model(
    model: Any,
    eval_examples: list[dict[str, Any]],
    stoi: dict[str, int],
    itos: list[str],
    pair_dir: Path,
    generated_image_dir: Path,
    rdkit: dict[str, Any],
    pillow: dict[str, Any],
    torch: Any,
    device: Any,
    max_length: int,
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
    image_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in eval_examples:
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
            candidate_smiles = _canonical_candidate_list(raw_samples, rdkit)
            target_smiles = example["smiles"]
            top1_smiles = candidate_smiles[0] if candidate_smiles else ""
            top1_valid = bool(top1_smiles)
            target_mol = rdkit["Chem"].MolFromSmiles(target_smiles)
            top1_mol = rdkit["Chem"].MolFromSmiles(top1_smiles) if top1_smiles else None
            top1_tanimoto = _tanimoto(target_mol, top1_mol, rdkit)
            best_tanimoto = max((_tanimoto(target_mol, rdkit["Chem"].MolFromSmiles(smiles), rdkit) for smiles in candidate_smiles), default=0.0)
            topk_exact = target_smiles in candidate_smiles
            generated_image_path = generated_image_dir / f"{example['pair_id']}.png"
            render_error = _render_smiles(top1_smiles, generated_image_path, image_size=image_size, rdkit=rdkit)
            target_image_path = _resolve_image_path(example["image_path"], pair_dir)
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
                    "top1_valid": top1_valid,
                    "top1_exact_match": bool(top1_smiles == target_smiles),
                    "topk_exact_match": bool(topk_exact),
                    "top1_target_tanimoto": float(top1_tanimoto),
                    "mean_best_tanimoto": float(best_tanimoto),
                    "top1_scaffold_match": bool(scaffold_match),
                    "target_image_path": str(target_image_path) if target_image_path else "",
                    "generated_image_path": str(generated_image_path),
                    "generated_image_exists": generated_image_path.exists(),
                    "render_error": render_error,
                    "paired_output_success": bool(top1_valid and generated_image_path.exists()),
                    **image_metrics,
                }
            )
    return rows


def _sample_smiles(
    model: Any,
    feature: list[float],
    stoi: dict[str, int],
    itos: list[str],
    torch: Any,
    device: Any,
    max_length: int,
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
) -> list[str]:
    samples: list[str] = []
    feature_tensor = torch.as_tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(int(samples_per_condition)):
        generated: list[str] = []
        token = torch.tensor([[stoi[BOS]]], dtype=torch.long, device=device)
        h = model.cond(feature_tensor).unsqueeze(0)
        for _step in range(int(max_length)):
            embedding = model.embedding(token[:, -1:])
            output, h = model.gru(embedding, h)
            logits = model.out(output[:, -1, :]) / max(float(temperature), 1e-6)
            next_id = int(_sample_token(logits[0], torch=torch, top_k=sample_top_k).item())
            if itos[next_id] == EOS:
                break
            if itos[next_id] not in (PAD, BOS):
                generated.append(itos[next_id])
            token = torch.cat([token, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        samples.append("".join(generated))
    return samples


def _sample_token(logits: Any, torch: Any, top_k: int) -> Any:
    if top_k and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        values, indices = torch.topk(logits, int(top_k))
        probs = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, 1)
        return indices[sampled]
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def _canonical_candidate_list(raw_samples: list[str], rdkit: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for sample in raw_samples:
        smiles, _error = _canonicalize(sample, rdkit)
        if smiles and smiles not in seen:
            seen.add(smiles)
            candidates.append(smiles)
    return candidates


def _summarize_learned_decoder(
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
    samples_per_condition: int,
    temperature: float,
    sample_top_k: int,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    total = len(prediction_rows)
    compared = [row for row in prediction_rows if row["image_compared"]]
    image_mse_values = [float(row["image_mse"]) for row in compared if row["image_mse"] != ""]
    return {
        "phase": "phase5a1_oracle_conditioned_learned_smiles_decoder",
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
        "samples_per_condition": float(samples_per_condition),
        "temperature": float(temperature),
        "sample_top_k": float(sample_top_k),
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


def _prepare_examples(
    rows: list[dict[str, str]],
    stoi: dict[str, int],
    fingerprint_fn: Any,
    max_length: int,
    np: Any,
    allow_unknown: bool = False,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        smiles = row.get("canonical_smiles") or row.get("input_smiles", "")
        if not smiles:
            continue
        if len(smiles) + 1 > max_length:
            continue
        if not allow_unknown and any(ch not in stoi for ch in smiles):
            continue
        feature = fingerprint_fn(smiles)
        if feature is None:
            continue
        input_ids, target_ids = _encode_smiles(smiles, stoi=stoi, max_length=max_length)
        examples.append(
            {
                "pair_id": row.get("pair_id", ""),
                "smiles": smiles,
                "image_path": row.get("image_path", ""),
                "feature": np.asarray(feature, dtype=np.float32),
                "input_ids": input_ids,
                "target_ids": target_ids,
            }
        )
    return examples


def _encode_smiles(smiles: str, stoi: dict[str, int], max_length: int) -> tuple[list[int], list[int]]:
    input_tokens = [BOS] + list(smiles)
    target_tokens = list(smiles) + [EOS]
    input_ids = [stoi.get(token, stoi[PAD]) for token in input_tokens][:max_length]
    target_ids = [stoi.get(token, stoi[PAD]) for token in target_tokens][:max_length]
    while len(input_ids) < max_length:
        input_ids.append(stoi[PAD])
    while len(target_ids) < max_length:
        target_ids.append(stoi[PAD])
    return input_ids, target_ids


def _build_vocab(rows: list[dict[str, str]]) -> tuple[dict[str, int], list[str]]:
    chars = sorted({ch for row in rows for ch in (row.get("canonical_smiles") or row.get("input_smiles", ""))})
    itos = [PAD, BOS, EOS] + chars
    stoi = {token: idx for idx, token in enumerate(itos)}
    return stoi, itos


def _make_fingerprint_fn(rdkit: dict[str, Any], np: Any, fingerprint_bits: int) -> Any:
    generator = None
    try:
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=int(fingerprint_bits))
    except Exception:
        generator = None

    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    def _fingerprint(smiles: str) -> Any:
        mol = rdkit["Chem"].MolFromSmiles(smiles)
        if mol is None:
            return None
        if generator is not None:
            fp = generator.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=int(fingerprint_bits))
        arr = np.zeros((int(fingerprint_bits),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    return _fingerprint


def _tanimoto(left_mol: Any, right_mol: Any, rdkit: dict[str, Any]) -> float:
    if left_mol is None or right_mol is None:
        return 0.0
    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem

        left = AllChem.GetMorganFingerprintAsBitVect(left_mol, radius=2, nBits=2048)
        right = AllChem.GetMorganFingerprintAsBitVect(right_mol, radius=2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(left, right))
    except Exception:
        return 0.0


def _scaffold_match(target_smiles: str, generated_smiles: str, rdkit: dict[str, Any]) -> bool:
    if not target_smiles or not generated_smiles:
        return False
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        target = rdkit["Chem"].MolFromSmiles(target_smiles)
        generated = rdkit["Chem"].MolFromSmiles(generated_smiles)
        if target is None or generated is None:
            return False
        target_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=target)
        generated_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=generated)
        return bool(target_scaffold and target_scaffold == generated_scaffold)
    except Exception:
        return False


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return float(sum(values) / len(values)) if values else 0.0


def _load_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required for Phase 5A-1: {exc}") from exc


def _load_numpy() -> Any:
    try:
        import numpy as np

        return np
    except Exception as exc:
        raise RuntimeError(f"NumPy is required for Phase 5A-1: {exc}") from exc


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seeds(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_rdkit_error_logging(enabled: bool) -> None:
    try:
        from rdkit import RDLogger

        if enabled:
            RDLogger.EnableLog("rdApp.error")
        else:
            RDLogger.DisableLog("rdApp.error")
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5A-1 oracle-conditioned learned SMILES decoder.")
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/runs/phase5a1_learned_smiles_decoder")
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
    parser.add_argument("--samples-per-condition", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--contact-sheet-cols", type=int, default=8)
    parser.add_argument("--contact-thumb-size", type=int, default=144)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    metrics = run_learned_smiles_decoder(
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
        samples_per_condition=args.samples_per_condition,
        temperature=args.temperature,
        sample_top_k=args.sample_top_k,
        image_size=args.image_size,
        sample_count=args.sample_count,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_thumb_size=args.contact_thumb_size,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
