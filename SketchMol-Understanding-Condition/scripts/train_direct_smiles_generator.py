#!/usr/bin/env python3
"""Train/evaluate an MLLM-conditioned direct SMILES generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.direct_smiles_generation import (  # noqa: E402
    UNK,
    ConditionedSmilesDecoder,
    SmilesVocabulary,
    build_vocabulary,
    detokenize_smiles,
    tokenize_smiles,
)
from sketchmol_understanding_condition.chem import canonical_smiles, molecular_properties  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import PROPERTY_COLUMNS  # noqa: E402
from sketchmol_understanding_condition.univideo_molecule import FrozenConditionFeatureStore  # noqa: E402


PROPERTY_NORMALIZERS = {
    "MW": 500.0,
    "LogP": 6.0,
    "QED": 1.0,
    "TPSA": 160.0,
    "HBD": 8.0,
    "HBA": 12.0,
    "RB": 12.0,
    "SA": 8.0,
}
SKETCHMOL_STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
}
PROPERTY_VALUE_KEYS = {
    "MW": "MolWt",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "rotatable",
    "SA": "SA",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--eval-condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", default="query_tokens")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument(
        "--condition-mixing-mode",
        choices=("features_only", "append_property_program", "append_source_property_program", "property_program_only"),
        default="features_only",
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--reset-training-state",
        action="store_true",
        help="Load model/vocab weights from --resume-checkpoint but restart epoch, history, and optimizer state.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-smiles-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--parallel-samples", type=int, default=1)
    parser.add_argument("--max-parallel-sequences", type=int, default=1024)
    parser.add_argument("--disable-property-rerank", action="store_true")
    parser.add_argument("--property-count-curriculum-sampling", action="store_true")
    parser.add_argument("--property-count-curriculum-loss", action="store_true")
    parser.add_argument("--property-count-curriculum-power", type=float, default=1.0)
    parser.add_argument("--property-count-curriculum-baseline", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument(
        "--candidate-output-csv",
        type=Path,
        default=None,
        help="Optional CSV with every sampled candidate before final selection/rerank.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    train_rows = read_rows(args.train_csv, limit=args.limit) if args.train_csv else []
    eval_rows = read_rows(args.eval_csv, limit=args.eval_limit) if args.eval_csv else []
    if args.eval_only and args.resume_checkpoint is None:
        raise ValueError("--eval-only requires --resume-checkpoint")
    if not args.eval_only and not train_rows:
        raise ValueError("--train-csv is required unless --eval-only is set")

    train_store = load_store(args.condition_features_dir, args)
    eval_store = load_store(args.eval_condition_features_dir or args.condition_features_dir, args)

    checkpoint = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    checkpoint_args = dict(checkpoint.get("args", {})) if checkpoint else {}
    condition_mixing_mode = resolve_condition_mixing_mode(args, checkpoint_args)
    if checkpoint is not None:
        vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
        config = dict(checkpoint["model_config"])
        condition_dim = int(config["condition_dim"])
    else:
        vocab = build_vocabulary([row.get("target_smiles", "") for row in train_rows + eval_rows])
        condition_dim = infer_condition_dim(train_store, eval_store)
        config = {
            "vocab_size": len(vocab.token_to_id),
            "condition_dim": condition_dim,
            "d_model": int(args.d_model),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dim_feedforward": int(args.dim_feedforward),
            "dropout": float(args.dropout),
            "pad_id": vocab.pad_id,
            "max_length": int(args.max_smiles_length) + 8,
        }

    model = ConditionedSmilesDecoder(**config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])

    train_dataset = build_dataset(
        train_rows,
        vocab,
        train_store,
        condition_dim,
        max_smiles_length=args.max_smiles_length,
        condition_mixing_mode=condition_mixing_mode,
    )
    eval_dataset = build_dataset(
        eval_rows,
        vocab,
        eval_store,
        condition_dim,
        max_smiles_length=args.max_smiles_length,
        condition_mixing_mode=condition_mixing_mode,
    )

    warm_start = bool(args.reset_training_state) and checkpoint is not None and not args.eval_only
    history: list[dict[str, object]] = [] if warm_start else (list(checkpoint.get("history", [])) if checkpoint else [])
    start_epoch = 1 if warm_start else (int(checkpoint.get("epoch", 0)) + 1 if checkpoint and not args.eval_only else 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if checkpoint is not None and checkpoint.get("optimizer_state") and not args.eval_only and not warm_start:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if not args.eval_only:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            record = train_epoch(
                model,
                train_dataset,
                optimizer,
                batch_size=int(args.batch_size),
                device=device,
                seed=int(args.seed) + epoch,
                grad_clip=float(args.grad_clip),
                property_count_curriculum_sampling=bool(args.property_count_curriculum_sampling),
                property_count_curriculum_loss=bool(args.property_count_curriculum_loss),
                property_count_curriculum_power=float(args.property_count_curriculum_power),
                property_count_curriculum_baseline=float(args.property_count_curriculum_baseline),
            )
            record["epoch"] = epoch
            if eval_dataset:
                record.update(
                    {
                        f"eval_{key}": value
                        for key, value in evaluate_loss(
                            model,
                            eval_dataset,
                            batch_size=int(args.eval_batch_size),
                            device=device,
                        ).items()
                    }
                )
            history.append(record)
            save_checkpoint(args.output_dir / "latest_checkpoint.pt", model, optimizer, vocab, config, epoch, history, args)

    final_checkpoint = args.output_dir / "direct_smiles_generator.pt"
    save_checkpoint(
        final_checkpoint,
        model,
        optimizer if not args.eval_only else None,
        vocab,
        config,
        int(history[-1].get("epoch", 0)) if history else int(checkpoint.get("epoch", 0) if checkpoint else 0),
        history,
        args,
    )

    prediction_csv = args.prediction_csv or args.output_dir / "direct_smiles_predictions.csv"
    prediction_summary: dict[str, object] | None = None
    if eval_dataset:
        prediction_summary = write_predictions(
            model,
            eval_dataset,
            eval_rows,
            vocab,
            prediction_csv,
            batch_size=int(args.eval_batch_size),
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            num_samples=int(args.num_samples),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            min_new_tokens=int(args.min_new_tokens),
            parallel_samples=int(args.parallel_samples),
            max_parallel_sequences=int(args.max_parallel_sequences),
            property_rerank=not bool(args.disable_property_rerank),
            condition_mixing_mode=condition_mixing_mode,
            candidate_output_csv=args.candidate_output_csv,
        )

    summary = {
        "train_csv": str(args.train_csv) if args.train_csv else None,
        "eval_csv": str(args.eval_csv) if args.eval_csv else None,
        "condition_features_dir": str(args.condition_features_dir) if args.condition_features_dir else None,
        "eval_condition_features_dir": str(args.eval_condition_features_dir or args.condition_features_dir)
        if (args.eval_condition_features_dir or args.condition_features_dir)
        else None,
        "output_dir": str(args.output_dir),
        "checkpoint": str(final_checkpoint),
        "prediction_csv": str(prediction_csv) if eval_dataset else None,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "vocab_size": len(vocab.token_to_id),
        "condition_dim": int(config["condition_dim"]),
        "history": history,
        "prediction_summary": prediction_summary,
        "condition_mixing_mode": condition_mixing_mode,
        "reset_training_state": bool(args.reset_training_state),
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def read_rows(path: Path | None, *, limit: int = 0) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit and limit > 0 else rows


def load_store(path: Path | None, args: argparse.Namespace) -> FrozenConditionFeatureStore | None:
    if path is None:
        return None
    return FrozenConditionFeatureStore(
        path,
        array_name=str(args.condition_feature_array),
        variant=str(args.condition_feature_variant),
    )


def load_checkpoint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def infer_condition_dim(*stores: FrozenConditionFeatureStore | None) -> int:
    for store in stores:
        if store is not None:
            return int(store.input_hidden_dim)
    return 32


def resolve_condition_mixing_mode(args: argparse.Namespace, checkpoint_args: Mapping[str, object]) -> str:
    current = str(getattr(args, "condition_mixing_mode", "features_only") or "features_only")
    saved = str(checkpoint_args.get("condition_mixing_mode", "") or "").strip()
    if current == "features_only" and saved:
        return saved
    return current


def build_dataset(
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    store: FrozenConditionFeatureStore | None,
    condition_dim: int,
    *,
    max_smiles_length: int,
    condition_mixing_mode: str = "features_only",
) -> list[dict[str, object]]:
    dataset = []
    for row in rows:
        target = str(row.get("target_smiles", "") or "").strip()
        if not target:
            continue
        tokens = tokenize_smiles(target)[: max(1, int(max_smiles_length))]
        decoder_input = vocab.encode(tokens, add_bos=True, add_eos=False)
        target_ids = vocab.encode(tokens, add_bos=False, add_eos=True)
        condition = condition_array_for_row(row, store, condition_dim, condition_mixing_mode=condition_mixing_mode)
        dataset.append(
            {
                "condition": condition.astype(np.float32),
                "decoder_input_ids": np.asarray(decoder_input, dtype=np.int64),
                "target_ids": np.asarray(target_ids, dtype=np.int64),
                "property_count": row_property_count(row),
            }
        )
    return dataset


def condition_array_for_row(
    row: Mapping[str, str],
    store: FrozenConditionFeatureStore | None,
    condition_dim: int,
    *,
    condition_mixing_mode: str = "features_only",
) -> np.ndarray:
    condition_id = str(row.get("condition_id", "") or row.get("sample_id", "") or "").strip()
    value = store.get(condition_id) if store is not None and condition_id else None
    base = None
    if value is not None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape[-1] != condition_dim:
            raise ValueError(f"Condition feature dim mismatch for {condition_id}: {arr.shape[-1]} != {condition_dim}")
        base = arr
    else:
        base = fallback_condition_features(row, condition_dim)
    program = property_program_tokens(row, condition_dim)
    source = source_smiles_condition_tokens(row, condition_dim)
    if condition_mixing_mode == "property_program_only":
        return program
    if condition_mixing_mode == "append_source_property_program":
        return np.concatenate([base, source, program], axis=0)
    if condition_mixing_mode == "append_property_program":
        return np.concatenate([base, program], axis=0)
    return base


def fallback_condition_features(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    values = []
    active_props = {part.strip() for part in str(row.get("condition_properties", "") or "").split(",") if part.strip()}
    for prop in PROPERTY_COLUMNS:
        value = parse_float(row.get(f"target_{prop}"))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        values.append(0.0 if math.isnan(value) else float(value) / normalizer)
    for prop in PROPERTY_COLUMNS:
        active = truthy(row.get(f"{prop}_active"))
        values.append(1.0 if (active if active is not None else prop in active_props) else 0.0)
    for prop in PROPERTY_COLUMNS:
        direction = str(row.get(f"{prop}_direction", "") or "").strip().lower()
        values.append(1.0 if direction in {"increase", "up", "+", "higher"} else (-1.0 if direction else 0.0))
    values.append(float(len(active_props)) / max(len(PROPERTY_COLUMNS), 1))
    vec = np.zeros(max(1, int(condition_dim)), dtype=np.float32)
    source = np.asarray(values, dtype=np.float32)
    vec[: min(vec.shape[0], source.shape[0])] = source[: vec.shape[0]]
    return vec[None, :]


def property_program_tokens(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    selected = selected_properties(row)
    selected_set = set(selected)
    count = row_property_count(row)
    count_norm = float(count) / max(len(PROPERTY_COLUMNS), 1)
    directions = [parse_direction_value(row.get(f"{prop}_direction")) for prop in PROPERTY_COLUMNS]
    positive_direction_fraction = sum(1 for value in directions if value > 0) / max(len(PROPERTY_COLUMNS), 1)
    negative_direction_fraction = sum(1 for value in directions if value < 0) / max(len(PROPERTY_COLUMNS), 1)
    normalized_targets = []
    for prop in PROPERTY_COLUMNS:
        target = parse_float(row.get(f"target_{prop}"))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        if not math.isnan(target):
            normalized_targets.append(float(target) / max(normalizer, 1e-8))
    tokens = [
        _expand_condition_token(
            [
                0.25,
                count_norm,
                float(len(selected_set)) / max(len(PROPERTY_COLUMNS), 1),
                sum(normalized_targets) / len(normalized_targets) if normalized_targets else 0.0,
                max(normalized_targets) if normalized_targets else 0.0,
                min(normalized_targets) if normalized_targets else 0.0,
                positive_direction_fraction,
                negative_direction_fraction,
            ],
            condition_dim,
        )
    ]
    for idx, prop in enumerate(PROPERTY_COLUMNS):
        target = parse_float(row.get(f"target_{prop}"))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        tolerance = float(SKETCHMOL_STRICT_TOLERANCE.get(prop, normalizer))
        tokens.append(
            _expand_condition_token(
                [
                    1.0,
                    float(idx + 1) / max(len(PROPERTY_COLUMNS), 1),
                    0.0 if math.isnan(target) else float(target) / max(normalizer, 1e-8),
                    1.0 if prop in selected_set else 0.0,
                    float(parse_direction_value(row.get(f"{prop}_direction"))),
                    tolerance / max(normalizer, 1e-8),
                    count_norm,
                    0.0 if math.isnan(target) else 1.0,
                ],
                condition_dim,
            )
        )
    return np.stack(tokens, axis=0).astype(np.float32)


def source_smiles_condition_tokens(
    row: Mapping[str, str],
    condition_dim: int,
    *,
    max_source_tokens: int = 96,
) -> np.ndarray:
    source_smiles = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    if not source_smiles:
        return np.zeros((1, max(1, int(condition_dim))), dtype=np.float32)
    tokens = tokenize_smiles(source_smiles)[: max(1, int(max_source_tokens))]
    if not tokens:
        return np.zeros((1, max(1, int(condition_dim))), dtype=np.float32)
    source_length = max(len(tokens), 1)
    rows = [
        _source_token_feature(token, idx, source_length, condition_dim)
        for idx, token in enumerate(tokens)
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def _source_token_feature(token: str, index: int, source_length: int, condition_dim: int) -> np.ndarray:
    dim = max(1, int(condition_dim))
    vec = np.zeros(dim, dtype=np.float32)
    _safe_set(vec, 0, 2.0)
    _safe_set(vec, 1, float(index + 1) / max(float(source_length), 1.0))
    _safe_set(vec, 2, float(source_length) / 160.0)
    _safe_set(vec, 3, float(len(str(token))) / 16.0)
    _safe_set(vec, 4, 1.0 if _is_atom_token(token) else 0.0)
    _safe_set(vec, 5, 1.0 if str(token).islower() else 0.0)
    _safe_set(vec, 6, 1.0 if _is_bond_token(token) else 0.0)
    _safe_set(vec, 7, 1.0 if str(token).isdigit() or str(token).startswith("%") else 0.0)
    _safe_set(vec, 8, 1.0 if str(token) in {"(", ")"} else 0.0)
    _safe_set(vec, 9, 1.0 if str(token).startswith("[") and str(token).endswith("]") else 0.0)
    bucket_space = max(dim - 16, 1)
    token_hash = _stable_token_hash(str(token))
    bucket = 16 + (token_hash % bucket_space)
    if bucket < dim:
        vec[bucket] = 1.0
    second_bucket = 16 + ((token_hash // max(bucket_space, 1)) % bucket_space)
    if second_bucket < dim:
        vec[second_bucket] = max(vec[second_bucket], 0.5)
    return vec


def _stable_token_hash(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _safe_set(vec: np.ndarray, index: int, value: float) -> None:
    if 0 <= int(index) < vec.shape[0]:
        vec[int(index)] = float(value)


def _is_atom_token(token: str) -> bool:
    text = str(token)
    if not text:
        return False
    if text.startswith("[") and text.endswith("]"):
        return True
    return any(ch.isalpha() for ch in text)


def _is_bond_token(token: str) -> bool:
    return str(token) in {"=", "#", "-", "/", "\\", ":", "~", "."}


def _expand_condition_token(values: Sequence[float], condition_dim: int) -> np.ndarray:
    source = np.asarray(list(values), dtype=np.float32)
    if source.size == 0:
        return np.zeros(max(1, int(condition_dim)), dtype=np.float32)
    repeats = int(math.ceil(max(1, int(condition_dim)) / max(source.size, 1)))
    tiled = np.tile(source, repeats)[: max(1, int(condition_dim))]
    return tiled.astype(np.float32)


def row_property_count(row: Mapping[str, str]) -> int:
    explicit = parse_float(row.get("property_count"))
    if not math.isnan(explicit) and explicit > 0:
        return max(1, int(round(explicit)))
    selected = selected_properties(row)
    if selected:
        return len(selected)
    return 1


def parse_direction_value(value: object) -> int:
    text = str(value or "").strip().lower()
    if text in {"increase", "up", "+", "higher"}:
        return 1
    if text in {"decrease", "down", "-", "lower"}:
        return -1
    return 0


def train_epoch(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    grad_clip: float,
    property_count_curriculum_sampling: bool,
    property_count_curriculum_loss: bool,
    property_count_curriculum_power: float,
    property_count_curriculum_baseline: float,
) -> dict[str, object]:
    model.train()
    rows = list(dataset)
    rng = random.Random(seed)
    if property_count_curriculum_sampling and rows:
        weights = [
            property_count_curriculum_weight(
                int(row.get("property_count", 1)),
                power=property_count_curriculum_power,
                baseline=property_count_curriculum_baseline,
            )
            for row in rows
        ]
        sampled_indices = rng.choices(range(len(rows)), weights=weights, k=len(rows))
        rows = [rows[idx] for idx in sampled_indices]
    else:
        rng.shuffle(rows)
    total_loss = 0.0
    total_tokens = 0
    sampled_property_counts: list[float] = []
    sampled_curriculum_weights: list[float] = []
    for batch_rows in batches(rows, batch_size):
        batch = collate(batch_rows, pad_id=model.pad_id, device=device)
        logits = model(batch["condition"], batch["decoder_input_ids"], condition_mask=batch["condition_mask"])
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=model.pad_id,
            reduction="none",
        ).reshape(batch["target_ids"].shape)
        token_mask = batch["target_ids"].ne(model.pad_id)
        sample_token_count = token_mask.sum(dim=1).clamp_min(1)
        sample_loss = (token_loss * token_mask).sum(dim=1) / sample_token_count
        sample_weights = property_count_curriculum_weight_tensor(
            batch["property_count"],
            power=property_count_curriculum_power,
            baseline=property_count_curriculum_baseline,
        )
        if property_count_curriculum_loss:
            loss = (sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        else:
            loss = sample_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        token_count = int(batch["target_ids"].ne(model.pad_id).sum().item())
        total_loss += float(loss.detach().cpu()) * max(token_count, 1)
        total_tokens += token_count
        sampled_property_counts.extend(float(value) for value in batch["property_count"].detach().cpu().tolist())
        sampled_curriculum_weights.extend(float(value) for value in sample_weights.detach().cpu().tolist())
    return {
        "loss": total_loss / max(total_tokens, 1),
        "tokens": total_tokens,
        "batches": math.ceil(len(rows) / max(1, int(batch_size))),
        "mean_property_count": _mean(sampled_property_counts),
        "mean_curriculum_weight": _mean(sampled_curriculum_weights),
    }


@torch.no_grad()
def evaluate_loss(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch_rows in batches(dataset, batch_size):
        batch = collate(batch_rows, pad_id=model.pad_id, device=device)
        logits = model(batch["condition"], batch["decoder_input_ids"], condition_mask=batch["condition_mask"])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=model.pad_id,
        )
        token_count = int(batch["target_ids"].ne(model.pad_id).sum().item())
        total_loss += float(loss.detach().cpu()) * max(token_count, 1)
        total_tokens += token_count
    return {
        "loss": total_loss / max(total_tokens, 1),
        "perplexity": math.exp(min(20.0, total_loss / max(total_tokens, 1))),
        "tokens": total_tokens,
    }


@torch.no_grad()
def write_predictions(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    output_csv: Path,
    *,
    batch_size: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_samples: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    parallel_samples: int,
    max_parallel_sequences: int,
    property_rerank: bool,
    condition_mixing_mode: str = "features_only",
    candidate_output_csv: Path | None = None,
) -> dict[str, object]:
    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    generated_values = []
    candidate_counts = []
    valid_candidate_counts = []
    unique_valid_candidate_counts = []
    unique_candidate_counts = []
    strict_fractions = []
    property_distances = []
    dataset_index = 0
    written_rows = 0
    csv_initialized = False
    candidate_csv_initialized = False
    sample_count = max(1, int(num_samples))
    sample_parallel = max(1, int(parallel_samples))
    max_parallel_sequences = max(1, int(max_parallel_sequences))
    suppress_ids = [vocab.token_to_id[UNK]] if UNK in vocab.token_to_id else []
    selected_fieldnames = infer_csv_fieldnames(
        rows,
        extra_fields=(
            "generated_smiles",
            "method",
            "direct_candidate_count",
            "direct_unique_candidate_count",
            "direct_valid_candidate_count",
            "direct_unique_valid_candidate_count",
            "direct_best_candidate_rank",
            "direct_best_score",
            "direct_best_strict_fraction",
            "direct_best_property_distance",
        ),
    )
    candidate_fieldnames = infer_csv_fieldnames(
        rows,
        extra_fields=(
            "generated_smiles",
            "method",
            "direct_candidate_index",
            "direct_candidate_raw_smiles",
            "direct_candidate_canonical_smiles",
            "direct_candidate_score",
            "direct_candidate_strict_fraction",
            "direct_candidate_property_distance",
            "direct_candidate_count",
        ),
    )
    for batch_rows in batches(dataset, batch_size):
        batch = collate(batch_rows, pad_id=model.pad_id, device=device)
        batch_candidates: list[list[str]] = [[] for _ in batch_rows]
        batch_output_rows: list[dict[str, object]] = []
        prompt_count = len(batch_rows)
        remaining = sample_count
        while remaining > 0:
            chunk_limit = max(1, max_parallel_sequences // max(prompt_count, 1))
            chunk = min(remaining, sample_parallel, chunk_limit)
            expanded = _repeat_generation_batch(batch, repeats=chunk)
            generated = model.generate(
                expanded["condition"],
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_new_tokens=max_new_tokens,
                condition_mask=expanded["condition_mask"],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                min_new_tokens=min_new_tokens,
                suppress_ids=suppress_ids,
            ).cpu()
            for row_offset in range(prompt_count):
                start = row_offset * chunk
                end = start + chunk
                for ids in generated[start:end]:
                    tokens = vocab.decode(ids.tolist()[1:])
                    batch_candidates[row_offset].append(detokenize_smiles(tokens))
            remaining -= chunk
        for candidates in batch_candidates:
            source_row = dict(rows[dataset_index])
            selected = select_generated_candidate(source_row, candidates, property_rerank=property_rerank)
            smiles = selected["generated_smiles"]
            source_row["generated_smiles"] = smiles
            source_row["method"] = direct_smiles_method_name(sample_count, condition_mixing_mode)
            source_row["direct_candidate_count"] = selected["candidate_count"]
            source_row["direct_unique_candidate_count"] = selected["unique_candidate_count"]
            source_row["direct_valid_candidate_count"] = selected["valid_candidate_count"]
            source_row["direct_unique_valid_candidate_count"] = selected["unique_valid_candidate_count"]
            source_row["direct_best_candidate_rank"] = selected["best_candidate_rank"]
            source_row["direct_best_score"] = selected["score"]
            source_row["direct_best_strict_fraction"] = selected["strict_fraction"]
            source_row["direct_best_property_distance"] = selected["normalized_property_distance"]
            generated_values.append(smiles)
            candidate_counts.append(int(selected["candidate_count"]))
            valid_candidate_counts.append(int(selected["valid_candidate_count"]))
            unique_valid_candidate_counts.append(int(selected["unique_valid_candidate_count"]))
            unique_candidate_counts.append(int(selected["unique_candidate_count"]))
            strict_fractions.append(float(selected["strict_fraction"]))
            distance = float(selected["normalized_property_distance"])
            if math.isfinite(distance):
                property_distances.append(distance)
            batch_output_rows.append(source_row)
            if candidate_output_csv is not None:
                candidate_rows = build_candidate_output_rows(
                    rows[dataset_index],
                    candidates,
                    sample_count=sample_count,
                    condition_mixing_mode=condition_mixing_mode,
                    property_rerank=property_rerank,
                )
                append_csv_rows(
                    candidate_output_csv,
                    candidate_rows,
                    overwrite=not candidate_csv_initialized,
                    fieldnames=candidate_fieldnames,
                )
                candidate_csv_initialized = True
            dataset_index += 1
        append_csv_rows(output_csv, batch_output_rows, overwrite=not csv_initialized, fieldnames=selected_fieldnames)
        csv_initialized = True
        written_rows += len(batch_output_rows)
        print(
            f"[direct-smiles] wrote {written_rows}/{len(rows)} rows "
            f"(num_samples={sample_count}, parallel_samples={sample_parallel}, max_parallel_sequences={max_parallel_sequences})",
            flush=True,
        )
    return {
        "rows": written_rows,
        "nonempty_rate": sum(1 for value in generated_values if str(value).strip()) / max(len(generated_values), 1),
        "unique_generated": len({value for value in generated_values if str(value).strip()}),
        "num_samples": sample_count,
        "parallel_samples": sample_parallel,
        "max_parallel_sequences": max_parallel_sequences,
        "property_rerank": bool(property_rerank),
        "mean_candidate_count": _mean(candidate_counts),
        "mean_valid_candidate_count": _mean(valid_candidate_counts),
        "mean_unique_valid_candidate_count": _mean(unique_valid_candidate_counts),
        "mean_unique_candidate_count": _mean(unique_candidate_counts),
        "mean_selected_strict_fraction": _mean(strict_fractions),
        "mean_selected_property_distance": _mean(property_distances),
    }


def direct_smiles_method_name(sample_count: int, condition_mixing_mode: str) -> str:
    base = "direct_smiles_mllm"
    if condition_mixing_mode != "features_only":
        base += "_mixed_condition"
    if sample_count > 1:
        base += "_sampled_rerank"
    return base


def select_generated_candidate(
    row: Mapping[str, str],
    candidates: Sequence[str],
    *,
    property_rerank: bool = True,
) -> dict[str, object]:
    scored = [
        score_generated_candidate(row, candidate, rank=rank, property_rerank=property_rerank)
        for rank, candidate in enumerate(candidates)
    ]
    if not scored:
        return {
            "generated_smiles": "",
            "candidate_count": 0,
            "unique_candidate_count": 0,
            "valid_candidate_count": 0,
            "unique_valid_candidate_count": 0,
            "best_candidate_rank": -1,
            "score": -math.inf,
            "strict_fraction": 0.0,
            "normalized_property_distance": math.inf,
        }
    best = max(scored, key=lambda item: float(item["score"]))
    raw_unique = {str(value).strip() for value in candidates if str(value).strip()}
    valid_unique = {str(item["canonical_smiles"]) for item in scored if item.get("canonical_smiles")}
    return {
        "generated_smiles": str(best.get("canonical_smiles") or best.get("raw_smiles") or ""),
        "candidate_count": len(candidates),
        "unique_candidate_count": len(raw_unique),
        "valid_candidate_count": sum(1 for item in scored if item.get("canonical_smiles")),
        "unique_valid_candidate_count": len(valid_unique),
        "best_candidate_rank": int(best["rank"]),
        "score": float(best["score"]),
        "strict_fraction": float(best["strict_fraction"]),
        "normalized_property_distance": float(best["normalized_property_distance"]),
    }


def build_candidate_output_rows(
    row: Mapping[str, str],
    candidates: Sequence[str],
    *,
    sample_count: int,
    condition_mixing_mode: str,
    property_rerank: bool,
) -> list[dict[str, object]]:
    out = []
    method = direct_smiles_method_name(sample_count, condition_mixing_mode)
    for rank, candidate in enumerate(candidates):
        scored = score_generated_candidate(row, candidate, rank=rank, property_rerank=property_rerank)
        candidate_row = dict(row)
        candidate_row["generated_smiles"] = str(scored.get("canonical_smiles") or scored.get("raw_smiles") or "")
        candidate_row["method"] = f"{method}_candidate"
        candidate_row["direct_candidate_index"] = int(rank)
        candidate_row["direct_candidate_raw_smiles"] = str(scored.get("raw_smiles") or "")
        candidate_row["direct_candidate_canonical_smiles"] = str(scored.get("canonical_smiles") or "")
        candidate_row["direct_candidate_score"] = float(scored["score"])
        candidate_row["direct_candidate_strict_fraction"] = float(scored["strict_fraction"])
        candidate_row["direct_candidate_property_distance"] = float(scored["normalized_property_distance"])
        candidate_row["direct_candidate_count"] = int(len(candidates))
        out.append(candidate_row)
    return out


def score_generated_candidate(
    row: Mapping[str, str],
    smiles: str,
    *,
    rank: int = 0,
    property_rerank: bool = True,
) -> dict[str, object]:
    raw = str(smiles or "").strip()
    canonical = _safe_canonical_smiles(raw)
    if not canonical:
        return {
            "raw_smiles": raw,
            "canonical_smiles": "",
            "rank": int(rank),
            "score": -1_000_000.0 - float(rank) * 1e-6,
            "strict_fraction": 0.0,
            "normalized_property_distance": math.inf,
        }
    strict_fraction = 0.0
    distance = 0.0
    if property_rerank:
        strict_fraction, distance = property_score_components(row, canonical)
    score = 10.0 + 100.0 * strict_fraction - 10.0 * distance - float(rank) * 1e-6
    return {
        "raw_smiles": raw,
        "canonical_smiles": canonical,
        "rank": int(rank),
        "score": float(score),
        "strict_fraction": float(strict_fraction),
        "normalized_property_distance": float(distance),
    }


def property_score_components(row: Mapping[str, str], smiles: str) -> tuple[float, float]:
    props = _safe_normalized_properties(smiles)
    selected = selected_properties(row)
    if not props or not selected:
        return 0.0, 0.0
    successes = []
    distances = []
    for prop in selected:
        target = parse_float(row.get(f"target_{prop}"))
        actual = parse_float(props.get(prop))
        tolerance = float(SKETCHMOL_STRICT_TOLERANCE.get(prop, PROPERTY_NORMALIZERS.get(prop, 1.0)))
        if math.isnan(target) or math.isnan(actual):
            successes.append(False)
            distances.append(1e6)
            continue
        error = abs(actual - target)
        successes.append(error <= tolerance)
        distances.append(error / max(tolerance, 1e-8))
    return sum(1 for value in successes if value) / max(len(successes), 1), sum(distances) / max(len(distances), 1)


def selected_properties(row: Mapping[str, str]) -> list[str]:
    selected = [item.strip() for item in str(row.get("condition_properties", "") or "").split(",") if item.strip()]
    selected = [prop for prop in selected if prop in PROPERTY_COLUMNS]
    if selected:
        return selected
    return [prop for prop in PROPERTY_COLUMNS if truthy(row.get(f"{prop}_active"))]


def _safe_canonical_smiles(smiles: str) -> str:
    return _cached_canonical_smiles(str(smiles or ""))


def _safe_normalized_properties(smiles: str) -> dict[str, float]:
    return dict(_cached_normalized_properties(str(smiles or "")))


def _mean(values: Sequence[float | int]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return sum(items) / len(items) if items else 0.0


def property_count_curriculum_weight(count: int, *, power: float, baseline: float) -> float:
    base = max(float(baseline), 1e-8)
    normalized = max(float(count), 1.0) / base
    return max(1.0, normalized**float(power))


def property_count_curriculum_weight_tensor(
    property_count: torch.Tensor,
    *,
    power: float,
    baseline: float,
) -> torch.Tensor:
    base = max(float(baseline), 1e-8)
    normalized = torch.clamp(property_count.to(dtype=torch.float32), min=1.0) / base
    weights = torch.pow(normalized, float(power))
    return torch.clamp(weights, min=1.0)


def collate(rows: list[dict[str, object]], *, pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_condition_len = max(np.asarray(row["condition"]).shape[0] for row in rows)
    condition_dim = np.asarray(rows[0]["condition"]).shape[-1]
    max_seq_len = max(len(row["decoder_input_ids"]) for row in rows)
    condition = np.zeros((len(rows), max_condition_len, condition_dim), dtype=np.float32)
    condition_mask = np.zeros((len(rows), max_condition_len), dtype=bool)
    decoder_input_ids = np.full((len(rows), max_seq_len), int(pad_id), dtype=np.int64)
    target_ids = np.full((len(rows), max_seq_len), int(pad_id), dtype=np.int64)
    property_count = np.ones((len(rows),), dtype=np.float32)
    for idx, row in enumerate(rows):
        cond = np.asarray(row["condition"], dtype=np.float32)
        condition[idx, : cond.shape[0], :] = cond
        condition_mask[idx, : cond.shape[0]] = True
        dec = np.asarray(row["decoder_input_ids"], dtype=np.int64)
        tgt = np.asarray(row["target_ids"], dtype=np.int64)
        decoder_input_ids[idx, : dec.shape[0]] = dec
        target_ids[idx, : tgt.shape[0]] = tgt
        property_count[idx] = float(row.get("property_count", 1))
    return {
        "condition": torch.from_numpy(condition).to(device),
        "condition_mask": torch.from_numpy(condition_mask).to(device),
        "decoder_input_ids": torch.from_numpy(decoder_input_ids).to(device),
        "target_ids": torch.from_numpy(target_ids).to(device),
        "property_count": torch.from_numpy(property_count).to(device),
    }


def _repeat_generation_batch(batch: Mapping[str, torch.Tensor], *, repeats: int) -> dict[str, torch.Tensor]:
    repeats = max(1, int(repeats))
    return {
        "condition": batch["condition"].repeat_interleave(repeats, dim=0),
        "condition_mask": batch["condition_mask"].repeat_interleave(repeats, dim=0),
    }


def batches(rows: list[dict[str, object]], batch_size: int) -> Iterable[list[dict[str, object]]]:
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def save_checkpoint(
    path: Path,
    model: ConditionedSmilesDecoder,
    optimizer: torch.optim.Optimizer | None,
    vocab: SmilesVocabulary,
    config: dict[str, object],
    epoch: int,
    history: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "vocab": vocab.to_dict(),
        "model_config": config,
        "epoch": int(epoch),
        "history": history,
        "args": vars(args),
    }
    torch.save(payload, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_csv_fieldnames(rows: Sequence[Mapping[str, object]], *, extra_fields: Sequence[str] = ()) -> list[str]:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    for key in extra_fields:
        if key not in seen:
            seen.add(key)
            fieldnames.append(str(key))
    return fieldnames


def append_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
    *,
    overwrite: bool = False,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = infer_csv_fieldnames(rows)
    mode = "w" if overwrite or not path.exists() else "a"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return math.nan


def truthy(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@lru_cache(maxsize=200000)
def _cached_canonical_smiles(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        return canonical_smiles(smiles) or ""
    except RuntimeError:
        return ""


@lru_cache(maxsize=200000)
def _cached_normalized_properties(smiles: str) -> tuple[tuple[str, float], ...]:
    try:
        props = molecular_properties(smiles) or {}
    except RuntimeError:
        return tuple()
    normalized = {
        prop: float(props.get(key, math.nan))
        for prop, key in PROPERTY_VALUE_KEYS.items()
        if key in props
    }
    return tuple(sorted(normalized.items()))


if __name__ == "__main__":
    raise SystemExit(main())
