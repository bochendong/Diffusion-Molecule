#!/usr/bin/env python3
"""Train/evaluate an MLLM-conditioned direct SMILES generator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_understanding_condition.direct_smiles_generation import (  # noqa: E402
    ConditionedSmilesDecoder,
    SmilesVocabulary,
    build_vocabulary,
    detokenize_smiles,
    tokenize_smiles,
)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--eval-condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", default="query_tokens")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-csv", type=Path, default=None)
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

    train_dataset = build_dataset(train_rows, vocab, train_store, condition_dim, max_smiles_length=args.max_smiles_length)
    eval_dataset = build_dataset(eval_rows, vocab, eval_store, condition_dim, max_smiles_length=args.max_smiles_length)

    history: list[dict[str, object]] = list(checkpoint.get("history", [])) if checkpoint else []
    start_epoch = int(checkpoint.get("epoch", 0)) + 1 if checkpoint and not args.eval_only else 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if checkpoint is not None and checkpoint.get("optimizer_state") and not args.eval_only:
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
    return torch.load(path, map_location="cpu")


def infer_condition_dim(*stores: FrozenConditionFeatureStore | None) -> int:
    for store in stores:
        if store is not None:
            return int(store.input_hidden_dim)
    return 32


def build_dataset(
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    store: FrozenConditionFeatureStore | None,
    condition_dim: int,
    *,
    max_smiles_length: int,
) -> list[dict[str, object]]:
    dataset = []
    for row in rows:
        target = str(row.get("target_smiles", "") or "").strip()
        if not target:
            continue
        tokens = tokenize_smiles(target)[: max(1, int(max_smiles_length))]
        decoder_input = vocab.encode(tokens, add_bos=True, add_eos=False)
        target_ids = vocab.encode(tokens, add_bos=False, add_eos=True)
        condition = condition_array_for_row(row, store, condition_dim)
        dataset.append(
            {
                "condition": condition.astype(np.float32),
                "decoder_input_ids": np.asarray(decoder_input, dtype=np.int64),
                "target_ids": np.asarray(target_ids, dtype=np.int64),
            }
        )
    return dataset


def condition_array_for_row(
    row: Mapping[str, str],
    store: FrozenConditionFeatureStore | None,
    condition_dim: int,
) -> np.ndarray:
    condition_id = str(row.get("condition_id", "") or row.get("sample_id", "") or "").strip()
    value = store.get(condition_id) if store is not None and condition_id else None
    if value is not None:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape[-1] != condition_dim:
            raise ValueError(f"Condition feature dim mismatch for {condition_id}: {arr.shape[-1]} != {condition_dim}")
        return arr
    return fallback_condition_features(row, condition_dim)


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


def train_epoch(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    grad_clip: float,
) -> dict[str, object]:
    model.train()
    rows = list(dataset)
    random.Random(seed).shuffle(rows)
    total_loss = 0.0
    total_tokens = 0
    for batch_rows in batches(rows, batch_size):
        batch = collate(batch_rows, pad_id=model.pad_id, device=device)
        logits = model(batch["condition"], batch["decoder_input_ids"], condition_mask=batch["condition_mask"])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=model.pad_id,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        token_count = int(batch["target_ids"].ne(model.pad_id).sum().item())
        total_loss += float(loss.detach().cpu()) * max(token_count, 1)
        total_tokens += token_count
    return {
        "loss": total_loss / max(total_tokens, 1),
        "tokens": total_tokens,
        "batches": math.ceil(len(rows) / max(1, int(batch_size))),
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
) -> dict[str, object]:
    model.eval()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    generated_values = []
    dataset_index = 0
    for batch_rows in batches(dataset, batch_size):
        batch = collate(batch_rows, pad_id=model.pad_id, device=device)
        generated = model.generate(
            batch["condition"],
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_new_tokens=max_new_tokens,
            condition_mask=batch["condition_mask"],
            temperature=temperature,
            top_k=top_k,
        ).cpu()
        for ids in generated:
            source_row = dict(rows[dataset_index])
            tokens = vocab.decode(ids.tolist()[1:])
            smiles = detokenize_smiles(tokens)
            source_row["generated_smiles"] = smiles
            source_row["method"] = "direct_smiles_mllm"
            generated_values.append(smiles)
            out_rows.append(source_row)
            dataset_index += 1
    write_csv(output_csv, out_rows)
    return {
        "rows": len(out_rows),
        "nonempty_rate": sum(1 for value in generated_values if str(value).strip()) / max(len(generated_values), 1),
        "unique_generated": len({value for value in generated_values if str(value).strip()}),
    }


def collate(rows: list[dict[str, object]], *, pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_condition_len = max(np.asarray(row["condition"]).shape[0] for row in rows)
    condition_dim = np.asarray(rows[0]["condition"]).shape[-1]
    max_seq_len = max(len(row["decoder_input_ids"]) for row in rows)
    condition = np.zeros((len(rows), max_condition_len, condition_dim), dtype=np.float32)
    condition_mask = np.zeros((len(rows), max_condition_len), dtype=bool)
    decoder_input_ids = np.full((len(rows), max_seq_len), int(pad_id), dtype=np.int64)
    target_ids = np.full((len(rows), max_seq_len), int(pad_id), dtype=np.int64)
    for idx, row in enumerate(rows):
        cond = np.asarray(row["condition"], dtype=np.float32)
        condition[idx, : cond.shape[0], :] = cond
        condition_mask[idx, : cond.shape[0]] = True
        dec = np.asarray(row["decoder_input_ids"], dtype=np.int64)
        tgt = np.asarray(row["target_ids"], dtype=np.int64)
        decoder_input_ids[idx, : dec.shape[0]] = dec
        target_ids[idx, : tgt.shape[0]] = tgt
    return {
        "condition": torch.from_numpy(condition).to(device),
        "condition_mask": torch.from_numpy(condition_mask).to(device),
        "decoder_input_ids": torch.from_numpy(decoder_input_ids).to(device),
        "target_ids": torch.from_numpy(target_ids).to(device),
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


if __name__ == "__main__":
    raise SystemExit(main())
