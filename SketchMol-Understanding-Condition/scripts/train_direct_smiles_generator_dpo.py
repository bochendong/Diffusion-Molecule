#!/usr/bin/env python3
"""DPO-style fine-tuning for the MLLM-conditioned direct SMILES generator."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_direct_smiles_generator import (  # noqa: E402
    condition_array_for_row,
    load_checkpoint,
    load_store,
    read_rows,
    resolve_device,
    resolve_condition_mixing_mode,
    save_checkpoint,
    seed_everything,
)
from sketchmol_understanding_condition.direct_smiles_generation import (  # noqa: E402
    ConditionedSmilesDecoder,
    SmilesVocabulary,
    tokenize_smiles,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-preference-csv", required=True, type=Path)
    parser.add_argument("--eval-preference-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--eval-condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", default="query_tokens")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-weight", type=float, default=0.5)
    parser.add_argument("--max-smiles-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.resume_checkpoint)
    if checkpoint is None:
        raise ValueError("--resume-checkpoint is required")
    checkpoint_args = dict(checkpoint.get("args", {}))
    condition_mixing_mode = resolve_condition_mixing_mode(args, checkpoint_args)
    vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])

    train_rows = read_rows(args.train_preference_csv)
    eval_rows = read_rows(args.eval_preference_csv) if args.eval_preference_csv else []
    train_store = load_store(args.condition_features_dir, args)
    eval_store = load_store(args.eval_condition_features_dir or args.condition_features_dir, args)

    train_dataset = build_preference_dataset(
        train_rows,
        vocab,
        train_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        condition_mixing_mode=condition_mixing_mode,
    )
    eval_dataset = build_preference_dataset(
        eval_rows,
        vocab,
        eval_store,
        int(config["condition_dim"]),
        max_smiles_length=int(args.max_smiles_length),
        condition_mixing_mode=condition_mixing_mode,
    )

    policy_model = ConditionedSmilesDecoder(**config).to(device)
    policy_model.load_state_dict(checkpoint["model_state"])
    reference_model = ConditionedSmilesDecoder(**config).to(device)
    reference_model.load_state_dict(checkpoint["model_state"])
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        record = train_epoch_dpo(
            policy_model,
            reference_model,
            train_dataset,
            optimizer,
            pad_id=vocab.pad_id,
            batch_size=int(args.batch_size),
            device=device,
            beta=float(args.beta),
            sft_weight=float(args.sft_weight),
            grad_clip=float(args.grad_clip),
            seed=int(args.seed) + epoch,
        )
        record["epoch"] = epoch
        if eval_dataset:
            record.update(
                {
                    f"eval_{key}": value
                    for key, value in evaluate_dpo(
                        policy_model,
                        reference_model,
                        eval_dataset,
                        pad_id=vocab.pad_id,
                        batch_size=int(args.eval_batch_size),
                        device=device,
                        beta=float(args.beta),
                        sft_weight=float(args.sft_weight),
                    ).items()
                }
            )
        history.append(record)
        save_checkpoint(args.output_dir / "latest_dpo_checkpoint.pt", policy_model, optimizer, vocab, config, epoch, history, args)

    save_checkpoint(
        args.output_dir / "direct_smiles_generator_dpo.pt",
        policy_model,
        optimizer,
        vocab,
        config,
        len(history),
        history,
        args,
    )
    summary = {
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.output_dir / "direct_smiles_generator_dpo.pt"),
        "epochs": int(args.epochs),
        "train_pairs": len(train_dataset),
        "eval_pairs": len(eval_dataset),
        "history": history,
        "condition_mixing_mode": condition_mixing_mode,
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_preference_dataset(
    rows: Sequence[Mapping[str, str]],
    vocab: SmilesVocabulary,
    store,
    condition_dim: int,
    *,
    max_smiles_length: int,
    condition_mixing_mode: str = "features_only",
) -> list[dict[str, object]]:
    dataset = []
    for row in rows:
        chosen = str(row.get("chosen_smiles", "") or "").strip()
        rejected = str(row.get("rejected_smiles", "") or "").strip()
        if not chosen or not rejected:
            continue
        chosen_tokens = tokenize_smiles(chosen)[: max(1, int(max_smiles_length))]
        rejected_tokens = tokenize_smiles(rejected)[: max(1, int(max_smiles_length))]
        if not chosen_tokens or not rejected_tokens:
            continue
        condition = condition_array_for_row(
            row,
            store,
            condition_dim,
            condition_mixing_mode=condition_mixing_mode,
        )
        dataset.append(
            {
                "condition": np.asarray(condition, dtype=np.float32),
                "chosen_decoder_input_ids": np.asarray(vocab.encode(chosen_tokens, add_bos=True, add_eos=False), dtype=np.int64),
                "chosen_target_ids": np.asarray(vocab.encode(chosen_tokens, add_bos=False, add_eos=True), dtype=np.int64),
                "rejected_decoder_input_ids": np.asarray(vocab.encode(rejected_tokens, add_bos=True, add_eos=False), dtype=np.int64),
                "rejected_target_ids": np.asarray(vocab.encode(rejected_tokens, add_bos=False, add_eos=True), dtype=np.int64),
            }
        )
    return dataset


def collate_preference(rows: Sequence[dict[str, object]], *, pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_condition_len = max(np.asarray(row["condition"]).shape[0] for row in rows)
    condition_dim = np.asarray(rows[0]["condition"]).shape[-1]
    max_chosen_len = max(len(row["chosen_decoder_input_ids"]) for row in rows)
    max_rejected_len = max(len(row["rejected_decoder_input_ids"]) for row in rows)

    condition = np.zeros((len(rows), max_condition_len, condition_dim), dtype=np.float32)
    condition_mask = np.zeros((len(rows), max_condition_len), dtype=bool)
    chosen_decoder_input_ids = np.full((len(rows), max_chosen_len), int(pad_id), dtype=np.int64)
    chosen_target_ids = np.full((len(rows), max_chosen_len), int(pad_id), dtype=np.int64)
    rejected_decoder_input_ids = np.full((len(rows), max_rejected_len), int(pad_id), dtype=np.int64)
    rejected_target_ids = np.full((len(rows), max_rejected_len), int(pad_id), dtype=np.int64)

    for idx, row in enumerate(rows):
        cond = np.asarray(row["condition"], dtype=np.float32)
        condition[idx, : cond.shape[0], :] = cond
        condition_mask[idx, : cond.shape[0]] = True

        chosen_dec = np.asarray(row["chosen_decoder_input_ids"], dtype=np.int64)
        chosen_tgt = np.asarray(row["chosen_target_ids"], dtype=np.int64)
        rejected_dec = np.asarray(row["rejected_decoder_input_ids"], dtype=np.int64)
        rejected_tgt = np.asarray(row["rejected_target_ids"], dtype=np.int64)

        chosen_decoder_input_ids[idx, : chosen_dec.shape[0]] = chosen_dec
        chosen_target_ids[idx, : chosen_tgt.shape[0]] = chosen_tgt
        rejected_decoder_input_ids[idx, : rejected_dec.shape[0]] = rejected_dec
        rejected_target_ids[idx, : rejected_tgt.shape[0]] = rejected_tgt

    return {
        "condition": torch.from_numpy(condition).to(device),
        "condition_mask": torch.from_numpy(condition_mask).to(device),
        "chosen_decoder_input_ids": torch.from_numpy(chosen_decoder_input_ids).to(device),
        "chosen_target_ids": torch.from_numpy(chosen_target_ids).to(device),
        "rejected_decoder_input_ids": torch.from_numpy(rejected_decoder_input_ids).to(device),
        "rejected_target_ids": torch.from_numpy(rejected_target_ids).to(device),
    }


def train_epoch_dpo(
    policy_model: ConditionedSmilesDecoder,
    reference_model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    beta: float,
    sft_weight: float,
    grad_clip: float,
    seed: int,
) -> dict[str, object]:
    policy_model.train()
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    total_loss = 0.0
    total_dpo_loss = 0.0
    total_sft_loss = 0.0
    total_pref_rate = 0.0
    total_margin = 0.0
    total_batches = 0

    for batch_ids in batch_indices(indices, batch_size):
        batch_rows = [dataset[idx] for idx in batch_ids]
        batch = collate_preference(batch_rows, pad_id=pad_id, device=device)

        policy_chosen = sequence_logprobs_from_targets(
            policy_model,
            batch["condition"],
            batch["condition_mask"],
            batch["chosen_decoder_input_ids"],
            batch["chosen_target_ids"],
            pad_id=pad_id,
        )
        policy_rejected = sequence_logprobs_from_targets(
            policy_model,
            batch["condition"],
            batch["condition_mask"],
            batch["rejected_decoder_input_ids"],
            batch["rejected_target_ids"],
            pad_id=pad_id,
        )

        with torch.no_grad():
            ref_chosen = sequence_logprobs_from_targets(
                reference_model,
                batch["condition"],
                batch["condition_mask"],
                batch["chosen_decoder_input_ids"],
                batch["chosen_target_ids"],
                pad_id=pad_id,
            )
            ref_rejected = sequence_logprobs_from_targets(
                reference_model,
                batch["condition"],
                batch["condition_mask"],
                batch["rejected_decoder_input_ids"],
                batch["rejected_target_ids"],
                pad_id=pad_id,
            )

        logits = float(beta) * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
        dpo_loss = -F.logsigmoid(logits).mean()

        chosen_logits = policy_model(
            batch["condition"],
            batch["chosen_decoder_input_ids"],
            condition_mask=batch["condition_mask"],
        )
        sft_loss = F.cross_entropy(
            chosen_logits.reshape(-1, chosen_logits.shape[-1]),
            batch["chosen_target_ids"].reshape(-1),
            ignore_index=int(pad_id),
        )
        loss = dpo_loss + float(sft_weight) * sft_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), float(grad_clip))
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_dpo_loss += float(dpo_loss.detach().cpu())
        total_sft_loss += float(sft_loss.detach().cpu())
        total_pref_rate += float((policy_chosen > policy_rejected).float().mean().detach().cpu())
        total_margin += float(logits.mean().detach().cpu())
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "dpo_loss": total_dpo_loss / denom,
        "sft_loss": total_sft_loss / denom,
        "policy_pref_rate": total_pref_rate / denom,
        "mean_margin": total_margin / denom,
        "batches": total_batches,
    }


@torch.no_grad()
def evaluate_dpo(
    policy_model: ConditionedSmilesDecoder,
    reference_model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    *,
    pad_id: int,
    batch_size: int,
    device: torch.device,
    beta: float,
    sft_weight: float,
) -> dict[str, object]:
    policy_model.eval()
    total_loss = 0.0
    total_dpo_loss = 0.0
    total_sft_loss = 0.0
    total_pref_rate = 0.0
    total_margin = 0.0
    total_batches = 0

    for batch_rows in batch_rows_iter(dataset, batch_size):
        batch = collate_preference(batch_rows, pad_id=pad_id, device=device)
        policy_chosen = sequence_logprobs_from_targets(
            policy_model,
            batch["condition"],
            batch["condition_mask"],
            batch["chosen_decoder_input_ids"],
            batch["chosen_target_ids"],
            pad_id=pad_id,
        )
        policy_rejected = sequence_logprobs_from_targets(
            policy_model,
            batch["condition"],
            batch["condition_mask"],
            batch["rejected_decoder_input_ids"],
            batch["rejected_target_ids"],
            pad_id=pad_id,
        )
        ref_chosen = sequence_logprobs_from_targets(
            reference_model,
            batch["condition"],
            batch["condition_mask"],
            batch["chosen_decoder_input_ids"],
            batch["chosen_target_ids"],
            pad_id=pad_id,
        )
        ref_rejected = sequence_logprobs_from_targets(
            reference_model,
            batch["condition"],
            batch["condition_mask"],
            batch["rejected_decoder_input_ids"],
            batch["rejected_target_ids"],
            pad_id=pad_id,
        )
        logits = float(beta) * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
        dpo_loss = -F.logsigmoid(logits).mean()
        chosen_logits = policy_model(
            batch["condition"],
            batch["chosen_decoder_input_ids"],
            condition_mask=batch["condition_mask"],
        )
        sft_loss = F.cross_entropy(
            chosen_logits.reshape(-1, chosen_logits.shape[-1]),
            batch["chosen_target_ids"].reshape(-1),
            ignore_index=int(pad_id),
        )
        loss = dpo_loss + float(sft_weight) * sft_loss

        total_loss += float(loss.detach().cpu())
        total_dpo_loss += float(dpo_loss.detach().cpu())
        total_sft_loss += float(sft_loss.detach().cpu())
        total_pref_rate += float((policy_chosen > policy_rejected).float().mean().detach().cpu())
        total_margin += float(logits.mean().detach().cpu())
        total_batches += 1

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "dpo_loss": total_dpo_loss / denom,
        "sft_loss": total_sft_loss / denom,
        "policy_pref_rate": total_pref_rate / denom,
        "mean_margin": total_margin / denom,
        "batches": total_batches,
    }


def sequence_logprobs_from_targets(
    model: ConditionedSmilesDecoder,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    logits = model(condition, decoder_input_ids, condition_mask=condition_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    token_mask = target_ids.ne(int(pad_id)).to(token_log_probs.dtype)
    return (token_log_probs * token_mask).sum(dim=1)


def batch_indices(indices: Sequence[int], batch_size: int) -> Sequence[Sequence[int]]:
    size = max(1, int(batch_size))
    return [indices[start : start + size] for start in range(0, len(indices), size)]


def batch_rows_iter(rows: Sequence[dict[str, object]], batch_size: int):
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


if __name__ == "__main__":
    raise SystemExit(main())
