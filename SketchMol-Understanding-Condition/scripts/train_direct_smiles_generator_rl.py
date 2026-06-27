#!/usr/bin/env python3
"""Policy fine-tuning for the MLLM-conditioned direct SMILES generator."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_direct_smiles_generator import (  # noqa: E402
    UNK,
    _repeat_generation_batch,
    _safe_canonical_smiles,
    batches,
    build_dataset,
    collate,
    load_checkpoint,
    load_store,
    property_score_components,
    read_rows,
    resolve_device,
    resolve_condition_mixing_mode,
    save_checkpoint,
    seed_everything,
)
from sketchmol_understanding_condition.direct_smiles_generation import (  # noqa: E402
    ConditionedSmilesDecoder,
    SmilesVocabulary,
    detokenize_smiles,
)
from sketchmol_understanding_condition.chem import morgan_tanimoto  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--eval-condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", default="query_tokens")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument(
        "--condition-mixing-mode",
        choices=("features_only", "append_property_program", "append_source_property_program", "property_program_only"),
        default="features_only",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rollouts-per-prompt", type=int, default=16)
    parser.add_argument("--parallel-samples", type=int, default=4)
    parser.add_argument("--max-parallel-sequences", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--min-new-tokens", type=int, default=6)
    parser.add_argument("--sft-weight", type=float, default=0.25)
    parser.add_argument(
        "--advantage-mode",
        choices=("group_center", "group_zscore"),
        default="group_center",
    )
    parser.add_argument("--advantage-clip", type=float, default=0.0)
    parser.add_argument(
        "--sequence-logprob-reduction",
        choices=("sum", "mean"),
        default="sum",
    )
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument("--reward-valid-weight", type=float, default=1.0)
    parser.add_argument("--reward-strict-weight", type=float, default=1.0)
    parser.add_argument("--reward-distance-weight", type=float, default=0.1)
    parser.add_argument("--reward-distance-clip", type=float, default=10.0)
    parser.add_argument("--reward-source-similarity-weight", type=float, default=0.0)
    parser.add_argument("--reward-source-similarity-threshold", type=float, default=0.4)
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
    vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])
    checkpoint_args = dict(checkpoint.get("args", {}))
    condition_mixing_mode = resolve_condition_mixing_mode(args, checkpoint_args)

    train_rows = read_rows(args.train_csv)
    eval_rows = read_rows(args.eval_csv) if args.eval_csv else []
    train_store = load_store(args.condition_features_dir, args)
    eval_store = load_store(args.eval_condition_features_dir or args.condition_features_dir, args)

    train_dataset = build_dataset(
        train_rows,
        vocab,
        train_store,
        int(config["condition_dim"]),
        max_smiles_length=int(config["max_length"]) - 8,
        condition_mixing_mode=condition_mixing_mode,
    )
    eval_dataset = build_dataset(
        eval_rows,
        vocab,
        eval_store,
        int(config["condition_dim"]),
        max_smiles_length=int(config["max_length"]) - 8,
        condition_mixing_mode=condition_mixing_mode,
    )
    train_rows = [row for row in train_rows if str(row.get("target_smiles", "") or "").strip()]
    eval_rows = [row for row in eval_rows if str(row.get("target_smiles", "") or "").strip()]

    model = ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    reference_model = None
    if float(args.reference_kl_weight) > 0:
        reference_model = ConditionedSmilesDecoder(**config).to(device)
        reference_model.load_state_dict(checkpoint["model_state"])
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        record = train_epoch_rl(
            model,
            reference_model,
            train_dataset,
            train_rows,
            vocab,
            optimizer,
            batch_size=int(args.batch_size),
            device=device,
            rollouts_per_prompt=int(args.rollouts_per_prompt),
            parallel_samples=int(args.parallel_samples),
            max_parallel_sequences=int(args.max_parallel_sequences),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            min_new_tokens=int(args.min_new_tokens),
            sft_weight=float(args.sft_weight),
            advantage_mode=str(args.advantage_mode),
            advantage_clip=float(args.advantage_clip),
            sequence_logprob_reduction=str(args.sequence_logprob_reduction),
            reference_kl_weight=float(args.reference_kl_weight),
            reward_valid_weight=float(args.reward_valid_weight),
            reward_strict_weight=float(args.reward_strict_weight),
            reward_distance_weight=float(args.reward_distance_weight),
            reward_distance_clip=float(args.reward_distance_clip),
            reward_source_similarity_weight=float(args.reward_source_similarity_weight),
            reward_source_similarity_threshold=float(args.reward_source_similarity_threshold),
            grad_clip=float(args.grad_clip),
            seed=int(args.seed) + epoch,
        )
        record["epoch"] = epoch
        if eval_dataset and eval_rows:
            record.update(
                {
                    f"eval_{key}": value
                    for key, value in evaluate_rl(
                        model,
                        eval_dataset,
                        eval_rows,
                        vocab,
                        batch_size=int(args.eval_batch_size),
                        device=device,
                        rollouts_per_prompt=int(args.rollouts_per_prompt),
                        parallel_samples=int(args.parallel_samples),
                        max_parallel_sequences=int(args.max_parallel_sequences),
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        repetition_penalty=float(args.repetition_penalty),
                        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
                        min_new_tokens=int(args.min_new_tokens),
                        reward_valid_weight=float(args.reward_valid_weight),
                        reward_strict_weight=float(args.reward_strict_weight),
                        reward_distance_weight=float(args.reward_distance_weight),
                        reward_distance_clip=float(args.reward_distance_clip),
                        reward_source_similarity_weight=float(args.reward_source_similarity_weight),
                        reward_source_similarity_threshold=float(args.reward_source_similarity_threshold),
                    ).items()
                }
            )
        history.append(record)
        save_checkpoint(args.output_dir / "latest_rl_checkpoint.pt", model, optimizer, vocab, config, epoch, history, args)

    save_checkpoint(
        args.output_dir / "direct_smiles_generator_rl.pt",
        model,
        optimizer,
        vocab,
        config,
        len(history),
        history,
        args,
    )
    summary = {
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.output_dir / "direct_smiles_generator_rl.pt"),
        "epochs": int(args.epochs),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "history": history,
        "condition_mixing_mode": condition_mixing_mode,
        "advantage_mode": str(args.advantage_mode),
        "sequence_logprob_reduction": str(args.sequence_logprob_reduction),
        "reference_kl_weight": float(args.reference_kl_weight),
        "reward_source_similarity_weight": float(args.reward_source_similarity_weight),
        "reward_source_similarity_threshold": float(args.reward_source_similarity_threshold),
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def train_epoch_rl(
    model: ConditionedSmilesDecoder,
    reference_model: ConditionedSmilesDecoder | None,
    dataset: list[dict[str, object]],
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    device: torch.device,
    rollouts_per_prompt: int,
    parallel_samples: int,
    max_parallel_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    sft_weight: float,
    advantage_mode: str,
    advantage_clip: float,
    sequence_logprob_reduction: str,
    reference_kl_weight: float,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
    grad_clip: float,
    seed: int,
) -> dict[str, object]:
    model.train()
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    total_loss = 0.0
    total_pg_loss = 0.0
    total_sft_loss = 0.0
    total_kl_loss = 0.0
    total_reward = 0.0
    total_batches = 0
    suppress_ids = [vocab.token_to_id[UNK]] if UNK in vocab.token_to_id else []
    for batch_ids in batches(indices, batch_size):
        batch_rows = [dataset[idx] for idx in batch_ids]
        batch_meta = [rows[idx] for idx in batch_ids]
        batch = collate(batch_rows, pad_id=vocab.pad_id, device=device)
        generated = sample_rollouts(
            model,
            batch,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            rollouts_per_prompt=rollouts_per_prompt,
            parallel_samples=parallel_samples,
            max_parallel_sequences=max_parallel_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            suppress_ids=suppress_ids,
        )
        expanded = _repeat_generation_batch(batch, repeats=rollouts_per_prompt)
        seq_logprob = sequence_logprobs(
            model,
            expanded["condition"],
            expanded["condition_mask"],
            generated.to(device),
            eos_id=vocab.eos_id,
            reduction=sequence_logprob_reduction,
        )
        rewards = compute_rewards(
            batch_meta,
            generated,
            vocab,
            reward_valid_weight=reward_valid_weight,
            reward_strict_weight=reward_strict_weight,
            reward_distance_weight=reward_distance_weight,
            reward_distance_clip=reward_distance_clip,
            reward_source_similarity_weight=reward_source_similarity_weight,
            reward_source_similarity_threshold=reward_source_similarity_threshold,
        ).to(device)
        rewards_2d = rewards.view(len(batch_rows), rollouts_per_prompt)
        advantages = group_relative_advantages(
            rewards_2d,
            mode=advantage_mode,
            clip=advantage_clip,
        ).reshape(-1)
        pg_loss = -(advantages.detach() * seq_logprob).mean()
        kl_loss = torch.zeros((), dtype=pg_loss.dtype, device=device)
        if reference_model is not None and float(reference_kl_weight) > 0:
            ref_seq_logprob = sequence_logprobs(
                reference_model,
                expanded["condition"],
                expanded["condition_mask"],
                generated.to(device),
                eos_id=vocab.eos_id,
                reduction=sequence_logprob_reduction,
            )
            kl_gap = seq_logprob - ref_seq_logprob
            kl_loss = float(reference_kl_weight) * kl_gap.mean()

        logits = model(batch["condition"], batch["decoder_input_ids"], condition_mask=batch["condition_mask"])
        sft_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["target_ids"].reshape(-1),
            ignore_index=vocab.pad_id,
        )
        loss = pg_loss + float(sft_weight) * sft_loss + kl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_pg_loss += float(pg_loss.detach().cpu())
        total_sft_loss += float(sft_loss.detach().cpu())
        total_kl_loss += float(kl_loss.detach().cpu())
        total_reward += float(rewards.mean().detach().cpu())
        total_batches += 1
    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "pg_loss": total_pg_loss / denom,
        "sft_loss": total_sft_loss / denom,
        "kl_loss": total_kl_loss / denom,
        "mean_reward": total_reward / denom,
        "batches": total_batches,
    }


@torch.no_grad()
def evaluate_rl(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    *,
    batch_size: int,
    device: torch.device,
    rollouts_per_prompt: int,
    parallel_samples: int,
    max_parallel_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
) -> dict[str, float]:
    model.eval()
    rewards = []
    suppress_ids = [vocab.token_to_id[UNK]] if UNK in vocab.token_to_id else []
    dataset_index = 0
    for batch_rows in batches(dataset, batch_size):
        batch = collate(batch_rows, pad_id=vocab.pad_id, device=device)
        batch_meta = rows[dataset_index : dataset_index + len(batch_rows)]
        generated = sample_rollouts(
            model,
            batch,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            rollouts_per_prompt=rollouts_per_prompt,
            parallel_samples=parallel_samples,
            max_parallel_sequences=max_parallel_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            suppress_ids=suppress_ids,
        )
        reward = compute_rewards(
            batch_meta,
            generated,
            vocab,
            reward_valid_weight=reward_valid_weight,
            reward_strict_weight=reward_strict_weight,
            reward_distance_weight=reward_distance_weight,
            reward_distance_clip=reward_distance_clip,
            reward_source_similarity_weight=reward_source_similarity_weight,
            reward_source_similarity_threshold=reward_source_similarity_threshold,
        )
        rewards.append(float(reward.mean()))
        dataset_index += len(batch_rows)
    return {"mean_reward": sum(rewards) / max(len(rewards), 1)}


@torch.no_grad()
def sample_rollouts(
    model: ConditionedSmilesDecoder,
    batch: Mapping[str, torch.Tensor],
    *,
    bos_id: int,
    eos_id: int,
    rollouts_per_prompt: int,
    parallel_samples: int,
    max_parallel_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    suppress_ids: Sequence[int],
) -> torch.Tensor:
    prompt_count = int(batch["condition"].shape[0])
    remaining = max(1, int(rollouts_per_prompt))
    outputs: list[torch.Tensor] = []
    parallel_samples = max(1, int(parallel_samples))
    max_parallel_sequences = max(1, int(max_parallel_sequences))
    while remaining > 0:
        chunk_limit = max(1, max_parallel_sequences // max(prompt_count, 1))
        chunk = min(remaining, parallel_samples, chunk_limit)
        expanded = _repeat_generation_batch(batch, repeats=chunk)
        generated = model.generate(
            expanded["condition"],
            bos_id=bos_id,
            eos_id=eos_id,
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
        # Keep rollouts grouped by prompt before flattening so rewards/logprobs stay aligned.
        outputs.append(generated.view(prompt_count, chunk, generated.shape[1]))
        remaining -= chunk
    max_seq_len = max(tensor.shape[-1] for tensor in outputs)
    padded_outputs: list[torch.Tensor] = []
    for tensor in outputs:
        if tensor.shape[-1] < max_seq_len:
            pad = torch.full(
                (tensor.shape[0], tensor.shape[1], max_seq_len - tensor.shape[-1]),
                int(eos_id),
                dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, pad], dim=-1)
        padded_outputs.append(tensor)
    merged = torch.cat(padded_outputs, dim=1)
    return merged.reshape(prompt_count * merged.shape[1], max_seq_len)


def sequence_logprobs(
    model: ConditionedSmilesDecoder,
    condition: torch.Tensor,
    condition_mask: torch.Tensor,
    generated: torch.Tensor,
    *,
    eos_id: int,
    reduction: str = "sum",
) -> torch.Tensor:
    decoder_input = generated[:, :-1]
    target_ids = generated[:, 1:]
    logits = model(condition, decoder_input, condition_mask=condition_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    eos_hits = target_ids.eq(int(eos_id)).cumsum(dim=1)
    token_mask = eos_hits.le(1).to(token_log_probs.dtype)
    seq_logprob = (token_log_probs * token_mask).sum(dim=1)
    if reduction == "mean":
        token_count = token_mask.sum(dim=1).clamp_min(1.0)
        seq_logprob = seq_logprob / token_count
    return seq_logprob


def group_relative_advantages(
    rewards_2d: torch.Tensor,
    *,
    mode: str,
    clip: float,
) -> torch.Tensor:
    centered = rewards_2d - rewards_2d.mean(dim=1, keepdim=True)
    if mode == "group_zscore":
        std = rewards_2d.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        centered = centered / std
    if clip and clip > 0:
        centered = torch.clamp(centered, min=-float(clip), max=float(clip))
    return centered


def compute_rewards(
    rows: Sequence[Mapping[str, str]],
    generated: torch.Tensor,
    vocab: SmilesVocabulary,
    *,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_source_similarity_weight: float,
    reward_source_similarity_threshold: float,
) -> torch.Tensor:
    rewards = []
    rollout_count = max(1, int(generated.shape[0] // max(len(rows), 1)))
    for row_idx, row in enumerate(rows):
        start = row_idx * rollout_count
        end = start + rollout_count
        for ids in generated[start:end]:
            smiles = detokenize_smiles(vocab.decode(ids.tolist()[1:]))
            rewards.append(
                reward_for_smiles(
                    row,
                    smiles,
                    reward_valid_weight=reward_valid_weight,
                    reward_strict_weight=reward_strict_weight,
                    reward_distance_weight=reward_distance_weight,
                    reward_distance_clip=reward_distance_clip,
                    reward_source_similarity_weight=reward_source_similarity_weight,
                    reward_source_similarity_threshold=reward_source_similarity_threshold,
                )
            )
    return torch.as_tensor(rewards, dtype=torch.float32)


def reward_for_smiles(
    row: Mapping[str, str],
    smiles: str,
    *,
    reward_valid_weight: float,
    reward_strict_weight: float,
    reward_distance_weight: float,
    reward_distance_clip: float,
    reward_source_similarity_weight: float = 0.0,
    reward_source_similarity_threshold: float = 0.4,
) -> float:
    canonical = _safe_canonical_smiles(smiles)
    if not canonical:
        return -1.0
    strict_fraction, distance = property_score_components(row, canonical)
    distance = min(float(distance), float(reward_distance_clip))
    valid_reward = float(reward_valid_weight)
    strict_reward = float(reward_strict_weight) * float(strict_fraction)
    distance_penalty = float(reward_distance_weight) * float(distance)
    source_similarity_reward = float(reward_source_similarity_weight) * source_similarity_component(
        row,
        canonical,
        threshold=float(reward_source_similarity_threshold),
    )
    return valid_reward + strict_reward - distance_penalty + source_similarity_reward


def source_similarity_component(row: Mapping[str, str], smiles: str, *, threshold: float) -> float:
    source_smiles = str(row.get("source_smiles", "") or "").strip()
    if not source_smiles:
        return 0.0
    try:
        similarity = morgan_tanimoto(source_smiles, smiles)
    except RuntimeError:
        return 0.0
    if similarity is None or not math.isfinite(float(similarity)):
        return 0.0
    value = max(0.0, min(1.0, float(similarity)))
    threshold = max(0.0, min(0.999, float(threshold)))
    if threshold <= 0:
        return value
    return (value - threshold) / max(1.0 - threshold, 1e-6)


if __name__ == "__main__":
    raise SystemExit(main())
