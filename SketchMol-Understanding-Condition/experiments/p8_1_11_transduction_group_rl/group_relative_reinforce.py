#!/usr/bin/env python3
"""Group-relative REINFORCE for the constrained P8.1.2 transducer.

This is deliberately not named GRPO: it uses a single on-policy REINFORCE
update with within-prompt standardized rewards, a sampled squared log-ratio
reference penalty, and an
oracle-program SFT anchor. There is no PPO ratio or clipped surrogate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
P812_DIR = PROJECT_DIR / "experiments" / "p8_1_2_unified_transduction"
P6_DIR = PROJECT_DIR / "experiments" / "p6_unified_molecular_transition_policy"
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for path in (P812_DIR, P6_DIR, UNIFIED_DIR): sys.path.insert(0, str(path))
import p6_transition_program as p6  # noqa: E402
import sample_transduction_policy as sampler  # noqa: E402
import transduction_oracle as transduction  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path); parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--features-dir", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reward-aggregation", choices=("joint_bottleneck", "dense_softmin"), required=True)
    parser.add_argument("--rollouts", type=int, default=4); parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-6); parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--softmin-temperature", type=float, default=0.25); parser.add_argument("--reference-logratio-weight", type=float, default=0.05)
    parser.add_argument("--sft-weight", type=float, default=0.10); parser.add_argument("--seed", type=int, default=7); parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sequence_logprob(model, vocab, condition_np, programs, device):
    items = []
    for tokens in programs:
        items.append({
            "condition": condition_np,
            "decoder_input_ids": np.asarray(vocab.encode(tokens, add_bos=True), dtype=np.int64),
            "target_ids": np.asarray(vocab.encode(tokens, add_eos=True), dtype=np.int64),
            "task_mode": unified.EDIT_MODE,
        })
    batch = {key: value.to(device) for key, value in unified.collate_batch(items, model.pad_id).items()}
    logits = model(batch["condition"], batch["decoder_input_ids"], condition_mask=batch["condition_mask"])
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)
    mask = batch["target_ids"].ne(model.pad_id)
    return (selected * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def reward_components(row, smiles: str) -> list[float]:
    clean = dict(row); clean["target_smiles"] = ""  # target structure is never consulted by reward
    metrics = unified.candidate_metrics(clean, smiles, source_similarity_threshold=0.65)
    valid = float(bool(smiles))
    prop = max(0.0, min(1.0, float(metrics.get("unified_property_success_fraction") or 0.0)))
    components = [valid, prop]
    if unified.task_mode_for_row(clean) == unified.EDIT_MODE:
        try: similarity = float(metrics.get("source_tanimoto") or 0.0)
        except (TypeError, ValueError): similarity = 0.0
        components.append(max(0.0, min(1.0, similarity)))
    return components


def aggregate(values: list[float], mode: str, temperature: float) -> float:
    if mode == "joint_bottleneck": return min(values)
    tau = max(float(temperature), 1e-6)
    return -tau * math.log(sum(math.exp(-value / tau) for value in values) / len(values))


def decode_program(source_tokens, tokens):
    try: return transduction.canonical(transduction.decode_sf(transduction.execute_program(source_tokens, tokens)))
    except Exception: return ""


def main() -> int:
    args = parse_args(); unified.seed_everything(args.seed); device = unified.resolve_device(args.device)
    checkpoint = unified.load_checkpoint(args.base_checkpoint)
    if checkpoint is None: raise FileNotFoundError(args.base_checkpoint)
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"]); config = dict(checkpoint["model_config"])
    model = unified.ConditionedSmilesDecoder(**config).to(device); model.load_state_dict(checkpoint["model_state"])
    reference = unified.ConditionedSmilesDecoder(**config).to(device); reference.load_state_dict(checkpoint["model_state"]); reference.eval()
    for parameter in reference.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    rows = p6.read_rows(args.train_csv)[: args.max_prompts]
    store = unified.FeatureStore(args.features_dir, array_name="query_tokens", variant="full")
    semantic = sampler.observed_selfies_tokens(args.train_csv) & set(vocab.token_to_id)
    rewards_all, losses, active_groups = [], [], 0
    model.train()
    for index, row in enumerate(rows):
        source_smiles = transduction.canonical(p6.source_for_row(row)) if p6.source_for_row(row) else ""
        source_tokens = transduction.sf_tokens(source_smiles) if source_smiles else []
        condition_np = unified.condition_array_for_row(row, store, int(config["condition_dim"]), max_source_tokens=96, condition_layout="p6_transition").astype(np.float32)
        condition = torch.from_numpy(condition_np)[None].to(device).expand(args.rollouts, -1, -1)
        condition_mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
        with torch.no_grad():
            generated = sampler.generate(model, condition, condition_mask, vocab=vocab, source_length=len(source_tokens), semantic_tokens=semantic,
                max_new_tokens=128 if not source_tokens else 64, temperature=args.temperature, top_k=32, top_p=0.95)
        programs, rewards = [], []
        for ids in generated.tolist():
            tokens = vocab.decode(ids)
            if transduction.START in tokens: tokens = tokens[tokens.index(transduction.START):]
            programs.append(tokens)
            smiles = decode_program(source_tokens, tokens)
            rewards.append(aggregate(reward_components(row, smiles), args.reward_aggregation, args.softmin_temperature))
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = (reward_tensor - reward_tensor.mean()) / reward_tensor.std(unbiased=False).clamp_min(1e-4)
        current_logp = sequence_logprob(model, vocab, condition_np, programs, device)
        with torch.no_grad(): reference_logp = sequence_logprob(reference, vocab, condition_np, programs, device)
        reinforce = -(advantages.detach() * current_logp).mean()
        sampled_logratio_penalty = (current_logp - reference_logp).square().mean()
        try: oracle_program = json.loads(str(row.get("policy_target_tokens_json") or "[]"))
        except json.JSONDecodeError: oracle_program = []
        sft = -sequence_logprob(model, vocab, condition_np, [oracle_program], device).mean() if oracle_program else current_logp.new_zeros(())
        loss = reinforce + args.reference_logratio_weight * sampled_logratio_penalty + args.sft_weight * sft
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        active_groups += int(float(reward_tensor.std(unbiased=False)) > 1e-6); rewards_all.extend(rewards); losses.append(float(loss.detach()))
        if (index + 1) % 16 == 0: print(f"[p8.1.11] {index + 1}/{len(rows)} reward={sum(rewards_all)/len(rewards_all):.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "transduction_group_relative_reinforce.pt"
    history = [{"reward_mean": sum(rewards_all)/max(len(rewards_all), 1), "loss_mean": sum(losses)/max(len(losses), 1), "active_groups": active_groups}]
    unified.save_checkpoint(checkpoint_path, model, optimizer, vocab, config, 1, history, args)
    summary = {
        "protocol": "p8_1_11_group_relative_reinforce_v1", "algorithm": "group_relative_REINFORCE_not_GRPO",
        "checkpoint": str(checkpoint_path), "base_checkpoint": str(args.base_checkpoint), "reward_aggregation": args.reward_aggregation,
        "prompts": len(rows), "rollouts": args.rollouts, "active_groups": active_groups, "mean_reward": history[0]["reward_mean"],
        "eval_rows_used": 0, "eval_targets_used": 0, "target_structure_reward_access": False,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
