#!/usr/bin/env python3
"""C5: GRPO on the GraphEditDSL scorer, then rescore the frozen C1 action pool.

Family prior stays 0.5/0.5. B31 is frozen. The test program set is unchanged;
only graph_action_policy_logprob is rewritten. No ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
USG_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
UNIFIED_SCRIPTS = PROJECT_DIR.parent / "SketchMol-Unified-3MDiffusion" / "scripts"
for path in (USG_DIR, PROJECT_DIR / "scripts", UNIFIED_SCRIPTS, SCRIPT_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import umtp_graph_action_policy as action_policy  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402
from evaluate_moledit_table_metrics import (  # noqa: E402
    Chemistry,
    evaluate_prediction,
    task_specs_for_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c5-protocol-manifest", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-features-dir", required=True, type=Path)
    parser.add_argument("--eval-features-dir", required=True, type=Path)
    parser.add_argument("--graph-candidate-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--score-batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    c5 = json.loads(args.c5_protocol_manifest.read_text(encoding="utf-8"))
    device = unified.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unified.seed_everything(int(c5["seed"]))

    checkpoint = unified.load_checkpoint(args.base_checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(args.base_checkpoint)
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])
    model = unified.ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    trainable_scope = action_policy.configure_trainable_scope(
        model,
        scope=str(c5["trainable_scope"]),
        old_vocab_size=int(c5["old_vocab_size"]),
    )
    ref_model = None
    if float(c5.get("kl_coef") or 0.0) > 0.0:
        ref_model = unified.ConditionedSmilesDecoder(**config).to(device)
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(c5["lr"]),
        weight_decay=0.0,
    )
    train_store = unified.FeatureStore(
        args.train_features_dir, array_name="query_tokens", variant="full"
    )
    eval_store = unified.FeatureStore(
        args.eval_features_dir, array_name="query_tokens", variant="full"
    )
    chem = Chemistry()
    train_summary = train_grpo(
        model=model,
        vocab=vocab,
        config=config,
        optimizer=optimizer,
        train_csv=args.train_csv,
        train_store=train_store,
        c5=c5,
        device=device,
        chem=chem,
        train_limit=int(args.train_limit),
        score_batch_size=int(args.score_batch_size),
        trainable_scope=trainable_scope,
        ref_model=ref_model,
    )
    ckpt_path = args.output_dir / "umtp_graph_action_policy_grpo.pt"
    dummy_args = SimpleNamespace(command="c5_grpo", seed=int(c5["seed"]))
    unified.save_checkpoint(
        ckpt_path, model, optimizer, vocab, config, int(c5["grpo_epochs"]), [train_summary], dummy_args
    )
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "trained", **train_summary}, indent=2, sort_keys=True), flush=True)

    rescored_path = args.output_dir / "graph_action_candidates_rescored.csv"
    rescore_summary = rescore_pool(
        model=model,
        vocab=vocab,
        config=config,
        eval_csv=args.eval_csv,
        eval_store=eval_store,
        candidate_csv=args.graph_candidate_csv,
        output_csv=rescored_path,
        device=device,
        score_batch_size=int(args.score_batch_size),
    )
    (args.output_dir / "rescore_summary.json").write_text(
        json.dumps(rescore_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "rescored", **rescore_summary}, indent=2, sort_keys=True), flush=True)
    return 0


def train_grpo(
    *,
    model,
    vocab,
    config: Mapping[str, object],
    optimizer,
    train_csv: Path,
    train_store,
    c5: Mapping[str, object],
    device: torch.device,
    chem: Chemistry,
    train_limit: int,
    score_batch_size: int,
    trainable_scope: Mapping[str, object],
    ref_model=None,
) -> dict[str, object]:
    rows = action_policy.read_rows(train_csv)
    if train_limit > 0:
        rows = rows[: int(train_limit)]
    group_size = int(c5["grpo_group_size"])
    clip_eps = float(c5["grpo_clip_eps"])
    entropy_coef = float(c5["entropy_coef"])
    kl_coef = float(c5.get("kl_coef") or 0.0)
    epochs = int(c5["grpo_epochs"])
    updates = 0
    skipped = 0
    mean_rewards: list[float] = []
    clip_fractions: list[float] = []
    mean_kls: list[float] = []
    mean_entropies: list[float] = []
    for epoch in range(epochs):
        for index, row in enumerate(rows):
            candidates = action_policy.enumerate_action_candidates(
                row, site_limit=32, max_actions_per_row=256
            )
            if len(candidates) < 2:
                skipped += 1
                continue
            condition = unified.condition_array_for_row(
                row,
                train_store,
                int(config["condition_dim"]),
                max_source_tokens=96,
                condition_layout="transformation",
            ).astype(np.float32)
            programs = [program for _action, _smiles, program in candidates]
            smiles = [item[1] for item in candidates]
            with torch.no_grad():
                old_scores = score_programs_tensor(
                    model, vocab, condition, programs, batch_size=score_batch_size, device=device
                )
            old_logp = torch.log_softmax(old_scores, dim=0)
            probs = torch.softmax(old_scores, dim=0)
            sampled = torch.multinomial(probs, group_size, replacement=True)
            rewards = torch.tensor(
                [terminal_reward(row, smiles[int(idx)], chem) for idx in sampled],
                dtype=torch.float32,
                device=device,
            )
            mean_rewards.append(float(rewards.mean().item()))
            if float(rewards.std().item()) < 1e-8:
                skipped += 1
                continue
            advantages = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-6)
            model.train()
            new_scores = score_programs_tensor(
                model, vocab, condition, programs, batch_size=score_batch_size, device=device
            )
            new_logp = torch.log_softmax(new_scores, dim=0)
            chosen_new = new_logp[sampled]
            chosen_old = old_logp[sampled]
            ratio = torch.exp(chosen_new - chosen_old)
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg = torch.min(ratio * advantages, clipped * advantages)
            if ref_model is not None and kl_coef > 0.0:
                new_probs = torch.softmax(new_scores, dim=0)
                entropy = -(new_probs * new_logp).sum()
                with torch.no_grad():
                    ref_scores = score_programs_tensor(
                        ref_model,
                        vocab,
                        condition,
                        programs,
                        batch_size=score_batch_size,
                        device=device,
                    )
                    ref_logp = torch.log_softmax(ref_scores, dim=0)
                kl = (new_probs * (new_logp - ref_logp)).sum()
                loss = -pg.mean() - entropy_coef * entropy + kl_coef * kl
                mean_kls.append(float(kl.detach().item()))
            else:
                entropy = -(probs.detach() * old_logp).sum()
                loss = -pg.mean() - entropy_coef * entropy
            mean_entropies.append(float(entropy.detach().item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            updates += 1
            clip_fractions.append(float((ratio - clipped).abs().gt(1e-8).float().mean().item()))
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "stage": "grpo",
                            "epoch": epoch + 1,
                            "done": index + 1,
                            "total": len(rows),
                            "updates": updates,
                            "mean_reward": sum(mean_rewards[-20:]) / max(1, len(mean_rewards[-20:])),
                            "mean_kl": sum(mean_kls[-20:]) / max(1, len(mean_kls[-20:])),
                            "mean_entropy": sum(mean_entropies[-20:]) / max(1, len(mean_entropies[-20:])),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return {
        "protocol": c5["protocol"],
        "train_rows": len(rows),
        "updates": updates,
        "skipped": skipped,
        "epochs": epochs,
        "group_size": group_size,
        "mean_reward": sum(mean_rewards) / max(1, len(mean_rewards)),
        "mean_clip_fraction": sum(clip_fractions) / max(1, len(clip_fractions)),
        "mean_kl": sum(mean_kls) / max(1, len(mean_kls)),
        "kl_coef": kl_coef,
        "entropy_coef": entropy_coef,
        "mean_entropy": sum(mean_entropies) / max(1, len(mean_entropies)),
        "trainable_scope": dict(trainable_scope),
        "device": str(device),
    }


def rescore_pool(
    *,
    model,
    vocab,
    config: Mapping[str, object],
    eval_csv: Path,
    eval_store,
    candidate_csv: Path,
    output_csv: Path,
    device: torch.device,
    score_batch_size: int,
) -> dict[str, object]:
    eval_rows = {action_policy.row_id(row): row for row in action_policy.read_rows(eval_csv)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with candidate_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            condition_id = str(raw.get("condition_id", "") or "").strip()
            if condition_id:
                grouped[condition_id].append(raw)
    model.eval()
    out_rows: list[dict[str, str]] = []
    scored = 0
    missing = 0
    for condition_id, items in grouped.items():
        row = eval_rows.get(condition_id)
        if row is None:
            missing += 1
            out_rows.extend(items)
            continue
        programs = []
        scored_items = []
        for item in items:
            raw = str(item.get("graph_action_program_tokens_json", "") or "").strip()
            if not raw:
                out_rows.append(item)
                continue
            programs.append(json.loads(raw))
            scored_items.append(item)
        if not programs:
            continue
        condition = unified.condition_array_for_row(
            row,
            eval_store,
            int(config["condition_dim"]),
            max_source_tokens=96,
            condition_layout="transformation",
        ).astype(np.float32)
        with torch.no_grad():
            scores = score_programs_tensor(
                model, vocab, condition, programs, batch_size=score_batch_size, device=device
            )
        for item, score in zip(scored_items, scores.tolist()):
            updated = dict(item)
            updated["graph_action_policy_logprob"] = action_policy.format_float(float(score))
            updated["method"] = "umtp_graph_action_policy_grpo"
            out_rows.append(updated)
        scored += 1
        if scored % 50 == 0:
            print(json.dumps({"stage": "rescore", "scored": scored, "groups": len(grouped)}), flush=True)
    action_policy.write_rows(output_csv, out_rows)
    return {
        "candidate_rows": len(out_rows),
        "rescored_conditions": scored,
        "missing_eval_rows": missing,
        "candidate_csv": str(output_csv),
    }


def score_programs_tensor(
    model,
    vocab,
    condition: np.ndarray,
    programs: Sequence[Sequence[str]],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(programs), max(1, int(batch_size))):
        items = []
        for tokens in programs[start : start + max(1, int(batch_size))]:
            items.append(
                {
                    "condition": condition,
                    "decoder_input_ids": np.asarray(vocab.encode(tokens, add_bos=True), dtype=np.int64),
                    "target_ids": np.asarray(vocab.encode(tokens, add_eos=True), dtype=np.int64),
                    "task_mode": unified.EDIT_MODE,
                }
            )
        batch = {key: value.to(device) for key, value in unified.collate_batch(items, model.pad_id).items()}
        logits = model(
            batch["condition"],
            batch["decoder_input_ids"],
            condition_mask=batch["condition_mask"],
        )
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)
        mask = batch["target_ids"].ne(model.pad_id)
        chunks.append((token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
    return torch.cat(chunks, dim=0)


def terminal_reward(row: Mapping[str, str], smiles: str, chem: Chemistry) -> float:
    if not smiles:
        return 0.0
    scored = evaluate_prediction(
        row,
        smiles,
        task_specs_for_reference(row),
        chem=chem,
        thresholds=[0.65, 0.15],
    )
    tanimoto = scored.get("source_tanimoto")
    return (
        float(bool(scored.get("success_t0.65")))
        + 0.5 * float(bool(scored.get("property_success")))
        + 0.25 * (0.0 if tanimoto is None else float(tanimoto))
    )


if __name__ == "__main__":
    raise SystemExit(main())
