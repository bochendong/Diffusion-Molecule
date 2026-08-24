#!/usr/bin/env python3
"""Sample short executable edit transactions without target/property reranking."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
P6_DIR = PROJECT_DIR / "experiments" / "p6_unified_molecular_transition_policy"
UNIFIED_DIR = PROJECT_DIR / "experiments" / "unified_smiles_generator"
for path in (P6_DIR, UNIFIED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import p6_transition_program as p6  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402
import umtp_graph_action_policy as policy  # noqa: E402


def source_only_candidates(row: dict[str, str], *, site_limit: int, limit: int):
    """Executable support determined only by the supplied molecular state."""
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "")).strip()
    actions = policy.balanced_action_cap(
        policy.universal_actions(source, site_limit=int(site_limit)), int(limit)
    )
    seen: set[str] = set()
    canonical_source = unified.safe_canonical_smiles(source)
    if canonical_source:
        seen.add(canonical_source)
    out = []
    for action in actions:
        try:
            program = policy.action_program_tokens(action)
        except ValueError:
            continue
        generated = policy.graph.execute_graph_edit_action(source, action)
        canonical = unified.safe_canonical_smiles(generated)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append((action, canonical, program))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--eval-features-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2907)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    unified.seed_everything(int(args.seed))
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    device = unified.resolve_device(str(args.device))
    checkpoint = unified.load_checkpoint(args.checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(args.checkpoint)
    vocab = unified.SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])
    model = unified.ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    store = unified.FeatureStore(args.eval_features_dir, array_name="query_tokens", variant="full")
    rows = [row for row in policy.read_rows(args.eval_csv) if unified.task_mode_for_row(row) == unified.EDIT_MODE]
    output: list[dict[str, object]] = []
    pool_sizes: list[int] = []
    normalized_entropies: list[float] = []
    maximum_probabilities: list[float] = []
    for row_index, row in enumerate(rows):
        candidates = source_only_candidates(row, site_limit=args.site_limit, limit=args.max_actions)
        pool_sizes.append(len(candidates))
        if not candidates:
            continue
        condition = unified.condition_array_for_row(
            row, store, int(config["condition_dim"]), max_source_tokens=96,
            condition_layout="direct_compat",
        ).astype(np.float32)
        scores = policy.score_programs(
            model, vocab, condition, [item[2] for item in candidates],
            batch_size=int(args.score_batch_size), device=device,
        )
        logits = torch.tensor(scores, dtype=torch.float64) / max(float(args.temperature), 1e-6)
        probs = torch.softmax(logits, dim=0)
        entropy = float(-(probs * probs.clamp_min(1e-300).log()).sum())
        normalized_entropies.append(entropy / max(math.log(max(len(candidates), 2)), 1e-12))
        maximum_probabilities.append(float(probs.max()))
        take = min(int(args.num_samples), len(candidates))
        sampled = torch.multinomial(probs, take, replacement=False, generator=generator).tolist()
        for rank, candidate_index in enumerate(sampled, start=1):
            action, smiles, program = candidates[candidate_index]
            item = dict(row)
            item.update({
                "generated_smiles": smiles,
                "method": "p8_1_1_short_transaction",
                "generation_rank": rank,
                "candidate_rank": rank,
                "transaction_program_tokens_json": json.dumps(program),
                "transaction_action_json": json.dumps(asdict(action), sort_keys=True),
                "transaction_policy_logprob": policy.format_float(scores[candidate_index]),
                "transaction_sampling_probability": policy.format_float(float(probs[candidate_index])),
                "transaction_candidate_count": len(candidates),
            })
            item.update(unified.candidate_metrics(row, smiles, source_similarity_threshold=0.65))
            output.append(item)
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            print(f"[p8.1.1-edit] {row_index + 1}/{len(rows)}", flush=True)
    policy.write_rows(args.output_csv, output)
    summary = {
        "protocol": "p8_1_1_raw_transaction_sampling",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": policy.checkpoint_sha256(args.checkpoint),
        "eval_rows": len(rows),
        "rows_with_candidates": sum(size > 0 for size in pool_sizes),
        "num_samples": int(args.num_samples),
        "temperature": float(args.temperature),
        "mean_candidate_pool": sum(pool_sizes) / max(len(pool_sizes), 1),
        "min_candidate_pool": min(pool_sizes, default=0),
        "mean_normalized_policy_entropy": (
            sum(normalized_entropies) / max(len(normalized_entropies), 1)
        ),
        "mean_maximum_transaction_probability": (
            sum(maximum_probabilities) / max(len(maximum_probabilities), 1)
        ),
        "source_only_support": True,
        "property_reranking": False,
        "target_molecule_used_at_inference": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
