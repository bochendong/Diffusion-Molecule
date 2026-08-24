#!/usr/bin/env python3
"""Sample the raw de-novo arm and record the shared checkpoint identity."""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--eval-features-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    unified.seed_everything(int(args.seed))
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
    rows = p6.read_rows(args.eval_csv)
    action_ids = [
        token_id
        for token, token_id in vocab.token_to_id.items()
        if token.startswith("<") and token not in unified.SPECIAL_TOKENS
    ]
    output: list[dict[str, object]] = []
    valid = 0
    with torch.no_grad():
        for row_index, row in enumerate(rows):
            condition_np = unified.condition_array_for_row(
                row, store, int(config["condition_dim"]), max_source_tokens=96,
                condition_layout="direct_compat",
            ).astype(np.float32)
            condition = torch.from_numpy(condition_np)[None, :, :].to(device)
            condition = condition.expand(int(args.num_samples), -1, -1)
            mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=device)
            generated = model.generate(
                condition, bos_id=vocab.bos_id, eos_id=vocab.eos_id,
                max_new_tokens=96, condition_mask=mask, temperature=0.85,
                top_k=40, top_p=0.95, repetition_penalty=1.15,
                no_repeat_ngram_size=6, min_new_tokens=6,
                blocked_token_ids=action_ids,
            )
            for candidate_index, ids in enumerate(generated.tolist()):
                raw = unified.detokenize_smiles(vocab.decode(ids[1:]))
                canonical = unified.safe_canonical_smiles(raw)
                valid += int(bool(canonical))
                item = dict(row)
                item.update({
                    "generated_smiles": canonical,
                    "direct_candidate_index": candidate_index,
                    "direct_candidate_raw_smiles": raw,
                    "direct_candidate_canonical_smiles": canonical,
                    "method": "p8_1_1_short_transaction",
                    "generation_rank": candidate_index + 1,
                    "candidate_rank": candidate_index + 1,
                })
                item.update(unified.candidate_metrics(row, canonical, source_similarity_threshold=0.65))
                item["direct_candidate_strict_fraction"] = item["unified_property_success_fraction"]
                output.append(item)
            if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
                print(f"[p8.1.1-denovo] {row_index + 1}/{len(rows)}", flush=True)
    p6.write_rows(args.output_csv, output)
    total = len(rows) * int(args.num_samples)
    summary = {
        "protocol": "p8_1_1_raw_denovo",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": policy.checkpoint_sha256(args.checkpoint),
        "eval_rows": len(rows),
        "num_samples": int(args.num_samples),
        "valid_candidates": valid,
        "candidate_validity": valid / max(total, 1),
        "property_reranking": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

