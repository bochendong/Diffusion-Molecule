#!/usr/bin/env python3
"""Build preference pairs for DPO-style direct SMILES fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_direct_smiles_generator import (  # noqa: E402
    UNK,
    _repeat_generation_batch,
    append_csv_rows,
    batches,
    build_dataset,
    collate,
    load_checkpoint,
    load_store,
    read_rows,
    resolve_device,
    resolve_condition_mixing_mode,
    score_generated_candidate,
    seed_everything,
)
from sketchmol_understanding_condition.direct_smiles_generation import (  # noqa: E402
    ConditionedSmilesDecoder,
    SmilesVocabulary,
    detokenize_smiles,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--condition-features-dir", type=Path, default=None)
    parser.add_argument("--condition-feature-array", default="query_tokens")
    parser.add_argument("--condition-feature-variant", default="full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--parallel-samples", type=int, default=8)
    parser.add_argument("--max-parallel-sequences", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--min-new-tokens", type=int, default=6)
    parser.add_argument("--disable-property-rerank", action="store_true")
    parser.add_argument("--rejected-strategy", choices=("hard_valid", "best_invalid", "worst"), default="hard_valid")
    parser.add_argument("--min-score-gap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.resume_checkpoint)
    if checkpoint is None:
        raise ValueError("--resume-checkpoint is required")
    checkpoint_args = dict(checkpoint.get("args", {}))
    condition_mixing_mode = resolve_condition_mixing_mode(args, checkpoint_args)
    vocab = SmilesVocabulary.from_dict(checkpoint["vocab"])
    config = dict(checkpoint["model_config"])

    rows = read_rows(args.rows_csv, limit=args.limit)
    store = load_store(args.condition_features_dir, args)
    dataset = build_dataset(
        rows,
        vocab,
        store,
        int(config["condition_dim"]),
        max_smiles_length=int(config["max_length"]) - 8,
        condition_mixing_mode=condition_mixing_mode,
    )
    rows = [row for row in rows if str(row.get("target_smiles", "") or "").strip()]

    model = ConditionedSmilesDecoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    summary = write_preference_pairs(
        model,
        dataset,
        rows,
        vocab,
        args.output_csv,
        batch_size=int(args.batch_size),
        device=device,
        num_samples=int(args.num_samples),
        parallel_samples=int(args.parallel_samples),
        max_parallel_sequences=int(args.max_parallel_sequences),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
        min_new_tokens=int(args.min_new_tokens),
        rejected_strategy=str(args.rejected_strategy),
        min_score_gap=float(args.min_score_gap),
        property_rerank=not bool(args.disable_property_rerank),
    )
    payload = {
        "rows_csv": str(args.rows_csv),
        "output_csv": str(args.output_csv),
        "checkpoint": str(args.resume_checkpoint),
        "condition_features_dir": str(args.condition_features_dir) if args.condition_features_dir else None,
        "condition_mixing_mode": condition_mixing_mode,
        "summary": summary,
        "device": str(device),
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def write_preference_pairs(
    model: ConditionedSmilesDecoder,
    dataset: list[dict[str, object]],
    rows: list[dict[str, str]],
    vocab: SmilesVocabulary,
    output_csv: Path,
    *,
    batch_size: int,
    device: torch.device,
    num_samples: int,
    parallel_samples: int,
    max_parallel_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens: int,
    rejected_strategy: str,
    min_score_gap: float,
    property_rerank: bool,
) -> dict[str, object]:
    model.eval()
    sample_count = max(2, int(num_samples))
    sample_parallel = max(1, int(parallel_samples))
    max_parallel_sequences = max(1, int(max_parallel_sequences))
    suppress_ids = [vocab.token_to_id[UNK]] if UNK in vocab.token_to_id else []
    dataset_index = 0
    csv_initialized = False
    kept_rows = 0
    skipped_rows = 0
    invalid_only_rows = 0
    valid_negative_rows = 0
    invalid_negative_rows = 0
    score_gaps: list[float] = []

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
            source_row = rows[dataset_index]
            scored = [
                score_generated_candidate(source_row, candidate, rank=rank, property_rerank=property_rerank)
                for rank, candidate in enumerate(candidates)
            ]
            pair = select_preference_pair(
                source_row,
                scored,
                rejected_strategy=rejected_strategy,
                min_score_gap=min_score_gap,
            )
            if pair is None:
                if any(item.get("canonical_smiles") for item in scored):
                    skipped_rows += 1
                else:
                    invalid_only_rows += 1
                dataset_index += 1
                continue
            kept_rows += 1
            if pair["rejected_valid"] == "True":
                valid_negative_rows += 1
            else:
                invalid_negative_rows += 1
            score_gaps.append(float(pair["score_gap"]))
            batch_output_rows.append(pair)
            dataset_index += 1

        append_csv_rows(output_csv, batch_output_rows, overwrite=not csv_initialized)
        csv_initialized = True
        print(
            f"[preference-build] kept {kept_rows}/{len(rows)} rows "
            f"(invalid_only={invalid_only_rows}, skipped_gap={skipped_rows})",
            flush=True,
        )

    return {
        "rows": kept_rows,
        "skipped_rows": skipped_rows,
        "invalid_only_rows": invalid_only_rows,
        "valid_negative_rows": valid_negative_rows,
        "invalid_negative_rows": invalid_negative_rows,
        "mean_score_gap": sum(score_gaps) / len(score_gaps) if score_gaps else 0.0,
        "num_samples": sample_count,
        "parallel_samples": sample_parallel,
        "max_parallel_sequences": max_parallel_sequences,
        "rejected_strategy": rejected_strategy,
        "property_rerank": bool(property_rerank),
    }


def select_preference_pair(
    row: Mapping[str, str],
    scored: Sequence[Mapping[str, object]],
    *,
    rejected_strategy: str,
    min_score_gap: float,
) -> dict[str, object] | None:
    valid = [item for item in scored if str(item.get("canonical_smiles") or "").strip()]
    if not valid:
        return None
    chosen = max(valid, key=lambda item: float(item["score"]))
    remaining = [item for item in scored if item is not chosen and not _same_candidate(item, chosen)]
    if not remaining:
        return None
    rejected = choose_rejected_candidate(remaining, strategy=rejected_strategy)
    if rejected is None:
        return None
    score_gap = float(chosen["score"]) - float(rejected["score"])
    if score_gap < float(min_score_gap):
        return None
    pair = dict(row)
    pair.update(
        {
            "chosen_smiles": str(chosen.get("canonical_smiles") or chosen.get("raw_smiles") or ""),
            "rejected_smiles": str(rejected.get("canonical_smiles") or rejected.get("raw_smiles") or ""),
            "chosen_raw_smiles": str(chosen.get("raw_smiles") or ""),
            "rejected_raw_smiles": str(rejected.get("raw_smiles") or ""),
            "chosen_rank": int(chosen["rank"]),
            "rejected_rank": int(rejected["rank"]),
            "chosen_score": float(chosen["score"]),
            "rejected_score": float(rejected["score"]),
            "score_gap": float(score_gap),
            "chosen_strict_fraction": float(chosen["strict_fraction"]),
            "rejected_strict_fraction": float(rejected["strict_fraction"]),
            "chosen_property_distance": float(chosen["normalized_property_distance"]),
            "rejected_property_distance": float(rejected["normalized_property_distance"]),
            "chosen_valid": "True",
            "rejected_valid": "True" if str(rejected.get("canonical_smiles") or "").strip() else "False",
            "rejected_strategy": rejected_strategy,
            "candidate_count": len(scored),
            "valid_candidate_count": sum(1 for item in scored if str(item.get("canonical_smiles") or "").strip()),
            "unique_valid_candidate_count": len(
                {str(item.get("canonical_smiles") or "") for item in scored if str(item.get("canonical_smiles") or "").strip()}
            ),
        }
    )
    return pair


def choose_rejected_candidate(
    scored: Sequence[Mapping[str, object]],
    *,
    strategy: str,
) -> Mapping[str, object] | None:
    valid = [item for item in scored if str(item.get("canonical_smiles") or "").strip()]
    invalid = [item for item in scored if not str(item.get("canonical_smiles") or "").strip()]
    if strategy == "hard_valid":
        if valid:
            return min(valid, key=lambda item: float(item["score"]))
        if invalid:
            return max(invalid, key=lambda item: float(item["score"]))
    elif strategy == "best_invalid":
        if invalid:
            return max(invalid, key=lambda item: float(item["score"]))
        if valid:
            return min(valid, key=lambda item: float(item["score"]))
    elif strategy == "worst":
        return min(scored, key=lambda item: float(item["score"]))
    return None


def _same_candidate(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_canon = str(left.get("canonical_smiles") or "").strip()
    right_canon = str(right.get("canonical_smiles") or "").strip()
    if left_canon and right_canon:
        return left_canon == right_canon
    return str(left.get("raw_smiles") or "").strip() == str(right.get("raw_smiles") or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
