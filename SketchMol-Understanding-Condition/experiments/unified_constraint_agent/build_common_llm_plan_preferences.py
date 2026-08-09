#!/usr/bin/env python3
"""Build leakage-safe MuMO two-step plan preferences from train-only pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common_llm_plan_protocol as protocol  # noqa: E402
import rerank_common_llm_existing_action_plans as existing  # noqa: E402
import select_external_verifier_prefix as external  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-detail-csv", required=True, type=Path)
    parser.add_argument("--plan-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--max-negatives-per-condition", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1706)
    parser.add_argument("--require-input-split", default="train")
    parser.add_argument("--max-conditions", type=int, default=0)
    return parser.parse_args(argv)


def finite(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_validation(condition_id: str, *, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{condition_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < float(fraction)


def feedback(row: Mapping[str, object], *, action_count: int) -> dict[str, object]:
    return {
        "official_success": external.truthy(row.get("external_official_success")),
        "strict_success": external.truthy(row.get("external_strict_success")),
        "source_similarity_success": external.truthy(row.get("external_source_similarity_success")),
        "property_success_fraction": external.evaluated_success_fraction(row),
        "evaluated_property_fraction": finite(row.get("external_evaluated_property_fraction"), 0.0),
        "mean_relative_improvement": finite(row.get("external_mean_relative_improvement"), -1e6),
        "source_tanimoto": finite(row.get("external_source_tanimoto"), -1.0),
        "plan_action_count": int(action_count),
    }


def positive_key(item: tuple[Mapping[str, object], Sequence[Mapping[str, object]]]) -> tuple[float, ...]:
    row, actions = item
    return (
        finite(row.get("external_source_tanimoto"), -1.0),
        finite(row.get("external_mean_relative_improvement"), -1e6),
        external.evaluated_success_fraction(row),
        -len(actions),
        -external.candidate_rank(row),
    )


def negative_category(row: Mapping[str, object]) -> str:
    official_success = external.truthy(row.get("external_official_success"))
    similarity_success = external.truthy(row.get("external_source_similarity_success"))
    if official_success and not similarity_success:
        return "property_success_similarity_failure"
    if similarity_success and not official_success:
        return "similarity_success_property_failure"
    return "joint_near_miss"


def negative_key(item: tuple[Mapping[str, object], Sequence[Mapping[str, object]]]) -> tuple[float, ...]:
    row, actions = item
    return (
        external.evaluated_success_fraction(row),
        finite(row.get("external_evaluated_property_fraction"), 0.0),
        finite(row.get("external_source_tanimoto"), -1.0),
        finite(row.get("external_mean_relative_improvement"), -1e6),
        -len(actions),
        -external.candidate_rank(row),
    )


def select_negatives(
    items: Sequence[tuple[Mapping[str, object], Sequence[Mapping[str, object]]]],
    *,
    limit: int,
) -> list[tuple[str, Mapping[str, object], Sequence[Mapping[str, object]]]]:
    by_category: dict[str, list[tuple[Mapping[str, object], Sequence[Mapping[str, object]]]]] = defaultdict(list)
    for item in items:
        row, _actions = item
        if external.truthy(row.get("external_strict_success")):
            continue
        by_category[negative_category(row)].append(item)
    selected: list[tuple[str, Mapping[str, object], Sequence[Mapping[str, object]]]] = []
    category_order = (
        "property_success_similarity_failure",
        "similarity_success_property_failure",
        "joint_near_miss",
    )
    for category in category_order:
        if by_category.get(category):
            row, actions = max(by_category[category], key=negative_key)
            selected.append((category, row, actions))
            if len(selected) >= int(limit):
                return selected
    used = {existing.candidate_key(row) for _category, row, _actions in selected}
    remaining = [
        (row, actions)
        for item in by_category.values()
        for row, actions in item
        if existing.candidate_key(row) not in used
    ]
    for row, actions in sorted(remaining, key=negative_key, reverse=True):
        selected.append((negative_category(row), row, actions))
        if len(selected) >= int(limit):
            break
    return selected


def preference_pairs(
    rows: Sequence[Mapping[str, object]],
    plans: Mapping[tuple[str, str], Sequence[Mapping[str, object]]],
    *,
    max_negatives: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition_id", "") or "")].append(row)
    output: list[dict[str, object]] = []
    outcomes = Counter()
    for condition_id, group in grouped.items():
        candidates = [
            (row, plans[existing.candidate_key(row)])
            for row in group
            if existing.candidate_key(row) in plans
        ]
        if len(candidates) != len(group):
            outcomes["missing_reconstructed_plan"] += len(group) - len(candidates)
            continue
        positives = [item for item in candidates if external.truthy(item[0].get("external_strict_success"))]
        if not positives:
            outcomes["no_strict_positive_condition"] += 1
            continue
        chosen_row, chosen_actions = max(positives, key=positive_key)
        negatives = select_negatives(candidates, limit=max(1, int(max_negatives)))
        if not negatives:
            outcomes["no_hard_negative_condition"] += 1
            continue
        prompt_messages = protocol.plan_prompt_messages(chosen_row)
        prompt_text = json.dumps(prompt_messages, sort_keys=True)
        target_smiles = str(chosen_row.get("target_smiles", "") or "")
        source_smiles = str(chosen_row.get("source_smiles", "") or "")
        if target_smiles and target_smiles != source_smiles and target_smiles in prompt_text:
            raise ValueError(f"Target molecule leaked into plan prompt for {condition_id}")
        chosen_payload = protocol.plan_payload(chosen_actions)
        for negative_index, (category, rejected_row, rejected_actions) in enumerate(negatives):
            output.append(
                {
                    "pair_id": f"{condition_id}:{category}:{negative_index}",
                    "condition_id": condition_id,
                    "origin": "mumo",
                    "external_task_id": chosen_row.get("external_task_id", ""),
                    "external_task_split": chosen_row.get("external_task_split", ""),
                    "input_split": chosen_row.get("split", ""),
                    "data_role": "train_only_mumo_two_step_plan_preference",
                    "prompt_messages": prompt_messages,
                    "chosen": chosen_payload,
                    "rejected": protocol.plan_payload(rejected_actions),
                    "chosen_feedback": feedback(chosen_row, action_count=len(chosen_actions)),
                    "rejected_feedback": feedback(rejected_row, action_count=len(rejected_actions)),
                    "hard_negative_category": category,
                    "candidate_budget": len(group),
                }
            )
            outcomes[f"pair:{category}"] += 1
        outcomes["strict_preference_condition"] += 1
    return output, dict(sorted(outcomes.items()))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("Plan preference protocol fixes candidate_budget=20")
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    rows = existing.read_rows(args.official_detail_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        condition_id = str(row.get("condition_id", "") or "")
        if condition_id not in grouped:
            order.append(condition_id)
        grouped[condition_id].append(row)
    if args.max_conditions > 0:
        order = order[: int(args.max_conditions)]
        grouped = {key: grouped[key] for key in order}
    required_split = str(args.require_input_split or "").strip().lower()
    invalid_splits = sorted(
        {
            str(row.get("split", "") or "").strip().lower()
            for group in grouped.values()
            for row in group
            if required_split and str(row.get("split", "") or "").strip().lower() != required_split
        }
    )
    if invalid_splits:
        raise ValueError(f"Plan preference input is not train-only; unexpected splits: {invalid_splits}")
    incomplete = {key: len(group) for key, group in grouped.items() if len(group) != int(args.candidate_budget)}
    if incomplete:
        preview = list(incomplete.items())[:5]
        raise ValueError(f"Plan preference input has incomplete fixed n=20 pools: {preview}")
    selected_rows = [row for condition_id in order for row in grouped[condition_id]]
    checkpoint = args.output_dir / "reconstructed_candidate_plans.jsonl"
    if checkpoint.is_file():
        plans = existing.read_plan_checkpoint(checkpoint)
        desired = {existing.candidate_key(row) for row in selected_rows}
        if not desired.issubset(plans):
            plans = existing.reconstruct_candidate_plans(selected_rows, args.plan_jsonl, checkpoint)
        else:
            plans = {key: plans[key] for key in desired}
    else:
        plans = existing.reconstruct_candidate_plans(selected_rows, args.plan_jsonl, checkpoint)
    if len(plans) != len(selected_rows):
        raise ValueError(f"Plan reconstruction incomplete: {len(plans)}/{len(selected_rows)}")
    pairs, outcomes = preference_pairs(
        selected_rows,
        plans,
        max_negatives=int(args.max_negatives_per_condition),
    )
    if not pairs:
        raise SystemExit("No strict-positive two-step plan preferences were built")
    validation_ids = {
        condition_id
        for condition_id in order
        if stable_validation(condition_id, seed=int(args.seed), fraction=float(args.validation_fraction))
    }
    train = [row for row in pairs if str(row["condition_id"]) not in validation_ids]
    validation = [row for row in pairs if str(row["condition_id"]) in validation_ids]
    if not train or not validation:
        raise ValueError(f"Plan preference split is empty: train={len(train)} validation={len(validation)}")
    train_condition_ids = sorted({str(row["condition_id"]) for row in train})
    validation_condition_ids = sorted({str(row["condition_id"]) for row in validation})
    if set(train_condition_ids) & set(validation_condition_ids):
        raise AssertionError("Train/validation condition overlap in plan preference data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    (args.output_dir / "train_condition_ids.txt").write_text("\n".join(train_condition_ids) + "\n")
    (args.output_dir / "validation_condition_ids.txt").write_text("\n".join(validation_condition_ids) + "\n")
    manifest = {
        "protocol": "unified_constraint_mumo_two_step_plan_preference_v3",
        "data_role": "train_only",
        "official_detail_csv": str(args.official_detail_csv),
        "official_detail_sha256": sha256(args.official_detail_csv),
        "plan_jsonl": str(args.plan_jsonl),
        "plan_jsonl_sha256": sha256(args.plan_jsonl),
        "candidate_budget": int(args.candidate_budget),
        "conditions": len(order),
        "candidate_rows": len(selected_rows),
        "reconstructed_candidate_rows": len(plans),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "train_conditions": len(train_condition_ids),
        "validation_conditions": len(validation_condition_ids),
        "validation_fraction": float(args.validation_fraction),
        "seed": int(args.seed),
        "outcomes": outcomes,
        "pair_categories": dict(sorted(Counter(str(row["hard_negative_category"]) for row in pairs).items())),
        "task_counts": dict(sorted(Counter(str(row["external_task_id"]) for row in pairs).items())),
        "target_information_used_in_prompt": False,
        "planner_policy_score_exposed_to_model": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
