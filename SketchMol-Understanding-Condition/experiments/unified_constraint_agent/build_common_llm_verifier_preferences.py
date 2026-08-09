#!/usr/bin/env python3
"""Build strict-positive action preferences from train-only verifier scores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
for import_dir in (SCRIPT_DIR, POLICY_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--enumeration-attempt-budget", type=int, default=64)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--negatives-per-example", type=int, default=2)
    parser.add_argument("--table1-similarity-threshold", type=float, default=0.65)
    parser.add_argument("--mumo-similarity-threshold", type=float, default=0.40)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def strict_positive_key(record: Mapping[str, object]) -> tuple[float, ...]:
    return (
        finite(record.get("instruction_success_fraction"), -1.0),
        -finite(record.get("instruction_distance"), 1e6),
        finite(record.get("source_similarity"), -1.0),
        finite(record.get("instruction_mean_margin"), -1e6),
        finite(record.get("target_similarity"), -1.0),
    )


def hard_negative_key(record: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(bool(record.get("source_similarity_success"))),
        finite(record.get("instruction_success_fraction"), -1.0),
        -finite(record.get("instruction_distance"), 1e6),
        finite(record.get("source_similarity"), -1.0),
        finite(record.get("instruction_mean_margin"), -1e6),
    )


def verifier_feedback(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "strict_success": bool(record.get("strict_success")),
        "instruction_success_fraction": record.get("instruction_success_fraction"),
        "instruction_distance": record.get("instruction_distance"),
        "instruction_mean_margin": record.get("instruction_mean_margin"),
        "source_similarity": record.get("source_similarity"),
        "source_similarity_success": bool(record.get("source_similarity_success")),
    }


def select_verifier_preference(
    records: Sequence[Mapping[str, object]],
    *,
    negative_count: int,
) -> tuple[Mapping[str, object] | None, list[Mapping[str, object]]]:
    positives = [record for record in records if bool(record.get("strict_success"))]
    negatives = [record for record in records if not bool(record.get("strict_success"))]
    if not positives or not negatives:
        return None, []
    chosen = max(positives, key=strict_positive_key)
    rejected = sorted(negatives, key=hard_negative_key, reverse=True)[: max(1, int(negative_count))]
    return chosen, rejected


def preference_records(
    records: Sequence[Mapping[str, object]],
    *,
    candidate_budget: int,
    attempt_budget: int,
    site_limit: int,
    negatives_per_example: int,
    table1_similarity_threshold: float,
    mumo_similarity_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    policy = constrained.policy_module()
    by_id = {str(record.get("example_id", "")): record for record in records}
    output = []
    outcomes = Counter()
    for index, record in enumerate(by_id.values(), start=1):
        origin = str(record.get("origin", ""))
        if origin not in {"table1", "mumo"}:
            outcomes["non_edit"] += 1
            continue
        ir, _expected = constrained.constraint_payload(record)
        planner_row = constrained.planner_row_from_ir(ir)
        candidates = policy.enumerate_action_candidates(
            planner_row,
            site_limit=int(site_limit),
            max_actions_per_row=max(int(candidate_budget), int(attempt_budget)),
        )[: int(candidate_budget)]
        threshold = table1_similarity_threshold if origin == "table1" else mumo_similarity_threshold
        scored = [
            policy.action_oracle_record(
                planner_row,
                candidate,
                source_similarity_threshold=float(threshold),
            )
            for candidate in candidates
        ]
        chosen, rejected = select_verifier_preference(scored, negative_count=negatives_per_example)
        if chosen is None:
            outcomes["no_strict_positive"] += 1
            continue
        outcomes["strict_preference"] += 1
        chosen_action = chosen.get("action")
        if chosen_action is None:
            outcomes["missing_chosen_action"] += 1
            continue
        prompt_messages = record.get("messages", [])[:-1]
        chosen_payload = {"action_type": "graph_edit_dsl", "value": asdict(chosen_action)}
        for negative_index, negative in enumerate(rejected):
            rejected_action = negative.get("action")
            if rejected_action is None:
                continue
            output.append(
                {
                    "pair_id": f"{record.get('example_id', '')}:strict_negative_{negative_index}",
                    "example_id": record.get("example_id", ""),
                    "origin": origin,
                    "data_role": "train_only_verifier_preference",
                    "prompt_messages": prompt_messages,
                    "chosen": chosen_payload,
                    "rejected": {"action_type": "graph_edit_dsl", "value": asdict(rejected_action)},
                    "chosen_feedback": verifier_feedback(chosen),
                    "rejected_feedback": verifier_feedback(negative),
                    "candidate_count": len(candidates),
                    "candidate_budget": candidate_budget,
                }
            )
        if index % 100 == 0:
            print(f"[verifier-preference] {index}/{len(by_id)}", flush=True)
    return output, dict(sorted(outcomes.items()))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("Verifier preference v2 requires candidate_budget=20")
    common = {
        "candidate_budget": args.candidate_budget,
        "attempt_budget": args.enumeration_attempt_budget,
        "site_limit": args.site_limit,
        "negatives_per_example": args.negatives_per_example,
        "table1_similarity_threshold": args.table1_similarity_threshold,
        "mumo_similarity_threshold": args.mumo_similarity_threshold,
    }
    train, train_outcomes = preference_records(read_jsonl(args.train_jsonl), **common)
    validation, validation_outcomes = preference_records(read_jsonl(args.validation_jsonl), **common)
    if not train:
        raise SystemExit("No strict-positive train-only verifier preferences were built")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "protocol": "unified_constraint_verifier_preference_v2",
        "data_role": "train_only",
        "candidate_budget": args.candidate_budget,
        "enumeration_attempt_budget": args.enumeration_attempt_budget,
        "negatives_per_example": args.negatives_per_example,
        "table1_similarity_threshold": args.table1_similarity_threshold,
        "mumo_similarity_threshold": args.mumo_similarity_threshold,
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "train_origin_counts": dict(sorted(Counter(row["origin"] for row in train).items())),
        "validation_origin_counts": dict(sorted(Counter(row["origin"] for row in validation).items())),
        "train_outcomes": train_outcomes,
        "validation_outcomes": validation_outcomes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
