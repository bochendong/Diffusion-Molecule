#!/usr/bin/env python3
"""Build train-only chosen/rejected GraphEditDSL preferences."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--enumeration-attempt-budget", type=int, default=64)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--negatives-per-example", type=int, default=2)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def structural_similarity(expected: Mapping[str, object], candidate: Mapping[str, object]) -> tuple[float, ...]:
    expected_bond = expected.get("bond")
    candidate_bond = candidate.get("bond")
    if isinstance(expected_bond, list):
        expected_bond = tuple(expected_bond)
    if isinstance(candidate_bond, list):
        candidate_bond = tuple(candidate_bond)
    return (
        float(str(candidate.get("op", "")) == str(expected.get("op", ""))),
        float(str(candidate.get("prop", "")) == str(expected.get("prop", ""))),
        float(str(candidate.get("direction", "")) == str(expected.get("direction", ""))),
        float(candidate.get("site") == expected.get("site")),
        float(candidate_bond == expected_bond),
        float(candidate.get("atom") == expected.get("atom")),
        float(candidate.get("fragment") == expected.get("fragment")),
        float(candidate.get("bond_order") == expected.get("bond_order")),
        float(candidate.get("policy_score", 0.0) or 0.0),
    )


def select_hard_negatives(
    expected: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    *,
    count: int,
) -> list[dict[str, object]]:
    expected_key = constrained.structural_action_key(expected)
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for candidate in candidates:
        key = constrained.structural_action_key(candidate)
        if key is None or key == expected_key:
            continue
        unique.setdefault(key, dict(candidate))
    ordered = sorted(
        unique.values(),
        key=lambda candidate: structural_similarity(expected, candidate),
        reverse=True,
    )
    return ordered[: max(1, int(count))]


def preference_records(
    records: Sequence[Mapping[str, object]],
    *,
    attempt_budget: int,
    site_limit: int,
    negatives_per_example: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    policy = constrained.policy_module()
    by_id = {str(record.get("example_id", "")): record for record in records}
    output = []
    outcomes = Counter()
    for record in by_id.values():
        if record.get("origin") not in {"table1", "mumo"}:
            outcomes["non_edit"] += 1
            continue
        ir, expected_payload = constrained.constraint_payload(record)
        expected = expected_payload.get("value")
        if not isinstance(expected, Mapping):
            outcomes["missing_expected"] += 1
            continue
        planner_row = constrained.planner_row_from_ir(ir)
        candidates = policy.enumerate_action_candidates(
            planner_row,
            site_limit=int(site_limit),
            max_actions_per_row=int(attempt_budget),
        )
        candidate_actions = [asdict(action) for action, _smiles, _program in candidates]
        negatives = select_hard_negatives(
            expected,
            candidate_actions,
            count=negatives_per_example,
        )
        if not negatives:
            outcomes["no_negative"] += 1
            continue
        expected_in_pool = any(
            constrained.structural_action_key(candidate) == constrained.structural_action_key(expected)
            for candidate in candidate_actions
        )
        outcomes["expected_in_pool" if expected_in_pool else "expected_outside_pool"] += 1
        prompt_messages = record.get("messages", [])[:-1]
        chosen = {"action_type": "graph_edit_dsl", "value": dict(expected)}
        for negative_index, negative in enumerate(negatives):
            output.append(
                {
                    "pair_id": f"{record.get('example_id', '')}:negative_{negative_index}",
                    "example_id": record.get("example_id", ""),
                    "origin": record.get("origin", ""),
                    "data_role": "train_only_preference",
                    "prompt_messages": prompt_messages,
                    "chosen": chosen,
                    "rejected": {"action_type": "graph_edit_dsl", "value": negative},
                    "expected_in_enumerated_pool": expected_in_pool,
                    "enumerated_valid_actions": len(candidates),
                }
            )
    return output, dict(sorted(outcomes.items()))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train, train_outcomes = preference_records(
        read_jsonl(args.train_jsonl),
        attempt_budget=args.enumeration_attempt_budget,
        site_limit=args.site_limit,
        negatives_per_example=args.negatives_per_example,
    )
    validation, validation_outcomes = preference_records(
        read_jsonl(args.validation_jsonl),
        attempt_budget=args.enumeration_attempt_budget,
        site_limit=args.site_limit,
        negatives_per_example=args.negatives_per_example,
    )
    if not train:
        raise SystemExit("No train-only action preferences were built")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "protocol": "unified_constraint_action_preference_v1",
        "data_role": "train_only",
        "enumeration_attempt_budget": args.enumeration_attempt_budget,
        "negatives_per_example": args.negatives_per_example,
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
