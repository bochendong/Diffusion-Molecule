#!/usr/bin/env python3
"""Build train-only RetrievedDeltaEdit preferences from paired molecular edits.

Targets are used only to identify the positive executable delta in training.
They are never serialized into the common-LLM prompt or action payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_retrieved_delta_edit_candidates as delta  # noqa: E402
import audit_hierarchical_action_support as support  # noqa: E402
import retrieved_delta_plan_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-negatives-per-condition", type=int, default=3)
    parser.add_argument("--max-conditions-per-task", type=int, default=0)
    parser.add_argument("--max-transforms-per-query", type=int, default=96)
    parser.add_argument("--min-retrieval-similarity", type=float, default=0.15)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1709)
    return parser.parse_args(argv)


def source_group(row: Mapping[str, object]) -> str:
    return json.dumps(support.source_key(row), sort_keys=True, separators=(",", ":"))


def stable_validation(group: str, *, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64) < float(fraction)


def candidate_record(candidate: delta.Candidate) -> dict[str, object]:
    return {
        "delta_query_variable": candidate.query_variable,
        "delta_source_variable": candidate.source_variable,
        "delta_target_variable": candidate.target_variable,
    }


def safe_metadata(candidate: delta.Candidate) -> dict[str, object]:
    return {
        "source": candidate.source,
        "source_tanimoto": candidate.source_tanimoto,
        "admet_prior_score": candidate.admet_prior_score,
        "retrieval_similarity": candidate.retrieval_similarity,
        "transform_frequency": candidate.transform_frequency,
        "exact_variable_match": candidate.exact_variable_match,
    }


def preference_key(candidate: delta.Candidate) -> tuple[float, ...]:
    return (
        float(candidate.exact_variable_match),
        float(candidate.retrieval_similarity),
        float(candidate.source_tanimoto),
        float(candidate.admet_prior_score),
        float(candidate.transform_frequency),
    )


def hard_negatives(
    candidates: Sequence[delta.Candidate],
    *,
    target: str,
    limit: int,
) -> list[delta.Candidate]:
    eligible = [candidate for candidate in candidates if candidate.smiles != target]
    eligible.sort(key=preference_key, reverse=True)
    output: list[delta.Candidate] = []
    seen_targets: set[str] = set()
    for candidate in eligible:
        if candidate.target_variable in seen_targets:
            continue
        seen_targets.add(candidate.target_variable)
        output.append(candidate)
        if len(output) >= max(1, int(limit)):
            break
    return output


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 < float(args.validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    rows = delta.read_rows(args.train_csv)
    transform_index, transform_manifest = delta.build_transform_index(
        rows,
        min_core_heavy_atoms=int(args.min_core_heavy_atoms),
        max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
    )
    selected_rows = list(rows)
    if int(args.max_conditions_per_task) > 0:
        by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_task[delta.task_key(row)].append(row)
        selected_rows = []
        for offset, task in enumerate(sorted(by_task)):
            task_rows = list(by_task[task])
            random.Random(int(args.seed) + offset).shuffle(task_rows)
            selected_rows.extend(task_rows[: int(args.max_conditions_per_task)])
    pairs: list[dict[str, object]] = []
    outcomes = Counter()
    groups_by_condition: dict[str, str] = {}
    for row in selected_rows:
        condition_id = delta.row_key(row)
        target = delta.canonical_smiles(str(row.get("target_smiles", "") or ""))
        if not target:
            outcomes["missing_training_target"] += 1
            continue
        candidates, _match_summary = delta.retrieved_candidates(
            row,
            transform_index.get(delta.task_key(row), []),
            min_retrieval_similarity=float(args.min_retrieval_similarity),
            max_transforms_per_query=int(args.max_transforms_per_query),
            min_core_heavy_atoms=int(args.min_core_heavy_atoms),
            max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
        )
        positives = [candidate for candidate in candidates if candidate.smiles == target]
        if not positives:
            outcomes["no_executable_target_delta"] += 1
            continue
        chosen = max(positives, key=preference_key)
        negatives = hard_negatives(
            candidates,
            target=target,
            limit=int(args.max_negatives_per_condition),
        )
        if not negatives:
            outcomes["no_hard_negative"] += 1
            continue
        messages = protocol.prompt_messages(row)
        prompt_payload = json.loads(messages[-1]["content"])
        if "target_smiles" in prompt_payload or "target_smiles" in prompt_payload.get("constraint_ir", {}):
            raise ValueError(f"Training target field leaked into planner prompt for {condition_id}")
        group = source_group(row)
        groups_by_condition[condition_id] = group
        chosen_payload = protocol.action_payload(candidate_record(chosen))
        for negative_index, rejected in enumerate(negatives):
            pairs.append(
                {
                    "pair_id": f"{condition_id}:delta:{negative_index}",
                    "condition_id": condition_id,
                    "source_group": group,
                    "origin": "mumo_retrieved_delta",
                    "data_role": "train_only_paired_delta_supervision",
                    "prompt_target_access": False,
                    "training_target_role": "positive_label_only",
                    "prompt_messages": messages,
                    "chosen": chosen_payload,
                    "rejected": protocol.action_payload(candidate_record(rejected)),
                    "chosen_metadata": safe_metadata(chosen),
                    "rejected_metadata": safe_metadata(rejected),
                }
            )
        outcomes["preference_conditions"] += 1
        outcomes["preference_pairs"] += len(negatives)
    if not pairs:
        raise SystemExit("No RetrievedDeltaEdit preference pairs were built")
    validation_groups = {
        group
        for group in set(groups_by_condition.values())
        if stable_validation(group, seed=int(args.seed), fraction=float(args.validation_fraction))
    }
    train = [row for row in pairs if str(row["source_group"]) not in validation_groups]
    validation = [row for row in pairs if str(row["source_group"]) in validation_groups]
    if not train or not validation:
        raise ValueError(f"Preference split is empty: train={len(train)} validation={len(validation)}")
    if {str(row["source_group"]) for row in train} & {str(row["source_group"]) for row in validation}:
        raise AssertionError("Source-group overlap in RetrievedDeltaEdit preference split")
    random.Random(int(args.seed)).shuffle(train)
    validation.sort(key=lambda row: str(row["pair_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "protocol": "common_llm_retrieved_delta_preference_v1",
        "data_role": "train_only_paired_delta_supervision",
        "prompt_target_access": False,
        "training_target_role": "positive_label_only",
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "input_rows": len(rows),
        "selected_training_conditions": len(selected_rows),
        "max_conditions_per_task": int(args.max_conditions_per_task),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "train_source_groups": len({str(row["source_group"]) for row in train}),
        "validation_source_groups": len({str(row["source_group"]) for row in validation}),
        "source_group_overlap": 0,
        "outcomes": dict(sorted(outcomes.items())),
        "transform_index": transform_manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
