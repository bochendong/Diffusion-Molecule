#!/usr/bin/env python3
"""Build fit-only event-driven controller supervision plus common-task replay."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_composed_retrieved_delta_candidates as composed  # noqa: E402
import build_mumo_closed_loop_dev as closed_loop  # noqa: E402
import build_mumo_delta_shard as delta_shard  # noqa: E402
import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import feedback_repair_agent_protocol as feedback  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--stable-sft-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-states-per-task", type=int, default=384)
    parser.add_argument("--replay-per-origin", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-committed-edits", type=int, default=3)
    parser.add_argument("--max-proposals", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1716)
    return parser.parse_args(argv)


def supported_properties(effects: Mapping[str, float]) -> dict[str, dict[str, object]]:
    return {
        prop: {
            "executable_actions": max(1, int(round(abs(float(value)) * 4))),
            "best_train_effect": round(float(value), 6),
        }
        for prop, value in effects.items()
    }


def supervised_states(
    raw: Mapping[str, object],
    spec: export.ExternalTaskSpec,
    effects: Mapping[str, float],
    source_margins: Mapping[str, float],
    target_margins: Mapping[str, float],
    *,
    max_committed_edits: int,
    max_proposals: int,
) -> list[tuple[str, list[dict[str, str]], dict[str, object]]]:
    margins = dict(source_margins)
    support = supported_properties(effects)
    order = sorted(spec.properties, key=lambda prop: (float(margins[prop]), prop))
    first = order[0]
    first_messages = feedback.prompt_messages(
        raw,
        spec.properties,
        margins,
        support,
        current_smiles=str(raw["source_smiles"]),
        committed_edits=0,
        proposal_count=0,
        max_committed_edits=max_committed_edits,
        max_proposals=max_proposals,
        previous_event=None,
    )
    states = [("initial_repair", first_messages, feedback.repair_action(first))]

    rollback_margins = dict(margins)
    rollback_margins[first] = float(rollback_margins[first]) - 0.05
    alternatives = [prop for prop in order if prop != first]
    if alternatives:
        second = alternatives[0]
        rollback_messages = feedback.prompt_messages(
            raw,
            spec.properties,
            rollback_margins,
            support,
            current_smiles=str(raw["source_smiles"]),
            committed_edits=0,
            proposal_count=1,
            max_committed_edits=max_committed_edits,
            max_proposals=max_proposals,
            previous_event={
                "property": first,
                "outcome": "rollback",
                "reason": "focus_not_improved",
            },
        )
        states.append(("rollback_switch", rollback_messages, feedback.repair_action(second)))

    commit_margins = dict(target_margins)
    remaining = [prop for prop in order if float(commit_margins[prop]) < 0.0]
    commit_messages = feedback.prompt_messages(
        raw,
        spec.properties,
        commit_margins,
        support,
        current_smiles=str(raw["target_smiles"]),
        committed_edits=1,
        proposal_count=1,
        max_committed_edits=max_committed_edits,
        max_proposals=max_proposals,
        previous_event={"property": first, "outcome": "commit", "reason": "monotone_repair"},
    )
    commit_action = (
        feedback.repair_action(remaining[0])
        if remaining
        else feedback.stop_action("all_train_only_margins_satisfied")
    )
    states.append(("post_commit_replan", commit_messages, commit_action))
    return states


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models = closed_loop.load_models(args.evidence_root)
    specs = {item.task_id: item for item in export.TASK_SPECS if item.suite == "mumo"}
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(args.data_dir.glob("train_shard_*.jsonl")):
        for raw in protocol.read_jsonl(path):
            if raw.get("_uca_partition") == "fit":
                by_task[str(raw["_uca_task_id"])].append(raw)

    feedback_rows = []
    state_counts: Counter[str] = Counter()
    for task_id, rows in sorted(by_task.items()):
        rows.sort(key=lambda row: protocol.stable_fraction(
            str(row["_uca_source_group"]), seed=int(args.seed)
        ))
        accepted = 0
        for raw in rows:
            if accepted >= int(args.max_states_per_task):
                break
            spec = specs[task_id]
            normalized = delta_shard.normalized_row(raw, spec)
            effects = composed.observed_property_effects(normalized)
            if set(effects) != set(spec.properties) or min(effects.values()) < 1.0:
                continue
            source_feature = closed_loop.candidate_feature(str(raw["source_smiles"]))
            target_feature = closed_loop.candidate_feature(str(raw["target_smiles"]))
            if source_feature is None or target_feature is None:
                continue
            _scores, margin_rows = closed_loop.score_candidates_batch(
                source_feature,
                [source_feature, target_feature],
                properties=spec.properties,
                models=models,
                source_tanimotos=[1.0, 1.0],
                retrieval_similarities=[1.0, 1.0],
                frequencies=[0, 0],
            )
            states = supervised_states(
                raw,
                spec,
                effects,
                margin_rows[0],
                margin_rows[1],
                max_committed_edits=int(args.max_committed_edits),
                max_proposals=int(args.max_proposals),
            )
            for state_name, messages, action in states:
                if feedback.state_contains_evaluation_target(messages):
                    raise AssertionError("Evaluation target leaked into feedback controller prompt")
                feedback_rows.append(
                    {
                        "example_id": f"feedback:{raw['_uca_pair_digest']}:{state_name}",
                        "origin": "mumo_feedback_repair",
                        "task_mode": "edit",
                        "source_group": str(raw["_uca_source_group"]),
                        "state_type": state_name,
                        "messages": [
                            *messages,
                            {
                                "role": "assistant",
                                "content": json.dumps(action, sort_keys=True, separators=(",", ":")),
                            },
                        ],
                    }
                )
                state_counts[state_name] += 1
            accepted += 1

    validation_groups = {
        str(row["source_group"])
        for row in feedback_rows
        if protocol.stable_fraction(str(row["source_group"]), seed=int(args.seed) + 1)
        < float(args.validation_fraction)
    }
    feedback_train = [row for row in feedback_rows if str(row["source_group"]) not in validation_groups]
    feedback_validation = [row for row in feedback_rows if str(row["source_group"]) in validation_groups]
    if not feedback_train or not feedback_validation:
        raise ValueError("Feedback-controller train/validation split is empty")

    replay_by_origin: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in protocol.read_jsonl(args.stable_sft_dir / "train.jsonl"):
        replay_by_origin[str(row.get("origin", "unknown"))].append(row)
    replay = []
    for origin, rows in sorted(replay_by_origin.items()):
        rows.sort(key=lambda row: protocol.stable_fraction(
            str(row.get("example_id", "")), seed=int(args.seed)
        ))
        replay.extend(rows[: int(args.replay_per_origin)])
    train = [*feedback_train, *replay]
    random.Random(int(args.seed)).shuffle(train)
    feedback_validation.sort(key=lambda row: str(row["example_id"]))
    protocol.write_jsonl(args.output_dir / "train.jsonl", train)
    protocol.write_jsonl(args.output_dir / "validation.jsonl", feedback_validation)
    overlap = len(
        {str(row["source_group"]) for row in feedback_train}
        & {str(row["source_group"]) for row in feedback_validation}
    )
    manifest = {
        "protocol": "feedback_repair_controller_sft_v1",
        "data_role": "fit_only_feedback_states_plus_balanced_common_task_replay",
        "prompt_target_access": False,
        "fit_pair_target_used_as_post_commit_training_state": True,
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "feedback_train_rows": len(feedback_train),
        "feedback_validation_rows": len(feedback_validation),
        "feedback_state_counts": dict(sorted(state_counts.items())),
        "replay_rows": len(replay),
        "replay_origin_counts": dict(sorted(Counter(
            str(row.get("origin", "")) for row in replay
        ).items())),
        "source_group_overlap": overlap,
        "max_committed_edits": int(args.max_committed_edits),
        "max_proposals": int(args.max_proposals),
        "seed": int(args.seed),
    }
    protocol.write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
