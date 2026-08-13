#!/usr/bin/env python3
"""Generate exactly 20 direct source-preserving repair trajectories per condition.

Each trajectory samples and executes one train-derived symbolic delta at a time,
then observes train-only verifier margins before the next edit.  No larger
molecular candidate pool is built, scored, sorted, or truncated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_mumo_closed_loop_dev as closed_loop  # noqa: E402
import build_retrieved_delta_edit_candidates as delta  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


@dataclass
class TrajectoryState:
    attempt_index: int
    rng: random.Random
    current_smiles: str
    current_feature: np.ndarray
    margins: dict[str, float]
    source_tanimoto: float = 1.0
    active: bool = True
    stop_reason: str = "max_steps"
    used_actions: set[tuple[str, str, str]] = field(default_factory=set)
    trace: list[dict[str, object]] = field(default_factory=list)
    action_rejections: int = 0


def next_focus_property(
    order: Sequence[str], margins: Mapping[str, float], *, has_executed_edit: bool
) -> str | None:
    focus = focus_property(order, margins)
    if focus is not None:
        return focus
    return None if has_executed_edit else str(order[0])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--dev-sources-jsonl", required=True, type=Path)
    parser.add_argument("--plans-jsonl", required=True, type=Path)
    parser.add_argument("--plans-manifest", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--attempts-per-condition", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-symbolic-actions", type=int, default=256)
    parser.add_argument("--action-retry-limit", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--min-retrieval-similarity", type=float, default=0.15)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1715)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_plans(path: Path) -> dict[str, dict[str, object]]:
    output = {}
    for row in protocol.read_jsonl(path):
        output[str(row["condition_id"])] = row
    return output


def focus_property(order: Sequence[str], margins: Mapping[str, float]) -> str | None:
    for prop in order:
        if float(margins.get(prop, -math.inf)) < 0.0:
            return str(prop)
    return None


def symbolic_actions(
    current_smiles: str,
    transforms: Sequence[Mapping[str, object]],
    *,
    focus: str,
    order: Sequence[str],
    margins: Mapping[str, float],
    used: set[tuple[str, str, str]],
    min_similarity: float,
    min_core_heavy_atoms: int,
    max_variable_heavy_atoms: int,
    limit: int,
) -> list[tuple[float, str, Mapping[str, object], float]]:
    actions = []
    for split in delta.fragment_splits(
        current_smiles, int(min_core_heavy_atoms), int(max_variable_heavy_atoms)
    ):
        for transform in transforms:
            identity = (
                str(transform["task_key"]),
                str(transform["source_variable"]),
                str(transform["target_variable"]),
            )
            if identity in used:
                continue
            similarity = delta.variable_similarity(split.variable, str(transform["source_variable"]))
            if similarity < float(min_similarity):
                continue
            effects = dict(transform.get("effects", {}))
            unmet = [prop for prop in order if float(margins.get(prop, -1.0)) < 0.0]
            satisfied = [prop for prop in order if prop not in unmet]
            utility = (
                2.5 * float(effects.get(focus, 0.0))
                + sum(max(0.0, float(effects.get(prop, 0.0))) for prop in unmet)
                - 0.35 * sum(max(0.0, -float(effects.get(prop, 0.0))) for prop in satisfied)
                + 0.75 * float(similarity)
                + 0.05 * math.log1p(max(int(transform.get("frequency", 0)), 0))
            )
            actions.append((utility, split.core, transform, float(similarity)))
    actions.sort(
        key=lambda item: (
            item[0],
            item[3],
            int(item[2].get("frequency", 0)),
            str(item[2].get("target_variable", "")),
        ),
        reverse=True,
    )
    return actions[: max(1, int(limit))]


def sample_without_replacement(
    actions: Sequence[tuple[float, str, Mapping[str, object], float]],
    *,
    rng: random.Random,
    temperature: float,
) -> list[int]:
    remaining = list(range(len(actions)))
    output = []
    while remaining:
        values = [float(actions[index][0]) / max(float(temperature), 1e-6) for index in remaining]
        maximum = max(values)
        weights = [math.exp(max(-30.0, min(30.0, value - maximum))) for value in values]
        selected = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        output.append(remaining.pop(selected))
    return output


def choose_valid_edit(
    state: TrajectoryState,
    *,
    original_source: str,
    actions: Sequence[tuple[float, str, Mapping[str, object], float]],
    min_source_tanimoto: float,
    retry_limit: int,
    temperature: float,
) -> tuple[str, np.ndarray, float, Mapping[str, object], str, float] | None:
    for action_index in sample_without_replacement(
        actions, rng=state.rng, temperature=float(temperature)
    )[: int(retry_limit)]:
        _utility, core, transform, retrieval_similarity = actions[action_index]
        generated = delta.canonical_smiles(
            delta.join_fragments(core, str(transform["target_variable"]))
        )
        if not generated or generated == state.current_smiles:
            state.action_rejections += 1
            continue
        source_tanimoto = float(delta.graph.revise.morgan_tanimoto(original_source, generated))
        if not math.isfinite(source_tanimoto) or source_tanimoto < float(min_source_tanimoto):
            state.action_rejections += 1
            continue
        generated_feature = closed_loop.candidate_feature(generated)
        if generated_feature is None:
            state.action_rejections += 1
            continue
        return generated, generated_feature, source_tanimoto, transform, core, retrieval_similarity
    return None


def stable_seed(seed: int, condition_id: str, attempt_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{condition_id}:{attempt_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.attempts_per_condition) != 20:
        raise ValueError("Direct repair protocol fixes exactly 20 generation attempts")
    plans_manifest = json.loads(args.plans_manifest.read_text(encoding="utf-8"))
    if (
        plans_manifest.get("evaluation_target_access") is not False
        or plans_manifest.get("evaluation_oracle_access") is not False
        or plans_manifest.get("output_selection") != "none"
    ):
        raise ValueError("Repair-plan contract violation")
    evidence = json.loads((args.evidence_root / "merged" / "summary.json").read_text(encoding="utf-8"))
    if evidence.get("passed") is not True:
        raise ValueError("Train-only verifier evidence gate did not pass")
    models = closed_loop.load_models(args.evidence_root)
    transforms = closed_loop.load_transforms(args.evidence_root / "merged" / "transforms.jsonl")
    plans = load_plans(args.plans_jsonl)
    raw_rows = [
        row
        for row in protocol.read_jsonl(args.dev_sources_jsonl)
        if protocol.stable_shard(
            str(row["_uca_source_group"]),
            seed=int(args.seed),
            shard_count=int(args.shard_count),
        ) == int(args.shard_index)
    ]
    raw_rows.sort(key=lambda row: (str(row["_uca_task_id"]), str(row["_uca_source_group"])))
    output = []
    step_counts = []
    unique_counts = []
    stop_counts: Counter[str] = Counter()
    total_rejections = 0
    for row_index, raw in enumerate(raw_rows):
        row = closed_loop.condition_row(raw, row_index)
        condition_id = str(row["condition_id"])
        plan = plans.get(condition_id)
        if plan is None:
            raise ValueError(f"Missing repair plan: {condition_id}")
        order = tuple(str(prop) for prop in plan["property_order"])
        properties = tuple(str(row["external_task_properties"]).split(","))
        if set(order) != set(properties) or int(plan["max_steps"]) != int(args.max_steps):
            raise ValueError(f"Repair plan contract mismatch: {condition_id}")
        source = str(row["source_smiles"])
        source_feature = closed_loop.candidate_feature(source)
        if source_feature is None:
            raise ValueError(f"Invalid source: {condition_id}")
        _scores, initial_margin_rows = closed_loop.score_candidates_batch(
            source_feature,
            [source_feature],
            properties=properties,
            models=models,
            source_tanimotos=[1.0],
            retrieval_similarities=[1.0],
            frequencies=[0],
        )
        states = [
            TrajectoryState(
                attempt_index=attempt_index,
                rng=random.Random(stable_seed(int(args.seed), condition_id, attempt_index)),
                current_smiles=source,
                current_feature=source_feature.copy(),
                margins=dict(initial_margin_rows[0]),
            )
            for attempt_index in range(int(args.attempts_per_condition))
        ]
        task_transforms = transforms.get(str(row["external_task_key"]), [])
        for step_index in range(int(args.max_steps)):
            proposed = []
            proposed_states = []
            action_cache = {}
            for state in states:
                if not state.active:
                    continue
                focus = next_focus_property(
                    order, state.margins, has_executed_edit=bool(state.trace)
                )
                if focus is None:
                    state.active = False
                    state.stop_reason = "train_verifier_constraints_satisfied"
                    continue
                action_key = (
                    state.current_smiles,
                    focus,
                    tuple(sorted(state.used_actions)),
                    tuple(
                        sorted(
                            (prop, round(float(value), 6))
                            for prop, value in state.margins.items()
                        )
                    ),
                )
                actions = action_cache.get(action_key)
                if actions is None:
                    actions = symbolic_actions(
                        state.current_smiles,
                        task_transforms,
                        focus=focus,
                        order=order,
                        margins=state.margins,
                        used=state.used_actions,
                        min_similarity=float(args.min_retrieval_similarity),
                        min_core_heavy_atoms=int(args.min_core_heavy_atoms),
                        max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
                        limit=int(args.max_symbolic_actions),
                    )
                    action_cache[action_key] = actions
                edit = choose_valid_edit(
                    state,
                    original_source=source,
                    actions=actions,
                    min_source_tanimoto=float(args.min_source_tanimoto),
                    retry_limit=int(args.action_retry_limit),
                    temperature=float(args.temperature),
                )
                if edit is None:
                    state.active = False
                    state.stop_reason = "no_safe_executable_delta"
                    continue
                proposed.append((focus, edit))
                proposed_states.append(state)
            if not proposed:
                break
            _scores, margin_rows = closed_loop.score_candidates_batch(
                source_feature,
                [edit[1][1] for edit in proposed],
                properties=properties,
                models=models,
                source_tanimotos=[edit[1][2] for edit in proposed],
                retrieval_similarities=[edit[1][5] for edit in proposed],
                frequencies=[int(edit[1][3].get("frequency", 0)) for edit in proposed],
            )
            for state, (focus, edit), margins in zip(proposed_states, proposed, margin_rows):
                generated, feature, tanimoto, transform, _core, retrieval_similarity = edit
                before = dict(state.margins)
                identity = (
                    str(transform["task_key"]),
                    str(transform["source_variable"]),
                    str(transform["target_variable"]),
                )
                state.current_smiles = generated
                state.current_feature = feature
                state.source_tanimoto = float(tanimoto)
                state.margins = dict(margins)
                state.used_actions.add(identity)
                state.trace.append(
                    {
                        "step": step_index + 1,
                        "focus_property": focus,
                        "source_variable": transform["source_variable"],
                        "target_variable": transform["target_variable"],
                        "train_effects": transform.get("effects", {}),
                        "retrieval_similarity": round(float(retrieval_similarity), 6),
                        "source_tanimoto": round(float(tanimoto), 6),
                        "margins_before": before,
                        "margins_after": state.margins,
                    }
                )
        unique_counts.append(len({state.current_smiles for state in states}))
        for attempt_number, state in enumerate(states, start=1):
            if state.active:
                state.stop_reason = "max_steps"
            stop_counts[state.stop_reason] += 1
            step_counts.append(len(state.trace))
            total_rejections += state.action_rejections
            output.append(
                {
                    **row,
                    "generated_smiles": state.current_smiles,
                    "method": "common_llm_direct_constraint_repair_v12",
                    "generation_attempt_index": attempt_number,
                    "candidate_valid": True,
                    "candidate_source_similarity_pass": state.source_tanimoto >= float(args.min_source_tanimoto),
                    "candidate_attempt_is_repeat": state.current_smiles
                    in {item.current_smiles for item in states[: attempt_number - 1]},
                    "first_seen_attempt_index": next(
                        index + 1
                        for index, item in enumerate(states)
                        if item.current_smiles == state.current_smiles
                    ),
                    "candidate_is_noop": len(state.trace) == 0,
                    "source_tanimoto": state.source_tanimoto,
                    "trajectory_id": f"{condition_id}:trajectory:{attempt_number:02d}",
                    "trajectory_attempt_index": state.attempt_index,
                    "trajectory_step_count": len(state.trace),
                    "trajectory_stop_reason": state.stop_reason,
                    "trajectory_property_order_json": json.dumps(order),
                    "trajectory_final_margins_json": json.dumps(state.margins, sort_keys=True),
                    "trajectory_trace_json": json.dumps(state.trace, sort_keys=True),
                    "trajectory_action_rejections": state.action_rejections,
                    "output_selection": "none",
                }
            )
        if (row_index + 1) % 10 == 0 or row_index + 1 == len(raw_rows):
            print(f"[direct-repair] {row_index + 1}/{len(raw_rows)}", flush=True)

    write_csv(args.output_csv, output)
    manifest = {
        "protocol": "common_llm_direct_constraint_repair_trajectories_v1",
        "data_role": "fit_tools_to_source_only_disjoint_dev",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "output_selection": "none",
        "output_rows_have_rank": False,
        "output_rows_have_selected_flag": False,
        "internal_molecular_candidate_pool": False,
        "candidate_budget": 20,
        "attempts_per_condition": 20,
        "conditions": len(raw_rows),
        "candidate_rows": len(output),
        "unique_candidates_total": sum(unique_counts),
        "unique_valid_candidates_total": sum(unique_counts),
        "mean_unique_candidates_per_condition": sum(unique_counts) / max(len(unique_counts), 1),
        "min_unique_candidates_per_condition": min(unique_counts, default=0),
        "repeated_attempt_rows": len(output) - sum(unique_counts),
        "noop_attempt_rows": sum(str(row["candidate_is_noop"]).lower() == "true" for row in output),
        "max_steps": int(args.max_steps),
        "mean_steps_per_attempt": sum(step_counts) / max(len(step_counts), 1),
        "stop_reason_counts": dict(sorted(stop_counts.items())),
        "symbolic_action_rejections": total_rejections,
        "train_verifier_observation_after_each_edit": True,
        "official_oracle_used_only_after_freeze": True,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "seed": int(args.seed),
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
