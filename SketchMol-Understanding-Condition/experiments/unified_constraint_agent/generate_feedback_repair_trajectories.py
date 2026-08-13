#!/usr/bin/env python3
"""Generate exact-20 event-driven repair trajectories on a frozen dev signal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
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
import evaluate_common_llm_constrained_actions as constrained  # noqa: E402
import feedback_repair_agent_protocol as feedback  # noqa: E402
import generate_direct_repair_trajectories as direct  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--dev-sources-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--controller-mode", choices=("llm", "deterministic"), required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--conditions-per-split", type=int, default=100)
    parser.add_argument("--attempts-per-condition", type=int, default=20)
    parser.add_argument("--max-committed-edits", type=int, default=3)
    parser.add_argument("--max-proposals", type=int, default=6)
    parser.add_argument("--max-symbolic-actions", type=int, default=256)
    parser.add_argument("--action-retry-limit", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--min-retrieval-similarity", type=float, default=0.15)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1716)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_signal_rows(
    rows: Sequence[Mapping[str, object]], *, per_split: int, seed: int
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["_uca_task_id"])].append(dict(row))
    if set(grouped) != set(protocol.TASK_IDS):
        raise ValueError("Frozen dev signal does not contain all MuMO tasks")
    selected = []
    for split_tasks in (protocol.IND_TASK_IDS, protocol.OOD_TASK_IDS):
        task_ids = [task_id for task_id in protocol.TASK_IDS if task_id in split_tasks]
        total = sum(len(grouped[task_id]) for task_id in task_ids)
        if total < int(per_split):
            raise ValueError("Insufficient frozen dev rows for balanced split signal")
        exact = {
            task_id: int(per_split) * len(grouped[task_id]) / total
            for task_id in task_ids
        }
        quotas = {task_id: int(math.floor(value)) for task_id, value in exact.items()}
        remainder = int(per_split) - sum(quotas.values())
        for task_id in sorted(
            task_ids,
            key=lambda value: (exact[value] - quotas[value], value),
            reverse=True,
        )[:remainder]:
            quotas[task_id] += 1
        for task_id in task_ids:
            candidates = sorted(
                grouped[task_id],
                key=lambda row: protocol.stable_fraction(
                    str(row["_uca_source_group"]), seed=int(seed)
                ),
            )
            selected.extend(candidates[: quotas[task_id]])
    selected.sort(key=lambda row: (str(row["_uca_task_id"]), str(row["_uca_source_group"])))
    return selected


def action_support_summary(
    actions: Sequence[tuple[float, str, Mapping[str, object], float]], prop: str
) -> dict[str, object]:
    if not actions:
        return {"executable_actions": 0}
    utility, _core, transform, retrieval_similarity = actions[0]
    return {
        "executable_actions": len(actions),
        "best_train_effect": round(float(dict(transform.get("effects", {})).get(prop, 0.0)), 6),
        "best_retrieval_similarity": round(float(retrieval_similarity), 6),
        "best_symbolic_utility": round(float(utility), 6),
    }


def deterministic_action(
    properties: Sequence[str],
    margins: Mapping[str, float],
    support: Mapping[str, Mapping[str, object]],
    *,
    committed_edits: int,
) -> tuple[str, str | None]:
    available = [
        prop
        for prop in properties
        if int(dict(support.get(prop, {})).get("executable_actions", 0)) > 0
    ]
    unmet = [prop for prop in available if float(margins[prop]) < 0.0]
    if not unmet and int(committed_edits) > 0:
        return "stop", None
    candidates = unmet or available
    return ("repair", min(candidates, key=lambda prop: (float(margins[prop]), prop))) if candidates else ("stop", None)


def feedback_transaction_decision(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    focus: str,
) -> tuple[bool, str, dict[str, object]]:
    deltas = {prop: float(after[prop]) - float(before[prop]) for prop in before}
    violation_before = sum(max(0.0, -float(value)) for value in before.values())
    violation_after = sum(max(0.0, -float(value)) for value in after.values())
    lost = sorted(prop for prop in before if float(before[prop]) >= 0.0 > float(after[prop]))
    exact_qed_lost = "qed" in lost
    focus_progress = deltas[focus] > 0.0
    aggregate_progress = violation_after < violation_before
    accepted = bool(not exact_qed_lost and (focus_progress or aggregate_progress))
    reason = (
        "rollback_exact_qed_safety"
        if exact_qed_lost
        else "commit_progress_observed"
        if accepted
        else "rollback_no_progress"
    )
    return accepted, reason, {
        "focus_margin_delta": deltas[focus],
        "total_violation_before": violation_before,
        "total_violation_after": violation_after,
        "lost_satisfied_constraints": lost,
        "uncertain_nonexact_regressions_are_feedback_only": True,
    }


def load_controller(args: argparse.Namespace):
    if args.controller_mode != "llm":
        return None, None
    if args.adapter_dir is None or not args.adapter_dir.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError("Feedback LLM adapter is missing")
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("Feedback LLM controller requires CUDA")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # H100 inference does not need the float32 training footprint.  Bfloat16
    # preserves the adapter's scoring semantics while leaving enough room for
    # the causal-LM logits on a 20 GB MIG slice.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = peft.PeftModel.from_pretrained(model, args.adapter_dir).cuda().eval()
    model.config.use_cache = True
    return model, tokenizer


def choose_llm_actions(
    model: object,
    tokenizer: object,
    decisions: Sequence[dict[str, object]],
    *,
    max_length: int,
    batch_size: int,
) -> list[tuple[str, str | None]]:
    encoded = []
    spans = []
    for item in decisions:
        actions = item["actions"]
        start = len(encoded)
        encoded.extend(
            constrained.encoded_action(
                tokenizer,
                item["messages"],
                action,
                max_length=int(max_length),
            )
            for action in actions
        )
        spans.append((start, len(encoded)))
    scores = constrained.score_encoded_actions(
        model, tokenizer, encoded, batch_size=int(batch_size)
    )
    output = []
    for item, (start, end) in zip(decisions, spans):
        local = scores[start:end]
        selected = item["actions"][max(range(len(local)), key=local.__getitem__)]
        parsed = feedback.validate_action(
            selected,
            properties=item["properties"],
            allow_stop=int(item["committed_edits"]) > 0,
        )
        if parsed is None:
            raise AssertionError("Constrained feedback action became invalid")
        output.append(parsed)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.attempts_per_condition) != 20:
        raise ValueError("Feedback repair protocol fixes exactly 20 generation attempts")
    evidence = json.loads((args.evidence_root / "merged" / "summary.json").read_text())
    if evidence.get("passed") is not True:
        raise ValueError("Train-only verifier evidence gate did not pass")
    models = closed_loop.load_models(args.evidence_root)
    transforms = closed_loop.load_transforms(args.evidence_root / "merged" / "transforms.jsonl")
    raw_rows = select_signal_rows(
        protocol.read_jsonl(args.dev_sources_jsonl),
        per_split=int(args.conditions_per_split),
        seed=int(args.seed),
    )
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("Invalid feedback-repair shard index/count")
    raw_rows = [
        row
        for row in raw_rows
        if protocol.stable_shard(
            str(row["_uca_source_group"]),
            seed=int(args.seed),
            shard_count=int(args.shard_count),
        )
        == int(args.shard_index)
    ]
    model, tokenizer = load_controller(args)
    output = []
    action_counts: Counter[str] = Counter()
    decision_count = 0
    decision_divergence = 0
    committed_total = 0
    rollback_total = 0
    unique_counts = []
    noop_total = 0
    for row_index, raw in enumerate(raw_rows):
        row = closed_loop.condition_row(raw, row_index)
        condition_id = str(row["condition_id"])
        properties = tuple(str(row["external_task_properties"]).split(","))
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
            direct.TrajectoryState(
                attempt_index=index,
                rng=random.Random(direct.stable_seed(int(args.seed), condition_id, index)),
                current_smiles=source,
                current_feature=source_feature.copy(),
                margins=dict(initial_margin_rows[0]),
            )
            for index in range(20)
        ]
        task_transforms = transforms.get(str(row["external_task_key"]), [])
        previous_events: dict[int, dict[str, object] | None] = {index: None for index in range(20)}
        for proposal_index in range(int(args.max_proposals)):
            prepared = []
            for state in states:
                if not state.active:
                    continue
                if state.accepted_edits >= int(args.max_committed_edits):
                    state.active = False
                    state.stop_reason = "max_committed_edits"
                    continue
                actions_by_property = {}
                support = {}
                for prop in properties:
                    actions = direct.symbolic_actions(
                        state.current_smiles,
                        task_transforms,
                        focus=prop,
                        order=properties,
                        margins=state.margins,
                        used=state.used_actions,
                        min_similarity=float(args.min_retrieval_similarity),
                        min_core_heavy_atoms=int(args.min_core_heavy_atoms),
                        max_variable_heavy_atoms=int(args.max_variable_heavy_atoms),
                        limit=int(args.max_symbolic_actions),
                    )
                    actions_by_property[prop] = actions
                    support[prop] = action_support_summary(actions, prop)
                available_actions = feedback.candidate_actions(
                    properties, support, committed_edits=state.accepted_edits
                )
                if not available_actions:
                    state.active = False
                    state.stop_reason = "no_executable_property_support"
                    continue
                messages = feedback.prompt_messages(
                    raw,
                    properties,
                    state.margins,
                    support,
                    current_smiles=state.current_smiles,
                    committed_edits=state.accepted_edits,
                    proposal_count=proposal_index,
                    max_committed_edits=int(args.max_committed_edits),
                    max_proposals=int(args.max_proposals),
                    previous_event=previous_events[state.attempt_index],
                )
                if feedback.state_contains_evaluation_target(messages):
                    raise AssertionError("Evaluation target leaked into online feedback state")
                prepared.append(
                    {
                        "state": state,
                        "properties": properties,
                        "actions": available_actions,
                        "actions_by_property": actions_by_property,
                        "support": support,
                        "messages": messages,
                        "committed_edits": state.accepted_edits,
                    }
                )
            if not prepared:
                break
            if args.controller_mode == "llm":
                selected_actions = choose_llm_actions(
                    model,
                    tokenizer,
                    prepared,
                    max_length=int(args.max_length),
                    batch_size=int(args.score_batch_size),
                )
            else:
                selected_actions = [
                    deterministic_action(
                        item["properties"],
                        item["state"].margins,
                        item["support"],
                        committed_edits=item["state"].accepted_edits,
                    )
                    for item in prepared
                ]
            proposed = []
            for item, selected in zip(prepared, selected_actions):
                state = item["state"]
                deterministic = deterministic_action(
                    item["properties"],
                    state.margins,
                    item["support"],
                    committed_edits=state.accepted_edits,
                )
                decision_count += 1
                decision_divergence += int(selected != deterministic)
                action_type, prop = selected
                action_counts[f"{action_type}:{prop or ''}"] += 1
                if action_type == "stop":
                    state.active = False
                    state.stop_reason = "controller_stop"
                    continue
                edit = direct.choose_valid_edit(
                    state,
                    original_source=source,
                    actions=item["actions_by_property"][str(prop)],
                    min_source_tanimoto=float(args.min_source_tanimoto),
                    retry_limit=int(args.action_retry_limit),
                    temperature=float(args.temperature),
                )
                if edit is None:
                    event = {
                        "property": prop,
                        "outcome": "rollback",
                        "reason": "no_safe_executable_delta",
                    }
                    previous_events[state.attempt_index] = event
                    state.transaction_rollbacks += 1
                    rollback_total += 1
                    state.trace.append({"proposal": proposal_index + 1, **event})
                    continue
                proposed.append((item, str(prop), edit, deterministic))
            if not proposed:
                continue
            _scores, margin_rows = closed_loop.score_candidates_batch(
                source_feature,
                [item[2][1] for item in proposed],
                properties=properties,
                models=models,
                source_tanimotos=[item[2][2] for item in proposed],
                retrieval_similarities=[item[2][5] for item in proposed],
                frequencies=[int(item[2][3].get("frequency", 0)) for item in proposed],
            )
            for (item, prop, edit, deterministic), margins in zip(proposed, margin_rows):
                state = item["state"]
                generated, feature, tanimoto, transform, _core, retrieval_similarity = edit
                before = dict(state.margins)
                accepted, reason, diagnostics = feedback_transaction_decision(
                    before, margins, focus=prop
                )
                identity = (
                    str(transform["task_key"]),
                    str(transform["source_variable"]),
                    str(transform["target_variable"]),
                )
                state.used_actions.add(identity)
                event = {
                    "property": prop,
                    "outcome": "commit" if accepted else "rollback",
                    "reason": reason,
                    "controller_mode": str(args.controller_mode),
                    "deterministic_action": list(deterministic),
                }
                if accepted:
                    state.current_smiles = generated
                    state.current_feature = feature
                    state.source_tanimoto = float(tanimoto)
                    state.margins = dict(margins)
                    state.accepted_edits += 1
                    committed_total += 1
                else:
                    state.transaction_rollbacks += 1
                    rollback_total += 1
                previous_events[state.attempt_index] = event
                state.trace.append(
                    {
                        "proposal": proposal_index + 1,
                        **event,
                        "proposed_smiles": generated,
                        "committed_smiles_after": state.current_smiles,
                        "source_variable": transform["source_variable"],
                        "target_variable": transform["target_variable"],
                        "retrieval_similarity": float(retrieval_similarity),
                        "source_tanimoto": float(tanimoto),
                        "margins_before": before,
                        "proposed_margins": dict(margins),
                        "margins_after": dict(state.margins),
                        **diagnostics,
                    }
                )
        unique_counts.append(len({state.current_smiles for state in states}))
        for attempt_number, state in enumerate(states, start=1):
            if state.active:
                state.stop_reason = "max_proposals"
            noop_total += int(state.accepted_edits == 0)
            output.append(
                {
                    **row,
                    "generated_smiles": state.current_smiles,
                    "method": f"common_llm_feedback_repair_v14a_{args.controller_mode}",
                    "generation_attempt_index": attempt_number,
                    "candidate_valid": True,
                    "candidate_source_similarity_pass": state.source_tanimoto
                    >= float(args.min_source_tanimoto),
                    "candidate_is_noop": state.accepted_edits == 0,
                    "source_tanimoto": state.source_tanimoto,
                    "trajectory_id": f"{condition_id}:trajectory:{attempt_number:02d}",
                    "trajectory_step_count": state.accepted_edits,
                    "trajectory_proposal_count": len(state.trace),
                    "trajectory_transaction_rollbacks": state.transaction_rollbacks,
                    "trajectory_stop_reason": state.stop_reason,
                    "trajectory_final_margins_json": json.dumps(state.margins, sort_keys=True),
                    "trajectory_trace_json": json.dumps(state.trace, sort_keys=True),
                    "output_selection": "none",
                }
            )
        print(f"[feedback-repair:{args.controller_mode}] {row_index + 1}/{len(raw_rows)}", flush=True)

    write_csv(args.output_csv, output)
    manifest = {
        "protocol": "common_llm_feedback_repair_signal_v14a",
        "controller_mode": str(args.controller_mode),
        "data_role": "fit_tools_to_stable_disjoint_dev_signal",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "output_selection": "none",
        "internal_molecular_candidate_pool": False,
        "candidate_budget": 20,
        "attempts_per_condition": 20,
        "conditions_per_split": int(args.conditions_per_split),
        "ind_conditions": sum(
            str(row["_uca_task_id"]) in protocol.IND_TASK_IDS for row in raw_rows
        ),
        "ood_conditions": sum(
            str(row["_uca_task_id"]) in protocol.OOD_TASK_IDS for row in raw_rows
        ),
        "conditions": len(raw_rows),
        "candidate_rows": len(output),
        "max_committed_edits": int(args.max_committed_edits),
        "max_proposals": int(args.max_proposals),
        "committed_edits_total": committed_total,
        "transaction_rollbacks_total": rollback_total,
        "noop_attempt_rows": noop_total,
        "mean_unique_candidates_per_condition": sum(unique_counts) / max(len(unique_counts), 1),
        "min_unique_candidates_per_condition": min(unique_counts, default=0),
        "controller_decisions": decision_count,
        "controller_action_counts": dict(sorted(action_counts.items())),
        "controller_deterministic_divergence_rate": decision_divergence / max(decision_count, 1),
        "soft_uncertainty_feedback": True,
        "exact_qed_safety_critic": True,
        "nonexact_verifier_is_hard_veto": False,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "seed": int(args.seed),
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
