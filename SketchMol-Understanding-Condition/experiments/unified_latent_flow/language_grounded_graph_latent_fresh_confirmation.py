#!/usr/bin/env python3
"""Leakage-free prospective confirmation for language-grounded graph latent flow.

The experiment has three physically separated stages. ``prepare`` may read
train/validation targets, but writes a target-free generation manifest and a
separate evaluation manifest. ``arm`` trains an architecture-matched residual
adapter using fit-only source/target pairs and freezes exactly twenty molecules
per fresh condition without accepting an evaluation-target path. ``evaluate``
opens the fresh targets only after both arms have frozen their raw attempts.

Unlike the predecessor, every generation-time graph condition is built only
from the source SMILES and increase/decrease instructions. Numeric properties
copied from the target molecule are deliberately removed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for module_path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import language_grounded_graph_latent_flow as v1  # noqa: E402


state_guidance = v1.state_guidance
base = v1.base
belief = v1.belief
graph = v1.graph
b41 = v1.b41
b40 = v1.b40
b39 = v1.b39
b37 = state_guidance.b37
b36 = state_guidance.b36
delta = state_guidance.delta
hierarchical = state_guidance.hierarchical
unified = state_guidance.unified

PROTOCOL = "direction_only_language_grounded_graph_latent_fresh_v2"
ARMS = ("property_memory", "common_llm_memory")
FORBIDDEN_GENERATION_TERMS = (
    "target_smiles",
    "target_",
    "delta_",
    "ground_truth",
    "reference_smiles",
    "property_oracle",
    "strict_success",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-csv", type=Path, required=True)
    prepare.add_argument("--validation-csv", type=Path, required=True)
    prepare.add_argument("--b22-checkpoint", type=Path, required=True)
    prepare.add_argument("--b22-summary", type=Path, required=True)
    prepare.add_argument("--predecessor-manifest", type=Path, required=True)
    prepare.add_argument("--known-source-csv", action="append", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    arm = subparsers.add_parser("arm")
    arm.add_argument("--arm", choices=ARMS, required=True)
    arm.add_argument("--prepare-summary", type=Path, required=True)
    arm.add_argument("--fit-bundle", type=Path, required=True)
    arm.add_argument("--generation-conditions", type=Path, required=True)
    arm.add_argument("--representation-checkpoint", type=Path, required=True)
    arm.add_argument("--representation-summary", type=Path, required=True)
    arm.add_argument("--b41-checkpoint", type=Path, required=True)
    arm.add_argument("--sft-adapter-dir", type=Path, required=True)
    arm.add_argument("--output-dir", type=Path, required=True)
    arm.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prepare-summary", type=Path, required=True)
    evaluate.add_argument("--evaluation-targets", type=Path, required=True)
    for arm_name in ARMS:
        evaluate.add_argument(f"--{arm_name.replace('_', '-')}-summary", type=Path, required=True)
        evaluate.add_argument(f"--{arm_name.replace('_', '-')}-candidates", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def file_sha256(path: Path) -> str:
    return belief.file_sha256(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "arms": list(ARMS),
        "direction_only_generation_conditions": True,
        "numeric_target_property_access_during_generation": False,
        "fresh_target_process_isolation": True,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "official_test_access": False,
        "model_training": True,
        "single_seed": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Direction-only fresh preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Direction-only fresh implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    if sum(int(value) for value in dict(payload["fresh_property_count_quotas"]).values()) != int(
        payload["fresh_condition_count"]
    ):
        raise ValueError("Fresh condition quotas do not sum to the locked count")
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing direction-only fresh inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Direction-only fresh locked-input drift: {drift}")
    return actual


def stable_key(seed: int, *values: object) -> str:
    payload = "\0".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def direction_only_row(row: Mapping[str, str], source_smiles: str) -> dict[str, str]:
    specs = [(str(name), int(direction)) for name, direction in base.task_specs(row)]
    if not specs or any(direction == 0 for _name, direction in specs):
        raise ValueError("Direction-only condition requires non-zero property instructions")
    safe: dict[str, str] = {
        "task_mode": "edit",
        "source_smiles": str(source_smiles),
        "property_count": str(len(specs)),
        "condition_properties": ",".join(name for name, _direction in specs),
        "instruction_tasks": json.dumps(
            [
                {
                    "property": name,
                    "direction": "increase" if direction > 0 else "decrease",
                }
                for name, direction in specs
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for name, direction in specs:
        safe[f"{name}_active"] = "1"
        safe[f"{name}_direction"] = "increase" if direction > 0 else "decrease"
    serialized = json.dumps(safe, sort_keys=True).lower()
    forbidden = [term for term in FORBIDDEN_GENERATION_TERMS if term in serialized]
    if forbidden:
        raise ValueError(f"Direction-only condition leaked forbidden fields: {forbidden}")
    return safe


def direction_only_pair(pair: object, condition_dim: int) -> object:
    copied = copy.copy(pair)
    copied.row = direction_only_row(pair.row, pair.source_smiles)
    copied.condition = hierarchical.property_latent_slot_tokens(copied.row, condition_dim)
    copied.task = base.task_key(copied.row)
    return copied


def known_sources(paths: Sequence[Path]) -> set[str]:
    sources: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                canonical = graph.canonical_smiles(str(row.get("source_smiles", "") or ""))
                if canonical:
                    sources.add(canonical)
    return sources


def reconstruct_predecessor_pairs(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    predecessor: Mapping[str, object],
):
    predecessor_args = SimpleNamespace(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        b22_checkpoint=args.b22_checkpoint,
        b22_summary=args.b22_summary,
    )
    b22_summary, checkpoint = b36.load_locked_b22(predecessor_args, predecessor)
    selected, reconstruction = b36.reconstruct_b22_train_pairs(
        predecessor_args, predecessor, checkpoint, b22_summary
    )
    fit, development, split = b37.strict_source_group_split(
        selected,
        seed=int(predecessor["development_split_seed"]),
        development_source_limit=int(predecessor["development_source_limit"]),
    )
    for pair in [*fit, *development]:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(predecessor["condition_dim"])
        )
    trajectory = state_guidance.select_trajectory_pairs(fit, predecessor)
    train_indices, validation_indices = state_guidance.split_trajectory_conditions(
        trajectory, predecessor
    )
    vocabulary = b37.checkpoint_vocabulary(checkpoint)
    support = b40.build_support(fit, vocabulary)
    return selected, fit, development, trajectory, train_indices, validation_indices, vocabulary, support, {
        "reconstruction": reconstruction,
        "split": split,
    }


def select_fresh_pairs(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    predecessor: Mapping[str, object],
    forbidden_sources: set[str],
) -> tuple[list[object], dict[str, object]]:
    rows = base.read_rows(args.validation_csv)
    candidates, filter_counts = base.build_pairs(
        rows,
        max_atoms=int(preregistration["max_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        condition_dim=int(preregistration["condition_dim"]),
        allowed_counts={int(value) for value in preregistration["fresh_property_counts"]},
        timeout=int(preregistration["mcs_timeout"]),
        min_common_fraction=float(preregistration["min_common_fraction"]),
        limit=len(rows),
        seed=int(preregistration["fresh_selection_seed"]),
        forbidden_sources=forbidden_sources,
    )
    quotas = {int(key): int(value) for key, value in dict(preregistration["fresh_property_count_quotas"]).items()}
    ordered = {
        count: sorted(
            [pair for pair in candidates if int(pair.property_count) == count],
            key=lambda pair: stable_key(
                int(preregistration["fresh_selection_seed"]),
                count,
                pair.source_smiles,
                base.task_key(pair.row),
            ),
        )
        for count in quotas
    }
    selected: list[object] = []
    used_sources: set[str] = set()
    positions = {count: 0 for count in quotas}
    remaining = dict(quotas)
    while any(value > 0 for value in remaining.values()):
        progressed = False
        for count in sorted(quotas):
            if remaining[count] <= 0:
                continue
            values = ordered[count]
            while positions[count] < len(values):
                pair = values[positions[count]]
                positions[count] += 1
                if pair.source_smiles in used_sources:
                    continue
                selected.append(pair)
                used_sources.add(pair.source_smiles)
                remaining[count] -= 1
                progressed = True
                break
        if not progressed:
            raise ValueError(
                f"Fresh source quotas cannot be satisfied without source reuse: remaining={remaining}"
            )
    selected.sort(
        key=lambda pair: stable_key(
            int(preregistration["fresh_selection_seed"]),
            "final",
            pair.source_smiles,
            base.task_key(pair.row),
        )
    )
    if len(selected) != int(preregistration["fresh_condition_count"]):
        raise ValueError("Fresh condition count drift after quota selection")
    counts = defaultdict(int)
    for pair in selected:
        counts[int(pair.property_count)] += 1
    return selected, {
        "candidate_filter_counts": filter_counts,
        "eligible_pairs": len(candidates),
        "eligible_unique_sources": len({pair.source_smiles for pair in candidates}),
        "selected_conditions": len(selected),
        "selected_unique_sources": len(used_sources),
        "selected_by_property_count": {str(key): counts[key] for key in sorted(counts)},
        "forbidden_sources": len(forbidden_sources),
    }


def run_prepare(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed direction-only prepare exists: {summary_path}")
    source_names = ["b26_candidates_sha256", "b33_candidates_sha256"]
    if len(args.known_source_csv) != len(source_names):
        raise ValueError("Prepare requires the locked B26 and B33 source CSVs exactly once")
    check_locked_inputs(
        preregistration,
        {
            "train_csv_sha256": args.train_csv,
            "validation_csv_sha256": args.validation_csv,
            "b22_checkpoint_sha256": args.b22_checkpoint,
            "b22_summary_sha256": args.b22_summary,
            "predecessor_manifest_sha256": args.predecessor_manifest,
            **{name: path for name, path in zip(source_names, args.known_source_csv, strict=True)},
        },
    )
    predecessor = v1.read_preregistration(args.predecessor_manifest)
    (
        selected,
        fit,
        development,
        trajectory,
        train_indices,
        validation_indices,
        vocabulary,
        support,
        lineage,
    ) = reconstruct_predecessor_pairs(args, preregistration, predecessor)
    forbidden = {
        pair.source_smiles for pair in [*selected, *fit, *development, *trajectory]
    }
    historical = known_sources(args.known_source_csv)
    forbidden |= historical
    fresh, fresh_selection = select_fresh_pairs(
        args, preregistration, predecessor, forbidden
    )
    direction_fit = [
        direction_only_pair(pair, int(preregistration["condition_dim"]))
        for pair in trajectory
    ]
    fit_bundle_path = args.output_dir / "fit_only_direction_pairs.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "pairs": direction_fit,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "vocabulary": vocabulary,
            "support": support,
            "lineage": lineage,
        },
        fit_bundle_path,
    )
    generation_records: list[dict[str, object]] = []
    evaluation_records: list[dict[str, object]] = []
    for index, pair in enumerate(fresh):
        safe = direction_only_row(pair.row, pair.source_smiles)
        condition = hierarchical.property_latent_slot_tokens(
            safe, int(preregistration["condition_dim"])
        )
        condition_id = f"prospective_direction_only_{index:04d}"
        generation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": index,
                "source_smiles": pair.source_smiles,
                "property_count": int(pair.property_count),
                "task": base.task_key(safe),
                "condition_row": safe,
                "condition": condition.tolist(),
            }
        )
        evaluation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": index,
                "source_smiles": pair.source_smiles,
                "target_smiles": pair.target_smiles,
                "property_count": int(pair.property_count),
                "row": dict(pair.row),
            }
        )
    generation_path = args.output_dir / "generation_conditions.json"
    evaluation_path = args.output_dir / "sealed_evaluation_targets.json"
    generation_payload = {
        "protocol": PROTOCOL,
        "role": "generation_conditions_without_answers",
        "records": generation_records,
        "selection": fresh_selection,
    }
    generation_text = json.dumps(generation_payload, indent=2, sort_keys=True) + "\n"
    leaks = [term for term in FORBIDDEN_GENERATION_TERMS if term in generation_text.lower()]
    if leaks:
        raise ValueError(f"Generation manifest contains forbidden target material: {leaks}")
    generation_path.write_text(generation_text, encoding="utf-8")
    evaluation_path.write_text(
        json.dumps(
            {"protocol": PROTOCOL, "role": "post_freeze_evaluation_targets", "records": evaluation_records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selected_sources = {pair.source_smiles for pair in fresh}
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare",
        "decision": "direction_only_fit_and_fresh_manifests_frozen",
        "fit_conditions": len(direction_fit),
        "fit_train_conditions": len(train_indices),
        "fit_validation_conditions": len(validation_indices),
        "fresh_conditions": len(fresh),
        "fresh_unique_sources": len(selected_sources),
        "fresh_forbidden_source_overlap": len(selected_sources & forbidden),
        "fresh_selection": fresh_selection,
        "artifacts": {
            "fit_bundle_sha256": file_sha256(fit_bundle_path),
            "generation_conditions_sha256": file_sha256(generation_path),
            "evaluation_targets_sha256": file_sha256(evaluation_path),
        },
        "contract": {
            "direction_only_generation_conditions": True,
            "numeric_target_property_access_during_generation": False,
            "fresh_target_process_isolation": True,
            "fit_target_access": True,
            "generation_target_access": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def load_generation_pairs(path: Path, preregistration: Mapping[str, object]) -> list[object]:
    payload = read_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("role") != "generation_conditions_without_answers":
        raise ValueError("Invalid target-free generation manifest")
    records = list(payload["records"])
    pairs = []
    for expected_index, record in enumerate(records):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("Generation pair order drift")
        safe = dict(record["condition_row"])
        serialized = json.dumps(safe, sort_keys=True).lower()
        if any(term in serialized for term in FORBIDDEN_GENERATION_TERMS):
            raise ValueError("Generation condition contains target-derived fields")
        source = graph.molecule_example(
            str(record["source_smiles"]),
            int(preregistration["max_atoms"]),
            int(preregistration["fingerprint_bits"]),
        )
        if source is None:
            raise ValueError(f"Cannot materialize fresh source {expected_index}")
        pairs.append(
            SimpleNamespace(
                row=safe,
                source_smiles=str(record["source_smiles"]),
                source=source,
                condition=np.asarray(record["condition"], dtype=np.float32),
                property_count=int(record["property_count"]),
                task=str(record["task"]),
            )
        )
    return pairs


def load_frozen_stack_for_arm(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    bundle: Mapping[str, object],
    device: torch.device,
):
    representation, representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = dict(bundle["vocabulary"])
    support = dict(bundle["support"])
    support_tensors = b40._device_support(support, device)
    checkpoint = torch.load(args.b41_checkpoint, map_location=device, weights_only=False)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(preregistration["condition_dim"]),
        transport_dim=int(preregistration["transport_dim"]),
        hidden_dim=int(preregistration["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(preregistration["max_jumps"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(preregistration["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)
    representation.eval().requires_grad_(False)
    return model, representation, vocabulary, support, support_tensors, representation_summary


def train_flow_only_adapter(
    adapter: nn.Module,
    model: nn.Module,
    representation: nn.Module,
    pairs: Sequence[object],
    indices: Sequence[int],
    memories: torch.Tensor,
    masks: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(preregistration["adapter_learning_rate"]),
        weight_decay=float(preregistration["adapter_weight_decay"]),
    )
    history = []
    batch_size = int(preregistration["adapter_batch_size"])
    for epoch in range(1, int(preregistration["adapter_epochs"]) + 1):
        order = list(indices)
        random.Random(int(preregistration["adapter_training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        adapter.train()
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            loss, values = v1.flow_matching_batch(
                adapter,
                model,
                representation,
                [pairs[index] for index in chosen],
                memories[chosen],
                masks[chosen],
                preregistration,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), float(preregistration["adapter_grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in values.items():
                totals[name] += float(value)
            batches += 1
        row = {"epoch": epoch, **{name: value / max(1, batches) for name, value in totals.items()}}
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite direction-only adapter metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "direction_only_adapter_epoch", **row}, sort_keys=True), flush=True)
    adapter.eval()
    return history


@torch.no_grad()
def validate_flow_only_adapter(
    adapter: nn.Module,
    model: nn.Module,
    representation: nn.Module,
    pairs: Sequence[object],
    indices: Sequence[int],
    memories: torch.Tensor,
    masks: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    batch_size = int(preregistration["adapter_batch_size"])
    adapter.eval()
    for start in range(0, len(indices), batch_size):
        chosen = list(indices[start : start + batch_size])
        _loss, values = v1.flow_matching_batch(
            adapter,
            model,
            representation,
            [pairs[index] for index in chosen],
            memories[chosen],
            masks[chosen],
            preregistration,
            device,
        )
        for name, value in values.items():
            totals[name] += float(value)
        batches += 1
    result = {name: value / max(1, batches) for name, value in totals.items()}
    result["relative_flow_mse_reduction"] = (
        result["base_flow_loss"] - result["flow_loss"]
    ) / max(1e-12, result["base_flow_loss"])
    return result


def run_arm(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed direction-only arm exists: {summary_path}")
    check_locked_inputs(
        preregistration,
        {
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "b41_checkpoint_sha256": args.b41_checkpoint,
            "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
        },
    )
    prepare = read_json(args.prepare_summary)
    artifacts = dict(prepare["artifacts"])
    for path, key in (
        (args.fit_bundle, "fit_bundle_sha256"),
        (args.generation_conditions, "generation_conditions_sha256"),
    ):
        if file_sha256(path) != artifacts[key]:
            raise ValueError(f"Prepared artifact drift: {path}")
    bundle = torch.load(args.fit_bundle, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != PROTOCOL:
        raise ValueError("Direction-only fit bundle protocol drift")
    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["adapter_training_seed"]))
    model, representation, vocabulary, support, support_tensors, representation_summary = (
        load_frozen_stack_for_arm(args, preregistration, bundle, device)
    )
    pairs = list(bundle["pairs"])
    memory_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    memories, masks, llm_manifest = state_guidance.development_memories(
        args.arm, pairs, preregistration, memory_args, device
    )
    memories = v1.equalize_memory_dimension(memories, preregistration)
    base.seed_everything(int(preregistration["adapter_initialization_seed"]))
    adapter = v1.LanguageGroundedTransportAdapter(
        latent_dim=int(preregistration["transport_dim"]),
        source_dim=int(preregistration["representation_node_dim"]),
        memory_dim=int(memories.shape[-1]),
        hidden_dim=int(preregistration["adapter_hidden_dim"]),
        residual_scale=float(preregistration["adapter_residual_scale"]),
    ).to(device)
    training = train_flow_only_adapter(
        adapter,
        model,
        representation,
        pairs,
        list(bundle["train_indices"]),
        memories,
        masks,
        preregistration,
        device,
    )
    base.seed_everything(int(preregistration["fit_validation_seed"]))
    validation = validate_flow_only_adapter(
        adapter,
        model,
        representation,
        pairs,
        list(bundle["validation_indices"]),
        memories,
        masks,
        preregistration,
        device,
    )
    checkpoint_path = args.output_dir / "direction_only_transport_adapter.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": args.arm,
            "state_dict": adapter.state_dict(),
            "memory_dim": int(memories.shape[-1]),
        },
        checkpoint_path,
    )
    fresh_pairs = load_generation_pairs(args.generation_conditions, preregistration)
    fresh_memories, fresh_masks, _ = state_guidance.development_memories(
        args.arm, fresh_pairs, preregistration, memory_args, device
    )
    fresh_memories = v1.equalize_memory_dimension(fresh_memories, preregistration)
    frozen, transport_metrics, support_manifest = v1.freeze_adapted_candidates(
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        fresh_pairs,
        fresh_memories,
        fresh_masks,
        adapter,
        preregistration,
        device,
    )
    records = list(read_json(args.generation_conditions)["records"])
    for row in frozen:
        row["condition_id"] = str(records[int(row["pair_index"])]["condition_id"])
    frozen_path = args.output_dir / "frozen_prospective_candidates.csv"
    base.write_candidate_rows(frozen_path, frozen)
    summary = {
        "protocol": PROTOCOL,
        "stage": "arm_freeze",
        "arm": args.arm,
        "decision": "await_post_freeze_target_evaluation",
        "training": training,
        "fit_validation": validation,
        "transport_metrics": transport_metrics,
        "conditions": len(fresh_pairs),
        "candidate_rows": len(frozen),
        "artifacts": {
            "adapter_checkpoint_sha256": file_sha256(checkpoint_path),
            "frozen_candidates_sha256": file_sha256(frozen_path),
        },
        "manifest": {
            "direction_only_generation_conditions": True,
            "numeric_target_property_access_during_generation": False,
            "generation_target_path_accepted": False,
            "fresh_target_process_isolation": True,
            "fit_target_access": True,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
            "posthoc_molecule_repair": False,
            "common_llm": llm_manifest,
            "exact_molecule_stop_support": support_manifest,
            "representation_protocol": representation_summary.get("protocol"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def coerce_frozen_rows(path: Path) -> list[dict[str, object]]:
    integer_fields = {
        "pair_index", "attempt", "property_count", "particle_index", "predicted_atom_count",
        "predicted_cardinality", "event_count", "cardinality_residual_at_stop",
        "node_delete_events", "node_write_events", "edge_delete_events", "edge_set_events",
        "affected_node_count", "affected_components", "stop_masked_steps",
    }
    float_fields = {
        "latent_norm", "dynamic_support_mask_fraction", "initial_particle_mean_abs_cosine",
        "initial_particle_max_abs_cosine", "final_particle_mean_abs_cosine",
        "final_particle_max_abs_cosine", "final_particle_centered_rms",
        "minimum_transport_particle_rms",
    }
    bool_fields = {"stopped_by_model", "max_horizon_hit", "outside_source_invariant"}
    rows = base.read_rows(path)
    for row in rows:
        for key in integer_fields & row.keys():
            row[key] = int(row[key])
        for key in float_fields & row.keys():
            row[key] = float(row[key])
        for key in bool_fields & row.keys():
            row[key] = str(row[key]).strip().lower() == "true"
    return rows


def load_evaluation_pairs(path: Path, preregistration: Mapping[str, object]) -> list[object]:
    payload = read_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("role") != "post_freeze_evaluation_targets":
        raise ValueError("Invalid sealed evaluation-target manifest")
    pairs = []
    for expected_index, record in enumerate(payload["records"]):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("Evaluation pair order drift")
        row = {str(key): str(value) for key, value in dict(record["row"]).items()}
        aligned = base.align_pair(
            str(record["source_smiles"]),
            str(record["target_smiles"]),
            max_atoms=int(preregistration["max_atoms"]),
            fingerprint_bits=int(preregistration["fingerprint_bits"]),
            timeout=int(preregistration["mcs_timeout"]),
            min_common_fraction=float(preregistration["min_common_fraction"]),
        )
        if aligned is None:
            raise ValueError(f"Cannot reconstruct sealed evaluation pair {expected_index}")
        source, target, common = aligned
        pairs.append(
            base.EditPair(
                row=row,
                source_smiles=str(record["source_smiles"]),
                target_smiles=str(record["target_smiles"]),
                source=source,
                target=target,
                condition=np.zeros((1, int(preregistration["condition_dim"])), dtype=np.float32),
                property_count=int(record["property_count"]),
                task=base.task_key(row),
                common_atoms=int(common),
            )
        )
    return pairs


def grouped_metrics(rows: Sequence[Mapping[str, object]], counts: set[int]) -> dict[str, float]:
    by_condition: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["property_count"]) in counts:
            by_condition[str(row["condition_id"])].append(row)
    conditions = list(by_condition.values())
    flat = [row for values in conditions for row in values]
    valid = [row for row in flat if bool(row["valid"])]
    return {
        "conditions": len(conditions),
        "validity": len(valid) / max(1, len(flat)),
        "property_any20": sum(any(bool(row["property_success"]) for row in values) for values in conditions) / max(1, len(conditions)),
        "strict_any20": sum(any(bool(row["strict_success"]) for row in values) for values in conditions) / max(1, len(conditions)),
    }


def task_breakdown(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_condition: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    by_task: defaultdict[str, list[list[Mapping[str, object]]]] = defaultdict(list)
    for values in by_condition.values():
        by_task[str(values[0]["task"])].append(values)
    return {
        task: {
            "conditions": len(conditions),
            "property_any20": sum(any(bool(row["property_success"]) for row in values) for values in conditions) / len(conditions),
            "strict_any20": sum(any(bool(row["strict_success"]) for row in values) for values in conditions) / len(conditions),
        }
        for task, conditions in sorted(by_task.items())
    }


def condition_outcomes(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, bool]:
    output: dict[str, bool] = defaultdict(bool)
    for row in rows:
        output[str(row["condition_id"])] |= bool(row[field])
    return dict(output)


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed direction-only evaluation exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    if file_sha256(args.evaluation_targets) != dict(prepare["artifacts"])["evaluation_targets_sha256"]:
        raise ValueError("Sealed evaluation targets drifted after prepare")
    pairs = load_evaluation_pairs(args.evaluation_targets, preregistration)
    arm_paths = {
        "property_memory": (args.property_memory_summary, args.property_memory_candidates),
        "common_llm_memory": (args.common_llm_memory_summary, args.common_llm_memory_candidates),
    }
    arm_summaries: dict[str, object] = {}
    evaluated_by_arm: dict[str, list[dict[str, object]]] = {}
    for arm, (arm_summary_path, candidates_path) in arm_paths.items():
        arm_summary = read_json(arm_summary_path)
        if arm_summary.get("protocol") != PROTOCOL or arm_summary.get("arm") != arm:
            raise ValueError(f"Invalid frozen arm summary: {arm}")
        if file_sha256(candidates_path) != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]:
            raise ValueError(f"Frozen candidate drift for {arm}")
        frozen = coerce_frozen_rows(candidates_path)
        expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
        if len(frozen) != expected:
            raise ValueError(f"{arm} expected {expected} frozen rows, found {len(frozen)}")
        evaluated, metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = dict(metrics)
        metrics["core_2p_3p"] = grouped_metrics(evaluated, {2, 3})
        metrics["hard_4p_7p"] = grouped_metrics(evaluated, {4, 5, 6, 7})
        metrics["by_task"] = task_breakdown(evaluated)
        evaluated_path = args.output_dir / f"evaluated_{arm}_candidates.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        arm_summaries[arm] = {
            "fit_validation": arm_summary["fit_validation"],
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
        evaluated_by_arm[arm] = evaluated
    prop = dict(arm_summaries["property_memory"])["metrics"]
    llm = dict(arm_summaries["common_llm_memory"])["metrics"]
    prop_strict = condition_outcomes(evaluated_by_arm["property_memory"], "strict_success")
    llm_strict = condition_outcomes(evaluated_by_arm["common_llm_memory"], "strict_success")
    llm_wins = sum(llm_strict[key] and not prop_strict[key] for key in llm_strict)
    property_wins = sum(prop_strict[key] and not llm_strict[key] for key in llm_strict)
    discordant = llm_wins + property_wins
    tail = sum(math.comb(discordant, index) for index in range(0, min(llm_wins, property_wins) + 1)) if discordant else 1
    sign_p = min(1.0, 2.0 * tail / (2**discordant)) if discordant else 1.0
    gates = dict(preregistration["gates"])
    checks = {
        "fresh_condition_count": {"value": len(pairs), "threshold": int(preregistration["fresh_condition_count"])},
        "fresh_unique_sources": {"value": prepare["fresh_unique_sources"], "threshold": int(preregistration["fresh_condition_count"])},
        "fresh_forbidden_source_overlap": {"value": prepare["fresh_forbidden_source_overlap"], "threshold": 0},
        "exact_attempts": {"value": llm["attempted_per_condition"], "threshold": 20},
        "candidate_rows": {"value": llm["candidate_rows"], "threshold": len(pairs) * 20},
        "llm_validity": {"value": llm["validity"], "threshold": gates["llm_validity"]},
        "llm_mean_source_tanimoto": {"value": llm["mean_source_tanimoto"], "threshold": gates["llm_mean_source_tanimoto"]},
        "llm_mean_unique_valid": {"value": llm["mean_unique_valid"], "threshold": gates["llm_mean_unique_valid"]},
        "llm_overall_strict_any20": {"value": llm["strict_any20"], "threshold": gates["llm_overall_strict_any20"]},
        "llm_hard_strict_any20": {"value": llm["hard_4p_7p"]["strict_any20"], "threshold": gates["llm_hard_strict_any20"]},
        "llm_strict_gain_vs_property": {"value": llm["strict_any20"] - prop["strict_any20"], "threshold": gates["llm_strict_gain_vs_property"]},
        "llm_validity_delta_vs_property": {"value": llm["validity"] - prop["validity"], "threshold": gates["llm_validity_delta_vs_property"]},
        "core_strict_delta_vs_property": {"value": llm["core_2p_3p"]["strict_any20"] - prop["core_2p_3p"]["strict_any20"], "threshold": gates["core_strict_delta_vs_property"]},
        "hard_strict_delta_vs_property": {"value": llm["hard_4p_7p"]["strict_any20"] - prop["hard_4p_7p"]["strict_any20"], "threshold": gates["hard_strict_delta_vs_property"]},
        "llm_fit_relative_flow_reduction": {"value": arm_summaries["common_llm_memory"]["fit_validation"]["relative_flow_mse_reduction"], "threshold": gates["llm_fit_relative_flow_reduction"]},
    }
    exact = {"fresh_condition_count", "fresh_unique_sources", "fresh_forbidden_source_overlap", "exact_attempts", "candidate_rows"}
    failures = [
        name
        for name, item in checks.items()
        if (item["value"] != item["threshold"] if name in exact else float(item["value"]) < float(item["threshold"]))
    ]
    summary = {
        "protocol": PROTOCOL,
        "stage": "post_freeze_evaluation",
        "arms": arm_summaries,
        "paired_strict": {
            "llm_wins": llm_wins,
            "property_wins": property_wins,
            "ties": len(pairs) - discordant,
            "two_sided_sign_p": sign_p,
        },
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "decision": (
            "advance_direction_only_language_grounded_flow_to_lora_signal"
            if not failures
            else "stop_direction_only_language_grounded_flow_without_fresh_retuning"
        ),
        "contract": {
            "direction_only_generation_conditions": True,
            "numeric_target_property_access_during_generation": False,
            "fresh_target_process_isolation": True,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
            "official_test_access": False,
            "repeat_on_same_fresh_sources": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    if args.stage == "prepare":
        return run_prepare(args, preregistration)
    if args.stage == "arm":
        return run_arm(args, preregistration)
    if args.stage == "evaluate":
        return run_evaluate(args, preregistration)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
