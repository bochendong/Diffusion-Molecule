#!/usr/bin/env python3
"""Prospective source-disjoint confirmation for graph-jump and language controls.

The protocol is deliberately split into three processes. ``prepare`` may inspect
training targets to choose unused source molecules, but writes a target-free
generation manifest and a separately sealed evaluation manifest. ``freeze``
accepts no evaluation-target path and emits exactly twenty raw attempts per
condition. ``evaluate`` opens the sealed targets only after every arm is frozen.
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
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
for module_path in (SCRIPT_DIR, PROJECT_DIR, LATENT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import language_grounded_graph_latent_fresh_confirmation as fresh_v3  # noqa: E402
from dead_end_safe_support import DeadEndSafeSupport  # noqa: E402
from sketchmol_understanding_condition.text_features import hashed_text_vector  # noqa: E402


base = fresh_v3.base
graph = fresh_v3.graph
b41 = fresh_v3.b41
b40 = fresh_v3.b40
b39 = fresh_v3.b39
b37 = fresh_v3.b37
delta = fresh_v3.delta
hierarchical = fresh_v3.hierarchical
unified = fresh_v3.unified
valid_terminal = fresh_v3.v1.valid_terminal

PROTOCOL = "fresh_graph_jump_language_causal_confirmation_v1"
GRAPH_ARMS = ("numeric_b41", "numeric_canonical", "numeric_d3_grpo")
LANGUAGE_ARMS = (
    "language_template",
    "language_paraphrase",
    "language_keyword",
    "language_scrambled",
    "language_reversed",
)
ALL_ARMS = (*GRAPH_ARMS, *LANGUAGE_ARMS)
FORBIDDEN_GENERATION_TERMS = fresh_v3.FORBIDDEN_GENERATION_TERMS


class InstructionConditionHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, token_count: int, condition_dim: int) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(token_count) * int(condition_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1, self.token_count, self.condition_dim)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-csv", type=Path, required=True)
    prepare.add_argument("--validation-csv", type=Path, required=True)
    prepare.add_argument("--b22-checkpoint", type=Path, required=True)
    prepare.add_argument("--b22-summary", type=Path, required=True)
    prepare.add_argument("--representation-checkpoint", type=Path, required=True)
    prepare.add_argument("--representation-summary", type=Path, required=True)
    prepare.add_argument("--b36-records", type=Path, required=True)
    prepare.add_argument("--trajectory-dataset", type=Path, required=True)
    prepare.add_argument("--predecessor-manifest", type=Path, required=True)
    prepare.add_argument("--predecessor-fit-bundle", type=Path, required=True)
    prepare.add_argument("--known-source", action="append", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--arm-group", choices=("graph", "language"), required=True)
    freeze.add_argument("--prepare-summary", type=Path, required=True)
    freeze.add_argument("--fit-bundle", type=Path, required=True)
    freeze.add_argument("--generation-conditions", type=Path, required=True)
    freeze.add_argument("--representation-checkpoint", type=Path, required=True)
    freeze.add_argument("--representation-summary", type=Path, required=True)
    freeze.add_argument("--b41-checkpoint", type=Path, required=True)
    freeze.add_argument("--canonical-checkpoint", type=Path, required=True)
    freeze.add_argument("--d3-checkpoint", type=Path, required=True)
    freeze.add_argument("--e1-head-checkpoint", type=Path, required=True)
    freeze.add_argument("--e1-manifest", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--device", default="auto")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prepare-summary", type=Path, required=True)
    evaluate.add_argument("--evaluation-targets", type=Path, required=True)
    evaluate.add_argument("--frozen-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def file_sha256(path: Path) -> str:
    return fresh_v3.file_sha256(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "arms": list(ALL_ARMS),
        "exact_raw_attempts_per_condition": 20,
        "fresh_target_process_isolation": True,
        "generation_target_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "official_test_target_access": False,
        "model_training": False,
        "single_seed": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Fresh confirmation preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            f"Fresh confirmation implementation drift: expected {payload.get('implementation_sha256')}, found {actual}"
        )
    quotas = {str(key): int(value) for key, value in dict(payload["fresh_task_quotas"]).items()}
    if sum(quotas.values()) != int(payload["fresh_condition_count"]):
        raise ValueError("Fresh task quotas do not sum to fresh_condition_count")
    return payload


def check_locked_inputs(preregistration: Mapping[str, object], paths: Mapping[str, Path]) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fresh confirmation inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Fresh confirmation locked-input drift: {drift}")
    return actual


def stable_key(seed: int, *values: object) -> str:
    text = "\0".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def table1_task(row: Mapping[str, str]) -> str:
    directions = {str(prop): int(direction) for prop, direction in base.task_specs(row)}
    known = {
        (("GSK3B", 1),): "GSK3B:increase",
        (("MW", 1),): "MW:increase",
        (("SA", -1),): "SA:decrease",
        (("RB", -1),): "RB:decrease",
        (("DRD2", -1), ("MW", -1), ("SA", -1)): "DRD2:decrease+MW:decrease+SA:decrease",
    }
    return known.get(tuple(sorted(directions.items())), "unknown")


def sources_from_path(path: Path) -> set[str]:
    sources: set[str] = set()
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        records = list(payload.get("records") or [])
        for row in records:
            value = graph.canonical_smiles(str(dict(row).get("source_smiles", "") or ""))
            if value:
                sources.add(value)
        return sources
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = graph.canonical_smiles(str(row.get("source_smiles", "") or ""))
            if value:
                sources.add(value)
    return sources


def load_frozen_predecessor_bundle(path: Path) -> dict[str, object]:
    """Load the successful V3 fit bundle without rerunning timeout-bound MCS.

    V3 was serialized while the graph modules were imported under their legacy
    names.  The classes are identical to the current modules, so the aliases
    only restore pickle compatibility; no data or model state is changed.
    """
    sys.modules.setdefault("categorical_graph_belief_base", base)
    sys.modules.setdefault("categorical_graph_latent_ae", graph)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != fresh_v3.PROTOCOL:
        raise ValueError("Frozen predecessor fit-bundle protocol drift")
    required = {"pairs", "vocabulary", "support", "lineage"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Frozen predecessor fit bundle missing fields: {missing}")
    return bundle


def locked_b36_sources(path: Path) -> set[str]:
    sources: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = graph.canonical_smiles(str(record.get("source_smiles", "") or ""))
        if value:
            sources.add(value)
    return sources


def all_validation_sources(path: Path) -> set[str]:
    sources: set[str] = set()
    for row in base.read_rows(path):
        value = graph.canonical_smiles(str(row.get("source_smiles", "") or ""))
        if value:
            sources.add(value)
    return sources


def select_fresh_pairs(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    forbidden_sources: set[str],
) -> tuple[list[object], dict[str, object]]:
    quotas = {str(key): int(value) for key, value in dict(preregistration["fresh_task_quotas"]).items()}
    rows = base.read_rows(args.train_csv)
    candidates, filter_counts = base.build_pairs(
        rows,
        max_atoms=int(preregistration["max_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        condition_dim=int(preregistration["condition_dim"]),
        allowed_counts={1, 3},
        timeout=int(preregistration["mcs_timeout"]),
        min_common_fraction=float(preregistration["min_common_fraction"]),
        limit=int(preregistration["fresh_alignment_limit"]),
        seed=int(preregistration["fresh_selection_seed"]),
        forbidden_sources=forbidden_sources,
    )
    buckets: dict[str, list[object]] = defaultdict(list)
    for pair in candidates:
        task = table1_task(pair.row)
        if task in quotas:
            buckets[task].append(pair)
    for task in buckets:
        buckets[task].sort(
            key=lambda pair: stable_key(
                int(preregistration["fresh_selection_seed"]), task, pair.source_smiles
            )
        )
    selected: list[object] = []
    used_sources: set[str] = set()
    for task in quotas:
        for pair in buckets.get(task, []):
            if pair.source_smiles in used_sources:
                continue
            selected.append(pair)
            used_sources.add(pair.source_smiles)
            if sum(table1_task(item.row) == task for item in selected) >= quotas[task]:
                break
        count = sum(table1_task(item.row) == task for item in selected)
        if count != quotas[task]:
            raise ValueError(f"Fresh task quota unavailable for {task}: wanted {quotas[task]}, found {count}")
    selected.sort(
        key=lambda pair: stable_key(
            int(preregistration["fresh_selection_seed"]), "final", table1_task(pair.row), pair.source_smiles
        )
    )
    return selected, {
        "filter_counts": filter_counts,
        "eligible_pairs": len(candidates),
        "eligible_unique_sources": len({pair.source_smiles for pair in candidates}),
        "selected_conditions": len(selected),
        "selected_unique_sources": len(used_sources),
        "selected_by_task": {
            task: sum(table1_task(pair.row) == task for pair in selected) for task in quotas
        },
        "forbidden_sources": len(forbidden_sources),
    }


def run_prepare(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed fresh prepare exists: {summary_path}")
    fixed_paths = {
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "b36_records_sha256": args.b36_records,
        "trajectory_dataset_sha256": args.trajectory_dataset,
        "predecessor_manifest_sha256": args.predecessor_manifest,
        "predecessor_fit_bundle_sha256": args.predecessor_fit_bundle,
    }
    fixed_paths.update({f"known_source_{index}_sha256": path for index, path in enumerate(args.known_source)})
    check_locked_inputs(preregistration, fixed_paths)
    predecessor = fresh_v3.v1.read_preregistration(args.predecessor_manifest)
    bundle = load_frozen_predecessor_bundle(args.predecessor_fit_bundle)
    fit = list(bundle["pairs"])
    vocabulary = dict(bundle["vocabulary"])
    support = dict(bundle["support"])
    lineage = dict(bundle["lineage"])
    b36_sources = locked_b36_sources(args.b36_records)
    validation_sources = all_validation_sources(args.validation_csv)
    forbidden = set(b36_sources) | set(validation_sources)
    known_counts = {}
    for index, path in enumerate(args.known_source):
        values = sources_from_path(path)
        known_counts[str(index)] = len(values)
        forbidden |= values
    fresh, selection = select_fresh_pairs(args, preregistration, forbidden)
    fit_bundle_path = args.output_dir / "fit_support_bundle.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "fit_pairs": fit,
            "vocabulary": vocabulary,
            "support": support,
            "lineage": lineage,
        },
        fit_bundle_path,
    )
    generation_records: list[dict[str, object]] = []
    evaluation_records: list[dict[str, object]] = []
    for index, pair in enumerate(fresh):
        safe = fresh_v3.direction_only_row(pair.row, pair.source_smiles)
        condition = hierarchical.property_latent_slot_tokens(safe, int(preregistration["condition_dim"]))
        condition_id = f"fresh_graph_jump_{index:04d}"
        task = table1_task(pair.row)
        generation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": index,
                "source_smiles": pair.source_smiles,
                "property_count": int(pair.property_count),
                "task": task,
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
                "task": task,
                "row": dict(pair.row),
            }
        )
    generation_path = args.output_dir / "generation_conditions.json"
    evaluation_path = args.output_dir / "sealed_evaluation_targets.json"
    generation_text = json.dumps(
        {"protocol": PROTOCOL, "role": "generation_conditions_without_answers", "records": generation_records},
        indent=2,
        sort_keys=True,
    ) + "\n"
    leaks = [term for term in FORBIDDEN_GENERATION_TERMS if term in generation_text.lower()]
    if leaks:
        raise ValueError(f"Generation manifest contains forbidden target material: {leaks}")
    generation_path.write_text(generation_text, encoding="utf-8")
    evaluation_path.write_text(
        json.dumps(
            {"protocol": PROTOCOL, "role": "post_freeze_evaluation_targets", "records": evaluation_records},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    selected_sources = {pair.source_smiles for pair in fresh}
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare",
        "decision": "fresh_manifests_frozen",
        "fresh_conditions": len(fresh),
        "fresh_unique_sources": len(selected_sources),
        "fresh_forbidden_source_overlap": len(selected_sources & forbidden),
        "fresh_selection": selection,
        "deterministic_predecessor_reuse": {
            "fit_bundle_protocol": str(bundle["protocol"]),
            "fit_pairs": len(fit),
            "b36_record_sources": len(b36_sources),
            "all_validation_sources": len(validation_sources),
            "timeout_bound_mcs_reconstruction": False,
        },
        "known_source_counts": known_counts,
        "artifacts": {
            "fit_bundle_sha256": file_sha256(fit_bundle_path),
            "generation_conditions_sha256": file_sha256(generation_path),
            "evaluation_targets_sha256": file_sha256(evaluation_path),
        },
        "contract": {
            "generation_target_access": False,
            "fresh_target_process_isolation": True,
            "historical_table1_source_exclusion_access": True,
            "official_test_target_access": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def load_generation_pairs(path: Path, preregistration: Mapping[str, object]) -> list[object]:
    payload = read_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("role") != "generation_conditions_without_answers":
        raise ValueError("Invalid target-free generation manifest")
    pairs = []
    for expected_index, record in enumerate(payload["records"]):
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
                condition_id=str(record["condition_id"]),
            )
        )
    return pairs


def normalize_instruction(text: str) -> str:
    return str(text or "").replace("β", "b").replace("Β", "b").lower().strip()


def hashed_char_ngrams(text: str, dim: int, n: int) -> np.ndarray:
    padded = f"{' ' * (n - 1)}{normalize_instruction(text)}{' ' * (n - 1)}"
    vector = np.zeros(int(dim), dtype=np.float32)
    for index in range(max(0, len(padded) - n + 1)):
        gram = padded[index : index + n]
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % max(1, int(dim))
        sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def instruction_features(text: str, e1: Mapping[str, object]) -> np.ndarray:
    word = hashed_text_vector(normalize_instruction(text), int(e1["word_hash_dim"]))
    char = hashed_char_ngrams(text, int(e1["char_hash_dim"]), int(e1["char_ngram"]))
    return np.concatenate([word, char]).astype(np.float32)


def specs_for_row(row: Mapping[str, str]) -> list[tuple[str, int]]:
    return [(str(prop), int(direction)) for prop, direction in base.task_specs(row) if int(direction) != 0]


def render_template(specs: Sequence[tuple[str, int]], e1: Mapping[str, object]) -> str:
    names = dict(e1["property_names"])
    verbs = dict(e1["train_verbs"])
    parts = []
    for prop, direction in specs:
        direction_name = "increase" if direction > 0 else "decrease"
        options = list(verbs.get(direction_name, ["change"]))
        parts.append(f"{options[0]} {names.get(prop, prop)}")
    if len(parts) == 1:
        return f"Edit the molecule to {parts[0]}."
    return f"Edit the molecule to {', '.join(parts[:-1])}, and {parts[-1]}."


def render_keyword(specs: Sequence[tuple[str, int]], e1: Mapping[str, object]) -> str:
    names = dict(e1["property_names"])
    return " ".join(str(names.get(prop, prop)) for prop, _direction in specs) or "molecule"


def scramble_instruction(text: str, seed: int, condition_id: str) -> str:
    digest = hashlib.blake2b(f"{seed}|{condition_id}|{text}".encode("utf-8"), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "little"))
    output = []
    for token in text.split():
        body = list(token)
        if len(body) >= 3:
            rng.shuffle(body)
        output.append("".join(body))
    return " ".join(output)


def instruction_for_arm(arm: str, pair: object, e1: Mapping[str, object], seed: int) -> str:
    specs = specs_for_row(pair.row)
    template = render_template(specs, e1)
    if arm == "language_template":
        return template
    if arm == "language_paraphrase":
        return str(dict(e1["eval_paraphrases"]).get(pair.task, template))
    if arm == "language_keyword":
        return render_keyword(specs, e1)
    if arm == "language_scrambled":
        return scramble_instruction(template, seed, pair.condition_id)
    if arm == "language_reversed":
        return render_template([(prop, -direction) for prop, direction in specs], e1)
    raise ValueError(f"Unknown language arm: {arm}")


def build_model(representation_config: Mapping[str, object], preregistration: Mapping[str, object], vocabulary, device):
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    return b39.LatentCardinalityGraphJumpBridge(
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


def unique_smiles_mean(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = str(row.get("generated_smiles", "") or "")
        if value:
            grouped[str(row["condition_id"])].add(value)
    return float(np.mean([len(values) for values in grouped.values()])) if grouped else 0.0


def run_freeze(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "canonical_checkpoint_sha256": args.canonical_checkpoint,
        "d3_checkpoint_sha256": args.d3_checkpoint,
        "e1_head_checkpoint_sha256": args.e1_head_checkpoint,
        "e1_manifest_sha256": args.e1_manifest,
    }
    check_locked_inputs(preregistration, paths)
    prepare = read_json(args.prepare_summary)
    artifacts = dict(prepare["artifacts"])
    if file_sha256(args.fit_bundle) != artifacts["fit_bundle_sha256"]:
        raise ValueError("Prepared fit bundle drift")
    if file_sha256(args.generation_conditions) != artifacts["generation_conditions_sha256"]:
        raise ValueError("Prepared generation manifest drift")
    bundle = torch.load(args.fit_bundle, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != PROTOCOL:
        raise ValueError("Prepared fit bundle protocol drift")
    device = base.resolve_device(str(args.device))
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    representation.eval().requires_grad_(False)
    fit_pairs = list(bundle["fit_pairs"])
    vocabulary = dict(bundle["vocabulary"])
    support = dict(bundle["support"])
    support_tensors = b40._device_support(support, device)
    pairs = load_generation_pairs(args.generation_conditions, preregistration)
    checkpoints = {
        "numeric_b41": args.b41_checkpoint,
        "numeric_canonical": args.canonical_checkpoint,
        "numeric_d3_grpo": args.d3_checkpoint,
    }
    e1 = read_json(args.e1_manifest)
    head_blob = torch.load(args.e1_head_checkpoint, map_location="cpu", weights_only=False)
    head = InstructionConditionHead(
        int(head_blob["feature_dim"]),
        int(e1["head_hidden_dim"]),
        int(head_blob["token_count"]),
        int(head_blob["condition_dim"]),
    ).to(device)
    head.load_state_dict(dict(head_blob["model_state"]), strict=True)
    head.eval().requires_grad_(False)
    arms = GRAPH_ARMS if args.arm_group == "graph" else LANGUAGE_ARMS
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    safe_support = DeadEndSafeSupport(exact_support)
    original_support = b41.viability_event_mask
    group_rows = {}
    try:
        b41.viability_event_mask = safe_support
        for arm in arms:
            arm_dir = args.output_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = arm_dir / "frozen_candidates.csv"
            arm_summary_path = arm_dir / "summary.json"
            if arm_summary_path.is_file() and candidate_path.is_file():
                existing = read_json(arm_summary_path)
                if file_sha256(candidate_path) == dict(existing["artifacts"])["frozen_candidates_sha256"]:
                    group_rows[arm] = existing
                    print(json.dumps({"stage": "skip_frozen_arm", "arm": arm}, sort_keys=True), flush=True)
                    continue
            checkpoint_path = checkpoints.get(arm, args.canonical_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(representation_config, preregistration, vocabulary, device)
            model.load_state_dict(dict(checkpoint["model_state"]), strict=True)
            model.eval().requires_grad_(False)
            rows: list[dict[str, object]] = []
            started = time.perf_counter()
            for index, pair in enumerate(pairs):
                instruction = ""
                tokens = np.asarray(pair.condition, dtype=np.float32)
                if arm in LANGUAGE_ARMS:
                    instruction = instruction_for_arm(arm, pair, e1, int(preregistration["generation_seed"]))
                    features = torch.from_numpy(instruction_features(instruction, e1)[None, :]).to(device)
                    with torch.no_grad():
                        tokens = head(features)[0].detach().cpu().numpy().astype(np.float32)
                try:
                    generated = b41.sample_from_source(
                        model,
                        representation,
                        vocabulary,
                        support,
                        support_tensors,
                        pair.source,
                        tokens,
                        preregistration,
                        device,
                        int(preregistration["generation_seed"]) * 100000 + index,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {"stage": "sample_failed", "arm": arm, "condition_id": pair.condition_id, "error": f"{type(exc).__name__}: {exc}"},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    generated = [{"generated_smiles": ""}] * 20
                generated = (list(generated) + [{"generated_smiles": ""}] * 20)[:20]
                for attempt, candidate in enumerate(generated, start=1):
                    rows.append(
                        {
                            "condition_id": pair.condition_id,
                            "pair_index": index,
                            "attempt": attempt,
                            "property_count": pair.property_count,
                            "task": pair.task,
                            "source_smiles": pair.source_smiles,
                            "arm": arm,
                            "instruction": instruction,
                            **candidate,
                        }
                    )
                if (index + 1) % 12 == 0 or index + 1 == len(pairs):
                    print(
                        json.dumps({"stage": "freeze_progress", "arm": arm, "done": index + 1, "total": len(pairs)}, sort_keys=True),
                        flush=True,
                    )
            expected = len(pairs) * 20
            if len(rows) != expected:
                raise RuntimeError(f"{arm} expected {expected} rows, found {len(rows)}")
            base.write_candidate_rows(candidate_path, rows)
            arm_summary = {
                "protocol": PROTOCOL,
                "stage": "arm_freeze",
                "arm": arm,
                "candidate_rows": len(rows),
                "conditions": len(pairs),
                "attempts_per_condition": 20,
                "mean_unique_smiles": unique_smiles_mean(rows),
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "artifacts": {"frozen_candidates_sha256": file_sha256(candidate_path)},
                "contract": {
                    "generation_target_path_accepted": False,
                    "molecular_candidate_ranking": False,
                    "oracle_selection": False,
                    "retry_or_resampling": False,
                },
            }
            arm_summary_path.write_text(json.dumps(arm_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            group_rows[arm] = arm_summary
    finally:
        b41.viability_event_mask = original_support
    group_summary = {
        "protocol": PROTOCOL,
        "stage": "freeze_group",
        "arm_group": args.arm_group,
        "arms": group_rows,
        "exact_stop_support": safe_support.manifest(),
    }
    (args.output_dir / f"freeze_{args.arm_group}_summary.json").write_text(
        json.dumps(group_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(group_summary, indent=2, sort_keys=True), flush=True)
    return 0


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
                task=str(record["task"]),
                common_atoms=int(common),
            )
        )
    return pairs


def task_breakdown(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_condition: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    by_task: dict[str, list[list[Mapping[str, object]]]] = defaultdict(list)
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


def paired_summary(left: Mapping[str, bool], right: Mapping[str, bool]) -> dict[str, object]:
    keys = sorted(set(left) & set(right))
    left_wins = sum(bool(left[key]) and not bool(right[key]) for key in keys)
    right_wins = sum(bool(right[key]) and not bool(left[key]) for key in keys)
    discordant = left_wins + right_wins
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(left_wins, right_wins) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": len(keys) - discordant,
        "two_sided_sign_p": p_value,
    }


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed fresh evaluation exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    if file_sha256(args.evaluation_targets) != dict(prepare["artifacts"])["evaluation_targets_sha256"]:
        raise ValueError("Sealed evaluation targets drifted")
    pairs = load_evaluation_pairs(args.evaluation_targets, preregistration)
    expected = len(pairs) * 20
    metrics_by_arm = {}
    evaluated_by_arm = {}
    outcomes = {}
    for arm in ALL_ARMS:
        arm_dir = args.frozen_root / arm
        arm_summary = read_json(arm_dir / "summary.json")
        candidate_path = arm_dir / "frozen_candidates.csv"
        if file_sha256(candidate_path) != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]:
            raise ValueError(f"Frozen candidate drift for {arm}")
        frozen = fresh_v3.coerce_frozen_rows(candidate_path)
        if len(frozen) != expected:
            raise ValueError(f"{arm} expected {expected} rows, found {len(frozen)}")
        evaluated, metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = dict(metrics)
        metrics["by_task"] = task_breakdown(evaluated)
        evaluated_path = args.output_dir / f"evaluated_{arm}.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        metrics_by_arm[arm] = {
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
        evaluated_by_arm[arm] = evaluated
        outcomes[arm] = condition_outcomes(evaluated, "strict_success")
    strict = {arm: float(dict(metrics_by_arm[arm]["metrics"])["strict_any20"]) for arm in ALL_ARMS}
    semantic_floor = min(strict["language_template"], strict["language_paraphrase"])
    control_ceiling = max(strict["language_keyword"], strict["language_scrambled"])
    semantic_margin = semantic_floor - control_ceiling
    reversed_drop = semantic_floor - strict["language_reversed"]
    graph_gates = {
        "canonical_strict_any20": strict["numeric_canonical"] >= float(preregistration["gates"]["canonical_strict_any20"]),
        "canonical_gain_vs_b41": strict["numeric_canonical"] - strict["numeric_b41"] >= float(preregistration["gates"]["canonical_gain_vs_b41"]),
        "d3_not_worse_than_b41": strict["numeric_d3_grpo"] - strict["numeric_b41"] >= float(preregistration["gates"]["d3_gain_vs_b41"]),
    }
    language_gates = {
        "semantic_margin": semantic_margin >= float(preregistration["gates"]["language_semantic_margin"]),
        "reversed_drop": reversed_drop >= float(preregistration["gates"]["language_reversed_drop"]),
        "paraphrase_stability": abs(strict["language_template"] - strict["language_paraphrase"]) <= float(preregistration["gates"]["language_paraphrase_max_gap"]),
    }
    quality_gates = {}
    for arm in ALL_ARMS:
        metrics = dict(metrics_by_arm[arm]["metrics"])
        quality_gates[f"{arm}_validity"] = float(metrics["validity"]) >= float(preregistration["gates"]["validity"])
        quality_gates[f"{arm}_source_tanimoto"] = float(metrics["mean_source_tanimoto"]) >= float(preregistration["gates"]["mean_source_tanimoto"])
        quality_gates[f"{arm}_unique"] = float(metrics["mean_unique_valid"]) >= float(preregistration["gates"]["mean_unique_valid"])
    contract_gates = {
        "fresh_condition_count": len(pairs) == int(preregistration["fresh_condition_count"]),
        "fresh_unique_sources": int(prepare["fresh_unique_sources"]) == int(preregistration["fresh_condition_count"]),
        "fresh_forbidden_source_overlap": int(prepare["fresh_forbidden_source_overlap"]) == 0,
        "candidate_rows": all(int(dict(metrics_by_arm[arm]["metrics"])["candidate_rows"]) == expected for arm in ALL_ARMS),
        "exact_attempts": all(int(dict(metrics_by_arm[arm]["metrics"])["attempted_per_condition"]) == 20 for arm in ALL_ARMS),
    }
    graph_passed = all(graph_gates.values()) and all(quality_gates.values()) and all(contract_gates.values())
    language_passed = graph_passed and all(language_gates.values())
    if language_passed:
        decision = "advance_canonical_graph_jump_with_language"
    elif graph_passed:
        decision = "advance_canonical_graph_jump_without_language"
    else:
        decision = "stop_fresh_confirmation_without_retuning"
    summary = {
        "protocol": PROTOCOL,
        "stage": "post_freeze_evaluation",
        "decision": decision,
        "arms": metrics_by_arm,
        "strict_any20": strict,
        "effects": {
            "canonical_gain_vs_b41": strict["numeric_canonical"] - strict["numeric_b41"],
            "d3_gain_vs_b41": strict["numeric_d3_grpo"] - strict["numeric_b41"],
            "language_semantic_margin": semantic_margin,
            "language_reversed_drop": reversed_drop,
        },
        "paired": {
            "canonical_vs_b41": paired_summary(outcomes["numeric_canonical"], outcomes["numeric_b41"]),
            "d3_vs_b41": paired_summary(outcomes["numeric_d3_grpo"], outcomes["numeric_b41"]),
            "template_vs_scrambled": paired_summary(outcomes["language_template"], outcomes["language_scrambled"]),
            "paraphrase_vs_reversed": paired_summary(outcomes["language_paraphrase"], outcomes["language_reversed"]),
        },
        "gates": {
            "graph_passed": graph_passed,
            "language_passed": language_passed,
            "graph": graph_gates,
            "language": language_gates,
            "quality": quality_gates,
            "contract": contract_gates,
        },
        "contract": {
            "exact_raw_attempts_per_condition": 20,
            "generation_target_access": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
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
    if args.stage == "freeze":
        return run_freeze(args, preregistration)
    if args.stage == "evaluate":
        return run_evaluate(args, preregistration)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
