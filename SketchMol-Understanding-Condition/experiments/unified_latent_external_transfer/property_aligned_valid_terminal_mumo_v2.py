#!/usr/bin/env python3
"""Property-aligned valid-terminal graph transport on a sealed MuMO OOD probe.

This experiment is deliberately not another numeric-property adapter.  It
warm-starts B41, replaces the eight native condition slots with an explicit
six-element signed MuMO property set, rebuilds graph-event support from MuMO
train pairs, and jointly updates the condition router, transport, cardinality,
and event field.  Candidate generation receives only source SMILES and the
requested signed set; targets and property oracles are opened after the twenty
raw attempts per condition have been frozen.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
TABLE1_DIR = PROJECT_DIR / "experiments" / "unified_latent_table1"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, UCA_DIR, LATENT_DIR, TABLE1_DIR, SCRIPTS_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import b_series_external_mumo_transfer_v1 as prior  # noqa: E402


PROTOCOL = "property_aligned_valid_terminal_mumo_v2"
OOD_TASKS = prior.OOD_TASKS
PROPERTIES = prior.EXTERNAL_PROPERTIES
SIGNS = prior.EXTERNAL_SIGNS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "trainfreeze", "gate"):
        child = sub.add_parser(name)
        child.add_argument("--preregistration", type=Path, required=True)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--previous-conditions", type=Path, required=True)
    prepare.add_argument("--b22-checkpoint", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=1)

    train = sub.choices["trainfreeze"]
    train.add_argument("--prepare-summary", type=Path, required=True)
    train.add_argument("--fit-pairs", type=Path, required=True)
    train.add_argument("--calibration-pairs", type=Path, required=True)
    train.add_argument("--generation-conditions", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="auto")
    for argument in (
        "train_csv", "validation_csv", "representation_checkpoint",
        "representation_summary", "b22_checkpoint", "b22_summary",
        "b36_summary", "b37_summary", "b38_checkpoint", "b38_summary",
        "b39_checkpoint", "b39_summary", "b39_evaluated_candidates",
        "b40_summary", "b40_evaluated_candidates", "b41_checkpoint",
        "b41_summary", "b41_protocol_manifest", "valid_terminal_summary",
    ):
        train.add_argument("--" + argument.replace("_", "-"), type=Path, required=True)

    gate = sub.choices["gate"]
    gate.add_argument("--prepare-summary", type=Path, required=True)
    gate.add_argument("--freeze-summary", type=Path, required=True)
    gate.add_argument("--candidates", type=Path, required=True)
    gate.add_argument("--evaluation-detail", type=Path, required=True)
    gate.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "amended_after_engineering_failure_before_training_or_candidate_generation",
        "warm_start_b41": True,
        "train_only_graph_state_vocabulary_expansion": True,
        "numeric_adapter": False,
        "signed_property_set_direct_conditioning": True,
        "mumo_train_graph_event_support": True,
        "condition_router_training": True,
        "transport_training": True,
        "event_kernel_training": True,
        "external_task_split": "ood",
        "conditions_per_ood_task": 15,
        "condition_count": 75,
        "exact_raw_attempts_per_condition": 20,
        "candidate_pool_size": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "official_test_access": False,
        "table1_benchmark_access": False,
        "denovo_benchmark_access": False,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"preregistration drift: {drift}")
    if tuple(payload.get("ood_task_ids", ())) != OOD_TASKS:
        raise ValueError("OOD task contract drift")
    implementation = prior.sha256_file(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation:
        raise ValueError(
            f"implementation drift: expected {payload.get('implementation_sha256')}, found {implementation}"
        )
    return payload


def signed_property_tokens(task_id: str, condition_dim: int = 64) -> np.ndarray:
    """One zero global token plus an unordered set of explicit signed tokens."""

    active = set(prior.task_spec(task_id).properties)
    tokens = np.zeros((1 + len(PROPERTIES), int(condition_dim)), dtype=np.float32)
    for index, prop in enumerate(PROPERTIES):
        if prop not in active:
            continue
        # Identity and sign are separate so the representation is injective.
        tokens[1 + index, index] = 1.0
        tokens[1 + index, len(PROPERTIES)] = float(SIGNS[prop])
        tokens[1 + index, len(PROPERTIES) + 1] = 1.0
    return tokens


def _stable_key(seed: int, *values: object) -> str:
    import hashlib

    return hashlib.sha256(
        (str(seed) + ":" + ":".join(map(str, values))).encode("utf-8")
    ).hexdigest()


def _canonical(value: object) -> str:
    from sketchmol_understanding_condition.chem import canonical_smiles

    return canonical_smiles(str(value or "")) or ""


def _load_shards(data_dir: Path) -> tuple[list[dict[str, object]], list[Path]]:
    paths = sorted(data_dir.glob("train_shard_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no MuMO train shards in {data_dir}")
    rows = [row for path in paths for row in prior.read_jsonl(path)]
    return rows, paths


def _dedupe_rows(rows: Sequence[Mapping[str, object]], seed: int) -> list[dict[str, object]]:
    chosen: dict[tuple[str, str, str], tuple[str, dict[str, object]]] = {}
    for raw in rows:
        row = dict(raw)
        task = str(row.get("_uca_task_id", ""))
        source = _canonical(row.get("source_smiles"))
        target = _canonical(row.get("target_smiles"))
        if task not in OOD_TASKS or not source or not target or source == target:
            continue
        key = (task, source, target)
        rank = _stable_key(seed, task, source, target, row.get("_uca_raw_index", ""))
        if key not in chosen or rank < chosen[key][0]:
            row["source_smiles"] = source
            row["target_smiles"] = target
            chosen[key] = (rank, row)
    return [chosen[key][1] for key in sorted(chosen, key=lambda item: chosen[item][0])]


def _select_probe(
    fit_rows: Sequence[Mapping[str, object]],
    forbidden_sources: set[str],
    *,
    per_task: int,
    seed: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used = set(forbidden_sources)
    for task in OOD_TASKS:
        candidates = [
            dict(row)
            for row in fit_rows
            if str(row.get("_uca_task_id")) == task
            and _canonical(row.get("source_smiles")) not in used
        ]
        candidates.sort(
            key=lambda row: _stable_key(seed, task, _canonical(row.get("source_smiles")))
        )
        task_selected = []
        for row in candidates:
            source = _canonical(row.get("source_smiles"))
            if source in used:
                continue
            row["source_smiles"] = source
            task_selected.append(row)
            used.add(source)
            if len(task_selected) == int(per_task):
                break
        if len(task_selected) != int(per_task):
            raise ValueError(f"fresh probe quota unavailable for {task}: {len(task_selected)}")
        selected.extend(task_selected)
    return selected


def _split_calibration_sources(
    rows: Sequence[Mapping[str, object]], *, per_task: int, seed: int
) -> set[str]:
    selected: set[str] = set()
    for task in OOD_TASKS:
        sources = sorted(
            {_canonical(row.get("source_smiles")) for row in rows if row.get("_uca_task_id") == task},
            key=lambda source: _stable_key(seed, task, source),
        )
        selected.update(sources[: int(per_task)])
    return selected


def _materialize_one(payload: tuple[dict[str, object], int, int, float, int]):
    row, max_atoms, fingerprint_bits, min_common_fraction, timeout = payload
    import categorical_graph_latent_flow as base

    source = _canonical(row.get("source_smiles"))
    target = _canonical(row.get("target_smiles"))
    aligned = base.align_pair(
        source,
        target,
        max_atoms=int(max_atoms),
        fingerprint_bits=int(fingerprint_bits),
        timeout=int(timeout),
        min_common_fraction=float(min_common_fraction),
    )
    if aligned is None:
        return None
    source_graph, target_graph, common = aligned
    task = str(row["_uca_task_id"])
    return base.EditPair(
        row={
            "source_smiles": source,
            "target_smiles": target,
            "external_task_id": task,
            "external_task_properties": ",".join(prior.task_spec(task).properties),
        },
        source_smiles=source,
        target_smiles=target,
        source=source_graph,
        target=target_graph,
        condition=signed_property_tokens(task, 64),
        property_count=len(prior.task_spec(task).properties),
        task=task,
        common_atoms=int(common),
    )


def _materialize(
    rows: Sequence[Mapping[str, object]], prereg: Mapping[str, object], workers: int
) -> tuple[list[object], dict[str, object]]:
    payloads = [
        (
            dict(row),
            int(prereg["max_atoms"]),
            int(prereg["fingerprint_bits"]),
            float(prereg["min_common_fraction"]),
            int(prereg["mcs_timeout"]),
        )
        for row in rows
    ]
    if int(workers) == 1:
        values = [_materialize_one(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            values = list(pool.map(_materialize_one, payloads, chunksize=8))
    pairs = [value for value in values if value is not None]
    by_task = Counter(pair.task for pair in pairs)
    return pairs, {
        "input_rows": len(rows),
        "aligned_pairs": len(pairs),
        "alignment_rejected": len(rows) - len(pairs),
        "by_task": dict(sorted(by_task.items())),
    }


def _event_signatures(pairs: Sequence[object], vocabulary: Mapping[str, object]):
    import torch
    import categorical_graph_latent_flow as base
    import source_relative_delta_diffusion as delta

    signatures: list[tuple[object, ...]] = []
    counts: list[int] = []
    for pair in pairs:
        source = base.graph.collate([pair.source])
        target = base.graph.collate([pair.target])
        node_targets, edge_targets = delta.delta_action_targets(source, target, vocabulary)
        node = node_targets[0]
        edge = edge_targets[0]
        source_atoms = source["atomic_number"][0]
        source_bonds = source["bond"][0]
        local: list[tuple[object, ...]] = []
        for index in torch.nonzero(node.ne(delta.NODE_KEEP), as_tuple=False).flatten().tolist():
            action = int(node[index])
            if action == delta.NODE_DELETE:
                local.append(("node_delete", int(source_atoms[index])))
            else:
                local.append(("node_write", action))
        nodes = edge.shape[0]
        for left in range(nodes - 1):
            for right in range(left + 1, nodes):
                action = int(edge[left, right])
                if action == delta.EDGE_KEEP:
                    continue
                if action == delta.EDGE_DELETE:
                    local.append(("edge_delete", int(source_bonds[left, right])))
                else:
                    local.append(("edge_set", action))
        signatures.extend(local)
        counts.append(len(local))
    return signatures, counts


def _expanded_vocabulary(
    old_vocabulary: Mapping[str, object], fit_pairs: Sequence[object]
) -> dict[str, object]:
    """Append MuMO-train-only states while preserving every B22 action index."""

    import hashlib
    import json as json_module
    import discrete_graph_diffusion_decoder as full_graph

    observed = full_graph.build_joint_state_vocabulary(fit_pairs)
    old_nodes = [tuple(map(int, row)) for row in np.asarray(old_vocabulary["node_states"])]
    old_edges = [tuple(map(int, row)) for row in np.asarray(old_vocabulary["edge_states"])]
    observed_nodes = {
        tuple(map(int, row)) for row in np.asarray(observed["node_states"])
    }
    observed_edges = {
        tuple(map(int, row)) for row in np.asarray(observed["edge_states"])
    }
    nodes = [*old_nodes, *sorted(observed_nodes - set(old_nodes))]
    edges = [*old_edges, *sorted(observed_edges - set(old_edges))]
    if nodes[: len(old_nodes)] != old_nodes or edges[: len(old_edges)] != old_edges:
        raise AssertionError("B22 action indices were not preserved during vocabulary expansion")
    payload = json_module.dumps(
        {"node_states": nodes, "edge_states": edges}, separators=(",", ":")
    ).encode("utf-8")
    return {
        "node_states": np.asarray(nodes, dtype=np.int64),
        "edge_states": np.asarray(edges, dtype=np.int64),
        "blank_node_id": int(old_vocabulary["blank_node_id"]),
        "blank_edge_id": int(old_vocabulary["blank_edge_id"]),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "old_node_state_count": len(old_nodes),
        "old_edge_state_count": len(old_edges),
        "added_node_state_count": len(nodes) - len(old_nodes),
        "added_edge_state_count": len(edges) - len(old_edges),
    }


def _load_expanded_b41_state(
    model: object,
    old_state: Mapping[str, object],
    old_vocabulary: Mapping[str, object],
    vocabulary: Mapping[str, object],
) -> dict[str, object]:
    """Transplant B41 weights; new train-only actions keep fresh initialization."""

    current = model.state_dict()
    expanded_keys: dict[str, dict[str, object]] = {}
    for key, target in current.items():
        if key not in old_state:
            raise ValueError(f"B41 warm-start is missing parameter {key}")
        source = old_state[key]
        if tuple(source.shape) == tuple(target.shape):
            current[key] = source
            continue
        if (
            key
            not in {
                "denoiser.node_embedding.weight",
                "denoiser.edge_embedding.weight",
                "denoiser.node_write_head.1.weight",
                "denoiser.node_write_head.1.bias",
                "denoiser.edge_set_head.weight",
                "denoiser.edge_set_head.bias",
            }
            or source.ndim != target.ndim
            or tuple(source.shape[1:]) != tuple(target.shape[1:])
            or source.shape[0] >= target.shape[0]
        ):
            raise ValueError(
                f"unexpected B41 expansion shape for {key}: {tuple(source.shape)} -> {tuple(target.shape)}"
            )
        transplanted = target.clone()
        transplanted[: source.shape[0]] = source
        current[key] = transplanted
        expanded_keys[key] = {
            "old_rows": int(source.shape[0]),
            "new_rows": int(target.shape[0]),
        }
    model.load_state_dict(current, strict=True)
    expected_node_additions = int(vocabulary["added_node_state_count"])
    expected_edge_additions = int(vocabulary["added_edge_state_count"])
    if expected_node_additions and "denoiser.node_embedding.weight" not in expanded_keys:
        raise ValueError("node vocabulary expanded without expanding the node action field")
    if expected_edge_additions and "denoiser.edge_embedding.weight" not in expanded_keys:
        raise ValueError("edge vocabulary expanded without expanding the edge action field")
    return {
        "old_node_state_count": len(old_vocabulary["node_states"]),
        "new_node_state_count": len(vocabulary["node_states"]),
        "old_edge_state_count": len(old_vocabulary["edge_states"]),
        "new_edge_state_count": len(vocabulary["edge_states"]),
        "expanded_parameters": expanded_keys,
        "old_action_indices_preserved": True,
    }


def _support_audit(
    fit_pairs: Sequence[object],
    calibration_pairs: Sequence[object],
    b22_checkpoint: Path,
    prereg: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    import torch
    import eval_d0_b41_table1 as d0

    b41, b40, b39, delta = d0.b41, d0.b40, d0.b39, d0.delta
    checkpoint = torch.load(b22_checkpoint, map_location="cpu", weights_only=False)
    old_vocabulary = d0.b37.checkpoint_vocabulary(checkpoint)
    vocabulary = _expanded_vocabulary(old_vocabulary, fit_pairs)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, torch.device("cpu"))
    node_actions, edge_actions = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=16,
        edge_dim=16,
        condition_dim=64,
        transport_dim=16,
        hidden_dim=32,
        max_atoms=int(prereg["max_atoms"]),
        max_jumps=int(prereg["max_jumps"]),
        property_count=len(PROPERTIES),
        node_state_count=node_actions,
        edge_state_count=edge_actions,
        message_layers=1,
    )
    replay = b41.support_replay_gate(
        model,
        calibration_pairs,
        vocabulary,
        support,
        support_tensors,
        prereg,
        torch.device("cpu"),
    )
    fit_signatures, fit_counts = _event_signatures(fit_pairs, vocabulary)
    calibration_signatures, calibration_counts = _event_signatures(calibration_pairs, vocabulary)
    fit_set = set(fit_signatures)
    covered = sum(signature in fit_set for signature in calibration_signatures)
    coverage = covered / max(1, len(calibration_signatures))
    fit_task_counts = Counter(pair.task for pair in fit_pairs)
    checks = {
        "fit_pairs": len(fit_pairs) >= int(prereg["support_gates"]["fit_pairs"]),
        "calibration_pairs": len(calibration_pairs) >= int(prereg["support_gates"]["calibration_pairs"]),
        "minimum_fit_pairs_per_task": min(fit_task_counts.get(task, 0) for task in OOD_TASKS)
        >= int(prereg["support_gates"]["fit_pairs_per_task"]),
        "unique_event_signatures": len(fit_set)
        >= int(prereg["support_gates"]["unique_event_signatures"]),
        "calibration_event_coverage": coverage
        >= float(prereg["support_gates"]["calibration_event_coverage"]),
        "calibration_horizon_coverage": float(np.mean(np.asarray(calibration_counts) <= int(prereg["max_jumps"])))
        >= float(prereg["support_gates"]["calibration_horizon_coverage"]),
        "terminal_support_replay": bool(replay["passed"]),
    }
    audit = {
        "fit_event_instances": len(fit_signatures),
        "calibration_event_instances": len(calibration_signatures),
        "unique_fit_event_signatures": len(fit_set),
        "calibration_event_signature_coverage": coverage,
        "max_fit_target_events": max(fit_counts, default=0),
        "max_calibration_target_events": max(calibration_counts, default=0),
        "terminal_support_replay": replay,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "passed": all(checks.values()),
    }
    audit["vocabulary"] = {
        "sha256": vocabulary["sha256"],
        "node_state_count": len(vocabulary["node_states"]),
        "edge_state_count": len(vocabulary["edge_states"]),
        "added_node_state_count": vocabulary["added_node_state_count"],
        "added_edge_state_count": vocabulary["added_edge_state_count"],
        "old_action_indices_preserved": True,
    }
    return audit, vocabulary


def prepare(args: argparse.Namespace, prereg: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, shards = _load_shards(args.data_dir)
    locked = dict(prereg["locked_inputs"])
    shard_digest = prior.aggregate_shard_digest(shards)
    if shard_digest != locked["mumo_v8_shards_aggregate_sha256"]:
        raise ValueError(f"MuMO shard drift: {shard_digest}")
    previous_digest = prior.sha256_file(args.previous_conditions)
    if previous_digest != locked["previous_opened_conditions_sha256"]:
        raise ValueError(f"previous opened-condition drift: {previous_digest}")
    b22_digest = prior.sha256_file(args.b22_checkpoint)
    if b22_digest != locked["b22_checkpoint_sha256"]:
        raise ValueError(f"B22 checkpoint drift: {b22_digest}")
    dev_sources = {
        _canonical(row.get("source_smiles"))
        for row in rows
        if str(row.get("_uca_partition", "")) == "dev"
    }
    previous_sources = {
        _canonical(row.get("source_smiles")) for row in prior.read_jsonl(args.previous_conditions)
    }
    fit_rows = _dedupe_rows(
        [row for row in rows if str(row.get("_uca_partition", "")) == "fit"],
        int(prereg["pair_selection_seed"]),
    )
    probe = _select_probe(
        fit_rows,
        dev_sources | previous_sources,
        per_task=int(prereg["conditions_per_ood_task"]),
        seed=int(prereg["probe_selection_seed"]),
    )
    probe_sources = {_canonical(row["source_smiles"]) for row in probe}
    eligible = [row for row in fit_rows if _canonical(row["source_smiles"]) not in probe_sources]
    calibration_sources = _split_calibration_sources(
        eligible,
        per_task=int(prereg["calibration_sources_per_task"]),
        seed=int(prereg["calibration_split_seed"]),
    )
    calibration_rows = [row for row in eligible if _canonical(row["source_smiles"]) in calibration_sources]
    training_rows = [row for row in eligible if _canonical(row["source_smiles"]) not in calibration_sources]
    fit_pairs, fit_stats = _materialize(training_rows, prereg, int(args.workers))
    calibration_pairs, calibration_stats = _materialize(
        calibration_rows, prereg, int(args.workers)
    )
    with (args.output_dir / "fit_pairs.pkl").open("wb") as handle:
        pickle.dump(fit_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.output_dir / "calibration_pairs.pkl").open("wb") as handle:
        pickle.dump(calibration_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    generation = [prior.generation_condition(row) for row in probe]
    sealed = [
        {
            "condition_id": generation[index]["condition_id"],
            "external_task_id": row["_uca_task_id"],
            "source_smiles": row["source_smiles"],
            "target_smiles": row["target_smiles"],
            "pair_digest": row.get("_uca_pair_digest", ""),
        }
        for index, row in enumerate(probe)
    ]
    prior.write_jsonl(args.output_dir / "generation_conditions.jsonl", generation)
    prior.write_jsonl(args.output_dir / "sealed_probe_targets.jsonl", sealed)
    audit, vocabulary = _support_audit(
        fit_pairs, calibration_pairs, args.b22_checkpoint, prereg
    )
    vocabulary_path = args.output_dir / "expanded_vocabulary.json"
    prior.write_json(
        vocabulary_path,
        {
            "node_states": np.asarray(vocabulary["node_states"]).tolist(),
            "edge_states": np.asarray(vocabulary["edge_states"]).tolist(),
            "blank_node_id": int(vocabulary["blank_node_id"]),
            "blank_edge_id": int(vocabulary["blank_edge_id"]),
            "sha256": vocabulary["sha256"],
            "old_node_state_count": vocabulary["old_node_state_count"],
            "old_edge_state_count": vocabulary["old_edge_state_count"],
            "added_node_state_count": vocabulary["added_node_state_count"],
            "added_edge_state_count": vocabulary["added_edge_state_count"],
        },
    )
    source_overlap = len(
        {pair.source_smiles for pair in fit_pairs}
        & {pair.source_smiles for pair in calibration_pairs}
    )
    probe_train_overlap = len(
        probe_sources
        & ({pair.source_smiles for pair in fit_pairs} | {pair.source_smiles for pair in calibration_pairs})
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare_and_train_only_support_audit",
        "mumo_rows": len(rows),
        "shard_aggregate_sha256": shard_digest,
        "fit_materialization": fit_stats,
        "calibration_materialization": calibration_stats,
        "probe_conditions": len(generation),
        "probe_task_counts": dict(sorted(Counter(row["external_task_id"] for row in generation).items())),
        "probe_unique_sources": len(probe_sources),
        "fit_calibration_source_overlap": source_overlap,
        "probe_train_source_overlap": probe_train_overlap,
        "previous_opened_source_overlap": len(probe_sources & previous_sources),
        "original_dev_source_overlap": len(probe_sources & dev_sources),
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "support_audit_used_probe_targets": False,
        "support_audit": audit,
        "fit_pairs_sha256": prior.sha256_file(args.output_dir / "fit_pairs.pkl"),
        "calibration_pairs_sha256": prior.sha256_file(args.output_dir / "calibration_pairs.pkl"),
        "expanded_vocabulary_sha256": prior.sha256_file(vocabulary_path),
        "generation_conditions_sha256": prior.sha256_file(args.output_dir / "generation_conditions.jsonl"),
        "sealed_probe_targets_sha256": prior.sha256_file(args.output_dir / "sealed_probe_targets.jsonl"),
    }
    if any((source_overlap, probe_train_overlap, summary["previous_opened_source_overlap"], summary["original_dev_source_overlap"])):
        raise ValueError(f"source isolation failure: {summary}")
    prior.write_json(args.output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _b41_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        representation_checkpoint=args.representation_checkpoint,
        representation_summary=args.representation_summary,
        b22_checkpoint=args.b22_checkpoint,
        b22_summary=args.b22_summary,
        b36_summary=args.b36_summary,
        b37_summary=args.b37_summary,
        b38_checkpoint=args.b38_checkpoint,
        b38_summary=args.b38_summary,
        b39_checkpoint=args.b39_checkpoint,
        b39_summary=args.b39_summary,
        b39_evaluated_candidates=args.b39_evaluated_candidates,
        b40_summary=args.b40_summary,
        b40_evaluated_candidates=args.b40_evaluated_candidates,
    )


def trainfreeze(args: argparse.Namespace, prereg: Mapping[str, object]) -> int:
    import torch
    import eval_d0_b41_table1 as d0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_summary = json.loads(args.prepare_summary.read_text(encoding="utf-8"))
    if prepare_summary.get("protocol") != PROTOCOL:
        raise ValueError("prepare protocol drift")
    if not bool(prepare_summary["support_audit"]["passed"]):
        summary = {
            "protocol": PROTOCOL,
            "stage": "trainfreeze",
            "execution_skipped": True,
            "reason": "train_only_support_gate_failed",
            "support_failures": prepare_summary["support_audit"]["failures"],
        }
        prior.write_json(args.output_dir / "freeze_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    locks = dict(prereg["locked_inputs"])
    for name, path in {
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_summary_sha256": args.b41_summary,
        "valid_terminal_summary_sha256": args.valid_terminal_summary,
    }.items():
        actual = prior.sha256_file(path)
        if actual != locks[name]:
            raise ValueError(f"locked B artifact drift for {name}: {actual}")
    if prior.sha256_file(args.fit_pairs) != prepare_summary["fit_pairs_sha256"]:
        raise ValueError("fit-pair artifact drift")
    if prior.sha256_file(args.calibration_pairs) != prepare_summary["calibration_pairs_sha256"]:
        raise ValueError("calibration-pair artifact drift")
    if prior.sha256_file(args.generation_conditions) != prepare_summary["generation_conditions_sha256"]:
        raise ValueError("generation condition artifact drift")
    with args.fit_pairs.open("rb") as handle:
        fit_pairs = pickle.load(handle)
    with args.calibration_pairs.open("rb") as handle:
        calibration_pairs = pickle.load(handle)
    conditions = prior.read_jsonl(args.generation_conditions)
    b41, b40, b39, base, graph = d0.b41, d0.b40, d0.b39, d0.base, d0.graph
    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    namespace = _b41_namespace(args)
    (_b22, b22_checkpoint, _b36, _b37, _b39, _b40) = b41.check_locked_inputs(
        namespace, b41_prereg
    )
    device = base.resolve_device(str(args.device))
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    old_vocabulary = d0.b37.checkpoint_vocabulary(b22_checkpoint)
    vocabulary = _expanded_vocabulary(old_vocabulary, fit_pairs)
    if vocabulary["sha256"] != prepare_summary["support_audit"]["vocabulary"]["sha256"]:
        raise ValueError("expanded train-only vocabulary drift")
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_actions, edge_actions = d0.delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(prereg["condition_dim"]),
        transport_dim=int(prereg["transport_dim"]),
        hidden_dim=int(prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(prereg["max_jumps"]),
        property_count=len(PROPERTIES),
        node_state_count=node_actions,
        edge_state_count=edge_actions,
        message_layers=int(prereg["message_layers"]),
    ).to(device)
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)
    warm_start = _load_expanded_b41_state(
        model,
        dict(b41_checkpoint["model_state"]),
        old_vocabulary,
        vocabulary,
    )
    training_config = {**dict(b41_prereg), **dict(prereg)}
    history = b39.train_model(
        model, representation, fit_pairs, vocabulary, training_config, device
    )
    replay = b41.support_replay_gate(
        model,
        calibration_pairs,
        vocabulary,
        support,
        support_tensors,
        training_config,
        device,
    )
    event_config = {**training_config, "epochs": int(prereg["event_finetune_epochs"])}
    event_history = b41.fine_tune_event_kernel(
        model,
        representation,
        fit_pairs,
        vocabulary,
        support,
        support_tensors,
        event_config,
        device,
    )
    checkpoint_path = args.output_dir / "property_aligned_valid_terminal_mumo.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "model_state": model.state_dict(),
            "property_order": PROPERTIES,
            "signed_property_tokens": True,
            "warm_start_b41_sha256": locks["b41_checkpoint_sha256"],
            "fit_pairs_sha256": prepare_summary["fit_pairs_sha256"],
        },
        checkpoint_path,
    )
    exact_support = d0.valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_support = b41.viability_event_mask
    rows: list[dict[str, object]] = []
    try:
        b41.viability_event_mask = exact_support
        for index, condition in enumerate(conditions):
            source = _canonical(condition["source_smiles"])
            source_graph = graph.molecule_example(
                source,
                max_atoms=int(representation_config["max_atoms"]),
                fingerprint_bits=int(prereg["fingerprint_bits"]),
            )
            if source_graph is None:
                raise ValueError(f"probe source not representable: {condition['condition_id']}")
            tokens = signed_property_tokens(
                str(condition["external_task_id"]), int(prereg["condition_dim"])
            )
            generated = b41.sample_from_source(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
                source_graph,
                tokens,
                training_config,
                device,
                int(prereg["generation_seed"]) * 100_000 + index,
            )
            if len(generated) != int(prereg["exact_raw_attempts_per_condition"]):
                raise ValueError(f"attempt count drift for {condition['condition_id']}")
            for attempt, candidate in enumerate(generated, start=1):
                rows.append(
                    {
                        **dict(condition),
                        "generated_smiles": str(candidate.get("generated_smiles", "")),
                        "sample_index": attempt,
                        "candidate_index": attempt,
                        "candidate_rank": attempt,
                        "candidate_selected": True,
                        "method": PROTOCOL,
                        "family": "property_aligned_valid_terminal_graph_transport",
                        "numeric_adapter": False,
                    }
                )
            if (index + 1) % 5 == 0 or index + 1 == len(conditions):
                print(json.dumps({"stage": "frozen_generation", "done": index + 1, "total": len(conditions)}, sort_keys=True), flush=True)
    finally:
        b41.viability_event_mask = original_support
    expected = int(prereg["condition_count"]) * int(prereg["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise ValueError(f"candidate row drift: {len(rows)} != {expected}")
    candidates_path = args.output_dir / "frozen_candidates.csv"
    prior.write_csv(candidates_path, rows)
    summary = {
        "protocol": PROTOCOL,
        "stage": "train_and_freeze",
        "execution_skipped": False,
        "device": str(device),
        "fit_pairs": len(fit_pairs),
        "calibration_pairs": len(calibration_pairs),
        "training": history,
        "event_finetuning": event_history,
        "calibration_support_replay": replay,
        "train_only_vocabulary": prepare_summary["support_audit"]["vocabulary"],
        "warm_start_transplant": warm_start,
        "candidate_rows": len(rows),
        "attempts_per_condition": int(prereg["exact_raw_attempts_per_condition"]),
        "checkpoint_sha256": prior.sha256_file(checkpoint_path),
        "candidates_sha256": prior.sha256_file(candidates_path),
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "exact_molecule_stop_support": exact_support.manifest(),
    }
    prior.write_json(args.output_dir / "freeze_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def gate(args: argparse.Namespace, prereg: Mapping[str, object]) -> int:
    prepare_summary = json.loads(args.prepare_summary.read_text(encoding="utf-8"))
    freeze_summary = json.loads(args.freeze_summary.read_text(encoding="utf-8"))
    if freeze_summary.get("execution_skipped") is True:
        result = {
            "protocol": PROTOCOL,
            "decision": "stop_before_accelerator_or_oracle_due_to_train_only_support_gate",
            "support_audit": prepare_summary["support_audit"],
            "internal_gate": {"passed": False, "failures": prepare_summary["support_audit"]["failures"]},
        }
        prior.write_json(args.output_json, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if prior.sha256_file(args.candidates) != freeze_summary["candidates_sha256"]:
        raise ValueError("candidate artifact drift")
    metrics = prior.aggregate_evaluation(prior.read_csv(args.evaluation_detail))
    thresholds = dict(prereg["external_gates"])
    checks = {
        "conditions": {"value": metrics["conditions"], "threshold": int(prereg["condition_count"])},
        "candidate_rows": {"value": metrics["candidate_rows"], "threshold": int(prereg["condition_count"]) * 20},
        "validity": {"value": metrics["validity"], "threshold": thresholds["validity"]},
        "property_any20": {"value": metrics["property_any20"], "threshold": thresholds["property_any20"]},
        "strict_any20": {"value": metrics["strict_any20"], "threshold": thresholds["strict_any20"]},
        "support_ceiling": {"value": metrics["support_ceiling"], "threshold": thresholds["support_ceiling"]},
        "mean_source_tanimoto": {"value": metrics["mean_source_tanimoto"], "threshold": thresholds["mean_source_tanimoto"]},
        "mean_unique_valid": {"value": metrics["mean_unique_valid"], "threshold": thresholds["mean_unique_valid"]},
        "fit_calibration_source_overlap": {"value": prepare_summary["fit_calibration_source_overlap"], "threshold": 0, "comparison": "at_most"},
        "probe_train_source_overlap": {"value": prepare_summary["probe_train_source_overlap"], "threshold": 0, "comparison": "at_most"},
    }
    failures = []
    for name, check in checks.items():
        if check.get("comparison") == "at_most":
            failed = float(check["value"]) > float(check["threshold"])
        else:
            failed = float(check["value"]) < float(check["threshold"])
        if failed:
            failures.append(name)
    result = {
        "protocol": PROTOCOL,
        "decision": "advance_to_three_benchmark_pilots" if not failures else "stop_b_method_iteration_and_write_external_support_diagnostic",
        "metrics": metrics,
        "reference_results": prereg["reference_results"],
        "internal_gate": {"checks": checks, "failures": failures, "passed": not failures},
        "contracts": {
            "exact_n20": True,
            "generation_target_access": False,
            "generation_property_oracle_access": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
            "official_test_access": False,
        },
    }
    prior.write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    # A scientific STOP is a valid completed experiment, not an execution error.
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prereg = read_preregistration(args.preregistration)
    if args.stage == "prepare":
        return prepare(args, prereg)
    if args.stage == "trainfreeze":
        return trainfreeze(args, prereg)
    if args.stage == "gate":
        return gate(args, prereg)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
