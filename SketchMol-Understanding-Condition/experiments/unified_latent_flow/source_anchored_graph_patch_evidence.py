#!/usr/bin/env python3
"""Audit a source-anchored set-of-graph-patches representation on train data.

B22 showed that the smallest valid, property-successful intermediate is a much
better target than the complete paired molecule, but its factorized node/edge
decoder still produced chemically unstable graphs.  This evidence stage does
not train or generate molecules.  It reconstructs the locked B22 train-only
trajectory labels and asks whether each strict intermediate can be represented
as at most a few connected graph patches while every node and edge outside the
patch set remains an exact source invariant.

The result decides the B37 decoder state space before any new held-out source
is opened.  B26, B33 fresh sources, Table1 benchmark rows, and official test
rows are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import valid_early_stop_delta_diffusion as b22  # noqa: E402


base = b22.base
belief = b22.belief
delta = b22.delta
full_graph = b22.full_graph
graph = b22.graph
hierarchical = b22.hierarchical

PROTOCOL = "train_only_source_anchored_graph_patch_evidence_v36"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "b22_protocol": b22.PROTOCOL,
        "train_only_representation_audit": True,
        "model_training": False,
        "molecular_candidate_generation": False,
        "evaluation_target_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "future_exact_raw_attempts_per_condition": 20,
        "selected_full_train_pairs": 1451,
        "max_patch_components": 3,
        "max_nodes_per_patch": 12,
        "max_boundary_anchors_per_patch": 2,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B36 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B36 property-count contract drift")
    implementation_sha256 = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "B36 implementation drift: "
            f"expected {payload.get('implementation_sha256')}, "
            f"found {implementation_sha256}"
        )
    if set(dict(payload.get("locked_inputs", {}))) != {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }:
        raise ValueError("B36 locked-input manifest is incomplete")
    return payload


def load_locked_b22(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    inputs = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
    }
    drift = {
        name: {"expected": locked[name], "actual": belief.file_sha256(path)}
        for name, path in inputs.items()
        if belief.file_sha256(path) != locked[name]
    }
    if drift:
        raise ValueError(f"B36 locked input drift: {drift}")

    summary = json.loads(args.b22_summary.read_text(encoding="utf-8"))
    if summary.get("protocol") != b22.PROTOCOL:
        raise ValueError("B36 requires the locked B22 protocol")
    if not bool(dict(summary.get("trajectory_gate", {})).get("passed")):
        raise ValueError("B36 requires B22's passing train-only trajectory gate")
    if summary.get("next_stage") != "replace_independent_delta_tokens_with_local_rewrite_kernel":
        raise ValueError("B36 refuses a B22 decision drift")
    manifest = dict(summary.get("manifest", {}))
    expected_manifest = {
        "train_csv_sha256": locked["train_csv_sha256"],
        "validation_csv_sha256": locked["validation_csv_sha256"],
        "representation_checkpoint_sha256": locked[
            "representation_checkpoint_sha256"
        ],
        "selected_full_train_pairs": int(preregistration["selected_full_train_pairs"]),
        "train_selection_seed": int(preregistration["train_selection_seed"]),
        "validation_selection_seed": int(
            preregistration["validation_selection_seed"]
        ),
        "validation_exclusion_seed": int(
            preregistration["validation_exclusion_seed"]
        ),
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "oracle_reranking": False,
        "exact_raw_attempts_per_condition": 20,
    }
    manifest_drift = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if manifest_drift:
        raise ValueError(f"B36 refuses B22 manifest drift: {manifest_drift}")

    checkpoint = torch.load(args.b22_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != b22.PROTOCOL:
        raise ValueError("B36 refuses a non-B22 checkpoint")
    if dict(checkpoint.get("manifest", {})) != manifest:
        raise ValueError("B36 B22 checkpoint/summary manifest mismatch")
    return summary, checkpoint


def reconstruct_b22_train_pairs(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    checkpoint: Mapping[str, object],
    summary: Mapping[str, object],
) -> tuple[list[object], dict[str, object]]:
    config = dict(checkpoint["model_config"])
    allowed_counts = set(int(value) for value in preregistration["property_counts"])
    validation_rows = base.read_rows(args.validation_csv)
    common = {
        "max_atoms": int(config["max_atoms"]),
        "fingerprint_bits": int(preregistration["fingerprint_bits"]),
        "condition_dim": int(preregistration["condition_dim"]),
        "allowed_counts": allowed_counts,
        "timeout": int(preregistration["mcs_timeout"]),
        "min_common_fraction": float(preregistration["min_common_fraction"]),
        "limit": int(preregistration["historical_validation_limit"]),
    }
    excluded_pairs, _ = base.build_pairs(
        validation_rows,
        seed=int(preregistration["validation_exclusion_seed"]),
        **common,
    )
    excluded_sources = {pair.source_smiles for pair in excluded_pairs}
    excluded_keys = {(pair.source_smiles, pair.target_smiles) for pair in excluded_pairs}
    validation_pairs, _ = base.build_pairs(
        validation_rows,
        seed=int(preregistration["validation_selection_seed"]),
        forbidden_sources=excluded_sources,
        forbidden_pairs=excluded_keys,
        **common,
    )
    validation_sources = {pair.source_smiles for pair in validation_pairs}
    validation_keys = {(pair.source_smiles, pair.target_smiles) for pair in validation_pairs}
    full_train_pairs, filter_counts = base.build_pairs(
        base.read_rows(args.train_csv),
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        condition_dim=int(preregistration["condition_dim"]),
        allowed_counts=allowed_counts,
        timeout=int(preregistration["mcs_timeout"]),
        min_common_fraction=float(preregistration["min_common_fraction"]),
        limit=int(preregistration["train_limit"]),
        seed=int(preregistration["train_selection_seed"]),
        forbidden_sources=validation_sources,
        forbidden_pairs=validation_keys,
    )
    expected_pairs = int(preregistration["selected_full_train_pairs"])
    if len(full_train_pairs) != expected_pairs:
        raise ValueError(
            f"B36 reconstructed {len(full_train_pairs)} B22 pairs, expected {expected_pairs}"
        )
    for pair in full_train_pairs:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(preregistration["condition_dim"])
        )
    trajectory_args = SimpleNamespace(
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        trajectory_fractions=",".join(
            str(value) for value in preregistration["trajectory_fractions"]
        ),
        trajectory_max_orders=int(preregistration["trajectory_max_orders"]),
    )
    selected_pairs, trajectory = b22.select_early_stop_pairs(
        full_train_pairs, checkpoint["vocabulary"], trajectory_args
    )
    locked_trajectory = dict(summary["trajectory_evidence"])
    exact_keys = ("pairs", "early_stop_selected", "strict_selected", "no_op_selected")
    float_keys = (
        "early_stop_coverage",
        "selected_strict_rate",
        "mean_full_actions",
        "mean_selected_actions",
    )
    trajectory_drift = {
        key: {"expected": locked_trajectory[key], "actual": trajectory[key]}
        for key in exact_keys
        if trajectory[key] != locked_trajectory[key]
    }
    trajectory_drift.update(
        {
            key: {"expected": locked_trajectory[key], "actual": trajectory[key]}
            for key in float_keys
            if not math.isclose(
                float(trajectory[key]), float(locked_trajectory[key]), abs_tol=1e-10
            )
        }
    )
    if trajectory_drift:
        raise ValueError(f"B36 B22 trajectory reconstruction drift: {trajectory_drift}")
    return selected_pairs, {
        "filter_counts": filter_counts,
        "historical_excluded_sources": len(excluded_sources),
        "development_excluded_sources": len(validation_sources),
        "trajectory_evidence": trajectory,
    }


def connected_components(changed: Sequence[int], adjacency: np.ndarray) -> list[list[int]]:
    remaining = set(int(index) for index in changed)
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue: deque[int] = deque([seed])
        component = [seed]
        while queue:
            current = queue.popleft()
            for neighbour in np.flatnonzero(adjacency[current]).tolist():
                neighbour = int(neighbour)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


def tensor_state(
    value: Mapping[str, torch.Tensor], fields: Sequence[str], index: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(int(value[field][(0, *index)].item()) for field in fields)


def component_signature(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    component: Sequence[int],
    changed: set[int],
    adjacency: np.ndarray,
) -> str:
    nodes = [
        (
            tensor_state(source, full_graph.NODE_FIELDS, (index,)),
            tensor_state(target, full_graph.NODE_FIELDS, (index,)),
        )
        for index in component
    ]
    internal_edges = []
    for offset, left in enumerate(component):
        for right in component[offset + 1 :]:
            before = tensor_state(source, full_graph.EDGE_FIELDS, (left, right))
            after = tensor_state(target, full_graph.EDGE_FIELDS, (left, right))
            if before != after or adjacency[left, right]:
                internal_edges.append((before, after))
    boundary = []
    for index in component:
        for neighbour in np.flatnonzero(adjacency[index]).tolist():
            neighbour = int(neighbour)
            if neighbour in changed:
                continue
            boundary.append(
                (
                    tensor_state(source, full_graph.NODE_FIELDS, (neighbour,)),
                    tensor_state(source, full_graph.EDGE_FIELDS, (index, neighbour)),
                    tensor_state(target, full_graph.EDGE_FIELDS, (index, neighbour)),
                )
            )
    payload = {
        "nodes": sorted(nodes),
        "internal_edges": sorted(internal_edges),
        "boundary": sorted(boundary),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def pair_record(
    pair: object,
    vocabulary: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    source = graph.collate([pair.source])
    target = graph.collate([pair.target])
    node_actions, edge_actions = delta.delta_action_targets(source, target, vocabulary)
    changed, adjacency = b22.changed_nodes_and_adjacency(
        source, target, node_actions, edge_actions
    )
    components = connected_components(changed, adjacency)
    changed_set = set(changed)
    source_active = source["atomic_number"][0].gt(0)
    target_active = target["atomic_number"][0].gt(0)
    retained = source_active & target_active & node_actions[0].eq(delta.NODE_KEEP)
    retained_fraction = float(retained.sum().item()) / max(1, int(source_active.sum().item()))
    component_sizes = [len(component) for component in components]
    boundary_counts = []
    signatures = []
    for component in components:
        anchors = {
            int(neighbour)
            for index in component
            for neighbour in np.flatnonzero(adjacency[index]).tolist()
            if int(neighbour) not in changed_set
            and bool(source_active[int(neighbour)])
            and bool(target_active[int(neighbour)])
        }
        boundary_counts.append(len(anchors))
        signatures.append(
            component_signature(source, target, component, changed_set, adjacency)
        )

    compact = bool(
        components
        and len(components) <= int(preregistration["max_patch_components"])
        and max(component_sizes, default=0) <= int(preregistration["max_nodes_per_patch"])
        and min(boundary_counts, default=0) >= 1
        and max(boundary_counts, default=0)
        <= int(preregistration["max_boundary_anchors_per_patch"])
        and retained_fraction >= float(preregistration["minimum_retained_atom_fraction"])
    )
    replay = delta.apply_delta_actions(source, node_actions, edge_actions, vocabulary)
    replay_smiles = b22.graph_result_smiles(replay)
    target_smiles = graph.canonical_smiles(pair.target_smiles)
    replay_exact = bool(replay_smiles and replay_smiles == target_smiles)
    outside = sorted(set(range(node_actions.shape[1])) - changed_set)
    outside_nodes_exact = all(
        torch.equal(source[field][0, outside], replay[field][0, outside])
        for field in full_graph.NODE_FIELDS
    )
    if outside:
        outside_tensor = torch.as_tensor(outside, dtype=torch.long)
        outside_edges_exact = all(
            torch.equal(
                source[field][0][outside_tensor[:, None], outside_tensor[None, :]],
                replay[field][0][outside_tensor[:, None], outside_tensor[None, :]],
            )
            for field in full_graph.EDGE_FIELDS
        )
    else:
        outside_edges_exact = True
    fraction, strict, similarity = b22.property_outcome(pair, pair.target_smiles)
    return {
        "source_smiles": pair.source_smiles,
        "target_smiles": pair.target_smiles,
        "task": base.task_key(pair.row),
        "property_count": int(pair.property_count),
        "property_fraction": float(fraction),
        "strict": bool(strict),
        "source_tanimoto": float(similarity),
        "changed_nodes": len(changed),
        "patch_components": len(components),
        "component_sizes": component_sizes,
        "boundary_anchor_counts": boundary_counts,
        "component_signatures": signatures,
        "retained_atom_fraction": retained_fraction,
        "compact_patchable": compact,
        "replay_exact": replay_exact,
        "replay_valid": bool(replay_smiles),
        "outside_source_invariant": bool(outside_nodes_exact and outside_edges_exact),
    }


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    strict = [record for record in records if bool(record["strict"])]
    patchable = [record for record in strict if bool(record["compact_patchable"])]
    signatures = {
        str(signature)
        for record in patchable
        for signature in record["component_signatures"]
    }
    component_histogram = Counter(int(record["patch_components"]) for record in strict)
    by_property_count = {}
    for count in sorted({int(record["property_count"]) for record in records}):
        selected = [record for record in records if int(record["property_count"]) == count]
        selected_strict = [record for record in selected if bool(record["strict"])]
        by_property_count[str(count)] = {
            "pairs": len(selected),
            "strict_pairs": len(selected_strict),
            "selected_strict_rate": len(selected_strict) / max(1, len(selected)),
            "strict_compact_patch_coverage": sum(
                bool(record["compact_patchable"]) for record in selected_strict
            )
            / max(1, len(selected_strict)),
        }
    return {
        "pairs": len(records),
        "selected_strict_pairs": len(strict),
        "selected_strict_rate": len(strict) / max(1, len(records)),
        "strict_compact_patch_pairs": len(patchable),
        "strict_compact_patch_coverage": len(patchable) / max(1, len(strict)),
        "strict_multi_component_rate": sum(
            int(record["patch_components"]) > 1 for record in strict
        )
        / max(1, len(strict)),
        "strict_component_count_histogram": {
            str(key): value for key, value in sorted(component_histogram.items())
        },
        "unique_strict_patch_signatures": len(signatures),
        "exact_replay_rate": sum(bool(record["replay_exact"]) for record in records)
        / max(1, len(records)),
        "valid_replay_rate": sum(bool(record["replay_valid"]) for record in records)
        / max(1, len(records)),
        "outside_source_invariant_rate": sum(
            bool(record["outside_source_invariant"]) for record in records
        )
        / max(1, len(records)),
        "mean_retained_atom_fraction": mean(
            [float(record["retained_atom_fraction"]) for record in strict]
        ),
        "mean_changed_nodes": mean(
            [float(record["changed_nodes"]) for record in strict]
        ),
        "mean_patch_components": mean(
            [float(record["patch_components"]) for record in strict]
        ),
        "by_property_count": by_property_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B36 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    b22_summary, checkpoint = load_locked_b22(args, preregistration)
    selected_pairs, reconstruction = reconstruct_b22_train_pairs(
        args, preregistration, checkpoint, b22_summary
    )
    records = []
    for index, pair in enumerate(selected_pairs, start=1):
        records.append(pair_record(pair, checkpoint["vocabulary"], preregistration))
        if index % 128 == 0 or index == len(selected_pairs):
            print(
                json.dumps(
                    {"stage": "graph_patch_evidence", "pairs": index}, sort_keys=True
                ),
                flush=True,
            )
    metrics = summarize(records)
    gates = dict(preregistration["gates"])
    checks = {
        "selected_strict_rate": {
            "value": metrics["selected_strict_rate"],
            "threshold": gates["selected_strict_rate"],
        },
        "strict_compact_patch_coverage": {
            "value": metrics["strict_compact_patch_coverage"],
            "threshold": gates["strict_compact_patch_coverage"],
        },
        "unique_strict_patch_signatures": {
            "value": metrics["unique_strict_patch_signatures"],
            "threshold": gates["unique_strict_patch_signatures"],
        },
        "exact_replay_rate": {
            "value": metrics["exact_replay_rate"],
            "threshold": gates["exact_replay_rate"],
        },
        "valid_replay_rate": {
            "value": metrics["valid_replay_rate"],
            "threshold": gates["valid_replay_rate"],
        },
        "outside_source_invariant_rate": {
            "value": metrics["outside_source_invariant_rate"],
            "threshold": gates["outside_source_invariant_rate"],
        },
        "mean_retained_atom_fraction": {
            "value": metrics["mean_retained_atom_fraction"],
            "threshold": gates["mean_retained_atom_fraction"],
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if float(item["value"]) < float(item["threshold"])
    ]
    passed = not failures
    set_latent = (
        float(metrics["strict_multi_component_rate"])
        >= float(preregistration["set_latent_trigger"])
    )
    decision = (
        "train_source_anchored_set_graph_patch_flow_v37"
        if passed and set_latent
        else "train_source_anchored_single_graph_patch_flow_v37"
        if passed
        else "stop_graph_patch_representation_after_evidence_gate"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "train_patch_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "protocol": PROTOCOL,
        "implementation_sha256": preregistration["implementation_sha256"],
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "locked_inputs": preregistration["locked_inputs"],
        "b22_protocol": b22_summary.get("protocol"),
        "selected_full_train_pairs": len(selected_pairs),
        "train_only_representation_audit": True,
        "train_target_access_for_supervision": True,
        "train_only_property_oracle_for_trajectory_labels": True,
        "moledit_table1_training_lineage": True,
        "molecular_candidate_generation": False,
        "evaluation_target_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "future_exact_raw_attempts_per_condition": 20,
        "max_patch_components": preregistration["max_patch_components"],
        "max_nodes_per_patch": preregistration["max_nodes_per_patch"],
        "max_boundary_anchors_per_patch": preregistration[
            "max_boundary_anchors_per_patch"
        ],
        "records_sha256": belief.file_sha256(records_path),
        "split_reconstruction": reconstruction,
    }
    result = {
        "protocol": PROTOCOL,
        "manifest": manifest,
        "metrics": metrics,
        "gate": {"passed": passed, "checks": checks, "failures": failures},
        "set_latent_triggered": set_latent,
        "decision": decision,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
