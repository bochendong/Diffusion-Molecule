#!/usr/bin/env python3
"""Train source-relative delta diffusion on valid early-stop trajectories.

B21 learned the edit size of the full supervised target, but successful valid
candidates used substantially smaller deltas.  B22 changes only the train-time
endpoint distribution.  For every selected train pair it constructs connected
prefixes of the aligned source-to-target delta, materializes them on the source,
and keeps the earliest prefix that is chemically valid and already satisfies
the requested properties.  All prefix construction, validity checks, and
property labels are train-only.

Generation is exactly the source-relative absorbing delta diffusion from B21:
it receives source, sanitized property tokens, and noise, then emits exactly 20
raw candidates.  It performs no RDKit validation, repair, candidate filtering,
ranking, finalization, or oracle access before those candidates are frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import source_relative_delta_diffusion as delta


base = delta.base
belief = delta.belief
full_graph = delta.full_graph
graph = delta.graph
hierarchical = delta.hierarchical
unified = delta.unified

PROTOCOL = "train_only_valid_early_stop_delta_diffusion_pilot_v22"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--transport-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-loss-weight", type=float, default=0.50)
    parser.add_argument("--latent-usage-weight", type=float, default=0.20)
    parser.add_argument("--latent-usage-margin", type=float, default=0.10)
    parser.add_argument("--latent-variance-weight", type=float, default=0.10)
    parser.add_argument("--latent-min-std", type=float, default=0.20)
    parser.add_argument("--latent-noise-scale", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--birth-capacity", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.75)
    parser.add_argument("--trajectory-fractions", default="0.25,0.50,0.75,1.0")
    parser.add_argument("--trajectory-max-orders", type=int, default=4)
    parser.add_argument("--gate-early-stop-coverage", type=float, default=0.20)
    parser.add_argument("--gate-selected-strict-rate", type=float, default=0.80)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--gate-validity", type=float, default=0.95)
    parser.add_argument("--gate-source-tanimoto", type=float, default=0.40)
    parser.add_argument("--gate-strict-any20", type=float, default=0.25)
    parser.add_argument("--gate-3p-strict-any20", type=float, default=0.20)
    parser.add_argument("--gate-mean-unique-valid", type=float, default=10.0)
    parser.add_argument("--validation-selection-seed", type=int, default=2719)
    parser.add_argument("--validation-exclusion-seed", type=int, default=1742)
    parser.add_argument("--train-selection-seed", type=int, default=1741)
    parser.add_argument("--seed", type=int, default=1757)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def parse_fractions(value: str) -> list[float]:
    fractions = sorted(
        {float(part.strip()) for part in str(value).split(",") if part.strip()}
    )
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("trajectory-fractions must be in (0, 1]")
    if 1.0 not in fractions:
        fractions.append(1.0)
    return fractions


def changed_nodes_and_adjacency(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
) -> tuple[list[int], np.ndarray]:
    node_changed = node_actions[0].ne(delta.NODE_KEEP)
    edge_changed = edge_actions[0].ne(delta.EDGE_KEEP)
    edge_endpoints = edge_changed.any(dim=0) | edge_changed.any(dim=1)
    changed = torch.nonzero(node_changed | edge_endpoints, as_tuple=False).flatten().tolist()
    adjacency = (
        source["bond"][0].gt(graph.BOND_NONE)
        | target["bond"][0].gt(graph.BOND_NONE)
    ).cpu().numpy()
    return [int(index) for index in changed], np.asarray(adjacency, dtype=bool)


def bfs_changed_order(
    changed: Sequence[int], adjacency: np.ndarray, seed: int
) -> list[int]:
    if not changed:
        return []
    changed_set = set(int(index) for index in changed)
    distance = {int(seed): 0}
    queue: deque[int] = deque([int(seed)])
    while queue:
        current = queue.popleft()
        for neighbour in np.flatnonzero(adjacency[current]).tolist():
            neighbour = int(neighbour)
            if neighbour not in distance:
                distance[neighbour] = distance[current] + 1
                queue.append(neighbour)
    unreachable = adjacency.shape[0] + 1
    return sorted(changed_set, key=lambda index: (distance.get(index, unreachable), index))


def local_changed_orders(
    changed: Sequence[int], adjacency: np.ndarray, max_orders: int
) -> list[list[int]]:
    if not changed:
        return [[]]
    ordered_changed = sorted(set(int(index) for index in changed))
    source_degrees = adjacency.sum(axis=1)
    seeds = sorted(
        ordered_changed,
        key=lambda index: (-int(source_degrees[index]), index),
    )[: max(1, int(max_orders))]
    seeds.extend([ordered_changed[0], ordered_changed[-1]])
    orders: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        order = bfs_changed_order(ordered_changed, adjacency, int(seed))
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            orders.append(order)
        if len(orders) >= max(1, int(max_orders)):
            break
    return orders


def prefix_actions(
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    selected_nodes: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_node = torch.full_like(node_actions, delta.NODE_KEEP)
    prefix_edge = torch.full_like(edge_actions, delta.EDGE_KEEP)
    if not selected_nodes:
        return prefix_node, prefix_edge
    selected = torch.zeros_like(node_actions, dtype=torch.bool)
    selected[:, sorted(selected_nodes)] = True
    prefix_node = torch.where(selected, node_actions, prefix_node)
    incident = selected[:, :, None] | selected[:, None, :]
    prefix_edge = torch.where(incident, edge_actions, prefix_edge)
    return prefix_node, prefix_edge


def graph_result_smiles(result: Mapping[str, torch.Tensor]) -> str:
    prediction = {
        field: result[field].detach().cpu().numpy()
        for field in (*full_graph.NODE_FIELDS, *full_graph.EDGE_FIELDS)
    }
    smiles, _ = graph.graph_to_smiles(prediction, 0)
    return graph.canonical_smiles(smiles or "")


def action_count(node_actions: torch.Tensor, edge_actions: torch.Tensor) -> int:
    nodes = int(node_actions.ne(delta.NODE_KEEP).sum())
    upper = torch.triu(
        torch.ones(edge_actions.shape[1:], dtype=torch.bool), diagonal=1
    ).unsqueeze(0)
    edges = int((edge_actions.ne(delta.EDGE_KEEP).cpu() & upper).sum())
    return nodes + edges


def property_outcome(pair: object, smiles: str) -> tuple[float, bool, float]:
    specs = base.task_specs(pair.row)
    fraction, _, _, success = unified.instruction_success_and_distance(
        pair.row, smiles, task_specs=specs
    )
    similarity = graph.morgan_tanimoto(pair.source_smiles, smiles) or 0.0
    return float(fraction), bool(success and similarity >= 0.4), float(similarity)


def aligned_variant(
    pair: object,
    result: Mapping[str, torch.Tensor],
    smiles: str,
    args: argparse.Namespace,
) -> object | None:
    fingerprint = graph.morgan_fingerprint_bits(
        smiles, radius=2, n_bits=int(args.fingerprint_bits)
    )
    if fingerprint is None:
        return None
    arrays = {
        field: result[field][0].detach().cpu().numpy().copy()
        for field in (*full_graph.NODE_FIELDS, *full_graph.EDGE_FIELDS)
    }
    node_mask = (arrays["atomic_number"] > 0).astype(np.float32)
    target = graph.GraphExample(
        smiles,
        arrays["atomic_number"],
        arrays["formal_charge"],
        arrays["chirality"],
        arrays["aromatic"],
        arrays["explicit_hs"],
        arrays["no_implicit"],
        arrays["bond"],
        arrays["bond_stereo"],
        node_mask,
        np.asarray(fingerprint, dtype=np.float32),
    )
    variant = copy.copy(pair)
    variant.target = target
    variant.target_smiles = target.smiles
    variant.common_atoms = int(
        np.logical_and(pair.source.node_mask > 0, node_mask > 0).sum()
    )
    return variant


def valid_prefix_candidates(
    pair: object,
    vocabulary: Mapping[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], int]:
    source = graph.collate([pair.source])
    target = graph.collate([pair.target])
    node_actions, edge_actions = delta.delta_action_targets(source, target, vocabulary)
    full_count = action_count(node_actions, edge_actions)
    changed, adjacency = changed_nodes_and_adjacency(
        source, target, node_actions, edge_actions
    )
    orders = local_changed_orders(changed, adjacency, int(args.trajectory_max_orders))
    fractions = parse_fractions(str(args.trajectory_fractions))
    candidates: dict[str, dict[str, object]] = {}

    source_fraction, source_strict, source_similarity = property_outcome(
        pair, pair.source_smiles
    )
    source_variant = copy.copy(pair)
    source_variant.target = pair.source
    source_variant.target_smiles = pair.source_smiles
    source_variant.common_atoms = int(pair.source.node_mask.sum())
    candidates[pair.source_smiles] = {
        "smiles": pair.source_smiles,
        "fraction": source_fraction,
        "strict": source_strict,
        "similarity": source_similarity,
        "actions": 0,
        "strength": 0.0,
        "variant": source_variant,
    }
    for order in orders:
        for strength in fractions:
            count = min(len(order), max(1, int(math.ceil(len(order) * strength))))
            prefix_node, prefix_edge = prefix_actions(
                node_actions, edge_actions, set(order[:count])
            )
            result = delta.apply_delta_actions(
                source, prefix_node, prefix_edge, vocabulary
            )
            smiles = graph_result_smiles(result)
            if not smiles or smiles in candidates:
                continue
            variant = aligned_variant(pair, result, smiles, args)
            if variant is None:
                continue
            fraction, strict, similarity = property_outcome(pair, smiles)
            candidates[smiles] = {
                "smiles": smiles,
                "fraction": fraction,
                "strict": strict,
                "similarity": similarity,
                "actions": action_count(prefix_node, prefix_edge),
                "strength": float(strength),
                "variant": variant,
            }
    # The original aligned target is valid by dataset construction and is the
    # fallback even if a reconstructed full-prefix canonicalization was deduped.
    target_fraction, target_strict, target_similarity = property_outcome(
        pair, pair.target_smiles
    )
    candidates.setdefault(
        pair.target_smiles,
        {
            "smiles": pair.target_smiles,
            "fraction": target_fraction,
            "strict": target_strict,
            "similarity": target_similarity,
            "actions": full_count,
            "strength": 1.0,
            "variant": copy.copy(pair),
        },
    )
    return list(candidates.values()), full_count


def select_early_stop_pairs(
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    args: argparse.Namespace,
) -> tuple[list[object], dict[str, object]]:
    selected_pairs: list[object] = []
    full_counts: list[int] = []
    selected_counts: list[int] = []
    valid_candidate_counts: list[int] = []
    early_stop = 0
    strict_selected = 0
    no_op = 0
    strengths: list[float] = []
    for pair in pairs:
        candidates, full_count = valid_prefix_candidates(pair, vocabulary, args)
        strict = [candidate for candidate in candidates if bool(candidate["strict"])]
        if strict:
            chosen = min(
                strict,
                key=lambda candidate: (
                    int(candidate["actions"]),
                    -float(candidate["fraction"]),
                    -float(candidate["similarity"]),
                    str(candidate["smiles"]),
                ),
            )
        else:
            chosen = max(
                candidates,
                key=lambda candidate: (
                    float(candidate["fraction"]),
                    float(candidate["similarity"]),
                    -int(candidate["actions"]),
                    str(candidate["smiles"]),
                ),
            )
        selected_pairs.append(chosen["variant"])
        full_counts.append(int(full_count))
        selected_counts.append(int(chosen["actions"]))
        valid_candidate_counts.append(len(candidates))
        strengths.append(float(chosen["strength"]))
        strict_selected += int(bool(chosen["strict"]))
        no_op += int(int(chosen["actions"]) == 0)
        early_stop += int(bool(chosen["strict"]) and int(chosen["actions"]) < int(full_count))
    count = max(1, len(pairs))
    return selected_pairs, {
        "pairs": len(pairs),
        "early_stop_selected": early_stop,
        "early_stop_coverage": early_stop / count,
        "strict_selected": strict_selected,
        "selected_strict_rate": strict_selected / count,
        "no_op_selected": no_op,
        "no_op_rate": no_op / count,
        "mean_full_actions": float(np.mean(full_counts)) if full_counts else 0.0,
        "mean_selected_actions": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "mean_action_reduction": (
            float(np.mean(np.asarray(full_counts) - np.asarray(selected_counts)))
            if full_counts
            else 0.0
        ),
        "mean_selected_strength": float(np.mean(strengths)) if strengths else 0.0,
        "mean_valid_prefix_candidates": (
            float(np.mean(valid_candidate_counts)) if valid_candidate_counts else 0.0
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("The protocol requires exactly 20 raw attempts per condition")
    base.seed_everything(int(args.seed))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation, config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    validation_rows = base.read_rows(args.validation_csv)
    excluded_pairs, excluded_counts = base.build_pairs(
        validation_rows,
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_exclusion_seed),
    )
    excluded_sources = {pair.source_smiles for pair in excluded_pairs}
    excluded_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in excluded_pairs}
    validation_pairs, validation_counts = base.build_pairs(
        validation_rows,
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_selection_seed),
        forbidden_sources=excluded_sources,
        forbidden_pairs=excluded_pair_keys,
    )
    if not validation_pairs:
        raise ValueError("No development edit pairs survived the fixed filters")
    validation_sources = {pair.source_smiles for pair in validation_pairs}
    validation_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in validation_pairs}
    full_train_pairs, train_counts = base.build_pairs(
        base.read_rows(args.train_csv),
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.train_limit),
        seed=int(args.train_selection_seed),
        forbidden_sources=validation_sources,
        forbidden_pairs=validation_pair_keys,
    )
    if len(full_train_pairs) < 32:
        raise ValueError(f"Need at least 32 train pairs, found {len(full_train_pairs)}")
    for pair in [*full_train_pairs, *validation_pairs]:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(args.condition_dim)
        )
    vocabulary = full_graph.build_joint_state_vocabulary(full_train_pairs)
    train_pairs, trajectory = select_early_stop_pairs(
        full_train_pairs, vocabulary, args
    )
    trajectory_checks = {
        "early_stop_coverage": {
            "value": trajectory["early_stop_coverage"],
            "threshold": float(args.gate_early_stop_coverage),
        },
        "selected_strict_rate": {
            "value": trajectory["selected_strict_rate"],
            "threshold": float(args.gate_selected_strict_rate),
        },
    }
    trajectory_failures = [
        name
        for name, item in trajectory_checks.items()
        if float(item["value"]) < float(item["threshold"])
    ]
    print(
        json.dumps(
            {
                "stage": "trajectory_evidence",
                "trajectory_evidence": trajectory,
                "trajectory_gate": {
                    "passed": not trajectory_failures,
                    "checks": trajectory_checks,
                    "failures": trajectory_failures,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_sources = {pair.source_smiles for pair in full_train_pairs}
    train_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in full_train_pairs}
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "heldout_role": "development_not_final_audit",
        "device": str(device),
        "representation_protocol": representation_summary.get("protocol"),
        "representation_gate_passed": bool(representation_summary.get("gate", {}).get("passed")),
        "representation_checkpoint": str(args.representation_checkpoint),
        "representation_checkpoint_sha256": belief.file_sha256(args.representation_checkpoint),
        "train_csv": str(args.train_csv),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "selected_full_train_pairs": len(full_train_pairs),
        "selected_early_stop_train_pairs": len(train_pairs),
        "selected_validation_pairs": len(validation_pairs),
        "train_filter_counts": train_counts,
        "validation_filter_counts": validation_counts,
        "historical_validation_filter_counts": excluded_counts,
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_validation_pair_overlap": len(train_pair_keys & validation_pair_keys),
        "historical_validation_source_overlap": len(excluded_sources & validation_sources),
        "historical_validation_pair_overlap": len(excluded_pair_keys & validation_pair_keys),
        "property_counts": sorted(allowed_counts),
        "source_relative_sparse_delta_diffusion": True,
        "train_only_valid_early_stop_supervision": True,
        "train_only_property_oracle_for_trajectory_labels": True,
        "train_only_rdkit_validity_for_trajectory_labels": True,
        "trajectory_fractions": parse_fractions(str(args.trajectory_fractions)),
        "trajectory_max_orders": int(args.trajectory_max_orders),
        "continuous_transport_latent": True,
        "latent_usage_contrast": True,
        "latent_variance_floor": float(args.latent_min_std),
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "generation_rdkit_validity_access": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "posthoc_molecule_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "train_selection_seed": int(args.train_selection_seed),
        "validation_selection_seed": int(args.validation_selection_seed),
        "validation_exclusion_seed": int(args.validation_exclusion_seed),
    }
    if trajectory_failures:
        summary = {
            "protocol": PROTOCOL,
            "manifest": manifest,
            "trajectory_evidence": trajectory,
            "trajectory_gate": {
                "passed": False,
                "checks": trajectory_checks,
                "failures": trajectory_failures,
            },
            "training": [],
            "evaluation": None,
            "gate": {"passed": False, "checks": {}, "failures": ["trajectory_evidence"]},
            "next_stage": "replace_prefix_supervision_with_learned_local_rewrite_kernel",
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = full_graph.ContinuousDiscreteGraphDiffusion(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        transport_dim=int(args.transport_dim),
        hidden_dim=int(args.hidden_dim),
        max_atoms=int(config["max_atoms"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(args.message_layers),
    ).to(device)
    history = delta.train_model(
        model, representation, train_pairs, vocabulary, args, device
    )
    candidate_rows, metrics = delta.evaluate(
        model, representation, vocabulary, validation_pairs, args, device
    )
    three_property_strict = float(
        metrics["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": float(args.gate_validity)},
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": float(args.gate_mean_unique_valid),
        },
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": float(args.gate_source_tanimoto),
        },
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": float(args.gate_strict_any20),
        },
        "three_property_strict_any20": {
            "value": three_property_strict,
            "threshold": float(args.gate_3p_strict_any20),
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name == "exact_attempts"
            else item["value"] < item["threshold"]
        )
    ]
    checkpoint_path = args.output_dir / "valid_early_stop_delta_diffusion.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": int(args.condition_dim),
                "transport_dim": int(args.transport_dim),
                "hidden_dim": int(args.hidden_dim),
                "max_atoms": int(config["max_atoms"]),
                "property_count": len(unified.PROPERTY_COLUMNS),
                "message_layers": int(args.message_layers),
                "node_action_count": node_action_count,
                "edge_action_count": edge_action_count,
            },
            "vocabulary": {
                "node_states": np.asarray(vocabulary["node_states"]).tolist(),
                "edge_states": np.asarray(vocabulary["edge_states"]).tolist(),
                "sha256": vocabulary["sha256"],
            },
            "trajectory_evidence": trajectory,
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "trajectory_evidence": trajectory,
        "trajectory_gate": {
            "passed": True,
            "checks": trajectory_checks,
            "failures": [],
        },
        "training": history,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "scale_valid_early_stop_delta_diffusion_to_unified_2p_7p"
            if not failures
            else "replace_independent_delta_tokens_with_local_rewrite_kernel"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
