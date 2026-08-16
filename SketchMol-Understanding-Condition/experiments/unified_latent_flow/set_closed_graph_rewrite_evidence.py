#!/usr/bin/env python3
"""Audit a chemically closed, compositional graph-rewrite state space.

This is the first experiment after the B42 architecture stop.  It does not
train a generator and it does not generate or rank molecules.  Instead, it
reconstructs the locked B22 train-only successful trajectories and represents
each edit as one atomic transaction:

* a set of connected changed regions;
* the complete target node/edge state inside every region; and
* every source-boundary bond transition needed to attach the region.

The whole transaction is committed at once, so a decoder never exposes an
intermediate partially-valenced graph.  A source-grouped meta split then asks
whether the transaction can be expressed by node, edge, and boundary grammar
tokens learned only from the fit split.  Whole-rewrite signatures are expected
to be novel; compositional token coverage, not rewrite lookup, is the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import source_anchored_graph_patch_evidence as patch_evidence  # noqa: E402


b22 = patch_evidence.b22
base = patch_evidence.base
belief = patch_evidence.belief
delta = patch_evidence.delta
full_graph = patch_evidence.full_graph
graph = patch_evidence.graph

PROTOCOL = "train_only_set_closed_graph_rewrite_evidence_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-records", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "architecture_reset_after_b42": True,
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
        "future_joint_set_decoder": True,
        "future_independent_event_decoder": False,
        "selected_full_train_pairs": 1451,
        "max_rewrite_components": 3,
        "max_nodes_per_component": 32,
        "max_boundary_anchors_per_component": 8,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Set-closed rewrite preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Set-closed rewrite property-count contract drift")
    implementation_sha256 = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "Set-closed rewrite implementation drift: "
            f"expected {payload.get('implementation_sha256')}, "
            f"found {implementation_sha256}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
        "b36_records_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Set-closed rewrite locked-input manifest is incomplete")
    return payload


def reconstruct_locked_b36_pairs(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    checkpoint: Mapping[str, object],
    summary: Mapping[str, object],
) -> tuple[list[object], dict[str, object]]:
    """Rebuild the exact B36 train lineage despite RDKit MCS timeout jitter."""

    locked_sha = str(dict(preregistration["locked_inputs"])["b36_records_sha256"])
    actual_sha = belief.file_sha256(args.b36_records)
    if actual_sha != locked_sha:
        raise ValueError(
            f"Locked B36 records drift: expected {locked_sha}, found {actual_sha}"
        )
    locked_records = [
        json.loads(line)
        for line in args.b36_records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_pairs = int(preregistration["selected_full_train_pairs"])
    if len(locked_records) != expected_pairs:
        raise ValueError(
            f"Locked B36 records contain {len(locked_records)} rows, "
            f"expected {expected_pairs}"
        )
    locked_multiset = Counter(
        (str(record["source_smiles"]), str(record["task"]))
        for record in locked_records
    )

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
    excluded_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in excluded_pairs
    }
    validation_pairs, _ = base.build_pairs(
        validation_rows,
        seed=int(preregistration["validation_selection_seed"]),
        forbidden_sources=excluded_sources,
        forbidden_pairs=excluded_keys,
        **common,
    )
    validation_sources = {pair.source_smiles for pair in validation_pairs}
    validation_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in validation_pairs
    }
    reconstructed_pairs, filter_counts = base.build_pairs(
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
    remaining = Counter(locked_multiset)
    locked_pairs = []
    excluded_timeout_jitter = []
    for pair in reconstructed_pairs:
        key = (pair.source_smiles, base.task_key(pair.row))
        if remaining[key] > 0:
            locked_pairs.append(pair)
            remaining[key] -= 1
        else:
            excluded_timeout_jitter.append(key)
    missing = {key: count for key, count in remaining.items() if count > 0}
    if missing:
        raise ValueError(f"Current reconstruction is missing locked B36 pairs: {missing}")
    if len(locked_pairs) != expected_pairs:
        raise ValueError(
            f"B36 lineage filter retained {len(locked_pairs)} pairs, "
            f"expected {expected_pairs}"
        )

    for pair in locked_pairs:
        pair.condition = patch_evidence.hierarchical.property_latent_slot_tokens(
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
        locked_pairs, checkpoint["vocabulary"], trajectory_args
    )
    locked_trajectory = dict(summary["trajectory_evidence"])
    exact_keys = ("pairs", "early_stop_selected", "strict_selected", "no_op_selected")
    float_keys = (
        "early_stop_coverage",
        "selected_strict_rate",
        "mean_full_actions",
        "mean_selected_actions",
    )
    drift = {
        key: {"expected": locked_trajectory[key], "actual": trajectory[key]}
        for key in exact_keys
        if trajectory[key] != locked_trajectory[key]
    }
    drift.update(
        {
            key: {"expected": locked_trajectory[key], "actual": trajectory[key]}
            for key in float_keys
            if not math.isclose(
                float(trajectory[key]), float(locked_trajectory[key]), abs_tol=1e-10
            )
        }
    )
    if drift:
        raise ValueError(f"Locked B36 trajectory reconstruction drift: {drift}")
    return selected_pairs, {
        "raw_reconstructed_pairs": len(reconstructed_pairs),
        "locked_lineage_pairs": len(locked_pairs),
        "excluded_timeout_jitter_pairs": len(excluded_timeout_jitter),
        "excluded_timeout_jitter_keys": [
            list(key) for key in sorted(excluded_timeout_jitter)
        ],
        "filter_counts": filter_counts,
        "historical_excluded_sources": len(excluded_sources),
        "development_excluded_sources": len(validation_sources),
        "trajectory_evidence": trajectory,
    }


def stable_meta_assignment(source_smiles: str, seed: int, folds: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{source_smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(folds) == 0


def encoded_token(kind: str, payload: object) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{kind}:{serialised}"


def node_token(value: Mapping[str, torch.Tensor], index: int) -> str:
    if int(value["atomic_number"][0, index].item()) == 0:
        return "node:DELETE"
    return encoded_token(
        "node", patch_evidence.tensor_state(value, full_graph.NODE_FIELDS, (index,))
    )


def edge_token(
    left_node: str,
    right_node: str,
    value: Mapping[str, torch.Tensor],
    left: int,
    right: int,
) -> str:
    return encoded_token(
        "edge",
        {
            "ends": sorted((left_node, right_node)),
            "state": patch_evidence.tensor_state(
                value, full_graph.EDGE_FIELDS, (left, right)
            ),
        },
    )


def record_tokens(
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
    components = patch_evidence.connected_components(changed, adjacency)
    changed_set = set(changed)

    node_tokens: list[str] = []
    internal_edge_tokens: list[str] = []
    boundary_tokens: list[str] = []
    component_sizes: list[int] = []
    boundary_counts: list[int] = []
    component_signatures: list[str] = []
    for component in components:
        component_sizes.append(len(component))
        local_nodes = {index: node_token(target, index) for index in component}
        node_tokens.extend(local_nodes.values())
        for offset, left in enumerate(component):
            for right in component[offset + 1 :]:
                if int(target["bond"][0, left, right].item()) == 0:
                    continue
                internal_edge_tokens.append(
                    edge_token(
                        local_nodes[left], local_nodes[right], target, left, right
                    )
                )

        anchors: set[int] = set()
        for local in component:
            for neighbour in np.flatnonzero(adjacency[local]).tolist():
                neighbour = int(neighbour)
                if neighbour in changed_set:
                    continue
                anchors.add(neighbour)
                boundary_tokens.append(
                    encoded_token(
                        "boundary",
                        {
                            "anchor": node_token(source, neighbour),
                            "local": local_nodes[local],
                            "source_edge": patch_evidence.tensor_state(
                                source, full_graph.EDGE_FIELDS, (local, neighbour)
                            ),
                            "target_edge": patch_evidence.tensor_state(
                                target, full_graph.EDGE_FIELDS, (local, neighbour)
                            ),
                        },
                    )
                )
        boundary_counts.append(len(anchors))
        component_signatures.append(
            patch_evidence.component_signature(
                source, target, component, changed_set, adjacency
            )
        )

    replay = delta.apply_delta_actions(source, node_actions, edge_actions, vocabulary)
    replay_smiles = b22.graph_result_smiles(replay)
    target_smiles = graph.canonical_smiles(pair.target_smiles)
    outside = sorted(set(range(node_actions.shape[1])) - changed_set)
    outside_nodes_exact = all(
        torch.equal(source[field][0, outside], replay[field][0, outside])
        for field in full_graph.NODE_FIELDS
    )
    if outside:
        indices = torch.as_tensor(outside, dtype=torch.long)
        outside_edges_exact = all(
            torch.equal(
                source[field][0][indices[:, None], indices[None, :]],
                replay[field][0][indices[:, None], indices[None, :]],
            )
            for field in full_graph.EDGE_FIELDS
        )
    else:
        outside_edges_exact = True
    _fraction, strict, _similarity = b22.property_outcome(pair, pair.target_smiles)
    envelope_supported = bool(
        components
        and len(components) <= int(preregistration["max_rewrite_components"])
        and max(component_sizes, default=0)
        <= int(preregistration["max_nodes_per_component"])
        and max(boundary_counts, default=0)
        <= int(preregistration["max_boundary_anchors_per_component"])
    )
    return {
        "source_smiles": pair.source_smiles,
        "target_smiles": pair.target_smiles,
        "task": base.task_key(pair.row),
        "property_count": int(pair.property_count),
        "strict": bool(strict),
        "component_sizes": component_sizes,
        "boundary_anchor_counts": boundary_counts,
        "component_signatures": component_signatures,
        "node_tokens": sorted(node_tokens),
        "internal_edge_tokens": sorted(internal_edge_tokens),
        "boundary_tokens": sorted(boundary_tokens),
        "envelope_supported": envelope_supported,
        "replay_exact": bool(replay_smiles and replay_smiles == target_smiles),
        "replay_valid": bool(replay_smiles),
        "outside_source_invariant": bool(outside_nodes_exact and outside_edges_exact),
    }


def token_coverage(records: Iterable[Mapping[str, object]], key: str, vocab: set[str]) -> float:
    tokens = [str(token) for record in records for token in record[key]]
    return sum(token in vocab for token in tokens) / max(1, len(tokens))


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed set-closed rewrite evidence exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    b22_summary, checkpoint = patch_evidence.load_locked_b22(args, preregistration)
    pairs, reconstruction = reconstruct_locked_b36_pairs(
        args, preregistration, checkpoint, b22_summary
    )

    records = []
    for index, pair in enumerate(pairs, start=1):
        records.append(record_tokens(pair, checkpoint["vocabulary"], preregistration))
        if index % 128 == 0 or index == len(pairs):
            print(
                json.dumps(
                    {"stage": "set_closed_rewrite_evidence", "pairs": index},
                    sort_keys=True,
                ),
                flush=True,
            )
    strict_records = [record for record in records if bool(record["strict"])]
    fit_records = []
    meta_records = []
    for record in strict_records:
        target = (
            meta_records
            if stable_meta_assignment(
                str(record["source_smiles"]),
                int(preregistration["meta_split_seed"]),
                int(preregistration["meta_split_folds"]),
            )
            else fit_records
        )
        target.append(record)
    fit_sources = {str(record["source_smiles"]) for record in fit_records}
    meta_sources = {str(record["source_smiles"]) for record in meta_records}
    fit_pairs = {
        (str(record["source_smiles"]), str(record["target_smiles"]))
        for record in fit_records
    }
    meta_pairs = {
        (str(record["source_smiles"]), str(record["target_smiles"]))
        for record in meta_records
    }
    vocabularies = {
        key: {str(token) for record in fit_records for token in record[key]}
        for key in ("node_tokens", "internal_edge_tokens", "boundary_tokens")
    }
    for record in meta_records:
        all_tokens = [
            str(token)
            for key in ("node_tokens", "internal_edge_tokens", "boundary_tokens")
            for token in record[key]
        ]
        record["fit_grammar_supported"] = bool(
            all_tokens
            and all(
                token in vocabularies[key]
                for key in ("node_tokens", "internal_edge_tokens", "boundary_tokens")
                for token in record[key]
            )
        )

    fit_signatures = {
        str(signature)
        for record in fit_records
        for signature in record["component_signatures"]
    }
    meta_signatures = {
        str(signature)
        for record in meta_records
        for signature in record["component_signatures"]
    }
    metrics = {
        "all_pairs": len(records),
        "strict_pairs": len(strict_records),
        "selected_strict_rate": len(strict_records) / max(1, len(records)),
        "fit_strict_pairs": len(fit_records),
        "meta_strict_pairs": len(meta_records),
        "fit_sources": len(fit_sources),
        "meta_sources": len(meta_sources),
        "fit_meta_source_overlap": len(fit_sources & meta_sources),
        "fit_meta_pair_overlap": len(fit_pairs & meta_pairs),
        "meta_exact_replay_rate": mean(
            [float(bool(record["replay_exact"])) for record in meta_records]
        ),
        "meta_valid_replay_rate": mean(
            [float(bool(record["replay_valid"])) for record in meta_records]
        ),
        "meta_outside_source_invariant_rate": mean(
            [
                float(bool(record["outside_source_invariant"]))
                for record in meta_records
            ]
        ),
        "meta_envelope_coverage": mean(
            [float(bool(record["envelope_supported"])) for record in meta_records]
        ),
        "meta_node_token_coverage": token_coverage(
            meta_records, "node_tokens", vocabularies["node_tokens"]
        ),
        "meta_internal_edge_token_coverage": token_coverage(
            meta_records,
            "internal_edge_tokens",
            vocabularies["internal_edge_tokens"],
        ),
        "meta_boundary_token_coverage": token_coverage(
            meta_records, "boundary_tokens", vocabularies["boundary_tokens"]
        ),
        "meta_full_grammar_pair_coverage": mean(
            [
                float(bool(record["fit_grammar_supported"]))
                for record in meta_records
            ]
        ),
        "fit_unique_component_signatures": len(fit_signatures),
        "meta_unique_component_signatures": len(meta_signatures),
        "meta_novel_component_signature_rate": len(meta_signatures - fit_signatures)
        / max(1, len(meta_signatures)),
        "fit_vocabulary_sizes": {
            key: len(value) for key, value in vocabularies.items()
        },
        "max_component_nodes": max(
            (
                max(record["component_sizes"], default=0)
                for record in strict_records
            ),
            default=0,
        ),
        "max_boundary_anchors": max(
            (
                max(record["boundary_anchor_counts"], default=0)
                for record in strict_records
            ),
            default=0,
        ),
        "max_rewrite_components": max(
            (len(record["component_sizes"]) for record in strict_records), default=0
        ),
    }
    gates = dict(preregistration["gates"])
    checks = {
        name: {"value": metrics[name], "threshold": threshold}
        for name, threshold in gates.items()
    }
    failures = [
        name
        for name, item in checks.items()
        if float(item["value"]) < float(item["threshold"])
    ]
    if metrics["fit_meta_source_overlap"] != 0:
        failures.append("fit_meta_source_overlap")
    if metrics["fit_meta_pair_overlap"] != 0:
        failures.append("fit_meta_pair_overlap")
    failures = sorted(set(failures))
    passed = not failures

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "rewrite_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "protocol": PROTOCOL,
        "implementation_sha256": preregistration["implementation_sha256"],
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "locked_inputs": preregistration["locked_inputs"],
        "b22_protocol": b22_summary.get("protocol"),
        "selected_full_train_pairs": len(records),
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
        "future_joint_set_decoder": True,
        "future_independent_event_decoder": False,
        "atomic_valence_closed_transaction": True,
        "fit_meta_source_group_split": True,
        "records_sha256": belief.file_sha256(records_path),
        "split_reconstruction": reconstruction,
    }
    result = {
        "protocol": PROTOCOL,
        "manifest": manifest,
        "metrics": metrics,
        "gate": {"passed": passed, "checks": checks, "failures": failures},
        "decision": (
            "train_set_closed_graph_transport_exact_n20"
            if passed
            else "stop_and_revise_closed_rewrite_grammar_without_opening_heldout_data"
        ),
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
