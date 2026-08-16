#!/usr/bin/env python3
"""Train a bond-derived canonical graph transport on the frozen B41 lineage.

B41 made STOP reliable and raised strict success, but validity and molecule-level
diversity still failed the preregistered gate.  Its discrete graph state gives
aromaticity two independent representations: an atom flag and an aromatic bond
category.  This experiment replaces that redundant state with a quotient graph
state.  Atom edit payloads are identified modulo their aromatic flag, while
atom aromaticity is deterministically derived from the final bond graph.

The same quotient is used for fit-target encoding, event support, STOP, and raw
graph materialization.  It is therefore a generative state definition, not an
RDKit validity repair.  B41's interacting particles, exact n=20 budget, gates,
and train-only development split remain fixed.  No pool, ranking, retry, oracle
selection, validation-target access, or molecule repair is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import viability_preserving_interacting_particle_transport as b41  # noqa: E402


b40 = b41.b40
b39 = b41.b39
b38 = b41.b38
b37 = b41.b37
b36 = b41.b36
base = b41.base
belief = b41.belief
delta = b41.delta
full_graph = b41.full_graph
graph = b41.graph
hierarchical = b41.hierarchical
unified = b41.unified

PROTOCOL = "train_only_bond_derived_canonical_graph_transport_v42"
CANONICAL_NODE_FIELDS = (
    "atomic_number",
    "formal_charge",
    "chirality",
    "explicit_hs",
    "no_implicit",
)
GRAMMAR_FIELDS = (
    "atomic_number",
    "formal_charge",
    "explicit_hs",
    "no_implicit",
)

_ORIGINAL_DELTA_TARGETS = delta.delta_action_targets
_ORIGINAL_APPLY_DELTA = delta.apply_delta_actions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-summary", type=Path, required=True)
    parser.add_argument("--b37-summary", type=Path, required=True)
    parser.add_argument("--b38-checkpoint", type=Path, required=True)
    parser.add_argument("--b38-summary", type=Path, required=True)
    parser.add_argument("--b39-checkpoint", type=Path, required=True)
    parser.add_argument("--b39-summary", type=Path, required=True)
    parser.add_argument("--b39-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--b40-summary", type=Path, required=True)
    parser.add_argument("--b40-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--b41-checkpoint", type=Path, required=True)
    parser.add_argument("--b41-summary", type=Path, required=True)
    parser.add_argument("--b41-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_b41_checkpoint": True,
        "bond_derived_atom_aromaticity": True,
        "aromatic_flag_quotient_node_actions": True,
        "canonical_state_used_for_targets_support_stop_and_decode": True,
        "b41_particle_transport_frozen": True,
        "exact_raw_attempts_per_condition": 20,
        "particle_pool_size": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "development_source_limit": 160,
        "max_jumps": 64,
        "flow_steps": 8,
        "birth_capacity": 8,
        "epochs": 2,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B42 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B42 property-count contract drift")
    if payload.get("implementation_sha256") != belief.file_sha256(Path(__file__).resolve()):
        raise ValueError("B42 implementation hash drift")
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_summary_sha256",
        "b37_summary_sha256",
        "b38_checkpoint_sha256",
        "b38_summary_sha256",
        "b39_checkpoint_sha256",
        "b39_evaluated_candidates_sha256",
        "b39_summary_sha256",
        "b40_evaluated_candidates_sha256",
        "b40_summary_sha256",
        "b41_checkpoint_sha256",
        "b41_evaluated_candidates_sha256",
        "b41_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("B42 locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_summary_sha256": args.b36_summary,
        "b37_summary_sha256": args.b37_summary,
        "b38_checkpoint_sha256": args.b38_checkpoint,
        "b38_summary_sha256": args.b38_summary,
        "b39_checkpoint_sha256": args.b39_checkpoint,
        "b39_evaluated_candidates_sha256": args.b39_evaluated_candidates,
        "b39_summary_sha256": args.b39_summary,
        "b40_evaluated_candidates_sha256": args.b40_evaluated_candidates,
        "b40_summary_sha256": args.b40_summary,
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_evaluated_candidates_sha256": args.b41_evaluated_candidates,
        "b41_summary_sha256": args.b41_summary,
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
    }
    drift = {
        name: {"expected": locked[name], "actual": belief.file_sha256(path)}
        for name, path in paths.items()
        if belief.file_sha256(path) != locked[name]
    }
    if drift:
        raise ValueError(f"B42 locked input drift: {drift}")

    b41_prereg = json.loads(
        (SCRIPT_DIR / "viability_preserving_interacting_particle_transport_v41_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    b22_summary, checkpoint, b36_summary, b37_summary, _, _ = b41.check_locked_inputs(
        args, b41_prereg
    )
    b41_summary = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    if b41_summary.get("protocol") != b41.PROTOCOL:
        raise ValueError("B42 requires the locked B41 protocol")
    if b41_summary.get("decision") != (
        "stop_and_diagnose_viability_or_particle_support_without_gate_changes"
    ):
        raise ValueError("B42 refuses a B41 decision drift")
    b41_manifest = dict(b41_summary.get("manifest", {}))
    if b41_manifest.get("checkpoint_sha256") != locked["b41_checkpoint_sha256"]:
        raise ValueError("B42 B41 checkpoint/summary hash mismatch")
    for key in (
        "generation_target_access",
        "generation_property_oracle_access",
        "molecular_candidate_ranking",
        "oracle_selection",
        "retry_or_resampling",
        "posthoc_molecule_repair",
    ):
        if b41_manifest.get(key) is not False:
            raise ValueError(f"B42 refuses B41 contract drift: {key}")
    evidence_drift = {}
    for key, expected in dict(preregistration["b41_failure_trigger"]).items():
        actual = dict(b41_summary.get("metrics", {})).get(key)
        if actual is None or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            evidence_drift[key] = {"expected": expected, "actual": actual}
    if evidence_drift:
        raise ValueError(f"B42 refuses B41 failure-evidence drift: {evidence_drift}")
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)
    if b41_checkpoint.get("stage") != b41.PROTOCOL:
        raise ValueError("B42 refuses a non-B41 warm-start checkpoint")
    return b22_summary, checkpoint, b36_summary, b37_summary, b41_checkpoint


def _node_signatures(values: torch.Tensor) -> torch.Tensor:
    indices = [full_graph.NODE_FIELDS.index(field) for field in CANONICAL_NODE_FIELDS]
    return values[..., indices]


def canonical_vocabulary(vocabulary: Mapping[str, object]) -> dict[str, object]:
    """Attach an aromatic-invariant quotient map to the frozen vocabulary."""
    result = dict(vocabulary)
    node_states = np.asarray(vocabulary["node_states"], dtype=np.int64)
    signatures = _node_signatures(torch.as_tensor(node_states)).numpy()
    representative: dict[tuple[int, ...], int] = {}
    state_to_representative = np.zeros(len(node_states), dtype=np.int64)
    for state_id, row in enumerate(signatures.tolist()):
        key = tuple(int(value) for value in row)
        if state_id == 0:
            representative[key] = 0
        elif key not in representative or representative[key] == 0:
            representative[key] = state_id
        state_to_representative[state_id] = representative[key]
    # A second pass handles an unlikely signature first seen at blank state zero.
    for state_id, row in enumerate(signatures.tolist()):
        state_to_representative[state_id] = representative[tuple(row)]
    payload_representative = state_to_representative[1:]
    representative_payload_mask = payload_representative == np.arange(1, len(node_states))
    payload = json.dumps(
        {
            "canonical_fields": CANONICAL_NODE_FIELDS,
            "state_to_representative": state_to_representative.tolist(),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    result.update(
        {
            "canonical_node_fields": CANONICAL_NODE_FIELDS,
            "node_state_to_canonical_representative": state_to_representative,
            "canonical_representative_payload_mask": representative_payload_mask,
            "canonical_node_state_count": int(len(set(payload_representative.tolist()))),
            "canonical_quotient_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return result


def canonical_delta_action_targets(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    vocabulary: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode aligned targets modulo atom aromaticity."""
    states = torch.as_tensor(
        np.asarray(vocabulary["node_states"]), device=source["atomic_number"].device
    )
    representatives = torch.as_tensor(
        np.asarray(vocabulary["node_state_to_canonical_representative"]),
        device=states.device,
        dtype=torch.long,
    )
    state_signatures = _node_signatures(states)
    source_values = torch.stack(
        [source[field].long() for field in CANONICAL_NODE_FIELDS], dim=-1
    )
    target_values = torch.stack(
        [target[field].long() for field in CANONICAL_NODE_FIELDS], dim=-1
    )
    target_match = target_values.unsqueeze(-2).eq(state_signatures).all(dim=-1)
    if not bool(target_match.any(dim=-1).all()):
        raise ValueError("B42 target contains a canonical node state outside train vocabulary")
    target_state_ids = target_match.long().argmax(dim=-1)
    target_state_ids = representatives[target_state_ids]
    source_active = source["atomic_number"].gt(0)
    target_active = target["atomic_number"].gt(0)
    node_same = source_values.eq(target_values).all(dim=-1)
    node_actions = torch.full_like(target_state_ids, delta.NODE_KEEP)
    node_actions = torch.where(
        source_active & ~target_active,
        torch.full_like(node_actions, delta.NODE_DELETE),
        node_actions,
    )
    node_write = target_active & (~source_active | ~node_same)
    node_actions = torch.where(
        node_write,
        target_state_ids + delta.NODE_WRITE_OFFSET,
        node_actions,
    )

    edge_states = torch.as_tensor(
        np.asarray(vocabulary["edge_states"]), device=source["bond"].device
    )
    target_edge_values = torch.stack(
        [target[field].long() for field in full_graph.EDGE_FIELDS], dim=-1
    )
    target_edge_match = target_edge_values.unsqueeze(-2).eq(edge_states).all(dim=-1)
    if not bool(target_edge_match.any(dim=-1).all()):
        raise ValueError("B42 target contains an edge state outside train vocabulary")
    target_edge_ids = target_edge_match.long().argmax(dim=-1)
    source_bond = source["bond"].gt(graph.BOND_NONE)
    target_bond = target["bond"].gt(graph.BOND_NONE)
    edge_same = torch.ones_like(source_bond)
    for field in full_graph.EDGE_FIELDS:
        edge_same &= source[field].eq(target[field])
    edge_actions = torch.full_like(target_edge_ids, delta.EDGE_KEEP)
    edge_actions = torch.where(
        source_bond & ~target_bond,
        torch.full_like(edge_actions, delta.EDGE_DELETE),
        edge_actions,
    )
    edge_set = target_bond & (~source_bond | ~edge_same)
    edge_actions = torch.where(
        edge_set,
        target_edge_ids + delta.EDGE_SET_OFFSET,
        edge_actions,
    )
    return node_actions, edge_actions


def canonical_apply_delta_actions(
    source: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    vocabulary: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Materialize one quotient graph and derive aromaticity from its bonds."""
    result = _ORIGINAL_APPLY_DELTA(source, node_actions, edge_actions, vocabulary)
    active = result["atomic_number"].gt(0)
    bond_aromatic = result["bond"].eq(graph.BOND_AROMATIC)
    result["aromatic"] = (bond_aromatic.any(dim=2) & active).to(
        result["aromatic"].dtype
    )
    return result


def build_canonical_support(
    fit_pairs: Sequence[object], vocabulary: Mapping[str, object]
) -> dict[str, object]:
    """Build train-only valence/bond support on aromatic-invariant atom states."""
    maximum_valence: dict[tuple[int, ...], int] = {}
    unit_table = np.asarray(hierarchical.BOND_VALENCE_UNITS, dtype=np.int64)
    for pair in fit_pairs:
        for example in (pair.source, pair.target):
            active = np.asarray(example.atomic_number) > 0
            bond = np.asarray(example.bond, dtype=np.int64)
            explicit_hs = np.asarray(example.explicit_hs, dtype=np.int64)
            total_valence = unit_table[bond].sum(axis=1) + 2 * explicit_hs
            values = {
                field: np.asarray(getattr(example, field), dtype=np.int64)
                for field in GRAMMAR_FIELDS
            }
            for index in np.flatnonzero(active).tolist():
                state = tuple(int(values[field][index]) for field in GRAMMAR_FIELDS)
                maximum_valence[state] = max(
                    maximum_valence.get(state, 0),
                    min(hierarchical.MAX_VALENCE_UNITS, int(total_valence[index])),
                )
    ordered = sorted(maximum_valence.items())
    states = np.asarray([state for state, _ in ordered], dtype=np.int64)
    capacities = np.asarray([capacity for _, capacity in ordered], dtype=np.int64)
    state_to_id = {tuple(state): index for index, state in enumerate(states.tolist())}
    coarse_states = np.asarray(sorted({tuple(state[:2]) for state in states.tolist()}))
    coarse_to_id = {
        tuple(state): index for index, state in enumerate(coarse_states.tolist())
    }
    bond_support = np.zeros(
        (len(states), len(states), len(hierarchical.BOND_VALENCE_UNITS)), dtype=np.bool_
    )
    coarse_bond_support = np.zeros(
        (len(coarse_states), len(coarse_states), len(hierarchical.BOND_VALENCE_UNITS)),
        dtype=np.bool_,
    )
    for pair in fit_pairs:
        for example in (pair.source, pair.target):
            values = {
                field: np.asarray(getattr(example, field), dtype=np.int64)
                for field in GRAMMAR_FIELDS
            }
            atom_states = [
                tuple(int(values[field][index]) for field in GRAMMAR_FIELDS)
                for index in range(len(values["atomic_number"]))
            ]
            bonds = np.asarray(example.bond, dtype=np.int64)
            for left, right in np.argwhere(np.triu(bonds > graph.BOND_NONE, k=1)).tolist():
                bond = int(bonds[left, right])
                left_id, right_id = state_to_id[atom_states[left]], state_to_id[atom_states[right]]
                bond_support[left_id, right_id, bond] = True
                bond_support[right_id, left_id, bond] = True
                left_coarse = coarse_to_id[atom_states[left][:2]]
                right_coarse = coarse_to_id[atom_states[right][:2]]
                coarse_bond_support[left_coarse, right_coarse, bond] = True
                coarse_bond_support[right_coarse, left_coarse, bond] = True

    node_states = np.asarray(vocabulary["node_states"], dtype=np.int64)[1:]
    field_indices = [full_graph.NODE_FIELDS.index(field) for field in GRAMMAR_FIELDS]
    payload_states = node_states[:, field_indices]
    payload_full_ids = np.asarray(
        [state_to_id.get(tuple(row), -1) for row in payload_states.tolist()], dtype=np.int64
    )
    payload_coarse_ids = np.asarray(
        [coarse_to_id.get(tuple(row[:2]), -1) for row in payload_states.tolist()],
        dtype=np.int64,
    )
    payload_caps = np.asarray(
        [capacities[index] if index >= 0 else -1 for index in payload_full_ids],
        dtype=np.int64,
    )
    payload_edges = np.asarray(vocabulary["edge_states"], dtype=np.int64)[1:]
    payload_bonds = payload_edges[:, full_graph.EDGE_FIELDS.index("bond")]
    grammar_payload = json.dumps(
        [list(state) + [int(capacity)] for state, capacity in ordered],
        separators=(",", ":"),
    ).encode("utf-8")
    bond_payload = json.dumps(
        {
            "full": np.argwhere(bond_support).tolist(),
            "coarse": np.argwhere(coarse_bond_support).tolist(),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "fields": GRAMMAR_FIELDS,
        "states": states,
        "capacities": capacities,
        "coarse_states": coarse_states,
        "bond_support": bond_support,
        "coarse_bond_support": coarse_bond_support,
        "payload_nodes": node_states,
        "payload_full_ids": payload_full_ids,
        "payload_coarse_ids": payload_coarse_ids,
        "payload_caps": payload_caps,
        "payload_bonds": payload_bonds,
        "representative_payload_mask": np.asarray(
            vocabulary["canonical_representative_payload_mask"], dtype=np.bool_
        ),
        "atom_state_grammar_sha256": hashlib.sha256(grammar_payload).hexdigest(),
        "bond_support_sha256": hashlib.sha256(bond_payload).hexdigest(),
    }


def device_support(
    support: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    names = (
        "states",
        "capacities",
        "coarse_states",
        "bond_support",
        "coarse_bond_support",
        "payload_nodes",
        "payload_full_ids",
        "payload_coarse_ids",
        "payload_caps",
        "payload_bonds",
        "representative_payload_mask",
    )
    tensors = {
        name: torch.as_tensor(np.asarray(support[name]), device=device)
        for name in names
    }
    tensors["bond_units"] = torch.as_tensor(
        hierarchical.BOND_VALENCE_UNITS, device=device, dtype=torch.long
    )
    return tensors


def source_support_ids(
    source: Mapping[str, torch.Tensor], tensors: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.stack([source[field].long() for field in GRAMMAR_FIELDS], dim=-1)
    states = tensors["states"].long()
    matches = values.unsqueeze(-2).eq(states).all(dim=-1)
    full = matches.long().argmax(dim=-1)
    full = torch.where(matches.any(dim=-1), full, torch.full_like(full, -1))
    coarse_values = values[..., :2]
    coarse_states = tensors["coarse_states"].long()
    coarse_matches = coarse_values.unsqueeze(-2).eq(coarse_states).all(dim=-1)
    coarse = coarse_matches.long().argmax(dim=-1)
    coarse = torch.where(
        coarse_matches.any(dim=-1), coarse, torch.full_like(coarse, -1)
    )
    matched_caps = (
        matches.long() * tensors["capacities"].long().view(1, 1, -1)
    ).max(dim=-1).values
    current_valence = tensors["bond_units"][source["bond"].long()].sum(dim=2)
    current_valence += 2 * source["explicit_hs"].long()
    caps = torch.where(matches.any(dim=-1), matched_caps, current_valence)
    return full, coarse, caps


def pair_bond_support(
    left_full: torch.Tensor,
    right_full: torch.Tensor,
    left_coarse: torch.Tensor,
    right_coarse: torch.Tensor,
    bonds: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    full_known = left_full.ge(0) & right_full.ge(0)
    coarse_known = left_coarse.ge(0) & right_coarse.ge(0)
    full_ok = tensors["bond_support"][
        left_full.clamp_min(0), right_full.clamp_min(0), bonds
    ]
    coarse_ok = tensors["coarse_bond_support"][
        left_coarse.clamp_min(0), right_coarse.clamp_min(0), bonds
    ]
    return (full_known & full_ok) | (coarse_known & coarse_ok)


def canonical_event_mask(
    field: b38.JumpEventField,
    source: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    working: torch.Tensor,
    support: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply valence/bond support in the aromatic-invariant quotient state."""
    base_legal = b38.legal_event_mask(field, source, node_actions, edge_actions, working)
    payload_nodes = tensors["payload_nodes"].long()
    payload_full = tensors["payload_full_ids"].long()
    payload_coarse = tensors["payload_coarse_ids"].long()
    payload_caps = tensors["payload_caps"].long()
    payload_bonds = tensors["payload_bonds"].long()
    bond_units = tensors["bond_units"].long()
    representative = tensors["representative_payload_mask"].bool()
    batch, _ = node_actions.shape
    pair_left, pair_right = field.pair_left, field.pair_right

    source_full, source_coarse, source_caps = source_support_ids(source, tensors)
    write = node_actions.ge(delta.NODE_WRITE_OFFSET + 1)
    write_payload = (node_actions - (delta.NODE_WRITE_OFFSET + 1)).clamp(
        0, max(0, len(payload_full) - 1)
    )
    current_full = torch.where(write, payload_full[write_payload], source_full)
    current_coarse = torch.where(write, payload_coarse[write_payload], source_coarse)
    current_caps = torch.where(write, payload_caps[write_payload], source_caps)
    explicit_index = full_graph.NODE_FIELDS.index("explicit_hs")
    current_h = torch.where(
        write, payload_nodes[write_payload, explicit_index], source["explicit_hs"].long()
    )
    current_active = delta.action_active_nodes(
        source["atomic_number"].gt(0), node_actions
    )

    edge_set = edge_actions.ge(delta.EDGE_SET_OFFSET + 1)
    edge_payload = (edge_actions - (delta.EDGE_SET_OFFSET + 1)).clamp(
        0, max(0, len(payload_bonds) - 1)
    )
    committed_bonds = torch.where(
        edge_set, payload_bonds[edge_payload], torch.zeros_like(edge_actions)
    )
    committed_units = bond_units[committed_bonds].sum(dim=2)

    payload_h = payload_nodes[:, explicit_index]
    node_write_ok = (
        committed_units.unsqueeze(-1) + 2 * payload_h.view(1, 1, -1)
    ).le(payload_caps.view(1, 1, -1))
    node_write_ok &= payload_caps.view(1, 1, -1).ge(0)
    node_write_ok &= representative.view(1, 1, -1)
    for batch_index in range(batch):
        committed_pairs = torch.nonzero(
            torch.triu(edge_set[batch_index], diagonal=1), as_tuple=False
        ).tolist()
        for left, right in committed_pairs:
            bond = int(committed_bonds[batch_index, left, right])
            for node, other in ((left, right), (right, left)):
                full_ok = pair_bond_support(
                    payload_full,
                    current_full[batch_index, other].expand_as(payload_full),
                    payload_coarse,
                    current_coarse[batch_index, other].expand_as(payload_coarse),
                    torch.full_like(payload_full, bond),
                    tensors,
                )
                node_write_ok[batch_index, node] &= full_ok

    left_caps, right_caps = current_caps[:, pair_left], current_caps[:, pair_right]
    left_used, right_used = committed_units[:, pair_left], committed_units[:, pair_right]
    left_h, right_h = current_h[:, pair_left], current_h[:, pair_right]
    new_units = bond_units[payload_bonds].view(1, 1, -1)
    edge_valence_ok = (
        left_used.unsqueeze(-1) + new_units + 2 * left_h.unsqueeze(-1)
    ).le(left_caps.unsqueeze(-1)) & (
        right_used.unsqueeze(-1) + new_units + 2 * right_h.unsqueeze(-1)
    ).le(right_caps.unsqueeze(-1))
    edge_grammar_ok = pair_bond_support(
        current_full[:, pair_left].unsqueeze(-1),
        current_full[:, pair_right].unsqueeze(-1),
        current_coarse[:, pair_left].unsqueeze(-1),
        current_coarse[:, pair_right].unsqueeze(-1),
        payload_bonds.view(1, 1, -1),
        tensors,
    )
    edge_set_ok = edge_valence_ok & edge_grammar_ok & payload_bonds.view(
        1, 1, -1
    ).gt(graph.BOND_NONE)

    source_bond = source["bond"].long()
    edge_delete = edge_actions.eq(delta.EDGE_DELETE)
    final_bond = torch.where(edge_delete, torch.zeros_like(source_bond), source_bond)
    final_bond = torch.where(edge_set, committed_bonds, final_bond)
    active_pair = current_active[:, :, None] & current_active[:, None, :]
    final_bond = torch.where(active_pair, final_bond, torch.zeros_like(final_bond))
    final_valence = bond_units[final_bond].sum(dim=2) + 2 * current_h
    stop_valence = (
        ((~current_active) | final_valence.le(current_caps)).all(dim=1)
        & current_active.any(dim=1)
    )
    pair_bond = final_bond[:, pair_left, pair_right]
    changed_context = (
        edge_set[:, pair_left, pair_right]
        | write[:, pair_left]
        | write[:, pair_right]
    )
    changed_bond = changed_context & pair_bond.gt(graph.BOND_NONE)
    changed_support = pair_bond_support(
        current_full[:, pair_left],
        current_full[:, pair_right],
        current_coarse[:, pair_left],
        current_coarse[:, pair_right],
        pair_bond,
        tensors,
    )
    stop_bond_support = (~changed_bond | changed_support).all(dim=1)
    stop_atom_support = (
        (~current_active) | (current_full.ge(0) & current_caps.ge(0))
    ).all(dim=1)
    stop_ok = stop_valence & stop_bond_support & stop_atom_support

    layout = field.layout
    constrained = base_legal.clone()
    constrained[:, 0] &= stop_ok
    constrained[:, layout.node_write_offset : layout.edge_delete_offset] &= (
        node_write_ok.flatten(1)
    )
    constrained[:, layout.edge_set_offset :] &= edge_set_ok.flatten(1)
    if not bool(constrained.any(dim=1).all()):
        bad = torch.nonzero(~constrained.any(dim=1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"B42 canonical support reached a dead end: {bad}")
    return constrained, {
        "base_legal": base_legal.sum(dim=1),
        "constrained_legal": constrained.sum(dim=1),
        "stop_masked": ~stop_ok,
        "stop_valence_legal": stop_valence,
        "stop_changed_bonds_supported": stop_bond_support,
        "stop_atom_states_supported": stop_atom_support,
    }


@torch.no_grad()
def canonical_replay_gate(
    model: b39.LatentCardinalityGraphJumpBridge,
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    complete_stop_legal = 0
    canonical_aromatic_exact = 0
    target_events = 0
    batch_size = int(preregistration["batch_size"])
    component_failures: defaultdict[str, int] = defaultdict(int)
    for start in range(0, len(pairs), batch_size):
        items = list(pairs[start : start + batch_size])
        collated = base.pair_collate(items)
        source = base.move_graph_batch(collated["source"], device)
        target = base.move_graph_batch(collated["target"], device)
        node_targets, edge_targets = canonical_delta_action_targets(
            source, target, vocabulary
        )
        working = full_graph.working_node_mask(
            source["node_mask"], int(preregistration["birth_capacity"]), target["node_mask"]
        )
        legal, diagnostics = canonical_event_mask(
            model.denoiser,
            source,
            node_targets,
            edge_targets,
            working,
            support,
            support_tensors,
        )
        complete_stop_legal += int(legal[:, 0].sum())
        materialized = canonical_apply_delta_actions(
            source, node_targets, edge_targets, vocabulary
        )
        expected_aromatic = (
            materialized["bond"].eq(graph.BOND_AROMATIC).any(dim=2)
            & materialized["atomic_number"].gt(0)
        ).to(materialized["aromatic"].dtype)
        canonical_aromatic_exact += int(
            materialized["aromatic"].eq(expected_aromatic).all(dim=1).sum()
        )
        for name, values in diagnostics.items():
            if name.startswith("stop_") and name != "stop_masked":
                component_failures[name] += int((~values).sum())
        target_events += int(
            node_targets.ne(delta.NODE_KEEP).sum()
            + torch.triu(edge_targets.ne(delta.EDGE_KEEP), diagonal=1).sum()
        )
    coverage = canonical_aromatic_exact / max(1, len(pairs))
    stop_rate = complete_stop_legal / max(1, len(pairs))
    return {
        "fit_pairs": len(pairs),
        "fit_complete_stop_legal": complete_stop_legal,
        "fit_complete_stop_legal_rate": stop_rate,
        "canonical_aromatic_materialization_exact": canonical_aromatic_exact,
        "canonical_aromatic_materialization_rate": coverage,
        "canonical_node_state_count": int(vocabulary["canonical_node_state_count"]),
        "original_active_node_state_count": int(len(vocabulary["node_states"]) - 1),
        "component_failures": dict(sorted(component_failures.items())),
        "target_events": target_events,
        "passed": stop_rate == 1.0 and coverage == 1.0,
    }


def install_canonical_state_contract() -> None:
    """Route all reused B41 training/generation paths through the quotient state."""
    delta.delta_action_targets = canonical_delta_action_targets
    delta.apply_delta_actions = canonical_apply_delta_actions
    b41.viability_event_mask = canonical_event_mask


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, checkpoint, b36_summary, b37_summary, b41_checkpoint = (
        check_locked_inputs(args, preregistration)
    )
    selected_pairs, reconstruction = b36.reconstruct_b22_train_pairs(
        args, preregistration, checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    for pair in [*fit_pairs, *development_pairs]:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(preregistration["condition_dim"])
        )
    representation, representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = canonical_vocabulary(b37.checkpoint_vocabulary(checkpoint))
    support = build_canonical_support(fit_pairs, vocabulary)
    support_tensors = device_support(support, device)
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
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    install_canonical_state_contract()
    replay = canonical_replay_gate(
        model,
        fit_pairs,
        vocabulary,
        support,
        support_tensors,
        preregistration,
        device,
    )
    print(json.dumps({"stage": "canonical_replay_gate", **replay}, sort_keys=True), flush=True)
    if not bool(replay["passed"]):
        raise ValueError(f"B42 canonical replay gate failed: {replay}")
    history = b41.fine_tune_event_kernel(
        model,
        representation,
        fit_pairs,
        vocabulary,
        support,
        support_tensors,
        preregistration,
        device,
    )
    training_manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "representation_protocol": representation_summary.get("protocol"),
        "reconstruction": reconstruction,
        "split": split,
        "canonical_replay_gate": replay,
        "frozen_b41_checkpoint": True,
        "b41_particle_transport_frozen": True,
        "bond_derived_atom_aromaticity": True,
        "aromatic_flag_quotient_node_actions": True,
        "canonical_quotient_sha256": vocabulary["canonical_quotient_sha256"],
        "atom_state_grammar_sha256": support["atom_state_grammar_sha256"],
        "bond_support_sha256": support["bond_support_sha256"],
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
    }
    checkpoint_path = args.output_dir / "bond_derived_canonical_graph_transport.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "vocabulary": dict(b41_checkpoint["vocabulary"]),
            "canonical_quotient_sha256": vocabulary["canonical_quotient_sha256"],
            "history": history,
            "manifest": training_manifest,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = belief.file_sha256(checkpoint_path)
    print(
        json.dumps(
            {
                "stage": "checkpoint_frozen_before_generation",
                "checkpoint": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frozen = b41.freeze_candidates(
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        development_pairs,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, frozen)
    evaluated, metrics = b41.evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    internal_gate = b38.gate(metrics, dict(preregistration["gates"]))
    horizon_threshold = float(preregistration["additional_gates"]["max_horizon_hit_rate"])
    internal_gate["checks"]["max_horizon_hit_rate"] = {
        "threshold": horizon_threshold,
        "value": float(metrics["max_horizon_hit_rate"]),
    }
    if float(metrics["max_horizon_hit_rate"]) > horizon_threshold:
        internal_gate["failures"].append("max_horizon_hit_rate")
        internal_gate["passed"] = False
    b41_metrics = dict(json.loads(args.b41_summary.read_text(encoding="utf-8"))["metrics"])
    comparison = {
        key: float(metrics[key]) - float(b41_metrics[key])
        for key in (
            "validity",
            "strict_any20",
            "property_any20",
            "mean_unique_valid",
            "mean_source_tanimoto",
            "max_horizon_hit_rate",
        )
    }
    manifest = {
        **training_manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "post_freeze_train_only_dev_target_access": True,
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
        "b41_decision": json.loads(args.b41_summary.read_text(encoding="utf-8")).get(
            "decision"
        ),
    }
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "metrics": metrics,
        "delta_vs_b41": comparison,
        "internal_gate": internal_gate,
        "decision": (
            "advance_bond_derived_graph_transport_to_once_only_prospective_confirmation"
            if internal_gate["passed"]
            else "stop_and_diagnose_canonical_graph_support_or_diversity_without_gate_changes"
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
