#!/usr/bin/env python3
"""Evaluate a valence-constrained, repulsive latent-particle B39 bridge.

B39 established a strong random-finite-set graph transport signal, but its
independent latent draws collapsed to 7.49 unique valid molecules and its
unconstrained NODE_WRITE/EDGE_SET support reduced validity to 89.51%.  B40
freezes the B39 checkpoint and changes the *generative support*, not the
resulting molecules: each requested candidate is one of exactly twenty
orthogonal Gaussian particles, and every jump is sampled from the subset that
admits a train-supported atom-state/bond grammar and valence-feasible terminal
graph.

There is no candidate pool, ranking, retry, property-oracle selection, or
post-hoc repair.  Development targets and property scorers are opened only
after the twenty raw candidates per condition have been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

import latent_cardinality_graph_jump_bridge as b39  # noqa: E402


b38 = b39.b38
b37 = b39.b37
b36 = b39.b36
base = b39.base
belief = b39.belief
delta = b39.delta
full_graph = b39.full_graph
graph = b39.graph
hierarchical = b39.hierarchical
unified = b39.unified

PROTOCOL = "train_only_valence_constrained_latent_particle_bridge_v40"


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
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_b39_checkpoint": True,
        "train_only_atom_state_valence_grammar": True,
        "dynamic_valence_event_support": True,
        "train_observed_bond_support": True,
        "orthogonal_latent_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
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
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B40 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B40 property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            f"B40 implementation drift: expected {payload.get('implementation_sha256')}, found {actual}"
        )
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
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("B40 locked-input manifest is incomplete")
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
        raise ValueError(f"B40 locked input drift: {drift}")
    b22_summary, checkpoint, b36_summary, b37_summary, b38_summary, _ = (
        b39.check_locked_inputs(args, preregistration)
    )
    b39_summary = json.loads(args.b39_summary.read_text(encoding="utf-8"))
    if b39_summary.get("protocol") != b39.PROTOCOL:
        raise ValueError("B40 requires the locked B39 protocol")
    if b39_summary.get("decision") != (
        "stop_and_diagnose_cardinality_or_mark_transport_without_length_clipping"
    ):
        raise ValueError("B40 refuses a B39 decision drift")
    b39_manifest = dict(b39_summary.get("manifest", {}))
    if b39_manifest.get("checkpoint_sha256") != locked["b39_checkpoint_sha256"]:
        raise ValueError("B40 B39 checkpoint/summary hash mismatch")
    if b39_manifest.get("generation_target_access") is not False:
        raise ValueError("B40 refuses a target-exposed B39 checkpoint")
    if b39_manifest.get("oracle_selection") is not False:
        raise ValueError("B40 refuses an oracle-selected B39 checkpoint")
    trigger = dict(preregistration["b39_failure_trigger"])
    metrics = dict(b39_summary.get("metrics", {}))
    evidence_drift = {}
    for key, expected in trigger.items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                evidence_drift[key] = {"expected": expected, "actual": actual}
        elif actual != expected:
            evidence_drift[key] = {"expected": expected, "actual": actual}
    if evidence_drift:
        raise ValueError(f"B40 refuses B39 failure-evidence drift: {evidence_drift}")
    verify_b39_diagnostic(args.b39_evaluated_candidates, preregistration)
    b39_checkpoint = torch.load(
        args.b39_checkpoint, map_location="cpu", weights_only=False
    )
    if b39_checkpoint.get("stage") != b39.PROTOCOL:
        raise ValueError("B40 refuses a non-B39 checkpoint")
    return b22_summary, checkpoint, b36_summary, b37_summary, b39_checkpoint


def verify_b39_diagnostic(
    path: Path, preregistration: Mapping[str, object]
) -> None:
    """Reproduce the preregistered event-family validity diagnosis."""
    counts = {
        "no_write_no_set": [0, 0],
        "write_and_set": [0, 0],
        "three_property": [0, 0],
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            valid = str(row.get("valid", "")).strip().lower() == "true"
            node_write = int(row["node_write_events"]) > 0
            edge_set = int(row["edge_set_events"]) > 0
            categories = []
            if not node_write and not edge_set:
                categories.append("no_write_no_set")
            if node_write and edge_set:
                categories.append("write_and_set")
            if int(row["property_count"]) == 3:
                categories.append("three_property")
            for category in categories:
                counts[category][0] += 1
                counts[category][1] += int(valid)
    expected = dict(preregistration["b39_frozen_candidate_diagnostic"])
    actual = {
        category: {"rows": total, "valid_rows": valid}
        for category, (total, valid) in counts.items()
    }
    if actual != expected:
        raise ValueError(
            f"B40 frozen B39 diagnostic drift: expected {expected}, found {actual}"
        )


def build_support(
    fit_pairs: Sequence[object], vocabulary: Mapping[str, object]
) -> dict[str, object]:
    """Compile fit-only atom-state capacities and hierarchical bond support."""
    grammar = hierarchical.build_train_atom_state_grammar(fit_pairs)
    fields = tuple(hierarchical.ATOM_STATE_FIELDS)
    field_indices = [full_graph.NODE_FIELDS.index(field) for field in fields]
    states = np.asarray(grammar["states"], dtype=np.int64)
    coarse_states = np.asarray(grammar["coarse_states"], dtype=np.int64)
    capacities = np.asarray(grammar["capacities"], dtype=np.int64)
    state_to_id = {tuple(row): index for index, row in enumerate(states.tolist())}
    coarse_to_id = {
        tuple(row): index for index, row in enumerate(coarse_states.tolist())
    }
    # Vocabulary state zero is the inactive state and is never a WRITE payload.
    payload_nodes = np.asarray(vocabulary["node_states"], dtype=np.int64)[1:]
    payload_states = payload_nodes[:, field_indices]
    payload_full_ids = np.asarray(
        [state_to_id.get(tuple(row), -1) for row in payload_states.tolist()],
        dtype=np.int64,
    )
    payload_coarse_ids = np.asarray(
        [coarse_to_id.get(tuple(row[:3]), -1) for row in payload_states.tolist()],
        dtype=np.int64,
    )
    payload_caps = np.asarray(
        [capacities[index] if index >= 0 else -1 for index in payload_full_ids],
        dtype=np.int64,
    )
    payload_edges = np.asarray(vocabulary["edge_states"], dtype=np.int64)[1:]
    bond_field = full_graph.EDGE_FIELDS.index("bond")
    return {
        "grammar": grammar,
        "payload_nodes": payload_nodes,
        "payload_full_ids": payload_full_ids,
        "payload_coarse_ids": payload_coarse_ids,
        "payload_caps": payload_caps,
        "payload_bonds": payload_edges[:, bond_field],
        "atom_state_count": int(len(states)),
        "coarse_atom_state_count": int(len(coarse_states)),
        "supported_node_payloads": int((payload_full_ids >= 0).sum()),
        "node_payloads": int(len(payload_full_ids)),
        "atom_state_grammar_sha256": grammar["sha256"],
        "bond_support_sha256": grammar["bond_support_sha256"],
    }


def _device_support(
    support: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    grammar = dict(support["grammar"])
    return {
        "payload_nodes": torch.as_tensor(
            np.asarray(support["payload_nodes"]), device=device, dtype=torch.long
        ),
        "payload_full_ids": torch.as_tensor(
            np.asarray(support["payload_full_ids"]), device=device, dtype=torch.long
        ),
        "payload_coarse_ids": torch.as_tensor(
            np.asarray(support["payload_coarse_ids"]), device=device, dtype=torch.long
        ),
        "payload_caps": torch.as_tensor(
            np.asarray(support["payload_caps"]), device=device, dtype=torch.long
        ),
        "payload_bonds": torch.as_tensor(
            np.asarray(support["payload_bonds"]), device=device, dtype=torch.long
        ),
        "bond_support": torch.as_tensor(
            np.asarray(grammar["bond_support"]), device=device, dtype=torch.bool
        ),
        "coarse_bond_support": torch.as_tensor(
            np.asarray(grammar["coarse_bond_support"]), device=device, dtype=torch.bool
        ),
        "bond_units": torch.as_tensor(
            hierarchical.BOND_VALENCE_UNITS, device=device, dtype=torch.long
        ),
    }


def _pair_bond_support(
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


def constrained_event_mask(
    field: b38.JumpEventField,
    source: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    working: torch.Tensor,
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return dynamic grammar/valence support before an event is sampled."""
    base_legal = b38.legal_event_mask(
        field, source, node_actions, edge_actions, working
    )
    device = node_actions.device
    tensors = support_tensors
    payload_nodes = tensors["payload_nodes"]
    payload_full = tensors["payload_full_ids"]
    payload_coarse = tensors["payload_coarse_ids"]
    payload_caps = tensors["payload_caps"]
    payload_bonds = tensors["payload_bonds"]
    bond_units = tensors["bond_units"]
    batch, nodes = node_actions.shape
    pair_left, pair_right = field.pair_left, field.pair_right

    source_full, source_coarse = hierarchical.source_atom_state_ids(
        source, support["grammar"]
    )
    source_caps = hierarchical.source_atom_state_valence_caps(
        source, support["grammar"]
    )
    write = node_actions.ge(delta.NODE_WRITE_OFFSET + 1)
    write_payload = (node_actions - (delta.NODE_WRITE_OFFSET + 1)).clamp(
        0, max(0, len(payload_full) - 1)
    )
    current_full = torch.where(write, payload_full[write_payload], source_full)
    current_coarse = torch.where(
        write, payload_coarse[write_payload], source_coarse
    )
    current_caps = torch.where(write, payload_caps[write_payload], source_caps)
    explicit_index = full_graph.NODE_FIELDS.index("explicit_hs")
    current_explicit_h = torch.where(
        write,
        payload_nodes[write_payload, explicit_index],
        source["explicit_hs"].long(),
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

    # A WRITE is feasible if all already committed SET bonds fit its capacity.
    payload_h = payload_nodes[:, explicit_index]
    node_write_ok = (
        committed_units.unsqueeze(-1) + 2 * payload_h.view(1, 1, -1)
    ).le(payload_caps.view(1, 1, -1))
    node_write_ok &= payload_caps.view(1, 1, -1).ge(0)
    # A committed bond must also be in train-observed full or coarse support.
    for batch_index in range(batch):
        committed_pairs = torch.nonzero(
            torch.triu(edge_set[batch_index], diagonal=1), as_tuple=False
        ).tolist()
        for left, right in committed_pairs:
            bond = int(committed_bonds[batch_index, left, right])
            for node, other in ((left, right), (right, left)):
                full_ok = _pair_bond_support(
                    payload_full,
                    current_full[batch_index, other].expand_as(payload_full),
                    payload_coarse,
                    current_coarse[batch_index, other].expand_as(payload_coarse),
                    torch.full_like(payload_full, bond),
                    tensors,
                )
                node_write_ok[batch_index, node] &= full_ok

    left_caps = current_caps[:, pair_left]
    right_caps = current_caps[:, pair_right]
    left_used = committed_units[:, pair_left]
    right_used = committed_units[:, pair_right]
    left_h = current_explicit_h[:, pair_left]
    right_h = current_explicit_h[:, pair_right]
    new_units = bond_units[payload_bonds].view(1, 1, -1)
    edge_valence_ok = (
        left_used.unsqueeze(-1) + new_units + 2 * left_h.unsqueeze(-1)
    ).le(left_caps.unsqueeze(-1)) & (
        right_used.unsqueeze(-1) + new_units + 2 * right_h.unsqueeze(-1)
    ).le(right_caps.unsqueeze(-1))
    edge_grammar_ok = _pair_bond_support(
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

    # STOP materializes all remaining KEEP source bonds.  It is legal only if
    # that terminal graph is inside the train-derived atom/bond support.
    source_bond = source["bond"].long()
    edge_delete = edge_actions.eq(delta.EDGE_DELETE)
    final_bond = torch.where(edge_delete, torch.zeros_like(source_bond), source_bond)
    final_bond = torch.where(edge_set, committed_bonds, final_bond)
    active_pair = current_active[:, :, None] & current_active[:, None, :]
    final_bond = torch.where(active_pair, final_bond, torch.zeros_like(final_bond))
    final_valence = bond_units[final_bond].sum(dim=2) + 2 * current_explicit_h
    stop_ok = (
        (~current_active) | final_valence.le(current_caps)
    ).all(dim=1) & current_active.any(dim=1)
    aromatic_index = full_graph.NODE_FIELDS.index("aromatic")
    current_aromatic = torch.where(
        write,
        payload_nodes[write_payload, aromatic_index].bool(),
        source["aromatic"].bool(),
    ) & current_active
    aromatic_bond = final_bond.eq(graph.BOND_AROMATIC)
    aromatic_degree = aromatic_bond.sum(dim=2)
    aromatic_endpoints = (
        ~aromatic_bond
        | (current_aromatic[:, :, None] & current_aromatic[:, None, :])
    ).all(dim=(1, 2))
    aromatic_atoms_supported = (
        (~current_aromatic) | aromatic_degree.ge(2)
    ).all(dim=1)
    stop_ok &= aromatic_endpoints & aromatic_atoms_supported
    pair_bond = final_bond[:, pair_left, pair_right]
    changed_context = (
        edge_set[:, pair_left, pair_right]
        | write[:, pair_left]
        | write[:, pair_right]
    )
    changed_bond = changed_context & pair_bond.gt(graph.BOND_NONE)
    changed_support = _pair_bond_support(
        current_full[:, pair_left],
        current_full[:, pair_right],
        current_coarse[:, pair_left],
        current_coarse[:, pair_right],
        pair_bond,
        tensors,
    )
    stop_ok &= (~changed_bond | changed_support).all(dim=1)

    layout = field.layout
    constrained = base_legal.clone()
    constrained[:, 0] &= stop_ok
    constrained[
        :, layout.node_write_offset : layout.edge_delete_offset
    ] &= node_write_ok.flatten(1)
    constrained[:, layout.edge_set_offset :] &= edge_set_ok.flatten(1)
    if not bool(constrained.any(dim=1).all()):
        bad = torch.nonzero(~constrained.any(dim=1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"B40 dynamic support reached a dead end: {bad}")
    diagnostics = {
        "base_legal": base_legal.sum(dim=1),
        "constrained_legal": constrained.sum(dim=1),
        "stop_masked": ~stop_ok,
    }
    return constrained, diagnostics


def orthogonal_latent_particles(
    attempts: int,
    dimension: int,
    generator: torch.Generator,
    device: torch.device,
    scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Draw exactly n direct particles with orthogonal directions, no pool."""
    if attempts > dimension:
        raise ValueError("B40 needs transport_dim >= exact particle count")
    raw = torch.randn(
        attempts, dimension, generator=generator, device=device, dtype=torch.float32
    )
    radii = raw.norm(dim=1, keepdim=True).clamp_min(1e-8)
    basis, triangular = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
    signs = torch.sign(torch.diagonal(triangular)).clamp_min(0.0) * 2.0 - 1.0
    particles = basis.transpose(0, 1) * signs[:, None] * radii * float(scale)
    normalized = torch.nn.functional.normalize(particles, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, device=device, dtype=torch.bool)
    return particles, {
        "initial_particle_mean_abs_cosine": float(
            cosine[off_diagonal].abs().mean().detach().cpu()
        ),
        "initial_particle_max_abs_cosine": float(
            cosine[off_diagonal].abs().max().detach().cpu()
        ),
    }


@torch.no_grad()
def sample_from_source(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: torch.nn.Module,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    source_example: object,
    condition_tokens: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    attempts = int(preregistration["exact_raw_attempts_per_condition"])
    batch_size = int(preregistration["sample_batch_size"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    particles, particle_metrics = orthogonal_latent_particles(
        attempts,
        model.transport_dim,
        generator,
        device,
        float(preregistration["latent_noise_scale"]),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    outputs: list[dict[str, object]] = []
    final_latents: list[torch.Tensor] = []
    model.eval()
    for start in range(0, attempts, batch_size):
        count = min(batch_size, attempts - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        tokens = torch.from_numpy(
            np.repeat(condition_tokens[None, ...], count, axis=0)
        ).to(device)
        latent = particles[start : start + count]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            source_node, source_edge = representation.encode(source)
            for flow_index in range(int(preregistration["flow_steps"])):
                flow_time = torch.full(
                    (count,),
                    (flow_index + 0.5) / int(preregistration["flow_steps"]),
                    device=device,
                    dtype=source_node.dtype,
                )
                latent = latent + model.transport_velocity(
                    latent, flow_time, source_node, source["node_mask"], tokens
                ) / int(preregistration["flow_steps"])
            condition = model.route_condition(tokens)
            cardinality_logits = model.cardinality_logits(
                source_node, source["node_mask"].bool(), condition, latent
            ).float()
            cardinality_probability = torch.softmax(
                cardinality_logits / float(preregistration["cardinality_temperature"]),
                dim=1,
            )
            predicted_cardinality = torch.multinomial(
                cardinality_probability, 1, generator=generator
            ).squeeze(1)
            working = full_graph.working_node_mask(
                source["node_mask"], int(preregistration["birth_capacity"])
            )
            node_actions = torch.full_like(source["atomic_number"], delta.NODE_KEEP)
            edge_actions = torch.full_like(source["bond"], delta.EDGE_KEEP)
            stopped = torch.zeros(count, dtype=torch.bool, device=device)
            event_counts = torch.zeros(count, dtype=torch.long, device=device)
            kind_counts = torch.zeros(count, 5, dtype=torch.long, device=device)
            masked_events = torch.zeros(count, dtype=torch.long, device=device)
            base_events = torch.zeros(count, dtype=torch.long, device=device)
            stop_masked_steps = torch.zeros(count, dtype=torch.long, device=device)
            for _ in range(int(preregistration["max_jumps"])):
                jump_time = event_counts.float() / float(preregistration["max_jumps"])
                remaining_mass = (
                    predicted_cardinality.float() - event_counts.float()
                ) / float(preregistration["max_jumps"])
                logits = model.denoiser(
                    node_actions,
                    edge_actions,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    jump_time,
                    condition,
                    latent,
                    remaining_mass,
                ).float()
                legal, support_diagnostics = constrained_event_mask(
                    model.denoiser,
                    source,
                    node_actions,
                    edge_actions,
                    working,
                    support,
                    support_tensors,
                )
                masked_events += (
                    support_diagnostics["base_legal"]
                    - support_diagnostics["constrained_legal"]
                )
                base_events += support_diagnostics["base_legal"]
                stop_masked_steps += support_diagnostics["stop_masked"].long()
                if bool(stopped.any()):
                    legal[stopped] = False
                    legal[stopped, 0] = True
                probability = torch.softmax(
                    logits.masked_fill(~legal, -torch.inf)
                    / float(preregistration["event_temperature"]),
                    dim=1,
                )
                sampled = torch.multinomial(
                    probability, 1, generator=generator
                ).squeeze(1)
                for index in range(count):
                    if bool(stopped[index]):
                        continue
                    event = model.denoiser.layout.decode(int(sampled[index]))
                    kind_counts[index, event.kind] += 1
                    if b38.execute_flat_event(
                        int(sampled[index]),
                        model.denoiser.layout,
                        node_actions,
                        edge_actions,
                        index,
                    ):
                        stopped[index] = True
                    else:
                        event_counts[index] += 1
                if bool(stopped.all()):
                    break
            result = delta.apply_delta_actions(
                source, node_actions, edge_actions, vocabulary
            )
        final_latents.append(latent.float().detach().cpu())
        prediction = {
            key: value.detach().cpu().numpy() for key, value in result.items()
        }
        source_prediction = {
            key: value.detach().cpu().numpy()
            for key, value in source.items()
            if isinstance(value, torch.Tensor)
        }
        node_values = node_actions.detach().cpu()
        edge_values = edge_actions.detach().cpu()
        upper = torch.triu(
            torch.ones(edge_values.shape[1:], dtype=torch.bool), diagonal=1
        )
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            changed_edges = edge_values[index].ne(delta.EDGE_KEEP) & upper
            affected = node_values[index].ne(delta.NODE_KEEP)
            affected |= changed_edges.any(dim=0) | changed_edges.any(dim=1)
            outside = ~affected.numpy()
            outside_nodes_exact = all(
                np.array_equal(
                    prediction[field][index][outside],
                    source_prediction[field][index][outside],
                )
                for field in full_graph.NODE_FIELDS
            )
            outside_index = np.flatnonzero(outside)
            outside_edges_exact = all(
                np.array_equal(
                    prediction[field][index][np.ix_(outside_index, outside_index)],
                    source_prediction[field][index][np.ix_(outside_index, outside_index)],
                )
                for field in full_graph.EDGE_FIELDS
            )
            adjacency = (prediction["bond"][index] > graph.BOND_NONE) | (
                source_prediction["bond"][index] > graph.BOND_NONE
            )
            outputs.append(
                {
                    "particle_index": start + index,
                    "generated_smiles": graph.canonical_smiles(smiles or ""),
                    "predicted_atom_count": int(
                        (prediction["atomic_number"][index] > 0).sum()
                    ),
                    "latent_norm": float(latent[index].float().norm().detach().cpu()),
                    "predicted_cardinality": int(predicted_cardinality[index].cpu()),
                    "event_count": int(event_counts[index].cpu()),
                    "cardinality_residual_at_stop": int(
                        predicted_cardinality[index].cpu() - event_counts[index].cpu()
                    ),
                    "stopped_by_model": bool(stopped[index].cpu()),
                    "max_horizon_hit": bool(not stopped[index].cpu()),
                    "node_delete_events": int(kind_counts[index, b38.NODE_DELETE].cpu()),
                    "node_write_events": int(kind_counts[index, b38.NODE_WRITE].cpu()),
                    "edge_delete_events": int(kind_counts[index, b38.EDGE_DELETE].cpu()),
                    "edge_set_events": int(kind_counts[index, b38.EDGE_SET].cpu()),
                    "affected_node_count": int(affected.sum()),
                    "affected_components": b37.component_count(
                        affected.numpy(), adjacency
                    ),
                    "outside_source_invariant": bool(
                        outside_nodes_exact and outside_edges_exact
                    ),
                    "dynamic_support_mask_fraction": float(
                        masked_events[index].float()
                        / base_events[index].clamp_min(1).float()
                    ),
                    "stop_masked_steps": int(stop_masked_steps[index].cpu()),
                }
            )
    if len(outputs) != attempts:
        raise RuntimeError(f"B40 expected {attempts} attempts, produced {len(outputs)}")
    final = torch.cat(final_latents, dim=0)
    final_normalized = torch.nn.functional.normalize(final, dim=1)
    final_cosine = final_normalized @ final_normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, dtype=torch.bool)
    particle_metrics["final_particle_mean_abs_cosine"] = float(
        final_cosine[off_diagonal].abs().mean()
    )
    particle_metrics["final_particle_max_abs_cosine"] = float(
        final_cosine[off_diagonal].abs().max()
    )
    for row in outputs:
        row.update(particle_metrics)
    return outputs


def freeze_candidates(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: torch.nn.Module,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        generated = sample_from_source(
            model,
            representation,
            vocabulary,
            support,
            support_tensors,
            pair.source,
            np.asarray(pair.condition),
            preregistration,
            device,
            int(preregistration["seed"]) * 100000 + pair_index,
        )
        condition_id = f"train_only_dev_{pair_index:04d}"
        for attempt, candidate in enumerate(generated, start=1):
            rows.append(
                {
                    "condition_id": condition_id,
                    "pair_index": pair_index,
                    "attempt": attempt,
                    "property_count": int(pair.property_count),
                    "task": base.task_key(pair.row),
                    "source_smiles": pair.source_smiles,
                    **candidate,
                }
            )
        if (pair_index + 1) % 16 == 0 or pair_index + 1 == len(pairs):
            print(
                json.dumps(
                    {
                        "stage": "freeze_train_only_valence_particle_candidates",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise RuntimeError(f"B40 freeze expected {expected} rows, found {len(rows)}")
    return rows


def evaluate_frozen_candidates(
    frozen: Sequence[Mapping[str, object]], pairs: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated, metrics = b39.evaluate_frozen_candidates(frozen, pairs)
    metrics["mean_dynamic_support_mask_fraction"] = float(
        np.mean([float(row["dynamic_support_mask_fraction"]) for row in evaluated])
    )
    metrics["mean_stop_masked_steps"] = float(
        np.mean([float(row["stop_masked_steps"]) for row in evaluated])
    )
    by_condition: dict[str, Mapping[str, object]] = {}
    for row in evaluated:
        by_condition.setdefault(str(row["condition_id"]), row)
    metrics["mean_initial_particle_abs_cosine"] = float(
        np.mean(
            [
                float(row["initial_particle_mean_abs_cosine"])
                for row in by_condition.values()
            ]
        )
    )
    metrics["mean_final_particle_abs_cosine"] = float(
        np.mean(
            [
                float(row["final_particle_mean_abs_cosine"])
                for row in by_condition.values()
            ]
        )
    )
    return evaluated, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, checkpoint, b36_summary, b37_summary, b39_checkpoint = (
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
    representation, representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    vocabulary = b37.checkpoint_vocabulary(checkpoint)
    support = build_support(fit_pairs, vocabulary)
    if int(support["supported_node_payloads"]) <= 0:
        raise ValueError("B40 fit-only grammar covers no locked node payload")
    support_tensors = _device_support(support, device)
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
    model.load_state_dict(dict(b39_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)
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
        "frozen_b39_checkpoint": True,
        "b39_checkpoint_retrained": False,
        "train_only_atom_state_valence_grammar": True,
        "dynamic_valence_event_support": True,
        "train_observed_bond_support": True,
        "atom_state_grammar_sha256": support["atom_state_grammar_sha256"],
        "bond_support_sha256": support["bond_support_sha256"],
        "atom_state_count": support["atom_state_count"],
        "coarse_atom_state_count": support["coarse_atom_state_count"],
        "supported_node_payloads": support["supported_node_payloads"],
        "node_payloads": support["node_payloads"],
        "orthogonal_latent_particles": True,
        "particle_pool_size": 20,
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
    print(
        json.dumps(
            {
                "stage": "frozen_b39_with_train_only_dynamic_support",
                "split": split,
                "support": {
                    key: value
                    for key, value in support.items()
                    if key != "grammar" and not isinstance(value, np.ndarray)
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frozen = freeze_candidates(
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
    evaluated, metrics = evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    internal_gate = b38.gate(metrics, dict(preregistration["gates"]))
    manifest = {
        **training_manifest,
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "post_freeze_train_only_dev_target_access": True,
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "manifest": manifest,
        "metrics": metrics,
        "internal_gate": internal_gate,
        "decision": (
            "advance_valence_particle_bridge_to_once_only_prospective_confirmation"
            if internal_gate["passed"]
            else "stop_and_diagnose_support_or_particle_transport_without_gate_changes"
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
