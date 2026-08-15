#!/usr/bin/env python3
"""Source-relative discrete delta diffusion for compositional molecular editing.

B21 keeps the train-only set-compositional continuous transport and replaces
B20's full-target graph diffusion with an absorbing diffusion over sparse edit
actions.  Node tokens are KEEP, DELETE, or WRITE(a complete train-supported
atom state); edge tokens are KEEP, DELETE, or SET(a complete train-supported
bond state).  The source graph is therefore an exact invariant base rather
than something the decoder must reconstruct.

Generation receives only the source graph, sanitized property tokens, and
Gaussian/categorical noise.  Exactly 20 raw attempts are frozen before the
development target and property scorer are opened.  There is no candidate
library, selector, finalizer, oracle reranking, post-hoc chemistry repair, or
validation-derived state support.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import discrete_graph_diffusion_decoder as full_graph


base = full_graph.base
belief = full_graph.belief
continuous = full_graph.continuous
graph = full_graph.graph
hierarchical = full_graph.hierarchical
unified = full_graph.unified

PROTOCOL = "source_relative_sparse_delta_diffusion_pilot_v21"
NODE_KEEP, NODE_DELETE, NODE_WRITE_OFFSET = 0, 1, 1
EDGE_KEEP, EDGE_DELETE, EDGE_SET_OFFSET = 0, 1, 1


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
    parser.add_argument("--seed", type=int, default=1755)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def action_space_sizes(vocabulary: Mapping[str, object]) -> tuple[int, int]:
    # State zero is blank and is never used as a WRITE/SET payload.
    return len(vocabulary["node_states"]) + 1, len(vocabulary["edge_states"]) + 1


def delta_action_targets(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    vocabulary: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one aligned source-target pair as joint sparse edit actions."""
    source_node_ids, source_edge_ids = full_graph.graph_state_ids(source, vocabulary)
    target_node_ids, target_edge_ids = full_graph.graph_state_ids(target, vocabulary)
    source_active = source["atomic_number"].gt(0)
    target_active = target["atomic_number"].gt(0)
    node_same = torch.ones_like(source_active)
    for field in full_graph.NODE_FIELDS:
        node_same &= source[field].eq(target[field])
    node_actions = torch.full_like(source_node_ids, NODE_KEEP)
    node_actions = torch.where(
        source_active & ~target_active,
        torch.full_like(node_actions, NODE_DELETE),
        node_actions,
    )
    node_write = target_active & (~source_active | ~node_same)
    node_actions = torch.where(
        node_write,
        target_node_ids + NODE_WRITE_OFFSET,
        node_actions,
    )

    source_bond = source["bond"].gt(graph.BOND_NONE)
    target_bond = target["bond"].gt(graph.BOND_NONE)
    edge_same = torch.ones_like(source_bond)
    for field in full_graph.EDGE_FIELDS:
        edge_same &= source[field].eq(target[field])
    edge_actions = torch.full_like(source_edge_ids, EDGE_KEEP)
    edge_actions = torch.where(
        source_bond & ~target_bond,
        torch.full_like(edge_actions, EDGE_DELETE),
        edge_actions,
    )
    edge_set = target_bond & (~source_bond | ~edge_same)
    edge_actions = torch.where(
        edge_set,
        target_edge_ids + EDGE_SET_OFFSET,
        edge_actions,
    )
    return node_actions, edge_actions


def apply_delta_actions(
    source: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    vocabulary: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Materialize sampled actions once; no molecule-level repair is applied."""
    result = {
        field: source[field].clone()
        for field in (*full_graph.NODE_FIELDS, *full_graph.EDGE_FIELDS)
    }
    node_states = torch.as_tensor(
        np.asarray(vocabulary["node_states"]), device=node_actions.device
    )
    edge_states = torch.as_tensor(
        np.asarray(vocabulary["edge_states"]), device=edge_actions.device
    )
    node_delete = node_actions.eq(NODE_DELETE)
    node_write = node_actions.ge(NODE_WRITE_OFFSET + 1)
    payload_node_id = (node_actions - NODE_WRITE_OFFSET).clamp(
        0, node_states.shape[0] - 1
    )
    payload_nodes = node_states[payload_node_id]
    for index, field in enumerate(full_graph.NODE_FIELDS):
        result[field] = torch.where(node_write, payload_nodes[..., index], result[field])
    result["atomic_number"] = torch.where(
        node_delete, torch.zeros_like(result["atomic_number"]), result["atomic_number"]
    )
    active = result["atomic_number"].gt(0)
    defaults = {
        "formal_charge": int(graph.CHARGE_OFFSET),
        "chirality": 0,
        "aromatic": 0,
        "explicit_hs": 0,
        "no_implicit": 0,
    }
    for field, default in defaults.items():
        result[field] = torch.where(
            active, result[field], torch.full_like(result[field], int(default))
        )

    nodes = node_actions.shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=node_actions.device, dtype=torch.bool), diagonal=1
    ).unsqueeze(0)
    edge_delete = edge_actions.eq(EDGE_DELETE) & upper
    edge_set = edge_actions.ge(EDGE_SET_OFFSET + 1) & upper
    payload_edge_id = (edge_actions - EDGE_SET_OFFSET).clamp(
        0, edge_states.shape[0] - 1
    )
    payload_edges = edge_states[payload_edge_id]
    edge_delete = edge_delete | edge_delete.transpose(1, 2)
    edge_set = edge_set | edge_set.transpose(1, 2)
    for index, field in enumerate(full_graph.EDGE_FIELDS):
        payload = payload_edges[..., index]
        payload = torch.where(upper, payload, torch.zeros_like(payload))
        payload = payload + payload.transpose(1, 2)
        result[field] = torch.where(edge_set, payload, result[field])
        result[field] = torch.where(
            edge_delete, torch.zeros_like(result[field]), result[field]
        )
    active_pair = active[:, :, None] & active[:, None, :]
    diagonal = torch.eye(nodes, device=node_actions.device, dtype=torch.bool).unsqueeze(0)
    for field in full_graph.EDGE_FIELDS:
        result[field] = torch.where(
            active_pair & ~diagonal, result[field], torch.zeros_like(result[field])
        )
    return result


def legal_node_action_mask(
    source_active: torch.Tensor, action_count: int
) -> torch.Tensor:
    legal = torch.ones(
        *source_active.shape,
        int(action_count),
        dtype=torch.bool,
        device=source_active.device,
    )
    legal[..., NODE_DELETE] = source_active
    return legal


def legal_edge_action_mask(
    source_bond: torch.Tensor,
    active_after_node_actions: torch.Tensor,
    action_count: int,
) -> torch.Tensor:
    active_pair = (
        active_after_node_actions[:, :, None]
        & active_after_node_actions[:, None, :]
    )
    legal = torch.ones(
        *source_bond.shape,
        int(action_count),
        dtype=torch.bool,
        device=source_bond.device,
    )
    legal[..., EDGE_DELETE] = source_bond
    legal[..., EDGE_SET_OFFSET + 1 :] = active_pair.unsqueeze(-1)
    return legal


def action_active_nodes(
    source_active: torch.Tensor, node_actions: torch.Tensor
) -> torch.Tensor:
    return (~node_actions.eq(NODE_DELETE)) & (
        source_active | node_actions.ge(NODE_WRITE_OFFSET + 1)
    )


def train_model(
    model: full_graph.ContinuousDiscreteGraphDiffusion,
    representation,
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    representation.eval().requires_grad_(False)
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(pairs)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            items = [pairs[index] for index in order[start : start + int(args.batch_size)]]
            collated = base.pair_collate(items)
            source = base.move_graph_batch(collated["source"], device)
            target = base.move_graph_batch(collated["target"], device)
            tokens = collated["condition"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                target_node, target_edge = representation.encode(target)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                condition = model.route_condition(tokens)
                endpoint = model.posterior_endpoint(
                    source, target, source_node, source_edge, target_node, target_edge, condition
                )
                noise = torch.randn_like(endpoint) * float(args.latent_noise_scale)
                flow_time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
                current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
                velocity = model.transport_velocity(
                    current, flow_time, source_node, source["node_mask"], tokens
                )
                target_velocity = endpoint - noise
                flow_loss = F.mse_loss(velocity.float(), target_velocity.float())
                predicted_endpoint = current + (1.0 - flow_time[:, None]) * velocity

                node_actions, edge_actions = delta_action_targets(source, target, vocabulary)
                working = full_graph.working_node_mask(
                    source["node_mask"], int(args.birth_capacity), target["node_mask"]
                )
                diffusion_index = torch.randint(
                    1, int(args.diffusion_steps) + 1, (len(items),), device=device
                )
                diffusion_time = diffusion_index.float() / max(1, int(args.diffusion_steps))
                noisy_node, noisy_edge, node_corrupted, edge_corrupted = (
                    full_graph.corrupt_joint_states(
                        node_actions,
                        edge_actions,
                        working,
                        diffusion_time,
                        model.denoiser.node_mask_id,
                        model.denoiser.edge_mask_id,
                    )
                )
                node_logits, edge_logits = model.denoiser(
                    noisy_node,
                    noisy_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    diffusion_time,
                    condition,
                    predicted_endpoint,
                )
                edge_eligible = edge_corrupted & full_graph.upper_working_pairs(working)
                node_loss = full_graph.balanced_categorical_loss(
                    node_logits, node_actions, node_corrupted, NODE_KEEP, 1.0
                )
                edge_loss = full_graph.balanced_categorical_loss(
                    edge_logits, edge_actions, edge_eligible, EDGE_KEEP, 1.0
                )

                wrong_latent = torch.roll(predicted_endpoint, shifts=1, dims=0)
                wrong_node_logits, wrong_edge_logits = model.denoiser(
                    noisy_node,
                    noisy_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    diffusion_time,
                    condition,
                    wrong_latent,
                )
                wrong_loss = full_graph.balanced_categorical_loss(
                    wrong_node_logits, node_actions, node_corrupted, NODE_KEEP, 1.0
                ) + full_graph.balanced_categorical_loss(
                    wrong_edge_logits, edge_actions, edge_eligible, EDGE_KEEP, 1.0
                )
                correct_loss = node_loss + edge_loss
                latent_usage = F.relu(
                    float(args.latent_usage_margin) + correct_loss - wrong_loss
                )
                latent_std = endpoint.float().std(dim=0, unbiased=False)
                variance_loss = F.relu(float(args.latent_min_std) - latent_std).mean()
                loss = (
                    correct_loss
                    + float(args.flow_loss_weight) * flow_loss
                    + float(args.latent_usage_weight) * latent_usage
                    + float(args.latent_variance_weight) * variance_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            with torch.no_grad():
                node_accuracy = (
                    node_logits.argmax(dim=-1)[node_corrupted]
                    .eq(node_actions[node_corrupted])
                    .float()
                    .mean()
                    if bool(node_corrupted.any())
                    else torch.ones((), device=device)
                )
                edge_accuracy = (
                    edge_logits.argmax(dim=-1)[edge_eligible]
                    .eq(edge_actions[edge_eligible])
                    .float()
                    .mean()
                    if bool(edge_eligible.any())
                    else torch.ones((), device=device)
                )
            totals["loss"] += float(loss.detach())
            totals["node_denoising_loss"] += float(node_loss.detach())
            totals["edge_denoising_loss"] += float(edge_loss.detach())
            totals["flow_matching_loss"] += float(flow_loss.detach())
            totals["latent_usage_loss"] += float(latent_usage.detach())
            totals["latent_variance_loss"] += float(variance_loss.detach())
            totals["node_masked_accuracy"] += float(node_accuracy)
            totals["edge_masked_accuracy"] += float(edge_accuracy)
            totals["posterior_std"] += float(latent_std.mean().detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def sample_from_source(
    model: full_graph.ContinuousDiscreteGraphDiffusion,
    representation,
    vocabulary: Mapping[str, object],
    source_example,
    condition_tokens: np.ndarray,
    *,
    attempts: int,
    batch_size: int,
    flow_steps: int,
    diffusion_steps: int,
    birth_capacity: int,
    latent_noise_scale: float,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int, float, int, int]]:
    """Sample sparse deltas without target/oracle access or molecular repair."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[tuple[str | None, int, float, int, int]] = []
    model.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        tokens = torch.from_numpy(np.repeat(condition_tokens[None, ...], count, axis=0)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            latent = torch.randn(
                count,
                model.transport_dim,
                generator=generator,
                device=device,
                dtype=source_node.dtype,
            ) * float(latent_noise_scale)
            for flow_index in range(int(flow_steps)):
                time = torch.full(
                    (count,),
                    (flow_index + 0.5) / max(1, int(flow_steps)),
                    device=device,
                    dtype=source_node.dtype,
                )
                latent = latent + model.transport_velocity(
                    latent, time, source_node, source["node_mask"], tokens
                ) / max(1, int(flow_steps))
            condition = model.route_condition(tokens)
            working = full_graph.working_node_mask(source["node_mask"], int(birth_capacity))
            working_pairs = full_graph.upper_working_pairs(working)
            node_actions = torch.full_like(
                source["atomic_number"], model.denoiser.node_mask_id
            )
            node_actions = torch.where(
                working, node_actions, torch.full_like(node_actions, NODE_KEEP)
            )
            edge_actions = torch.full_like(source["bond"], model.denoiser.edge_mask_id)
            symmetric_pairs = working_pairs | working_pairs.transpose(1, 2)
            edge_actions = torch.where(
                symmetric_pairs, edge_actions, torch.full_like(edge_actions, EDGE_KEEP)
            )
            source_active = source["atomic_number"].gt(0)
            source_bond = source["bond"].gt(graph.BOND_NONE)
            node_action_count = model.denoiser.node_mask_id
            edge_action_count = model.denoiser.edge_mask_id
            for reverse_index in range(int(diffusion_steps), 0, -1):
                time = torch.full(
                    (count,),
                    reverse_index / max(1, int(diffusion_steps)),
                    device=device,
                    dtype=source_node.dtype,
                )
                node_logits, edge_logits = model.denoiser(
                    node_actions,
                    edge_actions,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    time,
                    condition,
                    latent,
                )
                node_legal = legal_node_action_mask(source_active, node_action_count)
                node_logits = node_logits.float().masked_fill(~node_legal, -torch.inf)
                sampled_node, node_confidence = full_graph.sample_categorical(
                    node_logits, generator, temperature
                )
                sampled_node = torch.where(
                    working, sampled_node, torch.full_like(sampled_node, NODE_KEEP)
                )
                predicted_active = action_active_nodes(source_active, sampled_node)
                edge_legal = legal_edge_action_mask(
                    source_bond, predicted_active, edge_action_count
                )
                edge_logits = edge_logits.float().masked_fill(~edge_legal, -torch.inf)
                sampled_edge, edge_confidence = full_graph.sample_categorical(
                    edge_logits, generator, temperature
                )
                sampled_edge = torch.where(
                    working_pairs, sampled_edge, torch.full_like(sampled_edge, EDGE_KEEP)
                )
                sampled_edge = sampled_edge + sampled_edge.transpose(1, 2)
                edge_confidence = torch.where(
                    working_pairs, edge_confidence, torch.zeros_like(edge_confidence)
                )
                edge_confidence = edge_confidence + edge_confidence.transpose(1, 2)
                fraction = (reverse_index - 1) / max(1, int(diffusion_steps))
                node_actions = full_graph.remask_low_confidence(
                    sampled_node,
                    node_confidence,
                    working,
                    model.denoiser.node_mask_id,
                    fraction,
                )
                edge_actions = full_graph.remask_low_confidence(
                    sampled_edge,
                    edge_confidence,
                    working_pairs,
                    model.denoiser.edge_mask_id,
                    fraction,
                )
                edge_actions = torch.where(
                    working_pairs,
                    edge_actions,
                    torch.full_like(edge_actions, EDGE_KEEP),
                )
                edge_actions = edge_actions + edge_actions.transpose(1, 2)
            result = apply_delta_actions(source, node_actions, edge_actions, vocabulary)

        prediction = {key: value.detach().cpu().numpy() for key, value in result.items()}
        upper = torch.triu(
            torch.ones(source["bond"].shape[1:], dtype=torch.bool), diagonal=1
        )
        latent_norm = latent.float().norm(dim=1).detach().cpu().tolist()
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append(
                (
                    smiles,
                    int((prediction["atomic_number"][index] > 0).sum()),
                    float(latent_norm[index]),
                    int(node_actions[index].ne(NODE_KEEP).sum()),
                    int((edge_actions[index].ne(EDGE_KEEP).detach().cpu() & upper).sum()),
                )
            )
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    model: full_graph.ContinuousDiscreteGraphDiffusion,
    representation,
    vocabulary: Mapping[str, object],
    pairs: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        generated = sample_from_source(
            model,
            representation,
            vocabulary,
            pair.source,
            pair.condition,
            attempts=int(args.num_attempts),
            batch_size=int(args.sample_batch_size),
            flow_steps=int(args.flow_steps),
            diffusion_steps=int(args.diffusion_steps),
            birth_capacity=int(args.birth_capacity),
            latent_noise_scale=float(args.latent_noise_scale),
            temperature=float(args.sample_temperature),
            device=device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        source_copy_target = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        for rank, (smiles, atom_count, latent_norm, node_edits, edge_edits) in enumerate(
            generated, start=1
        ):
            canonical = graph.canonical_smiles(smiles or "")
            valid = bool(canonical)
            source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            fraction, _, evaluated, all_success = unified.instruction_success_and_distance(
                pair.row, canonical or "", task_specs=specs
            )
            similarity_success = bool(source_tanimoto is not None and source_tanimoto >= 0.4)
            candidate_rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "latent_norm": float(latent_norm),
                    "node_edit_count": int(node_edits),
                    "edge_edit_count": int(edge_edits),
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": int(atom_count),
                    "valid": valid,
                    "source_tanimoto": float(source_tanimoto or 0.0),
                    "target_tanimoto": float(target_tanimoto or 0.0),
                    "source_copy_target_tanimoto": float(source_copy_target),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(all_success),
                    "strict_success": bool(all_success and similarity_success),
                    "source_similarity_success": similarity_success,
                }
            )
    metrics = base.summarize_candidates(candidate_rows, int(args.num_attempts))
    metrics["mean_latent_norm"] = float(np.mean([row["latent_norm"] for row in candidate_rows]))
    metrics["mean_node_edits"] = float(np.mean([row["node_edit_count"] for row in candidate_rows]))
    metrics["mean_edge_edits"] = float(np.mean([row["edge_edit_count"] for row in candidate_rows]))
    metrics["source_copy_rate"] = sum(
        str(row["generated_smiles"]) == graph.canonical_smiles(str(row["source_smiles"]))
        for row in candidate_rows
    ) / max(1, len(candidate_rows))
    return candidate_rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("The protocol requires exactly 20 raw attempts per condition")
    if int(args.diffusion_steps) < 2:
        raise ValueError("Delta diffusion requires at least two reverse steps")
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
    train_pairs, train_counts = base.build_pairs(
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
    if len(train_pairs) < 32:
        raise ValueError(f"Need at least 32 train pairs, found {len(train_pairs)}")
    for pair in [*train_pairs, *validation_pairs]:
        pair.condition = hierarchical.property_latent_slot_tokens(pair.row, int(args.condition_dim))
    vocabulary = full_graph.build_joint_state_vocabulary(train_pairs)
    node_action_count, edge_action_count = action_space_sizes(vocabulary)
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
    history = train_model(model, representation, train_pairs, vocabulary, args, device)
    candidate_rows, metrics = evaluate(
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
    train_sources = {pair.source_smiles for pair in train_pairs}
    train_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in train_pairs}
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
        "selected_train_pairs": len(train_pairs),
        "selected_validation_pairs": len(validation_pairs),
        "train_filter_counts": train_counts,
        "validation_filter_counts": validation_counts,
        "historical_validation_filter_counts": excluded_counts,
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_validation_pair_overlap": len(train_pair_keys & validation_pair_keys),
        "historical_validation_source_overlap": len(excluded_sources & validation_sources),
        "historical_validation_pair_overlap": len(excluded_pair_keys & validation_pair_keys),
        "property_counts": sorted(allowed_counts),
        "continuous_transport_latent": True,
        "conditional_flow_matching": True,
        "set_compositional_unary_property_fields": True,
        "symmetric_pairwise_property_fields": True,
        "property_order_permutation_invariant": True,
        "source_relative_sparse_delta_diffusion": True,
        "full_target_graph_diffusion": False,
        "source_graph_exact_invariant_base": True,
        "joint_node_action_tokens": True,
        "joint_edge_action_tokens": True,
        "train_only_action_payload_vocabulary": True,
        "state_vocabulary_sha256": vocabulary["sha256"],
        "node_action_count": node_action_count,
        "edge_action_count": edge_action_count,
        "diffusion_steps": int(args.diffusion_steps),
        "fixed_target_blind_birth_capacity": int(args.birth_capacity),
        "latent_usage_contrast": True,
        "latent_usage_margin": float(args.latent_usage_margin),
        "latent_variance_floor": float(args.latent_min_std),
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "posthoc_molecule_repair": False,
        "valence_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "source_target_mcs_alignment_training_only": True,
        "train_selection_seed": int(args.train_selection_seed),
        "validation_selection_seed": int(args.validation_selection_seed),
        "validation_exclusion_seed": int(args.validation_exclusion_seed),
    }
    checkpoint_path = args.output_dir / "source_relative_delta_diffusion.pt"
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
        "training": history,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "scale_source_relative_delta_diffusion_to_unified_2p_7p"
            if not failures
            else "diagnose_delta_support_vs_edit_calibration"
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
