#!/usr/bin/env python3
"""Continuous constraint transport with an absorbing discrete graph decoder.

The train-only posterior and set-compositional continuous transport from B18
are retained.  The one-shot continuous-to-graph endpoint is replaced by a
categorical denoiser over joint atom states and joint bond states.  Training
draws a diffusion time, masks target graph states with the corresponding
absorbing probability, and predicts the clean aligned target graph.  Sampling
starts from a fully masked source-sized graph with a bounded number of birth
slots and progressively unmasks it.

Generation receives the source graph, sanitized property tokens, and Gaussian
noise only.  Exactly 20 raw attempts are frozen before development targets and
property scorers are opened.  There is no candidate library, selector,
finalizer, oracle reranking, chemistry repair, or validation-derived grammar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import continuous_constraint_transport as continuous


base = continuous.base
belief = continuous.belief
graph = continuous.graph
hierarchical = continuous.hierarchical
unified = continuous.unified
vq = continuous.vq

PROTOCOL = "compositional_absorbing_discrete_graph_diffusion_pilot_v20"
NODE_FIELDS = tuple(vq.NODE_FIELDS)
EDGE_FIELDS = tuple(vq.EDGE_FIELDS)


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
    parser.add_argument("--latent-noise-scale", type=float, default=1.0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--birth-capacity", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.85)
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
    parser.add_argument("--seed", type=int, default=1753)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _state_rows(example: object, fields: Sequence[str], edge: bool) -> list[tuple[int, ...]]:
    arrays = [np.asarray(getattr(example, field), dtype=np.int64) for field in fields]
    if edge:
        bond = np.asarray(example.bond)
        left, right = np.triu_indices(bond.shape[0], k=1)
        active = bond[left, right] > graph.BOND_NONE
        return [
            tuple(int(array[i, j]) for array in arrays)
            for i, j in zip(left[active], right[active])
        ]
    active = np.asarray(example.atomic_number) > 0
    return [
        tuple(int(array[index]) for array in arrays)
        for index in np.flatnonzero(active)
    ]


def build_joint_state_vocabulary(pairs: Sequence[object]) -> dict[str, object]:
    """Build complete atom/bond state support from selected train pairs only."""
    blank_node = (0, graph.CHARGE_OFFSET, 0, 0, 0, 0)
    blank_edge = (0, 0)
    node_states = {blank_node}
    edge_states = {blank_edge}
    for pair in pairs:
        for example in (pair.source, pair.target):
            node_states.update(_state_rows(example, NODE_FIELDS, edge=False))
            edge_states.update(_state_rows(example, EDGE_FIELDS, edge=True))
    ordered_nodes = [blank_node, *sorted(node_states - {blank_node})]
    ordered_edges = [blank_edge, *sorted(edge_states - {blank_edge})]
    payload = json.dumps(
        {"node_states": ordered_nodes, "edge_states": ordered_edges},
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "node_states": np.asarray(ordered_nodes, dtype=np.int64),
        "edge_states": np.asarray(ordered_edges, dtype=np.int64),
        "blank_node_id": 0,
        "blank_edge_id": 0,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def graph_state_ids(
    batch: Mapping[str, torch.Tensor],
    vocabulary: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map complete categorical field tuples to joint train-support IDs."""
    node_values = torch.stack([batch[field].long() for field in NODE_FIELDS], dim=-1)
    edge_values = torch.stack([batch[field].long() for field in EDGE_FIELDS], dim=-1)
    node_states = torch.as_tensor(
        np.asarray(vocabulary["node_states"]), device=node_values.device
    )
    edge_states = torch.as_tensor(
        np.asarray(vocabulary["edge_states"]), device=edge_values.device
    )
    node_match = node_values.unsqueeze(-2).eq(node_states).all(dim=-1)
    edge_match = edge_values.unsqueeze(-2).eq(edge_states).all(dim=-1)
    if not bool(node_match.any(dim=-1).all()):
        raise ValueError("Graph contains a node state outside train-only vocabulary")
    if not bool(edge_match.any(dim=-1).all()):
        raise ValueError("Graph contains an edge state outside train-only vocabulary")
    return node_match.long().argmax(dim=-1), edge_match.long().argmax(dim=-1)


def working_node_mask(
    source_mask: torch.Tensor,
    birth_capacity: int,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expose source slots plus a fixed, target-blind prefix of free birth slots."""
    source = source_mask.bool()
    free_rank = (~source).long().cumsum(dim=1)
    births = (~source) & free_rank.le(max(0, int(birth_capacity)))
    working = source | births
    if target_mask is not None:
        working |= target_mask.bool()
    return working


def upper_working_pairs(node_mask: torch.Tensor) -> torch.Tensor:
    nodes = node_mask.shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=node_mask.device, dtype=torch.bool),
        diagonal=1,
    )
    return upper.unsqueeze(0) & node_mask[:, :, None] & node_mask[:, None, :]


def corrupt_joint_states(
    node_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    node_mask: torch.Tensor,
    time: torch.Tensor,
    node_mask_id: int,
    edge_mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward absorbing process q(x_t|x_0) with alpha_bar(t)=1-t."""
    node_selected = torch.rand_like(node_ids, dtype=torch.float32).lt(time[:, None])
    node_selected &= node_mask
    edge_eligible = upper_working_pairs(node_mask)
    edge_selected = torch.rand_like(edge_ids, dtype=torch.float32).lt(
        time[:, None, None]
    ) & edge_eligible
    edge_selected = edge_selected | edge_selected.transpose(1, 2)
    noisy_node = torch.where(node_selected, node_mask_id, node_ids)
    noisy_edge = torch.where(edge_selected, edge_mask_id, edge_ids)
    return noisy_node, noisy_edge, node_selected, edge_selected


class DenseDiscreteGraphLayer(nn.Module):
    """Permutation-equivariant dense message layer for masked graph states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        source_edge: torch.Tensor,
        context: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = node.shape[1]
        sender = node[:, None, :, :].expand(-1, nodes, -1, -1)
        messages = self.message(torch.cat([sender, edge, source_edge], dim=-1))
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        diagonal = torch.eye(nodes, device=node.device, dtype=torch.bool).unsqueeze(0)
        pair_mask &= ~diagonal
        messages = messages * pair_mask.unsqueeze(-1)
        aggregate = messages.sum(dim=2) / pair_mask.sum(dim=2, keepdim=True).clamp_min(1).sqrt()
        node = self.node_norm(
            node + self.node_update(torch.cat([node, aggregate, context[:, None, :].expand_as(node)], dim=-1))
        )
        node = node * node_mask.unsqueeze(-1)
        left, right = node[:, :, None, :], node[:, None, :, :]
        edge_delta = self.edge_update(
            torch.cat(
                [edge, source_edge, left + right, (left - right).abs(), context[:, None, None, :].expand_as(edge)],
                dim=-1,
            )
        )
        edge = self.edge_norm(edge + edge_delta)
        edge = 0.5 * (edge + edge.transpose(1, 2))
        edge = edge * pair_mask.unsqueeze(-1)
        return node, edge


class JointGraphDenoiser(nn.Module):
    """Predict clean joint atom and bond states from a masked discrete graph."""

    def __init__(
        self,
        *,
        node_state_count: int,
        edge_state_count: int,
        source_node_dim: int,
        source_edge_dim: int,
        context_dim: int,
        hidden_dim: int,
        max_atoms: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.node_mask_id = int(node_state_count)
        self.edge_mask_id = int(edge_state_count)
        self.node_embedding = nn.Embedding(node_state_count + 1, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_state_count + 1, hidden_dim)
        self.source_node = nn.Linear(source_node_dim, hidden_dim)
        self.source_edge = nn.Linear(source_edge_dim, hidden_dim)
        self.birth_rank = nn.Embedding(max_atoms + 1, hidden_dim)
        self.time = nn.Sequential(
            continuous.TimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [DenseDiscreteGraphLayer(hidden_dim) for _ in range(int(layers))]
        )
        self.node_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, node_state_count)
        )
        self.edge_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_state_count),
        )

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        working_mask: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context(torch.cat([condition, latent], dim=-1)) + self.time(time)
        ranks = belief.source_birth_ranks(source_mask).clamp_max(self.birth_rank.num_embeddings - 1)
        source_node_value = self.source_node(source_node)
        source_edge_value = self.source_edge(source_edge)
        node = (
            self.node_embedding(node_ids)
            + source_node_value
            + self.birth_rank(ranks)
            + context[:, None, :]
        ) * working_mask.unsqueeze(-1)
        edge = self.edge_embedding(edge_ids) + source_edge_value
        pair_mask = working_mask[:, :, None] & working_mask[:, None, :]
        edge = edge * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            node, edge = layer(node, edge, source_edge_value, context, working_mask)
        left, right = node[:, :, None, :], node[:, None, :, :]
        edge_logits = self.edge_head(
            torch.cat([edge, source_edge_value, left + right, (left - right).abs()], dim=-1)
        )
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        return self.node_head(node), edge_logits


class ContinuousDiscreteGraphDiffusion(nn.Module):
    """Set-compositional transport followed by a categorical graph denoiser."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        condition_dim: int,
        transport_dim: int,
        hidden_dim: int,
        max_atoms: int,
        property_count: int,
        node_state_count: int,
        edge_state_count: int,
        message_layers: int,
    ) -> None:
        super().__init__()
        self.transport_dim = int(transport_dim)
        self.condition_router = hierarchical.PropertyInteractionLatentComposer(
            condition_dim, hidden_dim, property_count
        )
        posterior_input = 2 * (node_dim + edge_dim) + condition_dim
        self.posterior = nn.Sequential(
            nn.LayerNorm(posterior_input),
            nn.Linear(posterior_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, transport_dim),
            nn.Tanh(),
        )
        self.velocity = continuous.CompositionalTransportVelocity(
            transport_dim, node_dim, condition_dim, hidden_dim, property_count
        )
        self.denoiser = JointGraphDenoiser(
            node_state_count=node_state_count,
            edge_state_count=edge_state_count,
            source_node_dim=node_dim,
            source_edge_dim=edge_dim,
            context_dim=condition_dim + transport_dim,
            hidden_dim=hidden_dim,
            max_atoms=max_atoms,
            layers=message_layers,
        )

    def route_condition(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.condition_router(tokens)

    def posterior_endpoint(
        self,
        source: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        target_node: torch.Tensor,
        target_edge: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        node_changed, edge_changed = vq.change_masks(source, target)
        union_node = source["node_mask"].bool() | target["node_mask"].bool()
        nodes = union_node.shape[1]
        upper = torch.triu(
            torch.ones(nodes, nodes, device=union_node.device, dtype=torch.bool), diagonal=1
        )
        union_edge = upper.unsqueeze(0) & union_node[:, :, None] & union_node[:, None, :]
        pieces = [
            hierarchical.masked_pool(target_node - source_node, union_node, (1,)),
            hierarchical.masked_pool(target_edge - source_edge, union_edge, (1, 2)),
            hierarchical.masked_pool(target_node - source_node, node_changed, (1,)),
            hierarchical.masked_pool(target_edge - source_edge, edge_changed, (1, 2)),
            condition,
        ]
        return self.posterior(torch.cat(pieces, dim=-1))

    def transport_velocity(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        source_node: torch.Tensor,
        source_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(
            latent,
            time,
            continuous.ContinuousConstraintTransport.source_pool(source_node, source_mask),
            tokens,
        )


def balanced_categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    blank_id: int,
    blank_weight: float,
) -> torch.Tensor:
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    selected_logits = logits[eligible].float()
    selected_target = target[eligible].long()
    classes = logits.shape[-1]
    counts = torch.bincount(selected_target, minlength=classes).float()
    weights = counts.sum().sqrt() / counts.clamp_min(1.0).sqrt()
    weights = (weights / weights[counts.gt(0)].mean().clamp_min(1e-6)).clamp(0.25, 4.0)
    weights[int(blank_id)] *= float(blank_weight)
    return F.cross_entropy(selected_logits, selected_target, weight=weights)


def train_model(
    model: ContinuousDiscreteGraphDiffusion,
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

                node_ids, edge_ids = graph_state_ids(target, vocabulary)
                working = working_node_mask(
                    source["node_mask"], int(args.birth_capacity), target["node_mask"]
                )
                diffusion_index = torch.randint(
                    1, int(args.diffusion_steps) + 1, (len(items),), device=device
                )
                diffusion_time = diffusion_index.float() / max(1, int(args.diffusion_steps))
                noisy_node, noisy_edge, node_corrupted, edge_corrupted = corrupt_joint_states(
                    node_ids,
                    edge_ids,
                    working,
                    diffusion_time,
                    model.denoiser.node_mask_id,
                    model.denoiser.edge_mask_id,
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
                node_loss = balanced_categorical_loss(
                    node_logits,
                    node_ids,
                    node_corrupted,
                    int(vocabulary["blank_node_id"]),
                    0.50,
                )
                edge_loss = balanced_categorical_loss(
                    edge_logits,
                    edge_ids,
                    edge_corrupted & upper_working_pairs(working),
                    int(vocabulary["blank_edge_id"]),
                    0.20,
                )
                loss = node_loss + edge_loss + float(args.flow_loss_weight) * flow_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            with torch.no_grad():
                node_accuracy = (
                    node_logits.argmax(dim=-1)[node_corrupted].eq(node_ids[node_corrupted]).float().mean()
                    if bool(node_corrupted.any())
                    else torch.ones((), device=device)
                )
                edge_eval = edge_corrupted & upper_working_pairs(working)
                edge_accuracy = (
                    edge_logits.argmax(dim=-1)[edge_eval].eq(edge_ids[edge_eval]).float().mean()
                    if bool(edge_eval.any())
                    else torch.ones((), device=device)
                )
            totals["loss"] += float(loss.detach())
            totals["node_denoising_loss"] += float(node_loss.detach())
            totals["edge_denoising_loss"] += float(edge_loss.detach())
            totals["flow_matching_loss"] += float(flow_loss.detach())
            totals["node_masked_accuracy"] += float(node_accuracy)
            totals["edge_masked_accuracy"] += float(edge_accuracy)
            totals["posterior_std"] += float(endpoint.float().std(dim=0, unbiased=False).mean())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def sample_categorical(
    logits: torch.Tensor, generator: torch.Generator, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits.float() / max(1e-4, float(temperature)), dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    sample = torch.multinomial(flat, 1, generator=generator).reshape(logits.shape[:-1])
    confidence = probabilities.gather(-1, sample.unsqueeze(-1)).squeeze(-1)
    return sample, confidence


def remask_low_confidence(
    values: torch.Tensor,
    confidence: torch.Tensor,
    eligible: torch.Tensor,
    mask_id: int,
    fraction: float,
) -> torch.Tensor:
    result = values.clone()
    for batch_index in range(values.shape[0]):
        positions = torch.nonzero(eligible[batch_index], as_tuple=False)
        count = min(len(positions), int(round(len(positions) * max(0.0, float(fraction)))))
        if count <= 0:
            continue
        scores = confidence[batch_index][tuple(positions.transpose(0, 1))]
        selected = positions[torch.argsort(scores)[:count]]
        result[batch_index][tuple(selected.transpose(0, 1))] = int(mask_id)
    return result


def decode_joint_states(
    node_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    vocabulary: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    node_states = torch.as_tensor(
        np.asarray(vocabulary["node_states"]), device=node_ids.device
    )
    edge_states = torch.as_tensor(
        np.asarray(vocabulary["edge_states"]), device=edge_ids.device
    )
    selected_nodes = node_states[node_ids]
    selected_edges = edge_states[edge_ids]
    result = {
        field: selected_nodes[..., index] for index, field in enumerate(NODE_FIELDS)
    }
    result.update(
        {field: selected_edges[..., index] for index, field in enumerate(EDGE_FIELDS)}
    )
    active = result["atomic_number"].gt(0)
    active_pair = active[:, :, None] & active[:, None, :]
    for field in EDGE_FIELDS:
        result[field] = torch.where(active_pair, result[field], torch.zeros_like(result[field]))
    nodes = node_ids.shape[1]
    diagonal = torch.eye(nodes, device=node_ids.device, dtype=torch.bool).unsqueeze(0)
    for field in EDGE_FIELDS:
        result[field] = result[field].masked_fill(diagonal, 0)
    return result


@torch.no_grad()
def sample_from_source(
    model: ContinuousDiscreteGraphDiffusion,
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
    """Generate exact raw attempts without target, property oracle, or repair."""
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
            working = working_node_mask(source["node_mask"], int(birth_capacity))
            working_pairs = upper_working_pairs(working)
            node_ids = torch.full_like(
                source["atomic_number"], model.denoiser.node_mask_id
            )
            node_ids = torch.where(
                working,
                node_ids,
                torch.full_like(node_ids, int(vocabulary["blank_node_id"])),
            )
            edge_ids = torch.full_like(source["bond"], model.denoiser.edge_mask_id)
            symmetric_pairs = working_pairs | working_pairs.transpose(1, 2)
            edge_ids = torch.where(
                symmetric_pairs,
                edge_ids,
                torch.full_like(edge_ids, int(vocabulary["blank_edge_id"])),
            )
            for reverse_index in range(int(diffusion_steps), 0, -1):
                time = torch.full(
                    (count,),
                    reverse_index / max(1, int(diffusion_steps)),
                    device=device,
                    dtype=source_node.dtype,
                )
                node_logits, edge_logits = model.denoiser(
                    node_ids,
                    edge_ids,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    time,
                    condition,
                    latent,
                )
                sampled_node, node_confidence = sample_categorical(
                    node_logits, generator, temperature
                )
                sampled_edge, edge_confidence = sample_categorical(
                    edge_logits, generator, temperature
                )
                sampled_node = torch.where(
                    working,
                    sampled_node,
                    torch.full_like(sampled_node, int(vocabulary["blank_node_id"])),
                )
                sampled_edge = torch.where(
                    working_pairs,
                    sampled_edge,
                    torch.full_like(sampled_edge, int(vocabulary["blank_edge_id"])),
                )
                sampled_edge = sampled_edge + sampled_edge.transpose(1, 2)
                edge_confidence = torch.where(
                    working_pairs, edge_confidence, torch.zeros_like(edge_confidence)
                )
                edge_confidence = edge_confidence + edge_confidence.transpose(1, 2)
                fraction = (reverse_index - 1) / max(1, int(diffusion_steps))
                node_ids = remask_low_confidence(
                    sampled_node,
                    node_confidence,
                    working,
                    model.denoiser.node_mask_id,
                    fraction,
                )
                edge_ids = remask_low_confidence(
                    sampled_edge,
                    edge_confidence,
                    working_pairs,
                    model.denoiser.edge_mask_id,
                    fraction,
                )
                edge_ids = torch.where(
                    working_pairs,
                    edge_ids,
                    torch.full_like(edge_ids, int(vocabulary["blank_edge_id"])),
                )
                edge_ids = edge_ids + edge_ids.transpose(1, 2)
            result = decode_joint_states(node_ids, edge_ids, vocabulary)

        prediction = {key: value.detach().cpu().numpy() for key, value in result.items()}
        source_cpu = {key: source[key].detach().cpu() for key in (*NODE_FIELDS, *EDGE_FIELDS)}
        upper = torch.triu(
            torch.ones(source["bond"].shape[1:], dtype=torch.bool), diagonal=1
        )
        latent_norm = latent.float().norm(dim=1).detach().cpu().tolist()
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            node_changed = torch.zeros_like(source_cpu["atomic_number"][index], dtype=torch.bool)
            for field in NODE_FIELDS:
                node_changed |= result[field][index].detach().cpu().ne(source_cpu[field][index])
            edge_changed = torch.zeros_like(source_cpu["bond"][index], dtype=torch.bool)
            for field in EDGE_FIELDS:
                edge_changed |= result[field][index].detach().cpu().ne(source_cpu[field][index])
            outputs.append(
                (
                    smiles,
                    int((prediction["atomic_number"][index] > 0).sum()),
                    float(latent_norm[index]),
                    int(node_changed.sum()),
                    int((edge_changed & upper).sum()),
                )
            )
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    model: ContinuousDiscreteGraphDiffusion,
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
        raise ValueError("Discrete diffusion requires at least two reverse steps")
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
    vocabulary = build_joint_state_vocabulary(train_pairs)
    model = ContinuousDiscreteGraphDiffusion(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        transport_dim=int(args.transport_dim),
        hidden_dim=int(args.hidden_dim),
        max_atoms=int(config["max_atoms"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=len(vocabulary["node_states"]),
        edge_state_count=len(vocabulary["edge_states"]),
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
        "absorbing_discrete_graph_diffusion": True,
        "joint_atom_state_diffusion": True,
        "joint_bond_state_diffusion": True,
        "train_only_state_vocabulary": True,
        "state_vocabulary_sha256": vocabulary["sha256"],
        "node_state_count": len(vocabulary["node_states"]),
        "edge_state_count": len(vocabulary["edge_states"]),
        "diffusion_steps": int(args.diffusion_steps),
        "fixed_target_blind_birth_capacity": int(args.birth_capacity),
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "posthoc_molecule_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "source_target_mcs_alignment_training_only": True,
        "train_selection_seed": int(args.train_selection_seed),
        "validation_selection_seed": int(args.validation_selection_seed),
        "validation_exclusion_seed": int(args.validation_exclusion_seed),
    }
    checkpoint_path = args.output_dir / "discrete_graph_diffusion_decoder.pt"
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
            "scale_discrete_graph_diffusion_to_unified_2p_7p"
            if not failures
            else "diagnose_discrete_support_vs_denoising_calibration"
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
