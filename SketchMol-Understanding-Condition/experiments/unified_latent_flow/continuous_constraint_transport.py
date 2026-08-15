#!/usr/bin/env python3
"""Set-compositional continuous graph-latent transport pilot.

This experiment removes the discrete constraint/motif VQ bottleneck.  A
train-only posterior embeds an aligned source-to-target graph delta into one
continuous transport endpoint.  Conditional flow matching learns to transport
Gaussian noise to that endpoint using a source field, a normalized sum of
unary property fields, and symmetric pairwise property-interaction fields.

Generation receives only the source graph and sanitized property tokens.  It
integrates 20 independent latent trajectories and decodes each trajectory with
the existing source-anchored motif graph decoder.  Validation targets and
property scorers are opened only after the 20 raw candidates are frozen.  No
VQ codebook, candidate library, selector, finalizer, oracle reranking, or
post-hoc molecule repair is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hierarchical_vq_motif_graph_flow as hierarchical


base = hierarchical.base
belief = hierarchical.belief
graph = hierarchical.graph
unified = hierarchical.unified
vq = hierarchical.vq

PROTOCOL = "set_compositional_continuous_constraint_transport_pilot_v18"


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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--flow-loss-weight", type=float, default=0.50)
    parser.add_argument("--latent-usage-weight", type=float, default=0.20)
    parser.add_argument("--latent-usage-margin", type=float, default=0.20)
    parser.add_argument("--latent-variance-weight", type=float, default=0.10)
    parser.add_argument("--latent-min-std", type=float, default=0.20)
    parser.add_argument("--latent-noise-scale", type=float, default=1.0)
    parser.add_argument("--edit-gate-loss-weight", type=float, default=0.50)
    parser.add_argument("--delta-loss-weight", type=float, default=0.50)
    parser.add_argument("--valence-budget-loss-weight", type=float, default=0.25)
    parser.add_argument("--motif-atom-count-loss-weight", type=float, default=0.25)
    parser.add_argument("--flow-steps", type=int, default=8)
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
    parser.add_argument("--seed", type=int, default=1751)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = max(1, self.dim // 2)
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        angle = time[:, None] * frequency[None, :]
        value = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return F.pad(value, (0, max(0, self.dim - value.shape[-1])))[:, : self.dim]


class CompositionalTransportVelocity(nn.Module):
    """Permutation-invariant unary and pairwise constraint vector fields."""

    def __init__(
        self,
        transport_dim: int,
        source_dim: int,
        condition_dim: int,
        hidden_dim: int,
        property_count: int,
    ) -> None:
        super().__init__()
        self.property_count = int(property_count)
        self.time = nn.Sequential(
            TimeEmbedding(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        common_dim = transport_dim + source_dim + condition_dim
        self.global_field = nn.Sequential(
            nn.LayerNorm(common_dim + condition_dim),
            nn.Linear(common_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, transport_dim),
        )
        self.unary_field = nn.Sequential(
            nn.LayerNorm(common_dim + condition_dim),
            nn.Linear(common_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, transport_dim),
        )
        self.pair_field = nn.Sequential(
            nn.LayerNorm(common_dim + condition_dim * 3),
            nn.Linear(common_dim + condition_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, transport_dim),
        )
        # The base source/global request field is available immediately.  The
        # compositional corrections enter through stable, zero-start residuals.
        nn.init.zeros_(self.unary_field[-1].weight)
        nn.init.zeros_(self.unary_field[-1].bias)
        nn.init.zeros_(self.pair_field[-1].weight)
        nn.init.zeros_(self.pair_field[-1].bias)

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        source: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.property_count + 1:
            raise ValueError(
                "Expected one global token plus one slot per property, got "
                f"{tuple(tokens.shape)}"
            )
        time_value = self.time(time)
        common = torch.cat([latent, source, time_value], dim=-1)
        global_token = tokens[:, 0, :]
        velocity = self.global_field(torch.cat([common, global_token], dim=-1))

        slots = tokens[:, 1:, :]
        active = slots.abs().sum(dim=-1).gt(0)
        unary_common = common[:, None, :].expand(-1, self.property_count, -1)
        unary = self.unary_field(torch.cat([unary_common, slots], dim=-1))
        unary_mask = active.unsqueeze(-1).to(unary.dtype)
        active_count = unary_mask.sum(dim=1).clamp_min(1.0)
        velocity = velocity + (unary * unary_mask).sum(dim=1) / active_count.sqrt()

        left = slots.unsqueeze(2)
        right = slots.unsqueeze(1)
        symmetric = torch.cat(
            [left + right, left * right, (left - right).abs()], dim=-1
        )
        pair_common = common[:, None, None, :].expand(
            -1, self.property_count, self.property_count, -1
        )
        pair = self.pair_field(torch.cat([pair_common, symmetric], dim=-1))
        upper = torch.triu(
            torch.ones(
                self.property_count,
                self.property_count,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        pair_mask = (
            active.unsqueeze(2) & active.unsqueeze(1) & upper.unsqueeze(0)
        ).unsqueeze(-1)
        pair_weight = pair_mask.to(pair.dtype)
        pair_count = pair_weight.sum(dim=(1, 2)).clamp_min(1.0)
        velocity = velocity + (pair * pair_weight).sum(dim=(1, 2)) / pair_count.sqrt()
        return velocity


class ContinuousConstraintTransport(nn.Module):
    """Train-only delta posterior, compositional flow, and motif decoder."""

    source_anchored = True
    connected_region = True
    categorical_delta = True
    valence_budget = True
    motif_attachment = True

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
        self.velocity = CompositionalTransportVelocity(
            transport_dim,
            node_dim,
            condition_dim,
            hidden_dim,
            property_count,
        )
        self.decoder = hierarchical.SourceAnchoredEndpointField(
            node_dim,
            edge_dim,
            condition_dim + transport_dim,
            hidden_dim,
            max_atoms,
        )

    @staticmethod
    def source_pool(
        source_node: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        weight = source_mask.unsqueeze(-1).to(source_node.dtype)
        return (source_node * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

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
            torch.ones(nodes, nodes, device=union_node.device, dtype=torch.bool),
            diagonal=1,
        )
        union_edge = (
            upper.unsqueeze(0)
            & union_node[:, :, None]
            & union_node[:, None, :]
        )
        global_node = hierarchical.masked_pool(
            target_node - source_node, union_node, (1,)
        )
        global_edge = hierarchical.masked_pool(
            target_edge - source_edge, union_edge, (1, 2)
        )
        local_node = hierarchical.masked_pool(
            target_node - source_node, node_changed, (1,)
        )
        local_edge = hierarchical.masked_pool(
            target_edge - source_edge, edge_changed, (1, 2)
        )
        return self.posterior(
            torch.cat(
                [global_node, global_edge, local_node, local_edge, condition],
                dim=-1,
            )
        )

    def transport_velocity(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        source_node: torch.Tensor,
        source_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity(
            latent, time, self.source_pool(source_node, source_mask), tokens
        )

    def decode_endpoint(
        self,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        birth_rank: torch.Tensor,
        condition: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        combined = torch.cat([condition, latent], dim=-1)
        time = torch.zeros(
            source_node.shape[0], device=source_node.device, dtype=source_node.dtype
        )
        return self.decoder(
            source_node,
            source_edge,
            source_node,
            source_edge,
            source_mask,
            source_mask,
            birth_rank,
            time,
            combined,
        )


def train_flow(
    flow: ContinuousConstraintTransport,
    representation,
    pairs: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(pairs)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        flow.train()
        for start in range(0, len(order), int(args.batch_size)):
            items = [pairs[index] for index in order[start : start + int(args.batch_size)]]
            collated = base.pair_collate(items)
            source = base.move_graph_batch(collated["source"], device)
            target = base.move_graph_batch(collated["target"], device)
            tokens = collated["condition"].to(device)
            node_target, node_eligible, edge_target, edge_eligible = (
                hierarchical.change_targets(source, target)
            )
            region_target = hierarchical.connected_region_target(
                node_target, edge_target
            )
            (
                node_delta_target,
                node_delta_eligible,
                edge_delta_target,
                edge_delta_eligible,
            ) = hierarchical.categorical_delta_targets(source, target)
            valence_target = hierarchical.valence_budget_targets(target)
            valence_eligible = (
                source["node_mask"].bool() | target["node_mask"].bool()
            )
            motif_atom_count_target = (
                region_target & target["node_mask"].bool()
            ).sum(dim=1).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                target_node, target_edge = representation.encode(target)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                condition = flow.route_condition(tokens)
                endpoint = flow.posterior_endpoint(
                    source,
                    target,
                    source_node,
                    source_edge,
                    target_node,
                    target_edge,
                    condition,
                )
                noise = torch.randn_like(endpoint) * float(args.latent_noise_scale)
                time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
                current = (
                    (1.0 - time[:, None]) * noise + time[:, None] * endpoint
                )
                velocity = flow.transport_velocity(
                    current, time, source_node, source["node_mask"], tokens
                )
                target_velocity = endpoint - noise
                flow_loss = F.mse_loss(velocity.float(), target_velocity.float())
                predicted_endpoint = current + (1.0 - time[:, None]) * velocity
                decoded = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    belief.source_birth_ranks(source["node_mask"]),
                    condition,
                    predicted_endpoint,
                )
                endpoint_node, endpoint_edge = decoded[:2]
                endpoint_loss, parts = graph.reconstruction_loss(
                    representation.decode(endpoint_node, endpoint_edge),
                    target,
                    endpoint_node,
                    geometry_weight=0.0,
                )
                auxiliary = hierarchical.decoder_auxiliary_losses(
                    flow,
                    decoded,
                    region_target,
                    node_target,
                    node_eligible,
                    edge_target,
                    edge_eligible,
                    node_delta_target,
                    node_delta_eligible,
                    edge_delta_target,
                    edge_delta_eligible,
                    valence_target,
                    valence_eligible,
                    motif_atom_count_target,
                    args,
                )
                structured_loss = endpoint_loss + auxiliary[-1]

                wrong_latent = torch.roll(predicted_endpoint, shifts=1, dims=0)
                wrong_decoded = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    belief.source_birth_ranks(source["node_mask"]),
                    condition,
                    wrong_latent,
                )
                wrong_node, wrong_edge = wrong_decoded[:2]
                wrong_loss, _ = graph.reconstruction_loss(
                    representation.decode(wrong_node, wrong_edge),
                    target,
                    wrong_node,
                    geometry_weight=0.0,
                )
                latent_usage = F.relu(
                    float(args.latent_usage_margin)
                    + endpoint_loss
                    - wrong_loss
                )
                latent_std = endpoint.float().std(dim=0, unbiased=False)
                variance_loss = F.relu(
                    float(args.latent_min_std) - latent_std
                ).mean()
                loss = (
                    structured_loss
                    + float(args.flow_loss_weight) * flow_loss
                    + float(args.latent_usage_weight) * latent_usage
                    + float(args.latent_variance_weight) * variance_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()
            for name, value in parts.items():
                totals[name] += float(value)
            totals["total_loss"] += float(loss.detach())
            totals["structured_loss"] += float(structured_loss.detach())
            totals["flow_matching_loss"] += float(flow_loss.detach())
            totals["latent_usage_loss"] += float(latent_usage.detach())
            totals["latent_variance_loss"] += float(variance_loss.detach())
            totals["posterior_std"] += float(latent_std.mean().detach())
            totals["node_gate_loss"] += float(auxiliary[0].detach())
            totals["region_size_loss"] += float(auxiliary[2].detach())
            totals["node_delta_loss"] += float(auxiliary[3].detach())
            totals["edge_delta_loss"] += float(auxiliary[4].detach())
            totals["valence_budget_loss"] += float(auxiliary[5].detach())
            totals["motif_atom_count_loss"] += float(auxiliary[6].detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{
                name: value / max(1, batches)
                for name, value in totals.items()
            },
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def sample_from_source(
    flow: ContinuousConstraintTransport,
    representation,
    source_example,
    condition_tokens: np.ndarray,
    *,
    attempts: int,
    batch_size: int,
    flow_steps: int,
    latent_noise_scale: float,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int, float, int, int]]:
    """Integrate continuous latents without target or property-oracle access."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[tuple[str | None, int, float, int, int]] = []
    flow.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = base.move_graph_batch(
            graph.collate([source_example] * count), device
        )
        tokens = torch.from_numpy(
            np.repeat(condition_tokens[None, ...], count, axis=0)
        ).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            source_node, source_edge = representation.encode(source)
            latent = torch.randn(
                count,
                flow.transport_dim,
                generator=generator,
                device=device,
                dtype=source_node.dtype,
            ) * float(latent_noise_scale)
            for step in range(int(flow_steps)):
                time = torch.full(
                    (count,),
                    (step + 0.5) / max(1, int(flow_steps)),
                    device=device,
                    dtype=source_node.dtype,
                )
                latent = latent + flow.transport_velocity(
                    latent, time, source_node, source["node_mask"], tokens
                ) / max(1, int(flow_steps))
            routed_condition = flow.route_condition(tokens)
            decoded = flow.decode_endpoint(
                source_node,
                source_edge,
                source["node_mask"],
                belief.source_birth_ranks(source["node_mask"]),
                routed_condition,
                latent,
            )
            endpoint_node, endpoint_edge = decoded[:2]
            logits = representation.decode(endpoint_node, endpoint_edge)

        candidate_nodes = {
            key: logits[key].argmax(dim=-1) for key in vq.NODE_FIELDS
        }
        nodes = source["node_mask"].shape[1]
        upper = torch.triu(
            torch.ones(nodes, nodes, device=device, dtype=torch.bool), diagonal=1
        )
        candidate_edges: dict[str, torch.Tensor] = {}
        for key in vq.EDGE_FIELDS:
            candidate = logits[key].argmax(dim=-1)
            candidate = torch.where(
                upper.unsqueeze(0), candidate, torch.zeros_like(candidate)
            )
            candidate_edges[key] = candidate + candidate.transpose(1, 2)
        node_eligible = source["node_mask"].bool() | candidate_nodes[
            "atomic_number"
        ].gt(0)
        adjacency = source["bond"].gt(graph.BOND_NONE) | candidate_edges[
            "bond"
        ].gt(graph.BOND_NONE)
        adjacency &= node_eligible[:, :, None] & node_eligible[:, None, :]
        diagonal = torch.eye(nodes, device=device, dtype=torch.bool).unsqueeze(0)
        adjacency &= ~diagonal
        region_size = decoded[4].argmax(dim=-1)
        node_edit = hierarchical.project_connected_region(
            decoded[2], region_size, node_eligible, adjacency
        )

        delta_nodes = dict(candidate_nodes)
        delta_nodes["atomic_number"] = (
            logits["atomic_number"][..., 1:].argmax(dim=-1) + 1
        )
        delta_edges = dict(candidate_edges)
        set_bond = logits["bond"][..., 1:].argmax(dim=-1) + 1
        set_bond = torch.where(
            upper.unsqueeze(0), set_bond, torch.zeros_like(set_bond)
        )
        delta_edges["bond"] = set_bond + set_bond.transpose(1, 2)
        result, node_edit, edge_edit_upper = (
            hierarchical.apply_motif_attachment_graph_delta(
                source,
                delta_nodes,
                delta_edges,
                logits["bond"],
                decoded,
                node_edit,
                upper,
                candidate_edges["bond"].gt(graph.BOND_NONE),
            )
        )
        prediction = {
            key: result[key].detach().cpu().numpy()
            for key in (*vq.NODE_FIELDS, *vq.EDGE_FIELDS)
        }
        atomic = prediction["atomic_number"] > 0
        node_edit_count = node_edit.sum(dim=1).detach().cpu().tolist()
        edge_edit_count = (
            edge_edit_upper.sum(dim=(1, 2)).detach().cpu().tolist()
        )
        latent_norm = latent.float().norm(dim=1).detach().cpu().tolist()
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append(
                (
                    smiles,
                    int(atomic[index].sum()),
                    float(latent_norm[index]),
                    int(node_edit_count[index]),
                    int(edge_edit_count[index]),
                )
            )
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    flow: ContinuousConstraintTransport,
    representation,
    pairs: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        generated = sample_from_source(
            flow,
            representation,
            pair.source,
            pair.condition,
            attempts=int(args.num_attempts),
            batch_size=int(args.sample_batch_size),
            flow_steps=int(args.flow_steps),
            latent_noise_scale=float(args.latent_noise_scale),
            device=device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        source_copy_target = (
            graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        )
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        for rank, (
            smiles,
            predicted_count,
            latent_norm,
            node_edit_count,
            edge_edit_count,
        ) in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(smiles or "")
            valid = bool(canonical)
            source_tanimoto = (
                graph.morgan_tanimoto(pair.source_smiles, canonical)
                if valid
                else None
            )
            target_tanimoto = (
                graph.morgan_tanimoto(pair.target_smiles, canonical)
                if valid
                else None
            )
            fraction, _, evaluated, all_success = (
                unified.instruction_success_and_distance(
                    pair.row, canonical or "", task_specs=specs
                )
            )
            similarity_success = bool(
                source_tanimoto is not None and source_tanimoto >= 0.4
            )
            candidate_rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "latent_norm": float(latent_norm),
                    "node_edit_count": int(node_edit_count),
                    "edge_edit_count": int(edge_edit_count),
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": int(predicted_count),
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
    metrics["mean_latent_norm"] = float(
        np.mean([float(row["latent_norm"]) for row in candidate_rows])
    )
    metrics["std_latent_norm"] = float(
        np.std([float(row["latent_norm"]) for row in candidate_rows])
    )
    metrics["mean_node_edits"] = float(
        np.mean([int(row["node_edit_count"]) for row in candidate_rows])
    )
    metrics["mean_edge_edits"] = float(
        np.mean([int(row["edge_edit_count"]) for row in candidate_rows])
    )
    metrics["source_copy_rate"] = sum(
        str(row["generated_smiles"])
        == graph.canonical_smiles(str(row["source_smiles"]))
        for row in candidate_rows
    ) / max(1, len(candidate_rows))
    return candidate_rows, metrics


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
    excluded_pair_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in excluded_pairs
    }
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
    validation_pair_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in validation_pairs
    }
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
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(args.condition_dim)
        )

    flow = ContinuousConstraintTransport(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        transport_dim=int(args.transport_dim),
        hidden_dim=int(args.hidden_dim),
        max_atoms=int(config["max_atoms"]),
        property_count=len(unified.PROPERTY_COLUMNS),
    ).to(device)
    history = train_flow(flow, representation, train_pairs, args, device)
    candidate_rows, metrics = evaluate(
        flow, representation, validation_pairs, args, device
    )
    three_property_strict = float(
        metrics["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    checks = {
        "exact_attempts": {
            "value": metrics["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {
            "value": metrics["validity"],
            "threshold": float(args.gate_validity),
        },
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
    train_pair_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in train_pairs
    }
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "train_selection_seed": int(args.train_selection_seed),
        "validation_selection_seed": int(args.validation_selection_seed),
        "validation_exclusion_seed": int(args.validation_exclusion_seed),
        "heldout_role": "development_not_final_audit",
        "device": str(device),
        "representation_protocol": representation_summary.get("protocol"),
        "representation_gate_passed": bool(
            representation_summary.get("gate", {}).get("passed")
        ),
        "representation_checkpoint": str(args.representation_checkpoint),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "train_csv": str(args.train_csv),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "selected_train_pairs": len(train_pairs),
        "selected_validation_pairs": len(validation_pairs),
        "train_filter_counts": train_counts,
        "validation_filter_counts": validation_counts,
        "historical_validation_filter_counts": excluded_counts,
        "train_validation_source_overlap": len(
            train_sources & validation_sources
        ),
        "train_validation_pair_overlap": len(
            train_pair_keys & validation_pair_keys
        ),
        "historical_validation_source_overlap": len(
            excluded_sources & validation_sources
        ),
        "historical_validation_pair_overlap": len(
            excluded_pair_keys & validation_pair_keys
        ),
        "property_counts": sorted(allowed_counts),
        "continuous_transport_latent": True,
        "vq_codebook": False,
        "posterior_train_only": True,
        "conditional_flow_matching": True,
        "gaussian_base_distribution": True,
        "set_compositional_unary_property_fields": True,
        "symmetric_pairwise_property_fields": True,
        "property_order_permutation_invariant": True,
        "source_anchored_residual_decoder": True,
        "connected_region_decoder": True,
        "categorical_graph_delta_grammar": True,
        "motif_attachment_decoder": True,
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
        "transport_dim": int(args.transport_dim),
        "flow_steps": int(args.flow_steps),
    }
    checkpoint_path = args.output_dir / "continuous_constraint_transport.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": flow.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": int(args.condition_dim),
                "transport_dim": int(args.transport_dim),
                "hidden_dim": int(args.hidden_dim),
                "max_atoms": int(config["max_atoms"]),
                "property_count": len(unified.PROPERTY_COLUMNS),
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
        "gate": {
            "passed": not failures,
            "checks": checks,
            "failures": failures,
        },
        "next_stage": (
            "scale_continuous_transport_to_unified_2p_7p"
            if not failures
            else "change_representation_or_decoder_backbone"
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
