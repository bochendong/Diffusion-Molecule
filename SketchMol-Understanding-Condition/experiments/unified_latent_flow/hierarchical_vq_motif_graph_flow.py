#!/usr/bin/env python3
"""Hierarchical constraint-token plus motif-token graph-latent flow pilot.

The train-only posterior quantizes a global source-to-target delta into a
constraint token, then quantizes the remaining local changed subgraph into a
motif token conditioned on that constraint.  At generation time two learned
priors use only source graph and requested properties: first constraint, then
motif.  A deterministic graph decoder consumes both tokens in one pass.  There
is no target/oracle access, GraphEditDSL action, candidate selector, independent
atom/bond sampling, finalizer, or chemistry repair during generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
VQ_PATH = SCRIPT_DIR / "vq_motif_graph_belief_flow.py"
PROTOCOL = "hierarchical_constraint_motif_vq_graph_flow_pilot_v6"
SOURCE_ANCHORED_PROTOCOL = "source_anchored_hierarchical_vq_graph_flow_pilot_v7"
CONNECTED_REGION_PROTOCOL = "connected_region_hierarchical_vq_graph_flow_pilot_v8"
CATEGORICAL_DELTA_PROTOCOL = "categorical_delta_hierarchical_vq_graph_flow_pilot_v9"
VALENCE_BUDGET_PROTOCOL = "valence_budget_hierarchical_vq_graph_flow_pilot_v10"
MOTIF_ATTACHMENT_PROTOCOL = "motif_attachment_hierarchical_vq_graph_flow_pilot_v11"
CONSTRAINT_ATTENTION_PROTOCOL = "constraint_attention_motif_graph_flow_pilot_v12"

NODE_KEEP, NODE_DELETE, NODE_BIRTH, NODE_REPLACE = range(4)
EDGE_KEEP, EDGE_DELETE, EDGE_SET = range(3)
MAX_VALENCE_UNITS = 12
BOND_VALENCE_UNITS = (0, 2, 4, 6, 3)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vq = load_module("hierarchical_vq_graph_base", VQ_PATH)
belief = vq.belief
base = vq.base
graph = vq.graph
unified = vq.unified


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
    parser.add_argument("--constraint-code-dim", type=int, default=32)
    parser.add_argument("--constraint-codebook-size", type=int, default=16)
    parser.add_argument("--motif-code-dim", type=int, default=64)
    parser.add_argument("--motif-codebook-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--prior-loss-weight", type=float, default=0.50)
    parser.add_argument("--vq-loss-weight", type=float, default=0.25)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.20)
    parser.add_argument("--contrastive-margin", type=float, default=0.20)
    parser.add_argument("--sampling-temperature", type=float, default=0.80)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--gate-validity", type=float, default=0.80)
    parser.add_argument("--gate-source-tanimoto", type=float, default=0.40)
    parser.add_argument("--gate-target-improvement-rate", type=float, default=0.25)
    parser.add_argument("--gate-strict-any20", type=float, default=0.20)
    parser.add_argument("--gate-min-constraint-codes", type=int, default=3)
    parser.add_argument("--gate-min-motif-codes", type=int, default=4)
    parser.add_argument("--source-anchored", action="store_true")
    parser.add_argument("--connected-region", action="store_true")
    parser.add_argument("--categorical-delta", action="store_true")
    parser.add_argument("--valence-budget", action="store_true")
    parser.add_argument("--motif-attachment", action="store_true")
    parser.add_argument("--condition-attention", action="store_true")
    parser.add_argument("--condition-attention-heads", type=int, default=4)
    parser.add_argument("--edit-gate-loss-weight", type=float, default=0.50)
    parser.add_argument("--delta-loss-weight", type=float, default=0.50)
    parser.add_argument("--valence-budget-loss-weight", type=float, default=0.25)
    parser.add_argument("--motif-atom-count-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1741)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def masked_pool(
    values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]
) -> torch.Tensor:
    return vq.masked_pool(values, mask, dimensions)


def perplexity(usage: Counter[int]) -> float:
    probability = np.asarray(list(usage.values()), dtype=np.float64)
    probability = probability / max(1.0, probability.sum())
    return float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))


def condition_tokens(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    """Keep the sanitized request as tokens instead of destroying it by averaging."""
    safe = base.sanitized_condition_row(row)
    fallback = unified.fallback_condition_features(safe, int(condition_dim))
    mode = unified.mode_condition_token(unified.EDIT_MODE, int(condition_dim))
    program = unified.property_program_tokens(safe, int(condition_dim))
    return np.concatenate([fallback, mode, program], axis=0).astype(np.float32)


class SourceConstraintCrossAttention(nn.Module):
    """Route individual constraint tokens through source atom queries."""

    def __init__(self, node_dim: int, condition_dim: int, heads: int) -> None:
        super().__init__()
        valid_heads = max(
            value for value in range(1, max(1, int(heads)) + 1) if int(node_dim) % value == 0
        )
        self.condition_projection = nn.Linear(condition_dim, node_dim)
        self.query_norm = nn.LayerNorm(node_dim)
        self.cross_attention = nn.MultiheadAttention(
            node_dim, valid_heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(node_dim + condition_dim),
            nn.Linear(node_dim + condition_dim, condition_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        source_node: torch.Tensor,
        source_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.condition_projection(tokens)
        attended, _ = self.cross_attention(
            self.query_norm(source_node), memory, memory, need_weights=False
        )
        mask = source_mask.unsqueeze(-1).to(attended.dtype)
        pooled = ((source_node + attended) * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1.0)
        token_mean = tokens.mean(dim=1)
        return self.output(torch.cat([pooled, token_mean], dim=-1))


class SourceAnchoredEndpointField(nn.Module):
    """Decode endpoint categories and learned source-relative edit blocks.

    The edit logits are part of the latent-conditioned decoder.  At generation
    time their deterministic sign chooses whether an entire atom or bond block
    comes from the decoded endpoint or is copied exactly from the source.  This
    is neither a chemistry repair nor a post-hoc candidate operation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        condition_dim: int,
        hidden_dim: int,
        max_atoms: int,
    ) -> None:
        super().__init__()
        self.endpoint = belief.CategoricalEndpointField(
            node_dim, edge_dim, condition_dim, hidden_dim, max_atoms
        )
        self.gate_condition = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.node_change = nn.Sequential(
            nn.LayerNorm(node_dim * 4),
            nn.Linear(node_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        edge_input = edge_dim * 2 + node_dim * 3
        self.edge_change = nn.Sequential(
            nn.LayerNorm(edge_input),
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.region_size = nn.Sequential(
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atoms + 1),
        )
        self.node_delta_operation = nn.Sequential(
            nn.LayerNorm(node_dim * 4),
            nn.Linear(node_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.edge_delta_operation = nn.Sequential(
            nn.LayerNorm(edge_input),
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.node_valence_budget = nn.Sequential(
            nn.LayerNorm(node_dim * 4),
            nn.Linear(node_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, MAX_VALENCE_UNITS + 1),
        )
        self.motif_atom_count = nn.Sequential(
            nn.LayerNorm(node_dim * 2),
            nn.Linear(node_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atoms + 1),
        )

    def forward(
        self,
        current_node: torch.Tensor,
        current_edge: torch.Tensor,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        current_mask: torch.Tensor,
        source_mask: torch.Tensor,
        birth_rank: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        endpoint_node, endpoint_edge = self.endpoint(
            current_node,
            current_edge,
            source_node,
            source_edge,
            current_mask,
            source_mask,
            birth_rank,
            time,
            condition,
        )
        birth = self.endpoint.birth_rank(birth_rank)
        context = self.gate_condition(condition)[:, None, :].expand_as(source_node)
        node_input = torch.cat(
            [source_node, endpoint_node, birth, context], dim=-1
        )
        node_change_logits = self.node_change(node_input).squeeze(-1)
        source_left = source_node[:, :, None, :]
        source_right = source_node[:, None, :, :]
        endpoint_left = endpoint_node[:, :, None, :]
        endpoint_right = endpoint_node[:, None, :, :]
        edge_context = context[:, :, None, :] + context[:, None, :, :]
        edge_input = torch.cat(
            [
                source_edge,
                endpoint_edge,
                source_left + source_right,
                endpoint_left + endpoint_right,
                edge_context,
            ],
            dim=-1,
        )
        edge_change_logits = self.edge_change(edge_input).squeeze(-1)
        edge_change_logits = 0.5 * (
            edge_change_logits + edge_change_logits.transpose(1, 2)
        )
        source_total = (source_node * source_mask.unsqueeze(-1)).sum(dim=1)
        source_pool = source_total / source_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        region_size_logits = self.region_size(
            torch.cat([source_pool, context[:, 0, :]], dim=-1)
        )
        node_delta_logits = self.node_delta_operation(node_input)
        edge_delta_logits = self.edge_delta_operation(edge_input)
        edge_delta_logits = 0.5 * (
            edge_delta_logits + edge_delta_logits.transpose(1, 2)
        )
        valence_budget_logits = self.node_valence_budget(node_input)
        motif_atom_count_logits = self.motif_atom_count(
            torch.cat([source_pool, context[:, 0, :]], dim=-1)
        )
        return (
            endpoint_node,
            endpoint_edge,
            node_change_logits,
            edge_change_logits,
            region_size_logits,
            node_delta_logits,
            edge_delta_logits,
            valence_budget_logits,
            motif_atom_count_logits,
        )


class HierarchicalVQGraphFlow(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        condition_dim: int,
        constraint_code_dim: int,
        constraint_codebook_size: int,
        motif_code_dim: int,
        motif_codebook_size: int,
        hidden_dim: int,
        max_atoms: int,
        source_anchored: bool = False,
        connected_region: bool = False,
        categorical_delta: bool = False,
        valence_budget: bool = False,
        motif_attachment: bool = False,
        condition_attention: bool = False,
        condition_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.source_anchored = bool(source_anchored)
        self.connected_region = bool(connected_region)
        self.categorical_delta = bool(categorical_delta)
        self.valence_budget = bool(valence_budget)
        self.motif_attachment = bool(motif_attachment)
        self.condition_attention = bool(condition_attention)
        if self.connected_region and not self.source_anchored:
            raise ValueError("Connected-region decoding requires source anchoring")
        if self.categorical_delta and not self.connected_region:
            raise ValueError("Categorical graph deltas require connected-region decoding")
        if self.valence_budget and not self.categorical_delta:
            raise ValueError("Valence budgets require categorical graph deltas")
        if self.motif_attachment and not self.valence_budget:
            raise ValueError("Motif attachment requires valence-budget decoding")
        self.constraint_codebook_size = int(constraint_codebook_size)
        self.motif_codebook_size = int(motif_codebook_size)
        self.condition_router = (
            SourceConstraintCrossAttention(
                node_dim, condition_dim, condition_attention_heads
            )
            if self.condition_attention
            else None
        )
        global_input = node_dim + edge_dim + condition_dim
        local_input = node_dim + edge_dim + condition_dim + constraint_code_dim
        self.constraint_posterior = nn.Sequential(
            nn.LayerNorm(global_input),
            nn.Linear(global_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, constraint_code_dim),
        )
        self.motif_posterior = nn.Sequential(
            nn.LayerNorm(local_input),
            nn.Linear(local_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, motif_code_dim),
        )
        self.constraint_codebook = nn.Embedding(
            constraint_codebook_size, constraint_code_dim
        )
        self.motif_codebook = nn.Embedding(motif_codebook_size, motif_code_dim)
        nn.init.normal_(self.constraint_codebook.weight, std=0.10)
        nn.init.normal_(self.motif_codebook.weight, std=0.10)
        self.constraint_prior = nn.Sequential(
            nn.LayerNorm(node_dim + condition_dim),
            nn.Linear(node_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, constraint_codebook_size),
        )
        self.motif_prior = nn.Sequential(
            nn.LayerNorm(node_dim + condition_dim + constraint_code_dim),
            nn.Linear(node_dim + condition_dim + constraint_code_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, motif_codebook_size),
        )
        decoder_class = (
            SourceAnchoredEndpointField
            if self.source_anchored
            else belief.CategoricalEndpointField
        )
        self.decoder = decoder_class(
            node_dim,
            edge_dim,
            condition_dim + constraint_code_dim + motif_code_dim,
            hidden_dim,
            max_atoms,
        )

    @staticmethod
    def source_pool(source_node: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        total = (source_node * source_mask.unsqueeze(-1)).sum(dim=1)
        return total / source_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def route_condition(
        self,
        source_node: torch.Tensor,
        source_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim == 2:
            return condition
        if condition.ndim != 3:
            raise ValueError(f"Expected condition [B,D] or [B,L,D], got {condition.shape}")
        if self.condition_router is None:
            return condition.mean(dim=1)
        return self.condition_router(source_node, source_mask, condition)

    @staticmethod
    def quantize(
        posterior: torch.Tensor, codebook: nn.Embedding
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distance = (
            posterior.square().sum(dim=1, keepdim=True)
            - 2.0 * posterior @ codebook.weight.transpose(0, 1)
            + codebook.weight.square().sum(dim=1).unsqueeze(0)
        )
        index = distance.argmin(dim=1)
        quantized = codebook(index)
        straight_through = posterior + (quantized - posterior).detach()
        return straight_through, quantized, index

    def posterior_codes(
        self,
        source: Mapping[str, torch.Tensor],
        target: Mapping[str, torch.Tensor],
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        target_node: torch.Tensor,
        target_edge: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        node_changed, edge_changed = vq.change_masks(source, target)
        union_node = source["node_mask"].bool() | target["node_mask"].bool()
        nodes = union_node.shape[1]
        upper = torch.triu(
            torch.ones(nodes, nodes, device=union_node.device, dtype=torch.bool), diagonal=1
        )
        union_edge = upper.unsqueeze(0) & union_node[:, :, None] & union_node[:, None, :]
        global_node = masked_pool(target_node - source_node, union_node, (1,))
        global_edge = masked_pool(target_edge - source_edge, union_edge, (1, 2))
        constraint_posterior = self.constraint_posterior(
            torch.cat([global_node, global_edge, condition], dim=-1)
        )
        constraint_code, constraint_raw, constraint_index = self.quantize(
            constraint_posterior, self.constraint_codebook
        )
        local_node = masked_pool(target_node - source_node, node_changed, (1,))
        local_edge = masked_pool(target_edge - source_edge, edge_changed, (1, 2))
        motif_posterior = self.motif_posterior(
            torch.cat([local_node, local_edge, condition, constraint_code], dim=-1)
        )
        motif_code, motif_raw, motif_index = self.quantize(
            motif_posterior, self.motif_codebook
        )
        return (
            constraint_code,
            constraint_posterior,
            constraint_raw,
            constraint_index,
            motif_code,
            motif_posterior,
            motif_raw,
            motif_index,
        )

    def constraint_prior_logits(
        self, source_node: torch.Tensor, source_mask: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.source_pool(source_node, source_mask)
        return self.constraint_prior(torch.cat([pooled, condition], dim=-1))

    def motif_prior_logits(
        self,
        source_node: torch.Tensor,
        source_mask: torch.Tensor,
        condition: torch.Tensor,
        constraint_code: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.source_pool(source_node, source_mask)
        return self.motif_prior(
            torch.cat([pooled, condition, constraint_code], dim=-1)
        )

    def decode_endpoint(
        self,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        birth_rank: torch.Tensor,
        condition: torch.Tensor,
        constraint_code: torch.Tensor,
        motif_code: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        combined = torch.cat([condition, constraint_code, motif_code], dim=-1)
        time = torch.zeros(source_node.shape[0], device=source_node.device, dtype=source_node.dtype)
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


def vq_loss(
    posterior: torch.Tensor,
    quantized: torch.Tensor,
    commitment_weight: float,
) -> torch.Tensor:
    codebook_loss = F.mse_loss(quantized, posterior.detach())
    commitment_loss = F.mse_loss(posterior, quantized.detach())
    return codebook_loss + float(commitment_weight) * commitment_loss


def different_indices(index: torch.Tensor, size: int) -> torch.Tensor:
    rolled = torch.roll(index, shifts=1, dims=0)
    return torch.where(rolled.eq(index), (index + 1) % int(size), rolled)


def change_targets(
    source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_changed = torch.zeros_like(source["atomic_number"], dtype=torch.bool)
    for key in vq.NODE_FIELDS:
        node_changed |= source[key].ne(target[key])
    # Blank positions are important negative examples for spontaneous births.
    node_eligible = torch.ones_like(node_changed, dtype=torch.bool)
    edge_changed = torch.zeros_like(source["bond"], dtype=torch.bool)
    for key in vq.EDGE_FIELDS:
        edge_changed |= source[key].ne(target[key])
    nodes = source["atomic_number"].shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=node_changed.device, dtype=torch.bool),
        diagonal=1,
    )
    union_node = source["node_mask"].bool() | target["node_mask"].bool()
    edge_eligible = upper.unsqueeze(0) & union_node[:, :, None] & union_node[:, None, :]
    return node_changed, node_eligible, edge_changed, edge_eligible


def masked_binary_loss(
    logits: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    selected_target = target[eligible].float()
    positives = selected_target.sum().clamp_min(1.0)
    negatives = (1.0 - selected_target).sum().clamp_min(1.0)
    positive_weight = torch.sqrt(negatives / positives).clamp(1.0, 10.0)
    return F.binary_cross_entropy_with_logits(
        logits[eligible], selected_target, pos_weight=positive_weight
    )


def edit_gate_losses(
    decoded: tuple[torch.Tensor, ...],
    node_target: torch.Tensor,
    node_eligible: torch.Tensor,
    edge_target: torch.Tensor,
    edge_eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_gate, edge_gate = decoded[2], decoded[3]
    return (
        masked_binary_loss(node_gate, node_target, node_eligible),
        masked_binary_loss(edge_gate, edge_target, edge_eligible),
    )


def connected_region_target(
    node_changed: torch.Tensor, edge_changed: torch.Tensor
) -> torch.Tensor:
    edge_endpoint = edge_changed.any(dim=1) | edge_changed.any(dim=2)
    return node_changed | edge_endpoint


def connected_region_losses(
    decoded: tuple[torch.Tensor, ...], region_target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    region_loss = masked_binary_loss(
        decoded[2], region_target, torch.ones_like(region_target, dtype=torch.bool)
    )
    region_size = region_target.sum(dim=1).long().clamp_max(decoded[4].shape[-1] - 1)
    size_loss = F.cross_entropy(decoded[4], region_size)
    return region_loss, size_loss


def categorical_delta_targets(
    source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map an aligned train pair to the sparse source-relative delta grammar."""
    source_active = source["atomic_number"].gt(0)
    target_active = target["atomic_number"].gt(0)
    node_changed = torch.zeros_like(source_active)
    for key in vq.NODE_FIELDS:
        node_changed |= source[key].ne(target[key])
    node_operation = torch.full_like(source["atomic_number"], NODE_KEEP)
    node_operation = torch.where(
        source_active & ~target_active,
        torch.full_like(node_operation, NODE_DELETE),
        node_operation,
    )
    node_operation = torch.where(
        ~source_active & target_active,
        torch.full_like(node_operation, NODE_BIRTH),
        node_operation,
    )
    node_operation = torch.where(
        source_active & target_active & node_changed,
        torch.full_like(node_operation, NODE_REPLACE),
        node_operation,
    )
    node_eligible = torch.ones_like(source_active, dtype=torch.bool)

    source_bond = source["bond"].gt(graph.BOND_NONE)
    target_bond = target["bond"].gt(graph.BOND_NONE)
    edge_changed = torch.zeros_like(source_bond)
    for key in vq.EDGE_FIELDS:
        edge_changed |= source[key].ne(target[key])
    edge_operation = torch.full_like(source["bond"], EDGE_KEEP)
    edge_operation = torch.where(
        source_bond & ~target_bond,
        torch.full_like(edge_operation, EDGE_DELETE),
        edge_operation,
    )
    edge_operation = torch.where(
        target_bond & (~source_bond | edge_changed),
        torch.full_like(edge_operation, EDGE_SET),
        edge_operation,
    )
    nodes = source_active.shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=source_active.device, dtype=torch.bool),
        diagonal=1,
    )
    union_active = source_active | target_active
    edge_eligible = (
        upper.unsqueeze(0) & union_active[:, :, None] & union_active[:, None, :]
    )
    return node_operation, node_eligible, edge_operation, edge_eligible


def masked_categorical_loss(
    logits: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    selected = target[eligible].long()
    classes = int(logits.shape[-1])
    # Keep class reweighting and CE in FP32 under CUDA autocast. Assigning an
    # FP32 inverse-frequency value into BF16 weights fails before the first
    # training step, and the small categorical heads do not benefit from BF16.
    selected_logits = logits[eligible].float()
    counts = torch.bincount(selected, minlength=classes).float()
    weights = torch.ones(classes, device=logits.device, dtype=torch.float32)
    present = counts.gt(0)
    weights[present] = torch.sqrt(selected.numel() / counts[present]).clamp(1.0, 10.0)
    return F.cross_entropy(selected_logits, selected, weight=weights)


def categorical_delta_losses(
    decoded: tuple[torch.Tensor, ...],
    node_target: torch.Tensor,
    node_eligible: torch.Tensor,
    edge_target: torch.Tensor,
    edge_eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        masked_categorical_loss(decoded[5], node_target, node_eligible),
        masked_categorical_loss(decoded[6], edge_target, edge_eligible),
    )


def valence_budget_targets(target: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return total explicit valence in integer half-bond units per atom."""
    unit_table = torch.as_tensor(
        BOND_VALENCE_UNITS, device=target["bond"].device, dtype=torch.long
    )
    bond_units = unit_table[target["bond"].long()].sum(dim=2)
    explicit_h_units = 2 * target["explicit_hs"].long()
    return (bond_units + explicit_h_units).clamp_max(MAX_VALENCE_UNITS)


def decoder_auxiliary_losses(
    flow: HierarchicalVQGraphFlow,
    decoded: tuple[torch.Tensor, ...],
    region_target: torch.Tensor,
    node_change_target: torch.Tensor,
    node_change_eligible: torch.Tensor,
    edge_change_target: torch.Tensor,
    edge_change_eligible: torch.Tensor,
    node_delta_target: torch.Tensor,
    node_delta_eligible: torch.Tensor,
    edge_delta_target: torch.Tensor,
    edge_delta_eligible: torch.Tensor,
    valence_budget_target: torch.Tensor,
    valence_budget_eligible: torch.Tensor,
    motif_atom_count_target: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, ...]:
    zero = decoded[0].sum() * 0.0
    node_gate_loss = edge_gate_loss = region_size_loss = zero
    node_delta_loss = edge_delta_loss = zero
    valence_budget_loss = zero
    motif_atom_count_loss = zero
    if flow.connected_region:
        node_gate_loss, region_size_loss = connected_region_losses(
            decoded, region_target
        )
    elif flow.source_anchored:
        node_gate_loss, edge_gate_loss = edit_gate_losses(
            decoded,
            node_change_target,
            node_change_eligible,
            edge_change_target,
            edge_change_eligible,
        )
    if flow.categorical_delta:
        node_delta_loss, edge_delta_loss = categorical_delta_losses(
            decoded,
            node_delta_target,
            node_delta_eligible,
            edge_delta_target,
            edge_delta_eligible,
        )
    if flow.valence_budget:
        valence_budget_loss = masked_categorical_loss(
            decoded[7], valence_budget_target, valence_budget_eligible
        )
    if flow.motif_attachment:
        motif_atom_count_loss = F.cross_entropy(
            decoded[8].float(), motif_atom_count_target.long()
        )
    weighted = float(args.edit_gate_loss_weight) * (
        node_gate_loss + edge_gate_loss + region_size_loss
    ) + float(args.delta_loss_weight) * (
        node_delta_loss + edge_delta_loss
    ) + float(args.valence_budget_loss_weight) * valence_budget_loss + float(
        args.motif_atom_count_loss_weight
    ) * motif_atom_count_loss
    return (
        node_gate_loss,
        edge_gate_loss,
        region_size_loss,
        node_delta_loss,
        edge_delta_loss,
        valence_budget_loss,
        motif_atom_count_loss,
        weighted,
    )


def masked_operation_argmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    if logits.shape != legal.shape:
        raise ValueError(f"Operation logits {logits.shape} and mask {legal.shape} differ")
    return logits.float().masked_fill(~legal, -torch.inf).argmax(dim=-1)


def apply_categorical_graph_delta(
    source: Mapping[str, torch.Tensor],
    candidate_nodes: Mapping[str, torch.Tensor],
    candidate_edges: Mapping[str, torch.Tensor],
    decoded: tuple[torch.Tensor, ...],
    region: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Apply one legal sparse graph delta inside a latent-selected region."""
    result = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in source.items()
    }
    source_active = source["atomic_number"].gt(0)
    node_legal = torch.zeros_like(decoded[5], dtype=torch.bool)
    node_legal[..., NODE_KEEP] = True
    node_legal[..., NODE_DELETE] = region & source_active
    node_legal[..., NODE_BIRTH] = region & ~source_active
    node_legal[..., NODE_REPLACE] = region & source_active
    node_operation = masked_operation_argmax(decoded[5], node_legal)
    node_delete = node_operation.eq(NODE_DELETE)
    node_write = node_operation.eq(NODE_BIRTH) | node_operation.eq(NODE_REPLACE)
    for key in vq.NODE_FIELDS:
        result[key] = torch.where(node_write, candidate_nodes[key], result[key])
    result["atomic_number"] = torch.where(
        node_delete, torch.zeros_like(result["atomic_number"]), result["atomic_number"]
    )
    result = belief.enforce_categorical_consistency(result)

    active = result["atomic_number"].gt(0)
    internal = (
        region[:, :, None]
        & region[:, None, :]
        & active[:, :, None]
        & active[:, None, :]
        & upper.unsqueeze(0)
    )
    source_bond = source["bond"].gt(graph.BOND_NONE)
    edge_legal = torch.zeros_like(decoded[6], dtype=torch.bool)
    edge_legal[..., EDGE_KEEP] = True
    edge_legal[..., EDGE_DELETE] = internal & source_bond
    edge_legal[..., EDGE_SET] = internal
    edge_operation = masked_operation_argmax(decoded[6], edge_legal)
    edge_delete_upper = edge_operation.eq(EDGE_DELETE) & internal
    edge_set_upper = edge_operation.eq(EDGE_SET) & internal
    edge_delete = edge_delete_upper | edge_delete_upper.transpose(1, 2)
    edge_set = edge_set_upper | edge_set_upper.transpose(1, 2)
    for key in vq.EDGE_FIELDS:
        result[key] = torch.where(edge_set, candidate_edges[key], result[key])
        result[key] = torch.where(edge_delete, torch.zeros_like(result[key]), result[key])
    result = belief.enforce_categorical_consistency(result)
    node_changed = node_operation.ne(NODE_KEEP)
    edge_changed_upper = edge_operation.ne(EDGE_KEEP) & internal
    return result, node_changed, edge_changed_upper


def apply_valence_budget_graph_delta(
    source: Mapping[str, torch.Tensor],
    candidate_nodes: Mapping[str, torch.Tensor],
    candidate_edges: Mapping[str, torch.Tensor],
    decoded: tuple[torch.Tensor, ...],
    region: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Decode the delta grammar in fixed pair order under learned valence budgets.

    This is the support of the generative distribution, not a repair pass: the
    decoder never creates an edge operation that exceeds either endpoint's
    remaining learned target budget.
    """
    source_cpu = {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in source.items()
    }
    node_cpu = {key: value.detach().cpu() for key, value in candidate_nodes.items()}
    edge_cpu = {key: value.detach().cpu() for key, value in candidate_edges.items()}
    region_cpu = region.detach().cpu().bool()
    upper_cpu = upper.detach().cpu().bool()
    node_logits = decoded[5].detach().float().cpu()
    edge_logits = decoded[6].detach().float().cpu()
    budget = decoded[7].detach().float().cpu().argmax(dim=-1).long()
    unit_table = torch.as_tensor(BOND_VALENCE_UNITS, dtype=torch.long)

    result = source_cpu
    source_active = result["atomic_number"].gt(0)
    source_units = unit_table[result["bond"].long()]
    region_pair = region_cpu[:, :, None] & region_cpu[:, None, :]
    boundary_units = (source_units * ~region_pair).sum(dim=2)
    write_required = boundary_units + 2 * node_cpu["explicit_hs"].long()
    node_legal = torch.zeros_like(node_logits, dtype=torch.bool)
    node_legal[..., NODE_KEEP] = True
    node_legal[..., NODE_DELETE] = region_cpu & source_active
    node_legal[..., NODE_BIRTH] = (
        region_cpu & ~source_active & write_required.le(budget)
    )
    node_legal[..., NODE_REPLACE] = (
        region_cpu & source_active & write_required.le(budget)
    )
    node_operation = masked_operation_argmax(node_logits, node_legal)
    node_delete = node_operation.eq(NODE_DELETE)
    node_write = node_operation.eq(NODE_BIRTH) | node_operation.eq(NODE_REPLACE)
    for key in vq.NODE_FIELDS:
        result[key] = torch.where(node_write, node_cpu[key], result[key])
    result["atomic_number"] = torch.where(
        node_delete, torch.zeros_like(result["atomic_number"]), result["atomic_number"]
    )
    result = belief.enforce_categorical_consistency(result)

    active = result["atomic_number"].gt(0)
    active_pair = active[:, :, None] & active[:, None, :]
    internal = region_pair & active_pair & upper_cpu.unsqueeze(0)
    internal_symmetric = internal | internal.transpose(1, 2)
    current_units = unit_table[result["bond"].long()]
    preserved_units = (current_units * (active_pair & ~internal_symmetric)).sum(dim=2)
    remaining = (
        budget - 2 * result["explicit_hs"].long() - preserved_units
    ).clamp_min(0)
    edge_changed_upper = torch.zeros_like(internal)

    for batch_index in range(internal.shape[0]):
        for left, right in torch.nonzero(internal[batch_index], as_tuple=False).tolist():
            current_bond = int(result["bond"][batch_index, left, right].item())
            current_value = int(BOND_VALENCE_UNITS[current_bond])
            set_bond = int(edge_cpu["bond"][batch_index, left, right].item())
            set_value = int(BOND_VALENCE_UNITS[set_bond])
            legal = [False, False, False]
            legal[EDGE_KEEP] = (
                current_value <= int(remaining[batch_index, left])
                and current_value <= int(remaining[batch_index, right])
            )
            legal[EDGE_DELETE] = current_bond > graph.BOND_NONE
            legal[EDGE_SET] = (
                set_bond > graph.BOND_NONE
                and set_value <= int(remaining[batch_index, left])
                and set_value <= int(remaining[batch_index, right])
            )
            legal_tensor = torch.as_tensor(legal, dtype=torch.bool)
            operation = int(
                edge_logits[batch_index, left, right]
                .masked_fill(~legal_tensor, -torch.inf)
                .argmax()
                .item()
            )
            chosen_value = current_value if operation == EDGE_KEEP else 0
            if operation == EDGE_DELETE:
                for key in vq.EDGE_FIELDS:
                    result[key][batch_index, left, right] = 0
                    result[key][batch_index, right, left] = 0
            elif operation == EDGE_SET:
                chosen_value = set_value
                for key in vq.EDGE_FIELDS:
                    value = edge_cpu[key][batch_index, left, right]
                    result[key][batch_index, left, right] = value
                    result[key][batch_index, right, left] = value
            if operation != EDGE_KEEP:
                edge_changed_upper[batch_index, left, right] = True
            remaining[batch_index, left] -= chosen_value
            remaining[batch_index, right] -= chosen_value

    result = belief.enforce_categorical_consistency(result)
    return result, node_operation.ne(NODE_KEEP), edge_changed_upper


def apply_motif_attachment_graph_delta(
    source: Mapping[str, torch.Tensor],
    candidate_nodes: Mapping[str, torch.Tensor],
    candidate_edges: Mapping[str, torch.Tensor],
    bond_category_logits: torch.Tensor,
    decoded: tuple[torch.Tensor, ...],
    region: torch.Tensor,
    upper: torch.Tensor,
    candidate_adjacency: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Generate one connected latent motif from a single source attachment atom."""
    result = {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in source.items()
    }
    node_cpu = {key: value.detach().cpu() for key, value in candidate_nodes.items()}
    edge_cpu = {key: value.detach().cpu() for key, value in candidate_edges.items()}
    bond_logits = bond_category_logits.detach().float().cpu()
    region_cpu = region.detach().cpu().bool()
    upper_cpu = upper.detach().cpu().bool()
    region_scores = decoded[2].detach().float().cpu()
    edge_operation_logits = decoded[6].detach().float().cpu()
    budget = decoded[7].detach().float().cpu().argmax(dim=-1).long()
    desired_count = decoded[8].detach().float().cpu().argmax(dim=-1).long()
    source_active = result["atomic_number"].gt(0)
    source_adjacency = result["bond"].gt(graph.BOND_NONE)
    adjacency = source_adjacency | candidate_adjacency.detach().cpu().bool()
    adjacency &= region_cpu[:, :, None] & region_cpu[:, None, :]
    diagonal = torch.eye(adjacency.shape[1], dtype=torch.bool).unsqueeze(0)
    adjacency &= ~diagonal

    selected = torch.zeros_like(region_cpu)
    selection_orders: list[list[int]] = []
    for batch_index in range(region_cpu.shape[0]):
        available = region_cpu[batch_index]
        count = int(available.sum().item())
        if count == 0:
            selection_orders.append([])
            continue
        outside = ~region_cpu[batch_index]
        boundary = (
            available
            & source_active[batch_index]
            & (source_adjacency[batch_index] & outside.unsqueeze(0)).any(dim=1)
        )
        root_pool = boundary if bool(boundary.any()) else available & source_active[batch_index]
        if not bool(root_pool.any()):
            root_pool = available
        root = int(
            region_scores[batch_index]
            .masked_fill(~root_pool, -torch.inf)
            .argmax()
            .item()
        )
        requested = max(1, int(desired_count[batch_index].item()))
        requested = min(requested, count)
        selected[batch_index, root] = True
        order = [root]
        while len(order) < requested:
            frontier = adjacency[batch_index, selected[batch_index]].any(dim=0)
            frontier &= available & ~selected[batch_index]
            if not bool(frontier.any()):
                frontier = available & ~selected[batch_index]
            next_node = int(
                region_scores[batch_index]
                .masked_fill(~frontier, -torch.inf)
                .argmax()
                .item()
            )
            selected[batch_index, next_node] = True
            order.append(next_node)
        selection_orders.append(order)

    primary_anchor = torch.zeros_like(selected)
    for batch_index, order in enumerate(selection_orders):
        if order:
            primary_anchor[batch_index, order[0]] = True
    # Preserve a root only when it is an actual source attachment atom.  For a
    # de-novo region with no active source atom the selected root is itself a
    # generated motif atom and must therefore be written from the candidate.
    preserved_anchor = primary_anchor & source_active
    motif_write = selected & ~preserved_anchor
    motif_delete = region_cpu & ~selected
    for key in vq.NODE_FIELDS:
        result[key] = torch.where(motif_write, node_cpu[key], result[key])
    result["atomic_number"] = torch.where(
        motif_delete,
        torch.zeros_like(result["atomic_number"]),
        result["atomic_number"],
    )
    result = belief.enforce_categorical_consistency(result)

    active = result["atomic_number"].gt(0)
    region_pair = region_cpu[:, :, None] & region_cpu[:, None, :]
    clear_pair = region_pair & upper_cpu.unsqueeze(0)
    clear_pair = clear_pair | clear_pair.transpose(1, 2)
    for key in vq.EDGE_FIELDS:
        result[key] = torch.where(clear_pair, torch.zeros_like(result[key]), result[key])

    unit_table = torch.as_tensor(BOND_VALENCE_UNITS, dtype=torch.long)
    active_pair = active[:, :, None] & active[:, None, :]
    preserved_units = (unit_table[result["bond"].long()] * active_pair).sum(dim=2)
    remaining = (
        budget - 2 * result["explicit_hs"].long() - preserved_units
    ).clamp_min(0)
    attached = primary_anchor.clone()

    def best_bond(
        batch_index: int, left: int, right: int
    ) -> tuple[float, int] | None:
        allowed = [
            bond
            for bond in range(1, len(BOND_VALENCE_UNITS))
            if BOND_VALENCE_UNITS[bond] <= int(remaining[batch_index, left])
            and BOND_VALENCE_UNITS[bond] <= int(remaining[batch_index, right])
        ]
        if not allowed:
            return None
        best = max(allowed, key=lambda bond: float(bond_logits[batch_index, left, right, bond]))
        return float(bond_logits[batch_index, left, right, best]), int(best)

    tree_pairs: set[tuple[int, int, int]] = set()
    for batch_index, order in enumerate(selection_orders):
        for child in order[1:]:
            choices: list[tuple[float, int, int]] = []
            for parent in torch.nonzero(attached[batch_index], as_tuple=False).flatten().tolist():
                proposal = best_bond(batch_index, parent, child)
                if proposal is None:
                    continue
                score, bond = proposal
                if bool(adjacency[batch_index, parent, child]):
                    score += 1.0
                choices.append((score, int(parent), bond))
            if not choices:
                result["atomic_number"][batch_index, child] = 0
                selected[batch_index, child] = False
                continue
            _, parent, bond = max(choices)
            result["bond"][batch_index, parent, child] = bond
            result["bond"][batch_index, child, parent] = bond
            stereo = int(edge_cpu["bond_stereo"][batch_index, parent, child].item())
            if bond != graph.BOND_DOUBLE:
                stereo = 0
            result["bond_stereo"][batch_index, parent, child] = stereo
            result["bond_stereo"][batch_index, child, parent] = stereo
            units = int(BOND_VALENCE_UNITS[bond])
            remaining[batch_index, parent] -= units
            remaining[batch_index, child] -= units
            attached[batch_index, child] = True
            tree_pairs.add((batch_index, min(parent, child), max(parent, child)))

    active = result["atomic_number"].gt(0) & attached
    closure_pairs = active[:, :, None] & active[:, None, :] & upper_cpu.unsqueeze(0)
    for batch_index in range(closure_pairs.shape[0]):
        for left, right in torch.nonzero(
            closure_pairs[batch_index], as_tuple=False
        ).tolist():
            if (batch_index, left, right) in tree_pairs:
                continue
            if float(edge_operation_logits[batch_index, left, right, EDGE_SET]) <= float(
                edge_operation_logits[batch_index, left, right, EDGE_KEEP]
            ):
                continue
            proposal = best_bond(batch_index, left, right)
            if proposal is None:
                continue
            _, bond = proposal
            result["bond"][batch_index, left, right] = bond
            result["bond"][batch_index, right, left] = bond
            stereo = int(edge_cpu["bond_stereo"][batch_index, left, right].item())
            if bond != graph.BOND_DOUBLE:
                stereo = 0
            result["bond_stereo"][batch_index, left, right] = stereo
            result["bond_stereo"][batch_index, right, left] = stereo
            units = int(BOND_VALENCE_UNITS[bond])
            remaining[batch_index, left] -= units
            remaining[batch_index, right] -= units

    result = belief.enforce_categorical_consistency(result)
    node_changed = torch.zeros_like(region_cpu)
    for key in vq.NODE_FIELDS:
        node_changed |= result[key].ne(source[key].detach().cpu())
    edge_changed = torch.zeros_like(result["bond"], dtype=torch.bool)
    for key in vq.EDGE_FIELDS:
        edge_changed |= result[key].ne(source[key].detach().cpu())
    return result, node_changed, edge_changed & upper_cpu.unsqueeze(0)


def project_connected_region(
    scores: torch.Tensor,
    sizes: torch.Tensor,
    eligible: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Deterministically select one latent-scored connected graph region."""
    selected = torch.zeros_like(eligible, dtype=torch.bool)
    for batch_index in range(scores.shape[0]):
        available = eligible[batch_index].bool()
        available_count = int(available.sum().item())
        if available_count == 0:
            continue
        requested = max(1, int(sizes[batch_index].item()))
        requested = min(requested, available_count)
        masked = scores[batch_index].float().masked_fill(~available, -torch.inf)
        root = int(masked.argmax().item())
        selected[batch_index, root] = True
        while int(selected[batch_index].sum().item()) < requested:
            frontier = adjacency[batch_index, selected[batch_index]].any(dim=0)
            frontier &= available & ~selected[batch_index]
            if not bool(frontier.any()):
                break
            next_node = int(
                scores[batch_index].float().masked_fill(~frontier, -torch.inf).argmax().item()
            )
            selected[batch_index, next_node] = True
    return selected


def train_flow(
    flow: HierarchicalVQGraphFlow,
    representation,
    pairs: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], Counter[int], Counter[int]]:
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    final_constraint_usage: Counter[int] = Counter()
    final_motif_usage: Counter[int] = Counter()
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(pairs)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        constraint_usage: Counter[int] = Counter()
        motif_usage: Counter[int] = Counter()
        batches = 0
        flow.train()
        for start in range(0, len(order), int(args.batch_size)):
            items = [pairs[index] for index in order[start : start + int(args.batch_size)]]
            collated = base.pair_collate(items)
            source = base.move_graph_batch(collated["source"], device)
            target = base.move_graph_batch(collated["target"], device)
            raw_condition = collated["condition"].to(device)
            node_target, node_eligible, edge_target, edge_eligible = change_targets(
                source, target
            )
            region_target = connected_region_target(node_target, edge_target)
            (
                node_delta_target,
                node_delta_eligible,
                edge_delta_target,
                edge_delta_eligible,
            ) = categorical_delta_targets(source, target)
            valence_target = valence_budget_targets(target)
            valence_eligible = source["node_mask"].bool() | target["node_mask"].bool()
            motif_atom_count_target = (
                region_target & target["node_mask"].bool()
            ).sum(dim=1).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                target_node, target_edge = representation.encode(target)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                condition = flow.route_condition(
                    source_node, source["node_mask"], raw_condition
                )
                (
                    constraint_code,
                    constraint_posterior,
                    constraint_raw,
                    constraint_index,
                    motif_code,
                    motif_posterior,
                    motif_raw,
                    motif_index,
                ) = flow.posterior_codes(
                    source,
                    target,
                    source_node,
                    source_edge,
                    target_node,
                    target_edge,
                    condition,
                )
                constraint_prior = flow.constraint_prior_logits(
                    source_node, source["node_mask"], condition
                )
                motif_prior = flow.motif_prior_logits(
                    source_node, source["node_mask"], condition, constraint_code
                )
                birth_rank = belief.source_birth_ranks(source["node_mask"])
                decoded = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    birth_rank,
                    condition,
                    constraint_code,
                    motif_code,
                )
                endpoint_node, endpoint_edge = decoded[:2]
                logits = representation.decode(endpoint_node, endpoint_edge)
                endpoint_loss, parts = graph.reconstruction_loss(
                    logits, target, endpoint_node, geometry_weight=0.0
                )
                (
                    node_gate_loss,
                    edge_gate_loss,
                    region_size_loss,
                    node_delta_loss,
                    edge_delta_loss,
                    valence_budget_loss,
                    motif_atom_count_loss,
                    auxiliary_loss,
                ) = decoder_auxiliary_losses(
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
                structured_loss = endpoint_loss + auxiliary_loss

                wrong_constraint = flow.constraint_codebook(
                    different_indices(constraint_index, flow.constraint_codebook_size)
                ).detach()
                wrong_constraint_decoded = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    birth_rank,
                    condition,
                    wrong_constraint,
                    motif_code,
                )
                wrong_node, wrong_edge = wrong_constraint_decoded[:2]
                wrong_constraint_loss, _ = graph.reconstruction_loss(
                    representation.decode(wrong_node, wrong_edge),
                    target,
                    wrong_node,
                    geometry_weight=0.0,
                )
                wrong_constraint_auxiliary = decoder_auxiliary_losses(
                    flow,
                    wrong_constraint_decoded,
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
                )[-1]
                wrong_constraint_loss = (
                    wrong_constraint_loss + wrong_constraint_auxiliary
                )
                wrong_motif = flow.motif_codebook(
                    different_indices(motif_index, flow.motif_codebook_size)
                ).detach()
                wrong_motif_decoded = flow.decode_endpoint(
                    source_node,
                    source_edge,
                    source["node_mask"],
                    birth_rank,
                    condition,
                    constraint_code,
                    wrong_motif,
                )
                wrong_node, wrong_edge = wrong_motif_decoded[:2]
                wrong_motif_loss, _ = graph.reconstruction_loss(
                    representation.decode(wrong_node, wrong_edge),
                    target,
                    wrong_node,
                    geometry_weight=0.0,
                )
                wrong_motif_auxiliary = decoder_auxiliary_losses(
                    flow,
                    wrong_motif_decoded,
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
                )[-1]
                wrong_motif_loss = wrong_motif_loss + wrong_motif_auxiliary
                constraint_contrastive = F.relu(
                    float(args.contrastive_margin)
                    + structured_loss
                    - wrong_constraint_loss
                )
                motif_contrastive = F.relu(
                    float(args.contrastive_margin) + structured_loss - wrong_motif_loss
                )
                constraint_prior_loss = F.cross_entropy(
                    constraint_prior, constraint_index.detach()
                )
                motif_prior_loss = F.cross_entropy(motif_prior, motif_index.detach())
                constraint_vq_loss = vq_loss(
                    constraint_posterior,
                    constraint_raw,
                    float(args.commitment_weight),
                )
                motif_vq_loss = vq_loss(
                    motif_posterior, motif_raw, float(args.commitment_weight)
                )
                loss = (
                    structured_loss
                    + float(args.prior_loss_weight)
                    * (constraint_prior_loss + motif_prior_loss)
                    + float(args.vq_loss_weight)
                    * (constraint_vq_loss + motif_vq_loss)
                    + float(args.contrastive_loss_weight)
                    * (constraint_contrastive + motif_contrastive)
                )
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()
            for name, value in parts.items():
                totals[name] += float(value)
            totals["total_loss"] += float(loss.detach())
            totals["constraint_prior_loss"] += float(constraint_prior_loss.detach())
            totals["motif_prior_loss"] += float(motif_prior_loss.detach())
            totals["constraint_vq_loss"] += float(constraint_vq_loss.detach())
            totals["motif_vq_loss"] += float(motif_vq_loss.detach())
            totals["node_gate_loss"] += float(node_gate_loss.detach())
            totals["edge_gate_loss"] += float(edge_gate_loss.detach())
            totals["region_size_loss"] += float(region_size_loss.detach())
            totals["node_delta_loss"] += float(node_delta_loss.detach())
            totals["edge_delta_loss"] += float(edge_delta_loss.detach())
            totals["valence_budget_loss"] += float(valence_budget_loss.detach())
            totals["motif_atom_count_loss"] += float(motif_atom_count_loss.detach())
            totals["node_change_rate"] += float(
                node_target[node_eligible].float().mean()
            )
            totals["edge_change_rate"] += float(
                edge_target[edge_eligible].float().mean()
            )
            totals["constraint_contrastive_loss"] += float(constraint_contrastive.detach())
            totals["motif_contrastive_loss"] += float(motif_contrastive.detach())
            constraint_usage.update(
                int(value) for value in constraint_index.detach().cpu().tolist()
            )
            motif_usage.update(int(value) for value in motif_index.detach().cpu().tolist())
            batches += 1
        final_constraint_usage = constraint_usage
        final_motif_usage = motif_usage
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
            "active_constraint_codes": len(constraint_usage),
            "constraint_code_perplexity": perplexity(constraint_usage),
            "active_motif_codes": len(motif_usage),
            "motif_code_perplexity": perplexity(motif_usage),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history, final_constraint_usage, final_motif_usage


def sample_active_code(
    logits: torch.Tensor,
    active_codes: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    active_logits = logits[:, active_codes].float() / max(1e-4, float(temperature))
    probability = torch.softmax(active_logits, dim=-1)
    selected = torch.multinomial(probability, 1, replacement=True, generator=generator)
    return active_codes[selected.squeeze(1)]


@torch.no_grad()
def sample_from_source(
    flow: HierarchicalVQGraphFlow,
    representation,
    source_example,
    condition: np.ndarray,
    active_constraint_codes: Sequence[int],
    active_motif_codes: Sequence[int],
    *,
    attempts: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int, int, int, int, int]]:
    """Sample two latent tokens without a target graph or property oracle."""
    if not active_constraint_codes or not active_motif_codes:
        raise ValueError("Both train-used code supports are required")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    constraint_support = torch.as_tensor(
        active_constraint_codes, device=device, dtype=torch.long
    )
    motif_support = torch.as_tensor(active_motif_codes, device=device, dtype=torch.long)
    outputs: list[tuple[str | None, int, int, int, int, int]] = []
    flow.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        condition_batch = torch.from_numpy(
            np.repeat(condition[None, ...], count, axis=0)
        ).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            condition_batch = flow.route_condition(
                source_node, source["node_mask"], condition_batch
            )
            constraint_logits = flow.constraint_prior_logits(
                source_node, source["node_mask"], condition_batch
            )
        constraint_index = sample_active_code(
            constraint_logits, constraint_support, temperature, generator
        )
        constraint_code = flow.constraint_codebook(constraint_index)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            motif_logits = flow.motif_prior_logits(
                source_node, source["node_mask"], condition_batch, constraint_code
            )
        motif_index = sample_active_code(
            motif_logits, motif_support, temperature, generator
        )
        motif_code = flow.motif_codebook(motif_index)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            decoded = flow.decode_endpoint(
                source_node,
                source_edge,
                source["node_mask"],
                belief.source_birth_ranks(source["node_mask"]),
                condition_batch,
                constraint_code,
                motif_code,
            )
            endpoint_node, endpoint_edge = decoded[:2]
            logits = representation.decode(endpoint_node, endpoint_edge)
        if flow.source_anchored:
            result = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in source.items()
            }
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
            if flow.connected_region:
                adjacency = source["bond"].gt(graph.BOND_NONE) | candidate_edges[
                    "bond"
                ].gt(graph.BOND_NONE)
                adjacency &= node_eligible[:, :, None] & node_eligible[:, None, :]
                diagonal = torch.eye(nodes, device=device, dtype=torch.bool).unsqueeze(0)
                adjacency &= ~diagonal
                region_size = decoded[4].argmax(dim=-1)
                node_edit = project_connected_region(
                    decoded[2], region_size, node_eligible, adjacency
                )
            else:
                node_edit = decoded[2].gt(0) & node_eligible
            if flow.categorical_delta:
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
                if flow.motif_attachment:
                    result, node_edit, edge_edit_upper = (
                        apply_motif_attachment_graph_delta(
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
                else:
                    delta_decoder = (
                        apply_valence_budget_graph_delta
                        if flow.valence_budget
                        else apply_categorical_graph_delta
                    )
                    result, node_edit, edge_edit_upper = delta_decoder(
                        source, delta_nodes, delta_edges, decoded, node_edit, upper
                    )
            else:
                for key in vq.NODE_FIELDS:
                    result[key] = torch.where(
                        node_edit, candidate_nodes[key], source[key]
                    )
                result = belief.enforce_categorical_consistency(result)

                active = result["node_mask"].bool()
                active_pair = active[:, :, None] & active[:, None, :]
                if flow.connected_region:
                    region_pair = node_edit[:, :, None] & node_edit[:, None, :]
                    edge_edit_upper = region_pair & active_pair & upper.unsqueeze(0)
                else:
                    edge_eligible = active_pair & (
                        source["bond"].gt(graph.BOND_NONE)
                        | candidate_edges["bond"].gt(graph.BOND_NONE)
                    )
                    edge_edit_upper = (
                        decoded[3].gt(0) & edge_eligible & upper.unsqueeze(0)
                    )
                edge_edit = edge_edit_upper | edge_edit_upper.transpose(1, 2)
                for key in vq.EDGE_FIELDS:
                    result[key] = torch.where(
                        edge_edit, candidate_edges[key], result[key]
                    )
                result = belief.enforce_categorical_consistency(result)
            prediction = {
                key: result[key].detach().cpu().numpy()
                for key in (*vq.NODE_FIELDS, *vq.EDGE_FIELDS)
            }
            atomic = prediction["atomic_number"] > 0
            node_edit_count = node_edit.sum(dim=1).detach().cpu().tolist()
            edge_edit_count = edge_edit_upper.sum(dim=(1, 2)).detach().cpu().tolist()
        else:
            prediction = graph.predictions_from_logits(logits)
            atomic = prediction["atomic_number"] > 0
            active_pair = atomic[:, :, None] & atomic[:, None, :]
            prediction["bond"][~active_pair] = graph.BOND_NONE
            prediction["bond_stereo"][~active_pair] = 0
            node_edit_count = [0] * count
            edge_edit_count = [0] * count
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append(
                (
                    smiles,
                    int(atomic[index].sum()),
                    int(constraint_index[index].item()),
                    int(motif_index[index].item()),
                    int(node_edit_count[index]),
                    int(edge_edit_count[index]),
                )
            )
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    flow: HierarchicalVQGraphFlow,
    representation,
    pairs: Sequence[object],
    active_constraint_codes: Sequence[int],
    active_motif_codes: Sequence[int],
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
            active_constraint_codes,
            active_motif_codes,
            attempts=int(args.num_attempts),
            batch_size=int(args.sample_batch_size),
            temperature=float(args.sampling_temperature),
            device=device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        source_copy_target = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        for rank, (
            smiles,
            predicted_count,
            constraint_index,
            motif_index,
            node_edit_count,
            edge_edit_count,
        ) in enumerate(generated, start=1):
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
                    "constraint_code": int(constraint_index),
                    "motif_code": int(motif_index),
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
    metrics["sampled_constraint_codes"] = len(
        {int(row["constraint_code"]) for row in candidate_rows}
    )
    metrics["sampled_motif_codes"] = len(
        {int(row["motif_code"]) for row in candidate_rows}
    )
    metrics["sampled_code_pairs"] = len(
        {(int(row["constraint_code"]), int(row["motif_code"])) for row in candidate_rows}
    )
    metrics["mean_node_edits"] = float(
        np.mean([int(row["node_edit_count"]) for row in candidate_rows])
    )
    metrics["mean_edge_edits"] = float(
        np.mean([int(row["edge_edit_count"]) for row in candidate_rows])
    )
    metrics["source_copy_rate"] = sum(
        str(row["generated_smiles"]) == graph.canonical_smiles(str(row["source_smiles"]))
        for row in candidate_rows
    ) / max(1, len(candidate_rows))
    return candidate_rows, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.condition_attention:
        args.motif_attachment = True
    if args.motif_attachment:
        args.valence_budget = True
    if args.valence_budget:
        args.categorical_delta = True
    if args.categorical_delta:
        args.connected_region = True
    if args.connected_region:
        args.source_anchored = True
    protocol = (
        CONSTRAINT_ATTENTION_PROTOCOL
        if args.condition_attention
        else MOTIF_ATTACHMENT_PROTOCOL
        if args.motif_attachment
        else VALENCE_BUDGET_PROTOCOL
        if args.valence_budget
        else CATEGORICAL_DELTA_PROTOCOL
        if args.categorical_delta
        else CONNECTED_REGION_PROTOCOL
        if args.connected_region
        else SOURCE_ANCHORED_PROTOCOL
        if args.source_anchored
        else PROTOCOL
    )
    if int(args.num_attempts) != 20:
        raise ValueError("The protocol requires exactly 20 raw attempts per condition")
    base.seed_everything(int(args.seed))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation, config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    validation_pairs, validation_counts = base.build_pairs(
        base.read_rows(args.validation_csv),
        max_atoms=int(config["max_atoms"]),
        fingerprint_bits=int(args.fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.seed) + 1,
    )
    if not validation_pairs:
        raise ValueError("No validation edit pairs survived the fixed filters")
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
        seed=int(args.seed),
        forbidden_sources=validation_sources,
        forbidden_pairs=validation_pair_keys,
    )
    if len(train_pairs) < 32:
        raise ValueError(f"Need at least 32 train pairs, found {len(train_pairs)}")
    if args.condition_attention:
        for pair in [*train_pairs, *validation_pairs]:
            pair.condition = condition_tokens(pair.row, int(args.condition_dim))
    flow = HierarchicalVQGraphFlow(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        constraint_code_dim=int(args.constraint_code_dim),
        constraint_codebook_size=int(args.constraint_codebook_size),
        motif_code_dim=int(args.motif_code_dim),
        motif_codebook_size=int(args.motif_codebook_size),
        hidden_dim=int(args.hidden_dim),
        max_atoms=int(config["max_atoms"]),
        source_anchored=bool(args.source_anchored),
        connected_region=bool(args.connected_region),
        categorical_delta=bool(args.categorical_delta),
        valence_budget=bool(args.valence_budget),
        motif_attachment=bool(args.motif_attachment),
        condition_attention=bool(args.condition_attention),
        condition_attention_heads=int(args.condition_attention_heads),
    ).to(device)
    history, constraint_usage, motif_usage = train_flow(
        flow, representation, train_pairs, args, device
    )
    active_constraint_codes = sorted(constraint_usage)
    active_motif_codes = sorted(motif_usage)
    candidate_rows, metrics = evaluate(
        flow,
        representation,
        validation_pairs,
        active_constraint_codes,
        active_motif_codes,
        args,
        device,
    )
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "train_active_constraint_codes": {
            "value": len(active_constraint_codes),
            "threshold": int(args.gate_min_constraint_codes),
        },
        "train_active_motif_codes": {
            "value": len(active_motif_codes),
            "threshold": int(args.gate_min_motif_codes),
        },
        "validity": {"value": metrics["validity"], "threshold": float(args.gate_validity)},
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": float(args.gate_source_tanimoto),
        },
        "target_improvement_any20": {
            "value": metrics["target_improvement_any20"],
            "threshold": float(args.gate_target_improvement_rate),
        },
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": float(args.gate_strict_any20),
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
        "protocol": protocol,
        "seed": int(args.seed),
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
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_validation_pair_overlap": len(train_pair_keys & validation_pair_keys),
        "property_counts": sorted(allowed_counts),
        "constraint_codebook_size": int(args.constraint_codebook_size),
        "motif_codebook_size": int(args.motif_codebook_size),
        "train_active_constraint_codes": len(active_constraint_codes),
        "train_active_motif_codes": len(active_motif_codes),
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "hierarchical_constraint_then_motif_tokens": True,
        "constraint_posterior_train_only": True,
        "motif_posterior_train_only": True,
        "source_condition_constraint_prior": True,
        "constraint_conditioned_motif_prior": True,
        "condition_token_pooling": "source_node_cross_attention"
        if args.condition_attention
        else "mean",
        "condition_attention": bool(args.condition_attention),
        "condition_attention_heads": int(args.condition_attention_heads),
        "separate_token_contrastive_reconstruction": True,
        "deterministic_category_decode_given_tokens": True,
        "source_anchored_residual_decoder": bool(args.source_anchored),
        "learned_atom_and_bond_edit_blocks": bool(
            args.source_anchored and not args.connected_region
        ),
        "deterministic_edit_gates": bool(
            args.source_anchored and not args.connected_region
        ),
        "connected_region_decoder": bool(args.connected_region),
        "learned_region_size": bool(args.connected_region),
        "latent_scored_connected_projection": bool(args.connected_region),
        "whole_region_endpoint_subgraph": bool(
            args.connected_region and not args.categorical_delta
        ),
        "source_boundary_preserved": bool(args.connected_region),
        "categorical_graph_delta_grammar": bool(args.categorical_delta),
        "node_delta_operations": ["KEEP", "DELETE", "BIRTH", "REPLACE"]
        if args.categorical_delta
        else [],
        "edge_delta_operations": ["KEEP", "DELETE", "SET"]
        if args.categorical_delta
        else [],
        "explicit_keep_category": bool(args.categorical_delta),
        "legal_operation_mask_from_source_occupancy": bool(args.categorical_delta),
        "region_internal_sparse_delta": bool(args.categorical_delta),
        "grammar_native_valence_budget": bool(args.valence_budget),
        "learned_total_explicit_valence_units": bool(args.valence_budget),
        "fixed_order_autoregressive_edge_operations": bool(args.valence_budget),
        "bond_valence_half_units": list(BOND_VALENCE_UNITS)
        if args.valence_budget
        else [],
        "motif_attachment_decoder": bool(args.motif_attachment),
        "single_source_attachment_anchor": bool(args.motif_attachment),
        "learned_motif_atom_count": bool(args.motif_attachment),
        "connected_spanning_tree_support": bool(args.motif_attachment),
        "budgeted_ring_closure_edges": bool(args.motif_attachment),
        "edit_gate_loss_weight": float(args.edit_gate_loss_weight),
        "delta_loss_weight": float(args.delta_loss_weight),
        "valence_budget_loss_weight": float(args.valence_budget_loss_weight),
        "motif_atom_count_loss_weight": float(args.motif_atom_count_loss_weight),
        "posthoc_source_copy_heuristic": False,
        "independent_atom_or_bond_sampling": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "valence_projection_or_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "source_target_mcs_alignment_training_only": True,
    }
    checkpoint_name = (
        "motif_attachment_hierarchical_vq_graph_flow.pt"
        if args.motif_attachment
        else "valence_budget_hierarchical_vq_graph_flow.pt"
        if args.valence_budget
        else "categorical_delta_hierarchical_vq_graph_flow.pt"
        if args.categorical_delta
        else "connected_region_hierarchical_vq_graph_flow.pt"
        if args.connected_region
        else "source_anchored_hierarchical_vq_graph_flow.pt"
        if args.source_anchored
        else "hierarchical_vq_motif_graph_flow.pt"
    )
    checkpoint_path = args.output_dir / checkpoint_name
    torch.save(
        {
            "stage": protocol,
            "model_state": flow.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": int(args.condition_dim),
                "constraint_code_dim": int(args.constraint_code_dim),
                "constraint_codebook_size": int(args.constraint_codebook_size),
                "motif_code_dim": int(args.motif_code_dim),
                "motif_codebook_size": int(args.motif_codebook_size),
                "hidden_dim": int(args.hidden_dim),
                "max_atoms": int(config["max_atoms"]),
                "source_anchored": bool(args.source_anchored),
                "connected_region": bool(args.connected_region),
                "categorical_delta": bool(args.categorical_delta),
                "valence_budget": bool(args.valence_budget),
                "motif_attachment": bool(args.motif_attachment),
                "condition_attention": bool(args.condition_attention),
                "condition_attention_heads": int(args.condition_attention_heads),
            },
            "active_constraint_codes": active_constraint_codes,
            "active_motif_codes": active_motif_codes,
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "protocol": protocol,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            (
                "expand_motif_attachment_hierarchical_vq_signal"
                if args.motif_attachment
                else "expand_valence_budget_hierarchical_vq_signal"
                if args.valence_budget
                else "expand_categorical_delta_hierarchical_vq_signal"
                if args.categorical_delta
                else "expand_connected_region_hierarchical_vq_signal"
                if args.connected_region
                else "expand_source_anchored_hierarchical_vq_signal"
                if args.source_anchored
                else "expand_hierarchical_vq_signal"
            )
            if not failures
            else (
                "diagnose_motif_attachment_support_or_atom_count"
                if args.motif_attachment
                else "diagnose_valence_budget_support_or_connectivity"
                if args.valence_budget
                else "diagnose_categorical_delta_support_or_calibration"
                if args.categorical_delta
                else "diagnose_connected_region_endpoint_validity"
            )
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
