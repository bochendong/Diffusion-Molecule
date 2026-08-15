#!/usr/bin/env python3
"""Train an orderless, source-clamped latent graph jump process.

B38 removes B37's explicit editable-region variable.  The discrete generator is
a globally normalized jump process over single sparse graph events plus STOP.
Consequently one transition can never fan out into dozens of simultaneous node
and edge decisions.  Training uses a random topological prefix of each B22
strict edit set and maximizes the probability mass of *all* dependency-ready
next events, so no arbitrary edit order is treated as ground truth.

Only B22 train-derived strict pairs are used.  Generation receives a source
graph and sanitized condition tokens, freezes exactly 20 raw attempts, and only
then opens the source-disjoint train-only development evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import source_clamped_region_graph_diffusion as b37  # noqa: E402


b36 = b37.b36
b22 = b37.b22
base = b37.base
belief = b37.belief
delta = b37.delta
full_graph = b37.full_graph
graph = b37.graph
hierarchical = b37.hierarchical
unified = b37.unified

PROTOCOL = "train_only_source_clamped_latent_graph_jump_process_v38"
STOP, NODE_DELETE, NODE_WRITE, EDGE_DELETE, EDGE_SET = range(5)


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
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "explicit_region_mask": False,
        "globally_normalized_single_event_jumps": True,
        "learned_stop_event": True,
        "orderless_event_set_objective": True,
        "hard_patch_count": False,
        "hard_anchor_limit": False,
        "hard_edit_radius": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "exact_raw_attempts_per_condition": 20,
        "development_source_limit": 160,
        "max_jumps": 64,
        "epochs": 6,
        "flow_steps": 8,
        "birth_capacity": 8,
        "pretraining_support_amendment": {
            "failed_job_id": 19865432,
            "model_training_started": False,
            "candidate_generation_started": False,
            "exact_replay_rate": 1.0,
            "topological_coverage": 1.0,
            "horizon_coverage_at_32": 0.9328859060402684,
            "maximum_observed_target_events": 53,
            "revised_safety_horizon": 64,
            "scientific_gate_changed": False,
        },
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B38 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B38 property-count contract drift")
    implementation_sha256 = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "B38 implementation drift: "
            f"expected {payload.get('implementation_sha256')}, "
            f"found {implementation_sha256}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_summary_sha256",
        "b37_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("B38 locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_summary_sha256": args.b36_summary,
        "b37_summary_sha256": args.b37_summary,
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
        raise ValueError(f"B38 locked input drift: {drift}")

    b22_summary, checkpoint = b36.load_locked_b22(args, preregistration)
    b36_summary = json.loads(args.b36_summary.read_text(encoding="utf-8"))
    if b36_summary.get("protocol") != b36.PROTOCOL:
        raise ValueError("B38 requires the locked B36 protocol")
    if b36_summary.get("decision") != "stop_graph_patch_representation_after_evidence_gate":
        raise ValueError("B38 refuses a B36 decision drift")
    b36_metrics = dict(b36_summary.get("metrics", {}))
    b36_drift = {
        key: {"expected": expected, "actual": b36_metrics.get(key)}
        for key, expected in dict(preregistration["b36_evidence_trigger"]).items()
        if b36_metrics.get(key) != expected
    }
    if b36_drift:
        raise ValueError(f"B38 refuses B36 evidence drift: {b36_drift}")

    b37_summary = json.loads(args.b37_summary.read_text(encoding="utf-8"))
    if b37_summary.get("protocol") != b37.PROTOCOL:
        raise ValueError("B38 requires the locked B37 protocol")
    if b37_summary.get("decision") != "stop_and_diagnose_region_transport_without_patch_expansion":
        raise ValueError("B38 refuses a B37 decision drift")
    b37_metrics = dict(b37_summary.get("metrics", {}))
    b37_drift = {}
    for key, expected in dict(preregistration["b37_failure_trigger"]).items():
        actual = b37_metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                b37_drift[key] = {"expected": expected, "actual": actual}
        elif actual != expected:
            b37_drift[key] = {"expected": expected, "actual": actual}
    if b37_drift:
        raise ValueError(f"B38 refuses B37 failure-evidence drift: {b37_drift}")
    return b22_summary, checkpoint, b36_summary, b37_summary


@dataclass(frozen=True)
class GraphEvent:
    kind: int
    left: int
    right: int = -1
    action: int = 0


@dataclass(frozen=True)
class EventLayout:
    nodes: int
    node_payloads: int
    edge_payloads: int

    @property
    def pair_count(self) -> int:
        return self.nodes * (self.nodes - 1) // 2

    @property
    def node_delete_offset(self) -> int:
        return 1

    @property
    def node_write_offset(self) -> int:
        return self.node_delete_offset + self.nodes

    @property
    def edge_delete_offset(self) -> int:
        return self.node_write_offset + self.nodes * self.node_payloads

    @property
    def edge_set_offset(self) -> int:
        return self.edge_delete_offset + self.pair_count

    @property
    def total_events(self) -> int:
        return self.edge_set_offset + self.pair_count * self.edge_payloads

    def pair_index(self, left: int, right: int) -> int:
        if not 0 <= left < right < self.nodes:
            raise ValueError(f"Invalid upper-triangle pair {(left, right)}")
        return left * (2 * self.nodes - left - 1) // 2 + right - left - 1

    def pair(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.pair_count:
            raise ValueError(f"Invalid pair index {index}")
        remaining = int(index)
        for left in range(self.nodes - 1):
            width = self.nodes - left - 1
            if remaining < width:
                return left, left + 1 + remaining
            remaining -= width
        raise RuntimeError("Unreachable pair decode")

    def encode(self, event: GraphEvent) -> int:
        if event.kind == STOP:
            return 0
        if event.kind == NODE_DELETE:
            return self.node_delete_offset + event.left
        if event.kind == NODE_WRITE:
            payload = event.action - (delta.NODE_WRITE_OFFSET + 1)
            if not 0 <= payload < self.node_payloads:
                raise ValueError(f"Invalid node write action {event.action}")
            return self.node_write_offset + event.left * self.node_payloads + payload
        pair = self.pair_index(event.left, event.right)
        if event.kind == EDGE_DELETE:
            return self.edge_delete_offset + pair
        if event.kind == EDGE_SET:
            payload = event.action - (delta.EDGE_SET_OFFSET + 1)
            if not 0 <= payload < self.edge_payloads:
                raise ValueError(f"Invalid edge set action {event.action}")
            return self.edge_set_offset + pair * self.edge_payloads + payload
        raise ValueError(f"Unknown event kind {event.kind}")

    def decode(self, index: int) -> GraphEvent:
        if index == 0:
            return GraphEvent(STOP, -1)
        if index < self.node_write_offset:
            return GraphEvent(
                NODE_DELETE,
                index - self.node_delete_offset,
                action=delta.NODE_DELETE,
            )
        if index < self.edge_delete_offset:
            relative = index - self.node_write_offset
            left, payload = divmod(relative, self.node_payloads)
            return GraphEvent(
                NODE_WRITE, left, action=payload + delta.NODE_WRITE_OFFSET + 1
            )
        if index < self.edge_set_offset:
            left, right = self.pair(index - self.edge_delete_offset)
            return GraphEvent(
                EDGE_DELETE, left, right, action=delta.EDGE_DELETE
            )
        relative = index - self.edge_set_offset
        pair_index, payload = divmod(relative, self.edge_payloads)
        left, right = self.pair(pair_index)
        return GraphEvent(
            EDGE_SET,
            left,
            right,
            payload + delta.EDGE_SET_OFFSET + 1,
        )


def target_event_set(
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    layout: EventLayout,
) -> list[GraphEvent]:
    events: list[GraphEvent] = []
    for node, action in enumerate(node_actions.tolist()):
        action = int(action)
        if action == delta.NODE_DELETE:
            events.append(GraphEvent(NODE_DELETE, node, action=action))
        elif action >= delta.NODE_WRITE_OFFSET + 1:
            events.append(GraphEvent(NODE_WRITE, node, action=action))
        elif action != delta.NODE_KEEP:
            raise ValueError(f"Unsupported target node action {action}")
    for left in range(layout.nodes - 1):
        for right in range(left + 1, layout.nodes):
            action = int(edge_actions[left, right])
            if action == delta.EDGE_DELETE:
                events.append(GraphEvent(EDGE_DELETE, left, right, action))
            elif action >= delta.EDGE_SET_OFFSET + 1:
                events.append(GraphEvent(EDGE_SET, left, right, action))
            elif action != delta.EDGE_KEEP:
                raise ValueError(f"Unsupported target edge action {action}")
    encoded = [layout.encode(event) for event in events]
    if len(encoded) != len(set(encoded)):
        raise ValueError("A target graph contains duplicate jump events")
    return events


def event_dependencies(events: Sequence[GraphEvent]) -> dict[int, set[int]]:
    encoded_events = list(enumerate(events))
    node_writes = {
        event.left: index
        for index, event in encoded_events
        if event.kind == NODE_WRITE
    }
    incident_edges: defaultdict[int, set[int]] = defaultdict(set)
    dependencies: dict[int, set[int]] = {index: set() for index, _ in encoded_events}
    for index, event in encoded_events:
        if event.kind in (EDGE_DELETE, EDGE_SET):
            incident_edges[event.left].add(index)
            incident_edges[event.right].add(index)
            if event.kind == EDGE_SET:
                for node in (event.left, event.right):
                    if node in node_writes:
                        dependencies[index].add(node_writes[node])
    for index, event in encoded_events:
        if event.kind == NODE_DELETE:
            dependencies[index].update(incident_edges[event.left])
    return dependencies


def apply_event_to_actions(
    event: GraphEvent, node_actions: torch.Tensor, edge_actions: torch.Tensor
) -> None:
    if event.kind in (NODE_DELETE, NODE_WRITE):
        if int(node_actions[event.left]) != delta.NODE_KEEP:
            raise ValueError("A node received more than one jump event")
        node_actions[event.left] = int(event.action)
    elif event.kind in (EDGE_DELETE, EDGE_SET):
        if int(edge_actions[event.left, event.right]) != delta.EDGE_KEEP:
            raise ValueError("An edge received more than one jump event")
        edge_actions[event.left, event.right] = int(event.action)
        edge_actions[event.right, event.left] = int(event.action)
    elif event.kind != STOP:
        raise ValueError(f"Unknown jump event {event}")


def random_topological_prefix(
    events: Sequence[GraphEvent],
    layout: EventLayout,
    *,
    seed: int,
    completion_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rng = random.Random(int(seed))
    dependencies = event_dependencies(events)
    if events and rng.random() >= float(completion_probability):
        prefix_length = rng.randrange(len(events))
    else:
        prefix_length = len(events)
    applied: set[int] = set()
    order: list[int] = []
    for _ in range(prefix_length):
        ready = sorted(
            index
            for index in range(len(events))
            if index not in applied and dependencies[index] <= applied
        )
        if not ready:
            raise ValueError("Target event dependencies contain a cycle")
        selected = rng.choice(ready)
        applied.add(selected)
        order.append(selected)
    node_actions = torch.full((layout.nodes,), delta.NODE_KEEP, dtype=torch.long)
    edge_actions = torch.full(
        (layout.nodes, layout.nodes), delta.EDGE_KEEP, dtype=torch.long
    )
    for index in order:
        apply_event_to_actions(events[index], node_actions, edge_actions)
    next_target = torch.zeros(layout.total_events, dtype=torch.bool)
    if len(applied) == len(events):
        next_target[0] = True
    else:
        ready = [
            index
            for index in range(len(events))
            if index not in applied and dependencies[index] <= applied
        ]
        if not ready:
            raise ValueError("Target event set has no dependency-ready transition")
        for index in ready:
            next_target[layout.encode(events[index])] = True
    return node_actions, edge_actions, next_target, len(applied)


class JumpEventField(nn.Module):
    """Permutation-equivariant global field over one-event graph transitions."""

    def __init__(
        self,
        *,
        node_action_count: int,
        edge_action_count: int,
        source_node_dim: int,
        source_edge_dim: int,
        context_dim: int,
        hidden_dim: int,
        max_atoms: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.layout = EventLayout(
            max_atoms, node_action_count - 2, edge_action_count - 2
        )
        self.node_embedding = nn.Embedding(node_action_count, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_action_count, hidden_dim)
        self.source_node = nn.Linear(source_node_dim, hidden_dim)
        self.source_edge = nn.Linear(source_edge_dim, hidden_dim)
        self.birth_rank = nn.Embedding(max_atoms + 1, hidden_dim)
        self.time = nn.Sequential(
            full_graph.continuous.TimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [full_graph.DenseDiscreteGraphLayer(hidden_dim) for _ in range(int(layers))]
        )
        self.stop_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1)
        )
        self.node_delete_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.node_write_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.layout.node_payloads)
        )
        self.edge_features = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
        )
        self.edge_delete_head = nn.Linear(hidden_dim, 1)
        self.edge_set_head = nn.Linear(hidden_dim, self.layout.edge_payloads)
        upper = torch.triu_indices(max_atoms, max_atoms, offset=1)
        self.register_buffer("pair_left", upper[0], persistent=False)
        self.register_buffer("pair_right", upper[1], persistent=False)
        self._initialize_event_priors()

    def _initialize_event_priors(self) -> None:
        nn.init.constant_(self.stop_head[-1].bias, -1.5)
        nn.init.constant_(self.node_delete_head[-1].bias, -4.0)
        nn.init.constant_(self.node_write_head[-1].bias, -6.0)
        nn.init.constant_(self.edge_delete_head.bias, -5.0)
        nn.init.constant_(self.edge_set_head.bias, -8.0)

    def forward(
        self,
        node_actions: torch.Tensor,
        edge_actions: torch.Tensor,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        working_mask: torch.Tensor,
        jump_time: torch.Tensor,
        condition: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context(torch.cat([condition, latent], dim=-1)) + self.time(
            jump_time
        )
        ranks = belief.source_birth_ranks(source_mask).clamp_max(
            self.birth_rank.num_embeddings - 1
        )
        source_node_value = self.source_node(source_node)
        source_edge_value = self.source_edge(source_edge)
        node = (
            self.node_embedding(node_actions)
            + source_node_value
            + self.birth_rank(ranks)
            + context[:, None, :]
        ) * working_mask.unsqueeze(-1)
        edge = self.edge_embedding(edge_actions) + source_edge_value
        pair_mask = working_mask[:, :, None] & working_mask[:, None, :]
        edge = edge * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            node, edge = layer(node, edge, source_edge_value, context, working_mask)
        pooled = (node * working_mask.unsqueeze(-1)).sum(dim=1)
        pooled /= working_mask.sum(dim=1, keepdim=True).clamp_min(1).sqrt()
        left, right = node[:, :, None, :], node[:, None, :, :]
        edge_feature = self.edge_features(
            torch.cat(
                [edge, source_edge_value, left + right, (left - right).abs()], dim=-1
            )
        )
        pair_feature = edge_feature[:, self.pair_left, self.pair_right]
        return torch.cat(
            [
                self.stop_head(torch.cat([pooled, context], dim=-1)),
                self.node_delete_head(node).squeeze(-1),
                self.node_write_head(node).flatten(1),
                self.edge_delete_head(pair_feature).squeeze(-1),
                self.edge_set_head(pair_feature).flatten(1),
            ],
            dim=1,
        )


class LatentGraphJumpProcess(full_graph.ContinuousDiscreteGraphDiffusion):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.denoiser = JumpEventField(
            node_action_count=int(kwargs["node_state_count"]),
            edge_action_count=int(kwargs["edge_state_count"]),
            source_node_dim=int(kwargs["node_dim"]),
            source_edge_dim=int(kwargs["edge_dim"]),
            context_dim=int(kwargs["condition_dim"]) + int(kwargs["transport_dim"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            max_atoms=int(kwargs["max_atoms"]),
            layers=int(kwargs["message_layers"]),
        )


def legal_event_mask(
    field: JumpEventField,
    source: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    working: torch.Tensor,
) -> torch.Tensor:
    source_active = source["atomic_number"].gt(0)
    current_active = delta.action_active_nodes(source_active, node_actions)
    node_free = node_actions.eq(delta.NODE_KEEP) & working
    pair_left, pair_right = field.pair_left, field.pair_right
    pair_free = edge_actions[:, pair_left, pair_right].eq(delta.EDGE_KEEP)
    active_pair = current_active[:, pair_left] & current_active[:, pair_right]
    source_bond = source["bond"][:, pair_left, pair_right].gt(graph.BOND_NONE)
    batch = node_actions.shape[0]
    stop = torch.ones(batch, 1, dtype=torch.bool, device=node_actions.device)
    node_delete = node_free & source_active
    node_write = node_free.unsqueeze(-1).expand(-1, -1, field.layout.node_payloads)
    edge_delete = pair_free & source_bond & active_pair
    edge_set = pair_free.unsqueeze(-1) & active_pair.unsqueeze(-1)
    edge_set = edge_set.expand(-1, -1, field.layout.edge_payloads)
    legal = torch.cat(
        [
            stop,
            node_delete,
            node_write.flatten(1),
            edge_delete,
            edge_set.flatten(1),
        ],
        dim=1,
    )
    if legal.shape[1] != field.layout.total_events:
        raise RuntimeError("B38 legal-event layout mismatch")
    return legal


def orderless_jump_loss(
    logits: torch.Tensor,
    legal: torch.Tensor,
    target_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bool((target_next & ~legal).any()):
        bad = torch.nonzero(target_next & ~legal, as_tuple=False)[:8].tolist()
        raise ValueError(f"B38 target-next events are outside legal support: {bad}")
    if not bool(target_next.any(dim=1).all()):
        raise ValueError("B38 every prefix must have at least one next event")
    legal_logits = logits.float().masked_fill(~legal, -torch.inf)
    target_logits = logits.float().masked_fill(~target_next, -torch.inf)
    log_partition = torch.logsumexp(legal_logits, dim=1)
    log_target_mass = torch.logsumexp(target_logits, dim=1)
    loss = (log_partition - log_target_mass).mean()
    mass = (log_target_mass - log_partition).exp().mean()
    accuracy = target_next.gather(1, legal_logits.argmax(dim=1, keepdim=True)).float().mean()
    return loss, mass, accuracy


def build_prefix_batch(
    node_targets: torch.Tensor,
    edge_targets: torch.Tensor,
    layout: EventLayout,
    *,
    epoch: int,
    global_batch: int,
    seed: int,
    completion_probability: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    nodes, edges, targets, prefix_lengths, event_counts = [], [], [], [], []
    for index in range(node_targets.shape[0]):
        events = target_event_set(
            node_targets[index].detach().cpu(),
            edge_targets[index].detach().cpu(),
            layout,
        )
        prefix_seed = (
            int(seed) * 1_000_003
            + int(epoch) * 10_007
            + int(global_batch) * 101
            + index
        )
        current_node, current_edge, target_next, prefix_length = (
            random_topological_prefix(
                events,
                layout,
                seed=prefix_seed,
                completion_probability=completion_probability,
            )
        )
        nodes.append(current_node)
        edges.append(current_edge)
        targets.append(target_next)
        prefix_lengths.append(prefix_length)
        event_counts.append(len(events))
    jump_time = torch.as_tensor(
        [prefix / max(1, count + 1) for prefix, count in zip(prefix_lengths, event_counts)],
        dtype=torch.float32,
        device=device,
    )
    return (
        torch.stack(nodes).to(device),
        torch.stack(edges).to(device),
        torch.stack(targets).to(device),
        jump_time,
        event_counts,
    )


def preflight_event_replay(
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    layout: EventLayout,
    max_jumps: int,
) -> dict[str, object]:
    exact = 0
    topology = 0
    counts: list[int] = []
    for start in range(0, len(pairs), 32):
        items = pairs[start : start + 32]
        collated = base.pair_collate(items)
        source = collated["source"]
        target = collated["target"]
        node_targets, edge_targets = delta.delta_action_targets(
            source, target, vocabulary
        )
        replay_nodes = torch.full_like(node_targets, delta.NODE_KEEP)
        replay_edges = torch.full_like(edge_targets, delta.EDGE_KEEP)
        for index in range(len(items)):
            events = target_event_set(node_targets[index], edge_targets[index], layout)
            counts.append(len(events))
            dependencies = event_dependencies(events)
            applied: set[int] = set()
            while len(applied) < len(events):
                ready = sorted(
                    item
                    for item in range(len(events))
                    if item not in applied and dependencies[item] <= applied
                )
                if not ready:
                    break
                selected = ready[0]
                apply_event_to_actions(
                    events[selected], replay_nodes[index], replay_edges[index]
                )
                applied.add(selected)
            topology += int(len(applied) == len(events))
        replay = delta.apply_delta_actions(
            source, replay_nodes, replay_edges, vocabulary
        )
        for index in range(len(items)):
            exact += int(
                all(
                    torch.equal(replay[field][index], target[field][index])
                    for field in (*full_graph.NODE_FIELDS, *full_graph.EDGE_FIELDS)
                )
            )
    total = len(pairs)
    replay_rate = exact / max(1, total)
    topology_rate = topology / max(1, total)
    horizon_coverage = sum(value <= int(max_jumps) for value in counts) / max(1, total)
    return {
        "pairs": total,
        "exact_replay_rate": replay_rate,
        "topological_coverage": topology_rate,
        "max_jump_horizon": int(max_jumps),
        "horizon_coverage": horizon_coverage,
        "mean_target_events": float(np.mean(counts)),
        "median_target_events": float(np.median(counts)),
        "max_target_events": int(max(counts, default=0)),
        "passed": bool(
            replay_rate >= 0.99
            and topology_rate == 1.0
            and horizon_coverage >= 0.99
        ),
    }


def train_model(
    model: LatentGraphJumpProcess,
    representation: nn.Module,
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    representation.eval().requires_grad_(False)
    batch_size = int(preregistration["batch_size"])
    global_batch = 0
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(pairs)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), batch_size):
            items = [pairs[index] for index in order[start : start + batch_size]]
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
                    source,
                    target,
                    source_node,
                    source_edge,
                    target_node,
                    target_edge,
                    condition,
                )
                noise = torch.randn_like(endpoint) * float(
                    preregistration["latent_noise_scale"]
                )
                flow_time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
                current_latent = (
                    (1.0 - flow_time[:, None]) * noise
                    + flow_time[:, None] * endpoint
                )
                velocity = model.transport_velocity(
                    current_latent, flow_time, source_node, source["node_mask"], tokens
                )
                flow_loss = F.mse_loss(
                    velocity.float(), (endpoint - noise).float()
                )
                predicted_endpoint = current_latent + (
                    1.0 - flow_time[:, None]
                ) * velocity

                node_targets, edge_targets = delta.delta_action_targets(
                    source, target, vocabulary
                )
                working = full_graph.working_node_mask(
                    source["node_mask"],
                    int(preregistration["birth_capacity"]),
                    target["node_mask"],
                )
                current_node, current_edge, target_next, jump_time, event_counts = (
                    build_prefix_batch(
                        node_targets,
                        edge_targets,
                        model.denoiser.layout,
                        epoch=epoch,
                        global_batch=global_batch,
                        seed=int(preregistration["seed"]),
                        completion_probability=float(
                            preregistration["completion_prefix_probability"]
                        ),
                        device=device,
                    )
                )
                logits = model.denoiser(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    jump_time,
                    condition,
                    predicted_endpoint,
                )
                legal = legal_event_mask(
                    model.denoiser, source, current_node, current_edge, working
                )
                jump_loss, target_mass, jump_accuracy = orderless_jump_loss(
                    logits, legal, target_next
                )
                wrong_latent = torch.roll(predicted_endpoint, shifts=1, dims=0)
                wrong_logits = model.denoiser(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    jump_time,
                    condition,
                    wrong_latent,
                )
                wrong_jump_loss, _, _ = orderless_jump_loss(
                    wrong_logits, legal, target_next
                )
                latent_usage = F.relu(
                    float(preregistration["latent_usage_margin"])
                    + jump_loss
                    - wrong_jump_loss
                )
                latent_std = endpoint.float().std(dim=0, unbiased=False)
                variance_loss = F.relu(
                    float(preregistration["latent_min_std"]) - latent_std
                ).mean()
                loss = (
                    jump_loss
                    + float(preregistration["flow_loss_weight"]) * flow_loss
                    + float(preregistration["latent_usage_weight"]) * latent_usage
                    + float(preregistration["latent_variance_weight"]) * variance_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(preregistration["grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["jump_nll"] += float(jump_loss.detach())
            totals["target_next_probability_mass"] += float(target_mass.detach())
            totals["next_event_set_accuracy"] += float(jump_accuracy.detach())
            totals["flow_loss"] += float(flow_loss.detach())
            totals["latent_usage_loss"] += float(latent_usage.detach())
            totals["latent_variance_loss"] += float(variance_loss.detach())
            totals["posterior_std"] += float(latent_std.mean().detach())
            totals["mean_target_events"] += float(np.mean(event_counts))
            batches += 1
            global_batch += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"B38 non-finite training metrics: {row}")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def execute_flat_event(
    event_index: int,
    layout: EventLayout,
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    batch_index: int,
) -> bool:
    event = layout.decode(int(event_index))
    if event.kind == STOP:
        return True
    if event.kind in (NODE_DELETE, NODE_WRITE):
        node_actions[batch_index, event.left] = int(event.action)
    else:
        edge_actions[batch_index, event.left, event.right] = int(event.action)
        edge_actions[batch_index, event.right, event.left] = int(event.action)
    return False


@torch.no_grad()
def sample_from_source(
    model: LatentGraphJumpProcess,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    source_example: object,
    condition_tokens: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    attempts = int(preregistration["exact_raw_attempts_per_condition"])
    batch_size = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[dict[str, object]] = []
    model.eval()
    for start in range(0, attempts, batch_size):
        count = min(batch_size, attempts - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        tokens = torch.from_numpy(
            np.repeat(condition_tokens[None, ...], count, axis=0)
        ).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            source_node, source_edge = representation.encode(source)
            latent = torch.randn(
                count,
                model.transport_dim,
                generator=generator,
                device=device,
                dtype=source_node.dtype,
            ) * float(preregistration["latent_noise_scale"])
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
            working = full_graph.working_node_mask(
                source["node_mask"], int(preregistration["birth_capacity"])
            )
            node_actions = torch.full_like(source["atomic_number"], delta.NODE_KEEP)
            edge_actions = torch.full_like(source["bond"], delta.EDGE_KEEP)
            stopped = torch.zeros(count, dtype=torch.bool, device=device)
            event_counts = torch.zeros(count, dtype=torch.long, device=device)
            kind_counts = torch.zeros(count, 5, dtype=torch.long, device=device)
            for jump in range(int(preregistration["max_jumps"])):
                jump_time = torch.full(
                    (count,),
                    jump / max(1, int(preregistration["max_jumps"])),
                    device=device,
                    dtype=source_node.dtype,
                )
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
                ).float()
                legal = legal_event_mask(
                    model.denoiser, source, node_actions, edge_actions, working
                )
                if bool(stopped.any()):
                    legal[stopped] = False
                    legal[stopped, 0] = True
                probabilities = torch.softmax(
                    logits.masked_fill(~legal, -torch.inf)
                    / float(preregistration["temperature"]),
                    dim=1,
                )
                sampled = torch.multinomial(
                    probabilities, 1, generator=generator
                ).squeeze(1)
                for index in range(count):
                    if bool(stopped[index]):
                        continue
                    event = model.denoiser.layout.decode(int(sampled[index]))
                    kind_counts[index, event.kind] += 1
                    if execute_flat_event(
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
            adjacency = (
                prediction["bond"][index] > graph.BOND_NONE
            ) | (source_prediction["bond"][index] > graph.BOND_NONE)
            outputs.append(
                {
                    "generated_smiles": graph.canonical_smiles(smiles or ""),
                    "predicted_atom_count": int(
                        (prediction["atomic_number"][index] > 0).sum()
                    ),
                    "latent_norm": float(latent[index].float().norm().detach().cpu()),
                    "event_count": int(event_counts[index].detach().cpu()),
                    "stopped_by_model": bool(stopped[index].detach().cpu()),
                    "max_horizon_hit": bool(not stopped[index].detach().cpu()),
                    "node_delete_events": int(kind_counts[index, NODE_DELETE].cpu()),
                    "node_write_events": int(kind_counts[index, NODE_WRITE].cpu()),
                    "edge_delete_events": int(kind_counts[index, EDGE_DELETE].cpu()),
                    "edge_set_events": int(kind_counts[index, EDGE_SET].cpu()),
                    "affected_node_count": int(affected.sum()),
                    "affected_components": b37.component_count(
                        affected.numpy(), adjacency
                    ),
                    "outside_source_invariant": bool(
                        outside_nodes_exact and outside_edges_exact
                    ),
                }
            )
    if len(outputs) != attempts:
        raise RuntimeError(f"B38 expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def freeze_candidates(
    model: LatentGraphJumpProcess,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    """Freeze exact raw rows without inspecting pair targets or property oracles."""
    rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        generated = sample_from_source(
            model,
            representation,
            vocabulary,
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
                        "stage": "freeze_train_only_development_jump_candidates",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise RuntimeError(f"B38 freeze expected {expected} rows, found {len(rows)}")
    return rows


def evaluate_frozen_candidates(
    frozen: Sequence[Mapping[str, object]], pairs: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for row in frozen:
        pair = pairs[int(row["pair_index"])]
        smiles = str(row["generated_smiles"] or "")
        valid = bool(smiles)
        source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, smiles) if valid else None
        target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, smiles) if valid else None
        fraction, _, evaluated_properties, property_success = (
            unified.instruction_success_and_distance(
                pair.row, smiles, task_specs=base.task_specs(pair.row)
            )
        )
        similarity_success = bool(
            source_tanimoto is not None and source_tanimoto >= 0.4
        )
        evaluated.append(
            {
                **dict(row),
                "target_smiles": pair.target_smiles,
                "source_atom_count": int(np.asarray(pair.source.node_mask).sum()),
                "target_atom_count": int(np.asarray(pair.target.node_mask).sum()),
                "valid": valid,
                "source_tanimoto": float(source_tanimoto or 0.0),
                "target_tanimoto": float(target_tanimoto or 0.0),
                "property_fraction": float(fraction),
                "evaluated_properties": int(evaluated_properties),
                "property_success": bool(property_success),
                "source_similarity_success": similarity_success,
                "strict_success": bool(property_success and similarity_success),
                "source_copy_target_tanimoto": float(
                    graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
                ),
            }
        )
    metrics = base.summarize_candidates(evaluated, 20)
    for name in (
        "latent_norm",
        "event_count",
        "node_delete_events",
        "node_write_events",
        "edge_delete_events",
        "edge_set_events",
        "affected_node_count",
        "affected_components",
    ):
        metrics[f"mean_{name}"] = float(
            np.mean([float(row[name]) for row in evaluated])
        )
    metrics["outside_source_invariant_rate"] = sum(
        bool(row["outside_source_invariant"]) for row in evaluated
    ) / max(1, len(evaluated))
    metrics["learned_stop_rate"] = sum(
        bool(row["stopped_by_model"]) for row in evaluated
    ) / max(1, len(evaluated))
    metrics["max_horizon_hit_rate"] = sum(
        bool(row["max_horizon_hit"]) for row in evaluated
    ) / max(1, len(evaluated))
    return evaluated, metrics


def gate(metrics: Mapping[str, object], thresholds: Mapping[str, object]) -> dict[str, object]:
    by_count = dict(metrics["by_property_count"])
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": thresholds["validity"]},
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": thresholds["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": thresholds["mean_source_tanimoto"],
        },
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": thresholds["strict_any20"],
        },
        "two_property_strict_any20": {
            "value": dict(by_count.get("2", {})).get("strict_any20", 0.0),
            "threshold": thresholds["two_property_strict_any20"],
        },
        "three_property_strict_any20": {
            "value": dict(by_count.get("3", {})).get("strict_any20", 0.0),
            "threshold": thresholds["three_property_strict_any20"],
        },
        "outside_source_invariant_rate": {
            "value": metrics["outside_source_invariant_rate"],
            "threshold": 1.0,
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name == "exact_attempts"
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    return {"passed": not failures, "checks": checks, "failures": failures}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, checkpoint, b36_summary, b37_summary = check_locked_inputs(
        args, preregistration
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
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = LatentGraphJumpProcess(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(preregistration["condition_dim"]),
        transport_dim=int(preregistration["transport_dim"]),
        hidden_dim=int(preregistration["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(preregistration["message_layers"]),
    ).to(device)
    replay = preflight_event_replay(
        [*fit_pairs, *development_pairs],
        vocabulary,
        model.denoiser.layout,
        int(preregistration["max_jumps"]),
    )
    print(json.dumps({"stage": "event_replay_gate", **replay}, sort_keys=True), flush=True)
    if not bool(replay["passed"]):
        raise ValueError(f"B38 event replay gate failed: {replay}")

    history = train_model(
        model, representation, fit_pairs, vocabulary, preregistration, device
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
        "event_replay_gate": replay,
        "explicit_region_mask": False,
        "globally_normalized_single_event_jumps": True,
        "learned_stop_event": True,
        "orderless_event_set_objective": True,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "posthoc_molecule_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
        "de_novo_null_source_uses_same_jump_process_but_not_evaluated": True,
    }
    checkpoint_path = args.output_dir / "source_clamped_latent_graph_jump_process.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "node_dim": int(representation_config["node_dim"]),
                "edge_dim": int(representation_config["edge_dim"]),
                "condition_dim": int(preregistration["condition_dim"]),
                "transport_dim": int(preregistration["transport_dim"]),
                "hidden_dim": int(preregistration["hidden_dim"]),
                "max_atoms": int(representation_config["max_atoms"]),
                "property_count": len(unified.PROPERTY_COLUMNS),
                "node_action_count": node_action_count,
                "edge_action_count": edge_action_count,
                "message_layers": int(preregistration["message_layers"]),
            },
            "vocabulary": {
                "node_states": np.asarray(vocabulary["node_states"]).tolist(),
                "edge_states": np.asarray(vocabulary["edge_states"]).tolist(),
                "sha256": vocabulary["sha256"],
            },
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

    frozen = freeze_candidates(
        model,
        representation,
        vocabulary,
        development_pairs,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, list(frozen))
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    internal_gate = gate(metrics, dict(preregistration["gates"]))
    manifest = {
        **training_manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_candidates_sha256": frozen_sha256,
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "post_freeze_train_only_dev_target_access": True,
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "metrics": metrics,
        "internal_gate": internal_gate,
        "decision": (
            "advance_latent_graph_jump_process_to_prospective_transfer"
            if internal_gate["passed"]
            else "stop_and_diagnose_jump_support_or_transport_without_region_patches"
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
