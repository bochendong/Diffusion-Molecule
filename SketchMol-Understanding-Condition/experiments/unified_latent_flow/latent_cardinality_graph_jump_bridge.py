#!/usr/bin/env python3
"""Warm-start a latent-cardinality graph jump bridge from frozen B38.

B38 fixed simultaneous graph expansion, but teacher-forced STOP calibration
still produced 29.5 events for targets averaging 11.8.  B39 models a molecular
edit as a random finite set: a transported latent predicts its cardinality and
each sampled graph event consumes one unit of continuous remaining edit mass.
The STOP hazard and event-family intensities see that mass, while STOP remains a
learned stochastic event rather than a hard length constraint.

Training and inference use the same absolute jump clock.  Completed target sets
are sometimes extended by train-only legal overrun events and still labelled
STOP, exposing the model to the off-manifold states that B38 never saw.  The
frozen B38 checkpoint supplies the already learned event-location field; only
train-derived fit pairs are used for B39 optimization.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
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

import source_clamped_latent_graph_jump_process as b38  # noqa: E402


b37 = b38.b37
b36 = b38.b36
b22 = b38.b22
base = b38.base
belief = b38.belief
delta = b38.delta
full_graph = b38.full_graph
graph = b38.graph
hierarchical = b38.hierarchical
unified = b38.unified

PROTOCOL = "train_only_latent_cardinality_graph_jump_bridge_v39"


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
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "warm_start_from_frozen_b38": True,
        "latent_cardinality_distribution": True,
        "continuous_remaining_edit_mass": True,
        "hard_event_budget": False,
        "learned_stop_event": True,
        "absolute_jump_clock_train_inference_match": True,
        "train_only_completed_set_overrun_exposure": True,
        "explicit_region_mask": False,
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
        "epochs": 4,
        "flow_steps": 8,
        "birth_capacity": 8,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B39 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B39 property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            f"B39 implementation drift: expected {payload.get('implementation_sha256')}, found {actual}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_summary_sha256",
        "b37_summary_sha256",
        "b38_checkpoint_sha256",
        "b38_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("B39 locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_summary_sha256": args.b36_summary,
        "b37_summary_sha256": args.b37_summary,
        "b38_checkpoint_sha256": args.b38_checkpoint,
        "b38_summary_sha256": args.b38_summary,
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
        raise ValueError(f"B39 locked input drift: {drift}")
    b22_summary, b22_checkpoint, b36_summary, b37_summary = b38.check_locked_inputs(
        args, preregistration
    )
    b38_summary = json.loads(args.b38_summary.read_text(encoding="utf-8"))
    if b38_summary.get("protocol") != b38.PROTOCOL:
        raise ValueError("B39 requires the locked B38 protocol")
    if b38_summary.get("decision") != "stop_and_diagnose_jump_support_or_transport_without_region_patches":
        raise ValueError("B39 refuses a B38 decision drift")
    metrics = dict(b38_summary.get("metrics", {}))
    evidence_drift = {}
    for key, expected in dict(preregistration["b38_failure_trigger"]).items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                evidence_drift[key] = {"expected": expected, "actual": actual}
        elif actual != expected:
            evidence_drift[key] = {"expected": expected, "actual": actual}
    if evidence_drift:
        raise ValueError(f"B39 refuses B38 failure-evidence drift: {evidence_drift}")
    b38_checkpoint = torch.load(
        args.b38_checkpoint, map_location="cpu", weights_only=False
    )
    if b38_checkpoint.get("stage") != b38.PROTOCOL:
        raise ValueError("B39 refuses a non-B38 warm-start checkpoint")
    checkpoint_manifest = dict(b38_checkpoint.get("manifest", {}))
    if checkpoint_manifest.get("generation_target_access") is not False:
        raise ValueError("B39 refuses a target-exposed B38 checkpoint")
    if dict(b38_summary.get("manifest", {})).get("checkpoint_sha256") != locked[
        "b38_checkpoint_sha256"
    ]:
        raise ValueError("B39 B38 checkpoint/summary hash mismatch")
    return (
        b22_summary,
        b22_checkpoint,
        b36_summary,
        b37_summary,
        b38_summary,
        b38_checkpoint,
    )


class CardinalityJumpEventField(b38.JumpEventField):
    """B38 event-location field plus a continuous remaining-mass intensity."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        hidden_dim = int(kwargs["hidden_dim"])
        self.remaining_mass_bias = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )
        nn.init.zeros_(self.remaining_mass_bias[-1].weight)
        nn.init.zeros_(self.remaining_mass_bias[-1].bias)

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
        remaining_mass: torch.Tensor,
    ) -> torch.Tensor:
        logits = super().forward(
            node_actions,
            edge_actions,
            source_node,
            source_edge,
            source_mask,
            working_mask,
            jump_time,
            condition,
            latent,
        )
        features = torch.stack(
            [remaining_mass, remaining_mass.abs(), remaining_mass.clamp_min(0.0)],
            dim=1,
        )
        bias = self.remaining_mass_bias(features.float()).to(logits.dtype)
        layout = self.layout
        return torch.cat(
            [
                logits[:, :1] + bias[:, 0:1],
                logits[:, layout.node_delete_offset : layout.node_write_offset]
                + bias[:, 1:2],
                logits[:, layout.node_write_offset : layout.edge_delete_offset]
                + bias[:, 2:3],
                logits[:, layout.edge_delete_offset : layout.edge_set_offset]
                + bias[:, 3:4],
                logits[:, layout.edge_set_offset :] + bias[:, 4:5],
            ],
            dim=1,
        )


class LatentCardinalityGraphJumpBridge(b38.LatentGraphJumpProcess):
    def __init__(self, **kwargs: object) -> None:
        max_jumps = int(kwargs.pop("max_jumps"))
        super().__init__(**kwargs)
        self.denoiser = CardinalityJumpEventField(
            node_action_count=int(kwargs["node_state_count"]),
            edge_action_count=int(kwargs["edge_state_count"]),
            source_node_dim=int(kwargs["node_dim"]),
            source_edge_dim=int(kwargs["edge_dim"]),
            context_dim=int(kwargs["condition_dim"]) + int(kwargs["transport_dim"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            max_atoms=int(kwargs["max_atoms"]),
            layers=int(kwargs["message_layers"]),
        )
        head_input = (
            int(kwargs["node_dim"])
            + int(kwargs["condition_dim"])
            + int(kwargs["transport_dim"])
        )
        self.event_count_head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, int(kwargs["hidden_dim"])),
            nn.SiLU(),
            nn.Linear(int(kwargs["hidden_dim"]), max_jumps + 1),
        )

    def cardinality_logits(
        self,
        source_node: torch.Tensor,
        source_mask: torch.Tensor,
        condition: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        pooled = (source_node * source_mask.unsqueeze(-1)).sum(dim=1)
        pooled /= source_mask.sum(dim=1, keepdim=True).clamp_min(1).sqrt()
        return self.event_count_head(torch.cat([pooled, condition, latent], dim=-1))


def legal_event_indices_cpu(
    layout: b38.EventLayout,
    source_active: torch.Tensor,
    source_bond: torch.Tensor,
    working: torch.Tensor,
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
) -> list[int]:
    current_active = (~node_actions.eq(delta.NODE_DELETE)) & (
        source_active | node_actions.ge(delta.NODE_WRITE_OFFSET + 1)
    )
    values: list[int] = []
    for node in range(layout.nodes):
        if not bool(working[node]) or int(node_actions[node]) != delta.NODE_KEEP:
            continue
        if bool(source_active[node]):
            values.append(layout.encode(b38.GraphEvent(b38.NODE_DELETE, node, action=delta.NODE_DELETE)))
        for payload in range(layout.node_payloads):
            values.append(
                layout.encode(
                    b38.GraphEvent(
                        b38.NODE_WRITE,
                        node,
                        action=payload + delta.NODE_WRITE_OFFSET + 1,
                    )
                )
            )
    for left in range(layout.nodes - 1):
        for right in range(left + 1, layout.nodes):
            if int(edge_actions[left, right]) != delta.EDGE_KEEP:
                continue
            if not bool(current_active[left] and current_active[right]):
                continue
            if bool(source_bond[left, right]):
                values.append(
                    layout.encode(
                        b38.GraphEvent(
                            b38.EDGE_DELETE,
                            left,
                            right,
                            delta.EDGE_DELETE,
                        )
                    )
                )
            for payload in range(layout.edge_payloads):
                values.append(
                    layout.encode(
                        b38.GraphEvent(
                            b38.EDGE_SET,
                            left,
                            right,
                            payload + delta.EDGE_SET_OFFSET + 1,
                        )
                    )
                )
    return values


def build_absolute_prefix_batch(
    node_targets: torch.Tensor,
    edge_targets: torch.Tensor,
    source: Mapping[str, torch.Tensor],
    working: torch.Tensor,
    layout: b38.EventLayout,
    *,
    epoch: int,
    global_batch: int,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes, edges, targets = [], [], []
    event_counts, executed_counts, overrun_counts = [], [], []
    source_active = source["atomic_number"].gt(0).detach().cpu()
    source_bond = source["bond"].gt(graph.BOND_NONE).detach().cpu()
    working_cpu = working.detach().cpu()
    for index in range(node_targets.shape[0]):
        seed = (
            int(preregistration["seed"]) * 1_000_003
            + int(epoch) * 10_007
            + int(global_batch) * 101
            + index
        )
        rng = random.Random(seed)
        events = b38.target_event_set(
            node_targets[index].detach().cpu(),
            edge_targets[index].detach().cpu(),
            layout,
        )
        current_node, current_edge, target_next, prefix_length = (
            b38.random_topological_prefix(
                events,
                layout,
                seed=seed,
                completion_probability=float(
                    preregistration["completion_prefix_probability"]
                ),
            )
        )
        overrun = 0
        if (
            prefix_length == len(events)
            and rng.random()
            < float(preregistration["completed_set_overrun_probability"])
        ):
            requested = rng.randint(
                1, int(preregistration["maximum_overrun_events"])
            )
            for _ in range(requested):
                legal = legal_event_indices_cpu(
                    layout,
                    source_active[index],
                    source_bond[index],
                    working_cpu[index],
                    current_node,
                    current_edge,
                )
                if not legal:
                    break
                selected = rng.choice(legal)
                b38.apply_event_to_actions(
                    layout.decode(selected), current_node, current_edge
                )
                overrun += 1
            target_next.zero_()
            target_next[0] = True
        nodes.append(current_node)
        edges.append(current_edge)
        targets.append(target_next)
        event_counts.append(len(events))
        executed_counts.append(prefix_length + overrun)
        overrun_counts.append(overrun)
    executed = torch.as_tensor(executed_counts, dtype=torch.float32, device=device)
    jump_time = executed / float(preregistration["max_jumps"])
    return (
        torch.stack(nodes).to(device),
        torch.stack(edges).to(device),
        torch.stack(targets).to(device),
        jump_time,
        torch.as_tensor(event_counts, dtype=torch.long, device=device),
        executed,
        torch.as_tensor(overrun_counts, dtype=torch.float32, device=device),
    )


def train_model(
    model: LatentCardinalityGraphJumpBridge,
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
    count_values = torch.arange(
        int(preregistration["max_jumps"]) + 1, device=device, dtype=torch.float32
    )
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
                (
                    current_node,
                    current_edge,
                    target_next,
                    jump_time,
                    target_count,
                    executed_count,
                    overrun_count,
                ) = build_absolute_prefix_batch(
                    node_targets,
                    edge_targets,
                    source,
                    working,
                    model.denoiser.layout,
                    epoch=epoch,
                    global_batch=global_batch,
                    preregistration=preregistration,
                    device=device,
                )
                count_logits = model.cardinality_logits(
                    source_node,
                    source["node_mask"].bool(),
                    condition,
                    predicted_endpoint,
                )
                cardinality_loss = F.cross_entropy(
                    count_logits.float(), target_count
                )
                expected_count = (
                    count_logits.float().softmax(dim=1) * count_values[None, :]
                ).sum(dim=1)
                remaining_mass = (
                    expected_count - executed_count
                ) / float(preregistration["max_jumps"])
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
                    remaining_mass,
                )
                legal = b38.legal_event_mask(
                    model.denoiser, source, current_node, current_edge, working
                )
                jump_loss, target_mass, jump_accuracy = b38.orderless_jump_loss(
                    logits, legal, target_next
                )
                wrong_latent = torch.roll(predicted_endpoint, shifts=1, dims=0)
                wrong_count_logits = model.cardinality_logits(
                    source_node,
                    source["node_mask"].bool(),
                    condition,
                    wrong_latent,
                )
                wrong_expected_count = (
                    wrong_count_logits.float().softmax(dim=1) * count_values[None, :]
                ).sum(dim=1)
                wrong_remaining = (
                    wrong_expected_count - executed_count
                ) / float(preregistration["max_jumps"])
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
                    wrong_remaining,
                )
                wrong_jump_loss, _, _ = b38.orderless_jump_loss(
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
                    + float(preregistration["cardinality_loss_weight"])
                    * cardinality_loss
                    + float(preregistration["flow_loss_weight"]) * flow_loss
                    + float(preregistration["latent_usage_weight"]) * latent_usage
                    + float(preregistration["latent_variance_weight"]) * variance_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            predicted_count = count_logits.float().argmax(dim=1)
            totals["loss"] += float(loss.detach())
            totals["jump_nll"] += float(jump_loss.detach())
            totals["cardinality_nll"] += float(cardinality_loss.detach())
            totals["cardinality_mae"] += float(
                (predicted_count - target_count).abs().float().mean().detach()
            )
            totals["cardinality_exact"] += float(
                predicted_count.eq(target_count).float().mean().detach()
            )
            totals["target_next_probability_mass"] += float(target_mass.detach())
            totals["next_event_set_accuracy"] += float(jump_accuracy.detach())
            totals["flow_loss"] += float(flow_loss.detach())
            totals["latent_usage_loss"] += float(latent_usage.detach())
            totals["latent_variance_loss"] += float(variance_loss.detach())
            totals["posterior_std"] += float(latent_std.mean().detach())
            totals["mean_target_events"] += float(target_count.float().mean())
            totals["mean_executed_prefix_events"] += float(executed_count.mean())
            totals["overrun_exposure_rate"] += float(overrun_count.gt(0).float().mean())
            batches += 1
            global_batch += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"B39 non-finite training metrics: {row}")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def sample_from_source(
    model: LatentCardinalityGraphJumpBridge,
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
            cardinality_logits = model.cardinality_logits(
                source_node,
                source["node_mask"].bool(),
                condition,
                latent,
            ).float()
            cardinality_probability = torch.softmax(
                cardinality_logits
                / float(preregistration["cardinality_temperature"]),
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
                legal = b38.legal_event_mask(
                    model.denoiser, source, node_actions, edge_actions, working
                )
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
                }
            )
    if len(outputs) != attempts:
        raise RuntimeError(f"B39 expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def freeze_candidates(
    model: LatentCardinalityGraphJumpBridge,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
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
                        "stage": "freeze_train_only_cardinality_bridge_candidates",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise RuntimeError(f"B39 freeze expected {expected} rows, found {len(rows)}")
    return rows


def evaluate_frozen_candidates(
    frozen: Sequence[Mapping[str, object]], pairs: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated, metrics = b38.evaluate_frozen_candidates(frozen, pairs)
    metrics["mean_predicted_cardinality"] = float(
        np.mean([float(row["predicted_cardinality"]) for row in evaluated])
    )
    metrics["mean_abs_cardinality_execution_error"] = float(
        np.mean(
            [
                abs(float(row["predicted_cardinality"]) - float(row["event_count"]))
                for row in evaluated
            ]
        )
    )
    metrics["mean_cardinality_residual_at_stop"] = float(
        np.mean([float(row["cardinality_residual_at_stop"]) for row in evaluated])
    )
    return evaluated, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        b22_summary,
        checkpoint,
        b36_summary,
        b37_summary,
        b38_summary,
        b38_checkpoint,
    ) = check_locked_inputs(args, preregistration)
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
    model = LatentCardinalityGraphJumpBridge(
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
    incompatible = model.load_state_dict(
        dict(b38_checkpoint["model_state"]), strict=False
    )
    allowed_missing = (
        "denoiser.remaining_mass_bias.",
        "event_count_head.",
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing)
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(
            "B39 warm-start mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    replay = b38.preflight_event_replay(
        [*fit_pairs, *development_pairs],
        vocabulary,
        model.denoiser.layout,
        int(preregistration["max_jumps"]),
    )
    print(json.dumps({"stage": "event_replay_gate", **replay}, sort_keys=True), flush=True)
    if not bool(replay["passed"]):
        raise ValueError(f"B39 event replay gate failed: {replay}")
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
        "warm_start_from_frozen_b38": True,
        "warm_start_missing_parameters": list(incompatible.missing_keys),
        "latent_cardinality_distribution": True,
        "continuous_remaining_edit_mass": True,
        "hard_event_budget": False,
        "learned_stop_event": True,
        "absolute_jump_clock_train_inference_match": True,
        "train_only_completed_set_overrun_exposure": True,
        "explicit_region_mask": False,
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
    }
    checkpoint_path = args.output_dir / "latent_cardinality_graph_jump_bridge.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": dict(b38_checkpoint["model_config"]),
            "vocabulary": dict(b38_checkpoint["vocabulary"]),
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
    base.write_candidate_rows(frozen_path, frozen)
    evaluated, metrics = evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    internal_gate = b38.gate(metrics, dict(preregistration["gates"]))
    manifest = {
        **training_manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "post_freeze_train_only_dev_target_access": True,
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
        "b38_decision": b38_summary.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "metrics": metrics,
        "internal_gate": internal_gate,
        "decision": (
            "advance_latent_cardinality_jump_bridge_to_prospective_transfer"
            if internal_gate["passed"]
            else "stop_and_diagnose_cardinality_or_mark_transport_without_length_clipping"
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
