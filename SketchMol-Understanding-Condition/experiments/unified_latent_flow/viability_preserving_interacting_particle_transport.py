#!/usr/bin/env python3
"""Fine-tune a viability-preserving interacting-particle graph transport.

B40 proved that train-only atom/bond support can lift raw validity, but exposed
two coupled failures: 6.3% of particles never reached a legal learned STOP and
orthogonal particles contracted from near-zero cosine to 0.48 through the
frozen conditional flow.  B41 addresses those mechanisms inside one model.

The B39 event kernel is fine-tuned only on fit-target topological prefixes under
the same dynamic B40 support used at inference.  An explicit STOP margin teaches
the kernel to prefer the remaining target events before termination.  At
generation, exactly twenty direct particles interact after every flow step;
repulsion and a posterior-scale spread floor preserve finite-set coverage.  No
larger pool, ranking, retry, oracle selection, target access, or molecule repair
is used.
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

import valence_constrained_latent_particle_bridge as b40  # noqa: E402


b39 = b40.b39
b38 = b40.b38
b37 = b40.b37
b36 = b40.b36
base = b40.base
belief = b40.belief
delta = b40.delta
full_graph = b40.full_graph
graph = b40.graph
hierarchical = b40.hierarchical
unified = b40.unified

PROTOCOL = "train_only_viability_preserving_interacting_particle_transport_v41"


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
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "warm_start_from_frozen_b39": True,
        "support_consistent_event_finetuning": True,
        "transport_weights_frozen_during_event_finetuning": True,
        "interacting_particle_transport": True,
        "orthogonal_latent_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "max_jumps": 64,
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
        raise ValueError(f"B41 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B41 property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            f"B41 implementation drift: expected {payload.get('implementation_sha256')}, found {actual}"
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
        "b40_evaluated_candidates_sha256",
        "b40_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("B41 locked-input manifest is incomplete")
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
        "b39_checkpoint_sha256": args.b39_checkpoint,
        "b39_evaluated_candidates_sha256": args.b39_evaluated_candidates,
        "b39_summary_sha256": args.b39_summary,
        "b40_evaluated_candidates_sha256": args.b40_evaluated_candidates,
        "b40_summary_sha256": args.b40_summary,
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
        raise ValueError(f"B41 locked input drift: {drift}")
    b22_summary, checkpoint, b36_summary, b37_summary, b39_checkpoint = (
        b40.check_locked_inputs(args, preregistration)
    )
    b40_summary = json.loads(args.b40_summary.read_text(encoding="utf-8"))
    if b40_summary.get("protocol") != b40.PROTOCOL:
        raise ValueError("B41 requires the locked B40 protocol")
    if b40_summary.get("decision") != (
        "stop_and_diagnose_support_or_particle_transport_without_gate_changes"
    ):
        raise ValueError("B41 refuses a B40 decision drift")
    b40_manifest = dict(b40_summary.get("manifest", {}))
    if b40_manifest.get("generation_target_access") is not False:
        raise ValueError("B41 refuses target-exposed B40 evidence")
    if b40_manifest.get("molecular_candidate_ranking") is not False:
        raise ValueError("B41 refuses ranked B40 evidence")
    metrics = dict(b40_summary.get("metrics", {}))
    evidence_drift = {}
    for key, expected in dict(preregistration["b40_failure_trigger"]).items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                evidence_drift[key] = {"expected": expected, "actual": actual}
        elif actual != expected:
            evidence_drift[key] = {"expected": expected, "actual": actual}
    if evidence_drift:
        raise ValueError(f"B41 refuses B40 failure-evidence drift: {evidence_drift}")
    return (
        b22_summary,
        checkpoint,
        b36_summary,
        b37_summary,
        b39_checkpoint,
        b40_summary,
    )


def build_viable_prefix_batch(
    node_targets: torch.Tensor,
    edge_targets: torch.Tensor,
    layout: b38.EventLayout,
    *,
    epoch: int,
    global_batch: int,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes, edges, targets, event_counts, executed_counts = [], [], [], [], []
    for index in range(node_targets.shape[0]):
        seed = (
            int(preregistration["seed"]) * 1_000_003
            + int(epoch) * 10_007
            + int(global_batch) * 101
            + index
        )
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
        nodes.append(current_node)
        edges.append(current_edge)
        targets.append(target_next)
        event_counts.append(len(events))
        executed_counts.append(prefix_length)
    executed = torch.as_tensor(executed_counts, dtype=torch.float32, device=device)
    return (
        torch.stack(nodes).to(device),
        torch.stack(edges).to(device),
        torch.stack(targets).to(device),
        executed / float(preregistration["max_jumps"]),
        torch.as_tensor(event_counts, dtype=torch.long, device=device),
        executed,
    )


@torch.no_grad()
def support_replay_gate(
    model: b39.LatentCardinalityGraphJumpBridge,
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    batch_size = int(preregistration["batch_size"])
    complete_stop_legal = 0
    target_events = 0
    for start in range(0, len(pairs), batch_size):
        items = list(pairs[start : start + batch_size])
        collated = base.pair_collate(items)
        source = base.move_graph_batch(collated["source"], device)
        target = base.move_graph_batch(collated["target"], device)
        node_targets, edge_targets = delta.delta_action_targets(
            source, target, vocabulary
        )
        working = full_graph.working_node_mask(
            source["node_mask"],
            int(preregistration["birth_capacity"]),
            target["node_mask"],
        )
        legal, _ = b40.constrained_event_mask(
            model.denoiser,
            source,
            node_targets,
            edge_targets,
            working,
            support,
            support_tensors,
        )
        complete_stop_legal += int(legal[:, 0].sum())
        target_events += int(
            node_targets.ne(delta.NODE_KEEP).sum()
            + torch.triu(edge_targets.ne(delta.EDGE_KEEP), diagonal=1).sum()
        )
    rate = complete_stop_legal / max(1, len(pairs))
    return {
        "fit_pairs": len(pairs),
        "fit_complete_stop_legal": complete_stop_legal,
        "fit_complete_stop_legal_rate": rate,
        "target_events": target_events,
        "passed": rate == 1.0,
    }


def fine_tune_event_kernel(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: nn.Module,
    fit_pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    model.requires_grad_(False)
    model.denoiser.requires_grad_(True)
    representation.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    batch_size = int(preregistration["batch_size"])
    global_batch = 0
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(fit_pairs)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.denoiser.train()
        for start in range(0, len(order), batch_size):
            items = [fit_pairs[index] for index in order[start : start + batch_size]]
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
                count_logits = model.cardinality_logits(
                    source_node,
                    source["node_mask"].bool(),
                    condition,
                    endpoint,
                )
                count_values = torch.arange(
                    int(preregistration["max_jumps"]) + 1,
                    device=device,
                    dtype=torch.float32,
                )
                expected_count = (
                    count_logits.float().softmax(dim=1) * count_values[None, :]
                ).sum(dim=1)
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
                ) = build_viable_prefix_batch(
                    node_targets,
                    edge_targets,
                    model.denoiser.layout,
                    epoch=epoch,
                    global_batch=global_batch,
                    preregistration=preregistration,
                    device=device,
                )
                remaining_mass = (
                    expected_count - executed_count
                ) / float(preregistration["max_jumps"])
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                logits = model.denoiser(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    jump_time,
                    condition,
                    endpoint,
                    remaining_mass,
                )
                legal, _ = b40.constrained_event_mask(
                    model.denoiser,
                    source,
                    current_node,
                    current_edge,
                    working,
                    support,
                    support_tensors,
                )
                jump_loss, target_mass, jump_accuracy = b38.orderless_jump_loss(
                    logits, legal, target_next
                )
                target_logits = torch.logsumexp(
                    logits.float().masked_fill(~target_next, -torch.inf), dim=1
                )
                incomplete = ~target_next[:, 0]
                if bool(incomplete.any()):
                    stop_margin = F.relu(
                        float(preregistration["stop_margin"])
                        + logits.float()[:, 0]
                        - target_logits
                    )[incomplete].mean()
                else:
                    stop_margin = logits.float().sum() * 0.0
                wrong_endpoint = torch.roll(endpoint, shifts=1, dims=0)
                wrong_logits = model.denoiser(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    jump_time,
                    condition,
                    wrong_endpoint,
                    remaining_mass,
                )
                wrong_jump_loss, _, _ = b38.orderless_jump_loss(
                    wrong_logits, legal, target_next
                )
                latent_usage = F.relu(
                    float(preregistration["latent_usage_margin"])
                    + jump_loss
                    - wrong_jump_loss
                )
                loss = (
                    jump_loss
                    + float(preregistration["stop_margin_weight"]) * stop_margin
                    + float(preregistration["latent_usage_weight"]) * latent_usage
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.denoiser.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["jump_nll"] += float(jump_loss.detach())
            totals["stop_margin_loss"] += float(stop_margin.detach())
            totals["latent_usage_loss"] += float(latent_usage.detach())
            totals["target_next_probability_mass"] += float(target_mass.detach())
            totals["next_event_set_accuracy"] += float(jump_accuracy.detach())
            totals["mean_target_events"] += float(target_count.float().mean())
            totals["mean_executed_prefix_events"] += float(executed_count.mean())
            totals["incomplete_prefix_rate"] += float(incomplete.float().mean())
            batches += 1
            global_batch += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"B41 non-finite training metrics: {row}")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    model.eval().requires_grad_(False)
    return history


@torch.no_grad()
def interacting_transport_particles(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: nn.Module,
    source_example: object,
    condition_tokens: np.ndarray,
    particles: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    attempts = particles.shape[0]
    chunk = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = torch.from_numpy(
        np.repeat(condition_tokens[None, ...], attempts, axis=0)
    ).to(device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, _ = representation.encode(source)
    minimum_rms = math.sqrt(model.transport_dim) * float(
        preregistration["latent_min_std"]
    )
    latent = particles.float()
    minimum_observed_rms = math.inf
    for flow_index in range(int(preregistration["flow_steps"])):
        velocities = []
        for start in range(0, attempts, chunk):
            count = min(chunk, attempts - start)
            flow_time = torch.full(
                (count,),
                (flow_index + 0.5) / int(preregistration["flow_steps"]),
                device=device,
                dtype=source_node.dtype,
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                velocity = model.transport_velocity(
                    latent[start : start + count],
                    flow_time,
                    source_node.expand(count, -1, -1),
                    source["node_mask"].expand(count, -1),
                    tokens[start : start + count],
                )
            velocities.append(velocity.float())
        proposal = latent + torch.cat(velocities, dim=0) / int(
            preregistration["flow_steps"]
        )
        center = proposal.mean(dim=0, keepdim=True)
        residual = proposal - center
        normalized = F.normalize(residual, dim=1)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(-torch.inf)
        neighbours = torch.softmax(
            similarity / float(preregistration["particle_repulsion_temperature"]),
            dim=1,
        )
        repulsion = normalized - neighbours @ normalized
        proposal = proposal + (
            float(preregistration["particle_repulsion_strength"])
            / int(preregistration["flow_steps"])
        ) * minimum_rms * repulsion
        center = proposal.mean(dim=0, keepdim=True)
        residual = proposal - center
        rms = residual.norm(dim=1).square().mean().sqrt().clamp_min(1e-8)
        minimum_observed_rms = min(minimum_observed_rms, float(rms.detach().cpu()))
        if float(rms.detach().cpu()) < minimum_rms:
            residual = residual * (minimum_rms / rms)
        latent = center + residual
    normalized = F.normalize(latent, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, dtype=torch.bool, device=device)
    final_centered_rms = float(
        (latent - latent.mean(dim=0, keepdim=True))
        .norm(dim=1)
        .square()
        .mean()
        .sqrt()
        .detach()
        .cpu()
    )
    return latent, {
        "final_particle_mean_abs_cosine": float(
            cosine[off_diagonal].abs().mean().detach().cpu()
        ),
        "final_particle_max_abs_cosine": float(
            cosine[off_diagonal].abs().max().detach().cpu()
        ),
        "final_particle_centered_rms": final_centered_rms,
        "minimum_transport_particle_rms": minimum_observed_rms,
    }


@torch.no_grad()
def sample_from_source(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: nn.Module,
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
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    particles, initial_metrics = b40.orthogonal_latent_particles(
        attempts,
        model.transport_dim,
        generator,
        device,
        float(preregistration["latent_noise_scale"]),
    )
    particles, transport_metrics = interacting_transport_particles(
        model,
        representation,
        source_example,
        condition_tokens,
        particles,
        preregistration,
        device,
    )
    particle_metrics = {**initial_metrics, **transport_metrics}
    outputs: list[dict[str, object]] = []
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
                legal, diagnostics = b40.constrained_event_mask(
                    model.denoiser,
                    source,
                    node_actions,
                    edge_actions,
                    working,
                    support,
                    support_tensors,
                )
                masked_events += diagnostics["base_legal"] - diagnostics[
                    "constrained_legal"
                ]
                base_events += diagnostics["base_legal"]
                stop_masked_steps += diagnostics["stop_masked"].long()
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
                    **particle_metrics,
                }
            )
    if len(outputs) != attempts:
        raise RuntimeError(f"B41 expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def freeze_candidates(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation: nn.Module,
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
                        "stage": "freeze_train_only_viability_particle_candidates",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise RuntimeError(f"B41 freeze expected {expected} rows, found {len(rows)}")
    return rows


def evaluate_frozen_candidates(
    frozen: Sequence[Mapping[str, object]], pairs: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated, metrics = b40.evaluate_frozen_candidates(frozen, pairs)
    metrics["mean_final_particle_centered_rms"] = float(
        np.mean([float(row["final_particle_centered_rms"]) for row in evaluated])
    )
    metrics["mean_minimum_transport_particle_rms"] = float(
        np.mean([float(row["minimum_transport_particle_rms"]) for row in evaluated])
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
        b39_checkpoint,
        b40_summary,
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
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
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
    replay = support_replay_gate(
        model,
        fit_pairs,
        vocabulary,
        support,
        support_tensors,
        preregistration,
        device,
    )
    print(json.dumps({"stage": "support_replay_gate", **replay}, sort_keys=True), flush=True)
    if not bool(replay["passed"]):
        raise ValueError(f"B41 support replay gate failed: {replay}")
    history = fine_tune_event_kernel(
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
        "support_replay_gate": replay,
        "warm_start_from_frozen_b39": True,
        "support_consistent_event_finetuning": True,
        "transport_weights_frozen_during_event_finetuning": True,
        "interacting_particle_transport": True,
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
    checkpoint_path = args.output_dir / "viability_interacting_particle_transport.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "vocabulary": dict(b39_checkpoint["vocabulary"]),
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
    horizon_threshold = float(preregistration["additional_gates"]["max_horizon_hit_rate"])
    internal_gate["checks"]["max_horizon_hit_rate"] = {
        "threshold": horizon_threshold,
        "value": float(metrics["max_horizon_hit_rate"]),
    }
    if float(metrics["max_horizon_hit_rate"]) > horizon_threshold:
        internal_gate["failures"].append("max_horizon_hit_rate")
        internal_gate["passed"] = False
    manifest = {
        **training_manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "post_freeze_train_only_dev_target_access": True,
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
        "b40_decision": b40_summary.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "metrics": metrics,
        "internal_gate": internal_gate,
        "decision": (
            "advance_viability_interacting_transport_to_once_only_prospective_confirmation"
            if internal_gate["passed"]
            else "stop_and_diagnose_viability_or_particle_support_without_gate_changes"
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
