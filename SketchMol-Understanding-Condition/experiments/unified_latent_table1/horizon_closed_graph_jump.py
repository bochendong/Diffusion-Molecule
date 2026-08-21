#!/usr/bin/env python3
"""Finite-horizon closure for the frozen B41 molecular graph-jump process.

The B41 process may visit non-materializable intermediate edit states.  Its
exact-terminal support correctly prevents STOP in those states, but a particle
that reaches the fixed horizon can still be serialized from an unfinished
state.  This module augments the Markov state with the most recent exactly
materializable state.  On the final transition, an unfinished particle takes a
deterministic terminal transition to that checkpoint and stops.

The checkpoint is updated before event sampling, uses no target, property
oracle, or candidate comparison, and never creates an additional candidate.
It is therefore a state-space closure rather than post-hoc molecule repair,
retry, ranking, filtering, or resampling.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

import valid_terminal_molecule_latent_jump as valid_terminal
import viability_preserving_interacting_particle_transport as b41
from dead_end_safe_support import DeadEndSafeSupport


base = b41.base
b40 = b41.b40
b38 = b41.b38
b37 = b41.b37
delta = b41.delta
full_graph = b41.full_graph
graph = b41.graph


@torch.no_grad()
def sample_from_source(
    model,
    representation,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    source_example: object,
    condition_tokens: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    """Sample exact-n particles with an in-process valid terminal checkpoint."""

    attempts = int(preregistration["exact_raw_attempts_per_condition"])
    batch_size = int(preregistration["sample_batch_size"])
    max_jumps = int(preregistration["max_jumps"])
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
    particles, transport_metrics = b41.interacting_transport_particles(
        model,
        representation,
        source_example,
        condition_tokens,
        particles,
        preregistration,
        device,
    )
    particle_metrics = {**initial_metrics, **transport_metrics}
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    safe_support = DeadEndSafeSupport(exact_support)
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
            checkpoint_node = node_actions.clone()
            checkpoint_edge = edge_actions.clone()
            checkpoint_event_count = torch.zeros(count, dtype=torch.long, device=device)
            checkpoint_updates = torch.zeros(count, dtype=torch.long, device=device)
            checkpoint_restored = torch.zeros(count, dtype=torch.bool, device=device)
            horizon_forced_stop = torch.zeros(count, dtype=torch.bool, device=device)
            stopped = torch.zeros(count, dtype=torch.bool, device=device)
            event_counts = torch.zeros(count, dtype=torch.long, device=device)
            kind_counts = torch.zeros(count, 5, dtype=torch.long, device=device)
            masked_events = torch.zeros(count, dtype=torch.long, device=device)
            base_events = torch.zeros(count, dtype=torch.long, device=device)
            stop_masked_steps = torch.zeros(count, dtype=torch.long, device=device)
            for jump_index in range(max_jumps):
                jump_time = event_counts.float() / float(max_jumps)
                remaining_mass = (
                    predicted_cardinality.float() - event_counts.float()
                ) / float(max_jumps)
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
                legal, diagnostics = safe_support(
                    model.denoiser,
                    source,
                    node_actions,
                    edge_actions,
                    working,
                    support,
                    support_tensors,
                )
                materializable = diagnostics.get("exact_molecule_materializable")
                if materializable is None:
                    materializable = valid_terminal.materializable_terminal_states(
                        source, node_actions, edge_actions, vocabulary
                    )
                materializable = materializable.bool()
                checkpointable = materializable & ~stopped
                if bool(checkpointable.any()):
                    checkpoint_node[checkpointable] = node_actions[checkpointable]
                    checkpoint_edge[checkpointable] = edge_actions[checkpointable]
                    checkpoint_event_count[checkpointable] = event_counts[checkpointable]
                    checkpoint_updates[checkpointable] += 1

                # A dead-end fallback must not STOP from an invalid state.
                invalid_forced_stop = (
                    ~materializable & legal[:, 0] & legal.sum(dim=1).eq(1) & ~stopped
                )
                final_transition = jump_index + 1 == max_jumps
                close = invalid_forced_stop.clone()
                if final_transition:
                    close |= ~stopped
                    horizon_forced_stop |= ~stopped
                restore = close & ~materializable
                if bool(restore.any()):
                    node_actions[restore] = checkpoint_node[restore]
                    edge_actions[restore] = checkpoint_edge[restore]
                    checkpoint_restored |= restore
                if bool(close.any()):
                    legal = legal.clone()
                    legal[close] = False
                    legal[close, 0] = True

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
            canonical = graph.canonical_smiles(smiles or "")
            if not canonical:
                raise RuntimeError(
                    "Horizon-closed graph jump emitted a non-materializable state"
                )
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
                    "generated_smiles": canonical,
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
                    "max_horizon_hit": bool(horizon_forced_stop[index].cpu()),
                    "horizon_forced_stop": bool(horizon_forced_stop[index].cpu()),
                    "horizon_checkpoint_restored": bool(
                        checkpoint_restored[index].cpu()
                    ),
                    "last_valid_event_count": int(checkpoint_event_count[index].cpu()),
                    "valid_checkpoint_updates": int(checkpoint_updates[index].cpu()),
                    "node_delete_events": int(
                        kind_counts[index, b38.NODE_DELETE].cpu()
                    ),
                    "node_write_events": int(kind_counts[index, b38.NODE_WRITE].cpu()),
                    "edge_delete_events": int(
                        kind_counts[index, b38.EDGE_DELETE].cpu()
                    ),
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
        raise RuntimeError(
            f"Horizon-closed B41 expected {attempts} attempts, produced {len(outputs)}"
        )
    return outputs
