#!/usr/bin/env python3
"""D1: freeze B41, tilt its event grid with B31 joint energy. No DSL, no ranking.

Each of n=20 attempts draws one (site, token) from the frozen B31 energy,
joins that fragment, and MCS-aligns the product onto the source. The aligned
delta becomes an additive logit bias on B41's legal event grid. The emitted
molecule is always B41 STOP materialization, never the B31 join SMILES.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
WORKTREE_LATENT = PROJECT_DIR / "experiments" / "unified_latent_flow"
WORKTREE_PROJECT = PROJECT_DIR
C_DIR = PROJECT_DIR / "experiments" / "unified_action_categorical"
for path in (WORKTREE_LATENT, WORKTREE_PROJECT, C_DIR, PROJECT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import assay_joint_site_token_latent as b31  # noqa: E402
import eval_d0_b41_table1 as d0b  # noqa: E402
import eval_joint_graph_fragment_categorical_c1 as c1  # noqa: E402
import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402

b40 = b41.b40
b39 = b41.b39
b38 = b41.b38
b37 = b41.b37
base = b41.base
delta = b41.delta
graph = b41.graph
full_graph = b41.full_graph
kernel = b31.kernel
b27 = b31.b27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--fragment-checkpoint", required=True, type=Path)
    parser.add_argument("--b31-checkpoint", required=True, type=Path)
    parser.add_argument("--b31-protocol-manifest", required=True, type=Path)
    parser.add_argument("--b22-checkpoint", required=True, type=Path)
    parser.add_argument("--b22-summary", required=True, type=Path)
    parser.add_argument("--b36-summary", required=True, type=Path)
    parser.add_argument("--b37-summary", required=True, type=Path)
    parser.add_argument("--b38-checkpoint", required=True, type=Path)
    parser.add_argument("--b38-summary", required=True, type=Path)
    parser.add_argument("--b39-checkpoint", required=True, type=Path)
    parser.add_argument("--b39-summary", required=True, type=Path)
    parser.add_argument("--b39-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b40-summary", required=True, type=Path)
    parser.add_argument("--b40-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-checkpoint", required=True, type=Path)
    parser.add_argument("--b41-summary", required=True, type=Path)
    parser.add_argument("--b41-evaluated-candidates", required=True, type=Path)
    parser.add_argument("--b41-protocol-manifest", required=True, type=Path)
    parser.add_argument("--d1-protocol-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    d1 = json.loads(args.d1_protocol_manifest.read_text(encoding="utf-8"))
    b31_prereg = json.loads(args.b31_protocol_manifest.read_text(encoding="utf-8"))
    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        _b22_summary,
        b22_checkpoint,
        _b36_summary,
        _b37_summary,
        _b39_checkpoint,
        _b40_summary,
    ) = b41.check_locked_inputs(args, b41_prereg)
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)

    selected_pairs = d0b.reconstruct_support_pairs(args, b41_prereg, b22_checkpoint)
    fit_pairs, _development_pairs, _split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(b41_prereg["development_split_seed"]),
        development_source_limit=int(b41_prereg["development_source_limit"]),
    )
    representation, representation_config, _summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        transport_dim=int(b41_prereg["transport_dim"]),
        hidden_dim=int(b41_prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(b41_prereg["max_jumps"]),
        property_count=len(b41.unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)

    fragment_model, target_fragments, target_endpoints, _frozen = (
        b27.load_frozen_fragment_model(args.fragment_checkpoint, device, b31_prereg)
    )
    energy_payload = torch.load(args.b31_checkpoint, map_location=device, weights_only=False)
    energy_config = dict(energy_payload["model_config"])
    energy_model = b27.LatentPropertyEnergy(
        endpoint_dim=int(energy_config["endpoint_dim"]),
        context_dim=int(energy_config["context_dim"]),
        hidden_dim=int(energy_config["hidden_dim"]),
    ).to(device)
    energy_model.load_state_dict(energy_payload["model_state"])
    energy_model.eval()
    for parameter in energy_model.parameters():
        parameter.requires_grad_(False)

    conditions = d0b.load_table1_conditions(
        args.eval_csv,
        limit=int(args.eval_limit),
        condition_dim=int(b41_prereg["condition_dim"]),
        graph_fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
        max_atoms=int(representation_config["max_atoms"]),
    )
    latents = kernel.encode_sources(
        representation,
        conditions,
        device,
        batch_size=int(b31_prereg["encoding_batch_size"]),
    )
    fragment_config = SimpleNamespace(
        min_core_heavy_atoms=int(b31_prereg["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(b31_prereg["max_variable_heavy_atoms"]),
        fingerprint_bits=int(b31_prereg["fingerprint_bits"]),
        num_attempts=int(d1["exact_raw_attempts_per_condition"]),
        flow_steps=int(b31_prereg["flow_steps"]),
        site_temperature=float(b31_prereg["site_temperature"]),
        energy_chunk_size=int(b31_prereg["energy_chunk_size"]),
        energy_scale_floor=float(b31_prereg["energy_scale_floor"]),
        energy_weight=float(d1["energy_weight"]),
        seed=int(d1["seed"]),
    )
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_support = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    rows: list[dict[str, object]] = []
    skipped = 0
    aligned_goals = 0
    total_goals = 0
    goal_event_total = 0
    zero_bias = 0
    dead_end_stops = 0
    started = time.perf_counter()
    try:
        for index, condition in enumerate(conditions):
            try:
                event_bias, goal_stats = b31_event_bias(
                    fragment_model,
                    energy_model,
                    condition,
                    latents[index],
                    target_fragments,
                    target_endpoints,
                    fragment_config,
                    vocabulary,
                    model.denoiser.layout,
                    b41_prereg,
                    device,
                    seed=int(d1["seed"]) * 100000 + index + int(d1["b31_goal_seed_offset"]),
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "stage": "goal_bias_fallback",
                            "condition_id": condition.condition_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                event_bias = torch.zeros(
                    int(d1["exact_raw_attempts_per_condition"]),
                    int(model.denoiser.layout.total_events),
                    dtype=torch.float32,
                )
                goal_stats = {
                    "attempts": int(d1["exact_raw_attempts_per_condition"]),
                    "aligned": 0,
                    "goal_events": 0,
                    "zero_bias": int(d1["exact_raw_attempts_per_condition"]),
                }
            aligned_goals += int(goal_stats["aligned"])
            total_goals += int(goal_stats["attempts"])
            goal_event_total += int(goal_stats["goal_events"])
            zero_bias += int(goal_stats["zero_bias"])
            try:
                generated = sample_from_source_with_event_bias(
                    model,
                    representation,
                    vocabulary,
                    support,
                    support_tensors,
                    condition.source,
                    np.asarray(condition.condition),
                    b41_prereg,
                    device,
                    int(d1["seed"]) * 100000 + index,
                    event_bias,
                    float(d1["energy_weight"]),
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "stage": "sample_fallback_vanilla_b41",
                            "condition_id": condition.condition_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                generated = b41.sample_from_source(
                    model,
                    representation,
                    vocabulary,
                    support,
                    support_tensors,
                    condition.source,
                    np.asarray(condition.condition),
                    b41_prereg,
                    device,
                    int(d1["seed"]) * 100000 + index,
                )
            if len(generated) != int(d1["exact_raw_attempts_per_condition"]):
                skipped += 1
                continue
            for attempt, candidate in enumerate(generated, start=1):
                dead_end_stops += int(candidate.get("dead_end_stop", 0))
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "task": condition.task,
                        "source_smiles": condition.source_smiles,
                        "generated_smiles": candidate.get("generated_smiles", ""),
                        "sample_index": attempt,
                        "candidate_index": attempt,
                        "method": d1["protocol"],
                        "family": "b41_graph_event_b31_energy",
                        "op": "latent_graph_jump_energy_tilt",
                        "goal_aligned": int(candidate.get("goal_aligned", 0)),
                        "goal_event_count": int(candidate.get("goal_event_count", 0)),
                        "dead_end_stop": int(candidate.get("dead_end_stop", 0)),
                    }
                )
            if (index + 1) % 20 == 0 or index + 1 == len(conditions):
                elapsed = time.perf_counter() - started
                done = index + 1
                sec_per = elapsed / done
                print(
                    json.dumps(
                        {
                            "stage": "sampled",
                            "done": done,
                            "total": len(conditions),
                            "elapsed_sec": round(elapsed, 1),
                            "sec_per_condition": round(sec_per, 3),
                            "eta_sec": round(sec_per * (len(conditions) - done), 1),
                            "aligned_goal_fraction": aligned_goals / max(1, total_goals),
                            "dead_end_stops": dead_end_stops,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.viability_event_mask = original_support

    candidate_path = args.output_dir / "d1_b31_energy_on_b41_table1_n20_candidates.csv"
    d0b.write_rows(candidate_path, rows)
    sampling = {
        "protocol": d1["protocol"],
        "device": str(device),
        "eval_csv": str(args.eval_csv),
        "loaded_conditions": len(conditions),
        "candidate_rows": len(rows),
        "attempts_per_condition": int(d1["exact_raw_attempts_per_condition"]),
        "skipped_count": skipped,
        "candidate_csv": str(candidate_path),
        "molecular_candidate_ranking": False,
        "task_router": False,
        "oracle_in_environment": False,
        "family_mixer": False,
        "energy_weight": float(d1["energy_weight"]),
        "aligned_goal_count": aligned_goals,
        "aligned_goal_fraction": aligned_goals / max(1, total_goals),
        "mean_goal_events": goal_event_total / max(1, total_goals),
        "zero_bias_attempts": zero_bias,
        "dead_end_stops": dead_end_stops,
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "sec_per_condition": round(
            (time.perf_counter() - started) / max(1, len(conditions)), 3
        ),
        "exact_stop_support": exact_support.manifest(),
    }
    (args.output_dir / "sampling_summary.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(sampling, indent=2, sort_keys=True))
    return 0


@torch.no_grad()
def b31_event_bias(
    fragment_model,
    energy_model,
    condition,
    source_latent: np.ndarray,
    target_fragments,
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    vocabulary,
    layout: b38.EventLayout,
    b41_prereg: dict,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    attempts = int(config.num_attempts)
    bias = torch.zeros(attempts, int(layout.total_events), dtype=torch.float32)
    stats = {"attempts": attempts, "aligned": 0, "goal_events": 0, "zero_bias": 0}
    try:
        _sites, tokens, probabilities, _logits = c1.fragment_family_scores(
            fragment_model,
            energy_model,
            condition,
            source_latent,
            target_fragments,
            target_endpoints,
            config,
            device,
        )
    except Exception:
        stats["zero_bias"] = attempts
        return bias, stats
    if probabilities is None or int(probabilities.numel()) == 0:
        stats["zero_bias"] = attempts
        return bias, stats
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = torch.multinomial(
        probabilities.float().cpu(), attempts, replacement=True, generator=generator
    )
    energy_weight = float(config.energy_weight)
    for row, flat_index in enumerate(selected.tolist()):
        try:
            site_index, token_index = divmod(int(flat_index), len(tokens))
            product = graph.canonical_smiles(
                kernel.fragments.join_fragments(
                    _sites[site_index].core, tokens[token_index]
                )
            )
            events = goal_events_on_source(
                condition.source_smiles,
                product,
                condition.source,
                vocabulary,
                layout,
                max_atoms=int(layout.nodes),
                fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
                timeout=int(b41_prereg["mcs_timeout"]),
                min_common_fraction=float(b41_prereg["min_common_fraction"]),
            )
            if events:
                for event in events:
                    bias[row, layout.encode(event)] = energy_weight
                stats["aligned"] += 1
                stats["goal_events"] += len(events)
                continue
        except Exception:
            pass
        stats["zero_bias"] += 1
    return bias, stats


def goal_events_on_source(
    source_smiles: str,
    product_smiles: str,
    source_example,
    vocabulary,
    layout: b38.EventLayout,
    *,
    max_atoms: int,
    fingerprint_bits: int,
    timeout: int,
    min_common_fraction: float,
) -> list[b38.GraphEvent]:
    product = graph.canonical_smiles(product_smiles or "")
    if not product or product == source_smiles:
        return []
    try:
        aligned = base.align_pair(
            source_smiles,
            product,
            max_atoms=int(max_atoms),
            fingerprint_bits=int(fingerprint_bits),
            timeout=int(timeout),
            min_common_fraction=float(min_common_fraction),
        )
        if aligned is None:
            return []
        _aligned_source, target, _common = aligned
        source_batch = graph.collate([source_example])
        target_batch = graph.collate([target])
        node_actions, edge_actions = delta.delta_action_targets(
            source_batch, target_batch, vocabulary
        )
        return b38.target_event_set(node_actions[0], edge_actions[0], layout)
    except (ValueError, RuntimeError):
        return []


@torch.no_grad()
def sample_from_source_with_event_bias(
    model: b39.LatentCardinalityGraphJumpBridge,
    representation,
    vocabulary,
    support,
    support_tensors,
    source_example,
    condition_tokens: np.ndarray,
    preregistration,
    device: torch.device,
    seed: int,
    event_bias: torch.Tensor,
    energy_weight: float,
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
    bias = event_bias.to(device=device, dtype=torch.float32)
    goal_mask = bias.gt(0)
    goal_mask[:, 0] = False
    outputs: list[dict[str, object]] = []
    model.eval()
    for start in range(0, attempts, batch_size):
        count = min(batch_size, attempts - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        tokens = torch.from_numpy(
            np.repeat(condition_tokens[None, ...], count, axis=0)
        ).to(device)
        latent = particles[start : start + count]
        chunk_bias = bias[start : start + count]
        chunk_goal = goal_mask[start : start + count]
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
            dead_end = torch.zeros(count, dtype=torch.bool, device=device)
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
                legal, diagnostics, newly_dead = safe_viability_event_mask(
                    model.denoiser,
                    source,
                    node_actions,
                    edge_actions,
                    working,
                    support,
                    support_tensors,
                )
                dead_end |= newly_dead
                if bool(newly_dead.any()):
                    stopped = stopped | newly_dead
                masked_events += diagnostics["base_legal"] - diagnostics[
                    "constrained_legal"
                ]
                base_events += diagnostics["base_legal"]
                stop_masked_steps += diagnostics["stop_masked"].long()
                if bool(stopped.any()):
                    legal[stopped] = False
                    legal[stopped, 0] = True
                logits = logits + chunk_bias
                still_goal = legal & chunk_goal
                logits[:, 0] = logits[:, 0] - float(energy_weight) * still_goal.any(
                    dim=1
                ).float()
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
                    "stopped_by_model": bool(stopped[index].cpu()),
                    "affected_node_count": int(affected.sum()),
                    "affected_components": b37.component_count(
                        affected.numpy(), adjacency
                    ),
                    "outside_source_invariant": bool(
                        outside_nodes_exact and outside_edges_exact
                    ),
                    "goal_aligned": int(bool(chunk_goal[index].any().cpu())),
                    "goal_event_count": int(chunk_goal[index].sum().cpu()),
                    "dead_end_stop": int(bool(dead_end[index].cpu())),
                    **particle_metrics,
                }
            )
    if len(outputs) != attempts:
        raise RuntimeError(f"D1 expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def _batch_slice(source: dict, index: int) -> dict:
    size = int(source["atomic_number"].shape[0])
    return {
        key: (
            value[index : index + 1]
            if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == size
            else value
        )
        for key, value in source.items()
    }


def _dead_end_diagnostics(
    count: int, legal: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    ones = torch.ones(count, dtype=torch.long, device=device)
    return {
        "base_legal": ones,
        "constrained_legal": legal.sum(dim=1).long(),
        "stop_masked": ~legal[:, 0],
    }


def safe_viability_event_mask(
    field,
    source,
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    working: torch.Tensor,
    support,
    support_tensors,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """B40 raises if a particle has no legal event. Energy tilt can create that.

    Isolate the dead particle, force STOP, and keep the rest of the batch.
    """
    try:
        legal, diagnostics = b41.viability_event_mask(
            field,
            source,
            node_actions,
            edge_actions,
            working,
            support,
            support_tensors,
        )
        dead = torch.zeros(
            node_actions.shape[0], dtype=torch.bool, device=node_actions.device
        )
        return legal, diagnostics, dead
    except RuntimeError as exc:
        if "dead end" not in str(exc):
            raise
    count = int(node_actions.shape[0])
    n_events = int(field.layout.total_events)
    legal = torch.zeros(
        count, n_events, dtype=torch.bool, device=node_actions.device
    )
    dead = torch.zeros(count, dtype=torch.bool, device=node_actions.device)
    for index in range(count):
        try:
            one, _ = b41.viability_event_mask(
                field,
                _batch_slice(source, index),
                node_actions[index : index + 1],
                edge_actions[index : index + 1],
                working[index : index + 1],
                support,
                support_tensors,
            )
            legal[index] = one[0]
        except RuntimeError as exc:
            if "dead end" not in str(exc):
                raise
            legal[index, 0] = True
            dead[index] = True
    return legal, _dead_end_diagnostics(count, legal, node_actions.device), dead


if __name__ == "__main__":
    raise SystemExit(main())
