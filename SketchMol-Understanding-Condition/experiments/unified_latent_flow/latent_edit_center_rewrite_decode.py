#!/usr/bin/env python3
"""Decode B22 latents through a learned local graph-rewrite support.

This B23 pilot isolates the remaining validity failure without retraining the
B22 transport model.  At the first reverse-diffusion step, the denoiser's own
latent-conditioned node and edge logits select one edit center for 2-property
conditions and two centers for 3-property conditions.  Subsequent categorical
updates are confined to the source-graph neighbourhood of those centers plus
the fixed target-blind birth slots.  The locality support is therefore part of
the generative action grammar, not a molecule-level repair or a candidate
selector.

Generation receives no development target, property oracle, RDKit validity
result, or candidate feedback.  Exactly 20 raw action samples are materialized
and frozen before normal development evaluation opens those resources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

import source_relative_delta_diffusion as delta


base = delta.base
belief = delta.belief
full_graph = delta.full_graph
graph = delta.graph
hierarchical = delta.hierarchical
unified = delta.unified

PROTOCOL = "latent_edit_center_local_rewrite_decode_pilot_v23"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--delta-checkpoint", type=Path, required=True)
    parser.add_argument("--delta-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--birth-capacity", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.75)
    parser.add_argument("--latent-noise-scale", type=float, default=1.0)
    parser.add_argument("--rewrite-radius", type=int, default=1)
    parser.add_argument("--centers-per-extra-property", type=int, default=1)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--validation-selection-seed", type=int, default=2719)
    parser.add_argument("--validation-exclusion-seed", type=int, default=1742)
    parser.add_argument("--gate-validity-improvement", type=float, default=0.10)
    parser.add_argument("--gate-strict-retention", type=float, default=-0.05)
    parser.add_argument("--gate-3p-strict-any20", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=1759)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def checkpoint_vocabulary(payload: Mapping[str, object]) -> dict[str, object]:
    raw = payload["vocabulary"]
    node_states = np.asarray(raw["node_states"], dtype=np.int64)
    edge_states = np.asarray(raw["edge_states"], dtype=np.int64)
    return {
        "node_states": node_states,
        "edge_states": edge_states,
        "blank_node_id": 0,
        "blank_edge_id": 0,
        "sha256": str(raw["sha256"]),
    }


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[full_graph.ContinuousDiscreteGraphDiffusion, dict[str, object], dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(payload["model_config"])
    model = full_graph.ContinuousDiscreteGraphDiffusion(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(config["condition_dim"]),
        transport_dim=int(config["transport_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        max_atoms=int(config["max_atoms"]),
        property_count=int(config["property_count"]),
        node_state_count=int(config["node_action_count"]),
        edge_state_count=int(config["edge_action_count"]),
        message_layers=int(config["message_layers"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, checkpoint_vocabulary(payload), payload


def latent_edit_region(
    node_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    source_active: torch.Tensor,
    source_bond: torch.Tensor,
    *,
    property_count: int,
    centers_per_extra_property: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select latent-conditioned centers and expand them on the source graph."""
    node_nonkeep = torch.logsumexp(node_logits[..., 1:], dim=-1) - node_logits[..., 0]
    edge_nonkeep = torch.logsumexp(edge_logits[..., 1:], dim=-1) - edge_logits[..., 0]
    diagonal = torch.eye(
        edge_nonkeep.shape[1], device=edge_nonkeep.device, dtype=torch.bool
    ).unsqueeze(0)
    edge_nonkeep = edge_nonkeep.masked_fill(diagonal, -torch.inf)
    incident_score = edge_nonkeep.amax(dim=-1)
    scores = node_nonkeep + 0.5 * incident_score
    scores = scores.masked_fill(~source_active, -torch.inf)
    requested_centers = max(
        1, (int(property_count) - 1) * int(centers_per_extra_property)
    )
    center_count = min(int(source_active.sum(dim=1).min()), requested_centers)
    center_indices = scores.topk(max(1, center_count), dim=-1).indices
    centers = torch.zeros_like(source_active)
    centers.scatter_(1, center_indices, True)
    region = centers.clone()
    adjacency = source_bond.bool()
    for _ in range(max(0, int(radius))):
        neighbours = (adjacency & region[:, :, None]).any(dim=1)
        region |= neighbours
    region &= source_active
    return region, centers


def restrict_node_actions(
    legal: torch.Tensor, local_working: torch.Tensor
) -> torch.Tensor:
    keep = torch.zeros(legal.shape[-1], dtype=torch.bool, device=legal.device)
    keep[delta.NODE_KEEP] = True
    return legal & (local_working.unsqueeze(-1) | keep)


def restrict_edge_actions(
    legal: torch.Tensor, local_pairs: torch.Tensor
) -> torch.Tensor:
    keep = torch.zeros(legal.shape[-1], dtype=torch.bool, device=legal.device)
    keep[delta.EDGE_KEEP] = True
    symmetric = local_pairs | local_pairs.transpose(1, 2)
    return legal & (symmetric.unsqueeze(-1) | keep)


@torch.no_grad()
def sample_local_rewrite(
    model: full_graph.ContinuousDiscreteGraphDiffusion,
    representation,
    vocabulary: Mapping[str, object],
    source_example,
    condition_tokens: np.ndarray,
    *,
    property_count: int,
    attempts: int,
    batch_size: int,
    flow_steps: int,
    diffusion_steps: int,
    birth_capacity: int,
    latent_noise_scale: float,
    temperature: float,
    rewrite_radius: int,
    centers_per_extra_property: int,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int, float, int, int, int, int]]:
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[tuple[str | None, int, float, int, int, int, int]] = []
    model.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
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
            full_working = full_graph.working_node_mask(
                source["node_mask"], int(birth_capacity)
            )
            full_pairs = full_graph.upper_working_pairs(full_working)
            node_actions = torch.full_like(
                source["atomic_number"], model.denoiser.node_mask_id
            )
            node_actions = torch.where(
                full_working,
                node_actions,
                torch.full_like(node_actions, delta.NODE_KEEP),
            )
            edge_actions = torch.full_like(
                source["bond"], model.denoiser.edge_mask_id
            )
            symmetric_full_pairs = full_pairs | full_pairs.transpose(1, 2)
            edge_actions = torch.where(
                symmetric_full_pairs,
                edge_actions,
                torch.full_like(edge_actions, delta.EDGE_KEEP),
            )
            source_active = source["atomic_number"].gt(0)
            source_bond = source["bond"].gt(graph.BOND_NONE)
            node_action_count = model.denoiser.node_mask_id
            edge_action_count = model.denoiser.edge_mask_id
            local_working: torch.Tensor | None = None
            local_pairs: torch.Tensor | None = None
            centers: torch.Tensor | None = None
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
                    full_working if local_working is None else local_working,
                    time,
                    condition,
                    latent,
                )
                if local_working is None:
                    region, centers = latent_edit_region(
                        node_logits,
                        edge_logits,
                        source_active,
                        source_bond,
                        property_count=int(property_count),
                        centers_per_extra_property=int(centers_per_extra_property),
                        radius=int(rewrite_radius),
                    )
                    births = full_working & ~source_active
                    local_working = region | births
                    local_pairs = full_graph.upper_working_pairs(local_working)
                    node_actions = torch.where(
                        local_working,
                        node_actions,
                        torch.full_like(node_actions, delta.NODE_KEEP),
                    )
                    symmetric_local = local_pairs | local_pairs.transpose(1, 2)
                    edge_actions = torch.where(
                        symmetric_local,
                        edge_actions,
                        torch.full_like(edge_actions, delta.EDGE_KEEP),
                    )
                assert local_working is not None and local_pairs is not None
                node_legal = delta.legal_node_action_mask(
                    source_active, node_action_count
                )
                node_legal = restrict_node_actions(node_legal, local_working)
                node_logits = node_logits.float().masked_fill(~node_legal, -torch.inf)
                sampled_node, node_confidence = full_graph.sample_categorical(
                    node_logits, generator, temperature
                )
                sampled_node = torch.where(
                    local_working,
                    sampled_node,
                    torch.full_like(sampled_node, delta.NODE_KEEP),
                )
                predicted_active = delta.action_active_nodes(
                    source_active, sampled_node
                )
                edge_legal = delta.legal_edge_action_mask(
                    source_bond, predicted_active, edge_action_count
                )
                edge_legal = restrict_edge_actions(edge_legal, local_pairs)
                edge_logits = edge_logits.float().masked_fill(~edge_legal, -torch.inf)
                sampled_edge, edge_confidence = full_graph.sample_categorical(
                    edge_logits, generator, temperature
                )
                sampled_edge = torch.where(
                    local_pairs,
                    sampled_edge,
                    torch.full_like(sampled_edge, delta.EDGE_KEEP),
                )
                sampled_edge = sampled_edge + sampled_edge.transpose(1, 2)
                edge_confidence = torch.where(
                    local_pairs, edge_confidence, torch.zeros_like(edge_confidence)
                )
                edge_confidence = edge_confidence + edge_confidence.transpose(1, 2)
                fraction = (reverse_index - 1) / max(1, int(diffusion_steps))
                node_actions = full_graph.remask_low_confidence(
                    sampled_node,
                    node_confidence,
                    local_working,
                    model.denoiser.node_mask_id,
                    fraction,
                )
                edge_actions = full_graph.remask_low_confidence(
                    sampled_edge,
                    edge_confidence,
                    local_pairs,
                    model.denoiser.edge_mask_id,
                    fraction,
                )
                edge_actions = torch.where(
                    local_pairs,
                    edge_actions,
                    torch.full_like(edge_actions, delta.EDGE_KEEP),
                )
                edge_actions = edge_actions + edge_actions.transpose(1, 2)
            result = delta.apply_delta_actions(
                source, node_actions, edge_actions, vocabulary
            )

        prediction = {
            key: value.detach().cpu().numpy() for key, value in result.items()
        }
        upper = torch.triu(
            torch.ones(source["bond"].shape[1:], dtype=torch.bool), diagonal=1
        )
        latent_norms = latent.float().norm(dim=1).detach().cpu().tolist()
        region_sizes = (
            (local_working & source_active).sum(dim=1).detach().cpu().tolist()
        )
        center_counts = centers.sum(dim=1).detach().cpu().tolist()
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append(
                (
                    smiles,
                    int((prediction["atomic_number"][index] > 0).sum()),
                    float(latent_norms[index]),
                    int(node_actions[index].ne(delta.NODE_KEEP).sum()),
                    int(
                        (
                            edge_actions[index].ne(delta.EDGE_KEEP).detach().cpu()
                            & upper
                        ).sum()
                    ),
                    int(region_sizes[index]),
                    int(center_counts[index]),
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
        generated = sample_local_rewrite(
            model,
            representation,
            vocabulary,
            pair.source,
            pair.condition,
            property_count=int(pair.property_count),
            attempts=int(args.num_attempts),
            batch_size=int(args.sample_batch_size),
            flow_steps=int(args.flow_steps),
            diffusion_steps=int(args.diffusion_steps),
            birth_capacity=int(args.birth_capacity),
            latent_noise_scale=float(args.latent_noise_scale),
            temperature=float(args.sample_temperature),
            rewrite_radius=int(args.rewrite_radius),
            centers_per_extra_property=int(args.centers_per_extra_property),
            device=device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        source_copy_target = (
            graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        )
        for rank, item in enumerate(generated, start=1):
            smiles, atom_count, latent_norm, node_edits, edge_edits, region_size, centers = item
            canonical = graph.canonical_smiles(smiles or "")
            valid = bool(canonical)
            source_tanimoto = (
                graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            )
            target_tanimoto = (
                graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
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
                    "node_edit_count": int(node_edits),
                    "edge_edit_count": int(edge_edits),
                    "edit_region_size": int(region_size),
                    "edit_center_count": int(centers),
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
    for name in (
        "latent_norm",
        "node_edit_count",
        "edge_edit_count",
        "edit_region_size",
        "edit_center_count",
    ):
        metrics[f"mean_{name}"] = float(
            np.mean([float(row[name]) for row in candidate_rows])
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
        raise ValueError("B23 requires exactly 20 raw attempts per condition")
    if int(args.rewrite_radius) < 0:
        raise ValueError("rewrite-radius must be non-negative")
    base.seed_everything(int(args.seed))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation, representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    model, vocabulary, checkpoint = load_model(args.delta_checkpoint, device)
    baseline = json.loads(args.delta_summary.read_text(encoding="utf-8"))
    baseline_metrics = baseline["evaluation"]
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    validation_rows = base.read_rows(args.validation_csv)
    excluded_pairs, excluded_counts = base.build_pairs(
        validation_rows,
        max_atoms=int(representation_config["max_atoms"]),
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
        max_atoms=int(representation_config["max_atoms"]),
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
        raise ValueError("No development pairs survived fixed B22 filters")
    for pair in validation_pairs:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(args.condition_dim)
        )
    candidate_rows, metrics = evaluate(
        model, representation, vocabulary, validation_pairs, args, device
    )
    baseline_validity = float(baseline_metrics["validity"])
    baseline_strict = float(baseline_metrics["strict_any20"])
    three_property_strict = float(
        metrics["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity_delta": {
            "value": float(metrics["validity"]) - baseline_validity,
            "threshold": float(args.gate_validity_improvement),
        },
        "strict_delta": {
            "value": float(metrics["strict_any20"]) - baseline_strict,
            "threshold": float(args.gate_strict_retention),
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
    checkpoint_manifest = dict(checkpoint.get("manifest", {}))
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "heldout_role": "development_not_final_audit",
        "device": str(device),
        "representation_protocol": representation_summary.get("protocol"),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "delta_checkpoint_sha256": belief.file_sha256(args.delta_checkpoint),
        "delta_training_protocol": checkpoint.get("stage"),
        "delta_train_validation_source_overlap": checkpoint_manifest.get(
            "train_validation_source_overlap"
        ),
        "delta_train_validation_pair_overlap": checkpoint_manifest.get(
            "train_validation_pair_overlap"
        ),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "selected_validation_pairs": len(validation_pairs),
        "validation_filter_counts": validation_counts,
        "historical_validation_filter_counts": excluded_counts,
        "historical_validation_source_overlap": len(
            excluded_sources & {pair.source_smiles for pair in validation_pairs}
        ),
        "historical_validation_pair_overlap": len(
            excluded_pair_keys
            & {(pair.source_smiles, pair.target_smiles) for pair in validation_pairs}
        ),
        "latent_conditioned_edit_center": True,
        "property_cardinality_center_count": True,
        "rewrite_radius": int(args.rewrite_radius),
        "centers_per_extra_property": int(args.centers_per_extra_property),
        "target_blind_birth_capacity": int(args.birth_capacity),
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "generation_rdkit_validity_access": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "posthoc_molecule_repair": False,
        "valence_repair": False,
        "exact_raw_attempts_per_condition": 20,
    }
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
    summary = {
        "protocol": PROTOCOL,
        "manifest": manifest,
        "baseline": {
            "protocol": baseline.get("protocol"),
            "validity": baseline_validity,
            "strict_any20": baseline_strict,
            "three_property_strict_any20": baseline_metrics["by_property_count"]
            .get("3", {})
            .get("strict_any20", 0.0),
        },
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "train_two_step_residual_latent_local_rewrite"
            if not failures
            else "replace_independent_edge_actions_with_fragment_attachment_grammar"
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
