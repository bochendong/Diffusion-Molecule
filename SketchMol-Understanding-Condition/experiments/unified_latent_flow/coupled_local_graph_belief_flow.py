#!/usr/bin/env python3
"""Coupled local categorical graph-belief flow with a native no-edit state.

Each local transition first decides whether an aligned atom block changes.  It
then predicts atom categories, re-encodes that provisional graph, and decides
whether incident bond blocks change before predicting their categories.  The
learned no-edit gates retain the source categories as part of the model's
probability space.  This is not post-hoc repair: no valence rule, candidate
selector, property oracle, or finalizer is called during generation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
BELIEF_PATH = SCRIPT_DIR / "categorical_graph_belief_flow.py"
PROTOCOL = "coupled_local_categorical_graph_belief_flow_pilot_v4"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


belief = load_module("coupled_local_graph_belief_base", BELIEF_PATH)
base = belief.base
graph = belief.graph
unified = belief.unified
NODE_FIELDS = belief.NODE_FIELDS
EDGE_FIELDS = belief.EDGE_FIELDS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--validation-limit", type=int, default=12)
    parser.add_argument("--property-counts", default="2")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--source-anchor-probability", type=float, default=0.35)
    parser.add_argument("--gate-loss-weight", type=float, default=0.50)
    parser.add_argument("--sampling-temperature", type=float, default=0.70)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--gate-validity", type=float, default=0.80)
    parser.add_argument("--gate-source-tanimoto", type=float, default=0.40)
    parser.add_argument("--gate-target-improvement-rate", type=float, default=0.25)
    parser.add_argument("--gate-strict-any20", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1733)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


class CoupledLocalEndpointField(nn.Module):
    """Endpoint categories plus learned atom-block and bond-block no-edit gates."""

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
            nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim)
        )
        self.gate_time = nn.Sequential(
            belief.TimeEmbedding(node_dim), nn.Linear(node_dim, node_dim), nn.SiLU()
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        context = self.gate_condition(condition) + self.gate_time(time)
        node_gate_input = torch.cat(
            [
                current_node,
                source_node,
                birth,
                context[:, None, :].expand_as(current_node),
            ],
            dim=-1,
        )
        node_change_logits = self.node_change(node_gate_input).squeeze(-1)
        left = current_node[:, :, None, :]
        right = current_node[:, None, :, :]
        edge_context = context[:, None, None, :].expand(
            -1, current_node.shape[1], current_node.shape[1], -1
        )
        edge_gate_input = torch.cat(
            [
                current_edge,
                source_edge,
                left + right,
                torch.abs(left - right),
                edge_context,
            ],
            dim=-1,
        )
        edge_change_logits = self.edge_change(edge_gate_input).squeeze(-1)
        edge_change_logits = 0.5 * (
            edge_change_logits + edge_change_logits.transpose(1, 2)
        )
        return endpoint_node, endpoint_edge, node_change_logits, edge_change_logits


def change_targets(
    source: Mapping[str, object], target: Mapping[str, object]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_atomic = source["atomic_number"]
    if not isinstance(source_atomic, torch.Tensor):
        raise TypeError("source graph tensors are required")
    node_changed = torch.zeros_like(source_atomic, dtype=torch.bool)
    for key in NODE_FIELDS:
        node_changed |= source[key].ne(target[key])
    # Empty slots are real no-edit examples; training them prevents spontaneous
    # births in exchangeable slots that never participate in the paired edit.
    node_eligible = torch.ones_like(node_changed, dtype=torch.bool)

    edge_changed = torch.zeros_like(source["bond"], dtype=torch.bool)
    for key in EDGE_FIELDS:
        edge_changed |= source[key].ne(target[key])
    nodes = source_atomic.shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=source_atomic.device, dtype=torch.bool), diagonal=1
    )
    union_node = source["node_mask"].bool() | target["node_mask"].bool()
    edge_eligible = (
        upper.unsqueeze(0) & union_node[:, :, None] & union_node[:, None, :]
    )
    return node_changed, node_eligible, edge_changed, edge_eligible


def masked_binary_loss(
    logits: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    selected_target = target[eligible].float()
    positives = selected_target.sum().clamp_min(1.0)
    negatives = (1.0 - selected_target).sum().clamp_min(1.0)
    # Square-root balancing keeps rare true edits learnable without erasing the
    # empirical dominance of the native no-edit state.
    positive_weight = torch.sqrt(negatives / positives).clamp(1.0, 10.0)
    return F.binary_cross_entropy_with_logits(
        logits[eligible], selected_target, pos_weight=positive_weight
    )


def train_flow(
    flow: CoupledLocalEndpointField,
    representation,
    pairs: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) * 7919)
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
            condition = collated["condition"].to(device)
            batch_size = len(items)
            time = torch.rand(batch_size, generator=generator, device=device)
            anchored = torch.rand(batch_size, generator=generator, device=device).lt(
                float(args.source_anchor_probability)
            )
            time = torch.where(anchored, torch.zeros_like(time), time)
            current = belief.mixed_categorical_state(source, target, time, generator)
            node_target, node_eligible, edge_target, edge_eligible = change_targets(
                source, target
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                current_node, current_edge = representation.encode(current)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                endpoint_node, endpoint_edge, node_gate, edge_gate = flow(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    current["node_mask"],
                    source["node_mask"],
                    belief.source_birth_ranks(source["node_mask"]),
                    time,
                    condition,
                )
                logits = representation.decode(endpoint_node, endpoint_edge)
                endpoint_loss, parts = graph.reconstruction_loss(
                    logits, target, endpoint_node, geometry_weight=0.0
                )
                node_gate_loss = masked_binary_loss(node_gate, node_target, node_eligible)
                edge_gate_loss = masked_binary_loss(edge_gate, edge_target, edge_eligible)
                loss = endpoint_loss + float(args.gate_loss_weight) * (
                    node_gate_loss + edge_gate_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()
            for name, value in parts.items():
                totals[name] += float(value)
            totals["total_loss"] += float(loss.detach())
            totals["node_gate_loss"] += float(node_gate_loss.detach())
            totals["edge_gate_loss"] += float(edge_gate_loss.detach())
            totals["node_change_rate"] += float(node_target[node_eligible].float().mean())
            totals["edge_change_rate"] += float(edge_target[edge_eligible].float().mean())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def clone_state(state: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def sample_bernoulli(
    logits: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    return torch.rand(
        probability.shape, generator=generator, device=probability.device
    ).lt(probability)


def sample_symmetric_bernoulli(
    logits: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    nodes = logits.shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=logits.device, dtype=torch.bool), diagonal=1
    )
    draw = sample_bernoulli(logits, generator) & upper.unsqueeze(0)
    return draw | draw.transpose(1, 2)


@torch.no_grad()
def sample_from_source(
    flow: CoupledLocalEndpointField,
    representation,
    source_example,
    condition: np.ndarray,
    *,
    attempts: int,
    batch_size: int,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int]]:
    """Generate coupled local blocks without a target graph or property oracle."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[tuple[str | None, int]] = []
    flow.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        condition_batch = torch.from_numpy(
            np.repeat(condition[None, :], count, axis=0)
        ).to(device)
        birth_rank = belief.source_birth_ranks(source["node_mask"])
        time_zero = torch.zeros(count, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            endpoint_node, endpoint_edge, node_gate, _ = flow(
                source_node,
                source_edge,
                source_node,
                source_edge,
                source["node_mask"],
                source["node_mask"],
                birth_rank,
                time_zero,
                condition_batch,
            )
            node_logits = representation.decode(endpoint_node, endpoint_edge)

        provisional = clone_state(source)
        sampled_nodes = {
            key: belief.categorical_sample(node_logits[key], temperature, generator)
            for key in NODE_FIELDS
        }
        node_eligible = source["node_mask"].bool() | sampled_nodes["atomic_number"].gt(0)
        node_edit = sample_bernoulli(node_gate, generator) & node_eligible
        for key in NODE_FIELDS:
            provisional[key] = torch.where(node_edit, sampled_nodes[key], source[key])
        provisional = belief.enforce_categorical_consistency(provisional)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            provisional_node, provisional_edge = representation.encode(provisional)
            endpoint_node, endpoint_edge, _, edge_gate = flow(
                provisional_node,
                provisional_edge,
                source_node,
                source_edge,
                provisional["node_mask"],
                source["node_mask"],
                birth_rank,
                torch.full((count,), 0.5, device=device),
                condition_batch,
            )
            edge_logits = representation.decode(endpoint_node, endpoint_edge)

        result = clone_state(provisional)
        nodes = source["node_mask"].shape[1]
        upper = torch.triu(
            torch.ones(nodes, nodes, device=device, dtype=torch.bool), diagonal=1
        )
        active_pair = provisional["node_mask"].bool()[:, :, None] & provisional[
            "node_mask"
        ].bool()[:, None, :]
        sampled_edges: dict[str, torch.Tensor] = {}
        for key in EDGE_FIELDS:
            sampled = belief.categorical_sample(edge_logits[key], temperature, generator)
            sampled = torch.where(upper.unsqueeze(0), sampled, torch.zeros_like(sampled))
            sampled_edges[key] = sampled + sampled.transpose(1, 2)
        edge_eligible = active_pair & (
            source["bond"].gt(graph.BOND_NONE)
            | sampled_edges["bond"].gt(graph.BOND_NONE)
        )
        edge_edit = sample_symmetric_bernoulli(edge_gate, generator) & edge_eligible
        for key in EDGE_FIELDS:
            result[key] = torch.where(edge_edit, sampled_edges[key], provisional[key])
        result = belief.enforce_categorical_consistency(result)
        prediction = {
            key: result[key].detach().cpu().numpy() for key in (*NODE_FIELDS, *EDGE_FIELDS)
        }
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append((smiles, int(result["node_mask"][index].sum().item())))
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    flow: CoupledLocalEndpointField,
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
        for rank, (smiles, predicted_atom_count) in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(smiles or "")
            valid = bool(canonical)
            source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            fraction, _, evaluated, all_success = unified.instruction_success_and_distance(
                pair.row, canonical or "", task_specs=specs
            )
            source_similarity_success = bool(
                source_tanimoto is not None and source_tanimoto >= 0.4
            )
            candidate_rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": int(predicted_atom_count),
                    "valid": valid,
                    "source_tanimoto": float(source_tanimoto or 0.0),
                    "target_tanimoto": float(target_tanimoto or 0.0),
                    "source_copy_target_tanimoto": float(source_copy_target),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(all_success),
                    "strict_success": bool(all_success and source_similarity_success),
                    "source_similarity_success": source_similarity_success,
                }
            )
    return candidate_rows, base.summarize_candidates(candidate_rows, int(args.num_attempts))


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
    validation_pairs, validation_counts = base.build_pairs(
        validation_rows,
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
    train_rows = base.read_rows(args.train_csv)
    train_pairs, train_counts = base.build_pairs(
        train_rows,
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
    flow = CoupledLocalEndpointField(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        hidden_dim=int(args.hidden_dim),
        max_atoms=int(config["max_atoms"]),
    ).to(device)
    history = train_flow(flow, representation, train_pairs, args, device)
    candidate_rows, metrics = evaluate(flow, representation, validation_pairs, args, device)
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
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
        "protocol": PROTOCOL,
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
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "native_no_edit_category": True,
        "node_then_incident_edge_coupling": True,
        "edge_distribution_conditioned_on_sampled_nodes": True,
        "joint_atom_birth_death_categories": True,
        "joint_bond_birth_death_categories": True,
        "continuous_latent_regression_loss": False,
        "separate_target_count_head": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "valence_projection_or_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "source_target_mcs_alignment_training_only": True,
    }
    checkpoint_path = args.output_dir / "coupled_local_graph_belief_flow.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": flow.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": int(args.condition_dim),
                "hidden_dim": int(args.hidden_dim),
                "max_atoms": int(config["max_atoms"]),
            },
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
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "extend_coupled_local_flow_to_3p"
            if not failures
            else "diagnose_coupled_local_validity_or_support"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
