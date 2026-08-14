#!/usr/bin/env python3
"""Native categorical graph-belief flow pilot for source-conditioned editing.

The stochastic path lives in molecule space: atom occupancy/type/attributes and
bond/order/stereo are categorical states.  The model predicts a distribution
over the endpoint categories from the current categorical graph, source graph,
property program, and time.  Sampling starts from the source and performs
joint categorical birth/death transitions.  A frozen, gate-passed graph
autoencoder supplies chemistry-aware features and category heads; there is no
continuous-latent regression objective, count head, candidate selector,
property-oracle guidance, finalizer, or valence repair.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
BASE_PATH = SCRIPT_DIR / "categorical_graph_latent_flow.py"
PROTOCOL = "native_categorical_graph_belief_flow_pilot_v3"
NODE_FIELDS = (
    "atomic_number",
    "formal_charge",
    "chirality",
    "aromatic",
    "explicit_hs",
    "no_implicit",
)
EDGE_FIELDS = ("bond", "bond_stereo")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("categorical_graph_belief_base", BASE_PATH)
graph = base.graph
unified = base.unified


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--validation-limit", type=int, default=16)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--source-anchor-probability", type=float, default=0.35)
    parser.add_argument("--flow-steps", type=int, default=6)
    parser.add_argument("--sampling-temperature", type=float, default=0.80)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--gate-validity", type=float, default=0.90)
    parser.add_argument("--gate-source-tanimoto", type=float, default=0.40)
    parser.add_argument("--gate-target-improvement-rate", type=float, default=0.25)
    parser.add_argument("--gate-strict-any20", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1731)
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
        value = torch.cat(
            [torch.sin(time[:, None] * frequency), torch.cos(time[:, None] * frequency)],
            dim=-1,
        )
        return F.pad(value, (0, max(0, self.dim - value.shape[-1])))[:, : self.dim]


class CategoricalEndpointField(nn.Module):
    """Predict endpoint latents whose frozen heads parameterize categories.

    Blank source slots receive only an ordered birth query.  This breaks the
    otherwise exact symmetry among empty slots without predicting a molecule
    size or supplying the target occupancy mask.
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
        self.max_atoms = int(max_atoms)
        self.birth_rank = nn.Embedding(self.max_atoms + 1, node_dim)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim)
        )
        self.time = nn.Sequential(
            TimeEmbedding(node_dim), nn.Linear(node_dim, node_dim), nn.SiLU()
        )
        self.current_mask = nn.Linear(1, node_dim)
        self.source_mask = nn.Linear(1, node_dim)
        self.edge_summary = nn.Linear(edge_dim, node_dim)
        self.node_delta = nn.Sequential(
            nn.LayerNorm(node_dim * 7),
            nn.Linear(node_dim * 7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        edge_input = edge_dim * 2 + node_dim * 5
        self.edge_delta = nn.Sequential(
            nn.LayerNorm(edge_input),
            nn.Linear(edge_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        nn.init.zeros_(self.node_delta[-1].bias)
        nn.init.zeros_(self.edge_delta[-1].bias)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total = (values * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
        return total / mask.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        birth = self.birth_rank(birth_rank)
        context = self.condition(condition) + self.time(time)
        current_global = self.masked_mean(current_node, current_mask).expand_as(current_node)
        source_global = self.masked_mean(source_node, source_mask).expand_as(source_node)
        edge_mean = self.edge_summary(current_edge.mean(dim=2))
        masks = self.current_mask(current_mask.unsqueeze(-1)) + self.source_mask(
            source_mask.unsqueeze(-1)
        )
        node_input = torch.cat(
            [
                current_node,
                source_node,
                birth,
                edge_mean,
                current_global,
                source_global,
                context[:, None, :].expand_as(current_node) + masks,
            ],
            dim=-1,
        )
        endpoint_node = current_node + self.node_delta(node_input)

        left = current_node[:, :, None, :]
        right = current_node[:, None, :, :]
        source_left = source_node[:, :, None, :]
        source_right = source_node[:, None, :, :]
        birth_pair = birth[:, :, None, :] + birth[:, None, :, :]
        pair_context = context[:, None, None, :].expand(
            -1, current_node.shape[1], current_node.shape[1], -1
        )
        edge_input = torch.cat(
            [
                current_edge,
                source_edge,
                left + right,
                torch.abs(left - right),
                source_left + source_right,
                birth_pair,
                pair_context,
            ],
            dim=-1,
        )
        endpoint_edge = current_edge + self.edge_delta(edge_input)
        endpoint_edge = 0.5 * (endpoint_edge + endpoint_edge.transpose(1, 2))
        return endpoint_node, endpoint_edge


def source_birth_ranks(source_mask: torch.Tensor) -> torch.Tensor:
    inactive = (~source_mask.bool()).long()
    ranks = torch.cumsum(inactive, dim=1) * inactive
    return ranks.clamp_max(source_mask.shape[1])


def enforce_categorical_consistency(state: dict[str, object]) -> dict[str, object]:
    """Make the joint state obey its own occupancy category, without chemistry repair."""
    atomic = state["atomic_number"]
    if not isinstance(atomic, torch.Tensor):
        raise TypeError("categorical state tensors are required")
    active = atomic.gt(0)
    state["node_mask"] = active.float()
    defaults = {
        "formal_charge": int(graph.CHARGE_OFFSET),
        "chirality": 0,
        "aromatic": 0,
        "explicit_hs": 0,
        "no_implicit": 0,
    }
    for key, default in defaults.items():
        tensor = state[key]
        state[key] = torch.where(active, tensor, torch.full_like(tensor, default))
    active_pair = active[:, :, None] & active[:, None, :]
    diagonal = torch.eye(active.shape[1], device=active.device, dtype=torch.bool).unsqueeze(0)
    active_pair = active_pair & ~diagonal
    for key in EDGE_FIELDS:
        tensor = state[key]
        state[key] = torch.where(active_pair, tensor, torch.zeros_like(tensor))
    return state


def mixed_categorical_state(
    source: Mapping[str, object],
    target: Mapping[str, object],
    time: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, object]:
    """Sample q_t with joint node-group and unordered-edge-group transitions."""
    atomic = source["atomic_number"]
    if not isinstance(atomic, torch.Tensor):
        raise TypeError("source graph tensor is required")
    batch, nodes = atomic.shape
    node_target = torch.rand(
        (batch, nodes), generator=generator, device=atomic.device
    ).lt(time[:, None])
    upper = torch.triu(
        torch.ones(nodes, nodes, device=atomic.device, dtype=torch.bool), diagonal=1
    )
    edge_draw = torch.rand(
        (batch, nodes, nodes), generator=generator, device=atomic.device
    ).lt(time[:, None, None]) & upper.unsqueeze(0)
    edge_target = edge_draw | edge_draw.transpose(1, 2)
    state: dict[str, object] = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in source.items()
    }
    for key in NODE_FIELDS:
        state[key] = torch.where(node_target, target[key], source[key])
    for key in EDGE_FIELDS:
        state[key] = torch.where(edge_target, target[key], source[key])
    return enforce_categorical_consistency(state)


def categorical_sample(
    logits: torch.Tensor, temperature: float, generator: torch.Generator
) -> torch.Tensor:
    classes = logits.shape[-1]
    probabilities = torch.softmax(logits.float() / max(1e-4, float(temperature)), dim=-1)
    sampled = torch.multinomial(
        probabilities.reshape(-1, classes), 1, replacement=True, generator=generator
    )
    return sampled.reshape(logits.shape[:-1])


def sample_endpoint_state(
    logits: Mapping[str, torch.Tensor],
    template: Mapping[str, object],
    temperature: float,
    generator: torch.Generator,
) -> dict[str, object]:
    state: dict[str, object] = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in template.items()
    }
    for key in NODE_FIELDS:
        state[key] = categorical_sample(logits[key], temperature, generator)
    nodes = logits["bond"].shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=logits["bond"].device, dtype=torch.bool), diagonal=1
    )
    for key in EDGE_FIELDS:
        sampled = categorical_sample(logits[key], temperature, generator)
        sampled = torch.where(upper.unsqueeze(0), sampled, torch.zeros_like(sampled))
        state[key] = sampled + sampled.transpose(1, 2)
    return enforce_categorical_consistency(state)


def transition_to_endpoint(
    current: dict[str, object],
    endpoint: Mapping[str, object],
    probability: float,
    generator: torch.Generator,
) -> dict[str, object]:
    atomic = current["atomic_number"]
    if not isinstance(atomic, torch.Tensor):
        raise TypeError("current graph tensor is required")
    batch, nodes = atomic.shape
    if float(probability) >= 1.0:
        node_update = torch.ones((batch, nodes), device=atomic.device, dtype=torch.bool)
    else:
        node_update = torch.rand(
            (batch, nodes), generator=generator, device=atomic.device
        ).lt(float(probability))
    upper = torch.triu(
        torch.ones(nodes, nodes, device=atomic.device, dtype=torch.bool), diagonal=1
    )
    if float(probability) >= 1.0:
        edge_update = upper.unsqueeze(0).expand(batch, -1, -1)
    else:
        draw = torch.rand(
            (batch, nodes, nodes), generator=generator, device=atomic.device
        ).lt(float(probability)) & upper.unsqueeze(0)
        edge_update = draw | draw.transpose(1, 2)
    for key in NODE_FIELDS:
        current[key] = torch.where(node_update, endpoint[key], current[key])
    for key in EDGE_FIELDS:
        current[key] = torch.where(edge_update, endpoint[key], current[key])
    return enforce_categorical_consistency(current)


def train_flow(
    flow: CategoricalEndpointField,
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
            current = mixed_categorical_state(source, target, time, generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                current_node, current_edge = representation.encode(current)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                endpoint_node, endpoint_edge = flow(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    current["node_mask"],
                    source["node_mask"],
                    source_birth_ranks(source["node_mask"]),
                    time,
                    condition,
                )
                logits = representation.decode(endpoint_node, endpoint_edge)
                loss, parts = graph.reconstruction_loss(
                    logits, target, endpoint_node, geometry_weight=0.0
                )
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()
            for name, value in parts.items():
                totals[name] += float(value)
            totals["source_anchor_fraction"] += float(anchored.float().mean())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def sample_from_source(
    flow: CategoricalEndpointField,
    representation,
    source_example,
    condition: np.ndarray,
    *,
    attempts: int,
    batch_size: int,
    flow_steps: int,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[tuple[str | None, int]]:
    """Generate raw categorical graphs without a target or property oracle."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[tuple[str | None, int]] = []
    flow.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = base.move_graph_batch(graph.collate([source_example] * count), device)
        current = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in source.items()
        }
        conditions = torch.from_numpy(np.repeat(condition[None, :], count, axis=0)).to(device)
        birth_rank = source_birth_ranks(source["node_mask"])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            for step in range(int(flow_steps)):
                current_node, current_edge = representation.encode(current)
                time = torch.full(
                    (count,), step / max(1, int(flow_steps)), device=device
                )
                endpoint_node, endpoint_edge = flow(
                    current_node,
                    current_edge,
                    source_node,
                    source_edge,
                    current["node_mask"],
                    source["node_mask"],
                    birth_rank,
                    time,
                    conditions,
                )
                logits = representation.decode(endpoint_node, endpoint_edge)
                endpoint = sample_endpoint_state(logits, current, temperature, generator)
                current = transition_to_endpoint(
                    current, endpoint, 1.0 / max(1, int(flow_steps) - step), generator
                )
        prediction = {
            key: current[key].detach().cpu().numpy() for key in (*NODE_FIELDS, *EDGE_FIELDS)
        }
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append((smiles, int(current["node_mask"][index].sum().item())))
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def evaluate(
    flow: CategoricalEndpointField,
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("The protocol requires exactly 20 raw attempts per condition")
    if int(args.flow_steps) < 1:
        raise ValueError("flow-steps must be positive")
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

    flow = CategoricalEndpointField(
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
        "representation_checkpoint_sha256": file_sha256(args.representation_checkpoint),
        "train_csv": str(args.train_csv),
        "train_csv_sha256": file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv),
        "validation_csv_sha256": file_sha256(args.validation_csv),
        "selected_train_pairs": len(train_pairs),
        "selected_validation_pairs": len(validation_pairs),
        "train_filter_counts": train_counts,
        "validation_filter_counts": validation_counts,
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_validation_pair_overlap": len(train_pair_keys & validation_pair_keys),
        "mean_train_common_atoms": base.finite_mean([pair.common_atoms for pair in train_pairs]),
        "mean_validation_common_atoms": base.finite_mean(
            [pair.common_atoms for pair in validation_pairs]
        ),
        "property_counts": sorted(allowed_counts),
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "native_discrete_state_path": True,
        "joint_atom_birth_death_categories": True,
        "joint_bond_birth_death_categories": True,
        "birth_rank_queries": True,
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
    checkpoint_path = args.output_dir / "categorical_graph_belief_flow.pt"
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
            "native_categorical_graph_flow_scale_signal"
            if not failures
            else "diagnose_native_categorical_signal"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
