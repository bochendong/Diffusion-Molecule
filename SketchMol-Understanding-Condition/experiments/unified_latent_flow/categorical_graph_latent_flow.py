#!/usr/bin/env python3
"""Source-conditioned categorical graph-latent rectified-flow pilot.

The representation checkpoint is frozen.  A permutation-equivariant velocity
field learns source-to-target motion in its node and unordered-pair latent
slots.  Validation generation receives only the source graph and a sanitized
property-program vector; target molecules are used only after exactly 20 raw
decodes have been frozen for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
GRAPH_AE_PATH = SCRIPT_DIR / "train_graph_latent_autoencoder.py"
UNIFIED_GENERATOR_PATH = (
    PROJECT_DIR / "experiments" / "unified_smiles_generator" / "unified_smiles_generator.py"
)
PROTOCOL = "categorical_graph_latent_rectified_flow_pilot_v1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graph = load_module("categorical_graph_latent_ae", GRAPH_AE_PATH)
unified = load_module("categorical_graph_latent_unified", UNIFIED_GENERATOR_PATH)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--validation-limit", type=int, default=24)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--source-noise", type=float, default=0.08)
    parser.add_argument("--endpoint-weight", type=float, default=0.10)
    parser.add_argument("--flow-steps", type=int, default=6)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=5)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--gate-validity", type=float, default=0.90)
    parser.add_argument("--gate-source-tanimoto", type=float, default=0.40)
    parser.add_argument("--gate-target-improvement-rate", type=float, default=0.25)
    parser.add_argument("--gate-strict-any20", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1727)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(seed: int, *values: str) -> str:
    return hashlib.sha256((str(seed) + "\0" + "\0".join(values)).encode()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def parse_property_counts(value: str) -> set[int]:
    counts = {int(part.strip()) for part in str(value).split(",") if part.strip()}
    if not counts or min(counts) <= 0:
        raise ValueError("property-counts must contain positive integers")
    return counts


def task_specs(row: Mapping[str, str]) -> list[tuple[str, int]]:
    specs = list(unified.instruction_task_specs(row))
    if specs:
        return specs
    selected = unified.selected_properties(row)
    return [(prop, int(unified.property_direction(row, prop))) for prop in selected]


def task_key(row: Mapping[str, str]) -> str:
    specs = task_specs(row)
    return "+".join(f"{name}:{direction:+d}" for name, direction in specs) or "unknown"


def is_edit_pair(row: Mapping[str, str], allowed_counts: set[int]) -> bool:
    source = str(row.get("source_smiles", "") or "").strip()
    target = str(row.get("target_smiles", "") or "").strip()
    if not source or not target or source == target:
        return False
    specs = task_specs(row)
    return len(specs) in allowed_counts and all(direction != 0 for _, direction in specs)


def sanitized_condition_row(row: Mapping[str, str]) -> dict[str, str]:
    forbidden = {
        "target_smiles",
        "canonical_smiles",
        "generated_smiles",
        "reference_smiles",
        "ground_truth_smiles",
    }
    return {str(key): str(value) for key, value in row.items() if str(key) not in forbidden}


def condition_vector(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    safe = sanitized_condition_row(row)
    fallback = unified.fallback_condition_features(safe, int(condition_dim))
    program = unified.property_program_tokens(safe, int(condition_dim))
    values = np.concatenate([fallback, program], axis=0).mean(axis=0)
    return np.asarray(values, dtype=np.float32)


def place_graph_example(example, target_to_slot: Mapping[int, int]):
    max_atoms = len(example.node_mask)
    arrays = {
        "atomic_number": np.zeros(max_atoms, dtype=np.int64),
        "formal_charge": np.full(max_atoms, graph.CHARGE_OFFSET, dtype=np.int64),
        "chirality": np.zeros(max_atoms, dtype=np.int64),
        "aromatic": np.zeros(max_atoms, dtype=np.int64),
        "explicit_hs": np.zeros(max_atoms, dtype=np.int64),
        "no_implicit": np.zeros(max_atoms, dtype=np.int64),
        "bond": np.zeros((max_atoms, max_atoms), dtype=np.int64),
        "bond_stereo": np.zeros((max_atoms, max_atoms), dtype=np.int64),
        "node_mask": np.zeros(max_atoms, dtype=np.float32),
    }
    atom_count = int(example.node_mask.sum())
    for target_index in range(atom_count):
        slot = int(target_to_slot[target_index])
        for key in (
            "atomic_number",
            "formal_charge",
            "chirality",
            "aromatic",
            "explicit_hs",
            "no_implicit",
            "node_mask",
        ):
            arrays[key][slot] = getattr(example, key)[target_index]
    for left in range(atom_count):
        for right in range(atom_count):
            left_slot, right_slot = int(target_to_slot[left]), int(target_to_slot[right])
            arrays["bond"][left_slot, right_slot] = example.bond[left, right]
            arrays["bond_stereo"][left_slot, right_slot] = example.bond_stereo[left, right]
    return graph.GraphExample(
        example.smiles,
        arrays["atomic_number"],
        arrays["formal_charge"],
        arrays["chirality"],
        arrays["aromatic"],
        arrays["explicit_hs"],
        arrays["no_implicit"],
        arrays["bond"],
        arrays["bond_stereo"],
        arrays["node_mask"],
        example.fingerprint,
    )


def align_pair(
    source_smiles: str,
    target_smiles: str,
    *,
    max_atoms: int,
    fingerprint_bits: int,
    timeout: int,
    min_common_fraction: float,
):
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    source = graph.molecule_example(source_smiles, max_atoms, fingerprint_bits)
    target = graph.molecule_example(target_smiles, max_atoms, fingerprint_bits)
    if source is None or target is None:
        return None
    source_mol, target_mol = Chem.MolFromSmiles(source.smiles), Chem.MolFromSmiles(target.smiles)
    if source_mol is None or target_mol is None:
        return None
    result = rdFMCS.FindMCS(
        [source_mol, target_mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=max(1, int(timeout)),
    )
    query = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
    if query is None:
        return None
    source_match, target_match = source_mol.GetSubstructMatch(query), target_mol.GetSubstructMatch(query)
    common = min(len(source_match), len(target_match))
    denominator = max(1, min(source_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
    if common == 0 or common / denominator < float(min_common_fraction):
        return None
    target_to_slot = {int(target_index): int(source_index) for source_index, target_index in zip(source_match, target_match)}
    free_slots = [slot for slot in range(max_atoms) if slot not in set(target_to_slot.values())]
    for target_index in range(target_mol.GetNumAtoms()):
        if target_index not in target_to_slot:
            target_to_slot[target_index] = free_slots.pop(0)
    return source, place_graph_example(target, target_to_slot), common


@dataclass
class EditPair:
    row: dict[str, str]
    source_smiles: str
    target_smiles: str
    source: object
    target: object
    condition: np.ndarray
    property_count: int
    task: str
    common_atoms: int


def build_pairs(
    rows: Sequence[dict[str, str]],
    *,
    max_atoms: int,
    fingerprint_bits: int,
    condition_dim: int,
    allowed_counts: set[int],
    timeout: int,
    min_common_fraction: float,
    limit: int,
    seed: int,
    forbidden_sources: set[str] | None = None,
    forbidden_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[EditPair], dict[str, int]]:
    forbidden_sources = forbidden_sources or set()
    forbidden_pairs = forbidden_pairs or set()
    prepared: list[tuple[str, dict[str, str], str, str]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if not is_edit_pair(row, allowed_counts):
            continue
        source = graph.canonical_smiles(str(row.get("source_smiles", "") or ""))
        target = graph.canonical_smiles(str(row.get("target_smiles", "") or ""))
        if not source or not target or source == target:
            counts["invalid_or_identity"] += 1
            continue
        if source in forbidden_sources or (source, target) in forbidden_pairs:
            counts["forbidden_overlap"] += 1
            continue
        prepared.append((stable_hash(seed, task_key(row), source, target), dict(row), source, target))
    prepared.sort(key=lambda value: value[0])
    if int(limit) > 0:
        prepared = prepared[: int(limit)]
    pairs: list[EditPair] = []
    for _, row, source_smiles, target_smiles in prepared:
        aligned = align_pair(
            source_smiles,
            target_smiles,
            max_atoms=max_atoms,
            fingerprint_bits=fingerprint_bits,
            timeout=timeout,
            min_common_fraction=min_common_fraction,
        )
        if aligned is None:
            counts["alignment_rejected"] += 1
            continue
        source, target, common = aligned
        specs = task_specs(row)
        pairs.append(
            EditPair(
                row=row,
                source_smiles=source_smiles,
                target_smiles=target_smiles,
                source=source,
                target=target,
                condition=condition_vector(row, condition_dim),
                property_count=len(specs),
                task=task_key(row),
                common_atoms=int(common),
            )
        )
    counts["selected"] = len(pairs)
    return pairs, dict(counts)


def pair_collate(items: Sequence[EditPair]) -> dict[str, object]:
    return {
        "source": graph.collate([item.source for item in items]),
        "target": graph.collate([item.target for item in items]),
        "condition": torch.from_numpy(np.stack([item.condition for item in items])),
        "items": list(items),
    }


def move_graph_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = max(1, self.dim // 2)
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        angles = time[:, None] * frequencies[None, :]
        value = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return F.pad(value, (0, max(0, self.dim - value.shape[-1])))[:, : self.dim]


class EquivariantGraphVelocity(nn.Module):
    """Index-free node and symmetric unordered-pair velocity field."""

    def __init__(self, node_dim: int, edge_dim: int, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim)
        )
        self.time = nn.Sequential(
            TimeEmbedding(node_dim), nn.Linear(node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.edge_summary = nn.Linear(edge_dim, node_dim)
        self.source_mask = nn.Linear(1, node_dim)
        self.node_velocity = nn.Sequential(
            nn.LayerNorm(node_dim * 5),
            nn.Linear(node_dim * 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        pair_input = edge_dim * 2 + node_dim * 4
        self.edge_velocity = nn.Sequential(
            nn.LayerNorm(pair_input),
            nn.Linear(pair_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        nn.init.zeros_(self.node_velocity[-1].bias)
        nn.init.zeros_(self.edge_velocity[-1].bias)

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.condition(condition) + self.time(time)
        edge_mean = self.edge_summary(edge.mean(dim=2))
        global_mean = node.mean(dim=1, keepdim=True).expand_as(node)
        node_input = torch.cat(
            [
                node,
                source_node,
                edge_mean,
                global_mean,
                context[:, None, :] + self.source_mask(source_mask.unsqueeze(-1)),
            ],
            dim=-1,
        )
        node_velocity = self.node_velocity(node_input)
        left = node[:, :, None, :].expand(-1, -1, node.shape[1], -1)
        right = node[:, None, :, :].expand(-1, node.shape[1], -1, -1)
        pair_context = context[:, None, None, :].expand(-1, node.shape[1], node.shape[1], -1)
        edge_input = torch.cat(
            [edge, source_edge, left + right, torch.abs(left - right), pair_context, global_mean[:, :, None, :].expand(-1, -1, node.shape[1], -1)],
            dim=-1,
        )
        edge_velocity = self.edge_velocity(edge_input)
        edge_velocity = 0.5 * (edge_velocity + edge_velocity.transpose(1, 2))
        return node_velocity, edge_velocity


def load_representation(checkpoint_path: Path, summary_path: Path, device: torch.device):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("gate", {}).get("passed")):
        raise ValueError("Representation checkpoint is not authorized by a passed gate")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    model = graph.GraphLatentAutoencoder(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        layers=int(config["layers"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, summary


def weighted_latent_loss(
    predicted_node: torch.Tensor,
    predicted_edge: torch.Tensor,
    target_node: torch.Tensor,
    target_edge: torch.Tensor,
    union_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_weight = 0.05 + 0.95 * union_mask
    node_loss = ((predicted_node - target_node).square().mean(dim=-1) * node_weight).sum()
    node_loss = node_loss / node_weight.sum().clamp_min(1.0)
    pair_weight = 0.02 + 0.98 * (union_mask[:, :, None] * union_mask[:, None, :])
    edge_loss = ((predicted_edge - target_edge).square().mean(dim=-1) * pair_weight).sum()
    edge_loss = edge_loss / pair_weight.sum().clamp_min(1.0)
    return node_loss, edge_loss


def train_flow(
    flow: EquivariantGraphVelocity,
    representation,
    pairs: Sequence[EditPair],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
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
            collated = pair_collate(items)
            source = move_graph_batch(collated["source"], device)
            target = move_graph_batch(collated["target"], device)
            condition = collated["condition"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                source_node, source_edge = representation.encode(source)
                target_node, target_edge = representation.encode(target)
            noise_node = torch.randn_like(source_node) * float(args.source_noise)
            noise_edge = torch.randn_like(source_edge) * float(args.source_noise)
            noise_edge = 0.5 * (noise_edge + noise_edge.transpose(1, 2))
            node_zero, edge_zero = source_node + noise_node, source_edge + noise_edge
            time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
            node_t = (1.0 - time[:, None, None]) * node_zero + time[:, None, None] * target_node
            edge_t = (1.0 - time[:, None, None, None]) * edge_zero + time[:, None, None, None] * target_edge
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                predicted_node, predicted_edge = flow(
                    node_t,
                    edge_t,
                    source_node,
                    source_edge,
                    source["node_mask"],
                    time,
                    condition,
                )
                true_node, true_edge = target_node - node_zero, target_edge - edge_zero
                union_mask = torch.maximum(source["node_mask"], target["node_mask"])
                node_loss, edge_loss = weighted_latent_loss(
                    predicted_node, predicted_edge, true_node, true_edge, union_mask
                )
                endpoint_node = node_t + (1.0 - time[:, None, None]) * predicted_node
                endpoint_edge = edge_t + (1.0 - time[:, None, None, None]) * predicted_edge
                endpoint_logits = representation.decode(endpoint_node, endpoint_edge)
                endpoint_loss, _ = graph.reconstruction_loss(
                    endpoint_logits, target, endpoint_node, geometry_weight=0.0
                )
                loss = node_loss + edge_loss + float(args.endpoint_weight) * endpoint_loss
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), float(args.grad_clip))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["node_velocity_loss"] += float(node_loss.detach())
            totals["edge_velocity_loss"] += float(edge_loss.detach())
            totals["endpoint_reconstruction_loss"] += float(endpoint_loss.detach())
            batches += 1
        row = {"epoch": epoch, **{name: value / max(1, batches) for name, value in totals.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def sample_from_source(
    flow: EquivariantGraphVelocity,
    representation,
    source_example,
    condition: np.ndarray,
    *,
    attempts: int,
    batch_size: int,
    flow_steps: int,
    source_noise: float,
    device: torch.device,
    seed: int,
) -> list[str | None]:
    """Generate without accepting a target graph, target SMILES, or property oracle."""
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    outputs: list[str | None] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    flow.eval()
    for start in range(0, int(attempts), int(batch_size)):
        count = min(int(batch_size), int(attempts) - start)
        source = move_graph_batch(graph.collate([source_example] * count), device)
        conditions = torch.from_numpy(np.repeat(condition[None, :], count, axis=0)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            node = source_node + torch.randn(source_node.shape, generator=generator, device=device, dtype=source_node.dtype) * float(source_noise)
            edge_noise = torch.randn(source_edge.shape, generator=generator, device=device, dtype=source_edge.dtype) * float(source_noise)
            edge = source_edge + 0.5 * (edge_noise + edge_noise.transpose(1, 2))
            for step in range(int(flow_steps)):
                time = torch.full((count,), (step + 0.5) / max(1, int(flow_steps)), device=device)
                node_velocity, edge_velocity = flow(
                    node, edge, source_node, source_edge, source["node_mask"], time, conditions
                )
                node = node + node_velocity / max(1, int(flow_steps))
                edge = edge + edge_velocity / max(1, int(flow_steps))
            prediction = graph.predictions_from_logits(representation.decode(node, edge))
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outputs.append(smiles)
    if len(outputs) != int(attempts):
        raise RuntimeError(f"Expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else 0.0


def summarize_candidates(rows: Sequence[dict[str, object]], attempts: int) -> dict[str, object]:
    by_condition: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    condition_rows: list[dict[str, object]] = []
    for condition_id, values in sorted(by_condition.items()):
        if len(values) != int(attempts):
            raise ValueError(f"{condition_id}: expected {attempts} attempts, found {len(values)}")
        valid = [row for row in values if bool(row["valid"])]
        baseline = float(values[0]["source_copy_target_tanimoto"])
        best_target = max((float(row["target_tanimoto"]) for row in valid), default=0.0)
        condition_rows.append(
            {
                "condition_id": condition_id,
                "property_count": int(values[0]["property_count"]),
                "attempted": len(values),
                "valid": len(valid),
                "unique_valid": len({str(row["generated_smiles"]) for row in valid}),
                "strict_any": any(bool(row["strict_success"]) for row in values),
                "property_any": any(bool(row["property_success"]) for row in values),
                "best_target_tanimoto": best_target,
                "source_copy_target_tanimoto": baseline,
                "target_improved": best_target > baseline + 1e-8,
            }
        )
    valid_rows = [row for row in rows if bool(row["valid"])]
    summary: dict[str, object] = {
        "conditions": len(condition_rows),
        "candidate_rows": len(rows),
        "attempted_per_condition": int(attempts),
        "validity": len(valid_rows) / max(1, len(rows)),
        "mean_unique_valid": finite_mean([float(row["unique_valid"]) for row in condition_rows]),
        "mean_source_tanimoto": finite_mean([float(row["source_tanimoto"]) for row in valid_rows]),
        "mean_target_tanimoto": finite_mean([float(row["target_tanimoto"]) for row in valid_rows]),
        "mean_best_target_tanimoto": finite_mean([float(row["best_target_tanimoto"]) for row in condition_rows]),
        "source_copy_target_tanimoto": finite_mean([float(row["source_copy_target_tanimoto"]) for row in condition_rows]),
        "target_improvement_any20": sum(bool(row["target_improved"]) for row in condition_rows) / max(1, len(condition_rows)),
        "property_any20": sum(bool(row["property_any"]) for row in condition_rows) / max(1, len(condition_rows)),
        "strict_any20": sum(bool(row["strict_any"]) for row in condition_rows) / max(1, len(condition_rows)),
    }
    by_count = {}
    for count in sorted({int(row["property_count"]) for row in condition_rows}):
        selected = [row for row in condition_rows if int(row["property_count"]) == count]
        by_count[str(count)] = {
            "conditions": len(selected),
            "target_improvement_any20": sum(bool(row["target_improved"]) for row in selected) / len(selected),
            "property_any20": sum(bool(row["property_any"]) for row in selected) / len(selected),
            "strict_any20": sum(bool(row["strict_any"]) for row in selected) / len(selected),
        }
    summary["by_property_count"] = by_count
    return summary


def write_candidate_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    flow: EquivariantGraphVelocity,
    representation,
    pairs: Sequence[EditPair],
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
            source_noise=float(args.source_noise),
            device=device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        source_copy_target = graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        specs = task_specs(pair.row)
        condition_id = str(pair.row.get("condition_id", "") or pair.row.get("sample_id", "") or f"validation_{pair_index:04d}")
        for rank, smiles in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(smiles or "")
            valid = bool(canonical)
            source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            fraction, _, evaluated, all_success = unified.instruction_success_and_distance(
                pair.row, canonical or "", task_specs=specs
            )
            source_similarity_success = bool(source_tanimoto is not None and source_tanimoto >= 0.4)
            candidate_rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
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
    return candidate_rows, summarize_candidates(candidate_rows, int(args.num_attempts))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("The pilot contract requires exactly 20 raw attempts per condition")
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation, config, representation_summary = load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    allowed_counts = parse_property_counts(str(args.property_counts))
    validation_rows = read_rows(args.validation_csv)
    validation_pairs, validation_counts = build_pairs(
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
    train_rows = read_rows(args.train_csv)
    train_pairs, train_counts = build_pairs(
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

    flow = EquivariantGraphVelocity(
        node_dim=int(config["node_dim"]),
        edge_dim=int(config["edge_dim"]),
        condition_dim=int(args.condition_dim),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    history = train_flow(flow, representation, train_pairs, args, device)
    candidate_rows, metrics = evaluate(flow, representation, validation_pairs, args, device)
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": float(args.gate_validity)},
        "mean_source_tanimoto": {"value": metrics["mean_source_tanimoto"], "threshold": float(args.gate_source_tanimoto)},
        "target_improvement_any20": {"value": metrics["target_improvement_any20"], "threshold": float(args.gate_target_improvement_rate)},
        "strict_any20": {"value": metrics["strict_any20"], "threshold": float(args.gate_strict_any20)},
    }
    failures = [
        name
        for name, item in checks.items()
        if (item["value"] != item["threshold"] if name == "exact_attempts" else item["value"] < item["threshold"])
    ]
    train_sources = {pair.source_smiles for pair in train_pairs}
    train_pair_keys = {(pair.source_smiles, pair.target_smiles) for pair in train_pairs}
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "device": str(device),
        "representation_protocol": representation_summary.get("protocol"),
        "representation_gate_passed": bool(representation_summary.get("gate", {}).get("passed")),
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
        "mean_train_common_atoms": finite_mean([pair.common_atoms for pair in train_pairs]),
        "mean_validation_common_atoms": finite_mean([pair.common_atoms for pair in validation_pairs]),
        "property_counts": sorted(allowed_counts),
        "generation_target_access": False,
        "evaluation_target_access": True,
        "property_oracle_generation_access": False,
        "candidate_library": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "valence_projection_or_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "source_target_mcs_alignment_training_only": True,
        "permutation_equivariant_velocity": True,
    }
    checkpoint_path = args.output_dir / "categorical_graph_latent_flow.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": flow.state_dict(),
            "model_config": {
                "node_dim": int(config["node_dim"]),
                "edge_dim": int(config["edge_dim"]),
                "condition_dim": int(args.condition_dim),
                "hidden_dim": int(args.hidden_dim),
            },
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
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
        "next_stage": "conditioned_graph_flow_scale_signal" if not failures else "diagnose_graph_flow_signal",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
