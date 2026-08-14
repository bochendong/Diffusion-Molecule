#!/usr/bin/env python3
"""Gate a graph-native molecular latent representation before fitting a flow.

The representation keeps one continuous slot per atom and one continuous slot
per unordered atom pair. A permutation-equivariant message-passing encoder
maps categorical atom/bond tensors into those slots; a one-shot decoder maps
them back to atom and bond distributions. There is no string decoder, property
condition, oracle, selector, finalizer, or chemistry repair pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sketchmol_understanding_condition.chem import (
    canonical_smiles,
    morgan_fingerprint_bits,
    morgan_tanimoto,
    rdkit_version,
    scaffold_smiles,
)


PROTOCOL = "graph_latent_reconstruction_gate_v2_complete_schema"
BOND_NONE, BOND_SINGLE, BOND_DOUBLE, BOND_TRIPLE, BOND_AROMATIC = range(5)
BOND_CLASSES = 5
MAX_ATOMIC_NUMBER = 118
CHARGE_OFFSET = 5
CHARGE_CLASSES = 11
CHIRAL_CLASSES = 3  # none, R, S; invariant to atom ordering
EXPLICIT_H_CLASSES = 6
BOND_STEREO_CLASSES = 3  # none, E/trans, Z/cis
ATOM_MASK_ID = MAX_ATOMIC_NUMBER + 1
CHARGE_MASK_ID = CHARGE_CLASSES
CHIRAL_MASK_ID = CHIRAL_CLASSES
AROMATIC_MASK_ID = 2
EXPLICIT_H_MASK_ID = EXPLICIT_H_CLASSES
NO_IMPLICIT_MASK_ID = 2
BOND_MASK_ID = BOND_CLASSES
BOND_STEREO_MASK_ID = BOND_STEREO_CLASSES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=15000)
    parser.add_argument("--validation-limit", type=int, default=400)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--node-dim", type=int, default=192)
    parser.add_argument("--edge-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-noise", type=float, default=0.05)
    parser.add_argument("--stress-latent-noise", type=float, default=0.50)
    parser.add_argument("--category-mask-probability", type=float, default=0.03)
    parser.add_argument("--geometry-weight", type=float, default=0.05)
    parser.add_argument("--gate-validation-coverage", type=float, default=0.90)
    parser.add_argument("--gate-clean-validity", type=float, default=0.95)
    parser.add_argument("--gate-clean-connected", type=float, default=0.95)
    parser.add_argument("--gate-clean-tensor-exact", type=float, default=0.50)
    parser.add_argument("--gate-clean-topology-exact", type=float, default=0.50)
    parser.add_argument("--gate-clean-isomeric-exact", type=float, default=0.90)
    parser.add_argument("--gate-clean-tanimoto", type=float, default=0.85)
    parser.add_argument("--gate-clean-scaffold", type=float, default=0.80)
    parser.add_argument("--gate-noisy-validity", type=float, default=0.90)
    parser.add_argument("--gate-noisy-tanimoto", type=float, default=0.75)
    parser.add_argument("--gate-stress-validity", type=float, default=0.80)
    parser.add_argument("--gate-stress-tanimoto", type=float, default=0.70)
    parser.add_argument("--gate-masked-validity", type=float, default=0.70)
    parser.add_argument("--gate-masked-tanimoto", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=1723)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_subset(values: Sequence[str], limit: int, seed: int) -> list[str]:
    ordered = sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )
    return ordered if int(limit) <= 0 else ordered[: int(limit)]


def read_molecules(path: Path) -> tuple[set[str], dict[str, int]]:
    molecules: set[str] = set()
    counts = {"raw_values": 0, "invalid": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in ("source_smiles", "target_smiles"):
                raw = str(row.get(column, "") or "").strip()
                if not raw:
                    continue
                counts["raw_values"] += 1
                value = canonical_smiles(raw)
                if value:
                    molecules.add(value)
                else:
                    counts["invalid"] += 1
    counts["unique_canonical"] = len(molecules)
    return molecules, counts


def bond_class(bond) -> int:
    from rdkit import Chem

    mapping = {
        Chem.BondType.SINGLE: BOND_SINGLE,
        Chem.BondType.DOUBLE: BOND_DOUBLE,
        Chem.BondType.TRIPLE: BOND_TRIPLE,
        Chem.BondType.AROMATIC: BOND_AROMATIC,
    }
    if bond.GetBondType() not in mapping:
        raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
    return mapping[bond.GetBondType()]


@dataclass
class GraphExample:
    smiles: str
    atomic_number: np.ndarray
    formal_charge: np.ndarray
    chirality: np.ndarray
    aromatic: np.ndarray
    explicit_hs: np.ndarray
    no_implicit: np.ndarray
    bond: np.ndarray
    bond_stereo: np.ndarray
    node_mask: np.ndarray
    fingerprint: np.ndarray


def molecule_example(smiles: str, max_atoms: int, fingerprint_bits: int) -> GraphExample | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not (0 < molecule.GetNumAtoms() <= int(max_atoms)):
        return None
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    atomic_number = np.zeros(max_atoms, dtype=np.int64)
    formal_charge = np.full(max_atoms, CHARGE_OFFSET, dtype=np.int64)
    chirality = np.zeros(max_atoms, dtype=np.int64)
    aromatic = np.zeros(max_atoms, dtype=np.int64)
    explicit_hs = np.zeros(max_atoms, dtype=np.int64)
    no_implicit = np.zeros(max_atoms, dtype=np.int64)
    bond = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    bond_stereo = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    node_mask = np.zeros(max_atoms, dtype=np.float32)
    for index, atom in enumerate(molecule.GetAtoms()):
        number, charge = int(atom.GetAtomicNum()), int(atom.GetFormalCharge())
        chiral = {"R": 1, "S": 2}.get(atom.GetProp("_CIPCode"), 0) if atom.HasProp("_CIPCode") else 0
        if not (1 <= number <= MAX_ATOMIC_NUMBER):
            return None
        hydrogens = int(atom.GetNumExplicitHs())
        if not (-CHARGE_OFFSET <= charge <= CHARGE_OFFSET) or not (0 <= chiral < CHIRAL_CLASSES):
            return None
        if not (0 <= hydrogens < EXPLICIT_H_CLASSES):
            return None
        atomic_number[index] = number
        formal_charge[index] = charge + CHARGE_OFFSET
        chirality[index] = chiral
        aromatic[index] = int(atom.GetIsAromatic())
        explicit_hs[index] = hydrogens
        no_implicit[index] = int(atom.GetNoImplicit())
        node_mask[index] = 1.0
    try:
        for edge in molecule.GetBonds():
            left, right = int(edge.GetBeginAtomIdx()), int(edge.GetEndAtomIdx())
            value = bond_class(edge)
            bond[left, right] = bond[right, left] = value
            stereo_name = str(edge.GetStereo())
            stereo = {
                "STEREONONE": 0,
                "STEREOE": 1,
                "STEREOTRANS": 1,
                "STEREOZ": 2,
                "STEREOCIS": 2,
            }.get(stereo_name)
            if stereo is None:
                return None
            bond_stereo[left, right] = bond_stereo[right, left] = stereo
    except ValueError:
        return None
    fingerprint = morgan_fingerprint_bits(smiles, radius=2, n_bits=int(fingerprint_bits))
    if fingerprint is None:
        return None
    return GraphExample(
        smiles, atomic_number, formal_charge, chirality, aromatic,
        explicit_hs, no_implicit, bond, bond_stereo, node_mask,
        np.asarray(fingerprint, dtype=np.float32),
    )


def build_examples(
    molecules: Sequence[str], max_atoms: int, fingerprint_bits: int
) -> tuple[list[GraphExample], int]:
    examples: list[GraphExample] = []
    omitted = 0
    for smiles in molecules:
        example = molecule_example(smiles, max_atoms, fingerprint_bits)
        if example is None:
            omitted += 1
        else:
            examples.append(example)
    return examples, omitted


def permute_example(example: GraphExample, generator: random.Random) -> GraphExample:
    atom_count = int(example.node_mask.sum())
    active = list(range(atom_count))
    generator.shuffle(active)
    permutation = np.asarray(active + list(range(atom_count, len(example.node_mask))))
    return GraphExample(
        example.smiles,
        example.atomic_number[permutation],
        example.formal_charge[permutation],
        example.chirality[permutation],
        example.aromatic[permutation],
        example.explicit_hs[permutation],
        example.no_implicit[permutation],
        example.bond[np.ix_(permutation, permutation)],
        example.bond_stereo[np.ix_(permutation, permutation)],
        example.node_mask[permutation],
        example.fingerprint,
    )


def collate(items: Sequence[GraphExample]) -> dict[str, object]:
    return {
        "atomic_number": torch.from_numpy(np.stack([item.atomic_number for item in items])),
        "formal_charge": torch.from_numpy(np.stack([item.formal_charge for item in items])),
        "chirality": torch.from_numpy(np.stack([item.chirality for item in items])),
        "aromatic": torch.from_numpy(np.stack([item.aromatic for item in items])),
        "explicit_hs": torch.from_numpy(np.stack([item.explicit_hs for item in items])),
        "no_implicit": torch.from_numpy(np.stack([item.no_implicit for item in items])),
        "bond": torch.from_numpy(np.stack([item.bond for item in items])),
        "bond_stereo": torch.from_numpy(np.stack([item.bond_stereo for item in items])),
        "node_mask": torch.from_numpy(np.stack([item.node_mask for item in items])),
        "fingerprint": torch.from_numpy(np.stack([item.fingerprint for item in items])),
        "smiles": [item.smiles for item in items],
    }


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


class GraphMessageLayer(nn.Module):
    """Permutation-equivariant node update using typed bonded neighbours."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim * 2), nn.SiLU(), nn.Linear(node_dim * 2, node_dim)
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, bond: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        batch, nodes, _ = node.shape
        sender = node[:, None, :, :].expand(batch, nodes, nodes, -1)
        messages = self.message(torch.cat([sender, edge], dim=-1))
        neighbour_mask = bond.gt(BOND_NONE) & node_mask[:, :, None].bool() & node_mask[:, None, :].bool()
        messages = messages * neighbour_mask.unsqueeze(-1)
        local = messages.sum(dim=2) / neighbour_mask.sum(dim=2, keepdim=True).clamp_min(1).sqrt()
        global_mean = (node * node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
        global_mean = global_mean / node_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        updated = self.update(torch.cat([node, local, global_mean.expand(-1, nodes, -1)], dim=-1))
        return self.norm(node + updated) * node_mask.unsqueeze(-1)


class GraphLatentAutoencoder(nn.Module):
    """Variable-length atom slots plus symmetric atom-pair edge slots."""

    def __init__(self, node_dim: int, edge_dim: int, layers: int) -> None:
        super().__init__()
        self.atomic_embedding = nn.Embedding(MAX_ATOMIC_NUMBER + 2, node_dim)
        self.charge_embedding = nn.Embedding(CHARGE_CLASSES + 1, node_dim)
        self.chiral_embedding = nn.Embedding(CHIRAL_CLASSES + 1, node_dim)
        self.aromatic_embedding = nn.Embedding(3, node_dim)
        self.explicit_h_embedding = nn.Embedding(EXPLICIT_H_CLASSES + 1, node_dim)
        self.no_implicit_embedding = nn.Embedding(3, node_dim)
        self.bond_embedding = nn.Embedding(BOND_CLASSES + 1, edge_dim)
        self.bond_stereo_embedding = nn.Embedding(BOND_STEREO_CLASSES + 1, edge_dim)
        self.layers = nn.ModuleList([GraphMessageLayer(node_dim, edge_dim) for _ in range(int(layers))])
        self.node_latent = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), nn.LayerNorm(node_dim))
        self.edge_latent = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim), nn.SiLU(),
            nn.Linear(node_dim, edge_dim), nn.LayerNorm(edge_dim),
        )
        self.atomic_head = nn.Linear(node_dim, MAX_ATOMIC_NUMBER + 1)
        self.charge_head = nn.Linear(node_dim, CHARGE_CLASSES)
        self.chiral_head = nn.Linear(node_dim, CHIRAL_CLASSES)
        self.aromatic_head = nn.Linear(node_dim, 2)
        self.explicit_h_head = nn.Linear(node_dim, EXPLICIT_H_CLASSES)
        self.no_implicit_head = nn.Linear(node_dim, 2)
        self.bond_head = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, BOND_CLASSES)
        )
        self.bond_stereo_head = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, BOND_STEREO_CLASSES),
        )

    @staticmethod
    def pair_features(node: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        left = node[:, :, None, :].expand(-1, -1, node.shape[1], -1)
        right = node[:, None, :, :].expand(-1, node.shape[1], -1, -1)
        return torch.cat([left + right, torch.abs(left - right), edge], dim=-1)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_mask = batch["node_mask"]
        node = (
            self.atomic_embedding(batch["atomic_number"])
            + self.charge_embedding(batch["formal_charge"])
            + self.chiral_embedding(batch["chirality"])
            + self.aromatic_embedding(batch["aromatic"])
            + self.explicit_h_embedding(batch["explicit_hs"])
            + self.no_implicit_embedding(batch["no_implicit"])
        ) * node_mask.unsqueeze(-1)
        edge_embedding = (
            self.bond_embedding(batch["bond"])
            + self.bond_stereo_embedding(batch["bond_stereo"])
        )
        for layer in self.layers:
            node = layer(node, edge_embedding, batch["bond"], node_mask)
        node_latent = self.node_latent(node) * node_mask.unsqueeze(-1)
        edge_latent = self.edge_latent(self.pair_features(node_latent, edge_embedding))
        edge_latent = 0.5 * (edge_latent + edge_latent.transpose(1, 2))
        pair_mask = node_mask[:, :, None] * node_mask[:, None, :]
        return node_latent, edge_latent * pair_mask.unsqueeze(-1)

    def decode(self, node_latent: torch.Tensor, edge_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        bond_logits = self.bond_head(self.pair_features(node_latent, edge_latent))
        stereo_logits = self.bond_stereo_head(self.pair_features(node_latent, edge_latent))
        return {
            "atomic_number": self.atomic_head(node_latent),
            "formal_charge": self.charge_head(node_latent),
            "chirality": self.chiral_head(node_latent),
            "aromatic": self.aromatic_head(node_latent),
            "explicit_hs": self.explicit_h_head(node_latent),
            "no_implicit": self.no_implicit_head(node_latent),
            "bond": 0.5 * (bond_logits + bond_logits.transpose(1, 2)),
            "bond_stereo": 0.5 * (stereo_logits + stereo_logits.transpose(1, 2)),
        }

    def forward(
        self, batch: Mapping[str, torch.Tensor], latent_noise: float = 0.0
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        node_latent, edge_latent = self.encode(batch)
        if float(latent_noise) > 0:
            node_latent = node_latent + torch.randn_like(node_latent) * float(latent_noise)
            edge_latent = edge_latent + torch.randn_like(edge_latent) * float(latent_noise)
        return self.decode(node_latent, edge_latent), node_latent, edge_latent


def mask_graph_categories(
    batch: Mapping[str, torch.Tensor], probability: float
) -> dict[str, torch.Tensor | object]:
    """Mask true atom and bond categories before encoding for denoising."""
    masked: dict[str, torch.Tensor | object] = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    probability = max(0.0, min(1.0, float(probability)))
    if probability <= 0:
        return masked
    node_mask = batch["node_mask"].bool()
    selected_nodes = torch.rand_like(batch["node_mask"], dtype=torch.float32).lt(probability) & node_mask
    node_mask_ids = {
        "atomic_number": ATOM_MASK_ID,
        "formal_charge": CHARGE_MASK_ID,
        "chirality": CHIRAL_MASK_ID,
        "aromatic": AROMATIC_MASK_ID,
        "explicit_hs": EXPLICIT_H_MASK_ID,
        "no_implicit": NO_IMPLICIT_MASK_ID,
    }
    for key, mask_id in node_mask_ids.items():
        masked[key][selected_nodes] = int(mask_id)
    nodes = batch["node_mask"].shape[1]
    upper = torch.triu(
        torch.ones(nodes, nodes, device=batch["node_mask"].device, dtype=torch.bool), diagonal=1
    )
    bonded = upper.unsqueeze(0) & batch["bond"].gt(BOND_NONE)
    selected_edges = torch.rand_like(batch["bond"], dtype=torch.float32).lt(probability) & bonded
    selected_edges = selected_edges | selected_edges.transpose(1, 2)
    masked["bond"][selected_edges] = BOND_MASK_ID
    masked["bond_stereo"][selected_edges] = BOND_STEREO_MASK_ID
    return masked


def upper_pair_mask(node_mask: torch.Tensor) -> torch.Tensor:
    nodes = node_mask.shape[1]
    upper = torch.triu(torch.ones(nodes, nodes, device=node_mask.device, dtype=torch.bool), diagonal=1)
    return upper.unsqueeze(0) & node_mask[:, :, None].bool() & node_mask[:, None, :].bool()


def fingerprint_geometry_loss(
    node_latent: torch.Tensor, node_mask: torch.Tensor, fingerprint: torch.Tensor
) -> torch.Tensor:
    if node_latent.shape[0] < 2:
        return node_latent.sum() * 0.0
    pooled = (node_latent * node_mask.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = F.normalize(pooled, dim=-1)
    predicted = pooled @ pooled.transpose(0, 1)
    intersection = fingerprint @ fingerprint.transpose(0, 1)
    bit_counts = fingerprint.sum(dim=1)
    target = intersection / (bit_counts[:, None] + bit_counts[None, :] - intersection).clamp_min(1.0)
    off_diagonal = ~torch.eye(node_latent.shape[0], dtype=torch.bool, device=node_latent.device)
    return F.mse_loss(predicted[off_diagonal], target[off_diagonal])


def reconstruction_loss(
    logits: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    node_latent: torch.Tensor, geometry_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    node_mask = batch["node_mask"].bool()
    pair_mask = upper_pair_mask(batch["node_mask"])
    atom_weight = torch.ones(MAX_ATOMIC_NUMBER + 1, device=node_latent.device)
    atom_weight[0] = 0.25
    bond_weight = torch.tensor([0.20, 1.0, 1.0, 1.0, 1.0], device=node_latent.device)
    atom_loss = F.cross_entropy(
        logits["atomic_number"].reshape(-1, MAX_ATOMIC_NUMBER + 1),
        batch["atomic_number"].reshape(-1), weight=atom_weight,
    )
    charge_loss = F.cross_entropy(logits["formal_charge"][node_mask], batch["formal_charge"][node_mask])
    chiral_weight = torch.tensor([0.10, 1.0, 1.0], device=node_latent.device)
    explicit_h_weight = torch.tensor([0.10, 1.0, 1.0, 1.0, 1.0, 1.0], device=node_latent.device)
    no_implicit_weight = torch.tensor([0.10, 1.0], device=node_latent.device)
    stereo_weight = torch.tensor([0.02, 1.0, 1.0], device=node_latent.device)
    chiral_loss = F.cross_entropy(
        logits["chirality"][node_mask], batch["chirality"][node_mask], weight=chiral_weight
    )
    aromatic_loss = F.cross_entropy(logits["aromatic"][node_mask], batch["aromatic"][node_mask])
    explicit_h_loss = F.cross_entropy(
        logits["explicit_hs"][node_mask], batch["explicit_hs"][node_mask], weight=explicit_h_weight
    )
    no_implicit_loss = F.cross_entropy(
        logits["no_implicit"][node_mask], batch["no_implicit"][node_mask], weight=no_implicit_weight
    )
    bond_loss = F.cross_entropy(logits["bond"][pair_mask], batch["bond"][pair_mask], weight=bond_weight)
    stereo_loss = F.cross_entropy(
        logits["bond_stereo"][pair_mask],
        batch["bond_stereo"][pair_mask],
        weight=stereo_weight,
    )
    geometry = fingerprint_geometry_loss(node_latent, batch["node_mask"], batch["fingerprint"])
    total = (
        atom_loss
        + 0.5 * charge_loss
        + 0.2 * chiral_loss
        + 0.2 * aromatic_loss
        + 0.3 * explicit_h_loss
        + 0.2 * no_implicit_loss
        + 2.0 * bond_loss
        + 0.2 * stereo_loss
    )
    total = total + float(geometry_weight) * geometry
    parts = {
        "loss": float(total.detach()), "atom_loss": float(atom_loss.detach()),
        "charge_loss": float(charge_loss.detach()), "chiral_loss": float(chiral_loss.detach()),
        "aromatic_loss": float(aromatic_loss.detach()), "bond_loss": float(bond_loss.detach()),
        "explicit_h_loss": float(explicit_h_loss.detach()),
        "no_implicit_loss": float(no_implicit_loss.detach()),
        "stereo_loss": float(stereo_loss.detach()),
        "geometry_loss": float(geometry.detach()),
    }
    return total, parts


def train_model(
    model: GraphLatentAutoencoder, dataset: Sequence[GraphExample],
    args: argparse.Namespace, device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(dataset)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        permutation_rng = random.Random(int(args.seed) * 1000 + epoch)
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            examples = [
                permute_example(dataset[index], permutation_rng)
                for index in order[start : start + int(args.batch_size)]
            ]
            batch = move_batch(collate(examples), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                masked_input = mask_graph_categories(batch, float(args.category_mask_probability))
                logits, node_latent, _ = model(masked_input, latent_noise=float(args.latent_noise))
                loss, parts = reconstruction_loss(logits, batch, node_latent, float(args.geometry_weight))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            for name, value in parts.items():
                totals[name] += value
            batches += 1
        row = {"epoch": epoch, **{key: value / max(1, batches) for key, value in totals.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def predictions_from_logits(logits: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.argmax(dim=-1).detach().cpu().numpy() for key, value in logits.items()}


def graph_to_smiles(prediction: Mapping[str, np.ndarray], index: int) -> tuple[str | None, bool]:
    """Decode raw argmax classes; no valence projection or repair is applied."""
    from rdkit import Chem

    atom_values = prediction["atomic_number"][index]
    active = [position for position, value in enumerate(atom_values) if int(value) > 0]
    if not active:
        return None, False
    rw_molecule = Chem.RWMol()
    position_to_atom: dict[int, int] = {}
    desired_cip: dict[int, str] = {}
    desired_bond_stereo: list[tuple[int, int, int]] = []
    try:
        for position in active:
            atom = Chem.Atom(int(atom_values[position]))
            atom.SetFormalCharge(int(prediction["formal_charge"][index, position]) - CHARGE_OFFSET)
            chiral_value = int(prediction["chirality"][index, position])
            if chiral_value not in (0, 1, 2):
                raise ValueError(f"Unsupported predicted CIP class: {chiral_value}")
            if chiral_value:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                desired_cip[position] = "R" if chiral_value == 1 else "S"
            atom.SetIsAromatic(bool(prediction["aromatic"][index, position]))
            atom.SetNumExplicitHs(int(prediction["explicit_hs"][index, position]))
            atom.SetNoImplicit(bool(prediction["no_implicit"][index, position]))
            position_to_atom[position] = int(rw_molecule.AddAtom(atom))
        bond_types = {
            BOND_SINGLE: Chem.BondType.SINGLE, BOND_DOUBLE: Chem.BondType.DOUBLE,
            BOND_TRIPLE: Chem.BondType.TRIPLE, BOND_AROMATIC: Chem.BondType.AROMATIC,
        }
        for offset, left in enumerate(active):
            for right in active[offset + 1 :]:
                value = int(prediction["bond"][index, left, right])
                if value == BOND_NONE:
                    continue
                rw_molecule.AddBond(position_to_atom[left], position_to_atom[right], bond_types[value])
                if value == BOND_AROMATIC:
                    rw_molecule.GetAtomWithIdx(position_to_atom[left]).SetIsAromatic(True)
                    rw_molecule.GetAtomWithIdx(position_to_atom[right]).SetIsAromatic(True)
        for offset, left in enumerate(active):
            for right in active[offset + 1 :]:
                stereo_value = int(prediction["bond_stereo"][index, left, right])
                if stereo_value == 0 or int(prediction["bond"][index, left, right]) != BOND_DOUBLE:
                    continue
                desired_bond_stereo.append((left, right, stereo_value))
        molecule = rw_molecule.GetMol()
        Chem.SanitizeMol(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        for position, desired in desired_cip.items():
            atom = molecule.GetAtomWithIdx(position_to_atom[position])
            current = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
            if current and current != desired:
                atom.InvertChirality()
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        for left, right, stereo_value in desired_bond_stereo:
            left_index, right_index = position_to_atom[left], position_to_atom[right]
            decoded_bond = molecule.GetBondBetweenAtoms(left_index, right_index)
            left_neighbours = [
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(left_index).GetNeighbors()
                if atom.GetIdx() != right_index
            ]
            right_neighbours = [
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(right_index).GetNeighbors()
                if atom.GetIdx() != left_index
            ]
            if not left_neighbours or not right_neighbours:
                raise ValueError("Stereo bond lacks substituent atoms")

            def cip_rank(atom_index: int) -> int:
                atom = molecule.GetAtomWithIdx(atom_index)
                return int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else atom_index

            decoded_bond.SetStereoAtoms(
                max(left_neighbours, key=cip_rank), max(right_neighbours, key=cip_rank)
            )
            decoded_bond.SetStereo(
                Chem.BondStereo.STEREOE if stereo_value == 1 else Chem.BondStereo.STEREOZ
            )
        Chem.AssignStereochemistry(molecule, cleanIt=False, force=True)
        return Chem.MolToSmiles(molecule, canonical=True), len(Chem.GetMolFrags(molecule)) == 1
    except Exception:
        return None, False


def topology_smiles(smiles: str | None) -> str | None:
    if not smiles:
        return None
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def batch_tensor_metrics(
    prediction: Mapping[str, np.ndarray], batch: Mapping[str, object]
) -> list[dict[str, float]]:
    target = {
        key: batch[key].detach().cpu().numpy()
        for key in (
            "atomic_number", "formal_charge", "chirality", "aromatic",
            "explicit_hs", "no_implicit", "bond", "bond_stereo", "node_mask",
        )
    }
    rows: list[dict[str, float]] = []
    for index in range(len(batch["smiles"])):
        mask = target["node_mask"][index].astype(bool)
        pair = np.triu(np.outer(mask, mask), k=1).astype(bool)
        atom_exact = bool(np.array_equal(prediction["atomic_number"][index], target["atomic_number"][index]))
        attributes_exact = all(
            np.array_equal(prediction[key][index][mask], target[key][index][mask])
            for key in ("formal_charge", "chirality", "aromatic", "explicit_hs", "no_implicit")
        )
        bond_exact = bool(np.array_equal(prediction["bond"][index][pair], target["bond"][index][pair]))
        stereo_exact = bool(
            np.array_equal(prediction["bond_stereo"][index][pair], target["bond_stereo"][index][pair])
        )
        edge_exact = bond_exact and stereo_exact
        predicted_count = int((prediction["atomic_number"][index] > 0).sum())
        rows.append({
            "node_count_exact": float(predicted_count == int(mask.sum())),
            "atom_tensor_exact": float(atom_exact), "attribute_tensor_exact": float(attributes_exact),
            "bond_tensor_exact": float(bond_exact), "stereo_tensor_exact": float(stereo_exact),
            "edge_tensor_exact": float(edge_exact),
            "graph_tensor_exact": float(atom_exact and attributes_exact and edge_exact),
        })
    return rows


def summarize_reconstructions(rows: Sequence[Mapping[str, object]], prefix: str) -> dict[str, float]:
    count = len(rows)
    valid_rows = [row for row in rows if row[f"{prefix}_smiles"]]
    similarities = [float(row[f"{prefix}_tanimoto"]) for row in rows]
    scaffold_rows = [row for row in rows if row[f"{prefix}_scaffold_eligible"]]
    return {
        "count": count, "valid_count": len(valid_rows), "validity": len(valid_rows) / max(1, count),
        "connected_rate": sum(bool(row[f"{prefix}_connected"]) for row in rows) / max(1, count),
        "unique_valid": len({str(row[f"{prefix}_smiles"]) for row in valid_rows}),
        "topology_exact": sum(bool(row[f"{prefix}_topology_exact"]) for row in rows) / max(1, count),
        "isomeric_exact": sum(bool(row[f"{prefix}_isomeric_exact"]) for row in rows) / max(1, count),
        "mean_tanimoto": float(np.mean(similarities)) if similarities else 0.0,
        "median_tanimoto": float(np.median(similarities)) if similarities else 0.0,
        "scaffold_eligible_count": len(scaffold_rows),
        "scaffold_match": sum(bool(row[f"{prefix}_scaffold_match"]) for row in scaffold_rows)
        / max(1, len(scaffold_rows)),
        **{
            metric: sum(float(row[f"{prefix}_{metric}"]) for row in rows) / max(1, count)
            for metric in (
                "node_count_exact", "atom_tensor_exact", "attribute_tensor_exact",
                "bond_tensor_exact", "stereo_tensor_exact", "edge_tensor_exact", "graph_tensor_exact",
            )
        },
    }


@torch.no_grad()
def evaluate(
    model: GraphLatentAutoencoder, dataset: Sequence[GraphExample],
    args: argparse.Namespace, device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    pooled_latents: list[np.ndarray] = []
    fingerprints: list[np.ndarray] = []
    for start in range(0, len(dataset), int(args.eval_batch_size)):
        batch = move_batch(collate(dataset[start : start + int(args.eval_batch_size)]), device)
        node_latent, edge_latent = model.encode(batch)
        clean_prediction = predictions_from_logits(model.decode(node_latent, edge_latent))
        noisy_prediction = predictions_from_logits(model.decode(
            node_latent + torch.randn_like(node_latent) * float(args.latent_noise),
            edge_latent + torch.randn_like(edge_latent) * float(args.latent_noise),
        ))
        stress_prediction = predictions_from_logits(model.decode(
            node_latent + torch.randn_like(node_latent) * float(args.stress_latent_noise),
            edge_latent + torch.randn_like(edge_latent) * float(args.stress_latent_noise),
        ))
        masked_batch = mask_graph_categories(batch, float(args.category_mask_probability))
        masked_node_latent, masked_edge_latent = model.encode(masked_batch)
        masked_prediction = predictions_from_logits(model.decode(masked_node_latent, masked_edge_latent))
        clean_tensor = batch_tensor_metrics(clean_prediction, batch)
        noisy_tensor = batch_tensor_metrics(noisy_prediction, batch)
        stress_tensor = batch_tensor_metrics(stress_prediction, batch)
        masked_tensor = batch_tensor_metrics(masked_prediction, batch)
        pooled = (node_latent * batch["node_mask"].unsqueeze(-1)).sum(dim=1)
        pooled = pooled / batch["node_mask"].sum(dim=1, keepdim=True).clamp_min(1)
        pooled_latents.extend(pooled.detach().float().cpu().numpy())
        fingerprints.extend(batch["fingerprint"].detach().cpu().numpy())
        for index, reference in enumerate(batch["smiles"]):
            row: dict[str, object] = {"reference_smiles": reference}
            reference_topology = topology_smiles(reference)
            reference_scaffold = scaffold_smiles(reference)
            for prefix, prediction, tensor_metrics in (
                ("clean", clean_prediction, clean_tensor[index]),
                ("noisy", noisy_prediction, noisy_tensor[index]),
                ("stress", stress_prediction, stress_tensor[index]),
                ("masked", masked_prediction, masked_tensor[index]),
            ):
                decoded, connected = graph_to_smiles(prediction, index)
                decoded_topology = topology_smiles(decoded)
                decoded_scaffold = scaffold_smiles(decoded) if decoded else None
                similarity = morgan_tanimoto(reference, decoded) if decoded else None
                row.update({
                    f"{prefix}_smiles": decoded or "", f"{prefix}_connected": connected,
                    f"{prefix}_topology_exact": bool(decoded_topology and decoded_topology == reference_topology),
                    f"{prefix}_isomeric_exact": bool(decoded and decoded == reference),
                    f"{prefix}_tanimoto": float(similarity or 0.0),
                    f"{prefix}_scaffold_eligible": bool(reference_scaffold),
                    f"{prefix}_scaffold_match": bool(
                        reference_scaffold
                        and decoded_scaffold
                        and reference_scaffold == decoded_scaffold
                    ),
                    **{f"{prefix}_{key}": value for key, value in tensor_metrics.items()},
                })
            rows.append(row)
    latent_array, fingerprint_array = np.asarray(pooled_latents), np.asarray(fingerprints)
    geometry_spearman = 0.0
    if len(latent_array) >= 3:
        from scipy.stats import spearmanr

        latent_norm = latent_array / np.maximum(np.linalg.norm(latent_array, axis=1, keepdims=True), 1e-8)
        latent_similarity = latent_norm @ latent_norm.T
        intersection = fingerprint_array @ fingerprint_array.T
        counts = fingerprint_array.sum(axis=1)
        fingerprint_similarity = intersection / np.maximum(counts[:, None] + counts[None, :] - intersection, 1.0)
        upper = np.triu_indices(len(latent_array), k=1)
        geometry_spearman = float(spearmanr(latent_similarity[upper], fingerprint_similarity[upper]).statistic)
        if not math.isfinite(geometry_spearman):
            geometry_spearman = 0.0
    return rows, {
        "clean": summarize_reconstructions(rows, "clean"),
        "noisy": summarize_reconstructions(rows, "noisy"),
        "stress": summarize_reconstructions(rows, "stress"),
        "masked": summarize_reconstructions(rows, "masked"),
        "latent_fingerprint_geometry_spearman": geometry_spearman,
        "raw_argmax_decoder": True, "valence_projection_or_repair": False,
    }


def build_gate(
    metrics: Mapping[str, object], validation_coverage: float, args: argparse.Namespace
) -> dict[str, object]:
    clean, noisy, stress, masked = (
        metrics["clean"], metrics["noisy"], metrics["stress"], metrics["masked"]
    )
    checks = {
        "validation_coverage": {"value": validation_coverage, "threshold": args.gate_validation_coverage},
        "clean_validity": {"value": clean["validity"], "threshold": args.gate_clean_validity},
        "clean_connected": {"value": clean["connected_rate"], "threshold": args.gate_clean_connected},
        "clean_graph_tensor_exact": {"value": clean["graph_tensor_exact"], "threshold": args.gate_clean_tensor_exact},
        "clean_topology_exact": {"value": clean["topology_exact"], "threshold": args.gate_clean_topology_exact},
        "clean_isomeric_exact": {"value": clean["isomeric_exact"], "threshold": args.gate_clean_isomeric_exact},
        "clean_mean_tanimoto": {"value": clean["mean_tanimoto"], "threshold": args.gate_clean_tanimoto},
        "clean_scaffold_match": {"value": clean["scaffold_match"], "threshold": args.gate_clean_scaffold},
        "noisy_validity": {"value": noisy["validity"], "threshold": args.gate_noisy_validity},
        "noisy_mean_tanimoto": {"value": noisy["mean_tanimoto"], "threshold": args.gate_noisy_tanimoto},
        "stress_validity": {"value": stress["validity"], "threshold": args.gate_stress_validity},
        "stress_mean_tanimoto": {"value": stress["mean_tanimoto"], "threshold": args.gate_stress_tanimoto},
        "masked_validity": {"value": masked["validity"], "threshold": args.gate_masked_validity},
        "masked_mean_tanimoto": {"value": masked["mean_tanimoto"], "threshold": args.gate_masked_tanimoto},
    }
    failures = [name for name, check in checks.items() if float(check["value"]) < float(check["threshold"])]
    return {"passed": not failures, "checks": checks, "failures": failures}


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["reference_smiles"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_molecules, train_counts = read_molecules(args.train_csv)
    validation_molecules, validation_counts = read_molecules(args.validation_csv)
    overlap = train_molecules & validation_molecules
    train_molecules -= validation_molecules
    selected_train = stable_subset(sorted(train_molecules), int(args.train_limit), int(args.seed))
    selected_validation = stable_subset(sorted(validation_molecules), int(args.validation_limit), int(args.seed) + 1)
    train_dataset, train_omitted = build_examples(selected_train, int(args.max_atoms), int(args.fingerprint_bits))
    validation_dataset, validation_omitted = build_examples(
        selected_validation, int(args.max_atoms), int(args.fingerprint_bits)
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Empty graph latent train or validation dataset.")
    validation_coverage = len(validation_dataset) / max(1, len(selected_validation))
    manifest = {
        "protocol": PROTOCOL,
        "paper_lineage": {
            "EDM-SyCo_ICLR_2025": "variable-length latent points and one-shot node/edge decoding",
            "DeFoG_ICML_2025": "permutation-symmetric categorical graph state",
            "GrIDDD_NeurIPS_2025": "future size-adaptive node insertion/deletion",
            "GraphBSI_ICLR_2026": "future flow over categorical graph beliefs",
        },
        "seed": int(args.seed), "device": str(device), "rdkit_version": rdkit_version(),
        "representation_stage_only": True, "variable_length_atom_slots": True,
        "explicit_bond_latent_slots": True, "permutation_equivariant_encoder": True,
        "complete_chemical_state_schema": {
            "atom": ["atomic_number", "formal_charge", "R_S", "aromatic", "explicit_hs", "no_implicit"],
            "bond": ["bond_order", "E_Z"],
            "stereo_labels_are_permutation_invariant": True,
        },
        "one_shot_graph_decoder": True, "raw_argmax_decoder": True,
        "valence_projection_or_repair": False, "condition_access": False,
        "property_oracle_access": False, "candidate_library": False,
        "selector": False, "finalizer": False, "benchmark_generation_target_access": False,
        "evaluation_input_graph_access": True,
        "train_csv": str(args.train_csv), "train_csv_sha256": file_sha256(args.train_csv),
        "validation_csv": str(args.validation_csv), "validation_csv_sha256": file_sha256(args.validation_csv),
        "raw_train_counts": train_counts, "raw_validation_counts": validation_counts,
        "raw_canonical_overlap_removed_from_train": len(overlap),
        "train_validation_canonical_overlap_after_filter": 0,
        "selected_train_molecules": len(train_dataset),
        "selected_validation_molecules": len(validation_dataset),
        "train_omitted_over_max_atoms_or_unsupported": train_omitted,
        "validation_omitted_over_max_atoms_or_unsupported": validation_omitted,
        "validation_coverage": validation_coverage, "max_atoms": int(args.max_atoms),
        "node_dim": int(args.node_dim), "edge_dim": int(args.edge_dim), "layers": int(args.layers),
        "latent_noise": float(args.latent_noise),
        "stress_latent_noise": float(args.stress_latent_noise),
        "category_mask_probability": float(args.category_mask_probability),
        "fingerprint_bits": int(args.fingerprint_bits),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    model = GraphLatentAutoencoder(int(args.node_dim), int(args.edge_dim), int(args.layers)).to(device)
    history = train_model(model, train_dataset, args, device)
    checkpoint_path = args.output_dir / "graph_latent_autoencoder.pt"
    torch.save({
        "stage": "graph_latent_reconstruction_gate", "model_state": model.state_dict(),
        "model_config": {"node_dim": int(args.node_dim), "edge_dim": int(args.edge_dim),
                         "layers": int(args.layers), "max_atoms": int(args.max_atoms)},
        "history": history, "manifest": manifest,
    }, checkpoint_path)
    rows, metrics = evaluate(model, validation_dataset, args, device)
    gate = build_gate(metrics, validation_coverage, args)
    summary = {
        "protocol": PROTOCOL, "checkpoint": str(checkpoint_path), "training": history,
        "representation": metrics, "gate": gate,
        "next_stage": "categorical_graph_latent_flow" if gate["passed"] else "stop_before_flow",
    }
    write_rows(args.output_dir / "reconstructions.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
