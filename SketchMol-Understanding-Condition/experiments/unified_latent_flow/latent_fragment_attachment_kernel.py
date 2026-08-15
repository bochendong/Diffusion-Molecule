#!/usr/bin/env python3
"""Train a continuous latent fragment-attachment kernel on train pairs only.

Each training action is an exact one-cut source-to-target fragment transform.
The frozen graph encoder supplies the source latent and the set-compositional
property slots supply the request.  One head samples an attachment site among
source-only MMPA splits; a conditional flow transports Gaussian noise to a
target-fragment fingerprint endpoint.  The endpoint is quantized once to the
nearest train-only fragment token and attached to the chosen source core.

This is a VQ-style generative decoder, not molecular candidate ranking: every
latent attempt chooses exactly one site and one token and produces at most one
raw molecule.  There is no development target, property oracle, validity
feedback, retry, repair, or finalizer until all 20 raw attempts are frozen.
"""

from __future__ import annotations

import argparse
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

import build_retrieved_delta_edit_candidates as fragments  # noqa: E402
import source_relative_delta_diffusion as delta  # noqa: E402


base = delta.base
belief = delta.belief
graph = delta.graph
hierarchical = delta.hierarchical
unified = delta.unified

PROTOCOL = "continuous_latent_fragment_attachment_kernel_pilot_v24"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--coverage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1500)
    parser.add_argument("--validation-limit", type=int, default=20)
    parser.add_argument("--property-counts", default="2,3")
    parser.add_argument("--fingerprint-bits", type=int, default=256)
    parser.add_argument("--graph-fingerprint-bits", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--site-loss-weight", type=float, default=0.50)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--site-temperature", type=float, default=0.80)
    parser.add_argument("--num-attempts", type=int, default=20)
    parser.add_argument("--mcs-timeout", type=int, default=1)
    parser.add_argument("--min-common-fraction", type=float, default=0.45)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--validation-selection-seed", type=int, default=2719)
    parser.add_argument("--validation-exclusion-seed", type=int, default=1742)
    parser.add_argument("--train-selection-seed", type=int, default=1741)
    parser.add_argument("--gate-validity", type=float, default=0.90)
    parser.add_argument("--gate-strict-any20", type=float, default=0.45)
    parser.add_argument("--gate-3p-strict-any20", type=float, default=0.30)
    parser.add_argument("--gate-mean-unique-valid", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1761)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class Site:
    core: str
    variable: str
    feature: np.ndarray


@dataclass(frozen=True)
class AttachmentAction:
    pair_index: int
    site_features: np.ndarray
    positive_site: int
    target_fragment: str


def fragment_fingerprint(smiles: str, n_bits: int) -> np.ndarray | None:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise RuntimeError("B24 requires RDKit") from exc
    molecule = Chem.MolFromSmiles(str(smiles or ""))
    if molecule is None:
        return None
    editable = Chem.RWMol(molecule)
    for index in sorted(
        (atom.GetIdx() for atom in editable.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    ):
        editable.RemoveAtom(int(index))
    molecule = editable.GetMol()
    if molecule.GetNumHeavyAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=int(n_bits))
    fingerprint = generator.GetFingerprint(molecule)
    array = np.zeros(int(n_bits), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array


def site_feature(split: fragments.FragmentSplit, n_bits: int) -> np.ndarray | None:
    core = fragment_fingerprint(split.core, n_bits)
    variable = fragment_fingerprint(split.variable, n_bits)
    if core is None or variable is None:
        return None
    sizes = np.asarray(
        [
            min(float(split.core_heavy_atoms), 64.0) / 64.0,
            min(float(split.variable_heavy_atoms), 30.0) / 30.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([core, variable, sizes]).astype(np.float32)


def source_sites(smiles: str, args: argparse.Namespace) -> list[Site]:
    output: list[Site] = []
    seen: set[tuple[str, str]] = set()
    for split in fragments.fragment_splits(
        smiles,
        int(args.min_core_heavy_atoms),
        int(args.max_variable_heavy_atoms),
    ):
        key = (split.core, split.variable)
        if key in seen:
            continue
        feature = site_feature(split, int(args.fingerprint_bits))
        if feature is None:
            continue
        seen.add(key)
        output.append(Site(split.core, split.variable, feature))
    return sorted(output, key=lambda item: (item.core, item.variable))


def exact_attachment_actions(
    pair: object, pair_index: int, args: argparse.Namespace
) -> list[tuple[AttachmentAction, str]]:
    sites = source_sites(pair.source_smiles, args)
    if not sites:
        return []
    target_by_core: dict[str, set[str]] = defaultdict(set)
    for split in fragments.fragment_splits(
        pair.target_smiles,
        int(args.min_core_heavy_atoms),
        int(args.max_variable_heavy_atoms),
    ):
        target_by_core[split.core].add(split.variable)
    target_canonical = fragments.canonical_smiles(pair.target_smiles)
    site_matrix = np.stack([site.feature for site in sites]).astype(np.float32)
    output: list[tuple[AttachmentAction, str]] = []
    seen: set[tuple[int, str]] = set()
    for site_index, site in enumerate(sites):
        for target_variable in sorted(target_by_core.get(site.core, set())):
            if target_variable == site.variable:
                continue
            if fragment_fingerprint(target_variable, int(args.fingerprint_bits)) is None:
                continue
            joined = fragments.canonical_smiles(
                fragments.join_fragments(site.core, target_variable)
            )
            if joined != target_canonical:
                continue
            key = (site_index, target_variable)
            if key in seen:
                continue
            seen.add(key)
            output.append(
                (
                    AttachmentAction(
                        pair_index=pair_index,
                        site_features=site_matrix,
                        positive_site=site_index,
                        target_fragment=target_variable,
                    ),
                    site.variable,
                )
            )
    return output


@torch.no_grad()
def encode_sources(
    representation,
    pairs: Sequence[object],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    representation.eval()
    for start in range(0, len(pairs), int(batch_size)):
        items = pairs[start : start + int(batch_size)]
        batch = base.move_graph_batch(graph.collate([pair.source for pair in items]), device)
        node, _edge = representation.encode(batch)
        mask = batch["node_mask"].to(node.dtype)
        pooled = (node * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        rows.append(pooled.float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


class FragmentAttachmentKernel(nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        condition_dim: int,
        site_dim: int,
        endpoint_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.endpoint_dim = int(endpoint_dim)
        self.source_encoder = nn.Sequential(
            nn.LayerNorm(source_dim),
            nn.Linear(source_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.property_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.property_count = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.site_encoder = nn.Sequential(
            nn.LayerNorm(site_dim),
            nn.Linear(site_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.site_query = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )
        self.flow_context = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(endpoint_dim + hidden_dim + 3),
            nn.Linear(endpoint_dim + hidden_dim + 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, endpoint_dim),
        )

    def request_context(
        self, source_latent: torch.Tensor, condition_tokens: torch.Tensor
    ) -> torch.Tensor:
        active = condition_tokens.abs().sum(dim=-1).gt(0)
        encoded = self.property_encoder(condition_tokens)
        encoded = encoded * active.unsqueeze(-1)
        count = active.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = encoded.sum(dim=1) / count.to(encoded.dtype).sqrt()
        count_feature = (count.to(encoded.dtype) / condition_tokens.shape[1]).clamp(0, 1)
        return self.source_encoder(source_latent) + pooled + self.property_count(count_feature)

    def site_logits(
        self,
        request_context: torch.Tensor,
        site_features: torch.Tensor,
        site_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        site_hidden = self.site_encoder(site_features)
        query = self.site_query(request_context)
        logits = (site_hidden * query[:, None, :]).sum(dim=-1) / math.sqrt(
            site_hidden.shape[-1]
        )
        return logits.masked_fill(~site_mask, -torch.inf), site_hidden

    def context_for_site(
        self, request_context: torch.Tensor, site_hidden: torch.Tensor
    ) -> torch.Tensor:
        return self.flow_context(torch.cat([request_context, site_hidden], dim=-1))

    def transport_velocity(
        self, current: torch.Tensor, time: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        time_features = torch.stack(
            [time, torch.sin(math.pi * time), torch.cos(math.pi * time)], dim=-1
        )
        return self.velocity(torch.cat([current, context, time_features], dim=-1))


def collate_actions(
    items: Sequence[AttachmentAction],
    source_latents: np.ndarray,
    condition_tokens: np.ndarray,
    target_lookup: Mapping[str, int],
    target_endpoints: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_sites = max(item.site_features.shape[0] for item in items)
    site_dim = items[0].site_features.shape[1]
    sites = np.zeros((len(items), max_sites, site_dim), dtype=np.float32)
    mask = np.zeros((len(items), max_sites), dtype=bool)
    for index, item in enumerate(items):
        count = item.site_features.shape[0]
        sites[index, :count] = item.site_features
        mask[index, :count] = True
    pair_indices = np.asarray([item.pair_index for item in items], dtype=np.int64)
    target_indices = np.asarray(
        [target_lookup[item.target_fragment] for item in items], dtype=np.int64
    )
    return {
        "source": torch.from_numpy(source_latents[pair_indices]).to(device),
        "condition": torch.from_numpy(condition_tokens[pair_indices]).to(device),
        "sites": torch.from_numpy(sites).to(device),
        "site_mask": torch.from_numpy(mask).to(device),
        "positive_site": torch.as_tensor(
            [item.positive_site for item in items], device=device
        ),
        "target": torch.from_numpy(target_endpoints[target_indices]).to(device),
    }


def train_model(
    model: FragmentAttachmentKernel,
    actions: Sequence[AttachmentAction],
    source_latents: np.ndarray,
    condition_tokens: np.ndarray,
    target_lookup: Mapping[str, int],
    target_endpoints: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = random.Random(int(args.seed))
    history: list[dict[str, float]] = []
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(1, int(args.epochs) + 1):
        order = list(range(len(actions)))
        generator.shuffle(order)
        totals = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(args.batch_size)):
            items = [actions[index] for index in order[start : start + int(args.batch_size)]]
            batch = collate_actions(
                items,
                source_latents,
                condition_tokens,
                target_lookup,
                target_endpoints,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                request = model.request_context(batch["source"], batch["condition"])
                site_logits, site_hidden = model.site_logits(
                    request, batch["sites"], batch["site_mask"]
                )
                site_loss = F.cross_entropy(site_logits.float(), batch["positive_site"])
                selected = site_hidden[
                    torch.arange(len(items), device=device), batch["positive_site"]
                ]
                context = model.context_for_site(request, selected)
                noise = torch.randn_like(batch["target"])
                time = torch.rand(len(items), device=device, dtype=noise.dtype)
                current = (1.0 - time[:, None]) * noise + time[:, None] * batch["target"]
                velocity = model.transport_velocity(current, time, context)
                flow_loss = F.mse_loss(
                    velocity.float(), (batch["target"] - noise).float()
                )
                loss = flow_loss + float(args.site_loss_weight) * site_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            with torch.no_grad():
                site_accuracy = site_logits.argmax(dim=-1).eq(
                    batch["positive_site"]
                ).float().mean()
            totals["loss"] += float(loss.detach())
            totals["flow_loss"] += float(flow_loss.detach())
            totals["site_loss"] += float(site_loss.detach())
            totals["site_accuracy"] += float(site_accuracy)
            batches += 1
        row = {
            "epoch": float(epoch),
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def generate_actions(
    model: FragmentAttachmentKernel,
    pair: object,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    sites = source_sites(pair.source_smiles, args)
    if not sites:
        return [
            {
                "smiles": "",
                "site_core": "",
                "source_fragment": "",
                "target_fragment_token": "",
                "latent_norm": 0.0,
                "quantization_distance": float("inf"),
                "site_entropy": 0.0,
            }
            for _ in range(int(args.num_attempts))
        ]
    attempts = int(args.num_attempts)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    source_tensor = torch.from_numpy(
        np.repeat(source_latent[None, :], attempts, axis=0)
    ).to(device)
    condition = torch.from_numpy(
        np.repeat(pair.condition[None, ...], attempts, axis=0)
    ).to(device)
    site_matrix = np.stack([site.feature for site in sites]).astype(np.float32)
    site_tensor = torch.from_numpy(
        np.repeat(site_matrix[None, ...], attempts, axis=0)
    ).to(device)
    site_mask = torch.ones(
        attempts, len(sites), dtype=torch.bool, device=device
    )
    model.eval()
    request = model.request_context(source_tensor, condition)
    site_logits, site_hidden = model.site_logits(request, site_tensor, site_mask)
    probabilities = torch.softmax(
        site_logits.float() / max(float(args.site_temperature), 1e-4), dim=-1
    )
    selected_sites = torch.multinomial(
        probabilities, 1, replacement=True, generator=generator
    ).squeeze(-1)
    selected_hidden = site_hidden[
        torch.arange(attempts, device=device), selected_sites
    ]
    context = model.context_for_site(request, selected_hidden)
    latent = torch.randn(
        attempts,
        model.endpoint_dim,
        generator=generator,
        device=device,
        dtype=context.dtype,
    )
    for step in range(int(args.flow_steps)):
        time = torch.full(
            (attempts,),
            (step + 0.5) / max(1, int(args.flow_steps)),
            device=device,
            dtype=latent.dtype,
        )
        latent = latent + model.transport_velocity(latent, time, context) / max(
            1, int(args.flow_steps)
        )
    vocabulary = torch.from_numpy(target_endpoints).to(device=device, dtype=latent.dtype)
    distances = ((latent[:, None, :] - vocabulary[None, :, :]) ** 2).mean(dim=-1)
    for attempt, site_index in enumerate(selected_sites.tolist()):
        current_fragment = sites[int(site_index)].variable
        for token_index, token in enumerate(target_fragments):
            if token == current_fragment:
                distances[attempt, token_index] = torch.inf
    selected_tokens = distances.argmin(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    output: list[dict[str, object]] = []
    for attempt, (site_index, token_index) in enumerate(
        zip(selected_sites.tolist(), selected_tokens.tolist())
    ):
        site = sites[int(site_index)]
        target_fragment = target_fragments[int(token_index)]
        product = fragments.join_fragments(site.core, target_fragment)
        output.append(
            {
                "smiles": product,
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": target_fragment,
                "latent_norm": float(latent[attempt].float().norm().cpu()),
                "quantization_distance": float(
                    distances[attempt, token_index].float().cpu()
                ),
                "site_entropy": float(entropy[attempt].float().cpu()),
            }
        )
    return output


def evaluate(
    model: FragmentAttachmentKernel,
    pairs: Sequence[object],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frozen: list[tuple[object, list[dict[str, object]]]] = []
    for pair_index, pair in enumerate(pairs):
        generated = generate_actions(
            model,
            pair,
            source_latents[pair_index],
            target_fragments,
            target_endpoints,
            args,
            device,
            seed=int(args.seed) * 100000 + pair_index,
        )
        if len(generated) != int(args.num_attempts):
            raise RuntimeError("Fragment kernel did not freeze exactly 20 attempts")
        frozen.append((pair, generated))

    candidate_rows: list[dict[str, object]] = []
    for pair_index, (pair, generated) in enumerate(frozen):
        from rdkit import Chem

        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"validation_{pair_index:04d}"
        )
        source_copy_target = (
            graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
        )
        for rank, raw in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(str(raw["smiles"] or ""))
            valid = bool(canonical)
            generated_molecule = Chem.MolFromSmiles(canonical) if valid else None
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
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "site_core": raw["site_core"],
                    "source_fragment": raw["source_fragment"],
                    "target_fragment_token": raw["target_fragment_token"],
                    "latent_norm": raw["latent_norm"],
                    "quantization_distance": raw["quantization_distance"],
                    "site_entropy": raw["site_entropy"],
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": (
                        int(generated_molecule.GetNumAtoms())
                        if generated_molecule is not None
                        else 0
                    ),
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
    for name in ("latent_norm", "quantization_distance", "site_entropy"):
        finite = [
            float(row[name])
            for row in candidate_rows
            if math.isfinite(float(row[name]))
        ]
        metrics[f"mean_{name}"] = float(np.mean(finite)) if finite else 0.0
    metrics["source_copy_rate"] = sum(
        str(row["generated_smiles"])
        == graph.canonical_smiles(str(row["source_smiles"]))
        for row in candidate_rows
    ) / max(1, len(candidate_rows))
    return candidate_rows, metrics


def select_pairs(args: argparse.Namespace) -> tuple[list[object], list[object], dict[str, object]]:
    allowed_counts = base.parse_property_counts(str(args.property_counts))
    validation_rows = base.read_rows(args.validation_csv)
    historical_pairs, historical_counts = base.build_pairs(
        validation_rows,
        max_atoms=64,
        fingerprint_bits=int(args.graph_fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_exclusion_seed),
    )
    historical_sources = {pair.source_smiles for pair in historical_pairs}
    historical_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in historical_pairs
    }
    validation_pairs, validation_counts = base.build_pairs(
        validation_rows,
        max_atoms=64,
        fingerprint_bits=int(args.graph_fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.validation_limit),
        seed=int(args.validation_selection_seed),
        forbidden_sources=historical_sources,
        forbidden_pairs=historical_keys,
    )
    validation_sources = {pair.source_smiles for pair in validation_pairs}
    validation_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in validation_pairs
    }
    train_pairs, train_counts = base.build_pairs(
        base.read_rows(args.train_csv),
        max_atoms=64,
        fingerprint_bits=int(args.graph_fingerprint_bits),
        condition_dim=int(args.condition_dim),
        allowed_counts=allowed_counts,
        timeout=int(args.mcs_timeout),
        min_common_fraction=float(args.min_common_fraction),
        limit=int(args.train_limit),
        seed=int(args.train_selection_seed),
        forbidden_sources=validation_sources,
        forbidden_pairs=validation_keys,
    )
    for pair in [*train_pairs, *validation_pairs]:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(args.condition_dim)
        )
    return train_pairs, validation_pairs, {
        "train_filter_counts": train_counts,
        "validation_filter_counts": validation_counts,
        "historical_validation_filter_counts": historical_counts,
        "historical_sources": historical_sources,
        "historical_keys": historical_keys,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_attempts) != 20:
        raise ValueError("B24 requires exactly 20 raw attempts per condition")
    base.seed_everything(int(args.seed))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = json.loads(args.coverage_summary.read_text(encoding="utf-8"))
    if not bool(coverage.get("gate", {}).get("passed")):
        raise ValueError("B24 coverage gate did not pass")
    representation, representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    train_pairs, validation_pairs, split = select_pairs(args)
    if len(train_pairs) < 32 or not validation_pairs:
        raise ValueError("B24 split does not contain enough pairs")
    train_source_latents = encode_sources(representation, train_pairs, device)
    validation_source_latents = encode_sources(representation, validation_pairs, device)
    condition_tokens = np.stack([pair.condition for pair in train_pairs]).astype(np.float32)

    actions: list[AttachmentAction] = []
    source_fragments: Counter[str] = Counter()
    target_fragment_counts: Counter[str] = Counter()
    covered_pairs: set[int] = set()
    for pair_index, pair in enumerate(train_pairs):
        for action, source_fragment in exact_attachment_actions(pair, pair_index, args):
            actions.append(action)
            source_fragments[source_fragment] += 1
            target_fragment_counts[action.target_fragment] += 1
            covered_pairs.add(pair_index)
    if len(actions) < 100:
        raise ValueError(f"Only {len(actions)} exact fragment actions were built")
    target_fragments = sorted(target_fragment_counts)
    target_lookup = {fragment: index for index, fragment in enumerate(target_fragments)}
    target_bits = []
    for fragment in target_fragments:
        fingerprint = fragment_fingerprint(fragment, int(args.fingerprint_bits))
        if fingerprint is None:
            raise ValueError(f"Target fragment lost fingerprint support: {fragment}")
        target_bits.append(fingerprint)
    target_endpoints = (2.0 * np.stack(target_bits) - 1.0).astype(np.float32)
    site_dim = actions[0].site_features.shape[1]
    model = FragmentAttachmentKernel(
        source_dim=int(representation_config["node_dim"]),
        condition_dim=int(args.condition_dim),
        site_dim=int(site_dim),
        endpoint_dim=int(args.fingerprint_bits),
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    history = train_model(
        model,
        actions,
        train_source_latents,
        condition_tokens,
        target_lookup,
        target_endpoints,
        args,
        device,
    )
    candidate_rows, metrics = evaluate(
        model,
        validation_pairs,
        validation_source_latents,
        target_fragments,
        target_endpoints,
        args,
        device,
    )
    three_property_strict = float(
        metrics["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": float(args.gate_validity)},
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": float(args.gate_strict_any20),
        },
        "three_property_strict_any20": {
            "value": three_property_strict,
            "threshold": float(args.gate_3p_strict_any20),
        },
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": float(args.gate_mean_unique_valid),
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
    validation_sources = {pair.source_smiles for pair in validation_pairs}
    train_keys = {(pair.source_smiles, pair.target_smiles) for pair in train_pairs}
    validation_keys = {
        (pair.source_smiles, pair.target_smiles) for pair in validation_pairs
    }
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "heldout_role": "development_not_final_audit",
        "device": str(device),
        "coverage_protocol": coverage.get("protocol"),
        "coverage_gate_passed": True,
        "representation_protocol": representation_summary.get("protocol"),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "selected_train_pairs": len(train_pairs),
        "covered_train_pairs": len(covered_pairs),
        "train_attachment_actions": len(actions),
        "selected_validation_pairs": len(validation_pairs),
        "unique_source_fragments": len(source_fragments),
        "unique_target_fragment_tokens": len(target_fragments),
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "train_validation_pair_overlap": len(train_keys & validation_keys),
        "historical_validation_source_overlap": len(
            split["historical_sources"] & validation_sources
        ),
        "historical_validation_pair_overlap": len(
            split["historical_keys"] & validation_keys
        ),
        "train_filter_counts": split["train_filter_counts"],
        "validation_filter_counts": split["validation_filter_counts"],
        "historical_validation_filter_counts": split[
            "historical_validation_filter_counts"
        ],
        "frozen_graph_source_latent": True,
        "set_compositional_property_slots": True,
        "continuous_fragment_transport": True,
        "train_only_fragment_vocabulary": True,
        "single_vq_fragment_decode_per_attempt": True,
        "source_only_attachment_site_enumeration": True,
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "generation_rdkit_assembly_access": True,
        "generation_rdkit_validity_feedback": False,
        "candidate_library": False,
        "molecular_candidate_ranking": False,
        "selector": False,
        "finalizer": False,
        "oracle_reranking": False,
        "posthoc_molecule_repair": False,
        "failed_attachment_retry": False,
        "exact_raw_attempts_per_condition": 20,
        "train_selection_seed": int(args.train_selection_seed),
        "validation_selection_seed": int(args.validation_selection_seed),
        "validation_exclusion_seed": int(args.validation_exclusion_seed),
    }
    checkpoint_path = args.output_dir / "latent_fragment_attachment_kernel.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "source_dim": int(representation_config["node_dim"]),
                "condition_dim": int(args.condition_dim),
                "site_dim": int(site_dim),
                "endpoint_dim": int(args.fingerprint_bits),
                "hidden_dim": int(args.hidden_dim),
            },
            "target_fragments": target_fragments,
            "target_endpoints": target_endpoints,
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "next_stage": (
            "add_residual_second_fragment_step_for_unsatisfied_property_latent"
            if not failures
            else "retain_b23_local_delta_and_train_fragment_compatibility_energy"
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
