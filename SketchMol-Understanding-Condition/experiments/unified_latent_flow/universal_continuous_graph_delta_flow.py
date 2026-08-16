#!/usr/bin/env python3
"""Universal continuous graph-delta flow with one-shot closed decoding.

This protocol is the architecture reset after the property-aligned VQ result.
It removes the discrete codebook and the task-partitioned transaction support.
A frozen graph autoencoder provides pooled source and target graph latents on
fit pairs only.  A conditional rectified flow learns the continuous
source-relative graph-latent displacement jointly with fit-only property
deltas.  At generation time twenty continuous particles are transported from
the source and property request, and each state is decoded once into one
source-applicable complete reaction transaction from a universal fit grammar.

The closed transaction layer is only a chemical decoder constraint.  Decoder
scores are distances in the learned graph/property latent space; generated
molecule properties, development targets, and property oracles are unavailable
until all twenty rows have been frozen.  There is no VQ code, task-specific
vocabulary, molecule ranking, oracle selection, retry, repair, or second edit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import compositional_closed_transaction_vq_flow as vq  # noqa: E402
import property_aligned_balanced_transaction_transport as balanced  # noqa: E402


atomic = vq.atomic
base = vq.base
belief = vq.belief
evidence = vq.evidence
graph = vq.graph
reaction_probe = vq.reaction_probe

PROTOCOL = "train_only_universal_continuous_graph_delta_flow_v1"


@dataclass(frozen=True)
class UniversalClosedTransaction:
    reaction_smarts: tuple[str, ...]
    fit_source_smiles: str
    fit_target_smiles: str
    component_count: int
    origin_tasks: tuple[str, ...]


@dataclass(frozen=True)
class UniversalApplicableAction:
    smiles: str
    transaction_key: str
    component_count: int
    fit_source_smiles: str
    origin_tasks: tuple[str, ...]


class ContinuousGraphDeltaFlow(nn.Module):
    """Rectified velocity field over continuous graph/property displacements."""

    def __init__(
        self,
        source_dim: int,
        request_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.LayerNorm(int(source_dim) + int(request_dim)),
            nn.Linear(int(source_dim) + int(request_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(hidden_dim) + 3),
            nn.Linear(int(latent_dim) + int(hidden_dim) + 3, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(
        self,
        latent: torch.Tensor,
        source: torch.Tensor,
        request: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        context = self.condition(torch.cat([source.float(), request.float()], dim=1))
        time_features = torch.stack(
            [time, torch.sin(math.pi * time), torch.cos(math.pi * time)], dim=1
        )
        return self.velocity(
            torch.cat([latent.float(), context, time_features.float()], dim=1)
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-records", type=Path, required=True)
    parser.add_argument("--b41-checkpoint", type=Path, required=True)
    parser.add_argument("--b41-summary", type=Path, required=True)
    parser.add_argument("--set-evidence-summary", type=Path, required=True)
    parser.add_argument("--set-evidence-records", type=Path, required=True)
    parser.add_argument("--b43-checkpoint", type=Path, required=True)
    parser.add_argument("--b43-summary", type=Path, required=True)
    parser.add_argument("--atomic-checkpoint", type=Path, required=True)
    parser.add_argument("--atomic-summary", type=Path, required=True)
    parser.add_argument("--radius-one-support-probe", type=Path, required=True)
    parser.add_argument("--radius-zero-support-probe", type=Path, required=True)
    parser.add_argument("--balanced-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "architecture_reset_after_vq": True,
        "frozen_graph_autoencoder": True,
        "continuous_graph_delta_latent": True,
        "conditional_rectified_flow": True,
        "discrete_vq_codebook": False,
        "universal_cross_task_transaction_grammar": True,
        "task_partitioned_transaction_support": False,
        "fit_only_closed_reaction_grammar": True,
        "fit_only_property_delta_labels": True,
        "development_target_latent_access": False,
        "orthogonal_continuous_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "second_edit": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "development_source_limit": 160,
        "condition_slots": 18,
        "primary_evaluator_semantics_match_b38": True,
        "transaction_native_diagnostics": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Universal continuous-flow preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Universal continuous-flow property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Universal continuous-flow implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    expected_inputs = {
        "atomic_checkpoint_sha256",
        "atomic_summary_sha256",
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_records_sha256",
        "b41_checkpoint_sha256",
        "b41_summary_sha256",
        "b43_checkpoint_sha256",
        "b43_summary_sha256",
        "balanced_summary_sha256",
        "radius_one_support_probe_sha256",
        "radius_zero_support_probe_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "set_evidence_records_sha256",
        "set_evidence_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Universal continuous-flow locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    b22_summary, b22_checkpoint, _b43_summary, _atomic_summary = vq.check_locked_inputs(
        args, preregistration
    )
    locked = dict(preregistration["locked_inputs"])
    balanced_sha = belief.file_sha256(args.balanced_summary)
    if balanced_sha != locked["balanced_summary_sha256"]:
        raise ValueError(
            "Universal continuous-flow balanced-summary drift: "
            f"expected {locked['balanced_summary_sha256']}, found {balanced_sha}"
        )
    balanced_summary = json.loads(args.balanced_summary.read_text(encoding="utf-8"))
    if balanced_summary.get("protocol") != balanced.PROTOCOL:
        raise ValueError("Universal continuous-flow requires the locked balanced protocol")
    if balanced_summary.get("decision") != "stop_property_aligned_transport_without_gate_changes":
        raise ValueError("Universal continuous-flow refuses a balanced decision drift")
    metrics = dict(balanced_summary.get("metrics", {}))
    baseline_drift = {
        key: {"expected": expected, "actual": metrics.get(key)}
        for key, expected in dict(preregistration["balanced_baseline"]).items()
        if not math.isclose(
            float(metrics.get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if baseline_drift:
        raise ValueError(f"Universal continuous-flow baseline drift: {baseline_drift}")
    return b22_summary, b22_checkpoint, balanced_summary


def universal_transaction_key(reaction_smarts: Sequence[str]) -> str:
    payload = json.dumps(
        {"reaction_smarts": list(reaction_smarts)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_universal_fit_grammar(
    fit_pairs: Sequence[object], preregistration: Mapping[str, object]
) -> tuple[list[UniversalClosedTransaction], list[tuple[object, str]], dict[str, object]]:
    raw: dict[str, dict[str, object]] = {}
    examples: list[tuple[object, str]] = []
    counts: Counter[str] = Counter()
    for index, pair in enumerate(fit_pairs, start=1):
        task = base.task_key(pair.row)
        counts["fit_pairs"] += 1
        target = graph.canonical_smiles(pair.target_smiles)
        if not target or "." in target:
            counts["disconnected_or_invalid_target"] += 1
            continue
        try:
            templates = reaction_probe.extract_templates(
                pair.source,
                pair.target,
                int(preregistration["reaction_context_radius"]),
            )
        except Exception:
            counts["template_extraction_error"] += 1
            continue
        if not templates:
            counts["template_empty"] += 1
            continue
        products, _raw = reaction_probe.apply_component_tuple(
            pair.source_smiles,
            templates,
            max_frontier=int(preregistration["max_reaction_frontier"]),
        )
        if target not in products:
            counts["nonexact_self_replay"] += 1
            continue
        smarts = tuple(template.reaction_smarts for template in templates)
        key = universal_transaction_key(smarts)
        if key not in raw:
            raw[key] = {
                "reaction_smarts": smarts,
                "fit_source_smiles": pair.source_smiles,
                "fit_target_smiles": target,
                "component_count": len(smarts),
                "origin_tasks": set(),
            }
        raw[key]["origin_tasks"].add(task)  # type: ignore[union-attr]
        examples.append((pair, key))
        counts["exact_self_replay_pairs"] += 1
        if index % 128 == 0 or index == len(fit_pairs):
            print(
                json.dumps(
                    {
                        "stage": "build_universal_fit_grammar",
                        "fit_pairs": index,
                        "exact_pairs": counts["exact_self_replay_pairs"],
                        "unique_transactions": len(raw),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    transactions = [
        UniversalClosedTransaction(
            reaction_smarts=tuple(raw[key]["reaction_smarts"]),
            fit_source_smiles=str(raw[key]["fit_source_smiles"]),
            fit_target_smiles=str(raw[key]["fit_target_smiles"]),
            component_count=int(raw[key]["component_count"]),
            origin_tasks=tuple(sorted(raw[key]["origin_tasks"])),
        )
        for key in sorted(raw)
    ]
    cross_task = sum(len(transaction.origin_tasks) > 1 for transaction in transactions)
    manifest = {
        "counts": dict(counts),
        "unique_universal_transactions": len(transactions),
        "training_examples": len(examples),
        "transactions_seen_in_multiple_tasks": cross_task,
        "task_in_key": False,
        "connected_target_exact_replay_rate": counts["exact_self_replay_pairs"]
        / max(1, counts["fit_pairs"] - counts["disconnected_or_invalid_target"]),
    }
    return transactions, examples, manifest


def pool_graph_latent(
    node: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    node_weight = mask.float().unsqueeze(-1)
    node_pool = (node.float() * node_weight).sum(dim=1) / node_weight.sum(dim=1).clamp_min(1.0)
    pair_weight = mask.float()[:, :, None] * mask.float()[:, None, :]
    diagonal = torch.eye(mask.shape[1], device=mask.device)[None, :, :]
    pair_weight = pair_weight * (1.0 - diagonal)
    edge_pool = (edge.float() * pair_weight.unsqueeze(-1)).sum(dim=(1, 2))
    edge_pool = edge_pool / pair_weight.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1.0)
    return torch.cat([node_pool, edge_pool], dim=1)


@torch.no_grad()
def encode_fit_graph_latents(
    representation: nn.Module,
    examples: Sequence[tuple[object, str]],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    sources: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    representation.eval()
    for offset in range(0, len(examples), int(batch_size)):
        items = [pair for pair, _key in examples[offset : offset + int(batch_size)]]
        source = base.move_graph_batch(graph.collate([pair.source for pair in items]), device)
        target = base.move_graph_batch(graph.collate([pair.target for pair in items]), device)
        source_node, source_edge = representation.encode(source)
        target_node, target_edge = representation.encode(target)
        source_pool = pool_graph_latent(source_node, source_edge, source["node_mask"])
        target_pool = pool_graph_latent(target_node, target_edge, target["node_mask"])
        sources.append(source_pool.cpu().numpy().astype(np.float32))
        deltas.append((target_pool - source_pool).cpu().numpy().astype(np.float32))
    return np.concatenate(sources), np.concatenate(deltas)


@torch.no_grad()
def encode_development_sources(
    representation: nn.Module,
    pairs: Sequence[object],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode only source graphs; development target objects are never collated."""

    outputs: list[np.ndarray] = []
    representation.eval()
    for offset in range(0, len(pairs), int(batch_size)):
        items = pairs[offset : offset + int(batch_size)]
        source = base.move_graph_batch(graph.collate([pair.source for pair in items]), device)
        source_node, source_edge = representation.encode(source)
        outputs.append(
            pool_graph_latent(source_node, source_edge, source["node_mask"])
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return np.concatenate(outputs)


def deterministic_pca(
    values: np.ndarray, dimensions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Continuous graph-delta PCA requires a nontrivial matrix")
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(int(dimensions), right.shape[0])
    components = right[:dimensions].copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    projected = centered @ components.T
    scale = projected.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    normalized = projected / scale
    explained = np.square(singular[:dimensions])
    total = float(np.square(singular).sum())
    manifest = {
        "input_dimension": int(values.shape[1]),
        "graph_delta_dimensions": dimensions,
        "explained_variance_fraction": float(explained.sum() / max(total, 1e-12)),
        "fit_projected_std_min": float(normalized.std(axis=0).min()),
        "fit_projected_std_max": float(normalized.std(axis=0).max()),
    }
    return (
        normalized.astype(np.float32),
        mean.squeeze(0).astype(np.float32),
        components.astype(np.float32),
        scale.squeeze(0).astype(np.float32),
        manifest,
    )


def fit_latent_dataset(
    examples: Sequence[tuple[object, str]],
    source_latents: np.ndarray,
    graph_deltas: np.ndarray,
    preregistration: Mapping[str, object],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, object],
]:
    projected, mean, components, scale, pca_manifest = deterministic_pca(
        graph_deltas, int(preregistration["graph_delta_dimensions"])
    )
    vocabulary = balanced.property_vocabulary(examples)
    property_rows: list[np.ndarray] = []
    request_rows: list[np.ndarray] = []
    attempted = 0
    evaluated_count = 0
    by_key_graph: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    by_key_property: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for index, (pair, key) in enumerate(examples):
        effect, _observed, pair_attempted, pair_evaluated = balanced.fit_property_effect(
            pair, vocabulary, float(preregistration["property_delta_clip"])
        )
        property_rows.append(effect)
        request_rows.append(balanced.property_request_vector(pair, vocabulary))
        by_key_graph[key].append(projected[index])
        by_key_property[key].append(effect)
        attempted += pair_attempted
        evaluated_count += pair_evaluated
    property_matrix = np.stack(property_rows).astype(np.float32)
    requests = np.stack(request_rows).astype(np.float32)
    targets = np.concatenate(
        [
            float(preregistration["graph_target_scale"]) * projected,
            float(preregistration["property_target_scale"]) * property_matrix,
        ],
        axis=1,
    ).astype(np.float32)
    transaction_embeddings = {
        key: np.concatenate(
            [
                float(preregistration["graph_target_scale"])
                * np.mean(by_key_graph[key], axis=0),
                float(preregistration["property_target_scale"])
                * np.mean(by_key_property[key], axis=0),
            ]
        ).astype(np.float32)
        for key in by_key_graph
    }
    property_manifest = {
        "property_vocabulary": vocabulary,
        "attempted_fit_property_labels": attempted,
        "evaluated_fit_property_labels": evaluated_count,
        "fit_property_label_coverage": evaluated_count / max(1, attempted),
        "fit_only_source_target_property_access": True,
        "development_property_access": False,
    }
    pca_state = {"mean": mean, "components": components, "scale": scale}
    return (
        source_latents.astype(np.float32),
        requests,
        targets,
        vocabulary,
        transaction_embeddings,
        pca_state,
        pca_manifest,
        property_manifest,
    )


def train_flow(
    model: ContinuousGraphDeltaFlow,
    source_latents: np.ndarray,
    requests: np.ndarray,
    targets: np.ndarray,
    property_dimensions: int,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    sources = torch.from_numpy(source_latents)
    request_tensor = torch.from_numpy(requests)
    target_tensor = torch.from_numpy(targets)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    batch_size = int(preregistration["batch_size"])
    graph_dimensions = targets.shape[1] - int(property_dimensions)
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(targets)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            source = sources[indices].to(device)
            request = request_tensor[indices].to(device)
            target = target_tensor[indices].to(device)
            noise = torch.randn_like(target) * float(preregistration["latent_noise_scale"])
            time = torch.rand(len(indices), device=device).clamp_(0.02, 0.98)
            current = (1.0 - time[:, None]) * noise + time[:, None] * target
            optimizer.zero_grad(set_to_none=True)
            velocity = model(current, source, request, time)
            target_velocity = target - noise
            flow_loss = F.mse_loss(velocity, target_velocity)
            predicted_endpoint = current + (1.0 - time[:, None]) * velocity
            property_loss = F.mse_loss(
                predicted_endpoint[:, graph_dimensions:], target[:, graph_dimensions:]
            )
            direction_margin = (
                predicted_endpoint[:, graph_dimensions:] * request
            ).sum(dim=1)
            target_margin = (target[:, graph_dimensions:] * request).sum(dim=1)
            direction_loss = F.mse_loss(direction_margin, target_margin)
            loss = (
                flow_loss
                + float(preregistration["property_endpoint_weight"]) * property_loss
                + float(preregistration["property_direction_weight"]) * direction_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite universal continuous-flow loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(preregistration["grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["flow_matching_loss"] += float(flow_loss.detach())
            totals["property_endpoint_loss"] += float(property_loss.detach())
            totals["property_direction_loss"] += float(direction_loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "training_examples": len(targets),
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    model.eval().requires_grad_(False)
    return history


def templates_from_universal(
    transaction: UniversalClosedTransaction,
) -> tuple[reaction_probe.ComponentTemplate, ...]:
    return tuple(
        reaction_probe.ComponentTemplate(
            reaction_smarts=smarts,
            changed_slots=(),
            context_slots=(),
        )
        for smarts in transaction.reaction_smarts
    )


def universal_applicable_actions(
    source_smiles: str,
    transactions: Sequence[UniversalClosedTransaction],
    preregistration: Mapping[str, object],
) -> list[UniversalApplicableAction]:
    source = graph.canonical_smiles(source_smiles)
    by_key: dict[tuple[str, str], UniversalApplicableAction] = {}
    for transaction in transactions:
        products, _raw = reaction_probe.apply_component_tuple(
            source,
            templates_from_universal(transaction),
            max_frontier=int(preregistration["max_reaction_frontier"]),
        )
        key = universal_transaction_key(transaction.reaction_smarts)
        for product in products:
            if not product or product == source or "." in product:
                continue
            by_key.setdefault(
                (product, key),
                UniversalApplicableAction(
                    smiles=product,
                    transaction_key=key,
                    component_count=transaction.component_count,
                    fit_source_smiles=transaction.fit_source_smiles,
                    origin_tasks=transaction.origin_tasks,
                ),
            )
    return [by_key[key] for key in sorted(by_key)]


@torch.no_grad()
def transport_particles(
    model: ContinuousGraphDeltaFlow,
    source_latent: np.ndarray,
    request: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    latent_dim = int(preregistration["graph_delta_dimensions"]) + len(request)
    particles = vq.orthogonal_particles(
        20, latent_dim, int(preregistration["particle_seed"])
    ).to(device)
    latent = particles * math.sqrt(float(latent_dim)) * float(
        preregistration["latent_noise_scale"]
    )
    source = torch.from_numpy(source_latent)[None, :].to(device).expand(20, -1)
    condition = torch.from_numpy(request)[None, :].to(device).expand(20, -1)
    for step in range(int(preregistration["flow_steps"])):
        time = torch.full(
            (20,),
            (step + 0.5) / float(preregistration["flow_steps"]),
            device=device,
        )
        latent = latent + model(latent, source, condition, time) / float(
            preregistration["flow_steps"]
        )
    return latent


@torch.no_grad()
def freeze_candidates(
    model: ContinuousGraphDeltaFlow,
    development_pairs: Sequence[object],
    development_source_latents: np.ndarray,
    transactions: Sequence[UniversalClosedTransaction],
    transaction_embeddings: Mapping[str, np.ndarray],
    property_vocabulary: Sequence[str],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    support_counts: list[int] = []
    family_counts: list[int] = []
    sampled_family_counts: list[int] = []
    latent_pairwise: list[float] = []
    cross_task_attempts = 0
    identity_attempts = 0
    graph_dimensions = int(preregistration["graph_delta_dimensions"])
    for pair_index, pair in enumerate(development_pairs):
        task = base.task_key(pair.row)
        actions = universal_applicable_actions(
            pair.source_smiles, transactions, preregistration
        )
        support_counts.append(len(actions))
        family_counts.append(len({action.transaction_key for action in actions}))
        condition_id = f"train_only_dev_{pair_index:04d}"
        request = balanced.property_request_vector(pair, property_vocabulary)
        latent = transport_particles(
            model,
            development_source_latents[pair_index],
            request,
            preregistration,
            device,
        )
        if len(latent) > 1:
            latent_pairwise.append(float(torch.pdist(latent).mean().cpu()))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(preregistration["seed"]) * 100000 + pair_index)
        sampled: list[tuple[UniversalApplicableAction, float]] = []
        if actions:
            embeddings = torch.from_numpy(
                np.stack([transaction_embeddings[action.transaction_key] for action in actions])
            ).to(device)
            for particle in latent:
                graph_distance = (
                    particle[:graph_dimensions][None, :]
                    - embeddings[:, :graph_dimensions]
                ).square().mean(dim=1)
                property_distance = (
                    particle[graph_dimensions:][None, :]
                    - embeddings[:, graph_dimensions:]
                ).square().mean(dim=1)
                energy = (
                    float(preregistration["decoder_graph_distance_weight"])
                    * graph_distance
                    + float(preregistration["decoder_property_distance_weight"])
                    * property_distance
                )
                probability = torch.softmax(
                    -energy / float(preregistration["decoder_temperature"]), dim=0
                )
                action_index = int(
                    torch.multinomial(
                        probability.cpu(), 1, generator=generator
                    ).item()
                )
                sampled.append((actions[action_index], float(probability[action_index].cpu())))
            sampled_family_counts.append(
                len({action.transaction_key for action, _probability in sampled})
            )
        else:
            sampled_family_counts.append(0)

        for attempt in range(1, 21):
            if sampled:
                action, decoder_probability = sampled[attempt - 1]
                smiles = action.smiles
                transaction_key_value = action.transaction_key
                component_count = action.component_count
                fit_source = action.fit_source_smiles
                origin_tasks = action.origin_tasks
                identity = False
                cross_task = task not in origin_tasks
                cross_task_attempts += int(cross_task)
            else:
                smiles = pair.source_smiles
                transaction_key_value = "identity_no_applicable_transaction"
                component_count = 0
                fit_source = ""
                origin_tasks = ()
                identity = True
                cross_task = False
                decoder_probability = 1.0
                identity_attempts += 1
            rows.append(
                {
                    "condition_id": condition_id,
                    "pair_index": pair_index,
                    "attempt": attempt,
                    "property_count": int(pair.property_count),
                    "task": task,
                    "source_smiles": pair.source_smiles,
                    "particle_index": attempt - 1,
                    "generated_smiles": smiles,
                    "predicted_atom_count": atomic.molecule_atom_count(smiles),
                    "transaction_support_size": len(actions),
                    "unique_transaction_family_support": len(
                        {action.transaction_key for action in actions}
                    ),
                    "decoder_probability": decoder_probability,
                    "transaction_key": transaction_key_value,
                    "transaction_components": component_count,
                    "fit_source_smiles": fit_source,
                    "origin_tasks": "|".join(origin_tasks),
                    "cross_task_transaction": cross_task,
                    "identity_transaction": identity,
                    "latent_norm": float(latent[attempt - 1].norm().cpu()),
                    "event_count": component_count,
                    "affected_components": component_count,
                    "outside_source_invariant": True,
                    "stopped_by_model": True,
                    "max_horizon_hit": False,
                }
            )
        if (pair_index + 1) % 16 == 0 or pair_index + 1 == len(development_pairs):
            print(
                json.dumps(
                    {
                        "stage": "freeze_universal_continuous_graph_delta_flow",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    support = {
        "conditions": len(development_pairs),
        "conditions_with_nonidentity_support": sum(count > 0 for count in support_counts),
        "condition_support_rate": sum(count > 0 for count in support_counts)
        / max(1, len(support_counts)),
        "mean_applicable_actions": float(np.mean(support_counts)),
        "mean_applicable_transaction_families": float(np.mean(family_counts)),
        "mean_sampled_transaction_families": float(np.mean(sampled_family_counts)),
        "mean_continuous_latent_pairwise_distance": float(np.mean(latent_pairwise)),
        "cross_task_attempt_rate": cross_task_attempts / max(1, len(rows)),
        "identity_attempt_rate": identity_attempts / max(1, len(rows)),
    }
    return rows, support


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gate_result(
    metrics: Mapping[str, object],
    property_manifest: Mapping[str, object],
    support: Mapping[str, object],
    balanced_baseline: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    thresholds = dict(preregistration["gates"])
    by_count = dict(metrics["by_property_count"])
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
        "validity": {"value": metrics["validity"], "threshold": thresholds["validity"]},
        "mean_unique_valid": {"value": metrics["mean_unique_valid"], "threshold": thresholds["mean_unique_valid"]},
        "mean_source_tanimoto": {"value": metrics["mean_source_tanimoto"], "threshold": thresholds["mean_source_tanimoto"]},
        "property_any20": {"value": metrics["property_any20"], "threshold": thresholds["property_any20"]},
        "strict_any20": {"value": metrics["strict_any20"], "threshold": thresholds["strict_any20"]},
        "two_property_strict_any20": {"value": by_count["2"]["strict_any20"], "threshold": thresholds["two_property_strict_any20"]},
        "three_property_strict_any20": {"value": by_count["3"]["strict_any20"], "threshold": thresholds["three_property_strict_any20"]},
        "target_improvement_any20": {"value": metrics["target_improvement_any20"], "threshold": thresholds["target_improvement_any20"]},
        "fit_property_label_coverage": {"value": property_manifest["fit_property_label_coverage"], "threshold": thresholds["fit_property_label_coverage"]},
        "condition_support_rate": {"value": support["condition_support_rate"], "threshold": thresholds["condition_support_rate"]},
        "mean_applicable_transaction_families": {"value": support["mean_applicable_transaction_families"], "threshold": thresholds["mean_applicable_transaction_families"]},
        "mean_continuous_latent_pairwise_distance": {"value": support["mean_continuous_latent_pairwise_distance"], "threshold": thresholds["mean_continuous_latent_pairwise_distance"]},
        "strict_delta_vs_balanced": {"value": float(metrics["strict_any20"]) - float(balanced_baseline["strict_any20"]), "threshold": thresholds["strict_delta_vs_balanced"]},
        "target_improvement_delta_vs_balanced": {"value": float(metrics["target_improvement_any20"]) - float(balanced_baseline["target_improvement_any20"]), "threshold": thresholds["target_improvement_delta_vs_balanced"]},
    }
    failures = [
        name
        for name, check in checks.items()
        if float(check["value"]) < float(check["threshold"])
    ]
    return {"checks": checks, "failures": failures, "passed": not failures}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed universal continuous-flow result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = torch.device(str(args.device))
    if device.type != "cpu":
        raise ValueError("The preregistered universal continuous-flow signal is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, balanced_summary = check_locked_inputs(
        args, preregistration
    )
    representation, representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    selected_pairs, reconstruction = evidence.reconstruct_locked_b36_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = vq.b43.b41.b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    transactions, training_examples, grammar_manifest = build_universal_fit_grammar(
        fit_pairs, preregistration
    )
    fit_sources, fit_graph_deltas = encode_fit_graph_latents(
        representation,
        training_examples,
        int(preregistration["encoding_batch_size"]),
        device,
    )
    (
        fit_sources,
        requests,
        targets,
        property_vocabulary,
        transaction_embeddings,
        pca_state,
        pca_manifest,
        property_manifest,
    ) = fit_latent_dataset(
        training_examples, fit_sources, fit_graph_deltas, preregistration
    )
    model = ContinuousGraphDeltaFlow(
        source_dim=fit_sources.shape[1],
        request_dim=len(property_vocabulary),
        latent_dim=targets.shape[1],
        hidden_dim=int(preregistration["hidden_dim"]),
    ).to(device)
    history = train_flow(
        model,
        fit_sources,
        requests,
        targets,
        len(property_vocabulary),
        preregistration,
        device,
    )
    checkpoint_path = args.output_dir / "universal_continuous_graph_delta_flow.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "source_dim": fit_sources.shape[1],
                "request_dim": len(property_vocabulary),
                "latent_dim": targets.shape[1],
                "hidden_dim": int(preregistration["hidden_dim"]),
            },
            "property_vocabulary": property_vocabulary,
            "graph_delta_pca_state": pca_state,
            "transaction_embeddings": transaction_embeddings,
            "transaction_catalog": [
                {
                    "reaction_smarts": list(transaction.reaction_smarts),
                    "fit_source_smiles": transaction.fit_source_smiles,
                    "fit_target_smiles": transaction.fit_target_smiles,
                    "component_count": transaction.component_count,
                    "origin_tasks": list(transaction.origin_tasks),
                    "transaction_key": universal_transaction_key(
                        transaction.reaction_smarts
                    ),
                }
                for transaction in transactions
            ],
            "pca_manifest": pca_manifest,
        },
        checkpoint_path,
    )
    development_source_latents = encode_development_sources(
        representation,
        development_pairs,
        int(preregistration["encoding_batch_size"]),
        device,
    )
    frozen, development_support = freeze_candidates(
        model,
        development_pairs,
        development_source_latents,
        transactions,
        transaction_embeddings,
        property_vocabulary,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_transactions.csv"
    write_rows(frozen_path, frozen)
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = vq.evaluate_frozen_transactions(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_transactions.csv"
    write_rows(evaluated_path, evaluated)
    gate = gate_result(
        metrics,
        property_manifest,
        development_support,
        dict(preregistration["balanced_baseline"]),
        preregistration,
    )
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "representation_config": representation_config,
        "representation_gate_passed": bool(dict(representation_summary.get("gate", {})).get("passed")),
        "reconstruction": reconstruction,
        "split": split,
        "fit_grammar": grammar_manifest,
        "graph_delta_pca": pca_manifest,
        "property_labels": property_manifest,
        "development_support": development_support,
        "architecture_reset_after_vq": True,
        "frozen_graph_autoencoder": True,
        "continuous_graph_delta_latent": True,
        "conditional_rectified_flow": True,
        "discrete_vq_codebook": False,
        "universal_cross_task_transaction_grammar": True,
        "task_partitioned_transaction_support": False,
        "fit_only_closed_reaction_grammar": True,
        "fit_only_property_delta_labels": True,
        "development_target_latent_access": False,
        "orthogonal_continuous_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
        "only_sampled_transactions_committed": True,
        "frozen_before_target_or_property_evaluation": True,
        "primary_evaluator_semantics_match_b38": True,
        "transaction_native_diagnostics": True,
        "frozen_candidates_sha256": frozen_sha256,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "second_edit": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
        "checkpoint_sha256": belief.file_sha256(checkpoint_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
    }
    summary = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_universal_continuous_graph_delta_flow_to_fresh_confirmation"
            if gate["passed"]
            else "stop_universal_continuous_graph_delta_flow_without_gate_changes"
        ),
        "training": history,
        "balanced_baseline": dict(preregistration["balanced_baseline"]),
        "atomic_baseline": dict(preregistration["atomic_baseline"]),
        "b43_baseline": dict(preregistration["b43_baseline"]),
        "internal_gate": gate,
        "metrics": metrics,
        "manifest": manifest,
        "checkpoint": str(checkpoint_path),
        "locked_balanced_summary_decision": balanced_summary.get("decision"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
