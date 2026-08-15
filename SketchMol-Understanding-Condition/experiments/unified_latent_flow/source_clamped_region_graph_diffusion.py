#!/usr/bin/env python3
"""Train a source-clamped induced-subgraph diffusion for molecular editing.

B37 replaces the patch/anchor grammar tested by B36 with one generative state:
the editable region itself.  A binary region mask, source-relative node actions,
and every edge incident to the region are denoised by one shared message-passing
network.  Nodes outside the sampled region and outside--outside edges are copied
from the source after every reverse step.  Boundary connections are therefore
modelled without an anchor count, fragment library, edit radius, or molecule
ranking stage.

The pilot uses only B22 train-derived strict early-stop labels.  A deterministic
source-group split creates an internal train-only development set.  Generation
receives a stripped source graph and sanitized condition tokens, freezes exactly
20 raw attempts, and only then opens the train-only development evaluator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import defaultdict, deque
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

import source_anchored_graph_patch_evidence as b36  # noqa: E402


b22 = b36.b22
base = b36.base
belief = b36.belief
delta = b36.delta
full_graph = b36.full_graph
graph = b36.graph
hierarchical = b36.hierarchical
unified = b22.unified

PROTOCOL = "train_only_source_clamped_region_graph_diffusion_v37"
REGION_OUTSIDE, REGION_INSIDE, REGION_MASK = 0, 1, 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "region_mask_is_diffusion_state": True,
        "source_exterior_clamped_each_reverse_step": True,
        "region_incident_edges_jointly_denoised": True,
        "hard_patch_count": False,
        "hard_anchor_limit": False,
        "hard_edit_radius": False,
        "fragment_library": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "exact_raw_attempts_per_condition": 20,
        "fit_dev_source_group_overlap": 0,
        "development_source_limit": 160,
        "epochs": 6,
        "diffusion_steps": 8,
        "flow_steps": 8,
        "birth_capacity": 8,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B37 preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("B37 property-count contract drift")
    implementation_sha256 = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "B37 implementation drift: "
            f"expected {payload.get('implementation_sha256')}, "
            f"found {implementation_sha256}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("B37 locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_summary_sha256": args.b36_summary,
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
        raise ValueError(f"B37 locked input drift: {drift}")

    b22_summary, checkpoint = b36.load_locked_b22(args, preregistration)
    evidence = json.loads(args.b36_summary.read_text(encoding="utf-8"))
    metrics = dict(evidence.get("metrics", {}))
    expected_evidence = dict(preregistration["b36_evidence_trigger"])
    evidence_drift = {}
    for key, expected in expected_evidence.items():
        actual = metrics.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                evidence_drift[key] = {"expected": expected, "actual": actual}
        elif actual != expected:
            evidence_drift[key] = {"expected": expected, "actual": actual}
    if evidence.get("protocol") != b36.PROTOCOL:
        evidence_drift["protocol"] = {
            "expected": b36.PROTOCOL,
            "actual": evidence.get("protocol"),
        }
    if evidence.get("decision") != "stop_graph_patch_representation_after_evidence_gate":
        evidence_drift["decision"] = {
            "expected": "stop_graph_patch_representation_after_evidence_gate",
            "actual": evidence.get("decision"),
        }
    if evidence_drift:
        raise ValueError(f"B37 refuses B36 evidence drift: {evidence_drift}")
    return b22_summary, checkpoint, evidence


def stable_source_order(source: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}\0{source}".encode("utf-8")).hexdigest()


def strict_source_group_split(
    pairs: Sequence[object], *, seed: int, development_source_limit: int
) -> tuple[list[object], list[object], dict[str, object]]:
    strict_pairs = [
        pair
        for pair in pairs
        if bool(b22.property_outcome(pair, pair.target_smiles)[1])
    ]
    sources = sorted(
        {pair.source_smiles for pair in strict_pairs},
        key=lambda value: (stable_source_order(value, seed), value),
    )
    if len(sources) <= int(development_source_limit):
        raise ValueError(
            "B37 source-group split needs more strict sources than the locked dev limit"
        )
    development_sources = set(sources[: int(development_source_limit)])
    fit = [pair for pair in strict_pairs if pair.source_smiles not in development_sources]
    development = [
        pair for pair in strict_pairs if pair.source_smiles in development_sources
    ]
    fit_sources = {pair.source_smiles for pair in fit}
    dev_sources = {pair.source_smiles for pair in development}
    fit_keys = {(pair.source_smiles, pair.target_smiles) for pair in fit}
    dev_keys = {(pair.source_smiles, pair.target_smiles) for pair in development}
    if fit_sources & dev_sources or fit_keys & dev_keys:
        raise ValueError("B37 fit/dev source-group split overlap")
    if len(development) < 128:
        raise ValueError(f"B37 requires at least 128 dev conditions, found {len(development)}")
    return fit, development, {
        "strict_pairs": len(strict_pairs),
        "fit_pairs": len(fit),
        "development_pairs": len(development),
        "fit_sources": len(fit_sources),
        "development_sources": len(dev_sources),
        "fit_dev_source_overlap": 0,
        "fit_dev_pair_overlap": 0,
        "split_seed": int(seed),
    }


def checkpoint_vocabulary(checkpoint: Mapping[str, object]) -> dict[str, object]:
    raw = dict(checkpoint["vocabulary"])
    return {
        "node_states": np.asarray(raw["node_states"], dtype=np.int64),
        "edge_states": np.asarray(raw["edge_states"], dtype=np.int64),
        "blank_node_id": 0,
        "blank_edge_id": 0,
        "sha256": str(raw["sha256"]),
    }


def region_targets(
    node_actions: torch.Tensor, edge_actions: torch.Tensor
) -> torch.Tensor:
    node_changed = node_actions.ne(delta.NODE_KEEP)
    edge_changed = edge_actions.ne(delta.EDGE_KEEP)
    edge_endpoints = edge_changed.any(dim=1) | edge_changed.any(dim=2)
    return node_changed | edge_endpoints


def corrupt_region_and_actions(
    region: torch.Tensor,
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    working: torch.Tensor,
    time: torch.Tensor,
    *,
    node_mask_id: int,
    edge_mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    region_selected = torch.rand_like(region, dtype=torch.float32).lt(time[:, None])
    region_selected &= working
    noisy_region = torch.where(
        region_selected, torch.full_like(region.long(), REGION_MASK), region.long()
    )
    node_eligible = region & working
    node_selected = torch.rand_like(node_actions, dtype=torch.float32).lt(time[:, None])
    node_selected &= node_eligible
    noisy_node = torch.where(
        node_selected, torch.full_like(node_actions, int(node_mask_id)), node_actions
    )
    upper = full_graph.upper_working_pairs(working)
    incident = (region[:, :, None] | region[:, None, :]) & upper
    edge_selected = torch.rand_like(edge_actions, dtype=torch.float32).lt(
        time[:, None, None]
    ) & incident
    edge_selected = edge_selected | edge_selected.transpose(1, 2)
    noisy_edge = torch.where(
        edge_selected, torch.full_like(edge_actions, int(edge_mask_id)), edge_actions
    )
    return (
        noisy_region,
        noisy_node,
        noisy_edge,
        region_selected,
        node_selected,
        edge_selected,
    )


class RegionJointGraphDenoiser(nn.Module):
    """Shared region/node/edge reverse field with no hard region geometry."""

    def __init__(
        self,
        *,
        node_state_count: int,
        edge_state_count: int,
        source_node_dim: int,
        source_edge_dim: int,
        context_dim: int,
        hidden_dim: int,
        max_atoms: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.node_mask_id = int(node_state_count)
        self.edge_mask_id = int(edge_state_count)
        self.region_mask_id = REGION_MASK
        self.node_embedding = nn.Embedding(node_state_count + 1, hidden_dim)
        self.edge_embedding = nn.Embedding(edge_state_count + 1, hidden_dim)
        self.region_embedding = nn.Embedding(3, hidden_dim)
        self.source_node = nn.Linear(source_node_dim, hidden_dim)
        self.source_edge = nn.Linear(source_edge_dim, hidden_dim)
        self.birth_rank = nn.Embedding(max_atoms + 1, hidden_dim)
        self.time = nn.Sequential(
            full_graph.continuous.TimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [full_graph.DenseDiscreteGraphLayer(hidden_dim) for _ in range(int(layers))]
        )
        self.region_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2)
        )
        self.region_size_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_atoms + 1),
        )
        self.node_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, node_state_count)
        )
        self.edge_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_state_count),
        )

    def forward(
        self,
        region_ids: torch.Tensor,
        node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        source_node: torch.Tensor,
        source_edge: torch.Tensor,
        source_mask: torch.Tensor,
        working_mask: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context(torch.cat([condition, latent], dim=-1)) + self.time(time)
        ranks = belief.source_birth_ranks(source_mask).clamp_max(
            self.birth_rank.num_embeddings - 1
        )
        source_node_value = self.source_node(source_node)
        source_edge_value = self.source_edge(source_edge)
        node = (
            self.node_embedding(node_ids)
            + self.region_embedding(region_ids)
            + source_node_value
            + self.birth_rank(ranks)
            + context[:, None, :]
        ) * working_mask.unsqueeze(-1)
        edge = self.edge_embedding(edge_ids) + source_edge_value
        pair_mask = working_mask[:, :, None] & working_mask[:, None, :]
        edge = edge * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            node, edge = layer(node, edge, source_edge_value, context, working_mask)
        pooled = (node * working_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / working_mask.sum(dim=1, keepdim=True).clamp_min(1).sqrt()
        left, right = node[:, :, None, :], node[:, None, :, :]
        edge_logits = self.edge_head(
            torch.cat(
                [edge, source_edge_value, left + right, (left - right).abs()], dim=-1
            )
        )
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        return (
            self.region_head(node),
            self.region_size_head(torch.cat([pooled, context], dim=-1)),
            self.node_head(node),
            edge_logits,
        )


class SourceClampedRegionDiffusion(full_graph.ContinuousDiscreteGraphDiffusion):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.denoiser = RegionJointGraphDenoiser(
            node_state_count=int(kwargs["node_state_count"]),
            edge_state_count=int(kwargs["edge_state_count"]),
            source_node_dim=int(kwargs["node_dim"]),
            source_edge_dim=int(kwargs["edge_dim"]),
            context_dim=int(kwargs["condition_dim"]) + int(kwargs["transport_dim"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            max_atoms=int(kwargs["max_atoms"]),
            layers=int(kwargs["message_layers"]),
        )


def structured_losses(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    region: torch.Tensor,
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    region_selected: torch.Tensor,
    node_selected: torch.Tensor,
    edge_selected: torch.Tensor,
    working: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    region_logits, size_logits, node_logits, edge_logits = outputs
    region_loss = full_graph.balanced_categorical_loss(
        region_logits, region.long(), region_selected, REGION_OUTSIDE, 0.35
    )
    node_loss = full_graph.balanced_categorical_loss(
        node_logits, node_actions, node_selected, delta.NODE_KEEP, 0.50
    )
    edge_eval = edge_selected & full_graph.upper_working_pairs(working)
    edge_loss = full_graph.balanced_categorical_loss(
        edge_logits, edge_actions, edge_eval, delta.EDGE_KEEP, 0.25
    )
    size_target = region.sum(dim=1).long().clamp_max(size_logits.shape[-1] - 1)
    size_loss = F.cross_entropy(size_logits.float(), size_target)
    return region_loss, size_loss, node_loss, edge_loss


def train_model(
    model: SourceClampedRegionDiffusion,
    representation: nn.Module,
    pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    history: list[dict[str, float]] = []
    representation.eval().requires_grad_(False)
    batch_size = int(preregistration["batch_size"])
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(pairs)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), batch_size):
            items = [pairs[index] for index in order[start : start + batch_size]]
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
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
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
                noise = torch.randn_like(endpoint) * float(
                    preregistration["latent_noise_scale"]
                )
                flow_time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
                current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
                velocity = model.transport_velocity(
                    current, flow_time, source_node, source["node_mask"], tokens
                )
                flow_loss = F.mse_loss(velocity.float(), (endpoint - noise).float())
                predicted_endpoint = current + (1.0 - flow_time[:, None]) * velocity

                node_actions, edge_actions = delta.delta_action_targets(
                    source, target, vocabulary
                )
                region = region_targets(node_actions, edge_actions)
                working = full_graph.working_node_mask(
                    source["node_mask"],
                    int(preregistration["birth_capacity"]),
                    target["node_mask"],
                )
                diffusion_index = torch.randint(
                    1,
                    int(preregistration["diffusion_steps"]) + 1,
                    (len(items),),
                    device=device,
                )
                diffusion_time = diffusion_index.float() / int(
                    preregistration["diffusion_steps"]
                )
                corrupted = corrupt_region_and_actions(
                    region,
                    node_actions,
                    edge_actions,
                    working,
                    diffusion_time,
                    node_mask_id=model.denoiser.node_mask_id,
                    edge_mask_id=model.denoiser.edge_mask_id,
                )
                noisy_region, noisy_node, noisy_edge = corrupted[:3]
                selected = corrupted[3:]
                outputs = model.denoiser(
                    noisy_region,
                    noisy_node,
                    noisy_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    diffusion_time,
                    condition,
                    predicted_endpoint,
                )
                region_loss, size_loss, node_loss, edge_loss = structured_losses(
                    outputs,
                    region,
                    node_actions,
                    edge_actions,
                    *selected,
                    working,
                )
                correct_structured = region_loss + size_loss + node_loss + edge_loss
                wrong_outputs = model.denoiser(
                    noisy_region,
                    noisy_node,
                    noisy_edge,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    diffusion_time,
                    condition,
                    torch.roll(predicted_endpoint, shifts=1, dims=0),
                )
                wrong_structured = sum(
                    structured_losses(
                        wrong_outputs,
                        region,
                        node_actions,
                        edge_actions,
                        *selected,
                        working,
                    )
                )
                latent_usage = F.relu(
                    float(preregistration["latent_usage_margin"])
                    + correct_structured
                    - wrong_structured
                )
                latent_std = endpoint.float().std(dim=0, unbiased=False)
                variance_loss = F.relu(
                    float(preregistration["latent_min_std"]) - latent_std
                ).mean()
                loss = (
                    correct_structured
                    + float(preregistration["flow_loss_weight"]) * flow_loss
                    + float(preregistration["latent_usage_weight"]) * latent_usage
                    + float(preregistration["latent_variance_weight"]) * variance_loss
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(preregistration["grad_clip"]))
            optimizer.step()
            region_logits, size_logits, node_logits, edge_logits = outputs
            with torch.no_grad():
                region_accuracy = (
                    region_logits.argmax(-1)[selected[0]]
                    .eq(region.long()[selected[0]])
                    .float()
                    .mean()
                    if bool(selected[0].any())
                    else torch.ones((), device=device)
                )
                node_accuracy = (
                    node_logits.argmax(-1)[selected[1]]
                    .eq(node_actions[selected[1]])
                    .float()
                    .mean()
                    if bool(selected[1].any())
                    else torch.ones((), device=device)
                )
                edge_eval = selected[2] & full_graph.upper_working_pairs(working)
                edge_accuracy = (
                    edge_logits.argmax(-1)[edge_eval]
                    .eq(edge_actions[edge_eval])
                    .float()
                    .mean()
                    if bool(edge_eval.any())
                    else torch.ones((), device=device)
                )
                size_accuracy = size_logits.argmax(-1).eq(region.sum(1).long()).float().mean()
            values = {
                "loss": loss,
                "region_loss": region_loss,
                "region_size_loss": size_loss,
                "node_loss": node_loss,
                "edge_loss": edge_loss,
                "flow_loss": flow_loss,
                "latent_usage_loss": latent_usage,
                "latent_variance_loss": variance_loss,
                "region_masked_accuracy": region_accuracy,
                "region_size_accuracy": size_accuracy,
                "node_masked_accuracy": node_accuracy,
                "edge_masked_accuracy": edge_accuracy,
                "posterior_std": latent_std.mean(),
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"B37 non-finite training metrics: {row}")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def component_count(mask: np.ndarray, adjacency: np.ndarray) -> int:
    remaining = set(np.flatnonzero(mask).tolist())
    count = 0
    while remaining:
        count += 1
        seed = min(remaining)
        remaining.remove(seed)
        queue: deque[int] = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbour in np.flatnonzero(adjacency[current]).tolist():
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
    return count


@torch.no_grad()
def sample_from_source(
    model: SourceClampedRegionDiffusion,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    source_example: object,
    condition_tokens: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    """Generate without target, property oracle, RDKit repair, or ranking."""
    attempts = int(preregistration["exact_raw_attempts_per_condition"])
    batch_size = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    outputs: list[dict[str, object]] = []
    model.eval()
    for start in range(0, attempts, batch_size):
        count = min(batch_size, attempts - start)
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
            ) * float(preregistration["latent_noise_scale"])
            for flow_index in range(int(preregistration["flow_steps"])):
                time = torch.full(
                    (count,),
                    (flow_index + 0.5) / int(preregistration["flow_steps"]),
                    device=device,
                    dtype=source_node.dtype,
                )
                latent = latent + model.transport_velocity(
                    latent, time, source_node, source["node_mask"], tokens
                ) / int(preregistration["flow_steps"])
            condition = model.route_condition(tokens)
            working = full_graph.working_node_mask(
                source["node_mask"], int(preregistration["birth_capacity"])
            )
            upper = full_graph.upper_working_pairs(working)
            symmetric = upper | upper.transpose(1, 2)
            region_ids = torch.where(
                working,
                torch.full_like(source["atomic_number"], REGION_MASK),
                torch.full_like(source["atomic_number"], REGION_OUTSIDE),
            )
            node_actions = torch.where(
                working,
                torch.full_like(source["atomic_number"], model.denoiser.node_mask_id),
                torch.full_like(source["atomic_number"], delta.NODE_KEEP),
            )
            edge_actions = torch.where(
                symmetric,
                torch.full_like(source["bond"], model.denoiser.edge_mask_id),
                torch.full_like(source["bond"], delta.EDGE_KEEP),
            )
            source_active = source["atomic_number"].gt(0)
            source_bond = source["bond"].gt(graph.BOND_NONE)
            final_region = torch.zeros_like(working)
            final_node = torch.full_like(node_actions, delta.NODE_KEEP)
            final_edge = torch.full_like(edge_actions, delta.EDGE_KEEP)
            for reverse_index in range(
                int(preregistration["diffusion_steps"]), 0, -1
            ):
                time = torch.full(
                    (count,),
                    reverse_index / int(preregistration["diffusion_steps"]),
                    device=device,
                    dtype=source_node.dtype,
                )
                region_logits, _, node_logits, edge_logits = model.denoiser(
                    region_ids,
                    node_actions,
                    edge_actions,
                    source_node,
                    source_edge,
                    source["node_mask"].bool(),
                    working,
                    time,
                    condition,
                    latent,
                )
                sampled_region, region_confidence = full_graph.sample_categorical(
                    region_logits.float(), generator, float(preregistration["temperature"])
                )
                sampled_region = sampled_region.eq(REGION_INSIDE) & working
                node_legal = delta.legal_node_action_mask(
                    source_active, model.denoiser.node_mask_id
                )
                sampled_node, node_confidence = full_graph.sample_categorical(
                    node_logits.float().masked_fill(~node_legal, -torch.inf),
                    generator,
                    float(preregistration["temperature"]),
                )
                sampled_node = torch.where(
                    sampled_region,
                    sampled_node,
                    torch.full_like(sampled_node, delta.NODE_KEEP),
                )
                predicted_active = delta.action_active_nodes(source_active, sampled_node)
                edge_legal = delta.legal_edge_action_mask(
                    source_bond, predicted_active, model.denoiser.edge_mask_id
                )
                sampled_edge, edge_confidence = full_graph.sample_categorical(
                    edge_logits.float().masked_fill(~edge_legal, -torch.inf),
                    generator,
                    float(preregistration["temperature"]),
                )
                incident = (sampled_region[:, :, None] | sampled_region[:, None, :]) & upper
                sampled_edge = torch.where(
                    incident,
                    sampled_edge,
                    torch.full_like(sampled_edge, delta.EDGE_KEEP),
                )
                sampled_edge = sampled_edge + sampled_edge.transpose(1, 2)
                edge_confidence = torch.where(
                    incident, edge_confidence, torch.zeros_like(edge_confidence)
                )
                edge_confidence = edge_confidence + edge_confidence.transpose(1, 2)
                final_region, final_node, final_edge = (
                    sampled_region,
                    sampled_node,
                    sampled_edge,
                )
                fraction = (reverse_index - 1) / int(
                    preregistration["diffusion_steps"]
                )
                region_ids = full_graph.remask_low_confidence(
                    sampled_region.long(),
                    region_confidence,
                    working,
                    REGION_MASK,
                    fraction,
                )
                node_actions = full_graph.remask_low_confidence(
                    sampled_node,
                    node_confidence,
                    sampled_region,
                    model.denoiser.node_mask_id,
                    fraction,
                )
                node_actions = torch.where(
                    sampled_region,
                    node_actions,
                    torch.full_like(node_actions, delta.NODE_KEEP),
                )
                edge_actions = full_graph.remask_low_confidence(
                    sampled_edge,
                    edge_confidence,
                    incident,
                    model.denoiser.edge_mask_id,
                    fraction,
                )
                edge_actions = torch.where(
                    incident,
                    edge_actions,
                    torch.full_like(edge_actions, delta.EDGE_KEEP),
                )
                edge_actions = edge_actions + edge_actions.transpose(1, 2)
            result = delta.apply_delta_actions(source, final_node, final_edge, vocabulary)

        prediction = {
            key: value.detach().cpu().numpy() for key, value in result.items()
        }
        source_prediction = {
            key: value.detach().cpu().numpy()
            for key, value in source.items()
            if isinstance(value, torch.Tensor)
        }
        upper_cpu = torch.triu(
            torch.ones(source["bond"].shape[1:], dtype=torch.bool), diagonal=1
        )
        regions = final_region.detach().cpu().numpy().astype(bool)
        node_values = final_node.detach().cpu()
        edge_values = final_edge.detach().cpu()
        for index in range(count):
            smiles, _ = graph.graph_to_smiles(prediction, index)
            outside = ~regions[index]
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
            result_adjacency = prediction["bond"][index] > graph.BOND_NONE
            source_adjacency = source_prediction["bond"][index] > graph.BOND_NONE
            adjacency = result_adjacency | source_adjacency
            region_pair = regions[index][:, None] | regions[index][None, :]
            boundary = np.logical_xor(regions[index][:, None], regions[index][None, :])
            changed_edges = edge_values[index].ne(delta.EDGE_KEEP) & upper_cpu
            outputs.append(
                {
                    "generated_smiles": graph.canonical_smiles(smiles or ""),
                    "predicted_atom_count": int(
                        (prediction["atomic_number"][index] > 0).sum()
                    ),
                    "latent_norm": float(latent[index].float().norm().detach().cpu()),
                    "region_size": int(regions[index].sum()),
                    "region_components": component_count(regions[index], adjacency),
                    "node_edit_count": int(node_values[index].ne(delta.NODE_KEEP).sum()),
                    "edge_edit_count": int(changed_edges.sum()),
                    "boundary_edge_edit_count": int(
                        (changed_edges.numpy() & boundary).sum()
                    ),
                    "region_incident_pair_count": int(
                        (region_pair & np.triu(np.ones_like(region_pair), 1).astype(bool)).sum()
                    ),
                    "outside_source_invariant": bool(
                        outside_nodes_exact and outside_edges_exact
                    ),
                }
            )
    if len(outputs) != attempts:
        raise RuntimeError(f"B37 expected {attempts} attempts, produced {len(outputs)}")
    return outputs


def freeze_candidates(
    model: SourceClampedRegionDiffusion,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    """The generation boundary deliberately does not inspect pair.target."""
    rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pairs):
        generated = sample_from_source(
            model,
            representation,
            vocabulary,
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
                        "stage": "freeze_train_only_development_candidates",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise RuntimeError(f"B37 freeze expected {expected} rows, found {len(rows)}")
    return rows


def evaluate_frozen_candidates(
    frozen: Sequence[Mapping[str, object]], pairs: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Open train-only dev targets/oracles only after raw rows are frozen."""
    evaluated: list[dict[str, object]] = []
    for row in frozen:
        pair = pairs[int(row["pair_index"])]
        smiles = str(row["generated_smiles"] or "")
        valid = bool(smiles)
        source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, smiles) if valid else None
        target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, smiles) if valid else None
        fraction, _, evaluated_properties, property_success = (
            unified.instruction_success_and_distance(
                pair.row, smiles, task_specs=base.task_specs(pair.row)
            )
        )
        similarity_success = bool(
            source_tanimoto is not None and source_tanimoto >= 0.4
        )
        evaluated.append(
            {
                **dict(row),
                "target_smiles": pair.target_smiles,
                "valid": valid,
                "source_tanimoto": float(source_tanimoto or 0.0),
                "target_tanimoto": float(target_tanimoto or 0.0),
                "property_fraction": float(fraction),
                "evaluated_properties": int(evaluated_properties),
                "property_success": bool(property_success),
                "source_similarity_success": similarity_success,
                "strict_success": bool(property_success and similarity_success),
                "source_copy_target_tanimoto": float(
                    graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
                ),
            }
        )
    metrics = base.summarize_candidates(evaluated, 20)
    for name in (
        "latent_norm",
        "region_size",
        "region_components",
        "node_edit_count",
        "edge_edit_count",
        "boundary_edge_edit_count",
    ):
        metrics[f"mean_{name}"] = float(
            np.mean([float(row[name]) for row in evaluated])
        )
    metrics["outside_source_invariant_rate"] = sum(
        bool(row["outside_source_invariant"]) for row in evaluated
    ) / max(1, len(evaluated))
    metrics["nonempty_region_rate"] = sum(
        int(row["region_size"]) > 0 for row in evaluated
    ) / max(1, len(evaluated))
    return evaluated, metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, checkpoint, b36_evidence = check_locked_inputs(
        args, preregistration
    )
    selected_pairs, reconstruction = b36.reconstruct_b22_train_pairs(
        args, preregistration, checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = strict_source_group_split(
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
    vocabulary = checkpoint_vocabulary(checkpoint)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = SourceClampedRegionDiffusion(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(preregistration["condition_dim"]),
        transport_dim=int(preregistration["transport_dim"]),
        hidden_dim=int(preregistration["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(preregistration["message_layers"]),
    ).to(device)
    history = train_model(
        model, representation, fit_pairs, vocabulary, preregistration, device
    )
    frozen = freeze_candidates(
        model,
        representation,
        vocabulary,
        development_pairs,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, list(frozen))
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = evaluate_frozen_candidates(frozen, development_pairs)
    base.write_candidate_rows(args.output_dir / "evaluated_train_only_dev_candidates.csv", evaluated)

    gates = dict(preregistration["gates"])
    two_property = float(
        dict(metrics["by_property_count"]).get("2", {}).get("strict_any20", 0.0)
    )
    three_property = float(
        dict(metrics["by_property_count"]).get("3", {}).get("strict_any20", 0.0)
    )
    checks = {
        "exact_attempts": {
            "value": metrics["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": metrics["validity"], "threshold": gates["validity"]},
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": gates["strict_any20"],
        },
        "two_property_strict_any20": {
            "value": two_property,
            "threshold": gates["two_property_strict_any20"],
        },
        "three_property_strict_any20": {
            "value": three_property,
            "threshold": gates["three_property_strict_any20"],
        },
        "outside_source_invariant_rate": {
            "value": metrics["outside_source_invariant_rate"],
            "threshold": 1.0,
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name == "exact_attempts"
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "representation_protocol": representation_summary.get("protocol"),
        "b22_protocol": b22.PROTOCOL,
        "b36_protocol": b36.PROTOCOL,
        "b36_decision": b36_evidence.get("decision"),
        "reconstruction": reconstruction,
        "split": split,
        "train_only_strict_early_stop_supervision": True,
        "train_only_property_oracle_for_label_construction": True,
        "train_only_internal_development_evaluation": True,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "frozen_candidates_sha256": frozen_sha256,
        "exact_raw_attempts_per_condition": 20,
        "region_mask_is_diffusion_state": True,
        "source_exterior_clamped_each_reverse_step": True,
        "region_incident_edges_jointly_denoised": True,
        "hard_patch_count": False,
        "hard_anchor_limit": False,
        "hard_edit_radius": False,
        "fragment_library": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "posthoc_molecule_repair": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
        "de_novo_null_exterior_compatible_but_not_evaluated": True,
    }
    checkpoint_path = args.output_dir / "source_clamped_region_graph_diffusion.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "node_dim": int(representation_config["node_dim"]),
                "edge_dim": int(representation_config["edge_dim"]),
                "condition_dim": int(preregistration["condition_dim"]),
                "transport_dim": int(preregistration["transport_dim"]),
                "hidden_dim": int(preregistration["hidden_dim"]),
                "max_atoms": int(representation_config["max_atoms"]),
                "property_count": len(unified.PROPERTY_COLUMNS),
                "node_action_count": node_action_count,
                "edge_action_count": edge_action_count,
                "message_layers": int(preregistration["message_layers"]),
            },
            "vocabulary": {
                "node_states": np.asarray(vocabulary["node_states"]).tolist(),
                "edge_states": np.asarray(vocabulary["edge_states"]).tolist(),
                "sha256": vocabulary["sha256"],
            },
            "history": history,
            "manifest": manifest,
        },
        checkpoint_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "metrics": metrics,
        "internal_gate": {
            "passed": not failures,
            "checks": checks,
            "failures": failures,
        },
        "decision": (
            "advance_source_clamped_region_diffusion_to_prospective_transfer"
            if not failures
            else "stop_and_diagnose_region_transport_without_patch_expansion"
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
