#!/usr/bin/env python3
"""Fit a state-dependent language-grounded adapter inside a frozen graph flow.

Unlike inference-time classifier guidance, this pilot learns a transport
velocity residual directly from paired source-to-target flow-matching targets.
The current latent, source graph representation, and flow time query either
explicit property memory or frozen Common-LLM SFT token memory.  A terminal
reachability head shares the state-memory representation as an auxiliary
fit-only loss, but no gradient-based guidance is used during generation.

The B41 graph representation, graph-event decoder, exact molecule STOP support,
twenty raw attempts, and post-freeze evaluator remain fixed.  The Common LLM
does not emit text, decisions, routes, edits, molecules, or candidate scores.
"""

from __future__ import annotations

import argparse
import hashlib
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
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_llm_state_viability_guidance as state_guidance  # noqa: E402


operator = state_guidance.operator
valid_terminal = state_guidance.valid_terminal
b41 = state_guidance.b41
b40 = state_guidance.b40
b39 = state_guidance.b39
base = state_guidance.base
belief = state_guidance.belief
graph = state_guidance.graph

PROTOCOL = "train_only_language_grounded_graph_latent_flow_v1"
ARMS = ("property_memory", "common_llm_memory")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-summary", type=Path, required=True)
    parser.add_argument("--b37-summary", type=Path, required=True)
    parser.add_argument("--b38-checkpoint", type=Path, required=True)
    parser.add_argument("--b38-summary", type=Path, required=True)
    parser.add_argument("--b39-checkpoint", type=Path, required=True)
    parser.add_argument("--b39-summary", type=Path, required=True)
    parser.add_argument("--b39-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--b40-summary", type=Path, required=True)
    parser.add_argument("--b40-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--b41-checkpoint", type=Path, required=True)
    parser.add_argument("--b41-summary", type=Path, required=True)
    parser.add_argument("--b41-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--valid-terminal-summary", type=Path, required=True)
    parser.add_argument("--valid-terminal-candidates", type=Path, required=True)
    parser.add_argument("--state-guidance-summary", type=Path, required=True)
    parser.add_argument("--trajectory-dataset", type=Path, required=True)
    parser.add_argument("--sft-adapter-dir", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "arms": list(ARMS),
        "fit_property_counts": [2],
        "composition_diagnostic_property_counts": [3],
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "state_dependent_transport_adapter": True,
        "paired_flow_matching_supervision": True,
        "terminal_reachability_is_auxiliary_training_loss": True,
        "inference_classifier_gradient_guidance": False,
        "generation_seed_matches_valid_terminal_baseline": True,
        "common_llm_emits_text_or_actions": False,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "official_test_access": False,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Language-grounded flow preregistration drift: {drift}")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Language-grounded flow implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    expected_locks = {
        "valid_terminal_summary_sha256",
        "valid_terminal_candidates_sha256",
        "state_guidance_summary_sha256",
        "trajectory_dataset_sha256",
        "sft_adapter_config_sha256",
        "sft_adapter_model_sha256",
    }
    if set(dict(payload.get("locked_signal_inputs", {}))) != expected_locks:
        raise ValueError("Language-grounded flow locked inputs are incomplete")
    return payload


def check_signal_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    locks = dict(preregistration["locked_signal_inputs"])
    paths = {
        "valid_terminal_summary_sha256": args.valid_terminal_summary,
        "valid_terminal_candidates_sha256": args.valid_terminal_candidates,
        "state_guidance_summary_sha256": args.state_guidance_summary,
        "trajectory_dataset_sha256": args.trajectory_dataset,
        "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
        "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing language-grounded flow inputs: {missing}")
    actual = {name: belief.file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks[name], "actual": actual[name]}
        for name in paths
        if actual[name] != locks[name]
    }
    if drift:
        raise ValueError(f"Language-grounded flow locked-input drift: {drift}")
    baseline = read_json(args.valid_terminal_summary)
    predecessor = read_json(args.state_guidance_summary)
    if baseline.get("protocol") != valid_terminal.PROTOCOL:
        raise ValueError("Language-grounded flow requires the valid-terminal baseline")
    if predecessor.get("protocol") != state_guidance.PROTOCOL:
        raise ValueError("Language-grounded flow requires the state-guidance predecessor")
    if predecessor.get("decision") != "stop_state_viability_guidance_without_gate_changes":
        raise ValueError("Language-grounded flow refuses predecessor decision drift")
    return baseline, predecessor


class LanguageGroundedTransportAdapter(nn.Module):
    """Query constraint memory from the current latent and emit a velocity residual."""

    def __init__(
        self,
        latent_dim: int,
        source_dim: int,
        memory_dim: int,
        hidden_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.state = nn.Sequential(
            nn.LayerNorm(latent_dim + source_dim + 3),
            nn.Linear(latent_dim + source_dim + 3, hidden_dim),
            nn.SiLU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.memory = nn.Sequential(
            nn.LayerNorm(memory_dim), nn.Linear(memory_dim, hidden_dim)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.velocity = nn.Linear(hidden_dim, latent_dim)
        self.viability = nn.Linear(hidden_dim, 1)
        self.edit_projection = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

    def forward(
        self,
        latent: torch.Tensor,
        source_pool: torch.Tensor,
        flow_time: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = torch.stack(
            [
                flow_time,
                torch.sin(math.pi * flow_time),
                torch.cos(math.pi * flow_time),
            ],
            dim=1,
        )
        state = self.state(
            torch.cat([latent.float(), source_pool.float(), phase.float()], dim=1)
        )
        values = self.memory(memory.float())
        attention_logits = (
            self.query(state).unsqueeze(1) * values
        ).sum(dim=-1) / math.sqrt(values.shape[-1])
        attention_logits = attention_logits.masked_fill(
            ~memory_mask.bool(), -torch.inf
        )
        context = (
            torch.softmax(attention_logits, dim=1).unsqueeze(-1) * values
        ).sum(dim=1)
        fused = self.fusion(torch.cat([state, context], dim=1))
        residual = self.residual_scale * torch.tanh(self.velocity(fused))
        return residual, self.viability(fused).squeeze(1), self.edit_projection(context)


def equalize_memory_dimension(
    memory: torch.Tensor, preregistration: Mapping[str, object]
) -> torch.Tensor:
    """Use a fixed non-trainable projection so both arms have equal capacity."""

    target_dim = int(preregistration["memory_adapter_dim"])
    if int(memory.shape[-1]) == target_dim:
        return memory.to(torch.float16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(preregistration["llm_memory_projection_seed"]))
    projection = torch.randn(
        int(memory.shape[-1]), target_dim, generator=generator, dtype=torch.float32
    ) / math.sqrt(target_dim)
    return (memory.float() @ projection).to(torch.float16)


def pooled_source_node(
    source_node: torch.Tensor, source_mask: torch.Tensor
) -> torch.Tensor:
    mask = source_mask.unsqueeze(-1).to(source_node.dtype)
    return (source_node * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def flow_matching_batch(
    adapter: LanguageGroundedTransportAdapter,
    model: nn.Module,
    representation: nn.Module,
    items: Sequence[object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    collated = base.pair_collate(items)
    source = base.move_graph_batch(collated["source"], device)
    target = base.move_graph_batch(collated["target"], device)
    condition_tokens = collated["condition"].to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, source_edge = representation.encode(source)
        target_node, target_edge = representation.encode(target)
        condition = model.route_condition(condition_tokens)
        endpoint = model.posterior_endpoint(
            source,
            target,
            source_node,
            source_edge,
            target_node,
            target_edge,
            condition,
        ).float()
        noise = torch.randn_like(endpoint)
        flow_time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
        current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
        base_velocity = model.transport_velocity(
            current,
            flow_time.to(source_node.dtype),
            source_node,
            source["node_mask"],
            condition_tokens,
        ).float()
        source_pool = pooled_source_node(source_node, source["node_mask"])
        target_velocity = endpoint - noise
    residual, viability_logit, edit_code = adapter(
        current,
        source_pool,
        flow_time,
        memories.to(device),
        memory_masks.to(device),
    )
    adapted_velocity = base_velocity + residual
    flow_loss = F.mse_loss(adapted_velocity, target_velocity)
    base_flow_loss = F.mse_loss(base_velocity, target_velocity)
    edit_code = F.normalize(edit_code, dim=1)
    target_code = F.normalize(target_velocity.detach(), dim=1)
    contrastive_logits = edit_code @ target_code.transpose(0, 1)
    contrastive_logits = contrastive_logits / float(
        preregistration["contrastive_temperature"]
    )
    labels = torch.arange(len(items), device=device)
    contrastive_loss = 0.5 * (
        F.cross_entropy(contrastive_logits, labels)
        + F.cross_entropy(contrastive_logits.transpose(0, 1), labels)
    )
    positive_viability = F.binary_cross_entropy_with_logits(
        viability_logit, torch.ones_like(viability_logit)
    )
    residual_penalty = residual.square().mean()
    loss = (
        flow_loss
        + float(preregistration["contrastive_loss_weight"]) * contrastive_loss
        + float(preregistration["positive_viability_loss_weight"])
        * positive_viability
        + float(preregistration["residual_penalty_weight"]) * residual_penalty
    )
    values = {
        "flow_loss": float(flow_loss.detach()),
        "base_flow_loss": float(base_flow_loss.detach()),
        "contrastive_loss": float(contrastive_loss.detach()),
        "positive_viability_loss": float(positive_viability.detach()),
        "residual_penalty": float(residual_penalty.detach()),
        "residual_norm": float(residual.detach().norm(dim=1).mean()),
    }
    return loss, values


def terminal_auxiliary_batch(
    adapter: LanguageGroundedTransportAdapter,
    dataset: Mapping[str, object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    rows: Sequence[tuple[int, int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent, source_pool, flow_time, memory, mask, labels = state_guidance.batch_from_rows(
        dataset, memories, memory_masks, rows, device
    )
    _residual, logits, _edit_code = adapter(
        latent, source_pool, flow_time, memory, mask
    )
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss, {
        "terminal_loss": float(loss.detach()),
        "terminal_accuracy": float((logits.detach().ge(0) == labels.bool()).float().mean()),
    }


def balanced_terminal_rows(
    dataset: Mapping[str, object], condition_indices: Sequence[int]
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    rows = state_guidance.state_rows(dataset, condition_indices)
    positive = [
        row for row in rows if bool(dataset["terminal_labels"][row[0], row[2]])
    ]
    negative = [
        row for row in rows if not bool(dataset["terminal_labels"][row[0], row[2]])
    ]
    if not positive or not negative:
        raise ValueError("Transport adapter auxiliary labels require both classes")
    return positive, negative


def train_adapter(
    adapter: LanguageGroundedTransportAdapter,
    model: nn.Module,
    representation: nn.Module,
    trajectory_pairs: Sequence[object],
    dataset: Mapping[str, object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    train_conditions = list(dataset["critic_train_condition_indices"])
    positive_rows, negative_rows = balanced_terminal_rows(dataset, train_conditions)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(preregistration["adapter_learning_rate"]),
        weight_decay=float(preregistration["adapter_weight_decay"]),
    )
    batch_size = int(preregistration["adapter_batch_size"])
    terminal_batch_size = int(preregistration["terminal_batch_size"])
    history: list[dict[str, float]] = []
    model.eval().requires_grad_(False)
    representation.eval().requires_grad_(False)
    for epoch in range(1, int(preregistration["adapter_epochs"]) + 1):
        rng = random.Random(int(preregistration["adapter_training_seed"]) + epoch)
        order = train_conditions.copy()
        rng.shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        adapter.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            items = [trajectory_pairs[index] for index in indices]
            flow_loss, values = flow_matching_batch(
                adapter,
                model,
                representation,
                items,
                memories[indices],
                memory_masks[indices],
                preregistration,
                device,
            )
            half = max(1, terminal_batch_size // 2)
            terminal_rows = [
                *rng.choices(positive_rows, k=half),
                *rng.choices(negative_rows, k=half),
            ]
            rng.shuffle(terminal_rows)
            terminal_loss, terminal_values = terminal_auxiliary_batch(
                adapter,
                dataset,
                memories,
                memory_masks,
                terminal_rows,
                device,
            )
            loss = flow_loss + float(
                preregistration["terminal_auxiliary_loss_weight"]
            ) * terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                adapter.parameters(), float(preregistration["adapter_grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in {**values, **terminal_values}.items():
                totals[name] += value
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite language-grounded adapter metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "adapter_epoch", **row}, sort_keys=True), flush=True)
    adapter.eval()
    return history


@torch.no_grad()
def validate_adapter(
    adapter: LanguageGroundedTransportAdapter,
    model: nn.Module,
    representation: nn.Module,
    trajectory_pairs: Sequence[object],
    dataset: Mapping[str, object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    indices = list(dataset["critic_validation_condition_indices"])
    batch_size = int(preregistration["adapter_batch_size"])
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    adapter.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        _loss, values = flow_matching_batch(
            adapter,
            model,
            representation,
            [trajectory_pairs[index] for index in selected],
            memories[selected],
            memory_masks[selected],
            preregistration,
            device,
        )
        for name, value in values.items():
            totals[name] += value
        batches += 1
    validation_rows = state_guidance.state_rows(dataset, indices)
    scores: list[float] = []
    labels: list[bool] = []
    terminal_batch_size = int(preregistration["terminal_batch_size"])
    for start in range(0, len(validation_rows), terminal_batch_size):
        rows = validation_rows[start : start + terminal_batch_size]
        latent, source_pool, flow_time, memory, mask, target = state_guidance.batch_from_rows(
            dataset, memories, memory_masks, rows, device
        )
        _residual, logits, _edit = adapter(
            latent, source_pool, flow_time, memory, mask
        )
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(target.bool().cpu().tolist())
    labels_np = np.asarray(labels, dtype=bool)
    scores_np = np.asarray(scores, dtype=np.float64)
    flow_loss = totals["flow_loss"] / max(1, batches)
    base_flow_loss = totals["base_flow_loss"] / max(1, batches)
    return {
        **{name: value / max(1, batches) for name, value in totals.items()},
        "relative_flow_mse_reduction": (base_flow_loss - flow_loss)
        / max(1e-12, base_flow_loss),
        "terminal_state_auc_diagnostic": state_guidance.binary_auc(
            labels_np, scores_np
        ),
        "independent_negative_terminal_particles": int(
            dataset["validation_terminal_class_counts"]["negative"]
        ),
        "independent_positive_terminal_particles": int(
            dataset["validation_terminal_class_counts"]["positive"]
        ),
    }


def adapted_transport_particles(
    model: nn.Module,
    representation: nn.Module,
    source_example: object,
    condition_tokens: np.ndarray,
    particles: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
    *,
    adapter: LanguageGroundedTransportAdapter,
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    diagnostics: defaultdict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    attempts = particles.shape[0]
    chunk = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = torch.from_numpy(
        np.repeat(condition_tokens[None, ...], attempts, axis=0)
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, _ = representation.encode(source)
    source_pool = pooled_source_node(source_node, source["node_mask"]).float()
    memory = memory.to(device).float().unsqueeze(0).expand(attempts, -1, -1)
    memory_mask = memory_mask.to(device).bool().unsqueeze(0).expand(attempts, -1)
    minimum_rms = math.sqrt(model.transport_dim) * float(
        preregistration["latent_min_std"]
    )
    latent = particles.float()
    minimum_observed_rms = math.inf
    for flow_index in range(int(preregistration["flow_steps"])):
        velocities: list[torch.Tensor] = []
        for start in range(0, attempts, chunk):
            count = min(chunk, attempts - start)
            flow_time = torch.full(
                (count,),
                (flow_index + 0.5) / int(preregistration["flow_steps"]),
                device=device,
                dtype=source_node.dtype,
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                base_velocity = model.transport_velocity(
                    latent[start : start + count],
                    flow_time,
                    source_node.expand(count, -1, -1),
                    source["node_mask"].expand(count, -1),
                    tokens[start : start + count],
                ).float()
            residual, _viability, _edit = adapter(
                latent[start : start + count],
                source_pool.expand(count, -1),
                flow_time.float(),
                memory[start : start + count],
                memory_mask[start : start + count],
            )
            diagnostics["residual_norm_sum"] += float(
                residual.norm(dim=1).mean().detach().cpu()
            )
            diagnostics["residual_steps"] += 1.0
            velocities.append(base_velocity + residual.float())
        proposal = latent + torch.cat(velocities, dim=0) / int(
            preregistration["flow_steps"]
        )
        center = proposal.mean(dim=0, keepdim=True)
        residual = proposal - center
        normalized = F.normalize(residual, dim=1)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(-torch.inf)
        neighbours = torch.softmax(
            similarity / float(preregistration["particle_repulsion_temperature"]),
            dim=1,
        )
        repulsion = normalized - neighbours @ normalized
        proposal = proposal + (
            float(preregistration["particle_repulsion_strength"])
            / int(preregistration["flow_steps"])
        ) * minimum_rms * repulsion
        center = proposal.mean(dim=0, keepdim=True)
        residual = proposal - center
        rms = residual.norm(dim=1).square().mean().sqrt().clamp_min(1e-8)
        minimum_observed_rms = min(minimum_observed_rms, float(rms.detach().cpu()))
        if float(rms.detach().cpu()) < minimum_rms:
            residual = residual * (minimum_rms / rms)
        latent = (center + residual).detach()
    normalized = F.normalize(latent, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, dtype=torch.bool, device=device)
    return latent, {
        "final_particle_mean_abs_cosine": float(
            cosine[off_diagonal].abs().mean().detach().cpu()
        ),
        "final_particle_max_abs_cosine": float(
            cosine[off_diagonal].abs().max().detach().cpu()
        ),
        "final_particle_centered_rms": float(
            (latent - latent.mean(dim=0, keepdim=True))
            .norm(dim=1)
            .square()
            .mean()
            .sqrt()
            .detach()
            .cpu()
        ),
        "minimum_transport_particle_rms": minimum_observed_rms,
    }


def freeze_adapted_candidates(
    model: nn.Module,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    pairs: Sequence[object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    adapter: LanguageGroundedTransportAdapter,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, object]]:
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    original_transport = b41.interacting_transport_particles
    rows: list[dict[str, object]] = []
    diagnostics: defaultdict[str, float] = defaultdict(float)
    try:
        b41.viability_event_mask = exact_support
        for pair_index, pair in enumerate(pairs):
            def adapted(*transport_args, **transport_kwargs):
                return adapted_transport_particles(
                    *transport_args,
                    **transport_kwargs,
                    adapter=adapter,
                    memory=memories[pair_index],
                    memory_mask=memory_masks[pair_index],
                    diagnostics=diagnostics,
                )

            b41.interacting_transport_particles = adapted
            generated = b41.sample_from_source(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
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
                            "stage": "freeze_language_grounded_candidates",
                            "conditions": pair_index + 1,
                            "raw_rows": len(rows),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.interacting_transport_particles = original_transport
        b41.viability_event_mask = original_mask
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    if len(rows) != expected:
        raise RuntimeError(f"Language-grounded freeze expected {expected}, found {len(rows)}")
    transport = {
        "mean_adapter_residual_norm": diagnostics["residual_norm_sum"]
        / max(1.0, diagnostics["residual_steps"]),
        "adapter_residual_steps": int(diagnostics["residual_steps"]),
    }
    return rows, transport, exact_support.manifest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["adapter_training_seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed language-grounded arm exists: {summary_path}")
    baseline, predecessor = check_signal_inputs(args, preregistration)
    stack = state_guidance.load_frozen_stack(args, preregistration, device)
    (
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        trajectory_pairs,
        development_pairs,
        lineage,
    ) = stack
    dataset = torch.load(args.trajectory_dataset, map_location="cpu", weights_only=False)
    if dataset.get("protocol") != state_guidance.PROTOCOL:
        raise ValueError("Language-grounded flow trajectory protocol drift")
    if dataset["pair_keys"] != [state_guidance.pair_key(pair) for pair in trajectory_pairs]:
        raise ValueError("Language-grounded flow trajectory pair drift")
    fit_memories, fit_masks, llm_manifest = state_guidance.development_memories(
        args.arm, trajectory_pairs, preregistration, args, device
    )
    fit_memories = equalize_memory_dimension(fit_memories, preregistration)
    base.seed_everything(int(preregistration["adapter_initialization_seed"]))
    adapter = LanguageGroundedTransportAdapter(
        latent_dim=int(dataset["traces"].shape[-1]),
        source_dim=int(dataset["source_pool"].shape[-1]),
        memory_dim=int(fit_memories.shape[-1]),
        hidden_dim=int(preregistration["adapter_hidden_dim"]),
        residual_scale=float(preregistration["adapter_residual_scale"]),
    ).to(device)
    training = train_adapter(
        adapter,
        model,
        representation,
        trajectory_pairs,
        dataset,
        fit_memories,
        fit_masks,
        preregistration,
        device,
    )
    base.seed_everything(int(preregistration["fit_validation_seed"]))
    validation = validate_adapter(
        adapter,
        model,
        representation,
        trajectory_pairs,
        dataset,
        fit_memories,
        fit_masks,
        preregistration,
        device,
    )
    checkpoint_path = args.output_dir / "language_grounded_transport_adapter.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": args.arm,
            "state_dict": adapter.state_dict(),
            "latent_dim": int(dataset["traces"].shape[-1]),
            "source_dim": int(dataset["source_pool"].shape[-1]),
            "memory_dim": int(fit_memories.shape[-1]),
            "hidden_dim": int(preregistration["adapter_hidden_dim"]),
            "residual_scale": float(preregistration["adapter_residual_scale"]),
        },
        checkpoint_path,
    )
    development_memory, development_mask, _ = state_guidance.development_memories(
        args.arm, development_pairs, preregistration, args, device
    )
    development_memory = equalize_memory_dimension(
        development_memory, preregistration
    )
    frozen, transport_metrics, support_manifest = freeze_adapted_candidates(
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        development_pairs,
        development_memory,
        development_mask,
        adapter,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, frozen)
    evaluated, metrics = b41.evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    metrics = dict(metrics)
    metrics["by_property_count_diagnostic"] = operator.property_count_diagnostics(
        evaluated, metrics
    )
    manifest = {
        **lineage,
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "trajectory_dataset_sha256": belief.file_sha256(args.trajectory_dataset),
        "adapter_checkpoint_sha256": belief.file_sha256(checkpoint_path),
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "exact_molecule_stop_support": support_manifest,
        "common_llm": llm_manifest,
        "memory_adapter_dim": int(preregistration["memory_adapter_dim"]),
        "fixed_non_trainable_llm_memory_projection": args.arm
        == "common_llm_memory",
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "state_dependent_transport_adapter": True,
        "paired_flow_matching_supervision": True,
        "terminal_reachability_is_auxiliary_training_loss": True,
        "inference_classifier_gradient_guidance": False,
        "fit_property_counts": [2],
        "composition_diagnostic_property_counts": [3],
        "development_is_reused_method_development_split": True,
        "development_is_formal_fresh_ood": False,
        "common_llm_emits_text_or_actions": False,
        "frozen_before_target_or_property_evaluation": True,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
        "valid_terminal_decision": baseline.get("decision"),
        "state_guidance_decision": predecessor.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "arm": args.arm,
        "decision": "await_cross_arm_language_grounded_flow_gate",
        "training": training,
        "fit_validation": validation,
        "transport_metrics": transport_metrics,
        "metrics": metrics,
        "manifest": manifest,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
