#!/usr/bin/env python3
"""Guide a frozen molecular latent flow with a state-dependent Common-LLM critic.

This is a mechanism experiment, not a candidate selector.  A fit-only rollout
stage records the frozen B41 latent particles at every flow time and whether
each particle later reaches an exact, materializable molecule STOP.  Two
identical critics are then trained from those traces:

* ``property_memory`` queries the explicit property-slot tokens;
* ``common_llm_memory`` queries compressed hidden tokens from the frozen
  Common-LLM SFT model.

At generation time the current latent state queries the frozen memory and the
gradient of log terminal reachability is added to the *continuous* B41 vector
field.  The event decoder, STOP support, twenty raw attempts, and post-freeze
evaluation remain unchanged.  No LLM text/action, molecule pool, ranking,
retry, target, or property oracle is available to generation.
"""

from __future__ import annotations

import argparse
import copy
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

import llm_latent_operator_signal as operator  # noqa: E402


valid_terminal = operator.valid_terminal
b41 = operator.b41
b40 = operator.b40
b39 = operator.b39
b37 = operator.b37
b36 = operator.b36
base = operator.base
belief = operator.belief
delta = operator.delta
graph = operator.graph
hierarchical = operator.hierarchical
unified = operator.unified
full_graph = operator.full_graph

PROTOCOL = "train_only_common_llm_state_viability_guidance_v1"
STAGES = ("prepare", "property_memory", "common_llm_memory")
ARMS = ("property_memory", "common_llm_memory")
VALID_TERMINAL_PREREGISTRATION = (
    SCRIPT_DIR / "valid_terminal_molecule_latent_jump_v1_preregistration.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
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
    parser.add_argument("--operator-summary", type=Path, required=True)
    parser.add_argument("--sft-adapter-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dataset", type=Path, required=True)
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
        "stages": list(STAGES),
        "arms": list(ARMS),
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "fit_property_counts": [2],
        "composition_diagnostic_property_counts": [3],
        "common_llm_emits_text_or_actions": False,
        "current_latent_queries_constraint_memory_each_flow_step": True,
        "terminal_reachability_gradient_guides_latent_vector_field": True,
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
        raise ValueError(f"State-viability preregistration drift: {drift}")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "State-viability implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    expected_locks = {
        "valid_terminal_summary_sha256",
        "valid_terminal_candidates_sha256",
        "operator_summary_sha256",
        "sft_adapter_config_sha256",
        "sft_adapter_model_sha256",
    }
    if set(dict(payload.get("locked_signal_inputs", {}))) != expected_locks:
        raise ValueError("State-viability locked signal inputs are incomplete")
    return payload


def check_signal_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    locks = dict(preregistration["locked_signal_inputs"])
    paths = {
        "valid_terminal_summary_sha256": args.valid_terminal_summary,
        "valid_terminal_candidates_sha256": args.valid_terminal_candidates,
        "operator_summary_sha256": args.operator_summary,
        "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
        "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing state-viability inputs: {missing}")
    actual_hashes = {name: belief.file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks[name], "actual": actual_hashes[name]}
        for name in paths
        if actual_hashes[name] != locks[name]
    }
    if drift:
        raise ValueError(f"State-viability locked input drift: {drift}")
    valid_summary = read_json(args.valid_terminal_summary)
    operator_summary = read_json(args.operator_summary)
    if valid_summary.get("protocol") != valid_terminal.PROTOCOL:
        raise ValueError("State viability requires the locked valid-terminal baseline")
    if operator_summary.get("protocol") != operator.PROTOCOL:
        raise ValueError("State viability requires the locked latent-operator result")
    if operator_summary.get("decision") != "stop_common_llm_latent_operator_signal_without_gate_changes":
        raise ValueError("State viability refuses latent-operator decision drift")
    return valid_summary, operator_summary


def pair_key(pair: object) -> str:
    payload = f"{pair.source_smiles}|{base.task_key(pair.row)}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_trajectory_pairs(
    fit_pairs: Sequence[object], preregistration: Mapping[str, object]
) -> list[object]:
    eligible = [
        pair
        for pair in fit_pairs
        if int(pair.property_count) in set(preregistration["fit_property_counts"])
    ]
    seed = int(preregistration["trajectory_selection_seed"])
    eligible.sort(key=lambda pair: hashlib.sha256(f"{seed}|{pair_key(pair)}".encode()).hexdigest())
    limit = int(preregistration["trajectory_condition_limit"])
    selected = eligible[:limit]
    if len(selected) != limit:
        raise ValueError(f"Expected {limit} fit-only trajectory conditions, found {len(selected)}")
    return selected


def split_trajectory_conditions(
    pairs: Sequence[object], preregistration: Mapping[str, object]
) -> tuple[list[int], list[int]]:
    seed = int(preregistration["critic_validation_seed"])
    order = sorted(
        range(len(pairs)),
        key=lambda index: hashlib.sha256(f"{seed}|{pair_key(pairs[index])}".encode()).hexdigest(),
    )
    validation_count = int(round(len(pairs) * float(preregistration["critic_validation_fraction"])))
    validation = sorted(order[:validation_count])
    training = sorted(order[validation_count:])
    if set(training) & set(validation):
        raise RuntimeError("Critic trajectory split overlap")
    return training, validation


def load_frozen_stack(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
):
    valid_prereg = valid_terminal.read_preregistration(VALID_TERMINAL_PREREGISTRATION)
    b22_summary, b22_checkpoint, b36_summary, b37_summary, b41_checkpoint = (
        valid_terminal.check_locked_inputs(args, valid_prereg)
    )
    selected_pairs, reconstruction = b36.reconstruct_b22_train_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    for pair in [*fit_pairs, *development_pairs]:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(preregistration["condition_dim"])
        )
    trajectory_pairs = select_trajectory_pairs(fit_pairs, preregistration)
    if {pair_key(pair) for pair in trajectory_pairs} & {
        pair_key(pair) for pair in development_pairs
    }:
        raise ValueError("Trajectory/development pair overlap")
    representation, representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(preregistration["condition_dim"]),
        transport_dim=int(preregistration["transport_dim"]),
        hidden_dim=int(preregistration["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(preregistration["max_jumps"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(preregistration["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)
    representation.eval().requires_grad_(False)
    lineage = {
        "reconstruction": reconstruction,
        "split": split,
        "representation_protocol": representation_summary.get("protocol"),
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
    }
    return (
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        trajectory_pairs,
        development_pairs,
        lineage,
    )


def pooled_source(
    representation: nn.Module, source_example: object, device: torch.device
) -> torch.Tensor:
    source = base.move_graph_batch(graph.collate([source_example]), device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, _ = representation.encode(source)
    mask = source["node_mask"].unsqueeze(-1).to(source_node.dtype)
    return ((source_node * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).float()[0]


@torch.no_grad()
def traced_transport_particles(
    model: nn.Module,
    representation: nn.Module,
    source_example: object,
    condition_tokens: np.ndarray,
    particles: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
    capture: dict[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    attempts = particles.shape[0]
    chunk = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = torch.from_numpy(np.repeat(condition_tokens[None, ...], attempts, axis=0)).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        source_node, _ = representation.encode(source)
    minimum_rms = math.sqrt(model.transport_dim) * float(preregistration["latent_min_std"])
    latent = particles.float()
    traces = [latent.detach().cpu().to(torch.float16)]
    minimum_observed_rms = math.inf
    for flow_index in range(int(preregistration["flow_steps"])):
        velocities = []
        for start in range(0, attempts, chunk):
            count = min(chunk, attempts - start)
            flow_time = torch.full(
                (count,),
                (flow_index + 0.5) / int(preregistration["flow_steps"]),
                device=device,
                dtype=source_node.dtype,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                velocity = model.transport_velocity(
                    latent[start : start + count],
                    flow_time,
                    source_node.expand(count, -1, -1),
                    source["node_mask"].expand(count, -1),
                    tokens[start : start + count],
                )
            velocities.append(velocity.float())
        proposal = latent + torch.cat(velocities, dim=0) / int(preregistration["flow_steps"])
        center = proposal.mean(dim=0, keepdim=True)
        residual = proposal - center
        normalized = F.normalize(residual, dim=1)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(-torch.inf)
        neighbours = torch.softmax(
            similarity / float(preregistration["particle_repulsion_temperature"]), dim=1
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
        latent = center + residual
        traces.append(latent.detach().cpu().to(torch.float16))
    capture["traces"] = torch.stack(traces, dim=0)
    capture["source_pool"] = pooled_source(representation, source_example, device).cpu().to(torch.float16)
    normalized = F.normalize(latent, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, dtype=torch.bool, device=device)
    return latent, {
        "final_particle_mean_abs_cosine": float(cosine[off_diagonal].abs().mean().cpu()),
        "final_particle_max_abs_cosine": float(cosine[off_diagonal].abs().max().cpu()),
        "final_particle_centered_rms": float(
            (latent - latent.mean(dim=0, keepdim=True)).norm(dim=1).square().mean().sqrt().cpu()
        ),
        "minimum_transport_particle_rms": minimum_observed_rms,
    }


def collect_trajectory_dataset(
    model: nn.Module,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    trajectory_pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    original_transport = b41.interacting_transport_particles
    traces: list[torch.Tensor] = []
    source_pools: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    try:
        b41.viability_event_mask = exact_support
        for index, pair in enumerate(trajectory_pairs):
            capture: dict[str, object] = {}

            def traced(*transport_args, **transport_kwargs):
                return traced_transport_particles(
                    *transport_args, **transport_kwargs, capture=capture
                )

            b41.interacting_transport_particles = traced
            outputs = b41.sample_from_source(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
                pair.source,
                np.asarray(pair.condition),
                preregistration,
                device,
                int(preregistration["trajectory_generation_seed"]) * 100000 + index,
            )
            traces.append(capture["traces"])
            source_pools.append(capture["source_pool"])
            outputs.sort(key=lambda row: int(row["particle_index"]))
            labels.append(
                torch.as_tensor(
                    [
                        bool(row["stopped_by_model"])
                        and not bool(row["max_horizon_hit"])
                        and bool(str(row["generated_smiles"]))
                        for row in outputs
                    ],
                    dtype=torch.bool,
                )
            )
            if (index + 1) % 16 == 0 or index + 1 == len(trajectory_pairs):
                print(
                    json.dumps(
                        {
                            "stage": "fit_only_trajectory_collection",
                            "conditions": index + 1,
                            "terminal_particles": (index + 1)
                            * int(preregistration["exact_raw_attempts_per_condition"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.interacting_transport_particles = original_transport
        b41.viability_event_mask = original_mask
    training, validation = split_trajectory_conditions(trajectory_pairs, preregistration)
    label_tensor = torch.stack(labels)
    validation_labels = label_tensor[validation].reshape(-1)
    terminal_counts = {
        "positive": int(validation_labels.sum()),
        "negative": int((~validation_labels).sum()),
    }
    return {
        "protocol": PROTOCOL,
        "traces": torch.stack(traces),
        "source_pool": torch.stack(source_pools),
        "property_memory": torch.from_numpy(
            np.stack([np.asarray(pair.condition, dtype=np.float32) for pair in trajectory_pairs])
        ).to(torch.float16),
        "property_memory_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray([True, *[bool(np.any(slot)) for slot in pair.condition[1:]]])
                    for pair in trajectory_pairs
                ]
            )
        ).bool(),
        "terminal_labels": label_tensor,
        "pair_keys": [pair_key(pair) for pair in trajectory_pairs],
        "critic_train_condition_indices": training,
        "critic_validation_condition_indices": validation,
        "validation_terminal_class_counts": terminal_counts,
        "exact_stop_support": exact_support.manifest(),
    }


@torch.no_grad()
def compressed_llm_token_memories(
    model: object,
    tokenizer: object,
    pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    model.eval().requires_grad_(False)
    output: list[torch.Tensor] = []
    batch_size = int(preregistration["llm_embedding_batch_size"])
    slots = int(preregistration["llm_memory_slots"])
    for start in range(0, len(pairs), batch_size):
        items = pairs[start : start + batch_size]
        batch = operator.prompt_batch(
            tokenizer, items, int(preregistration["llm_max_length"]), device
        )
        hidden = operator._transformer_body(model)(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()
        for row, mask in zip(hidden, batch["attention_mask"], strict=True):
            valid = row[mask.bool()]
            pooled = F.adaptive_avg_pool1d(valid.transpose(0, 1).unsqueeze(0), slots)
            output.append(pooled.squeeze(0).transpose(0, 1).cpu().to(torch.float16))
        print(
            json.dumps(
                {
                    "stage": "frozen_common_llm_token_memory",
                    "conditions": min(start + batch_size, len(pairs)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return torch.stack(output)


class StateViabilityCritic(nn.Module):
    """A state query over frozen constraint memory, producing one reachability logit."""

    def __init__(
        self, latent_dim: int, source_dim: int, memory_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.state = nn.Sequential(
            nn.LayerNorm(latent_dim + source_dim + 3),
            nn.Linear(latent_dim + source_dim + 3, hidden_dim),
            nn.SiLU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.memory = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, hidden_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        source_pool: torch.Tensor,
        flow_time: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.stack(
            [
                flow_time,
                torch.sin(math.pi * flow_time),
                torch.cos(math.pi * flow_time),
            ],
            dim=1,
        )
        state = self.state(torch.cat([latent.float(), source_pool.float(), phase.float()], dim=1))
        query = self.query(state).unsqueeze(1)
        values = self.memory(memory.float())
        logits = (query * values).sum(dim=-1) / math.sqrt(values.shape[-1])
        logits = logits.masked_fill(~memory_mask.bool(), -torch.inf)
        attended = torch.softmax(logits, dim=1).unsqueeze(-1).mul(values).sum(dim=1)
        return self.output(torch.cat([state, attended], dim=1)).squeeze(1)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def state_rows(dataset: Mapping[str, object], condition_indices: Sequence[int]):
    trace = dataset["traces"]
    time_count, particle_count = int(trace.shape[1]), int(trace.shape[2])
    rows = [
        (condition, time_index, particle)
        for condition in condition_indices
        for time_index in range(time_count)
        for particle in range(particle_count)
    ]
    return rows


def batch_from_rows(
    dataset: Mapping[str, object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    rows: Sequence[tuple[int, int, int]],
    device: torch.device,
):
    condition = torch.as_tensor([row[0] for row in rows], dtype=torch.long)
    time_index = torch.as_tensor([row[1] for row in rows], dtype=torch.long)
    particle = torch.as_tensor([row[2] for row in rows], dtype=torch.long)
    denominator = max(1, int(dataset["traces"].shape[1]) - 1)
    return (
        dataset["traces"][condition, time_index, particle].to(device).float(),
        dataset["source_pool"][condition].to(device).float(),
        (time_index.float() / denominator).to(device),
        memories[condition].to(device).float(),
        memory_masks[condition].to(device),
        dataset["terminal_labels"][condition, particle].to(device).float(),
    )


def train_critic(
    dataset: Mapping[str, object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[StateViabilityCritic, list[dict[str, float]], dict[str, object]]:
    critic = StateViabilityCritic(
        int(dataset["traces"].shape[-1]),
        int(dataset["source_pool"].shape[-1]),
        int(memories.shape[-1]),
        int(preregistration["critic_hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(preregistration["critic_learning_rate"]),
        weight_decay=float(preregistration["critic_weight_decay"]),
    )
    train_rows = state_rows(dataset, dataset["critic_train_condition_indices"])
    positive = [row for row in train_rows if bool(dataset["terminal_labels"][row[0], row[2]])]
    negative = [row for row in train_rows if not bool(dataset["terminal_labels"][row[0], row[2]])]
    if not positive or not negative:
        raise ValueError("Fit-only critic labels do not contain both feasibility classes")
    batch_size = int(preregistration["critic_batch_size"])
    history: list[dict[str, float]] = []
    for epoch in range(1, int(preregistration["critic_epochs"]) + 1):
        rng = random.Random(int(preregistration["critic_training_seed"]) + epoch)
        sampled_positive = rng.sample(positive, min(len(positive), len(negative) * 2))
        balanced = [*negative, *sampled_positive]
        rng.shuffle(balanced)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        critic.train()
        for start in range(0, len(balanced), batch_size):
            batch_rows = balanced[start : start + batch_size]
            latent, source_pool, flow_time, memory, mask, labels = batch_from_rows(
                dataset, memories, memory_masks, batch_rows, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = critic(latent, source_pool, flow_time, memory, mask)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), float(preregistration["critic_grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["accuracy"] += float((logits.detach().ge(0) == labels.bool()).float().mean())
            batches += 1
        row = {"epoch": epoch, **{name: value / max(1, batches) for name, value in totals.items()}}
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite critic training metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "critic_epoch", **row}, sort_keys=True), flush=True)
    validation_rows = state_rows(dataset, dataset["critic_validation_condition_indices"])
    all_scores: list[float] = []
    all_labels: list[bool] = []
    critic.eval()
    with torch.no_grad():
        for start in range(0, len(validation_rows), batch_size):
            rows = validation_rows[start : start + batch_size]
            latent, source_pool, flow_time, memory, mask, labels = batch_from_rows(
                dataset, memories, memory_masks, rows, device
            )
            probabilities = torch.sigmoid(critic(latent, source_pool, flow_time, memory, mask))
            all_scores.extend(probabilities.cpu().tolist())
            all_labels.extend(labels.bool().cpu().tolist())
    labels_np = np.asarray(all_labels, dtype=bool)
    scores_np = np.asarray(all_scores, dtype=np.float64)
    diagnostics = {
        "state_auc": binary_auc(labels_np, scores_np),
        "state_balanced_accuracy": 0.5
        * (
            float((scores_np[labels_np] >= 0.5).mean())
            + float((scores_np[~labels_np] < 0.5).mean())
        ),
        "state_brier": float(np.mean((scores_np - labels_np.astype(float)) ** 2)),
        "validation_state_rows": len(validation_rows),
        "validation_terminal_class_counts": dict(dataset["validation_terminal_class_counts"]),
    }
    critic.eval().requires_grad_(False)
    return critic, history, diagnostics


def guided_transport_particles(
    model: nn.Module,
    representation: nn.Module,
    source_example: object,
    condition_tokens: np.ndarray,
    particles: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
    *,
    critic: StateViabilityCritic,
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    diagnostics: defaultdict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    attempts = particles.shape[0]
    chunk = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = torch.from_numpy(np.repeat(condition_tokens[None, ...], attempts, axis=0)).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, _ = representation.encode(source)
    source_pool = pooled_source(representation, source_example, device)
    memory = memory.to(device).float().unsqueeze(0).expand(attempts, -1, -1)
    memory_mask = memory_mask.to(device).bool().unsqueeze(0).expand(attempts, -1)
    minimum_rms = math.sqrt(model.transport_dim) * float(preregistration["latent_min_std"])
    latent = particles.float()
    minimum_observed_rms = math.inf
    for flow_index in range(int(preregistration["flow_steps"])):
        velocities = []
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
                velocity = model.transport_velocity(
                    latent[start : start + count],
                    flow_time,
                    source_node.expand(count, -1, -1),
                    source["node_mask"].expand(count, -1),
                    tokens[start : start + count],
                )
            velocities.append(velocity.float())
        time = torch.full(
            (attempts,),
            (flow_index + 0.5) / int(preregistration["flow_steps"]),
            device=device,
        )
        with torch.enable_grad():
            query_latent = latent.detach().requires_grad_(True)
            reachability = critic(
                query_latent,
                source_pool.unsqueeze(0).expand(attempts, -1),
                time,
                memory,
                memory_mask,
            )
            gradient = torch.autograd.grad(F.logsigmoid(reachability).sum(), query_latent)[0]
        gradient_norm = gradient.norm(dim=1)
        diagnostics["gradient_norm_sum"] += float(gradient_norm.mean().detach().cpu())
        diagnostics["gradient_steps"] += 1.0
        guidance = F.normalize(gradient, dim=1, eps=1e-8)
        proposal = latent + torch.cat(velocities, dim=0) / int(preregistration["flow_steps"])
        proposal = proposal + (
            float(preregistration["guidance_scale"])
            / int(preregistration["flow_steps"])
        ) * minimum_rms * guidance
        center = proposal.mean(dim=0, keepdim=True)
        residual = proposal - center
        normalized = F.normalize(residual, dim=1)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(-torch.inf)
        neighbours = torch.softmax(
            similarity / float(preregistration["particle_repulsion_temperature"]), dim=1
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
        "final_particle_mean_abs_cosine": float(cosine[off_diagonal].abs().mean().detach().cpu()),
        "final_particle_max_abs_cosine": float(cosine[off_diagonal].abs().max().detach().cpu()),
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


def freeze_guided_candidates(
    model: nn.Module,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    pairs: Sequence[object],
    memories: torch.Tensor,
    memory_masks: torch.Tensor,
    critic: StateViabilityCritic,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, object]]:
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_mask = b41.viability_event_mask
    original_transport = b41.interacting_transport_particles
    rows: list[dict[str, object]] = []
    guidance_diagnostics: defaultdict[str, float] = defaultdict(float)
    try:
        b41.viability_event_mask = exact_support
        for pair_index, pair in enumerate(pairs):
            def guided(*transport_args, **transport_kwargs):
                return guided_transport_particles(
                    *transport_args,
                    **transport_kwargs,
                    critic=critic,
                    memory=memories[pair_index],
                    memory_mask=memory_masks[pair_index],
                    diagnostics=guidance_diagnostics,
                )

            b41.interacting_transport_particles = guided
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
                            "stage": "freeze_state_guided_candidates",
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
        raise RuntimeError(f"State guidance expected {expected} rows, found {len(rows)}")
    diagnostics = {
        "mean_gradient_norm": guidance_diagnostics["gradient_norm_sum"]
        / max(1.0, guidance_diagnostics["gradient_steps"]),
        "gradient_steps": int(guidance_diagnostics["gradient_steps"]),
    }
    return rows, diagnostics, exact_support.manifest()


def development_memories(
    arm: str,
    pairs: Sequence[object],
    preregistration: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if arm == "property_memory":
        memory = torch.from_numpy(
            np.stack([np.asarray(pair.condition, dtype=np.float32) for pair in pairs])
        ).to(torch.float16)
        mask = torch.from_numpy(
            np.stack(
                [
                    np.asarray([True, *[bool(np.any(slot)) for slot in pair.condition[1:]]])
                    for pair in pairs
                ]
            )
        ).bool()
        return memory, mask, {"common_sft_adapter": False, "memory_slots": int(memory.shape[1])}
    llm, tokenizer = operator.load_common_llm(
        args, preregistration, device, sft=True, latent_lora=False
    )
    memory = compressed_llm_token_memories(llm, tokenizer, pairs, preregistration, device)
    mask = torch.ones(memory.shape[:2], dtype=torch.bool)
    del llm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return memory, mask, {
        "common_sft_adapter": True,
        "base_model": preregistration["base_model"],
        "base_model_revision": preregistration["base_model_revision"],
        "memory_slots": int(memory.shape[1]),
    }


def run_prepare(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
    stack,
    valid_summary: Mapping[str, object],
    operator_summary: Mapping[str, object],
) -> None:
    (
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        trajectory_pairs,
        _development_pairs,
        lineage,
    ) = stack
    dataset = collect_trajectory_dataset(
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        trajectory_pairs,
        preregistration,
        device,
    )
    args.trajectory_dataset.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, args.trajectory_dataset)
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare",
        "decision": "trajectory_dataset_frozen_for_parallel_critics",
        "conditions": len(trajectory_pairs),
        "state_rows": int(dataset["traces"].shape[0] * dataset["traces"].shape[1] * dataset["traces"].shape[2]),
        "trajectory_shape": list(dataset["traces"].shape),
        "validation_terminal_class_counts": dataset["validation_terminal_class_counts"],
        "trajectory_dataset_sha256": belief.file_sha256(args.trajectory_dataset),
        "valid_terminal_decision": valid_summary.get("decision"),
        "operator_decision": operator_summary.get("decision"),
        "manifest": {
            **lineage,
            "fit_only_trajectory_labels": True,
            "development_target_access": False,
            "development_property_oracle_access": False,
            "exact_molecule_stop_support": dataset["exact_stop_support"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def run_arm(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
    stack,
    valid_summary: Mapping[str, object],
) -> None:
    arm = args.stage
    if arm not in ARMS:
        raise ValueError(f"Not a critic arm: {arm}")
    if not args.trajectory_dataset.is_file():
        raise FileNotFoundError(f"Missing frozen trajectory dataset: {args.trajectory_dataset}")
    dataset = torch.load(args.trajectory_dataset, map_location="cpu", weights_only=False)
    if dataset.get("protocol") != PROTOCOL:
        raise ValueError("Trajectory dataset protocol drift")
    expected_dataset_sha = belief.file_sha256(args.trajectory_dataset)
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
    if dataset["pair_keys"] != [pair_key(pair) for pair in trajectory_pairs]:
        raise ValueError("Frozen trajectory pair order drift")
    fit_memories, fit_masks, llm_manifest = development_memories(
        arm, trajectory_pairs, preregistration, args, device
    )
    critic, history, critic_metrics = train_critic(
        dataset, fit_memories, fit_masks, preregistration, device
    )
    class_counts = dict(critic_metrics["validation_terminal_class_counts"])
    critic_failures = []
    if float(critic_metrics["state_auc"]) < float(preregistration["critic_gates"]["state_auc"]):
        critic_failures.append("state_auc")
    if int(class_counts["positive"]) < int(preregistration["critic_gates"]["min_positive_terminal_examples"]):
        critic_failures.append("positive_terminal_examples")
    if int(class_counts["negative"]) < int(preregistration["critic_gates"]["min_negative_terminal_examples"]):
        critic_failures.append("negative_terminal_examples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "state_viability_critic.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": arm,
            "state_dict": critic.state_dict(),
            "latent_dim": int(dataset["traces"].shape[-1]),
            "source_dim": int(dataset["source_pool"].shape[-1]),
            "memory_dim": int(fit_memories.shape[-1]),
            "hidden_dim": int(preregistration["critic_hidden_dim"]),
        },
        checkpoint_path,
    )
    development_memory, development_mask, _ = development_memories(
        arm, development_pairs, preregistration, args, device
    )
    frozen, guidance_metrics, support_manifest = freeze_guided_candidates(
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        development_pairs,
        development_memory,
        development_mask,
        critic,
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
    summary = {
        "protocol": PROTOCOL,
        "arm": arm,
        "decision": "await_cross_arm_state_viability_gate",
        "critic_training": history,
        "critic_metrics": critic_metrics,
        "critic_gate": {"passed": not critic_failures, "failures": critic_failures},
        "guidance_metrics": guidance_metrics,
        "metrics": metrics,
        "manifest": {
            **lineage,
            "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
            "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
            "trajectory_dataset_sha256": expected_dataset_sha,
            "critic_checkpoint_sha256": belief.file_sha256(checkpoint_path),
            "frozen_candidates_sha256": belief.file_sha256(frozen_path),
            "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
            "exact_molecule_stop_support": support_manifest,
            "common_llm": llm_manifest,
            "frozen_b41_checkpoint": True,
            "b41_training": False,
            "fit_property_counts": [2],
            "composition_diagnostic_property_counts": [3],
            "development_is_reused_method_development_split": True,
            "development_is_formal_fresh_ood": False,
            "common_llm_emits_text_or_actions": False,
            "current_latent_queries_constraint_memory_each_flow_step": True,
            "terminal_reachability_gradient_guides_latent_vector_field": True,
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
            "valid_terminal_decision": valid_summary.get("decision"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    valid_summary, operator_summary = check_signal_inputs(args, preregistration)
    stack = load_frozen_stack(args, preregistration, device)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed state-viability stage exists: {summary_path}")
    if args.stage == "prepare":
        run_prepare(args, preregistration, device, stack, valid_summary, operator_summary)
    else:
        run_arm(args, preregistration, device, stack, valid_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
