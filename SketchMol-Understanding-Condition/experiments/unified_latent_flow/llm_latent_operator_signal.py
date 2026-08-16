#!/usr/bin/env python3
"""Test whether a common LLM improves frozen molecular latent dynamics.

The experiment keeps the B41 graph transport, direct atom/bond event decoder,
and exact-molecule STOP support frozen.  A controller may only add one bounded
64-dimensional residual to the generation-safe global constraint token.  The
residual therefore changes the continuous latent vector field and graph-event
field; it never emits an action, molecule, route, score, or candidate rank.

Four independently runnable arms share the same fit/dev split and random
generation seeds:

* ``property_mlp``: explicit property/direction features plus a small MLP;
* ``base_frozen``: frozen base-Qwen hidden state plus the same MLP head;
* ``sft_frozen``: frozen common-LLM SFT hidden state plus the same MLP head;
* ``sft_lora``: common-LLM SFT initialization with a new latent-control LoRA.

Controller fitting uses only two-property fit pairs.  Three-property dev pairs
are therefore a held-out composition diagnostic.  Development targets and
property values are unavailable to prompt construction and generation, and
are opened only after exactly twenty raw candidates have been frozen.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

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

import valid_terminal_molecule_latent_jump as valid_terminal  # noqa: E402


b41 = valid_terminal.b41
b40 = valid_terminal.b40
b39 = valid_terminal.b39
b37 = valid_terminal.b37
b36 = valid_terminal.b36
base = valid_terminal.base
belief = valid_terminal.belief
delta = valid_terminal.delta
graph = valid_terminal.graph
hierarchical = valid_terminal.hierarchical
unified = valid_terminal.unified
full_graph = b41.full_graph
b38 = b41.b38

PROTOCOL = "train_only_common_llm_latent_operator_signal_v1"
ARMS = ("property_mlp", "base_frozen", "sft_frozen", "sft_lora")
VALID_TERMINAL_PREREGISTRATION = (
    SCRIPT_DIR / "valid_terminal_molecule_latent_jump_v1_preregistration.json"
)
SYSTEM_PROMPT = (
    "You are a unified molecular constraint agent. Encode the source molecule "
    "and requested property changes. The hidden representation will control a "
    "continuous molecular graph latent process."
)


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
    parser.add_argument("--sft-adapter-dir", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "controller_fit_property_counts": [2],
        "composition_ood_property_counts": [3],
        "common_llm_emits_text_or_actions": False,
        "common_llm_hidden_state_controls_latent_dynamics": True,
        "exact_molecule_materialization_is_stop_support": True,
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
        raise ValueError(f"LLM latent-operator preregistration drift: {drift}")
    if tuple(payload.get("arms", ())) != ARMS:
        raise ValueError("LLM latent-operator arm contract drift")
    implementation_sha = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation_sha:
        raise ValueError(
            "LLM latent-operator implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {implementation_sha}"
        )
    required_locks = {
        "valid_terminal_summary_sha256",
        "valid_terminal_candidates_sha256",
        "sft_adapter_config_sha256",
        "sft_adapter_model_sha256",
    }
    if set(dict(payload.get("locked_signal_inputs", {}))) != required_locks:
        raise ValueError("LLM latent-operator locked signal inputs are incomplete")
    return payload


def check_signal_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    locks = dict(preregistration["locked_signal_inputs"])
    paths = {
        "valid_terminal_summary_sha256": args.valid_terminal_summary,
        "valid_terminal_candidates_sha256": args.valid_terminal_candidates,
        "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
        "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LLM latent-operator inputs: {missing}")
    drift = {
        name: {"expected": locks[name], "actual": belief.file_sha256(path)}
        for name, path in paths.items()
        if belief.file_sha256(path) != locks[name]
    }
    if drift:
        raise ValueError(f"LLM latent-operator locked input drift: {drift}")
    valid_summary = json.loads(args.valid_terminal_summary.read_text(encoding="utf-8"))
    if valid_summary.get("protocol") != valid_terminal.PROTOCOL:
        raise ValueError("LLM latent operator requires the valid-terminal protocol")
    if valid_summary.get("decision") != "stop_valid_terminal_latent_jump_without_gate_changes":
        raise ValueError("LLM latent operator refuses valid-terminal decision drift")
    expected_baseline = dict(preregistration["valid_terminal_baseline"])
    actual_metrics = dict(valid_summary.get("metrics", {}))
    metric_drift = {
        key: {"expected": value, "actual": actual_metrics.get(key)}
        for key, value in expected_baseline.items()
        if not math.isclose(
            float(value), float(actual_metrics.get(key, math.nan)), rel_tol=0.0, abs_tol=1e-12
        )
    }
    if metric_drift:
        raise ValueError(f"LLM latent-operator baseline drift: {metric_drift}")
    return valid_summary, actual_metrics


def stable_pair_key(pair: object, seed: int) -> str:
    payload = f"{seed}|{pair.source_smiles}|{base.task_key(pair.row)}"
    return hashlib.sha256(payload.encode()).hexdigest()


def select_controller_fit_pairs(
    pairs: Sequence[object], preregistration: Mapping[str, object]
) -> list[object]:
    allowed = {int(value) for value in preregistration["controller_fit_property_counts"]}
    eligible = [pair for pair in pairs if int(pair.property_count) in allowed]
    eligible.sort(
        key=lambda pair: stable_pair_key(pair, int(preregistration["controller_fit_seed"]))
    )
    selected = eligible[: int(preregistration["controller_fit_limit"])]
    if len(selected) != int(preregistration["controller_fit_limit"]):
        raise ValueError(
            f"Expected {preregistration['controller_fit_limit']} controller fit pairs, "
            f"found {len(selected)}"
        )
    return selected


def property_features(pair: object) -> np.ndarray:
    specs = {name: int(direction) for name, direction in base.task_specs(pair.row)}
    directions = np.asarray(
        [float(specs.get(prop, 0)) for prop in unified.PROPERTY_COLUMNS], dtype=np.float32
    )
    active = np.asarray([float(value != 0) for value in directions], dtype=np.float32)
    count = np.asarray(
        [float(active.sum()) / max(1, len(unified.PROPERTY_COLUMNS))], dtype=np.float32
    )
    return np.concatenate([directions, active, count], axis=0)


def generation_safe_prompt(pair: object) -> list[dict[str, str]]:
    constraints = [
        {
            "property": str(name),
            "direction": "increase" if int(direction) > 0 else "decrease",
        }
        for name, direction in base.task_specs(pair.row)
    ]
    payload = {
        "schema_version": "llm_latent_operator_constraint_v1",
        "task_mode": "edit",
        "source_smiles": str(pair.source_smiles),
        "constraints": constraints,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    forbidden = (
        "target_smiles",
        "target_properties",
        "generated_smiles",
        "strict_success",
        "property_success",
        "oracle",
    )
    if any(value in serialized.lower() for value in forbidden):
        raise ValueError("Generation-safe LLM prompt contains a forbidden field")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": serialized},
    ]


class LatentConditionResidual(nn.Module):
    """A bounded residual head shared by every controller comparison arm."""

    def __init__(self, input_dim: int, hidden_dim: int, condition_dim: int, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.network = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(condition_dim)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.network(value.float()))


def add_global_residual(tokens: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or residual.ndim != 2:
        raise ValueError("Expected token tensor [B,L,D] and residual [B,D]")
    if tokens.shape[0] != residual.shape[0] or tokens.shape[2] != residual.shape[1]:
        raise ValueError("Condition residual shape mismatch")
    modified = tokens.clone()
    modified[:, 0, :] = modified[:, 0, :] + residual.to(modified.dtype)
    return modified


def _transformer_body(model: object) -> object:
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(causal, "model", None)
    if body is None:
        raise TypeError(f"Unsupported common-LLM model: {type(causal).__name__}")
    return body


def prompt_batch(tokenizer: object, pairs: Sequence[object], max_length: int, device: torch.device):
    texts = [
        tokenizer.apply_chat_template(
            generation_safe_prompt(pair), tokenize=False, add_generation_prompt=True
        )
        for pair in pairs
    ]
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def llm_hidden_batch(
    model: object,
    tokenizer: object,
    pairs: Sequence[object],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    batch = prompt_batch(tokenizer, pairs, max_length, device)
    body = _transformer_body(model)
    output = body(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    last = output.last_hidden_state
    index = batch["attention_mask"].sum(dim=1).long().sub(1).clamp_min(0)
    return last[torch.arange(last.shape[0], device=device), index].float()


@torch.no_grad()
def frozen_llm_embeddings(
    model: object,
    tokenizer: object,
    pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    batch_size = int(preregistration["llm_embedding_batch_size"])
    model.eval().requires_grad_(False)
    for start in range(0, len(pairs), batch_size):
        rows.append(
            llm_hidden_batch(
                model,
                tokenizer,
                pairs[start : start + batch_size],
                int(preregistration["llm_max_length"]),
                device,
            ).cpu()
        )
    return torch.cat(rows, dim=0)


def load_common_llm(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
    *,
    sft: bool,
    latent_lora: bool,
) -> tuple[object, object]:
    try:
        import peft
        import transformers
    except ImportError as exc:
        raise RuntimeError(f"Missing common-LLM dependency: {exc}") from exc
    model_id = str(preregistration["base_model"])
    revision = str(preregistration["base_model_revision"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    model.config.use_cache = False
    if sft:
        model = peft.PeftModel.from_pretrained(
            model,
            args.sft_adapter_dir,
            is_trainable=False,
            adapter_name="common_sft",
        )
    if latent_lora:
        if not sft:
            raise ValueError("Latent LoRA must initialize from the common SFT adapter")
        model = model.merge_and_unload()
        config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=int(preregistration["latent_lora_rank"]),
            lora_alpha=int(preregistration["latent_lora_alpha"]),
            lora_dropout=float(preregistration["latent_lora_dropout"]),
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = peft.get_peft_model(model, config)
        model.get_base_model().gradient_checkpointing_enable()
        model.enable_input_require_grads()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    return model, tokenizer


def _training_batch_loss(
    controller: LatentConditionResidual,
    inputs: torch.Tensor,
    model: nn.Module,
    representation: nn.Module,
    items: Sequence[object],
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
    *,
    epoch: int,
    global_batch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    collated = base.pair_collate(items)
    source = base.move_graph_batch(collated["source"], device)
    target = base.move_graph_batch(collated["target"], device)
    tokens = collated["condition"].to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, source_edge = representation.encode(source)
        target_node, target_edge = representation.encode(target)
        base_condition = model.route_condition(tokens)
        endpoint = model.posterior_endpoint(
            source,
            target,
            source_node,
            source_edge,
            target_node,
            target_edge,
            base_condition,
        )
        node_targets, edge_targets = delta.delta_action_targets(source, target, vocabulary)
        working = full_graph.working_node_mask(
            source["node_mask"], int(preregistration["birth_capacity"]), target["node_mask"]
        )
        (
            current_node,
            current_edge,
            target_next,
            jump_time,
            target_count,
            executed_count,
        ) = b41.build_viable_prefix_batch(
            node_targets,
            edge_targets,
            model.denoiser.layout,
            epoch=epoch,
            global_batch=global_batch,
            preregistration=preregistration,
            device=device,
        )
        noise = torch.randn_like(endpoint)
        flow_time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
        current_latent = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
    residual = controller(inputs)
    modified_tokens = add_global_residual(tokens, residual)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        condition = model.route_condition(modified_tokens)
        velocity = model.transport_velocity(
            current_latent, flow_time, source_node, source["node_mask"], modified_tokens
        )
        flow_loss = F.mse_loss(velocity.float(), (endpoint - noise).float())
        remaining_mass = (
            target_count.float() - executed_count.float()
        ) / float(preregistration["max_jumps"])
        logits = model.denoiser(
            current_node,
            current_edge,
            source_node,
            source_edge,
            source["node_mask"].bool(),
            working,
            jump_time,
            condition,
            endpoint,
            remaining_mass,
        )
        legal, _ = b41.viability_event_mask(
            model.denoiser,
            source,
            current_node,
            current_edge,
            working,
            support,
            support_tensors,
        )
        jump_loss, target_mass, jump_accuracy = b38.orderless_jump_loss(
            logits, legal, target_next
        )
        cardinality_logits = model.cardinality_logits(
            source_node, source["node_mask"].bool(), condition, endpoint
        )
        cardinality_loss = F.cross_entropy(
            cardinality_logits.float(),
            target_count.clamp_max(cardinality_logits.shape[1] - 1),
        )
        residual_penalty = residual.square().mean()
        loss = (
            float(preregistration["flow_loss_weight"]) * flow_loss
            + float(preregistration["event_loss_weight"]) * jump_loss
            + float(preregistration["cardinality_loss_weight"]) * cardinality_loss
            + float(preregistration["residual_penalty_weight"]) * residual_penalty
        )
    values = {
        "loss": float(loss.detach()),
        "flow_loss": float(flow_loss.detach()),
        "event_loss": float(jump_loss.detach()),
        "cardinality_loss": float(cardinality_loss.detach()),
        "residual_penalty": float(residual_penalty.detach()),
        "target_next_probability_mass": float(target_mass.detach()),
        "next_event_set_accuracy": float(jump_accuracy.detach()),
        "mean_residual_norm": float(residual.detach().norm(dim=1).mean()),
    }
    return loss, values


def train_controller(
    controller: LatentConditionResidual,
    model: nn.Module,
    representation: nn.Module,
    fit_pairs: Sequence[object],
    input_provider: Callable[[Sequence[int], Sequence[object]], torch.Tensor],
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
    *,
    extra_parameters: Sequence[nn.Parameter] = (),
) -> list[dict[str, float]]:
    model.eval().requires_grad_(False)
    representation.eval().requires_grad_(False)
    controller_parameters = list(controller.parameters())
    extra_parameters = list(extra_parameters)
    parameters = controller_parameters + extra_parameters
    parameter_groups: list[dict[str, object]] = [
        {
            "params": controller_parameters,
            "lr": float(preregistration["controller_learning_rate"]),
        }
    ]
    if extra_parameters:
        parameter_groups.append(
            {
                "params": extra_parameters,
                "lr": float(preregistration["latent_lora_learning_rate"]),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(preregistration["controller_weight_decay"]),
    )
    batch_size = int(preregistration["controller_batch_size"])
    history: list[dict[str, float]] = []
    global_batch = 0
    for epoch in range(1, int(preregistration["controller_epochs"]) + 1):
        order = list(range(len(fit_pairs)))
        random.Random(int(preregistration["controller_fit_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        controller.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            items = [fit_pairs[index] for index in indices]
            inputs = input_provider(indices, items).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, values = _training_batch_loss(
                controller,
                inputs,
                model,
                representation,
                items,
                vocabulary,
                support,
                support_tensors,
                preregistration,
                device,
                epoch=epoch,
                global_batch=global_batch,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, float(preregistration["controller_grad_clip"]))
            optimizer.step()
            for name, value in values.items():
                totals[name] += value
            batches += 1
            global_batch += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite LLM latent-controller metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "controller_epoch", **row}, sort_keys=True), flush=True)
    controller.eval()
    return history


@torch.no_grad()
def conditioned_pairs(
    pairs: Sequence[object],
    controller: LatentConditionResidual,
    embeddings: torch.Tensor,
    device: torch.device,
) -> list[object]:
    controller.eval()
    residuals = []
    for start in range(0, len(pairs), 64):
        residuals.append(controller(embeddings[start : start + 64].to(device)).cpu())
    residual = torch.cat(residuals, dim=0).numpy()
    output = []
    for pair, value in zip(pairs, residual, strict=True):
        cloned = copy.copy(pair)
        tokens = np.asarray(pair.condition, dtype=np.float32).copy()
        tokens[0] += np.asarray(value, dtype=np.float32)
        cloned.condition = tokens
        output.append(cloned)
    return output


def property_count_diagnostics(
    evaluated: Sequence[Mapping[str, object]], metrics: Mapping[str, object]
) -> dict[str, object]:
    output: dict[str, object] = {}
    by_count = dict(metrics["by_property_count"])
    for property_count, condition_metrics in by_count.items():
        rows = [row for row in evaluated if int(row["property_count"]) == int(property_count)]
        output[str(property_count)] = {
            **dict(condition_metrics),
            "validity": float(np.mean([bool(row["valid"]) for row in rows])),
            "max_horizon_hit_rate": float(
                np.mean([bool(row["max_horizon_hit"]) for row in rows])
            ),
            "mean_unique_valid": float(
                np.mean(
                    [
                        len(
                            {
                                str(candidate["generated_smiles"])
                                for candidate in rows
                                if candidate["condition_id"] == condition_id
                                and bool(candidate["valid"])
                                and str(candidate["generated_smiles"])
                            }
                        )
                        for condition_id in sorted({str(row["condition_id"]) for row in rows})
                    ]
                )
            ),
        }
    return output


def evaluate_arm(
    args: argparse.Namespace,
    arm: str,
    model: nn.Module,
    representation: nn.Module,
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    development_pairs: Sequence[object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_support = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    try:
        frozen = b41.freeze_candidates(
            model,
            representation,
            vocabulary,
            support,
            support_tensors,
            development_pairs,
            preregistration,
            device,
        )
    finally:
        b41.viability_event_mask = original_support
    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, frozen)
    evaluated, metrics = b41.evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    metrics = dict(metrics)
    metrics["by_property_count_diagnostic"] = property_count_diagnostics(evaluated, metrics)
    artifacts = {
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "exact_molecule_stop_support": exact_support.manifest(),
    }
    print(
        json.dumps(
            {
                "stage": "arm_evaluated",
                "arm": arm,
                "validity": metrics["validity"],
                "strict_any20": metrics["strict_any20"],
                "max_horizon_hit_rate": metrics["max_horizon_hit_rate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metrics, artifacts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed LLM latent-operator arm exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["controller_fit_seed"]))
    device = base.resolve_device(str(args.device))
    valid_terminal_prereg = valid_terminal.read_preregistration(
        VALID_TERMINAL_PREREGISTRATION
    )
    b22_summary, b22_checkpoint, b36_summary, b37_summary, b41_checkpoint = (
        valid_terminal.check_locked_inputs(args, valid_terminal_prereg)
    )
    valid_summary, _baseline_metrics = check_signal_inputs(args, preregistration)

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
    controller_fit_pairs = select_controller_fit_pairs(fit_pairs, preregistration)
    if any(int(pair.property_count) != 2 for pair in controller_fit_pairs):
        raise ValueError("Controller fit set contains a composition-OOD row")
    if not any(int(pair.property_count) == 3 for pair in development_pairs):
        raise ValueError("Development split lacks the registered three-property OOD set")

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

    llm_model = None
    tokenizer = None
    train_history: list[dict[str, float]]
    llm_manifest: dict[str, object] = {
        "base_model": None,
        "base_model_revision": None,
        "common_sft_adapter": False,
        "latent_lora_training": False,
    }
    if args.arm == "property_mlp":
        fit_embeddings = torch.from_numpy(
            np.stack([property_features(pair) for pair in controller_fit_pairs])
        )
        dev_embeddings = torch.from_numpy(
            np.stack([property_features(pair) for pair in development_pairs])
        )
        controller = LatentConditionResidual(
            fit_embeddings.shape[1],
            int(preregistration["controller_hidden_dim"]),
            int(preregistration["condition_dim"]),
            float(preregistration["controller_residual_scale"]),
        ).to(device)
        train_history = train_controller(
            controller,
            model,
            representation,
            controller_fit_pairs,
            lambda indices, _items: fit_embeddings[list(indices)],
            vocabulary,
            support,
            support_tensors,
            preregistration,
            device,
        )
    elif args.arm in {"base_frozen", "sft_frozen"}:
        use_sft = args.arm == "sft_frozen"
        llm_model, tokenizer = load_common_llm(
            args, preregistration, device, sft=use_sft, latent_lora=False
        )
        fit_embeddings = frozen_llm_embeddings(
            llm_model, tokenizer, controller_fit_pairs, preregistration, device
        )
        dev_embeddings = frozen_llm_embeddings(
            llm_model, tokenizer, development_pairs, preregistration, device
        )
        controller = LatentConditionResidual(
            fit_embeddings.shape[1],
            int(preregistration["controller_hidden_dim"]),
            int(preregistration["condition_dim"]),
            float(preregistration["controller_residual_scale"]),
        ).to(device)
        del llm_model
        llm_model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_history = train_controller(
            controller,
            model,
            representation,
            controller_fit_pairs,
            lambda indices, _items: fit_embeddings[list(indices)],
            vocabulary,
            support,
            support_tensors,
            preregistration,
            device,
        )
        llm_manifest = {
            "base_model": preregistration["base_model"],
            "base_model_revision": preregistration["base_model_revision"],
            "common_sft_adapter": use_sft,
            "latent_lora_training": False,
        }
    else:
        llm_model, tokenizer = load_common_llm(
            args, preregistration, device, sft=True, latent_lora=True
        )
        hidden_dim = int(llm_model.get_base_model().config.hidden_size)
        controller = LatentConditionResidual(
            hidden_dim,
            int(preregistration["controller_hidden_dim"]),
            int(preregistration["condition_dim"]),
            float(preregistration["controller_residual_scale"]),
        ).to(device)

        def live_provider(_indices: Sequence[int], items: Sequence[object]) -> torch.Tensor:
            return llm_hidden_batch(
                llm_model,
                tokenizer,
                items,
                int(preregistration["llm_max_length"]),
                device,
            )

        trainable_llm = [parameter for parameter in llm_model.parameters() if parameter.requires_grad]
        train_history = train_controller(
            controller,
            model,
            representation,
            controller_fit_pairs,
            live_provider,
            vocabulary,
            support,
            support_tensors,
            preregistration,
            device,
            extra_parameters=trainable_llm,
        )
        latent_adapter_dir = args.output_dir / "latent_lora_adapter"
        llm_model.save_pretrained(latent_adapter_dir, safe_serialization=True)
        dev_embeddings = frozen_llm_embeddings(
            llm_model, tokenizer, development_pairs, preregistration, device
        )
        llm_manifest = {
            "base_model": preregistration["base_model"],
            "base_model_revision": preregistration["base_model_revision"],
            "common_sft_adapter": True,
            "latent_lora_training": True,
            "latent_lora_adapter": str(latent_adapter_dir),
        }
        del llm_model
        llm_model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    controller_path = args.output_dir / "latent_condition_controller.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": args.arm,
            "state_dict": controller.state_dict(),
            "input_dim": int(controller.network[0].normalized_shape[0]),
            "condition_dim": int(preregistration["condition_dim"]),
        },
        controller_path,
    )
    dev_conditioned = conditioned_pairs(
        development_pairs, controller, dev_embeddings, device
    )
    metrics, artifacts = evaluate_arm(
        args,
        args.arm,
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        dev_conditioned,
        preregistration,
        device,
    )
    manifest = {
        "protocol": PROTOCOL,
        "arm": args.arm,
        "seed": int(preregistration["controller_fit_seed"]),
        "device": str(device),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "controller_checkpoint_sha256": belief.file_sha256(controller_path),
        "locked_signal_inputs": dict(preregistration["locked_signal_inputs"]),
        "valid_terminal_baseline_decision": valid_summary.get("decision"),
        "reconstruction": reconstruction,
        "split": split,
        "controller_fit_pairs": len(controller_fit_pairs),
        "controller_fit_property_counts": [2],
        "composition_ood_property_counts": [3],
        "development_conditions": len(development_pairs),
        "common_llm_emits_text_or_actions": False,
        "common_llm_hidden_state_controls_latent_dynamics": args.arm != "property_mlp",
        "llm": llm_manifest,
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "exact_molecule_materialization_is_stop_support": True,
        "exact_raw_attempts_per_condition": 20,
        "frozen_before_target_or_property_evaluation": True,
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
        "representation_protocol": representation_summary.get("protocol"),
        **artifacts,
    }
    summary = {
        "protocol": PROTOCOL,
        "arm": args.arm,
        "decision": "await_cross_arm_llm_latent_operator_gate",
        "training_history": train_history,
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
