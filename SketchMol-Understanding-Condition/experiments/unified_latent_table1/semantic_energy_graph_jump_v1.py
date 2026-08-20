#!/usr/bin/env python3
"""Language-grounded energy field inside a frozen canonical graph-jump process.

The Common LLM sees constraint text only: source SMILES, targets, property
oracles, candidate pools, routes, and actions are never part of its prompt.
Correct instructions are aligned to source-to-target graph-flow endpoints while
direction-reversed, character-scrambled, and property-swapped instructions are
explicit hard negatives.  A low-rank adapter emits both condition tokens and a
state-dependent continuous velocity residual; the frozen canonical graph-event
kernel and exact legal-action support still materialize molecules.

Prepare, train, freeze, and evaluate are separate processes.  In particular,
freeze cannot accept the sealed evaluation-target path.
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
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
for module_path in (SCRIPT_DIR, PROJECT_DIR, LATENT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import fresh_graph_jump_language_confirmation as fresh  # noqa: E402
import llm_latent_operator_signal as operator  # noqa: E402
from dead_end_safe_support import DeadEndSafeSupport  # noqa: E402


base = fresh.base
graph = fresh.graph
b41 = fresh.b41
b40 = fresh.b40
hierarchical = fresh.hierarchical
unified = fresh.unified
valid_terminal = fresh.valid_terminal

PROTOCOL = "train_only_semantic_energy_graph_jump_v1"
ARMS = (
    "numeric_canonical",
    "language_matched",
    "language_reversed",
    "language_scrambled",
    "language_property_swap",
)
LANGUAGE_ARMS = ARMS[1:]
NEGATIVE_VARIANTS = ("reversed", "scrambled", "property_swap")
FORBIDDEN_PROMPT_TERMS = (
    "source_smiles",
    "target_smiles",
    "target_properties",
    "generated_smiles",
    "oracle",
    "candidate",
    "route",
    "action",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare = stages.add_parser("prepare")
    prepare.add_argument("--predecessor-fit-bundle", required=True, type=Path)
    prepare.add_argument("--e1-manifest", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    train = stages.add_parser("train")
    train.add_argument("--prepare-summary", required=True, type=Path)
    train.add_argument("--fit-probe-bundle", required=True, type=Path)
    train.add_argument("--representation-checkpoint", required=True, type=Path)
    train.add_argument("--representation-summary", required=True, type=Path)
    train.add_argument("--canonical-checkpoint", required=True, type=Path)
    train.add_argument("--sft-adapter-dir", required=True, type=Path)
    train.add_argument("--e1-manifest", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--device", default="auto")

    freeze = stages.add_parser("freeze")
    freeze.add_argument("--prepare-summary", required=True, type=Path)
    freeze.add_argument("--fit-probe-bundle", required=True, type=Path)
    freeze.add_argument("--generation-conditions", required=True, type=Path)
    freeze.add_argument("--representation-checkpoint", required=True, type=Path)
    freeze.add_argument("--representation-summary", required=True, type=Path)
    freeze.add_argument("--canonical-checkpoint", required=True, type=Path)
    freeze.add_argument("--sft-adapter-dir", required=True, type=Path)
    freeze.add_argument("--adapter-checkpoint", required=True, type=Path)
    freeze.add_argument("--train-summary", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--device", default="auto")

    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--prepare-summary", required=True, type=Path)
    evaluate.add_argument("--evaluation-targets", required=True, type=Path)
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "arms": list(ARMS),
        "common_llm_prompt_contains_source": False,
        "explicit_semantic_hard_negatives": list(NEGATIVE_VARIANTS),
        "frozen_canonical_graph_jump": True,
        "canonical_training": False,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "generation_target_access": False,
        "official_test_access": False,
        "single_seed": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Semantic-energy preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Semantic-energy implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing semantic-energy inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Semantic-energy locked-input drift: {drift}")
    return actual


def specs_for_row(row: Mapping[str, object]) -> list[tuple[str, int]]:
    return [
        (str(name), int(direction))
        for name, direction in base.task_specs(row)
        if int(direction) != 0
    ]


def direction_word(direction: int, *, heldout: bool) -> str:
    if heldout:
        return "raise" if int(direction) > 0 else "lower"
    return "increase" if int(direction) > 0 else "decrease"


def render_constraint(
    specs: Sequence[tuple[str, int]],
    property_names: Mapping[str, object],
    *,
    heldout: bool,
) -> str:
    parts = [
        f"{direction_word(direction, heldout=heldout)} "
        f"{property_names.get(prop, prop)}"
        for prop, direction in specs
    ]
    if not parts:
        raise ValueError("Constraint-only prompt requires at least one property")
    if len(parts) == 1:
        return f"Modify the molecule to {parts[0]}."
    return f"Modify the molecule to {', '.join(parts[:-1])}, and {parts[-1]}."


def scramble_text(text: str, seed: int, key: str) -> str:
    rng = random.Random(int(text_sha256(f"{seed}|{key}|{text}")[:16], 16))
    output = []
    for token in text.split():
        chars = list(token)
        if len(chars) >= 4:
            rng.shuffle(chars)
        output.append("".join(chars))
    return " ".join(output)


def property_swap_specs(specs: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    used = {prop for prop, _direction in specs}
    columns = list(unified.PROPERTY_COLUMNS)
    swapped: list[tuple[str, int]] = []
    for prop, direction in specs:
        start = columns.index(prop) if prop in columns else 0
        replacement = None
        for offset in range(1, len(columns) + 1):
            candidate = str(columns[(start + offset) % len(columns)])
            if candidate not in used and candidate not in {item[0] for item in swapped}:
                replacement = candidate
                break
        if replacement is None:
            raise ValueError(f"Cannot property-swap {prop}")
        swapped.append((replacement, int(direction)))
    return swapped


def instruction_variants(
    row: Mapping[str, object],
    property_names: Mapping[str, object],
    *,
    seed: int,
    key: str,
    heldout: bool,
) -> dict[str, str]:
    specs = specs_for_row(row)
    matched = render_constraint(specs, property_names, heldout=heldout)
    values = {
        "matched": matched,
        "reversed": render_constraint(
            [(prop, -direction) for prop, direction in specs],
            property_names,
            heldout=heldout,
        ),
        "scrambled": scramble_text(matched, seed, key),
        "property_swap": render_constraint(
            property_swap_specs(specs), property_names, heldout=heldout
        ),
    }
    for name, text in values.items():
        lower = text.lower()
        leaked = [term for term in FORBIDDEN_PROMPT_TERMS if term in lower]
        if leaked:
            raise ValueError(f"Constraint-only {name} prompt leaked terms: {leaked}")
    return values


def run_prepare(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed semantic-energy prepare exists: {summary_path}")
    check_locked_inputs(
        preregistration,
        {
            "predecessor_fit_bundle_sha256": args.predecessor_fit_bundle,
            "e1_manifest_sha256": args.e1_manifest,
        },
    )
    bundle = fresh.load_frozen_predecessor_bundle(args.predecessor_fit_bundle)
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    if len(pairs) != int(preregistration["fit_probe_conditions"]):
        raise ValueError(f"Expected {preregistration['fit_probe_conditions']} pairs, found {len(pairs)}")
    if len(train_indices) != int(preregistration["fit_conditions"]):
        raise ValueError("Fit condition count drift")
    if len(validation_indices) != int(preregistration["probe_conditions"]):
        raise ValueError("Probe condition count drift")
    train_sources = {pairs[index].source_smiles for index in train_indices}
    probe_sources = {pairs[index].source_smiles for index in validation_indices}
    if train_sources & probe_sources:
        raise ValueError("Fit/probe source overlap")
    e1 = read_json(args.e1_manifest)
    property_names = dict(e1["property_names"])
    fit_bundle_path = args.output_dir / "fit_probe_bundle.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "pairs": pairs,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "vocabulary": dict(bundle["vocabulary"]),
            "support": dict(bundle["support"]),
            "lineage": dict(bundle["lineage"]),
        },
        fit_bundle_path,
    )
    generation_records = []
    evaluation_records = []
    for pair_index, original_index in enumerate(validation_indices):
        pair = pairs[original_index]
        safe = fresh.fresh_v3.direction_only_row(pair.row, pair.source_smiles)
        condition_id = f"semantic_energy_dev_{pair_index:04d}"
        instructions = instruction_variants(
            safe,
            property_names,
            seed=int(preregistration["hard_negative_seed"]),
            key=condition_id,
            heldout=True,
        )
        generation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": pair_index,
                "original_index": original_index,
                "source_smiles": pair.source_smiles,
                "property_count": int(pair.property_count),
                "task": base.task_key(safe),
                "condition_row": safe,
                "instructions": instructions,
            }
        )
        evaluation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": pair_index,
                "source_smiles": pair.source_smiles,
                "target_smiles": pair.target_smiles,
                "property_count": int(pair.property_count),
                "row": dict(pair.row),
            }
        )
    generation_path = args.output_dir / "generation_conditions.json"
    evaluation_path = args.output_dir / "sealed_evaluation_targets.json"
    generation_text = json.dumps(
        {
            "protocol": PROTOCOL,
            "role": "constraint_text_and_sources_without_targets",
            "records": generation_records,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    forbidden_generation = (
        "target_smiles",
        "target_properties",
        "generated_smiles",
        "strict_success",
        "oracle",
    )
    leaks = [term for term in forbidden_generation if term in generation_text.lower()]
    if leaks:
        raise ValueError(f"Generation manifest leaked target material: {leaks}")
    generation_path.write_text(generation_text, encoding="utf-8")
    evaluation_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "role": "sealed_post_freeze_targets",
                "records": evaluation_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare",
        "decision": "fit_probe_and_target_isolated_generation_manifests_frozen",
        "fit_probe_conditions": len(pairs),
        "fit_conditions": len(train_indices),
        "probe_conditions": len(validation_indices),
        "fit_sources": len(train_sources),
        "probe_sources": len(probe_sources),
        "fit_probe_source_overlap": len(train_sources & probe_sources),
        "artifacts": {
            "fit_probe_bundle_sha256": file_sha256(fit_bundle_path),
            "generation_conditions_sha256": file_sha256(generation_path),
            "evaluation_targets_sha256": file_sha256(evaluation_path),
        },
        "contract": {
            "common_llm_prompt_contains_source": False,
            "fit_target_access": True,
            "probe_target_access_during_training_diagnostics": True,
            "generation_target_access": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def constraint_only_chat(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Encode the requested molecular property changes as a continuous "
                "semantic control signal for a graph latent process."
            ),
        },
        {"role": "user", "content": text},
    ]


@torch.no_grad()
def embed_texts(
    model: object,
    tokenizer: object,
    texts: Sequence[str],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    rows = []
    body = operator._transformer_body(model)
    batch_size = int(preregistration["llm_embedding_batch_size"])
    model.eval().requires_grad_(False)
    for start in range(0, len(texts), batch_size):
        batch_texts = [
            tokenizer.apply_chat_template(
                constraint_only_chat(text), tokenize=False, add_generation_prompt=True
            )
            for text in texts[start : start + batch_size]
        ]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(preregistration["llm_max_length"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = body(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        rows.append(pooled.cpu())
    return torch.cat(rows, dim=0)


def load_constraint_embeddings(
    pairs: Sequence[object],
    validation_indices: set[int],
    e1: Mapping[str, object],
    preregistration: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    property_names = dict(e1["property_names"])
    texts: dict[str, list[str]] = {name: [] for name in ("matched", *NEGATIVE_VARIANTS)}
    prompt_records = []
    for index, pair in enumerate(pairs):
        variants = instruction_variants(
            pair.row,
            property_names,
            seed=int(preregistration["hard_negative_seed"]),
            key=f"fit_probe_{index:04d}",
            heldout=index in validation_indices,
        )
        prompt_records.append(variants)
        for name in texts:
            texts[name].append(variants[name])
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm, tokenizer = operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=False
    )
    try:
        embeddings = {
            name: embed_texts(llm, tokenizer, values, preregistration, device)
            for name, values in texts.items()
        }
    finally:
        del llm
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prompt_json = json.dumps(prompt_records, sort_keys=True, separators=(",", ":"))
    manifest = {
        "base_model": preregistration["base_model"],
        "base_model_revision": preregistration["base_model_revision"],
        "sft_adapter_config_sha256": file_sha256(args.sft_adapter_dir / "adapter_config.json"),
        "sft_adapter_model_sha256": file_sha256(args.sft_adapter_dir / "adapter_model.safetensors"),
        "prompt_records_sha256": text_sha256(prompt_json),
        "prompt_contains_source_smiles": False,
        "embedding_dim": int(embeddings["matched"].shape[1]),
        "hard_negative_variants": list(NEGATIVE_VARIANTS),
    }
    return embeddings, manifest


class SemanticEnergyAdapter(nn.Module):
    """Factorized language energy over condition tokens and graph-flow states."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        token_count: int,
        condition_dim: int,
        latent_dim: int,
        source_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.condition_dim = int(condition_dim)
        self.residual_scale = float(residual_scale)
        self.language = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.token_head = nn.Linear(int(hidden_dim), int(token_count) * int(condition_dim))
        self.edit_head = nn.Linear(int(hidden_dim), int(latent_dim))
        self.state = nn.Sequential(
            nn.LayerNorm(int(latent_dim) + int(source_dim) + 3),
            nn.Linear(int(latent_dim) + int(source_dim) + 3, int(hidden_dim)),
            nn.SiLU(),
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)

    def initialize_token_bias(self, mean_tokens: torch.Tensor) -> None:
        if mean_tokens.shape != (self.token_count, self.condition_dim):
            raise ValueError("Mean condition-token shape mismatch")
        nn.init.zeros_(self.token_head.weight)
        with torch.no_grad():
            self.token_head.bias.copy_(mean_tokens.reshape(-1).float())

    def language_code(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.language(embedding.float())

    def condition_tokens(self, embedding: torch.Tensor) -> torch.Tensor:
        code = self.language_code(embedding)
        return self.token_head(code).view(-1, self.token_count, self.condition_dim)

    def energy_score(self, embedding: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
        code = self.language_code(embedding)
        edit = F.normalize(self.edit_head(code), dim=1)
        return (edit * F.normalize(endpoint.float(), dim=1)).sum(dim=1)

    def velocity_residual(
        self,
        latent: torch.Tensor,
        source_pool: torch.Tensor,
        flow_time: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        phase = torch.stack(
            [flow_time, torch.sin(math.pi * flow_time), torch.cos(math.pi * flow_time)],
            dim=1,
        )
        state = self.state(torch.cat([latent.float(), source_pool.float(), phase.float()], dim=1))
        fused = state * self.language_code(embedding)
        return self.residual_scale * torch.tanh(self.velocity(fused))


def pooled_source_node(source_node: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
    mask = source_mask.unsqueeze(-1).to(source_node.dtype)
    return (source_node * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def semantic_batch_loss(
    adapter: SemanticEnergyAdapter,
    model: nn.Module,
    representation: nn.Module,
    items: Sequence[object],
    embeddings: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    collated = base.pair_collate(items)
    source = base.move_graph_batch(collated["source"], device)
    target = base.move_graph_batch(collated["target"], device)
    canonical_tokens = collated["condition"].to(device).float()
    selected = {
        name: value[list(indices)].to(device).float() for name, value in embeddings.items()
    }
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        source_node, source_edge = representation.encode(source)
        target_node, target_edge = representation.encode(target)
        teacher_condition = model.route_condition(canonical_tokens)
        endpoint = model.posterior_endpoint(
            source,
            target,
            source_node,
            source_edge,
            target_node,
            target_edge,
            teacher_condition,
        ).float()
        noise = torch.randn_like(endpoint)
        flow_time = torch.rand(len(items), device=device).clamp_(0.02, 0.98)
        current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
        source_pool = pooled_source_node(source_node, source["node_mask"]).float()
        target_velocity = endpoint - noise

    correct_tokens = adapter.condition_tokens(selected["matched"])
    negative_tokens = {
        name: adapter.condition_tokens(selected[name]) for name in NEGATIVE_VARIANTS
    }
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        base_velocity = model.transport_velocity(
            current,
            flow_time.to(source_node.dtype),
            source_node,
            source["node_mask"],
            correct_tokens,
        ).float()
    residual = adapter.velocity_residual(
        current, source_pool, flow_time, selected["matched"]
    )
    adapted_velocity = base_velocity + residual
    flow_loss = F.mse_loss(adapted_velocity, target_velocity)
    token_loss = F.mse_loss(correct_tokens, canonical_tokens)
    zero_token_mse = canonical_tokens.square().mean()

    correct_score = adapter.energy_score(selected["matched"], endpoint)
    negative_scores = torch.stack(
        [adapter.energy_score(selected[name], endpoint) for name in NEGATIVE_VARIANTS],
        dim=1,
    )
    hardest_negative_score = negative_scores.max(dim=1).values
    semantic_margin = correct_score - hardest_negative_score
    semantic_margin_loss = F.relu(
        float(preregistration["semantic_score_margin"]) - semantic_margin
    ).mean()
    hard_negative_accuracy = (correct_score[:, None] > negative_scores).all(dim=1).float().mean()

    per_row_correct_token_mse = (correct_tokens - canonical_tokens).square().mean(dim=(1, 2))
    per_row_negative_token_mse = torch.stack(
        [
            (negative_tokens[name] - canonical_tokens).square().mean(dim=(1, 2))
            for name in NEGATIVE_VARIANTS
        ],
        dim=1,
    )
    easiest_negative_token_mse = per_row_negative_token_mse.min(dim=1).values
    token_margin = easiest_negative_token_mse - per_row_correct_token_mse
    token_margin_loss = F.relu(
        float(preregistration["token_mse_margin"]) - token_margin
    ).mean()

    reversed_tokens = negative_tokens["reversed"]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        reversed_base_velocity = model.transport_velocity(
            current,
            flow_time.to(source_node.dtype),
            source_node,
            source["node_mask"],
            reversed_tokens,
        ).float()
    reversed_residual = adapter.velocity_residual(
        current, source_pool, flow_time, selected["reversed"]
    )
    reversed_flow_loss = F.mse_loss(
        reversed_base_velocity + reversed_residual, target_velocity
    )
    flow_advantage = reversed_flow_loss - flow_loss
    flow_margin_loss = F.relu(
        float(preregistration["flow_mse_margin"]) - flow_advantage
    )
    residual_penalty = residual.square().mean()
    loss = (
        float(preregistration["flow_loss_weight"]) * flow_loss
        + float(preregistration["token_reconstruction_weight"]) * token_loss
        + float(preregistration["semantic_margin_weight"]) * semantic_margin_loss
        + float(preregistration["token_margin_weight"]) * token_margin_loss
        + float(preregistration["flow_margin_weight"]) * flow_margin_loss
        + float(preregistration["residual_penalty_weight"]) * residual_penalty
    )
    return loss, {
        "loss": float(loss.detach()),
        "flow_loss": float(flow_loss.detach()),
        "reversed_flow_loss": float(reversed_flow_loss.detach()),
        "matched_flow_advantage": float(flow_advantage.detach()),
        "token_mse": float(token_loss.detach()),
        "zero_token_mse": float(zero_token_mse.detach()),
        "token_mse_ratio_vs_zero": float((token_loss / zero_token_mse.clamp_min(1e-12)).detach()),
        "semantic_margin_loss": float(semantic_margin_loss.detach()),
        "semantic_score_margin": float(semantic_margin.mean().detach()),
        "hard_negative_accuracy": float(hard_negative_accuracy.detach()),
        "token_margin": float(token_margin.mean().detach()),
        "token_margin_loss": float(token_margin_loss.detach()),
        "flow_margin_loss": float(flow_margin_loss.detach()),
        "residual_norm": float(residual.detach().norm(dim=1).mean()),
    }


def train_adapter(
    adapter: SemanticEnergyAdapter,
    model: nn.Module,
    representation: nn.Module,
    pairs: Sequence[object],
    indices: Sequence[int],
    embeddings: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(preregistration["adapter_learning_rate"]),
        weight_decay=float(preregistration["adapter_weight_decay"]),
    )
    history = []
    batch_size = int(preregistration["adapter_batch_size"])
    model.eval().requires_grad_(False)
    representation.eval().requires_grad_(False)
    for epoch in range(1, int(preregistration["adapter_epochs"]) + 1):
        order = list(indices)
        random.Random(int(preregistration["adapter_training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        adapter.train()
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            loss, metrics = semantic_batch_loss(
                adapter,
                model,
                representation,
                [pairs[index] for index in chosen],
                embeddings,
                chosen,
                preregistration,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), float(preregistration["adapter_grad_clip"]))
            optimizer.step()
            for name, value in metrics.items():
                totals[name] += value
            batches += 1
        row = {"epoch": epoch, **{name: value / max(1, batches) for name, value in totals.items()}}
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite semantic-energy metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "semantic_energy_epoch", **row}, sort_keys=True), flush=True)
    adapter.eval()
    return history


@torch.no_grad()
def validate_adapter(
    adapter: SemanticEnergyAdapter,
    model: nn.Module,
    representation: nn.Module,
    pairs: Sequence[object],
    indices: Sequence[int],
    embeddings: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    batch_size = int(preregistration["adapter_batch_size"])
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    base.seed_everything(int(preregistration["probe_seed"]))
    for start in range(0, len(indices), batch_size):
        chosen = list(indices[start : start + batch_size])
        _loss, metrics = semantic_batch_loss(
            adapter,
            model,
            representation,
            [pairs[index] for index in chosen],
            embeddings,
            chosen,
            preregistration,
            device,
        )
        for name, value in metrics.items():
            totals[name] += value
        batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


def load_graph_stack(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    bundle: Mapping[str, object],
    device: torch.device,
):
    representation, representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    model = fresh.build_model(representation_config, preregistration, bundle["vocabulary"], device)
    checkpoint = torch.load(args.canonical_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(dict(checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)
    representation.eval().requires_grad_(False)
    return model, representation, representation_config, representation_summary


def run_train(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed semantic-energy train exists: {summary_path}")
    check_locked_inputs(
        preregistration,
        {
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
            "e1_manifest_sha256": args.e1_manifest,
        },
    )
    prepare = read_json(args.prepare_summary)
    if file_sha256(args.fit_probe_bundle) != dict(prepare["artifacts"])["fit_probe_bundle_sha256"]:
        raise ValueError("Prepared fit/probe bundle drift")
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != PROTOCOL:
        raise ValueError("Fit/probe bundle protocol drift")
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["adapter_training_seed"]))
    e1 = read_json(args.e1_manifest)
    embeddings, llm_manifest = load_constraint_embeddings(
        pairs,
        set(validation_indices),
        e1,
        preregistration,
        args,
        device,
    )
    embedding_cache_path = args.output_dir / "constraint_embeddings.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "embeddings": embeddings,
            "llm_manifest": llm_manifest,
            "pair_count": len(pairs),
        },
        embedding_cache_path,
    )
    model, representation, representation_config, _representation_summary = load_graph_stack(
        args, preregistration, bundle, device
    )
    token_shape = np.asarray(pairs[0].condition).shape
    if token_shape != (int(preregistration["token_count"]), int(preregistration["condition_dim"])):
        raise ValueError(f"Condition token shape drift: {token_shape}")
    adapter = SemanticEnergyAdapter(
        embedding_dim=int(embeddings["matched"].shape[1]),
        hidden_dim=int(preregistration["adapter_hidden_dim"]),
        token_count=int(preregistration["token_count"]),
        condition_dim=int(preregistration["condition_dim"]),
        latent_dim=int(preregistration["transport_dim"]),
        source_dim=int(representation_config["node_dim"]),
        residual_scale=float(preregistration["adapter_residual_scale"]),
    ).to(device)
    mean_tokens = torch.from_numpy(
        np.mean(
            np.stack([np.asarray(pairs[index].condition, dtype=np.float32) for index in train_indices]),
            axis=0,
        )
    ).to(device)
    adapter.initialize_token_bias(mean_tokens)
    training = train_adapter(
        adapter,
        model,
        representation,
        pairs,
        train_indices,
        embeddings,
        preregistration,
        device,
    )
    validation = validate_adapter(
        adapter,
        model,
        representation,
        pairs,
        validation_indices,
        embeddings,
        preregistration,
        device,
    )
    gates = dict(preregistration["representation_gates"])
    checks = {
        "hard_negative_accuracy": validation["hard_negative_accuracy"] >= float(gates["hard_negative_accuracy"]),
        "semantic_score_margin": validation["semantic_score_margin"] >= float(gates["semantic_score_margin"]),
        "token_mse_ratio_vs_zero": validation["token_mse_ratio_vs_zero"] <= float(gates["token_mse_ratio_vs_zero"]),
        "matched_flow_advantage": validation["matched_flow_advantage"] >= float(gates["matched_flow_advantage"]),
    }
    passed = all(checks.values())
    checkpoint_path = args.output_dir / "semantic_energy_adapter.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "state_dict": adapter.state_dict(),
            "embedding_dim": int(embeddings["matched"].shape[1]),
            "hidden_dim": int(preregistration["adapter_hidden_dim"]),
            "token_count": int(preregistration["token_count"]),
            "condition_dim": int(preregistration["condition_dim"]),
            "latent_dim": int(preregistration["transport_dim"]),
            "source_dim": int(representation_config["node_dim"]),
            "residual_scale": float(preregistration["adapter_residual_scale"]),
        },
        checkpoint_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "train_and_probe",
        "decision": "advance_to_target_isolated_freeze" if passed else "stop_before_generation_no_semantic_representation_signal",
        "training": training,
        "probe_validation": validation,
        "representation_gate": {"passed": passed, "checks": checks, "thresholds": gates},
        "llm_manifest": llm_manifest,
        "artifacts": {
            "adapter_checkpoint_sha256": file_sha256(checkpoint_path),
            "embedding_cache_sha256": file_sha256(embedding_cache_path),
        },
        "contract": {
            "fit_target_access": True,
            "probe_target_access": True,
            "common_llm_prompt_contains_source": False,
            "generation_target_access": False,
            "frozen_canonical_graph_jump": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 3


def load_generation_pairs(
    path: Path, preregistration: Mapping[str, object]
) -> tuple[list[object], list[dict[str, object]]]:
    payload = read_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("role") != "constraint_text_and_sources_without_targets":
        raise ValueError("Invalid target-free semantic-energy generation manifest")
    records = list(payload["records"])
    pairs = []
    for expected_index, record in enumerate(records):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("Generation condition order drift")
        source = graph.molecule_example(
            str(record["source_smiles"]),
            int(preregistration["max_atoms"]),
            int(preregistration["fingerprint_bits"]),
        )
        if source is None:
            raise ValueError(f"Cannot materialize source {expected_index}")
        row = dict(record["condition_row"])
        pairs.append(
            SimpleNamespace(
                row=row,
                source_smiles=str(record["source_smiles"]),
                source=source,
                condition=np.zeros(
                    (int(preregistration["token_count"]), int(preregistration["condition_dim"])),
                    dtype=np.float32,
                ),
                property_count=int(record["property_count"]),
                task=str(record["task"]),
            )
        )
    return pairs, records


def embed_generation_records(
    records: Sequence[Mapping[str, object]],
    preregistration: Mapping[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    variant_to_key = {
        "language_matched": "matched",
        "language_reversed": "reversed",
        "language_scrambled": "scrambled",
        "language_property_swap": "property_swap",
    }
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm, tokenizer = operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=False
    )
    try:
        embeddings = {
            arm: embed_texts(
                llm,
                tokenizer,
                [str(dict(record["instructions"])[key]) for record in records],
                preregistration,
                device,
            )
            for arm, key in variant_to_key.items()
        }
    finally:
        del llm
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prompts = [dict(record["instructions"]) for record in records]
    return embeddings, {
        "prompt_records_sha256": text_sha256(json.dumps(prompts, sort_keys=True, separators=(",", ":"))),
        "common_llm_prompt_contains_source": False,
        "embedding_dim": int(embeddings["language_matched"].shape[1]),
    }


def semantic_transport_particles(
    model: nn.Module,
    representation: nn.Module,
    source_example: object,
    condition_tokens: np.ndarray,
    particles: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
    *,
    adapter: SemanticEnergyAdapter,
    language_embedding: torch.Tensor,
    diagnostics: defaultdict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    attempts = int(particles.shape[0])
    chunk = int(preregistration["sample_batch_size"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    source = base.move_graph_batch(graph.collate([source_example]), device)
    tokens = torch.from_numpy(np.repeat(condition_tokens[None, ...], attempts, axis=0)).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        source_node, _ = representation.encode(source)
    source_pool = pooled_source_node(source_node, source["node_mask"]).float()
    embedding = language_embedding.to(device).float().unsqueeze(0).expand(attempts, -1)
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
                ).float()
            residual = adapter.velocity_residual(
                latent[start : start + count],
                source_pool.expand(count, -1),
                flow_time.float(),
                embedding[start : start + count],
            )
            diagnostics["residual_norm_sum"] += float(residual.norm(dim=1).mean().detach().cpu())
            diagnostics["residual_steps"] += 1.0
            velocities.append(velocity + residual)
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
        latent = (center + residual).detach()
    normalized = F.normalize(latent, dim=1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = ~torch.eye(attempts, dtype=torch.bool, device=device)
    return latent, {
        "final_particle_mean_abs_cosine": float(cosine[off_diagonal].abs().mean().detach().cpu()),
        "final_particle_max_abs_cosine": float(cosine[off_diagonal].abs().max().detach().cpu()),
        "final_particle_centered_rms": float(
            (latent - latent.mean(dim=0, keepdim=True)).norm(dim=1).square().mean().sqrt().detach().cpu()
        ),
        "minimum_transport_particle_rms": minimum_observed_rms,
    }


def mean_unique_smiles(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = str(row.get("generated_smiles", "") or "")
        if value:
            grouped[str(row["condition_id"])].add(value)
    return float(np.mean([len(values) for values in grouped.values()])) if grouped else 0.0


def run_freeze(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_locked_inputs(
        preregistration,
        {
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
        },
    )
    prepare = read_json(args.prepare_summary)
    artifacts = dict(prepare["artifacts"])
    if file_sha256(args.fit_probe_bundle) != artifacts["fit_probe_bundle_sha256"]:
        raise ValueError("Freeze fit/probe bundle drift")
    if file_sha256(args.generation_conditions) != artifacts["generation_conditions_sha256"]:
        raise ValueError("Freeze generation manifest drift")
    train_summary = read_json(args.train_summary)
    if not bool(dict(train_summary["representation_gate"])["passed"]):
        raise ValueError("Semantic representation gate failed; generation is forbidden")
    if file_sha256(args.adapter_checkpoint) != dict(train_summary["artifacts"])["adapter_checkpoint_sha256"]:
        raise ValueError("Semantic-energy adapter checkpoint drift")
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    device = base.resolve_device(str(args.device))
    pairs, records = load_generation_pairs(args.generation_conditions, preregistration)
    embeddings, llm_manifest = embed_generation_records(records, preregistration, args, device)
    model, representation, representation_config, representation_summary = load_graph_stack(
        args, preregistration, bundle, device
    )
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    adapter = SemanticEnergyAdapter(
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        token_count=int(checkpoint["token_count"]),
        condition_dim=int(checkpoint["condition_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        source_dim=int(checkpoint["source_dim"]),
        residual_scale=float(checkpoint["residual_scale"]),
    ).to(device)
    adapter.load_state_dict(dict(checkpoint["state_dict"]), strict=True)
    adapter.eval().requires_grad_(False)
    vocabulary = dict(bundle["vocabulary"])
    support = dict(bundle["support"])
    support_tensors = b40._device_support(support, device)
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    safe_support = DeadEndSafeSupport(exact_support)
    original_support = b41.viability_event_mask
    original_transport = b41.interacting_transport_particles
    try:
        b41.viability_event_mask = safe_support
        for arm in ARMS:
            arm_dir = args.output_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = arm_dir / "frozen_candidates.csv"
            summary_path = arm_dir / "summary.json"
            if summary_path.exists() or candidate_path.exists():
                raise ValueError(f"Completed semantic-energy arm exists: {arm_dir}")
            rows = []
            diagnostics: defaultdict[str, float] = defaultdict(float)
            started = time.perf_counter()
            for index, original_pair in enumerate(pairs):
                pair = copy.copy(original_pair)
                if arm == "numeric_canonical":
                    pair.condition = hierarchical.property_latent_slot_tokens(
                        pair.row, int(preregistration["condition_dim"])
                    )
                    b41.interacting_transport_particles = original_transport
                else:
                    embedding = embeddings[arm][index]
                    with torch.no_grad():
                        pair.condition = (
                            adapter.condition_tokens(embedding.to(device).unsqueeze(0))[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                    def adapted(*transport_args, _embedding=embedding, **transport_kwargs):
                        return semantic_transport_particles(
                            *transport_args,
                            **transport_kwargs,
                            adapter=adapter,
                            language_embedding=_embedding,
                            diagnostics=diagnostics,
                        )

                    b41.interacting_transport_particles = adapted
                try:
                    generated = b41.sample_from_source(
                        model,
                        representation,
                        vocabulary,
                        support,
                        support_tensors,
                        pair.source,
                        np.asarray(pair.condition, dtype=np.float32),
                        preregistration,
                        device,
                        int(preregistration["generation_seed"]) * 100000 + index,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "stage": "sample_failed",
                                "arm": arm,
                                "condition_id": records[index]["condition_id"],
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    generated = []
                generated = (list(generated) + [{"generated_smiles": ""}] * 20)[:20]
                for attempt, candidate in enumerate(generated, start=1):
                    rows.append(
                        {
                            "condition_id": records[index]["condition_id"],
                            "pair_index": index,
                            "attempt": attempt,
                            "property_count": int(pair.property_count),
                            "task": pair.task,
                            "source_smiles": pair.source_smiles,
                            "arm": arm,
                            **candidate,
                        }
                    )
                if (index + 1) % 12 == 0 or index + 1 == len(pairs):
                    print(
                        json.dumps(
                            {"stage": "freeze_progress", "arm": arm, "done": index + 1, "total": len(pairs)},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
            if len(rows) != expected:
                raise RuntimeError(f"{arm} expected {expected} rows, found {len(rows)}")
            base.write_candidate_rows(candidate_path, rows)
            arm_summary = {
                "protocol": PROTOCOL,
                "stage": "target_isolated_freeze",
                "arm": arm,
                "candidate_rows": len(rows),
                "conditions": len(pairs),
                "attempts_per_condition": 20,
                "mean_unique_smiles": mean_unique_smiles(rows),
                "mean_adapter_residual_norm": (
                    diagnostics["residual_norm_sum"] / max(1.0, diagnostics["residual_steps"])
                ),
                "elapsed_sec": round(time.perf_counter() - started, 1),
                "artifacts": {"frozen_candidates_sha256": file_sha256(candidate_path)},
                "contract": {
                    "generation_target_path_accepted": False,
                    "common_llm_prompt_contains_source": False,
                    "exact_raw_attempts_per_condition": 20,
                    "molecular_candidate_ranking": False,
                    "oracle_selection": False,
                    "retry_or_resampling": False,
                },
            }
            summary_path.write_text(json.dumps(arm_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        b41.interacting_transport_particles = original_transport
        b41.viability_event_mask = original_support
    freeze_summary = {
        "protocol": PROTOCOL,
        "stage": "all_arms_frozen",
        "decision": "await_target_evaluation",
        "arms": list(ARMS),
        "conditions": len(pairs),
        "candidate_rows_per_arm": len(pairs) * 20,
        "llm_manifest": llm_manifest,
        "representation_protocol": representation_summary.get("protocol"),
        "contract": {
            "generation_target_access": False,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(freeze_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(freeze_summary, indent=2, sort_keys=True), flush=True)
    return 0


def coerce_frozen_rows(path: Path) -> list[dict[str, object]]:
    return fresh.fresh_v3.coerce_frozen_rows(path)


def load_evaluation_pairs(path: Path, preregistration: Mapping[str, object]) -> list[object]:
    payload = read_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("role") != "sealed_post_freeze_targets":
        raise ValueError("Invalid sealed semantic-energy evaluation targets")
    pairs = []
    for expected_index, record in enumerate(payload["records"]):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("Evaluation target order drift")
        row = {str(key): str(value) for key, value in dict(record["row"]).items()}
        aligned = base.align_pair(
            str(record["source_smiles"]),
            str(record["target_smiles"]),
            max_atoms=int(preregistration["max_atoms"]),
            fingerprint_bits=int(preregistration["fingerprint_bits"]),
            timeout=int(preregistration["mcs_timeout"]),
            min_common_fraction=float(preregistration["min_common_fraction"]),
        )
        if aligned is None:
            raise ValueError(f"Cannot reconstruct sealed pair {expected_index}")
        source, target, common = aligned
        pairs.append(
            base.EditPair(
                row=row,
                source_smiles=str(record["source_smiles"]),
                target_smiles=str(record["target_smiles"]),
                source=source,
                target=target,
                condition=np.zeros((1, int(preregistration["condition_dim"])), dtype=np.float32),
                property_count=int(record["property_count"]),
                task=base.task_key(row),
                common_atoms=int(common),
            )
        )
    return pairs


def task_breakdown(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_condition: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    by_task: defaultdict[str, list[list[Mapping[str, object]]]] = defaultdict(list)
    for values in by_condition.values():
        by_task[str(values[0]["task"])].append(values)
    return {
        task: {
            "conditions": len(conditions),
            "strict_any20": sum(
                any(bool(row["strict_success"]) for row in values) for values in conditions
            )
            / len(conditions),
        }
        for task, conditions in sorted(by_task.items())
    }


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed semantic-energy evaluation exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    if file_sha256(args.evaluation_targets) != dict(prepare["artifacts"])["evaluation_targets_sha256"]:
        raise ValueError("Sealed semantic-energy targets drift")
    pairs = load_evaluation_pairs(args.evaluation_targets, preregistration)
    expected = len(pairs) * 20
    metrics_by_arm = {}
    strict = {}
    for arm in ARMS:
        arm_dir = args.frozen_root / arm
        arm_summary = read_json(arm_dir / "summary.json")
        candidate_path = arm_dir / "frozen_candidates.csv"
        if file_sha256(candidate_path) != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]:
            raise ValueError(f"Frozen semantic-energy candidates drift: {arm}")
        frozen = coerce_frozen_rows(candidate_path)
        if len(frozen) != expected:
            raise ValueError(f"{arm} expected {expected} rows, found {len(frozen)}")
        evaluated, metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = dict(metrics)
        metrics["by_task"] = task_breakdown(evaluated)
        evaluated_path = args.output_dir / f"evaluated_{arm}.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        metrics_by_arm[arm] = {
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
        strict[arm] = float(metrics["strict_any20"])
    matched = dict(metrics_by_arm["language_matched"])["metrics"]
    control_ceiling = max(strict[arm] for arm in LANGUAGE_ARMS if arm != "language_matched")
    semantic_margin = strict["language_matched"] - control_ceiling
    gates = dict(preregistration["generation_gates"])
    checks = {
        "candidate_rows": all(int(dict(metrics_by_arm[arm]["metrics"])["candidate_rows"]) == expected for arm in ARMS),
        "exact_attempts": all(int(dict(metrics_by_arm[arm]["metrics"])["attempted_per_condition"]) == 20 for arm in ARMS),
        "all_arm_validity": all(float(dict(metrics_by_arm[arm]["metrics"])["validity"]) >= float(gates["validity"]) for arm in ARMS),
        "matched_source_tanimoto": float(matched["mean_source_tanimoto"]) >= float(gates["mean_source_tanimoto"]),
        "matched_unique": float(matched["mean_unique_valid"]) >= float(gates["mean_unique_valid"]),
        "matched_vs_numeric": strict["language_matched"] - strict["numeric_canonical"] >= float(gates["strict_delta_vs_numeric"]),
        "semantic_margin": semantic_margin >= float(gates["semantic_margin"]),
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "stage": "post_freeze_evaluation",
        "decision": "advance_semantic_energy_to_one_new_fresh_confirmation" if passed else "stop_semantic_energy_without_dev_retuning",
        "arms": metrics_by_arm,
        "strict_any20": strict,
        "effects": {
            "matched_delta_vs_numeric": strict["language_matched"] - strict["numeric_canonical"],
            "matched_semantic_margin_vs_best_control": semantic_margin,
        },
        "gate": {"passed": passed, "checks": checks, "thresholds": gates},
        "contract": {
            "train_only_reused_method_dev": True,
            "formal_fresh_confirmation": False,
            "generation_target_access": False,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "repeat_on_same_dev_for_retuning": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    if args.stage == "prepare":
        return run_prepare(args, preregistration)
    if args.stage == "train":
        return run_train(args, preregistration)
    if args.stage == "freeze":
        return run_freeze(args, preregistration)
    if args.stage == "evaluate":
        return run_evaluate(args, preregistration)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
