#!/usr/bin/env python3
"""Preregistered semantic-set compiler representation gate.

This experiment asks one narrow question: can a frozen Common LLM encode each
signed molecular-property phrase so that a permutation-invariant set network
recovers the explicit numeric condition and the corresponding frozen graph
transport field?  It is deliberately molecule-target-free.  The process only
accepts a train-only numeric basis and a source-only graph manifest; it has no
CLI for molecule targets, property oracles, candidate generation, or ranking.

The semantic set compiler is a DeepSets encoder with shared element weights.
Its pooled state emits signed property coefficients and a small low-rank token
field.  The latter is distilled from the frozen numeric canonical condition.
Unseen 3- and 4-property sets are compared with numeric canonical control, the
locked V5 token-slot router, and sign-reversed language.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations, product
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

import mass_conserving_property_set_router_v5 as v5  # noqa: E402
import mass_conserving_router_table1_bridge_v6 as v6  # noqa: E402
import property_factorized_language_graph_basis_v1 as property_basis  # noqa: E402
import semantic_energy_graph_jump_v1 as semantic  # noqa: E402
import token_slot_lora_property_compiler_v3 as v3  # noqa: E402


PROTOCOL = "train_only_permutation_invariant_semantic_set_compiler_v10"
ARMS = (
    "numeric_canonical",
    "semantic_set_compiler",
    "token_slot",
    "reversed_language",
)
base = property_basis.base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--basis-bundle", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--canonical-checkpoint", required=True, type=Path)
    parser.add_argument("--sft-adapter-dir", required=True, type=Path)
    parser.add_argument("--v5-root", required=True, type=Path)
    parser.add_argument("--e1-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "representation": "permutation_invariant_signed_property_set",
        "set_architecture": "deepsets_shared_phi_sum_rho_low_rank_field",
        "common_llm_frozen": True,
        "numeric_canonical_distillation": True,
        "fit_property_cardinalities": [1, 2],
        "probe_property_cardinalities": [3, 4],
        "probe_property_sets_unseen_in_fit": True,
        "generation_target_access": False,
        "molecule_target_path_accepted": False,
        "property_oracle_access": False,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "official_test_access": False,
        "threshold_search": False,
        "single_seed": True,
        "arms": list(ARMS),
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"V10 preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "V10 implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V10 locked inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    locked = dict(preregistration["locked_inputs"])
    drift = {
        name: {"expected": locked.get(name), "actual": digest}
        for name, digest in actual.items()
        if locked.get(name) != digest
    }
    if drift:
        raise ValueError(f"V10 locked-input drift: {drift}")
    return actual


def specs_signature(specs: Sequence[tuple[str, int]]) -> str:
    return "|".join(f"{name}:{int(direction):+d}" for name, direction in sorted(specs))


def fit_specs(property_columns: Sequence[str]) -> list[list[tuple[str, int]]]:
    rows: list[list[tuple[str, int]]] = []
    for name in map(str, property_columns):
        for direction in (-1, 1):
            rows.append([(name, direction)])
    for left, right in combinations(map(str, property_columns), 2):
        for directions in product((-1, 1), repeat=2):
            rows.append([(left, directions[0]), (right, directions[1])])
    return rows


def sample_probe_specs(
    property_columns: Sequence[str], preregistration: Mapping[str, object]
) -> list[list[tuple[str, int]]]:
    rng = random.Random(int(preregistration["probe_seed"]))
    rows: list[list[tuple[str, int]]] = []
    seen_sets: set[tuple[str, ...]] = set()
    per_cardinality = int(preregistration["probe_specs_per_cardinality"])
    for cardinality in map(int, preregistration["probe_property_cardinalities"]):
        accepted = 0
        while accepted < per_cardinality:
            names = tuple(sorted(rng.sample(list(map(str, property_columns)), cardinality)))
            if names in seen_sets:
                continue
            seen_sets.add(names)
            rows.append([(name, rng.choice((-1, 1))) for name in names])
            accepted += 1
    return rows


def examples_from_specs(
    specs_rows: Sequence[Sequence[tuple[str, int]]],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    templates: Sequence[str],
    prefix: str,
) -> list[v3.TextExample]:
    output: list[v3.TextExample] = []
    for index, specs in enumerate(specs_rows):
        for template in templates:
            output.append(
                v3.make_example(
                    specs,
                    property_columns,
                    property_names,
                    template,
                    f"{prefix}_{index:04d}_{template}",
                )
            )
    return output


def phrase_rows(examples: Sequence[v3.TextExample], reverse_order: bool = False) -> list[list[str]]:
    output: list[list[str]] = []
    for example in examples:
        values = [phrase for _index, phrase in sorted(example.phrases.items())]
        if reverse_order:
            values = list(reversed(values))
        output.append(values)
    return output


def reversed_examples(
    specs_rows: Sequence[Sequence[tuple[str, int]]],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    templates: Sequence[str],
) -> list[v3.TextExample]:
    reversed_rows = [
        [(name, -int(direction)) for name, direction in specs] for specs in specs_rows
    ]
    examples = examples_from_specs(
        reversed_rows, property_columns, property_names, templates, "probe_reversed"
    )
    original_targets = [
        property_basis.property_vector(specs, property_columns) for specs in specs_rows
    ]
    repeated = []
    for target in original_targets:
        repeated.extend([target] * len(templates))
    return [
        v3.TextExample(example.text, target, example.phrases, example.key)
        for example, target in zip(examples, repeated)
    ]


class SemanticSetCompiler(nn.Module):
    """Shared element encoder plus order-invariant aggregation and low-rank field."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        property_count: int,
        basis: torch.Tensor,
        field_rank: int,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.coefficient_head = nn.Linear(int(hidden_dim), int(property_count))
        self.support_head = nn.Linear(int(hidden_dim), int(property_count))
        self.field_head = nn.Linear(int(hidden_dim), int(field_rank))
        self.register_buffer("numeric_basis", basis.float().clone())
        token_dim = int(basis.shape[1])
        self.residual_field = nn.Parameter(
            torch.randn(int(field_rank), token_dim) * (0.001 / math.sqrt(max(1, field_rank)))
        )

    def forward(
        self, element_embeddings: torch.Tensor, element_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.phi(element_embeddings.float())
        mask = element_mask.float().unsqueeze(-1)
        cardinality = mask.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * mask).sum(dim=1) / cardinality.sqrt()
        state = self.rho(pooled)
        support_logits = self.support_head(state)
        coefficients = torch.tanh(self.coefficient_head(state)) * torch.sigmoid(
            support_logits
        )
        rank_weights = torch.tanh(self.field_head(state))
        design = torch.cat(
            [torch.ones(len(state), 1, device=state.device), coefficients], dim=-1
        )
        tokens = design @ self.numeric_basis
        tokens = tokens + rank_weights @ self.residual_field
        return coefficients, support_logits, rank_weights, tokens


@torch.no_grad()
def encode_unique_phrases(
    llm: object,
    tokenizer: object,
    phrases: Sequence[str],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    unique = sorted(set(map(str, phrases)))
    body = semantic.operator._transformer_body(llm)
    output: dict[str, torch.Tensor] = {}
    batch_size = int(preregistration["llm_embedding_batch_size"])
    for start in range(0, len(unique), batch_size):
        chosen = unique[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                semantic.constraint_only_chat(phrase),
                tokenize=False,
                add_generation_prompt=True,
            )
            for phrase in chosen
        ]
        encoded = tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=int(preregistration["llm_max_length"]),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            hidden = body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        final_index = attention_mask.sum(dim=-1).long().sub(1).clamp_min(0)
        pooled = hidden[torch.arange(len(chosen), device=device), final_index].float().cpu()
        for phrase, embedding in zip(chosen, pooled):
            output[phrase] = embedding
    return output


def materialize_sets(
    rows: Sequence[Sequence[str]], cache: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(row) for row in rows)
    embedding_dim = int(next(iter(cache.values())).numel())
    embeddings = torch.zeros(len(rows), maximum, embedding_dim, dtype=torch.float32)
    mask = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for index, row in enumerate(rows):
        for position, phrase in enumerate(row):
            embeddings[index, position] = cache[str(phrase)]
            mask[index, position] = True
    return embeddings, mask


def compose_numeric_tokens(
    coefficients: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    design = torch.cat(
        [torch.ones(len(coefficients), 1, device=coefficients.device), coefficients.float()],
        dim=-1,
    )
    return design @ basis.to(coefficients.device)


def train_compiler(
    compiler: SemanticSetCompiler,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        compiler.parameters(),
        lr=float(preregistration["compiler_learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    batch_size = int(preregistration["training_batch_size"])
    generator = torch.Generator().manual_seed(int(preregistration["training_seed"]))
    history: list[dict[str, float]] = []
    compiler.train()
    for epoch in range(1, int(preregistration["training_epochs"]) + 1):
        order = torch.randperm(len(targets), generator=generator)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            x = embeddings[chosen].to(device)
            m = mask[chosen].to(device)
            target = targets[chosen].to(device)
            coefficients, support_logits, _rank, tokens = compiler(x, m)
            numeric_tokens = compose_numeric_tokens(target, compiler.numeric_basis)
            coefficient_loss = F.mse_loss(coefficients, target)
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits, target.ne(0).float()
            )
            denominator = numeric_tokens.square().mean().clamp_min(1e-8)
            token_loss = F.mse_loss(tokens, numeric_tokens) / denominator
            loss = (
                coefficient_loss
                + float(preregistration["support_loss_weight"]) * support_loss
                + float(preregistration["token_loss_weight"]) * token_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                compiler.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["coefficient_loss"] += float(coefficient_loss.detach())
            totals["support_loss"] += float(support_loss.detach())
            totals["token_loss_ratio"] += float(token_loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite V10 training metrics: {row}")
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(preregistration["training_epochs"]):
            print(json.dumps({"stage": "semantic_set_epoch", **row}, sort_keys=True), flush=True)
    compiler.eval()
    return history


@torch.no_grad()
def predict_compiler(
    compiler: SemanticSetCompiler,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients = []
    support_logits = []
    tokens = []
    for start in range(0, len(embeddings), int(batch_size)):
        coefficient, support, _rank, token = compiler(
            embeddings[start : start + batch_size].to(device),
            mask[start : start + batch_size].to(device),
        )
        coefficients.append(coefficient.cpu())
        support_logits.append(support.cpu())
        tokens.append(token.cpu())
    return torch.cat(coefficients), torch.cat(support_logits), torch.cat(tokens)


def set_metrics(
    coefficients: torch.Tensor, support_logits: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    expected = target.ne(0)
    predicted = support_logits.sigmoid().ge(0.5)
    true_positive = (predicted & expected).sum().float()
    precision = true_positive / predicted.sum().clamp_min(1)
    recall = true_positive / expected.sum().clamp_min(1)
    active_sign = (
        torch.sign(coefficients[expected]) == torch.sign(target[expected])
    ).float()
    exact_support = predicted.eq(expected).all(dim=-1)
    exact_signed = exact_support & (
        (torch.sign(coefficients) == torch.sign(target)) | ~expected
    ).all(dim=-1)
    return {
        "coefficient_mse": float(F.mse_loss(coefficients, target)),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "exact_support_rate": float(exact_support.float().mean()),
        "active_sign_accuracy": float(active_sign.mean()),
        "exact_signed_set_rate": float(exact_signed.float().mean()),
    }


def token_metrics(
    arms: Mapping[str, torch.Tensor], numeric: torch.Tensor
) -> dict[str, dict[str, float]]:
    intercept = numeric.new_zeros(numeric.shape)
    denominator = F.mse_loss(intercept, numeric).clamp_min(1e-12)
    output: dict[str, dict[str, float]] = {}
    for name, tokens in arms.items():
        mse = F.mse_loss(tokens, numeric)
        cosine = F.cosine_similarity(tokens, numeric, dim=-1).mean()
        output[name] = {
            "mse": float(mse),
            "mse_ratio_vs_zero": float(mse / denominator),
            "cosine_to_numeric": float(cosine),
        }
    return output


def paired_bootstrap_advantage(
    baseline_error: torch.Tensor,
    compiler_error: torch.Tensor,
    seed: int,
    samples: int,
) -> dict[str, float]:
    delta = (baseline_error.double() - compiler_error.double()).cpu()
    generator = torch.Generator().manual_seed(int(seed))
    means = []
    for _ in range(int(samples)):
        indices = torch.randint(len(delta), (len(delta),), generator=generator)
        means.append(delta[indices].mean())
    distribution = torch.stack(means).sort().values
    lower_index = max(0, int(0.025 * len(distribution)))
    upper_index = min(len(distribution) - 1, int(0.975 * len(distribution)))
    return {
        "mean": float(delta.mean()),
        "ci95_lower": float(distribution[lower_index]),
        "ci95_upper": float(distribution[upper_index]),
    }


def per_row_token_error(tokens: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
    numerator = (tokens - numeric).square().mean(dim=-1)
    denominator = numeric.square().mean(dim=-1).clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def graph_velocity_metrics(
    graph_model: nn.Module,
    representation: nn.Module,
    sources: Sequence[object],
    arm_tokens: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    count = min(
        len(sources),
        len(next(iter(arm_tokens.values()))),
        int(preregistration["graph_probe_sources"]),
    )
    batch_size = int(preregistration["graph_probe_batch_size"])
    totals: defaultdict[str, float] = defaultdict(float)
    per_source: defaultdict[str, list[float]] = defaultdict(list)
    elements = 0
    base.seed_everything(int(preregistration["graph_probe_seed"]))
    generator = torch.Generator(device=device).manual_seed(
        int(preregistration["graph_probe_seed"])
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        selected = list(sources[start:stop])
        source = base.move_graph_batch(base.graph.collate(selected), device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            source_node, _source_edge = representation.encode(source)
        current = torch.randn(
            len(selected),
            int(preregistration["transport_dim"]),
            generator=generator,
            device=device,
        )
        flow_time = torch.full(
            (len(selected),),
            float(preregistration["probe_flow_time"]),
            device=device,
            dtype=source_node.dtype,
        )
        velocities = {}
        for name, token_rows in arm_tokens.items():
            tokens = token_rows[start:stop].to(device).view(
                len(selected),
                int(preregistration["token_count"]),
                int(preregistration["condition_dim"]),
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                velocity = graph_model.transport_velocity(
                    current,
                    flow_time,
                    source_node,
                    source["node_mask"],
                    tokens,
                ).float()
            velocities[name] = velocity
        numeric = velocities["numeric_canonical"]
        energy = numeric.square().sum().clamp_min(1e-12)
        for name, velocity in velocities.items():
            delta = velocity - numeric
            totals[f"{name}:squared_error"] += float(delta.square().sum())
            totals[f"{name}:numeric_energy"] += float(energy)
            totals[f"{name}:cosine_sum"] += float(
                F.cosine_similarity(velocity, numeric, dim=-1).sum()
            )
            row_error = delta.square().mean(dim=-1) / numeric.square().mean(
                dim=-1
            ).clamp_min(1e-12)
            per_source[name].extend(map(float, row_error.cpu().tolist()))
        elements += len(selected)
    output = {}
    for name in arm_tokens:
        output[name] = {
            "relative_mse_to_numeric_velocity": totals[f"{name}:squared_error"]
            / max(totals[f"{name}:numeric_energy"], 1e-12),
            "cosine_to_numeric_velocity": totals[f"{name}:cosine_sum"]
            / max(1, elements),
            "per_source_relative_mse": per_source[name],
        }
    return output


def load_frozen_language_stack(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
):
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError(f"Missing PEFT for V10: {exc}") from exc
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm, tokenizer = semantic.operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=False
    )
    llm = llm.merge_and_unload()
    full_root = args.v5_root / "full"
    llm = peft.PeftModel.from_pretrained(
        llm,
        full_root / "lora_adapter",
        is_trainable=False,
        adapter_name="v10_frozen_semantic_encoder",
    ).to(device)
    checkpoint = torch.load(
        full_root / "structured_sparse_router.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("protocol") != v5.PROTOCOL or checkpoint.get("arm") != "full":
        raise ValueError("V10 V5 full checkpoint drift")
    router = v5.MassConservingPropertySetRouter(
        int(checkpoint["llm_hidden_dim"]),
        int(checkpoint["slot_dim"]),
        len(checkpoint["property_columns"]),
        int(checkpoint["max_instruction_cardinality"]),
        bool(checkpoint["use_token_slots"]),
    ).to(device)
    router.load_state_dict(dict(checkpoint["state_dict"]), strict=True)
    llm.eval().requires_grad_(False)
    router.eval().requires_grad_(False)
    return llm, tokenizer, router, [str(name) for name in checkpoint["property_columns"]]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V10 execution exists: {summary_path}")
    full_root = args.v5_root / "full"
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "basis_bundle_sha256": args.basis_bundle,
            "source_manifest_sha256": args.source_manifest,
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "common_sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "common_sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
            "v5_full_router_sha256": full_root / "structured_sparse_router.pt",
            "v5_full_summary_sha256": full_root / "summary.json",
            "v5_full_lora_config_sha256": full_root / "lora_adapter" / "adapter_config.json",
            "v5_full_lora_model_sha256": full_root / "lora_adapter" / "adapter_model.safetensors",
            "e1_manifest_sha256": args.e1_manifest,
        },
    )
    bundle = torch.load(args.basis_bundle, map_location="cpu", weights_only=False)
    if bundle.get("role") != "target_free_frozen_generation_basis_and_support":
        raise ValueError("V10 requires target-free numeric basis bundle")
    basis = torch.as_tensor(bundle["basis"], dtype=torch.float32)
    token_shape = tuple(int(value) for value in bundle["token_shape"])
    if token_shape != (
        int(preregistration["token_count"]),
        int(preregistration["condition_dim"]),
    ):
        raise ValueError(f"V10 token shape drift: {token_shape}")
    e1 = read_json(args.e1_manifest)
    property_names = dict(e1["property_names"])
    property_columns = [str(name) for name in bundle["property_columns"]]
    if len(property_columns) != int(preregistration["property_count"]):
        raise ValueError("V10 property vocabulary drift")

    fit_rows = fit_specs(property_columns)
    probe_rows = sample_probe_specs(property_columns, preregistration)
    fit_examples = examples_from_specs(
        fit_rows,
        property_columns,
        property_names,
        list(preregistration["fit_templates"]),
        "fit",
    )
    probe_examples = examples_from_specs(
        probe_rows,
        property_columns,
        property_names,
        list(preregistration["probe_templates"]),
        "probe",
    )
    reversed_probe = reversed_examples(
        probe_rows,
        property_columns,
        property_names,
        list(preregistration["probe_templates"]),
    )
    if {len(specs) for specs in fit_rows} != {1, 2} or {len(specs) for specs in probe_rows} != {3, 4}:
        raise ValueError("V10 fit/probe cardinality contract failed")
    fit_sets = {tuple(sorted(name for name, _direction in specs)) for specs in fit_rows}
    probe_sets = {tuple(sorted(name for name, _direction in specs)) for specs in probe_rows}
    if fit_sets & probe_sets:
        raise ValueError("V10 unseen property-set leakage")

    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["training_seed"]))
    llm, tokenizer, token_router, checkpoint_columns = load_frozen_language_stack(
        args, preregistration, device
    )
    if checkpoint_columns != property_columns:
        raise ValueError("V10 V5 property ordering drift")
    all_phrase_rows = [
        *phrase_rows(fit_examples),
        *phrase_rows(probe_examples),
        *phrase_rows(reversed_probe),
    ]
    phrase_cache = encode_unique_phrases(
        llm,
        tokenizer,
        [phrase for row in all_phrase_rows for phrase in row],
        preregistration,
        device,
    )
    fit_embeddings, fit_mask = materialize_sets(phrase_rows(fit_examples), phrase_cache)
    probe_embeddings, probe_mask = materialize_sets(phrase_rows(probe_examples), phrase_cache)
    reversed_embeddings, reversed_mask = materialize_sets(
        phrase_rows(reversed_probe), phrase_cache
    )
    permuted_embeddings, permuted_mask = materialize_sets(
        phrase_rows(probe_examples, reverse_order=True), phrase_cache
    )
    fit_targets = torch.stack([example.target for example in fit_examples])
    probe_targets = torch.stack([example.target for example in probe_examples])

    compiler = SemanticSetCompiler(
        embedding_dim=int(next(iter(phrase_cache.values())).numel()),
        hidden_dim=int(preregistration["set_hidden_dim"]),
        property_count=len(property_columns),
        basis=basis,
        field_rank=int(preregistration["field_rank"]),
    ).to(device)
    history = train_compiler(
        compiler,
        fit_embeddings,
        fit_mask,
        fit_targets,
        preregistration,
        device,
    )
    compiler_coefficients, compiler_support, compiler_tokens = predict_compiler(
        compiler,
        probe_embeddings,
        probe_mask,
        int(preregistration["probe_batch_size"]),
        device,
    )
    reversed_coefficients, reversed_support, reversed_tokens = predict_compiler(
        compiler,
        reversed_embeddings,
        reversed_mask,
        int(preregistration["probe_batch_size"]),
        device,
    )
    permuted_coefficients, _permuted_support, permuted_tokens = predict_compiler(
        compiler,
        permuted_embeddings,
        permuted_mask,
        int(preregistration["probe_batch_size"]),
        device,
    )
    token_slot_coefficients, token_slot_support, _token_slot_cardinality = v5.predict_examples(
        llm,
        token_router,
        tokenizer,
        probe_examples,
        preregistration,
        device,
    )
    numeric_tokens = compose_numeric_tokens(probe_targets, basis)
    token_slot_tokens = compose_numeric_tokens(token_slot_coefficients, basis)
    arm_tokens = {
        "numeric_canonical": numeric_tokens,
        "semantic_set_compiler": compiler_tokens,
        "token_slot": token_slot_tokens,
        "reversed_language": reversed_tokens,
    }
    coefficient_metrics = {
        "semantic_set_compiler": set_metrics(
            compiler_coefficients, compiler_support, probe_targets
        ),
        "token_slot": set_metrics(
            token_slot_coefficients,
            torch.where(
                token_slot_support,
                torch.full_like(token_slot_coefficients, 20.0),
                torch.full_like(token_slot_coefficients, -20.0),
            ),
            probe_targets,
        ),
        "reversed_language": set_metrics(
            reversed_coefficients, reversed_support, probe_targets
        ),
    }
    tokens = token_metrics(arm_tokens, numeric_tokens)
    compiler_token_row_error = per_row_token_error(compiler_tokens, numeric_tokens)
    token_slot_row_error = per_row_token_error(token_slot_tokens, numeric_tokens)
    reversed_token_row_error = per_row_token_error(reversed_tokens, numeric_tokens)
    token_paired = {
        "semantic_set_vs_reversed": paired_bootstrap_advantage(
            reversed_token_row_error,
            compiler_token_row_error,
            int(preregistration["bootstrap_seed"]),
            int(preregistration["bootstrap_samples"]),
        ),
        "semantic_set_vs_token_slot": paired_bootstrap_advantage(
            token_slot_row_error,
            compiler_token_row_error,
            int(preregistration["bootstrap_seed"]) + 1,
            int(preregistration["bootstrap_samples"]),
        ),
    }
    permutation = {
        "coefficient_max_abs": float(
            (compiler_coefficients - permuted_coefficients).abs().max()
        ),
        "token_max_abs": float((compiler_tokens - permuted_tokens).abs().max()),
    }

    source_pairs, _source_records = v6.load_generation_pairs(
        args.source_manifest, preregistration
    )
    graph_model, representation, _config, _representation_summary = semantic.load_graph_stack(
        args, preregistration, bundle, device
    )
    velocity = graph_velocity_metrics(
        graph_model,
        representation,
        [pair.source for pair in source_pairs],
        arm_tokens,
        preregistration,
        device,
    )
    velocity_paired = {
        "semantic_set_vs_reversed": paired_bootstrap_advantage(
            torch.tensor(velocity["reversed_language"]["per_source_relative_mse"]),
            torch.tensor(velocity["semantic_set_compiler"]["per_source_relative_mse"]),
            int(preregistration["bootstrap_seed"]) + 2,
            int(preregistration["bootstrap_samples"]),
        ),
        "semantic_set_vs_token_slot": paired_bootstrap_advantage(
            torch.tensor(velocity["token_slot"]["per_source_relative_mse"]),
            torch.tensor(velocity["semantic_set_compiler"]["per_source_relative_mse"]),
            int(preregistration["bootstrap_seed"]) + 3,
            int(preregistration["bootstrap_samples"]),
        ),
    }

    matched_token_error = tokens["semantic_set_compiler"]["mse_ratio_vs_zero"]
    token_slot_error = tokens["token_slot"]["mse_ratio_vs_zero"]
    reversed_token_error = tokens["reversed_language"]["mse_ratio_vs_zero"]
    matched_velocity_error = velocity["semantic_set_compiler"][
        "relative_mse_to_numeric_velocity"
    ]
    token_slot_velocity_error = velocity["token_slot"][
        "relative_mse_to_numeric_velocity"
    ]
    reversed_velocity_error = velocity["reversed_language"][
        "relative_mse_to_numeric_velocity"
    ]
    gates = dict(preregistration["representation_gates"])
    checks = {
        "exact_support_rate": coefficient_metrics["semantic_set_compiler"][
            "exact_support_rate"
        ]
        >= float(gates["exact_support_rate"]),
        "active_sign_accuracy": coefficient_metrics["semantic_set_compiler"][
            "active_sign_accuracy"
        ]
        >= float(gates["active_sign_accuracy"]),
        "exact_signed_set_rate": coefficient_metrics["semantic_set_compiler"][
            "exact_signed_set_rate"
        ]
        >= float(gates["exact_signed_set_rate"]),
        "permutation_coefficient_max_abs": permutation["coefficient_max_abs"]
        <= float(gates["permutation_max_abs"]),
        "permutation_token_max_abs": permutation["token_max_abs"]
        <= float(gates["permutation_max_abs"]),
        "token_noninferiority_to_numeric": matched_token_error
        <= float(gates["token_mse_ratio_vs_zero"]),
        "matched_token_better_than_reversed": token_paired[
            "semantic_set_vs_reversed"
        ]["ci95_lower"]
        >= float(gates["matched_reversed_token_advantage_ci_lower"]),
        "set_token_better_than_token_slot": token_paired[
            "semantic_set_vs_token_slot"
        ]["ci95_lower"]
        >= float(gates["set_token_slot_advantage_ci_lower"]),
        "velocity_noninferiority_to_numeric": matched_velocity_error
        <= float(gates["velocity_relative_mse"]),
        "matched_velocity_better_than_reversed": velocity_paired[
            "semantic_set_vs_reversed"
        ]["ci95_lower"]
        >= float(gates["matched_reversed_velocity_advantage_ci_lower"]),
        "set_velocity_better_than_token_slot": velocity_paired[
            "semantic_set_vs_token_slot"
        ]["ci95_lower"]
        >= float(gates["set_token_slot_velocity_advantage_ci_lower"]),
    }
    passed = all(checks.values())

    checkpoint_path = args.output_dir / "semantic_set_compiler.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "state_dict": compiler.cpu().state_dict(),
            "property_columns": property_columns,
            "embedding_dim": int(next(iter(phrase_cache.values())).numel()),
            "hidden_dim": int(preregistration["set_hidden_dim"]),
            "field_rank": int(preregistration["field_rank"]),
            "token_shape": token_shape,
        },
        checkpoint_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "execution_complete_science_gate_deferred",
        "execution_status": "completed",
        "decision": "science_gate_deferred_to_separate_cpu_job",
        "training": {
            "common_llm_frozen": True,
            "compiler_fit_examples": len(fit_examples),
            "fit_unique_signed_specs": len(fit_rows),
            "fit_cardinalities": [1, 2],
            "history": history,
        },
        "probe": {
            "examples": len(probe_examples),
            "unique_property_sets": len(probe_rows),
            "cardinalities": [3, 4],
            "fit_probe_property_set_overlap": 0,
        },
        "coefficient_and_set_reconstruction": coefficient_metrics,
        "permutation_invariance": permutation,
        "numeric_token_equivalence": tokens,
        "paired_token_advantages": token_paired,
        "source_only_graph_velocity_equivalence": velocity,
        "paired_velocity_advantages": velocity_paired,
        "representation_gate_preview": {
            "passed": passed,
            "checks": checks,
            "thresholds": gates,
        },
        "artifacts": {
            "compiler_checkpoint_sha256": file_sha256(checkpoint_path),
            "locked_inputs": input_hashes,
        },
        "contract": {
            "common_llm_frozen": True,
            "numeric_canonical_distillation": True,
            "source_manifest_role": "constraint_text_and_sources_without_targets",
            "molecule_target_path_accepted": False,
            "molecule_target_access": False,
            "property_oracle_access": False,
            "generation_target_access": False,
            "molecule_generation": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "threshold_search": False,
            "official_test_access": False,
            "token_slot_training": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    del llm, tokenizer, token_router, graph_model, representation
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
