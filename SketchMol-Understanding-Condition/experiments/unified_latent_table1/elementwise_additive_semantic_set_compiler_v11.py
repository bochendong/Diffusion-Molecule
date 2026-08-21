#!/usr/bin/env python3
"""Final prospective representation gate for an additive semantic-set compiler.

V10 showed that a global DeepSets rho trained on 1--2-property sets did not
extrapolate to 3--4-property cardinalities.  V11 removes that failure mode
rather than tuning V10 on its opened probe.  A shared, frozen-LLM element head
predicts one property identity, one sign, and one low-rank token-field delta
for every language element independently; the set condition is their unordered
sum plus one numeric intercept.  Training uses singleton phrases only.  The
only science probe consists of newly preregistered 5--7-property sets.

The script accepts no molecule target, oracle, candidate, or ranking path.
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
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
for module_path in (SCRIPT_DIR, PROJECT_DIR, LATENT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import permutation_invariant_semantic_set_compiler_v10 as v10  # noqa: E402
import property_factorized_language_graph_basis_v1 as property_basis  # noqa: E402
import token_slot_lora_property_compiler_v3 as v3  # noqa: E402


PROTOCOL = "train_only_elementwise_additive_semantic_set_compiler_v11"
ARMS = v10.ARMS
base = v10.base


def build_parser() -> argparse.ArgumentParser:
    parser = v10.build_parser()
    parser.description = __doc__
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
        "representation": "elementwise_equivariant_signed_property_set",
        "set_architecture": "shared_element_identity_sign_field_then_unordered_sum",
        "common_llm_frozen": True,
        "numeric_canonical_distillation": True,
        "fit_property_cardinalities": [1],
        "probe_property_cardinalities": [5, 6, 7],
        "probe_property_sets_unseen_in_fit": True,
        "v10_probe_reuse": False,
        "final_representation_attempt": True,
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
        raise ValueError(f"V11 preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            f"V11 implementation drift: expected {payload.get('implementation_sha256')}, "
            f"found {actual}"
        )
    return payload


def singleton_fit_specs(property_columns: Sequence[str]) -> list[list[tuple[str, int]]]:
    return [
        [(str(name), direction)]
        for name in property_columns
        for direction in (-1, 1)
    ]


def sample_probe_specs(
    property_columns: Sequence[str], preregistration: Mapping[str, object]
) -> list[list[tuple[str, int]]]:
    rng = random.Random(int(preregistration["probe_seed"]))
    rows: list[list[tuple[str, int]]] = []
    seen: set[tuple[str, ...]] = set()
    per_cardinality = int(preregistration["probe_specs_per_cardinality"])
    for cardinality in map(int, preregistration["probe_property_cardinalities"]):
        accepted = 0
        while accepted < per_cardinality:
            names = tuple(sorted(rng.sample(list(map(str, property_columns)), cardinality)))
            if names in seen:
                continue
            seen.add(names)
            rows.append([(name, rng.choice((-1, 1))) for name in names])
            accepted += 1
    return rows


def render_fresh_example(
    specs: Sequence[tuple[str, int]],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    template: str,
    key: str,
    target_specs: Sequence[tuple[str, int]] | None = None,
) -> v3.TextExample:
    lookup = {str(name): index for index, name in enumerate(property_columns)}
    phrases: dict[int, str] = {}
    rendered: list[str] = []
    for name, direction in specs:
        readable = str(property_names.get(str(name), name))
        if template == "probe_enhance_suppress":
            phrase = f"{'enhance' if direction > 0 else 'suppress'} {readable}"
        elif template == "probe_promote_reduce":
            phrase = f"{'promote' if direction > 0 else 'reduce'} {readable}"
        else:
            raise ValueError(f"Unknown V11 fresh template: {template}")
        phrases[lookup[str(name)]] = phrase
        rendered.append(phrase)
    if template == "probe_enhance_suppress":
        text = "Seek a molecular edit that will " + "; ".join(rendered) + "."
    else:
        text = "The resulting molecule should " + " alongside ".join(rendered) + "."
    return v3.TextExample(
        text=text,
        target=property_basis.property_vector(target_specs or specs, property_columns),
        phrases=phrases,
        key=key,
    )


def probe_examples(
    rows: Sequence[Sequence[tuple[str, int]]],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    templates: Sequence[str],
    reversed_language: bool,
) -> list[v3.TextExample]:
    output = []
    for index, target_specs in enumerate(rows):
        language_specs = (
            [(name, -int(direction)) for name, direction in target_specs]
            if reversed_language
            else list(target_specs)
        )
        for template in templates:
            output.append(
                render_fresh_example(
                    language_specs,
                    property_columns,
                    property_names,
                    str(template),
                    f"{'reversed' if reversed_language else 'matched'}_{index:04d}_{template}",
                    target_specs=target_specs,
                )
            )
    return output


class ElementwiseAdditiveCompiler(nn.Module):
    """One shared semantic decoder per element; set output is an unordered sum."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        property_count: int,
        basis: torch.Tensor,
        field_rank: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(embedding_dim)),
            nn.Linear(int(embedding_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.property_head = nn.Linear(int(hidden_dim), int(property_count))
        self.direction_head = nn.Linear(int(hidden_dim), 1)
        self.field_head = nn.Linear(int(hidden_dim), int(field_rank))
        self.register_buffer("numeric_basis", basis.float().clone())
        self.residual_field = nn.Parameter(
            torch.randn(int(field_rank), int(basis.shape[1]))
            * (0.001 / math.sqrt(max(1, field_rank)))
        )

    def element_outputs(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encoder(embeddings.float())
        property_logits = self.property_head(state)
        direction_logits = self.direction_head(state).squeeze(-1)
        direction = torch.tanh(direction_logits)
        probabilities = torch.softmax(property_logits, dim=-1)
        coefficients = probabilities * direction.unsqueeze(-1)
        rank_weights = torch.tanh(self.field_head(state))
        token_delta = coefficients @ self.numeric_basis[1:]
        token_delta = token_delta + rank_weights @ self.residual_field
        return property_logits, direction_logits, coefficients, token_delta

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        property_logits, direction_logits, element_coefficients, token_delta = (
            self.element_outputs(embeddings)
        )
        active = mask.float().unsqueeze(-1)
        coefficients = (element_coefficients * active).sum(dim=1)
        tokens = self.numeric_basis[0].unsqueeze(0) + (token_delta * active).sum(dim=1)
        hard_property = property_logits.argmax(dim=-1)
        hard_direction = torch.where(
            direction_logits.ge(0),
            torch.ones_like(direction_logits),
            -torch.ones_like(direction_logits),
        )
        hard_coefficients = torch.zeros_like(coefficients)
        hard_coefficients.scatter_add_(
            1,
            hard_property,
            hard_direction * mask.float(),
        )
        hard_coefficients = hard_coefficients.clamp(-1.0, 1.0)
        return (
            coefficients,
            hard_coefficients,
            property_logits,
            direction_logits,
            tokens,
        )


def singleton_training_tensors(
    examples: Sequence[v3.TextExample],
    cache: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = []
    property_targets = []
    direction_targets = []
    coefficient_targets = []
    for example in examples:
        if len(example.phrases) != 1:
            raise ValueError("V11 training accepts singleton examples only")
        property_index, phrase = next(iter(example.phrases.items()))
        target = example.target
        direction = float(target[int(property_index)])
        embeddings.append(cache[str(phrase)])
        property_targets.append(int(property_index))
        direction_targets.append(1.0 if direction > 0 else 0.0)
        coefficient_targets.append(target)
    return (
        torch.stack(embeddings),
        torch.tensor(property_targets, dtype=torch.long),
        torch.tensor(direction_targets, dtype=torch.float32),
        torch.stack(coefficient_targets),
    )


def train_compiler(
    compiler: ElementwiseAdditiveCompiler,
    embeddings: torch.Tensor,
    property_targets: torch.Tensor,
    direction_targets: torch.Tensor,
    coefficient_targets: torch.Tensor,
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
    history = []
    compiler.train()
    for epoch in range(1, int(preregistration["training_epochs"]) + 1):
        order = torch.randperm(len(embeddings), generator=generator)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            x = embeddings[chosen].to(device)
            prop = property_targets[chosen].to(device)
            direction = direction_targets[chosen].to(device)
            coefficient = coefficient_targets[chosen].to(device)
            property_logits, direction_logits, predicted_coefficients, token_delta = (
                compiler.element_outputs(x)
            )
            property_loss = F.cross_entropy(property_logits, prop)
            direction_loss = F.binary_cross_entropy_with_logits(
                direction_logits, direction
            )
            coefficient_loss = F.mse_loss(predicted_coefficients, coefficient)
            target_delta = coefficient @ compiler.numeric_basis[1:]
            denominator = target_delta.square().mean().clamp_min(1e-8)
            token_loss = F.mse_loss(token_delta, target_delta) / denominator
            loss = (
                float(preregistration["property_loss_weight"]) * property_loss
                + float(preregistration["direction_loss_weight"]) * direction_loss
                + float(preregistration["coefficient_loss_weight"]) * coefficient_loss
                + float(preregistration["token_loss_weight"]) * token_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                compiler.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["property_loss"] += float(property_loss.detach())
            totals["direction_loss"] += float(direction_loss.detach())
            totals["coefficient_loss"] += float(coefficient_loss.detach())
            totals["token_loss_ratio"] += float(token_loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite V11 training metrics: {row}")
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == int(preregistration["training_epochs"]):
            print(json.dumps({"stage": "elementwise_compiler_epoch", **row}, sort_keys=True), flush=True)
    compiler.eval()
    return history


@torch.no_grad()
def predict(
    compiler: ElementwiseAdditiveCompiler,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soft_rows = []
    hard_rows = []
    token_rows = []
    for start in range(0, len(embeddings), int(batch_size)):
        soft, hard, _property, _direction, tokens = compiler(
            embeddings[start : start + batch_size].to(device),
            mask[start : start + batch_size].to(device),
        )
        soft_rows.append(soft.cpu())
        hard_rows.append(hard.cpu())
        token_rows.append(tokens.cpu())
    return torch.cat(soft_rows), torch.cat(hard_rows), torch.cat(token_rows)


def hard_set_metrics(hard: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    expected = target.ne(0)
    predicted = hard.ne(0)
    true_positive = (predicted & expected).sum().float()
    exact_support = predicted.eq(expected).all(dim=-1)
    exact_signed = exact_support & hard.eq(target).all(dim=-1)
    return {
        "support_precision": float(true_positive / predicted.sum().clamp_min(1)),
        "support_recall": float(true_positive / expected.sum().clamp_min(1)),
        "exact_support_rate": float(exact_support.float().mean()),
        "active_sign_accuracy": float(
            (torch.sign(hard[expected]) == torch.sign(target[expected])).float().mean()
        ),
        "exact_signed_set_rate": float(exact_signed.float().mean()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V11 execution exists: {summary_path}")
    full_root = args.v5_root / "full"
    input_hashes = v10.check_locked_inputs(
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
            "v10_gate_sha256": args.v5_root.parents[1]
            / "permutation_invariant_semantic_set_compiler_v10"
            / "seed_2151"
            / "gate"
            / "gate_summary.json",
        },
    )
    v10_gate = read_json(
        args.v5_root.parents[1]
        / "permutation_invariant_semantic_set_compiler_v10"
        / "seed_2151"
        / "gate"
        / "gate_summary.json"
    )
    if v10_gate.get("decision") != "stop_llm_core_generation_claim_before_molecule_pilots":
        raise ValueError("V11 requires the locked negative V10 gate")
    bundle = torch.load(args.basis_bundle, map_location="cpu", weights_only=False)
    if bundle.get("role") != "target_free_frozen_generation_basis_and_support":
        raise ValueError("V11 requires target-free numeric basis bundle")
    basis = torch.as_tensor(bundle["basis"], dtype=torch.float32)
    property_columns = [str(name) for name in bundle["property_columns"]]
    e1 = read_json(args.e1_manifest)
    property_names = dict(e1["property_names"])
    fit_rows = singleton_fit_specs(property_columns)
    probe_rows = sample_probe_specs(property_columns, preregistration)
    fit_examples = v10.examples_from_specs(
        fit_rows,
        property_columns,
        property_names,
        list(preregistration["fit_templates"]),
        "v11_singleton_fit",
    )
    matched_examples = probe_examples(
        probe_rows,
        property_columns,
        property_names,
        list(preregistration["probe_templates"]),
        reversed_language=False,
    )
    reversed_examples = probe_examples(
        probe_rows,
        property_columns,
        property_names,
        list(preregistration["probe_templates"]),
        reversed_language=True,
    )
    if {len(row) for row in probe_rows} != {5, 6, 7}:
        raise ValueError("V11 probe cardinality drift")

    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["training_seed"]))
    llm, tokenizer, token_router, checkpoint_columns = v10.load_frozen_language_stack(
        args, preregistration, device
    )
    if checkpoint_columns != property_columns:
        raise ValueError("V11 property vocabulary ordering drift")
    phrase_lists = [
        *v10.phrase_rows(fit_examples),
        *v10.phrase_rows(matched_examples),
        *v10.phrase_rows(reversed_examples),
    ]
    phrase_cache = v10.encode_unique_phrases(
        llm,
        tokenizer,
        [phrase for row in phrase_lists for phrase in row],
        preregistration,
        device,
    )
    fit_embeddings, fit_properties, fit_directions, fit_coefficients = (
        singleton_training_tensors(fit_examples, phrase_cache)
    )
    matched_embeddings, matched_mask = v10.materialize_sets(
        v10.phrase_rows(matched_examples), phrase_cache
    )
    reversed_embeddings, reversed_mask = v10.materialize_sets(
        v10.phrase_rows(reversed_examples), phrase_cache
    )
    permuted_embeddings, permuted_mask = v10.materialize_sets(
        v10.phrase_rows(matched_examples, reverse_order=True), phrase_cache
    )
    targets = torch.stack([example.target for example in matched_examples])
    compiler = ElementwiseAdditiveCompiler(
        embedding_dim=int(next(iter(phrase_cache.values())).numel()),
        hidden_dim=int(preregistration["set_hidden_dim"]),
        property_count=len(property_columns),
        basis=basis,
        field_rank=int(preregistration["field_rank"]),
    ).to(device)
    history = train_compiler(
        compiler,
        fit_embeddings,
        fit_properties,
        fit_directions,
        fit_coefficients,
        preregistration,
        device,
    )
    matched_soft, matched_hard, matched_tokens = predict(
        compiler, matched_embeddings, matched_mask, int(preregistration["probe_batch_size"]), device
    )
    reversed_soft, reversed_hard, reversed_tokens = predict(
        compiler, reversed_embeddings, reversed_mask, int(preregistration["probe_batch_size"]), device
    )
    permuted_soft, permuted_hard, permuted_tokens = predict(
        compiler, permuted_embeddings, permuted_mask, int(preregistration["probe_batch_size"]), device
    )
    token_slot_coefficients, token_slot_support, _cardinality = v10.v5.predict_examples(
        llm, token_router, tokenizer, matched_examples, preregistration, device
    )
    numeric_tokens = v10.compose_numeric_tokens(targets, basis)
    token_slot_tokens = v10.compose_numeric_tokens(token_slot_coefficients, basis)
    arm_tokens = {
        "numeric_canonical": numeric_tokens,
        "semantic_set_compiler": matched_tokens,
        "token_slot": token_slot_tokens,
        "reversed_language": reversed_tokens,
    }
    token_summary = v10.token_metrics(arm_tokens, numeric_tokens)
    compiler_token_error = v10.per_row_token_error(matched_tokens, numeric_tokens)
    slot_token_error = v10.per_row_token_error(token_slot_tokens, numeric_tokens)
    reversed_token_error = v10.per_row_token_error(reversed_tokens, numeric_tokens)
    token_paired = {
        "semantic_set_vs_reversed": v10.paired_bootstrap_advantage(
            reversed_token_error,
            compiler_token_error,
            int(preregistration["bootstrap_seed"]),
            int(preregistration["bootstrap_samples"]),
        ),
        "semantic_set_vs_token_slot": v10.paired_bootstrap_advantage(
            slot_token_error,
            compiler_token_error,
            int(preregistration["bootstrap_seed"]) + 1,
            int(preregistration["bootstrap_samples"]),
        ),
    }
    set_summary = {
        "semantic_set_compiler": hard_set_metrics(matched_hard, targets),
        "token_slot": hard_set_metrics(token_slot_coefficients.sign(), targets),
        "reversed_language": hard_set_metrics(reversed_hard, targets),
    }
    permutation = {
        "soft_coefficient_max_abs": float((matched_soft - permuted_soft).abs().max()),
        "hard_coefficient_max_abs": float((matched_hard - permuted_hard).abs().max()),
        "token_max_abs": float((matched_tokens - permuted_tokens).abs().max()),
    }
    source_pairs, _records = v10.v6.load_generation_pairs(args.source_manifest, preregistration)
    graph_model, representation, _config, _summary = v10.semantic.load_graph_stack(
        args, preregistration, bundle, device
    )
    velocity = v10.graph_velocity_metrics(
        graph_model,
        representation,
        [pair.source for pair in source_pairs],
        arm_tokens,
        preregistration,
        device,
    )
    velocity_paired = {
        "semantic_set_vs_reversed": v10.paired_bootstrap_advantage(
            torch.tensor(velocity["reversed_language"]["per_source_relative_mse"]),
            torch.tensor(velocity["semantic_set_compiler"]["per_source_relative_mse"]),
            int(preregistration["bootstrap_seed"]) + 2,
            int(preregistration["bootstrap_samples"]),
        ),
        "semantic_set_vs_token_slot": v10.paired_bootstrap_advantage(
            torch.tensor(velocity["token_slot"]["per_source_relative_mse"]),
            torch.tensor(velocity["semantic_set_compiler"]["per_source_relative_mse"]),
            int(preregistration["bootstrap_seed"]) + 3,
            int(preregistration["bootstrap_samples"]),
        ),
    }
    gates = dict(preregistration["representation_gates"])
    checks = {
        "exact_support_rate": set_summary["semantic_set_compiler"]["exact_support_rate"]
        >= float(gates["exact_support_rate"]),
        "active_sign_accuracy": set_summary["semantic_set_compiler"]["active_sign_accuracy"]
        >= float(gates["active_sign_accuracy"]),
        "exact_signed_set_rate": set_summary["semantic_set_compiler"]["exact_signed_set_rate"]
        >= float(gates["exact_signed_set_rate"]),
        "permutation_soft": permutation["soft_coefficient_max_abs"]
        <= float(gates["permutation_max_abs"]),
        "permutation_hard": permutation["hard_coefficient_max_abs"]
        <= float(gates["permutation_max_abs"]),
        "permutation_token": permutation["token_max_abs"]
        <= float(gates["permutation_max_abs"]),
        "token_numeric_equivalence": token_summary["semantic_set_compiler"][
            "mse_ratio_vs_zero"
        ]
        <= float(gates["token_mse_ratio_vs_zero"]),
        "token_matched_better_than_reversed": token_paired[
            "semantic_set_vs_reversed"
        ]["ci95_lower"]
        >= float(gates["matched_reversed_token_advantage_ci_lower"]),
        "token_noninferior_to_token_slot": token_paired[
            "semantic_set_vs_token_slot"
        ]["ci95_lower"]
        >= -float(gates["token_slot_noninferiority_margin"]),
        "velocity_numeric_equivalence": velocity["semantic_set_compiler"][
            "relative_mse_to_numeric_velocity"
        ]
        <= float(gates["velocity_relative_mse"]),
        "velocity_matched_better_than_reversed": velocity_paired[
            "semantic_set_vs_reversed"
        ]["ci95_lower"]
        >= float(gates["matched_reversed_velocity_advantage_ci_lower"]),
        "velocity_noninferior_to_token_slot": velocity_paired[
            "semantic_set_vs_token_slot"
        ]["ci95_lower"]
        >= -float(gates["token_slot_velocity_noninferiority_margin"]),
    }
    passed = all(checks.values())
    checkpoint_path = args.output_dir / "elementwise_additive_semantic_set_compiler.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "state_dict": compiler.cpu().state_dict(),
            "property_columns": property_columns,
            "token_shape": list(bundle["token_shape"]),
            "field_rank": int(preregistration["field_rank"]),
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
            "singleton_examples": len(fit_examples),
            "composition_supervision": False,
            "history": history,
        },
        "probe": {
            "examples": len(matched_examples),
            "unique_property_sets": len(probe_rows),
            "cardinalities": [5, 6, 7],
            "v10_probe_reuse": False,
        },
        "set_reconstruction": set_summary,
        "permutation_invariance": permutation,
        "numeric_token_equivalence": token_summary,
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
            "singleton_language_fit_only": True,
            "composition_supervision": False,
            "v10_probe_reuse": False,
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
            "final_representation_attempt": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    del llm, tokenizer, token_router, graph_model, representation
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
