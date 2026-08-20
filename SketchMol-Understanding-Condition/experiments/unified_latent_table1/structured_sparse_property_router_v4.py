#!/usr/bin/env python3
"""Train one preregistered arm of a structured sparse Common-LLM router.

The router predicts instruction cardinality explicitly, then activates exactly
that many named property slots by a deterministic top-k operation.  No support
threshold is tuned.  Four independently executed arms isolate LoRA, token-slot
attention, and composition supervision while preserving the same frozen graph
stack, seeds, probes, and target-access contract.

This stage is a representation experiment only.  It never generates, ranks,
repairs, or evaluates molecules.  A valid arm always exits zero; scientific
acceptance is performed by a separate gate job.
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

import evaluate_token_slot_sparse_support_repair_v3 as repair  # noqa: E402
import token_slot_lora_property_compiler_v3 as v3  # noqa: E402


PROTOCOL = "train_only_structured_sparse_property_router_v4"
ARMS = ("full", "no_lora", "no_token_slots", "no_composition")
base = v3.base
semantic = v3.semantic
unified = v3.unified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--v3-summary", required=True, type=Path)
    parser.add_argument("--repair-summary", required=True, type=Path)
    parser.add_argument("--prepare-summary", required=True, type=Path)
    parser.add_argument("--fit-probe-bundle", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--canonical-checkpoint", required=True, type=Path)
    parser.add_argument("--sft-adapter-dir", required=True, type=Path)
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
        "mechanism": "explicit_cardinality_exact_topk_property_router",
        "support_threshold_search": False,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "official_test_access": False,
        "language_fit_excludes_graph_probe_property_pairs": True,
        "single_seed": True,
        "arms": list(ARMS),
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Structured-router preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Structured-router implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing structured-router inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Structured-router locked-input drift: {drift}")
    return actual


def specs_signature(specs: Sequence[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(name), int(direction)) for name, direction in specs))


def contains_excluded_pair(
    specs: Sequence[tuple[str, int]], excluded_pairs: set[tuple[str, str]]
) -> bool:
    names = [str(name) for name, _direction in specs]
    return any(tuple(sorted(pair)) in excluded_pairs for pair in combinations(names, 2))


def sample_specs(
    property_columns: Sequence[str],
    excluded_pairs: set[tuple[str, str]],
    cardinalities: Sequence[int],
    samples_per_cardinality: int,
    seed: int,
    forbidden: set[tuple[tuple[str, int], ...]] | None = None,
) -> list[list[tuple[str, int]]]:
    rng = random.Random(int(seed))
    forbidden = set() if forbidden is None else set(forbidden)
    output: list[list[tuple[str, int]]] = []
    seen = set(forbidden)
    for cardinality in cardinalities:
        accepted = 0
        attempts = 0
        while accepted < int(samples_per_cardinality):
            attempts += 1
            if attempts > int(samples_per_cardinality) * 10000:
                raise ValueError(f"Cannot sample cardinality {cardinality} without leakage")
            names = rng.sample(list(map(str, property_columns)), int(cardinality))
            specs = [(name, rng.choice((-1, 1))) for name in names]
            signature = specs_signature(specs)
            if signature in seen or contains_excluded_pair(specs, excluded_pairs):
                continue
            seen.add(signature)
            output.append(specs)
            accepted += 1
    return output


def base_fit_specs(
    property_columns: Sequence[str], excluded_pairs: set[tuple[str, str]]
) -> list[list[tuple[str, int]]]:
    rows: list[list[tuple[str, int]]] = []
    for prop in property_columns:
        for direction in (-1, 1):
            rows.append([(str(prop), int(direction))])
    for left, right in combinations(property_columns, 2):
        if tuple(sorted((str(left), str(right)))) in excluded_pairs:
            continue
        for left_direction, right_direction in product((-1, 1), repeat=2):
            rows.append(
                [(str(left), int(left_direction)), (str(right), int(right_direction))]
            )
    return rows


def examples_from_specs(
    specs_rows: Sequence[Sequence[tuple[str, int]]],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    scramble_seed: int,
    prefix: str,
) -> list[v3.TextExample]:
    examples: list[v3.TextExample] = []
    for row_index, specs in enumerate(specs_rows):
        for template in ("canonical", "train_paraphrase", "schema"):
            examples.append(
                v3.make_example(
                    specs,
                    property_columns,
                    property_names,
                    template,
                    f"{prefix}_{row_index:04d}_{template}",
                )
            )
        canonical = examples[-3]
        examples.append(
            v3.TextExample(
                text=semantic.scramble_text(
                    canonical.text, scramble_seed, f"{prefix}_scramble_{row_index:04d}"
                ),
                target=torch.zeros(len(property_columns), dtype=torch.float32),
                phrases={},
                key=f"{prefix}_{row_index:04d}_scrambled",
            )
        )
    return examples


def training_and_multicardinality_probe_examples(
    arm: str,
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    excluded_pairs: set[tuple[str, str]],
    preregistration: Mapping[str, object],
) -> tuple[list[v3.TextExample], list[v3.TextExample], list[v3.TextExample]]:
    base_specs = base_fit_specs(property_columns, excluded_pairs)
    higher_specs = sample_specs(
        property_columns,
        excluded_pairs,
        range(3, int(preregistration["max_instruction_cardinality"]) + 1),
        int(preregistration["higher_cardinality_fit_specs_per_k"]),
        int(preregistration["composition_fit_seed"]),
    )
    full_specs = base_specs + higher_specs
    fit_specs = (
        [specs for specs in base_specs if len(specs) == 1]
        if arm == "no_composition"
        else full_specs
    )
    forbidden = {specs_signature(specs) for specs in full_specs}
    probe_specs = sample_specs(
        property_columns,
        excluded_pairs,
        range(3, int(preregistration["max_instruction_cardinality"]) + 1),
        int(preregistration["higher_cardinality_probe_specs_per_k"]),
        int(preregistration["composition_probe_seed"]),
        forbidden,
    )
    unique_fit_examples = examples_from_specs(
        fit_specs,
        property_columns,
        property_names,
        int(preregistration["scramble_seed"]),
        f"fit_{arm}",
    )
    if arm == "no_composition":
        # Preserve the full arm's number of optimizer updates.  This ablation
        # removes composition supervision only; it must not also receive 26x
        # fewer language examples and gradient steps.
        target_count = int(preregistration["no_composition_fit_examples"])
        fit_examples = [
            v3.TextExample(
                text=example.text,
                target=example.target,
                phrases=example.phrases,
                key=f"{example.key}_exposure_{index:04d}",
            )
            for index, example in enumerate(
                unique_fit_examples[index % len(unique_fit_examples)]
                for index in range(target_count)
            )
        ]
    else:
        fit_examples = unique_fit_examples
    first = [
        v3.make_example(
            specs,
            property_columns,
            property_names,
            "probe_raise_lower",
            f"multi_probe_{index:04d}_raise_lower",
        )
        for index, specs in enumerate(probe_specs)
    ]
    second = [
        v3.make_example(
            specs,
            property_columns,
            property_names,
            "probe_more_less",
            f"multi_probe_{index:04d}_more_less",
        )
        for index, specs in enumerate(probe_specs)
    ]
    return fit_examples, first, second


class StructuredSparsePropertyRouter(nn.Module):
    """Predict direction, support, and cardinality before exact top-k routing."""

    def __init__(
        self,
        llm_hidden_dim: int,
        slot_dim: int,
        property_count: int,
        max_cardinality: int,
        use_token_slots: bool,
    ) -> None:
        super().__init__()
        self.property_count = int(property_count)
        self.max_cardinality = int(max_cardinality)
        self.use_token_slots = bool(use_token_slots)
        self.key = nn.Linear(int(llm_hidden_dim), int(slot_dim), bias=False)
        self.value = nn.Linear(int(llm_hidden_dim), int(slot_dim), bias=False)
        self.queries = nn.Parameter(
            torch.randn(self.property_count, int(slot_dim)) / math.sqrt(float(slot_dim))
        )
        self.property_embedding = nn.Parameter(
            torch.randn(self.property_count, int(slot_dim)) / math.sqrt(float(slot_dim))
        )
        self.pooled_value = nn.Linear(int(llm_hidden_dim), int(slot_dim), bias=False)
        self.direction_weights = nn.Parameter(
            torch.randn(self.property_count, 2, int(slot_dim))
            / math.sqrt(float(slot_dim))
        )
        self.direction_bias = nn.Parameter(torch.full((self.property_count, 2), -2.0))
        self.support_weights = nn.Parameter(
            torch.randn(self.property_count, int(slot_dim)) / math.sqrt(float(slot_dim))
        )
        self.support_bias = nn.Parameter(torch.full((self.property_count,), -2.0))
        self.cardinality = nn.Sequential(
            nn.Linear(int(llm_hidden_dim), int(slot_dim)),
            nn.GELU(),
            nn.Linear(int(slot_dim), self.max_cardinality + 1),
        )
        self.scale = math.sqrt(float(slot_dim))

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = attention_mask.bool()
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / denominator
        if self.use_token_slots:
            keys = self.key(hidden.float())
            values = self.value(hidden.float())
            attention_logits = (
                torch.einsum("pd,bld->bpl", self.queries, keys) / self.scale
            )
            attention_logits = attention_logits.masked_fill(
                ~mask[:, None, :], torch.finfo(attention_logits.dtype).min
            )
            attention = torch.softmax(attention_logits, dim=-1)
            slot_states = torch.einsum("bpl,bld->bpd", attention, values)
        else:
            shared = self.pooled_value(pooled)
            slot_states = shared[:, None, :] + self.property_embedding[None, :, :]
            attention = mask[:, None, :].float() / denominator[:, None, :]
            attention = attention.expand(-1, self.property_count, -1)
        direction_logits = (
            torch.einsum("bpd,psd->bps", slot_states, self.direction_weights)
            + self.direction_bias
        )
        probabilities = torch.sigmoid(direction_logits)
        raw_coefficients = probabilities[..., 1] - probabilities[..., 0]
        support_logits = (
            torch.einsum("bpd,pd->bp", slot_states, self.support_weights)
            + self.support_bias
        )
        cardinality_logits = self.cardinality(pooled)
        return (
            raw_coefficients,
            direction_logits,
            support_logits,
            cardinality_logits,
            attention,
        )


def exact_topk_support(
    support_logits: torch.Tensor, cardinality_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cardinality = cardinality_logits.argmax(dim=-1)
    support = torch.zeros_like(support_logits, dtype=torch.bool)
    for row, count in enumerate(cardinality.tolist()):
        count = max(0, min(int(count), int(support_logits.shape[1])))
        if count:
            indices = torch.topk(support_logits[row], k=count, dim=-1).indices
            support[row, indices] = True
    return support, cardinality


def model_forward(
    llm_model: object,
    router: StructuredSparsePropertyRouter,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body = semantic.operator._transformer_body(llm_model)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    llm_trainable = any(parameter.requires_grad for parameter in llm_model.parameters())
    with torch.set_grad_enabled(llm_trainable), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
    ):
        output = body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return router(output.last_hidden_state, attention_mask)


def train_router(
    llm_model: object,
    router: StructuredSparsePropertyRouter,
    tokenized: Mapping[str, torch.Tensor],
    arm: str,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    lora_parameters = [
        parameter for parameter in llm_model.parameters() if parameter.requires_grad
    ]
    if arm != "no_lora" and not lora_parameters:
        raise ValueError(f"Arm {arm} expected trainable LoRA parameters")
    if arm == "no_lora" and lora_parameters:
        raise ValueError("no_lora arm unexpectedly has trainable LLM parameters")
    groups = [
        {
            "params": list(router.parameters()),
            "lr": float(preregistration["router_learning_rate"]),
        }
    ]
    if lora_parameters:
        groups.insert(
            0,
            {
                "params": lora_parameters,
                "lr": float(preregistration["lora_learning_rate"]),
            },
        )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(preregistration["weight_decay"])
    )
    parameters = lora_parameters + list(router.parameters())
    batch_size = int(preregistration["training_batch_size"])
    row_count = int(tokenized["input_ids"].shape[0])
    history = []
    for epoch in range(1, int(preregistration["training_epochs"]) + 1):
        order = list(range(row_count))
        random.Random(int(preregistration["training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        llm_model.train(bool(lora_parameters))
        router.train()
        for start in range(0, row_count, batch_size):
            indices = order[start : start + batch_size]
            input_ids = tokenized["input_ids"][indices].to(device)
            attention_mask = tokenized["attention_mask"][indices].to(device)
            targets = tokenized["targets"][indices].to(device)
            span_targets = tokenized["span_targets"][indices].to(device)
            span_active = tokenized["span_active"][indices].to(device)
            (
                raw,
                direction_logits,
                support_logits,
                cardinality_logits,
                slot_attention,
            ) = model_forward(llm_model, router, input_ids, attention_mask, device)
            active = targets.ne(0)
            coefficient_loss = (
                (raw[active] - targets[active]).square().mean()
                if bool(active.any())
                else torch.zeros((), device=device)
            )
            inactive_loss = (
                raw[~active].square().mean()
                if bool((~active).any())
                else torch.zeros((), device=device)
            )
            direction_loss = F.binary_cross_entropy_with_logits(
                direction_logits,
                v3.direction_labels(targets),
                pos_weight=torch.full(
                    (2,), float(preregistration["positive_direction_weight"]), device=device
                ),
            )
            support_loss = F.binary_cross_entropy_with_logits(
                support_logits,
                active.float(),
                pos_weight=torch.tensor(
                    float(preregistration["positive_support_weight"]), device=device
                ),
            )
            cardinality_target = active.sum(dim=-1).long()
            cardinality_loss = F.cross_entropy(cardinality_logits, cardinality_target)
            if router.use_token_slots and bool(span_active.any()):
                token_log_attention = slot_attention.clamp_min(1e-9).log()
                attention_rows = -(span_targets * token_log_attention).sum(dim=-1)
                attention_loss = attention_rows[span_active].mean()
            else:
                attention_loss = torch.zeros((), device=device)
            loss = (
                float(preregistration["coefficient_loss_weight"]) * coefficient_loss
                + float(preregistration["inactive_loss_weight"]) * inactive_loss
                + float(preregistration["direction_loss_weight"]) * direction_loss
                + float(preregistration["support_loss_weight"]) * support_loss
                + float(preregistration["cardinality_loss_weight"]) * cardinality_loss
                + float(preregistration["attention_loss_weight"]) * attention_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, float(preregistration["grad_clip"]))
            optimizer.step()
            for name, value in {
                "loss": loss,
                "coefficient_loss": coefficient_loss,
                "inactive_loss": inactive_loss,
                "direction_loss": direction_loss,
                "support_loss": support_loss,
                "cardinality_loss": cardinality_loss,
                "attention_loss": attention_loss,
            }.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite structured-router metrics: {row}")
        history.append(row)
        print(
            json.dumps(
                {"stage": "structured_sparse_router_epoch", "arm": arm, **row},
                sort_keys=True,
            ),
            flush=True,
        )
    llm_model.eval()
    router.eval()
    return history


@torch.no_grad()
def predict_examples(
    llm_model: object,
    router: StructuredSparsePropertyRouter,
    tokenizer: object,
    examples: Sequence[v3.TextExample],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized = v3.tokenize_examples(
        tokenizer,
        examples,
        int(preregistration["property_count"]),
        int(preregistration["llm_max_length"]),
    )
    coefficient_rows = []
    support_rows = []
    cardinality_rows = []
    batch_size = int(preregistration["probe_batch_size"])
    for start in range(0, len(examples), batch_size):
        raw, _directions, support_logits, cardinality_logits, _attention = model_forward(
            llm_model,
            router,
            tokenized["input_ids"][start : start + batch_size].to(device),
            tokenized["attention_mask"][start : start + batch_size].to(device),
            device,
        )
        support, cardinality = exact_topk_support(support_logits, cardinality_logits)
        coefficients = torch.where(support, raw, torch.zeros_like(raw))
        coefficient_rows.append(coefficients.cpu())
        support_rows.append(support.cpu())
        cardinality_rows.append(cardinality.cpu())
    return (
        torch.cat(coefficient_rows, dim=0),
        torch.cat(support_rows, dim=0),
        torch.cat(cardinality_rows, dim=0),
    )


def routing_metrics(
    coefficients: torch.Tensor,
    support: torch.Tensor,
    cardinality: torch.Tensor,
    examples: Sequence[v3.TextExample],
) -> dict[str, float]:
    targets = torch.stack([example.target for example in examples])
    target_support = targets.ne(0)
    true_positive = int((support & target_support).sum())
    false_positive = int((support & ~target_support).sum())
    false_negative = int((~support & target_support).sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        **v3.coefficient_summary(coefficients, targets),
        "support_precision": (
            true_positive / precision_denominator if precision_denominator else 1.0
        ),
        "support_recall": true_positive / recall_denominator if recall_denominator else 1.0,
        "exact_support_rate": float(
            support.eq(target_support).all(dim=-1).float().mean()
        ),
        "cardinality_exact_rate": float(
            cardinality.eq(target_support.sum(dim=-1)).float().mean()
        ),
        "mean_predicted_cardinality": float(cardinality.float().mean()),
        "mean_target_cardinality": float(target_support.sum(dim=-1).float().mean()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    arm_output = args.output_dir / args.arm
    arm_output.mkdir(parents=True, exist_ok=True)
    summary_path = arm_output / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed structured-router arm exists: {summary_path}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "v3_implementation_sha256": Path(v3.__file__).resolve(),
            "repair_implementation_sha256": Path(repair.__file__).resolve(),
            "v3_summary_sha256": args.v3_summary,
            "repair_summary_sha256": args.repair_summary,
            "prepare_summary_sha256": args.prepare_summary,
            "fit_probe_bundle_sha256": args.fit_probe_bundle,
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
            "e1_manifest_sha256": args.e1_manifest,
        },
    )
    previous = read_json(args.v3_summary)
    repaired = read_json(args.repair_summary)
    if previous.get("protocol") != v3.PROTOCOL:
        raise ValueError("Structured router requires locked v3 evidence")
    if repaired.get("protocol") != repair.PROTOCOL:
        raise ValueError("Structured router requires locked sparse-repair evidence")
    if dict(repaired["repair_gate"])["checks"].get("matched_support_precision") is not False:
        raise ValueError("Structured router premise drift: support precision was not the blocker")
    prepare = read_json(args.prepare_summary)
    if int(prepare.get("fit_probe_source_overlap", -1)) != 0:
        raise ValueError("Structured router requires source-disjoint fit/probe")
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != v3.PREDECESSOR_PROTOCOL:
        raise ValueError("Structured-router predecessor protocol drift")
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    property_columns = [str(name) for name in unified.PROPERTY_COLUMNS]
    if len(property_columns) != int(preregistration["property_count"]):
        raise ValueError("Structured-router property vocabulary drift")
    excluded_pairs = set(v3.heldout_property_pairs(pairs, validation_indices))
    expected_excluded = {
        tuple(sorted(map(str, row))) for row in preregistration["heldout_property_pairs"]
    }
    if excluded_pairs != expected_excluded:
        raise ValueError("Structured-router graph-probe property-pair drift")
    e1 = read_json(args.e1_manifest)
    property_names = dict(e1["property_names"])
    fit_examples, multi_first, multi_second = (
        training_and_multicardinality_probe_examples(
            args.arm,
            property_columns,
            property_names,
            excluded_pairs,
            preregistration,
        )
    )
    expected_count = int(
        preregistration[
            "no_composition_fit_examples"
            if args.arm == "no_composition"
            else "full_fit_examples"
        ]
    )
    if len(fit_examples) != expected_count:
        raise ValueError(
            f"Structured-router fit count drift: expected {expected_count}, "
            f"found {len(fit_examples)}"
        )
    graph_examples = v3.graph_probe_examples(
        pairs,
        validation_indices,
        property_columns,
        property_names,
        int(preregistration["scramble_seed"]),
    )
    device = base.resolve_device(str(args.device))
    arm_seed_offset = ARMS.index(args.arm) * int(preregistration["arm_seed_stride"])
    base.seed_everything(int(preregistration["training_seed"]) + arm_seed_offset)
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    use_lora = args.arm != "no_lora"
    llm_model, tokenizer = semantic.operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=use_lora
    )
    llm_hidden_dim = int(llm_model.get_base_model().config.hidden_size)
    router = StructuredSparsePropertyRouter(
        llm_hidden_dim,
        int(preregistration["slot_dim"]),
        len(property_columns),
        int(preregistration["max_instruction_cardinality"]),
        use_token_slots=args.arm != "no_token_slots",
    ).to(device)
    tokenized_fit = v3.tokenize_examples(
        tokenizer,
        fit_examples,
        len(property_columns),
        int(preregistration["llm_max_length"]),
    )
    history = train_router(
        llm_model, router, tokenized_fit, args.arm, preregistration, device
    )
    probe_predictions: dict[str, torch.Tensor] = {}
    probe_support: dict[str, torch.Tensor] = {}
    probe_cardinality: dict[str, torch.Tensor] = {}
    for name, examples in {
        "multi_first": multi_first,
        "multi_second": multi_second,
        **graph_examples,
    }.items():
        prediction, support, cardinality = predict_examples(
            llm_model, router, tokenizer, examples, preregistration, device
        )
        probe_predictions[name] = prediction
        probe_support[name] = support
        probe_cardinality[name] = cardinality
    multi_metrics = routing_metrics(
        probe_predictions["multi_first"],
        probe_support["multi_first"],
        probe_cardinality["multi_first"],
        multi_first,
    )
    multi_metrics["unseen_template_consistency_mse"] = float(
        F.mse_loss(probe_predictions["multi_first"], probe_predictions["multi_second"])
    )
    graph_routing = {
        name: routing_metrics(
            probe_predictions[name],
            probe_support[name],
            probe_cardinality[name],
            examples,
        )
        for name, examples in graph_examples.items()
    }
    router_path = arm_output / "structured_sparse_router.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "arm": args.arm,
            "state_dict": router.cpu().state_dict(),
            "property_columns": property_columns,
            "llm_hidden_dim": llm_hidden_dim,
            "slot_dim": int(preregistration["slot_dim"]),
            "max_instruction_cardinality": int(
                preregistration["max_instruction_cardinality"]
            ),
            "use_token_slots": args.arm != "no_token_slots",
        },
        router_path,
    )
    adapter_hashes: dict[str, str] = {}
    if use_lora:
        adapter_dir = arm_output / "lora_adapter"
        llm_model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_hashes = {
            "lora_adapter_config_sha256": file_sha256(
                adapter_dir / "adapter_config.json"
            ),
            "lora_adapter_model_sha256": file_sha256(
                adapter_dir / "adapter_model.safetensors"
            ),
        }
    del llm_model, router, tokenized_fit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    token_shape = tuple(int(value) for value in np.asarray(pairs[0].condition).shape)
    if token_shape != (
        int(preregistration["token_count"]),
        int(preregistration["condition_dim"]),
    ):
        raise ValueError("Structured-router graph-token shape drift")
    v1_targets = v3.v1.coefficient_targets(pairs, property_columns)["matched"]
    basis = v3.v1.fit_property_token_basis(
        pairs,
        train_indices,
        v1_targets,
        float(preregistration["basis_ridge"]),
    )
    matched_target = torch.stack(
        [example.target for example in graph_examples["matched"]]
    )
    token_metrics = repair.token_metrics(
        basis,
        pairs,
        validation_indices,
        probe_predictions["matched"],
        matched_target,
        token_shape,
        device,
    )
    graph_model, representation, _config, _representation_summary = (
        semantic.load_graph_stack(args, preregistration, bundle, device)
    )
    flow_metrics = repair.graph_flow_metrics(
        graph_model,
        representation,
        basis,
        pairs,
        validation_indices,
        probe_predictions["matched"],
        probe_predictions["reversed"],
        matched_target,
        token_shape,
        preregistration,
        device,
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "structured_sparse_router_arm_execution",
        "execution_status": "completed",
        "arm": args.arm,
        "mechanisms": {
            "lora": use_lora,
            "token_slots": args.arm != "no_token_slots",
            "composition_supervision": args.arm != "no_composition",
            "explicit_cardinality": True,
            "exact_topk_support": True,
        },
        "training": {
            "fit_examples": len(fit_examples),
            "unique_fit_examples": (
                int(preregistration["no_composition_unique_fit_examples"])
                if args.arm == "no_composition"
                else len(fit_examples)
            ),
            "optimizer_steps": len(history)
            * math.ceil(len(fit_examples) / int(preregistration["training_batch_size"])),
            "fit_examples_sha256": v3.examples_digest(fit_examples),
            "history": history,
        },
        "multicardinality_probe": multi_metrics,
        "graph_probe_routing": graph_routing,
        "graph_probe_tokens": token_metrics,
        "graph_probe_flow": flow_metrics,
        "artifacts": {
            "router_sha256": file_sha256(router_path),
            **adapter_hashes,
            "locked_inputs": input_hashes,
        },
        "contract": {
            "common_llm_prompt_contains_source": False,
            "language_fit_target_access": False,
            "language_fit_excludes_graph_probe_property_pairs": True,
            "graph_fit_probe_source_overlap": 0,
            "support_threshold_search": False,
            "explicit_cardinality": True,
            "exact_topk_support": True,
            "molecule_generation": False,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
