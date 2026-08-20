#!/usr/bin/env python3
"""Train a token-level LoRA property-coordinate compiler for frozen graph flow.

The Common LLM sees constraint-only language and no molecule.  Its token states
are read by one explicit attention slot per molecular property.  Only a small
LoRA adapter and the slot decoder are trained to emit signed property
coefficients.  Synthetic single-property and pairwise instructions cover all
properties, while the four property pairs used by the real graph probe are
completely excluded from language training.  Those unseen compositions are
then tested with unseen templates before any graph-flow diagnostic.

This is a representation probe only: it cannot generate, rank, repair, or
evaluate molecules.
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
from dataclasses import dataclass
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

import property_factorized_language_graph_basis_v1 as v1  # noqa: E402


PROTOCOL = "train_only_token_slot_lora_property_compiler_v3"
PREDECESSOR_PROTOCOL = v1.PREDECESSOR_PROTOCOL
base = v1.base
semantic = v1.previous
unified = v1.unified


@dataclass(frozen=True)
class TextExample:
    text: str
    target: torch.Tensor
    phrases: Mapping[int, str]
    key: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
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
        "single_mechanism_change": "token_slot_lora_property_compiler",
        "common_llm_prompt_contains_source": False,
        "molecule_generation": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "official_test_access": False,
        "language_fit_excludes_graph_probe_property_pairs": True,
        "single_seed": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Token-slot preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Token-slot implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locks = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing token-slot inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locks.get(name), "actual": digest}
        for name, digest in actual.items()
        if locks.get(name) != digest
    }
    if drift:
        raise ValueError(f"Token-slot locked-input drift: {drift}")
    return actual


def normalized_property_pair(specs: Sequence[tuple[str, int]]) -> tuple[str, str]:
    names = sorted(str(name) for name, _direction in specs)
    if len(names) != 2:
        raise ValueError(f"Expected a two-property composition, found {specs}")
    return names[0], names[1]


def heldout_property_pairs(
    pairs: Sequence[object], validation_indices: Sequence[int]
) -> tuple[tuple[str, str], ...]:
    values = {
        normalized_property_pair(semantic.specs_for_row(pairs[index].row))
        for index in validation_indices
    }
    return tuple(sorted(values))


def render_specs(
    specs: Sequence[tuple[str, int]],
    property_names: Mapping[str, object],
    property_lookup: Mapping[str, int],
    template: str,
) -> tuple[str, dict[int, str]]:
    phrases: list[str] = []
    slots: dict[int, str] = {}
    for prop, direction in specs:
        readable = str(property_names.get(prop, prop))
        if template == "canonical":
            phrase = f"{'increase' if direction > 0 else 'decrease'} {readable}"
        elif template == "train_paraphrase":
            phrase = f"make {readable} {'higher' if direction > 0 else 'lower'}"
        elif template == "schema":
            phrase = f"{readable}={'up' if direction > 0 else 'down'}"
        elif template == "probe_raise_lower":
            phrase = f"{'raise' if direction > 0 else 'lower'} {readable}"
        elif template == "probe_more_less":
            phrase = f"{'more' if direction > 0 else 'less'} {readable}"
        else:
            raise ValueError(f"Unknown language template: {template}")
        phrases.append(phrase)
        slots[int(property_lookup[str(prop)])] = phrase
    if template == "canonical":
        text = f"Modify the molecule to {' and '.join(phrases)}."
    elif template == "train_paraphrase":
        text = f"Please {' while '.join(phrases)}."
    elif template == "schema":
        text = f"Desired molecular property changes: {'; '.join(phrases)}."
    elif template == "probe_raise_lower":
        text = f"Edit the compound to {' but '.join(phrases)}."
    else:
        text = f"The resulting compound should have {' and '.join(phrases)}."
    return text, slots


def make_example(
    specs: Sequence[tuple[str, int]],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    template: str,
    key: str,
) -> TextExample:
    lookup = {str(name): index for index, name in enumerate(property_columns)}
    text, phrases = render_specs(specs, property_names, lookup, template)
    return TextExample(
        text=text,
        target=v1.property_vector(specs, property_columns),
        phrases=phrases,
        key=key,
    )


def synthetic_training_examples(
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    excluded_pairs: set[tuple[str, str]],
    scramble_seed: int,
) -> list[TextExample]:
    specs_rows: list[list[tuple[str, int]]] = []
    for prop in property_columns:
        for direction in (-1, 1):
            specs_rows.append([(str(prop), int(direction))])
    for left, right in combinations(property_columns, 2):
        pair_key = tuple(sorted((str(left), str(right))))
        if pair_key in excluded_pairs:
            continue
        for left_direction, right_direction in product((-1, 1), repeat=2):
            specs_rows.append(
                [
                    (str(left), int(left_direction)),
                    (str(right), int(right_direction)),
                ]
            )
    examples: list[TextExample] = []
    for row_index, specs in enumerate(specs_rows):
        for template in ("canonical", "train_paraphrase", "schema"):
            examples.append(
                make_example(
                    specs,
                    property_columns,
                    property_names,
                    template,
                    f"fit_{row_index:04d}_{template}",
                )
            )
        canonical = examples[-3]
        examples.append(
            TextExample(
                text=semantic.scramble_text(
                    canonical.text, scramble_seed, f"fit_scramble_{row_index:04d}"
                ),
                target=torch.zeros(len(property_columns), dtype=torch.float32),
                phrases={},
                key=f"fit_{row_index:04d}_scrambled",
            )
        )
    return examples


def synthetic_probe_examples(
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    excluded_pairs: Sequence[tuple[str, str]],
) -> tuple[list[TextExample], list[TextExample]]:
    first: list[TextExample] = []
    second: list[TextExample] = []
    for pair_index, (left, right) in enumerate(excluded_pairs):
        for sign_index, (left_direction, right_direction) in enumerate(
            product((-1, 1), repeat=2)
        ):
            specs = [(left, left_direction), (right, right_direction)]
            key = f"probe_{pair_index:02d}_{sign_index:02d}"
            first.append(
                make_example(
                    specs,
                    property_columns,
                    property_names,
                    "probe_raise_lower",
                    f"{key}_raise_lower",
                )
            )
            second.append(
                make_example(
                    specs,
                    property_columns,
                    property_names,
                    "probe_more_less",
                    f"{key}_more_less",
                )
            )
    return first, second


def graph_probe_examples(
    pairs: Sequence[object],
    validation_indices: Sequence[int],
    property_columns: Sequence[str],
    property_names: Mapping[str, object],
    scramble_seed: int,
) -> dict[str, list[TextExample]]:
    output = {
        name: []
        for name in ("matched", "paraphrase", "reversed", "property_swap", "scrambled")
    }
    for order, index in enumerate(validation_indices):
        specs = semantic.specs_for_row(pairs[index].row)
        output["matched"].append(
            make_example(
                specs,
                property_columns,
                property_names,
                "probe_raise_lower",
                f"graph_{order:03d}_matched",
            )
        )
        output["paraphrase"].append(
            make_example(
                specs,
                property_columns,
                property_names,
                "probe_more_less",
                f"graph_{order:03d}_paraphrase",
            )
        )
        output["reversed"].append(
            make_example(
                [(name, -direction) for name, direction in specs],
                property_columns,
                property_names,
                "probe_raise_lower",
                f"graph_{order:03d}_reversed",
            )
        )
        output["property_swap"].append(
            make_example(
                semantic.property_swap_specs(specs),
                property_columns,
                property_names,
                "probe_raise_lower",
                f"graph_{order:03d}_property_swap",
            )
        )
        matched = output["matched"][-1]
        output["scrambled"].append(
            TextExample(
                text=semantic.scramble_text(
                    matched.text, scramble_seed, f"graph_scramble_{order:03d}"
                ),
                target=torch.zeros(len(property_columns), dtype=torch.float32),
                phrases={},
                key=f"graph_{order:03d}_scrambled",
            )
        )
    return output


def examples_digest(examples: Sequence[TextExample]) -> str:
    payload = [
        {
            "key": example.key,
            "text": example.text,
            "target": example.target.tolist(),
            "phrases": {str(key): value for key, value in sorted(example.phrases.items())},
        }
        for example in examples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenize_examples(
    tokenizer: object,
    examples: Sequence[TextExample],
    property_count: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    rendered = [
        tokenizer.apply_chat_template(
            semantic.constraint_only_chat(example.text),
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]
    encoded = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    span_targets = torch.zeros(
        len(examples), int(property_count), int(input_ids.shape[1]), dtype=torch.float32
    )
    span_active = torch.zeros(len(examples), int(property_count), dtype=torch.bool)
    for row, example in enumerate(examples):
        for property_index, phrase in example.phrases.items():
            start = rendered[row].find(phrase)
            if start < 0:
                raise ValueError(f"Missing supervised phrase {phrase!r} in {example.key}")
            end = start + len(phrase)
            selected = []
            for token_index, (token_start, token_end) in enumerate(offsets[row].tolist()):
                if token_end > start and token_start < end:
                    selected.append(token_index)
            if not selected:
                raise ValueError(f"No token span for phrase {phrase!r} in {example.key}")
            span_targets[row, int(property_index), selected] = 1.0 / len(selected)
            span_active[row, int(property_index)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": torch.stack([example.target for example in examples]),
        "span_targets": span_targets,
        "span_active": span_active,
    }


def direction_labels(coefficients: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [(coefficients < 0).float(), (coefficients > 0).float()], dim=-1
    )


class TokenPropertySlotDecoder(nn.Module):
    """Attend one named property slot over direction-sensitive LLM tokens."""

    def __init__(
        self, llm_hidden_dim: int, slot_dim: int, property_count: int
    ) -> None:
        super().__init__()
        self.key = nn.Linear(int(llm_hidden_dim), int(slot_dim), bias=False)
        self.value = nn.Linear(int(llm_hidden_dim), int(slot_dim), bias=False)
        self.queries = nn.Parameter(
            torch.randn(int(property_count), int(slot_dim)) / math.sqrt(float(slot_dim))
        )
        self.direction_weights = nn.Parameter(
            torch.randn(int(property_count), 2, int(slot_dim))
            / math.sqrt(float(slot_dim))
        )
        self.direction_bias = nn.Parameter(torch.full((int(property_count), 2), -2.0))
        self.scale = math.sqrt(float(slot_dim))

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = self.key(hidden.float())
        values = self.value(hidden.float())
        attention_logits = torch.einsum("pd,bld->bpl", self.queries, keys) / self.scale
        attention_logits = attention_logits.masked_fill(
            ~attention_mask[:, None, :].bool(), torch.finfo(attention_logits.dtype).min
        )
        attention = torch.softmax(attention_logits, dim=-1)
        slot_states = torch.einsum("bpl,bld->bpd", attention, values)
        direction_logits = (
            torch.einsum("bpd,psd->bps", slot_states, self.direction_weights)
            + self.direction_bias
        )
        probabilities = torch.sigmoid(direction_logits)
        coefficients = probabilities[..., 1] - probabilities[..., 0]
        return coefficients, direction_logits, attention


def model_forward(
    llm_model: object,
    decoder: TokenPropertySlotDecoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body = semantic.operator._transformer_body(llm_model)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        output = body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return decoder(output.last_hidden_state, attention_mask)


def train_compiler(
    llm_model: object,
    decoder: TokenPropertySlotDecoder,
    tokenized: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    lora_parameters = [parameter for parameter in llm_model.parameters() if parameter.requires_grad]
    if not lora_parameters:
        raise ValueError("Token-slot compiler expected trainable LoRA parameters")
    decoder_parameters = list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": float(preregistration["lora_learning_rate"]),
            },
            {
                "params": decoder_parameters,
                "lr": float(preregistration["decoder_learning_rate"]),
            },
        ],
        weight_decay=float(preregistration["weight_decay"]),
    )
    parameters = lora_parameters + decoder_parameters
    batch_size = int(preregistration["training_batch_size"])
    row_count = int(tokenized["input_ids"].shape[0])
    history = []
    for epoch in range(1, int(preregistration["training_epochs"]) + 1):
        order = list(range(row_count))
        random.Random(int(preregistration["training_seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        llm_model.train()
        decoder.train()
        for start in range(0, row_count, batch_size):
            indices = order[start : start + batch_size]
            input_ids = tokenized["input_ids"][indices].to(device)
            attention_mask = tokenized["attention_mask"][indices].to(device)
            targets = tokenized["targets"][indices].to(device)
            span_targets = tokenized["span_targets"][indices].to(device)
            span_active = tokenized["span_active"][indices].to(device)
            prediction, direction_logits, slot_attention = model_forward(
                llm_model, decoder, input_ids, attention_mask, device
            )
            active = targets.abs()
            inactive = 1.0 - active
            coefficient_loss = (
                ((prediction - targets).square() * active).sum()
                / active.sum().clamp_min(1.0)
                + float(preregistration["inactive_loss_weight"])
                * (prediction.square() * inactive).sum()
                / inactive.sum().clamp_min(1.0)
            )
            labels = direction_labels(targets)
            direction_weight = 1.0 + float(preregistration["positive_direction_weight"]) * labels
            direction_loss = (
                F.binary_cross_entropy_with_logits(
                    direction_logits, labels, reduction="none"
                )
                * direction_weight
            ).mean()
            token_log_attention = slot_attention.clamp_min(1e-9).log()
            attention_rows = -(span_targets * token_log_attention).sum(dim=-1)
            attention_loss = (
                attention_rows[span_active].mean()
                if bool(span_active.any())
                else torch.zeros((), device=device)
            )
            loss = (
                float(preregistration["coefficient_loss_weight"]) * coefficient_loss
                + float(preregistration["direction_loss_weight"]) * direction_loss
                + float(preregistration["attention_loss_weight"]) * attention_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, float(preregistration["grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["coefficient_loss"] += float(coefficient_loss.detach())
            totals["direction_loss"] += float(direction_loss.detach())
            totals["attention_loss"] += float(attention_loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Non-finite token-slot metrics: {row}")
        history.append(row)
        print(json.dumps({"stage": "token_slot_lora_epoch", **row}, sort_keys=True), flush=True)
    llm_model.eval()
    decoder.eval()
    return history


@torch.no_grad()
def predict_examples(
    llm_model: object,
    decoder: TokenPropertySlotDecoder,
    tokenizer: object,
    examples: Sequence[TextExample],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    tokenized = tokenize_examples(
        tokenizer,
        examples,
        int(preregistration["property_count"]),
        int(preregistration["llm_max_length"]),
    )
    rows = []
    batch_size = int(preregistration["probe_batch_size"])
    for start in range(0, len(examples), batch_size):
        prediction, _logits, _attention = model_forward(
            llm_model,
            decoder,
            tokenized["input_ids"][start : start + batch_size].to(device),
            tokenized["attention_mask"][start : start + batch_size].to(device),
            device,
        )
        rows.append(prediction.cpu())
    return torch.cat(rows, dim=0)


def coefficient_summary(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    active = target.abs() > 0
    inactive = ~active
    per_row_correct = (
        (torch.sign(prediction) == torch.sign(target)) | inactive
    ).all(dim=-1)
    return {
        "mse": float(F.mse_loss(prediction, target)),
        "active_sign_accuracy": float(
            (torch.sign(prediction[active]) == torch.sign(target[active])).float().mean()
        )
        if bool(active.any())
        else 1.0,
        "exact_active_sign_rate": float(per_row_correct.float().mean()),
        "inactive_abs_mean": float(prediction[inactive].abs().mean())
        if bool(inactive.any())
        else 0.0,
        "mean_abs_coefficient": float(prediction.abs().mean()),
    }


def incompatible_target_order(target: torch.Tensor) -> list[int]:
    order = []
    for row in range(len(target)):
        candidate = None
        for offset in range(1, len(target) + 1):
            index = (row + offset) % len(target)
            if not torch.equal(target[row], target[index]):
                candidate = index
                break
        if candidate is None:
            raise ValueError("Cannot construct incompatible semantic targets")
        order.append(candidate)
    return order


def semantic_probe_metrics(
    first_prediction: torch.Tensor,
    second_prediction: torch.Tensor,
    first_examples: Sequence[TextExample],
) -> dict[str, object]:
    target = torch.stack([example.target for example in first_examples])
    summary = coefficient_summary(first_prediction, target)
    incompatible = target[incompatible_target_order(target)]
    aligned_mse = float(F.mse_loss(first_prediction, target))
    incompatible_mse = float(F.mse_loss(first_prediction, incompatible))
    return {
        **summary,
        "aligned_mse": aligned_mse,
        "incompatible_mse": incompatible_mse,
        "aligned_vs_incompatible_mse_gain": incompatible_mse - aligned_mse,
        "unseen_template_consistency_mse": float(
            F.mse_loss(first_prediction, second_prediction)
        ),
    }


def graph_probe_metrics(
    predictions: Mapping[str, torch.Tensor], examples: Mapping[str, Sequence[TextExample]]
) -> dict[str, object]:
    output = {}
    for variant, prediction in predictions.items():
        target = torch.stack([example.target for example in examples[variant]])
        output[variant] = coefficient_summary(prediction, target)
    output["paraphrase_consistency_mse"] = float(
        F.mse_loss(predictions["matched"], predictions["paraphrase"])
    )
    return output


@torch.no_grad()
def token_metrics(
    basis: torch.Tensor,
    pairs: Sequence[object],
    validation_indices: Sequence[int],
    matched_prediction: torch.Tensor,
    matched_target: torch.Tensor,
    token_shape: tuple[int, int],
    device: torch.device,
) -> dict[str, float]:
    canonical = torch.from_numpy(
        np.stack(
            [np.asarray(pairs[index].condition, dtype=np.float32) for index in validation_indices]
        )
    ).to(device)
    language_tokens = v1.compose_tokens(matched_prediction.to(device), basis, token_shape)
    oracle_tokens = v1.compose_tokens(matched_target.to(device), basis, token_shape)
    intercept_tokens = v1.compose_tokens(
        torch.zeros(len(validation_indices), basis.shape[0] - 1, device=device),
        basis,
        token_shape,
    )
    denominator = F.mse_loss(intercept_tokens, canonical).clamp_min(1e-12)
    language_mse = F.mse_loss(language_tokens, canonical)
    oracle_mse = F.mse_loss(oracle_tokens, canonical)
    return {
        "intercept_mse": float(denominator),
        "language_mse": float(language_mse),
        "oracle_basis_mse": float(oracle_mse),
        "language_mse_ratio_vs_intercept": float(language_mse / denominator),
        "oracle_basis_mse_ratio_vs_intercept": float(oracle_mse / denominator),
    }


@torch.no_grad()
def graph_flow_metrics(
    graph_model: nn.Module,
    representation: nn.Module,
    basis: torch.Tensor,
    pairs: Sequence[object],
    validation_indices: Sequence[int],
    matched_prediction: torch.Tensor,
    reversed_prediction: torch.Tensor,
    matched_target: torch.Tensor,
    token_shape: tuple[int, int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    batch_size = int(preregistration["graph_probe_batch_size"])
    base.seed_everything(int(preregistration["graph_probe_seed"]))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for start in range(0, len(validation_indices), batch_size):
        chosen = list(validation_indices[start : start + batch_size])
        local = slice(start, start + len(chosen))
        collated = base.pair_collate([pairs[index] for index in chosen])
        source = base.move_graph_batch(collated["source"], device)
        target_graph = base.move_graph_batch(collated["target"], device)
        canonical_tokens = collated["condition"].to(device).float()
        matched_tokens = v1.compose_tokens(
            matched_prediction[local].to(device), basis, token_shape
        )
        reversed_tokens = v1.compose_tokens(
            reversed_prediction[local].to(device), basis, token_shape
        )
        oracle_tokens = v1.compose_tokens(
            matched_target[local].to(device), basis, token_shape
        )
        intercept_tokens = v1.compose_tokens(
            torch.zeros(len(chosen), basis.shape[0] - 1, device=device),
            basis,
            token_shape,
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            source_node, source_edge = representation.encode(source)
            target_node, target_edge = representation.encode(target_graph)
            teacher_condition = graph_model.route_condition(canonical_tokens)
            endpoint = graph_model.posterior_endpoint(
                source,
                target_graph,
                source_node,
                source_edge,
                target_node,
                target_edge,
                teacher_condition,
            ).float()
        noise = torch.randn_like(endpoint)
        flow_time = torch.full(
            (len(chosen),), float(preregistration["probe_flow_time"]), device=device
        )
        current = (1.0 - flow_time[:, None]) * noise + flow_time[:, None] * endpoint
        target_velocity = endpoint - noise
        token_sets = {
            "canonical": canonical_tokens,
            "oracle_basis": oracle_tokens,
            "language_basis": matched_tokens,
            "reversed_language": reversed_tokens,
            "intercept": intercept_tokens,
        }
        for name, tokens in token_sets.items():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                velocity = graph_model.transport_velocity(
                    current,
                    flow_time.to(source_node.dtype),
                    source_node,
                    source["node_mask"],
                    tokens,
                ).float()
            totals[f"{name}_flow_mse"] += float(
                F.mse_loss(velocity, target_velocity, reduction="sum")
            )
        count += int(target_velocity.numel())
    metrics = {name: value / max(1, count) for name, value in totals.items()}
    metrics["matched_flow_advantage"] = (
        metrics["reversed_language_flow_mse"] - metrics["language_basis_flow_mse"]
    )
    metrics["language_flow_ratio_vs_intercept"] = (
        metrics["language_basis_flow_mse"] / max(metrics["intercept_flow_mse"], 1e-12)
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed token-slot probe exists: {summary_path}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
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
    prepare = read_json(args.prepare_summary)
    if int(prepare.get("fit_probe_source_overlap", -1)) != 0:
        raise ValueError("Token-slot graph probe requires source-disjoint fit/probe")
    bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != PREDECESSOR_PROTOCOL:
        raise ValueError("Token-slot predecessor protocol drift")
    pairs = list(bundle["pairs"])
    train_indices = list(bundle["train_indices"])
    validation_indices = list(bundle["validation_indices"])
    if len(pairs) != int(preregistration["fit_probe_conditions"]):
        raise ValueError("Token-slot pair count drift")
    if len(train_indices) != int(preregistration["fit_conditions"]):
        raise ValueError("Token-slot graph-basis fit count drift")
    if len(validation_indices) != int(preregistration["probe_conditions"]):
        raise ValueError("Token-slot graph probe count drift")
    property_columns = [str(name) for name in unified.PROPERTY_COLUMNS]
    if len(property_columns) != int(preregistration["property_count"]):
        raise ValueError("Token-slot property vocabulary drift")
    excluded_pairs = heldout_property_pairs(pairs, validation_indices)
    expected_excluded = tuple(
        sorted(tuple(sorted(map(str, row))) for row in preregistration["heldout_property_pairs"])
    )
    if excluded_pairs != expected_excluded:
        raise ValueError(
            f"Graph-probe property-pair drift: expected {expected_excluded}, found {excluded_pairs}"
        )
    e1 = read_json(args.e1_manifest)
    property_names = dict(e1["property_names"])
    fit_examples = synthetic_training_examples(
        property_columns,
        property_names,
        set(excluded_pairs),
        int(preregistration["scramble_seed"]),
    )
    if len(fit_examples) != int(preregistration["synthetic_fit_examples"]):
        raise ValueError(
            f"Synthetic fit count drift: expected {preregistration['synthetic_fit_examples']}, "
            f"found {len(fit_examples)}"
        )
    leaked_fit_pairs = {
        normalized_property_pair(
            [
                (property_columns[index], int(value))
                for index, value in enumerate(example.target.tolist())
                if int(value) != 0
            ]
        )
        for example in fit_examples
        if int((example.target != 0).sum()) == 2
    } & set(excluded_pairs)
    if leaked_fit_pairs:
        raise ValueError(f"Held-out property-pair leakage: {sorted(leaked_fit_pairs)}")
    semantic_first, semantic_second = synthetic_probe_examples(
        property_columns, property_names, excluded_pairs
    )
    graph_examples = graph_probe_examples(
        pairs,
        validation_indices,
        property_columns,
        property_names,
        int(preregistration["scramble_seed"]),
    )
    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["training_seed"]))
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm_model, tokenizer = semantic.operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=True
    )
    llm_hidden_dim = int(llm_model.get_base_model().config.hidden_size)
    decoder = TokenPropertySlotDecoder(
        llm_hidden_dim,
        int(preregistration["slot_dim"]),
        len(property_columns),
    ).to(device)
    tokenized_fit = tokenize_examples(
        tokenizer,
        fit_examples,
        len(property_columns),
        int(preregistration["llm_max_length"]),
    )
    history = train_compiler(
        llm_model, decoder, tokenized_fit, preregistration, device
    )
    semantic_first_prediction = predict_examples(
        llm_model, decoder, tokenizer, semantic_first, preregistration, device
    )
    semantic_second_prediction = predict_examples(
        llm_model, decoder, tokenizer, semantic_second, preregistration, device
    )
    graph_predictions = {
        variant: predict_examples(
            llm_model, decoder, tokenizer, examples, preregistration, device
        )
        for variant, examples in graph_examples.items()
    }
    semantic_metrics = semantic_probe_metrics(
        semantic_first_prediction, semantic_second_prediction, semantic_first
    )
    coefficient_metrics = graph_probe_metrics(graph_predictions, graph_examples)
    adapter_dir = args.output_dir / "token_slot_lora_adapter"
    llm_model.save_pretrained(adapter_dir, safe_serialization=True)
    decoder_path = args.output_dir / "token_property_slot_decoder.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "state_dict": decoder.cpu().state_dict(),
            "property_columns": property_columns,
            "llm_hidden_dim": llm_hidden_dim,
            "slot_dim": int(preregistration["slot_dim"]),
        },
        decoder_path,
    )
    del llm_model, decoder, tokenized_fit
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    token_shape = tuple(int(value) for value in np.asarray(pairs[0].condition).shape)
    expected_shape = (
        int(preregistration["token_count"]),
        int(preregistration["condition_dim"]),
    )
    if token_shape != expected_shape:
        raise ValueError(f"Token-slot condition-token shape drift: {token_shape}")
    basis = v1.fit_property_token_basis(
        pairs,
        train_indices,
        v1.coefficient_targets(pairs, property_columns)["matched"],
        float(preregistration["basis_ridge"]),
    )
    matched_target = torch.stack(
        [example.target for example in graph_examples["matched"]]
    )
    tokens = token_metrics(
        basis,
        pairs,
        validation_indices,
        graph_predictions["matched"],
        matched_target,
        token_shape,
        device,
    )
    graph_model, representation, _config, _representation_summary = semantic.load_graph_stack(
        args, preregistration, bundle, device
    )
    flows = graph_flow_metrics(
        graph_model,
        representation,
        basis,
        pairs,
        validation_indices,
        graph_predictions["matched"],
        graph_predictions["reversed"],
        matched_target,
        token_shape,
        preregistration,
        device,
    )
    gates = dict(preregistration["representation_gates"])
    checks = {
        "semantic_unseen_pair_active_sign_accuracy": semantic_metrics["active_sign_accuracy"]
        >= float(gates["semantic_unseen_pair_active_sign_accuracy"]),
        "semantic_aligned_vs_incompatible_mse_gain": semantic_metrics[
            "aligned_vs_incompatible_mse_gain"
        ]
        >= float(gates["semantic_aligned_vs_incompatible_mse_gain"]),
        "semantic_unseen_template_consistency_mse": semantic_metrics[
            "unseen_template_consistency_mse"
        ]
        <= float(gates["semantic_unseen_template_consistency_mse"]),
        "matched_active_sign_accuracy": float(
            dict(coefficient_metrics["matched"])["active_sign_accuracy"]
        )
        >= float(gates["matched_active_sign_accuracy"]),
        "paraphrase_active_sign_accuracy": float(
            dict(coefficient_metrics["paraphrase"])["active_sign_accuracy"]
        )
        >= float(gates["paraphrase_active_sign_accuracy"]),
        "reversed_active_sign_accuracy": float(
            dict(coefficient_metrics["reversed"])["active_sign_accuracy"]
        )
        >= float(gates["reversed_active_sign_accuracy"]),
        "property_swap_active_sign_accuracy": float(
            dict(coefficient_metrics["property_swap"])["active_sign_accuracy"]
        )
        >= float(gates["property_swap_active_sign_accuracy"]),
        "scrambled_mean_abs_coefficient": float(
            dict(coefficient_metrics["scrambled"])["mean_abs_coefficient"]
        )
        <= float(gates["scrambled_mean_abs_coefficient"]),
        "oracle_basis_mse_ratio_vs_intercept": tokens["oracle_basis_mse_ratio_vs_intercept"]
        <= float(gates["oracle_basis_mse_ratio_vs_intercept"]),
        "language_mse_ratio_vs_intercept": tokens["language_mse_ratio_vs_intercept"]
        <= float(gates["language_mse_ratio_vs_intercept"]),
        "matched_flow_advantage": flows["matched_flow_advantage"]
        >= float(gates["matched_flow_advantage"]),
        "language_flow_ratio_vs_intercept": flows["language_flow_ratio_vs_intercept"]
        <= float(gates["language_flow_ratio_vs_intercept"]),
    }
    passed = all(checks.values())
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    summary = {
        "protocol": PROTOCOL,
        "stage": "unseen_composition_token_slot_lora_representation_probe",
        "decision": (
            "advance_token_slot_compiler_to_target_isolated_generation"
            if passed
            else "stop_token_slot_lora_property_compiler"
        ),
        "language_training": {
            "synthetic_examples": len(fit_examples),
            "heldout_property_pairs": [list(value) for value in excluded_pairs],
            "heldout_pair_overlap": 0,
            "training_examples_sha256": examples_digest(fit_examples),
            "history": history,
        },
        "semantic_unseen_composition_probe": semantic_metrics,
        "graph_probe_coefficients": coefficient_metrics,
        "graph_probe_tokens": tokens,
        "graph_probe_flow": flows,
        "representation_gate": {
            "passed": passed,
            "checks": checks,
            "thresholds": gates,
        },
        "artifacts": {
            "decoder_sha256": file_sha256(decoder_path),
            "lora_adapter_config_sha256": file_sha256(adapter_config),
            "lora_adapter_model_sha256": file_sha256(adapter_model),
            "locked_inputs": input_hashes,
        },
        "contract": {
            "common_llm_prompt_contains_source": False,
            "language_fit_target_access": False,
            "language_fit_excludes_graph_probe_property_pairs": True,
            "graph_fit_probe_source_overlap": 0,
            "probe_target_access_for_postfit_flow_diagnostic": True,
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
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
