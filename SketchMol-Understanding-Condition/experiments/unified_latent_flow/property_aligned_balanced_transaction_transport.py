#!/usr/bin/env python3
"""Property-aligned balanced transport over chemically closed transactions.

This is one structural follow-up to the compositional VQ signal.  The closed
reaction grammar, train/development split, exact-20 budget, and evaluator stay
fixed.  Two things change together because they define one latent geometry:

* fit-only source/target property deltas are embedded into the transaction
  codebook alongside the structural reaction delta; and
* the twenty latent particles are coupled by one balanced Sinkhorn transport
  instead of sampling twenty codes independently.

The transport allocates code capacity before committing any molecule.  Within
each allocated code, one applicable complete transaction is sampled directly,
without replacement when support permits.  There is no larger molecular pool,
molecule ranking, target access, property-oracle access during generation,
retry, repair, or second edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
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


atomic = vq.atomic
base = vq.base
belief = vq.belief
evidence = vq.evidence
graph = vq.graph
reaction_probe = vq.reaction_probe
unified = base.unified

PROTOCOL = "train_only_property_aligned_balanced_transaction_transport_v1"


class PropertyAlignedCodeFlow(nn.Module):
    """Conditional code flow used as the learned part of the transport cost."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        codebook_size: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.base_logits = nn.Linear(int(hidden_dim), int(codebook_size))
        self.code_directions = nn.Parameter(
            torch.randn(int(codebook_size), int(latent_dim)) * 0.10
        )

    def forward(
        self,
        features: torch.Tensor,
        particles: torch.Tensor,
        latent_scale: float,
    ) -> torch.Tensor:
        state = self.encoder(features.float())
        base_logits = self.base_logits(state)
        directions = F.normalize(self.code_directions, dim=1)
        offsets = float(latent_scale) * particles @ directions.T
        return base_logits[:, None, :] + offsets[None, :, :]


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
    parser.add_argument("--vq-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "single_structural_change_after_vq": True,
        "fit_only_closed_reaction_grammar": True,
        "connected_fit_targets_only": True,
        "reaction_context_radius": 0,
        "fit_only_property_delta_labels": True,
        "property_aligned_codebook": True,
        "joint_balanced_particle_transport": True,
        "sinkhorn_code_assignment": True,
        "within_code_sampling_without_replacement": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
        "support_is_decoder_action_space": True,
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
        raise ValueError(f"Property-aligned transport preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Property-aligned transport property-count contract drift")
    if int(payload.get("balanced_min_codes", 0)) != 8:
        raise ValueError("Property-aligned transport must anchor eight codes")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Property-aligned transport implementation drift: "
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
        "radius_one_support_probe_sha256",
        "radius_zero_support_probe_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "set_evidence_records_sha256",
        "set_evidence_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
        "vq_summary_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Property-aligned transport locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    b22_summary, b22_checkpoint, b43_summary, atomic_summary = vq.check_locked_inputs(
        args, preregistration
    )
    locked = dict(preregistration["locked_inputs"])
    actual_vq_sha = belief.file_sha256(args.vq_summary)
    if actual_vq_sha != locked["vq_summary_sha256"]:
        raise ValueError(
            "Property-aligned transport VQ summary drift: "
            f"expected {locked['vq_summary_sha256']}, found {actual_vq_sha}"
        )
    vq_summary = json.loads(args.vq_summary.read_text(encoding="utf-8"))
    if vq_summary.get("protocol") != vq.PROTOCOL:
        raise ValueError("Property-aligned transport requires the locked VQ protocol")
    if vq_summary.get("decision") != (
        "stop_and_diagnose_vq_support_or_code_flow_without_gate_changes"
    ):
        raise ValueError("Property-aligned transport refuses a VQ decision drift")
    baseline = dict(vq_summary.get("metrics", {}))
    baseline_drift = {
        key: {"expected": expected, "actual": baseline.get(key)}
        for key, expected in dict(preregistration["vq_baseline"]).items()
        if not math.isclose(
            float(baseline.get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if baseline_drift:
        raise ValueError(f"Property-aligned transport VQ baseline drift: {baseline_drift}")
    return b22_summary, b22_checkpoint, b43_summary, atomic_summary, vq_summary


def property_vocabulary(examples: Sequence[tuple[object, str]]) -> list[str]:
    properties = {
        str(prop)
        for pair, _key in examples
        for prop, _direction in base.task_specs(pair.row)
    }
    if not properties:
        raise ValueError("No property labels found for property-aligned transport")
    return sorted(properties)


def property_request_vector(pair: object, vocabulary: Sequence[str]) -> np.ndarray:
    index = {prop: position for position, prop in enumerate(vocabulary)}
    request = np.zeros(len(vocabulary), dtype=np.float32)
    for prop, direction in base.task_specs(pair.row):
        if prop in index:
            request[index[prop]] = float(direction)
    scale = float(np.abs(request).sum())
    return request / max(1.0, scale)


def fit_property_effect(
    pair: object,
    vocabulary: Sequence[str],
    clip: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return normalized raw property deltas using fit source/target only."""

    index = {prop: position for position, prop in enumerate(vocabulary)}
    values = np.zeros(len(vocabulary), dtype=np.float32)
    observed = np.zeros(len(vocabulary), dtype=np.float32)
    attempted = 0
    evaluated_count = 0
    for prop, _direction in base.task_specs(pair.row):
        attempted += 1
        source_value = unified.score_property(pair.source_smiles, prop)
        target_value = unified.score_property(pair.target_smiles, prop)
        if source_value is None or target_value is None:
            continue
        if not math.isfinite(float(source_value)) or not math.isfinite(float(target_value)):
            continue
        normalizer = max(float(unified.PROPERTY_NORMALIZERS.get(prop, 1.0)), 1e-8)
        delta = (float(target_value) - float(source_value)) / normalizer
        values[index[prop]] = float(np.clip(delta, -float(clip), float(clip)))
        observed[index[prop]] = 1.0
        evaluated_count += 1
    return values, observed, attempted, evaluated_count


def property_aligned_transaction_vectors(
    transactions: Sequence[vq.ClosedReactionTransaction],
    examples: Sequence[tuple[object, str]],
    preregistration: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    vocabulary = property_vocabulary(examples)
    effect_sum: defaultdict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(vocabulary), dtype=np.float64)
    )
    effect_count: defaultdict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(vocabulary), dtype=np.float64)
    )
    attempted = 0
    evaluated_count = 0
    for pair, key in examples:
        values, observed, pair_attempted, pair_evaluated = fit_property_effect(
            pair, vocabulary, float(preregistration["property_delta_clip"])
        )
        effect_sum[key] += values.astype(np.float64)
        effect_count[key] += observed.astype(np.float64)
        attempted += pair_attempted
        evaluated_count += pair_evaluated

    structural_rows: list[np.ndarray] = []
    effect_rows: list[np.ndarray] = []
    observation_rows: list[np.ndarray] = []
    for transaction in transactions:
        key = vq.transaction_key(transaction.task, transaction.reaction_smarts)
        structural = vq.reaction_vector(
            transaction, int(preregistration["transaction_fingerprint_bits"])
        )
        structural = structural / max(float(np.linalg.norm(structural)), 1e-8)
        counts = effect_count[key]
        effect = np.divide(
            effect_sum[key],
            np.maximum(counts, 1.0),
            out=np.zeros_like(effect_sum[key]),
        ).astype(np.float32)
        observed = (counts > 0).astype(np.float32)
        structural_rows.append(structural.astype(np.float32))
        effect_rows.append(effect)
        observation_rows.append(observed)

    structural_matrix = np.stack(structural_rows)
    effect_matrix = np.stack(effect_rows)
    observation_matrix = np.stack(observation_rows)
    effect_norm = np.linalg.norm(effect_matrix, axis=1, keepdims=True)
    normalized_effect = effect_matrix / np.maximum(effect_norm, 1e-8)
    vectors = np.concatenate(
        [
            float(preregistration["structural_codebook_weight"]) * structural_matrix,
            float(preregistration["property_codebook_weight"]) * normalized_effect,
            float(preregistration["property_observation_weight"]) * observation_matrix,
        ],
        axis=1,
    ).astype(np.float32)
    manifest = {
        "property_vocabulary": vocabulary,
        "attempted_fit_property_labels": attempted,
        "evaluated_fit_property_labels": evaluated_count,
        "fit_property_label_coverage": evaluated_count / max(1, attempted),
        "transactions_with_property_effect": int((observation_matrix.sum(axis=1) > 0).sum()),
        "fit_only_source_target_property_access": True,
        "development_property_access": False,
    }
    return vectors, effect_matrix, vocabulary, manifest


def code_property_effects(
    transaction_effects: np.ndarray,
    transaction_codes: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    effects = np.zeros((int(codebook_size), transaction_effects.shape[1]), dtype=np.float32)
    for code in range(int(codebook_size)):
        members = transaction_effects[transaction_codes == code]
        if len(members):
            effects[code] = members.mean(axis=0)
    scale = np.linalg.norm(effects, axis=1, keepdims=True)
    return (effects / np.maximum(scale, 1e-8)).astype(np.float32)


def alignment_scores(
    requests: torch.Tensor, code_effects: torch.Tensor
) -> torch.Tensor:
    return torch.tanh(requests.float() @ code_effects.float().T)


def transport_loss(
    logits: torch.Tensor,
    target_codes: torch.Tensor,
    code_weights: torch.Tensor,
    preregistration: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mixture_logits = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
    classification = F.cross_entropy(
        mixture_logits, target_codes, weight=code_weights
    )
    probabilities = torch.softmax(logits, dim=2)
    normalized = F.normalize(probabilities, dim=2)
    cosine = normalized @ normalized.transpose(1, 2)
    off_diagonal = ~torch.eye(
        probabilities.shape[1], dtype=torch.bool, device=probabilities.device
    )
    particle_cosine = cosine[:, off_diagonal].mean()
    mean_mass = probabilities.mean(dim=(0, 1)).clamp_min(1e-8)
    load_balance = (
        mean_mass * torch.log(mean_mass * float(probabilities.shape[2]))
    ).sum()
    loss = (
        classification
        + float(preregistration["particle_diversity_weight"]) * particle_cosine
        + float(preregistration["load_balance_weight"]) * load_balance
    )
    return loss, {
        "classification_nll": classification,
        "particle_distribution_cosine": particle_cosine,
        "load_balance_kl": load_balance,
    }


def train_transport(
    model: PropertyAlignedCodeFlow,
    examples: Sequence[tuple[object, str]],
    key_to_code: Mapping[str, int],
    property_vocabulary_values: Sequence[str],
    code_effect_values: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    features = torch.from_numpy(
        np.stack(
            [
                vq.pair_feature(
                    pair,
                    int(preregistration["source_fingerprint_bits"]),
                    condition_slots=int(preregistration["condition_slots"]),
                    condition_dim=int(preregistration["condition_dim"]),
                )
                for pair, _key in examples
            ]
        )
    )
    requests = torch.from_numpy(
        np.stack(
            [
                property_request_vector(pair, property_vocabulary_values)
                for pair, _key in examples
            ]
        )
    )
    targets = torch.as_tensor(
        [key_to_code[key] for _pair, key in examples], dtype=torch.long
    )
    usage = torch.bincount(
        targets, minlength=int(preregistration["codebook_size"])
    ).float()
    code_weights = torch.where(usage > 0, usage.clamp_min(1.0).rsqrt(), torch.zeros_like(usage))
    positive = code_weights[code_weights > 0]
    code_weights = code_weights / positive.mean().clamp_min(1e-8)
    particles = vq.orthogonal_particles(
        20, int(preregistration["latent_dim"]), int(preregistration["latent_seed"])
    ).to(device)
    code_effect_tensor = torch.from_numpy(code_effect_values).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    batch_size = int(preregistration["batch_size"])
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(examples)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            batch_features = features[indices].to(device)
            batch_requests = requests[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch_features,
                particles,
                float(preregistration["latent_logit_scale"]),
            )
            logits = logits + float(
                preregistration["condition_alignment_logit_weight"]
            ) * alignment_scores(batch_requests, code_effect_tensor)[:, None, :]
            logits = logits / float(preregistration["training_temperature"])
            loss, metrics = transport_loss(
                logits, batch_targets, code_weights.to(device), preregistration
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite property-aligned transport loss: {metrics}"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in metrics.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "training_examples": len(examples),
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    model.eval().requires_grad_(False)
    return history


def balanced_code_capacities(
    available_codes: Sequence[int],
    preference: torch.Tensor,
    attempts: int,
    min_codes: int,
    temperature: float,
    gumbel_scale: float,
    generator: torch.Generator,
) -> dict[int, int]:
    """Sample one joint code-capacity vector with a preregistered diversity floor."""

    available = [int(value) for value in available_codes]
    if not available:
        return {}
    local_preference = preference[available].detach().float().cpu()
    uniform = torch.rand(len(available), generator=generator).clamp_(1e-7, 1 - 1e-7)
    gumbel = -torch.log(-torch.log(uniform))
    ordering = torch.argsort(
        local_preference + float(gumbel_scale) * gumbel, descending=True
    ).tolist()
    anchored = min(int(attempts), int(min_codes), len(available))
    capacities: Counter[int] = Counter(
        available[index] for index in ordering[:anchored]
    )
    remaining = int(attempts) - anchored
    if remaining > 0:
        probabilities = torch.softmax(
            local_preference / max(float(temperature), 1e-6), dim=0
        )
        sampled = torch.multinomial(
            probabilities, remaining, replacement=True, generator=generator
        )
        capacities.update(available[int(index)] for index in sampled.tolist())
    if sum(capacities.values()) != int(attempts):
        raise RuntimeError("Balanced code capacities do not sum to the attempt budget")
    return dict(capacities)


def sinkhorn_hard_assignment(
    logits: torch.Tensor,
    code_slots: Sequence[int],
    epsilon: float,
    iterations: int,
) -> tuple[list[int], list[float]]:
    """Couple particles and exact code slots, then round without changing capacity."""

    if logits.shape[0] != len(code_slots):
        raise ValueError("Sinkhorn transport needs one slot per particle")
    slot_index = torch.as_tensor(code_slots, dtype=torch.long, device=logits.device)
    score = logits[:, slot_index]
    scaled = score / max(float(epsilon), 1e-6)
    scaled = scaled - scaled.max()
    plan = torch.exp(scaled).clamp_min(1e-12)
    for _ in range(int(iterations)):
        plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        plan = plan / plan.sum(dim=0, keepdim=True).clamp_min(1e-12)

    remaining_rows = set(range(plan.shape[0]))
    remaining_slots = set(range(plan.shape[1]))
    assigned_slot = [-1] * plan.shape[0]
    assigned_mass = [0.0] * plan.shape[0]
    while remaining_rows:
        best = max(
            (
                (float(plan[row, slot].detach().cpu()), -row, -slot, row, slot)
                for row in remaining_rows
                for slot in remaining_slots
            ),
            key=lambda value: value[:3],
        )
        mass, _neg_row, _neg_slot, row, slot = best
        assigned_slot[row] = slot
        assigned_mass[row] = mass
        remaining_rows.remove(row)
        remaining_slots.remove(slot)
    assigned_codes = [int(code_slots[slot]) for slot in assigned_slot]
    if Counter(assigned_codes) != Counter(int(value) for value in code_slots):
        raise RuntimeError("Sinkhorn rounding changed the joint code capacity")
    return assigned_codes, assigned_mass


@torch.no_grad()
def freeze_balanced_candidates(
    model: PropertyAlignedCodeFlow,
    development_pairs: Sequence[object],
    transactions: Sequence[vq.ClosedReactionTransaction],
    transaction_codes: Sequence[int],
    property_vocabulary_values: Sequence[str],
    code_effect_values: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    particles = vq.orthogonal_particles(
        20, int(preregistration["latent_dim"]), int(preregistration["latent_seed"])
    ).to(device)
    code_effect_tensor = torch.from_numpy(code_effect_values).to(device)
    rows: list[dict[str, object]] = []
    support_counts: list[int] = []
    unique_support_counts: list[int] = []
    sampled_codes_by_condition: list[int] = []
    sampled_transactions_by_condition: list[int] = []
    sampled_code_counts: Counter[int] = Counter()
    identity_attempts = 0
    for pair_index, pair in enumerate(development_pairs):
        task = base.task_key(pair.row)
        actions = vq.applicable_actions(
            pair.source_smiles,
            task,
            transactions,
            transaction_codes,
            preregistration,
        )
        support_counts.append(len(actions))
        unique_support_counts.append(len({action.smiles for action in actions}))
        condition_id = f"train_only_dev_{pair_index:04d}"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(preregistration["seed"]) * 100000 + pair_index)
        if actions:
            features = torch.from_numpy(
                vq.pair_feature(
                    pair,
                    int(preregistration["source_fingerprint_bits"]),
                    condition_slots=int(preregistration["condition_slots"]),
                    condition_dim=int(preregistration["condition_dim"]),
                )
            )[None, :].to(device)
            request = torch.from_numpy(
                property_request_vector(pair, property_vocabulary_values)
            )[None, :].to(device)
            logits = model(
                features,
                particles,
                float(preregistration["latent_logit_scale"]),
            )[0]
            logits = logits + float(
                preregistration["condition_alignment_logit_weight"]
            ) * alignment_scores(request, code_effect_tensor)[0][None, :]
            logits = logits / float(preregistration["generation_temperature"])
            by_code: defaultdict[int, list[vq.ApplicableAction]] = defaultdict(list)
            for action in actions:
                by_code[action.code_index].append(action)
            available_codes = sorted(by_code)
            preference = logits.mean(dim=0)
            capacities = balanced_code_capacities(
                available_codes,
                preference,
                attempts=20,
                min_codes=int(preregistration["balanced_min_codes"]),
                temperature=float(preregistration["capacity_temperature"]),
                gumbel_scale=float(preregistration["capacity_gumbel_scale"]),
                generator=generator,
            )
            code_slots = [
                code
                for code, count in sorted(capacities.items())
                for _ in range(int(count))
            ]
            permutation = torch.randperm(len(code_slots), generator=generator).tolist()
            code_slots = [code_slots[index] for index in permutation]
            sampled_codes, transport_mass = sinkhorn_hard_assignment(
                logits,
                code_slots,
                epsilon=float(preregistration["sinkhorn_epsilon"]),
                iterations=int(preregistration["sinkhorn_iterations"]),
            )
            action_queues: dict[int, list[vq.ApplicableAction]] = {}
            action_offsets: Counter[int] = Counter()
            for code, choices in by_code.items():
                order = torch.randperm(len(choices), generator=generator).tolist()
                action_queues[code] = [choices[index] for index in order]
            sampled_actions: list[tuple[vq.ApplicableAction, float, float]] = []
            for particle_index, code in enumerate(sampled_codes):
                choices = action_queues[code]
                choice_index = action_offsets[code] % len(choices)
                action_offsets[code] += 1
                probability = float(torch.softmax(logits[particle_index], dim=0)[code].cpu())
                sampled_actions.append(
                    (choices[choice_index], probability, transport_mass[particle_index])
                )
                sampled_code_counts[code] += 1
            sampled_codes_by_condition.append(len(set(sampled_codes)))
            sampled_transactions_by_condition.append(
                len({action.transaction_key for action, _prob, _mass in sampled_actions})
            )
        else:
            sampled_actions = []
            sampled_codes = [-1] * 20
            capacities = {}
            sampled_codes_by_condition.append(0)
            sampled_transactions_by_condition.append(0)

        for attempt in range(1, 21):
            if sampled_actions:
                action, code_probability, transport_mass_value = sampled_actions[attempt - 1]
                smiles = action.smiles
                transaction_id = action.transaction_key
                component_count = action.component_count
                fit_source = action.fit_source_smiles
                identity = False
                code_index = sampled_codes[attempt - 1]
                code_capacity = capacities[code_index]
            else:
                smiles = pair.source_smiles
                transaction_id = "identity_no_applicable_transaction"
                component_count = 0
                fit_source = ""
                identity = True
                code_index = -1
                code_probability = 1.0
                transport_mass_value = 1.0 / 20.0
                code_capacity = 20
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
                    "unique_product_support_size": len({action.smiles for action in actions}),
                    "reaction_code": code_index,
                    "reaction_code_probability": code_probability,
                    "joint_code_capacity": code_capacity,
                    "sinkhorn_transport_mass": transport_mass_value,
                    "transaction_key": transaction_id,
                    "transaction_components": component_count,
                    "fit_source_smiles": fit_source,
                    "identity_transaction": identity,
                    "latent_norm": float(particles[attempt - 1].norm().cpu()),
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
                        "stage": "freeze_property_aligned_balanced_transport",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    code_probabilities = np.asarray(list(sampled_code_counts.values()), dtype=np.float64)
    code_probabilities = code_probabilities / max(1.0, code_probabilities.sum())
    sampled_perplexity = (
        float(np.exp(-(code_probabilities * np.log(code_probabilities)).sum()))
        if len(code_probabilities)
        else 0.0
    )
    manifest = {
        "conditions": len(development_pairs),
        "conditions_with_nonidentity_support": sum(count > 0 for count in support_counts),
        "condition_support_rate": sum(count > 0 for count in support_counts)
        / max(1, len(support_counts)),
        "mean_applicable_transactions": float(np.mean(support_counts)),
        "mean_unique_product_support": float(np.mean(unique_support_counts)),
        "mean_sampled_codes_per_condition": float(np.mean(sampled_codes_by_condition)),
        "mean_sampled_transactions_per_condition": float(
            np.mean(sampled_transactions_by_condition)
        ),
        "active_sampled_codes": len(sampled_code_counts),
        "sampled_code_perplexity": sampled_perplexity,
        "identity_attempt_rate": identity_attempts / max(1, len(rows)),
        "joint_capacity_before_molecule_commit": True,
        "within_code_without_replacement": True,
    }
    return rows, manifest


def gate_result(
    metrics: Mapping[str, object],
    codebook: Mapping[str, object],
    property_labels: Mapping[str, object],
    support: Mapping[str, object],
    vq_baseline: Mapping[str, object],
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
        "fit_active_codes": {"value": codebook["active_codes"], "threshold": thresholds["fit_active_codes"]},
        "fit_property_label_coverage": {"value": property_labels["fit_property_label_coverage"], "threshold": thresholds["fit_property_label_coverage"]},
        "condition_support_rate": {"value": support["condition_support_rate"], "threshold": thresholds["condition_support_rate"]},
        "mean_sampled_codes_per_condition": {"value": support["mean_sampled_codes_per_condition"], "threshold": thresholds["mean_sampled_codes_per_condition"]},
        "unique_delta_vs_vq": {"value": float(metrics["mean_unique_valid"]) - float(vq_baseline["mean_unique_valid"]), "threshold": thresholds["unique_delta_vs_vq"]},
        "strict_delta_vs_vq": {"value": float(metrics["strict_any20"]) - float(vq_baseline["strict_any20"]), "threshold": thresholds["strict_delta_vs_vq"]},
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
        raise ValueError(f"Completed property-aligned result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = torch.device(str(args.device))
    if device.type != "cpu":
        raise ValueError("The property-aligned balanced signal is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, _b43_summary, _atomic_summary, vq_summary = (
        check_locked_inputs(args, preregistration)
    )
    selected_pairs, reconstruction = evidence.reconstruct_locked_b36_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = vq.b43.b41.b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    transactions, training_examples, grammar_manifest = vq.build_fit_grammar(
        fit_pairs, preregistration
    )
    vectors, transaction_effects, vocabulary, property_manifest = (
        property_aligned_transaction_vectors(
            transactions, training_examples, preregistration
        )
    )
    centroids, transaction_codes, codebook_manifest = vq.deterministic_kmeans(
        vectors,
        int(preregistration["codebook_size"]),
        int(preregistration["kmeans_iterations"]),
        int(preregistration["codebook_seed"]),
    )
    code_effect_values = code_property_effects(
        transaction_effects,
        transaction_codes,
        int(preregistration["codebook_size"]),
    )
    keys = [
        vq.transaction_key(transaction.task, transaction.reaction_smarts)
        for transaction in transactions
    ]
    key_to_code = {
        key: int(code) for key, code in zip(keys, transaction_codes.tolist())
    }
    input_dim = int(preregistration["source_fingerprint_bits"]) + int(
        preregistration["condition_slots"]
    ) * int(preregistration["condition_dim"])
    model = PropertyAlignedCodeFlow(
        input_dim=input_dim,
        hidden_dim=int(preregistration["hidden_dim"]),
        codebook_size=int(preregistration["codebook_size"]),
        latent_dim=int(preregistration["latent_dim"]),
    ).to(device)
    history = train_transport(
        model,
        training_examples,
        key_to_code,
        vocabulary,
        code_effect_values,
        preregistration,
        device,
    )
    checkpoint_path = (
        args.output_dir / "property_aligned_balanced_transaction_transport.pt"
    )
    transaction_catalog = [
        {
            "task": transaction.task,
            "reaction_smarts": list(transaction.reaction_smarts),
            "fit_source_smiles": transaction.fit_source_smiles,
            "fit_target_smiles": transaction.fit_target_smiles,
            "component_count": transaction.component_count,
            "code_index": int(code),
        }
        for transaction, code in zip(transactions, transaction_codes.tolist())
    ]
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "model_config": {
                "input_dim": input_dim,
                "hidden_dim": int(preregistration["hidden_dim"]),
                "codebook_size": int(preregistration["codebook_size"]),
                "latent_dim": int(preregistration["latent_dim"]),
            },
            "property_vocabulary": vocabulary,
            "code_property_effects": torch.from_numpy(code_effect_values),
            "property_aligned_centroids": torch.from_numpy(centroids),
            "transaction_catalog": transaction_catalog,
            "codebook_manifest": codebook_manifest,
            "property_manifest": property_manifest,
        },
        checkpoint_path,
    )
    frozen, development_support = freeze_balanced_candidates(
        model,
        development_pairs,
        transactions,
        transaction_codes.tolist(),
        vocabulary,
        code_effect_values,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_transactions.csv"
    vq.write_rows(frozen_path, frozen)
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = vq.evaluate_frozen_transactions(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_transactions.csv"
    vq.write_rows(evaluated_path, evaluated)
    gate = gate_result(
        metrics,
        codebook_manifest,
        property_manifest,
        development_support,
        dict(preregistration["vq_baseline"]),
        preregistration,
    )
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "reconstruction": reconstruction,
        "split": split,
        "fit_grammar": grammar_manifest,
        "property_labels": property_manifest,
        "codebook": codebook_manifest,
        "development_support": development_support,
        "single_structural_change_after_vq": True,
        "fit_only_closed_reaction_grammar": True,
        "connected_fit_targets_only": True,
        "reaction_context_radius": 0,
        "fit_only_property_delta_labels": True,
        "property_aligned_codebook": True,
        "joint_balanced_particle_transport": True,
        "sinkhorn_code_assignment": True,
        "within_code_sampling_without_replacement": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
        "support_is_decoder_action_space": True,
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
            "advance_property_aligned_transport_to_once_only_fresh_confirmation"
            if gate["passed"]
            else "stop_property_aligned_transport_without_gate_changes"
        ),
        "training": history,
        "vq_baseline": dict(preregistration["vq_baseline"]),
        "atomic_baseline": dict(preregistration["atomic_baseline"]),
        "b43_baseline": dict(preregistration["b43_baseline"]),
        "internal_gate": gate,
        "metrics": metrics,
        "manifest": manifest,
        "checkpoint": str(checkpoint_path),
        "locked_vq_summary_decision": vq_summary.get("decision"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
