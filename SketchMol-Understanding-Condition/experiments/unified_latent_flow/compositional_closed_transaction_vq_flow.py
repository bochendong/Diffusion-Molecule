#!/usr/bin/env python3
"""Train a source-conditioned VQ flow over complete closed transactions.

The decoder vocabulary is learned only from fit source/target pairs.  Each
vocabulary item is a tuple of mapped local reaction components that exactly
replays a connected fit target and produces a sanitized single molecule.  A
deterministic VQ codebook clusters transaction deltas; a conditional latent flow
then maps source graph fingerprints, sanitized property tokens, and one of
twenty orthogonal particles to a distribution over transaction codes.

At generation time the complete set of source-applicable, sanitized transaction
actions is the decoder support.  Each particle samples one code and one action
from that code.  Only these twenty actions are committed.  There is no larger
molecular sample pool, molecule ranking, target access, property-oracle access,
retry, repair, or second edit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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

import atomic_closed_transaction_latent_decoder as atomic  # noqa: E402
import probe_compositional_closed_reaction_templates as reaction_probe  # noqa: E402


b43 = atomic.b43
base = atomic.base
belief = atomic.belief
evidence = atomic.evidence
graph = atomic.graph

PROTOCOL = "train_only_compositional_closed_transaction_vq_flow_v1"


@dataclass(frozen=True)
class ClosedReactionTransaction:
    task: str
    reaction_smarts: tuple[str, ...]
    fit_source_smiles: str
    fit_target_smiles: str
    component_count: int


@dataclass(frozen=True)
class ApplicableAction:
    smiles: str
    code_index: int
    transaction_key: str
    component_count: int
    fit_source_smiles: str


class ConditionalTransactionCodeFlow(nn.Module):
    """Source/condition flow whose latent particles transport code mass."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        codebook_size: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
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
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "fit_only_closed_reaction_grammar": True,
        "connected_fit_targets_only": True,
        "reaction_context_radius": 0,
        "deterministic_vq_codebook": True,
        "source_conditioned_code_flow": True,
        "orthogonal_latent_particles": True,
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
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Compositional VQ preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Compositional VQ property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Compositional VQ implementation drift: "
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
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Compositional VQ locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "atomic_checkpoint_sha256": args.atomic_checkpoint,
        "atomic_summary_sha256": args.atomic_summary,
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_records_sha256": args.b36_records,
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_summary_sha256": args.b41_summary,
        "b43_checkpoint_sha256": args.b43_checkpoint,
        "b43_summary_sha256": args.b43_summary,
        "radius_one_support_probe_sha256": args.radius_one_support_probe,
        "radius_zero_support_probe_sha256": args.radius_zero_support_probe,
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "set_evidence_records_sha256": args.set_evidence_records,
        "set_evidence_summary_sha256": args.set_evidence_summary,
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
    }
    drift = {
        name: {"expected": locked[name], "actual": belief.file_sha256(path)}
        for name, path in paths.items()
        if belief.file_sha256(path) != locked[name]
    }
    if drift:
        raise ValueError(f"Compositional VQ locked input drift: {drift}")

    b22_summary, b22_checkpoint = b43.evidence.patch_evidence.load_locked_b22(
        args, preregistration
    )
    set_summary = json.loads(args.set_evidence_summary.read_text(encoding="utf-8"))
    if set_summary.get("protocol") != evidence.PROTOCOL or not bool(
        dict(set_summary.get("gate", {})).get("passed")
    ):
        raise ValueError("Compositional VQ requires the passing B42 representation")

    b43_summary = json.loads(args.b43_summary.read_text(encoding="utf-8"))
    if b43_summary.get("protocol") != b43.PROTOCOL:
        raise ValueError("Compositional VQ requires the locked B43 result")
    atomic_summary = json.loads(args.atomic_summary.read_text(encoding="utf-8"))
    if atomic_summary.get("protocol") != atomic.PROTOCOL:
        raise ValueError("Compositional VQ requires the locked atomic result")
    atomic_drift = {
        key: {"expected": expected, "actual": atomic_summary.get("metrics", {}).get(key)}
        for key, expected in dict(preregistration["atomic_baseline"]).items()
        if not math.isclose(
            float(atomic_summary.get("metrics", {}).get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if atomic_drift:
        raise ValueError(f"Compositional VQ atomic baseline drift: {atomic_drift}")
    atomic_checkpoint = torch.load(
        args.atomic_checkpoint, map_location="cpu", weights_only=False
    )
    if atomic_checkpoint.get("stage") != atomic.PROTOCOL:
        raise ValueError("Compositional VQ refuses a non-atomic checkpoint")

    radius_one = json.loads(args.radius_one_support_probe.read_text(encoding="utf-8"))
    radius_zero = json.loads(args.radius_zero_support_probe.read_text(encoding="utf-8"))
    for name, payload in (("radius_one", radius_one), ("radius_zero", radius_zero)):
        if payload.get("protocol") != "source_disjoint_compositional_closed_reaction_support_v1":
            raise ValueError(f"Compositional VQ refuses {name} support-probe drift")
        if int(payload.get("split", {}).get("fit_development_source_overlap", -1)) != 0:
            raise ValueError(f"Compositional VQ refuses {name} source overlap")
    radius_zero_support = dict(radius_zero.get("cross_support", {}))
    if not math.isclose(
        float(radius_zero_support.get("condition_support_rate", math.nan)),
        float(preregistration["radius_zero_evidence"]["condition_support_rate"]),
        abs_tol=1e-12,
    ):
        raise ValueError("Compositional VQ radius-zero evidence drift")
    return b22_summary, b22_checkpoint, b43_summary, atomic_summary


def transaction_key(task: str, reaction_smarts: Sequence[str]) -> str:
    payload = json.dumps(
        {"task": task, "reaction_smarts": list(reaction_smarts)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fit_grammar(
    fit_pairs: Sequence[object], preregistration: Mapping[str, object]
) -> tuple[
    list[ClosedReactionTransaction],
    list[tuple[object, str]],
    dict[str, object],
]:
    by_key: dict[str, ClosedReactionTransaction] = {}
    training_examples: list[tuple[object, str]] = []
    counts: Counter[str] = Counter()
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for index, pair in enumerate(fit_pairs, start=1):
        task = base.task_key(pair.row)
        counts["fit_pairs"] += 1
        by_task[task]["pairs"] += 1
        target = graph.canonical_smiles(pair.target_smiles)
        if not target or "." in target:
            counts["disconnected_or_invalid_target"] += 1
            by_task[task]["disconnected_or_invalid_target"] += 1
            continue
        try:
            templates = reaction_probe.extract_templates(
                pair.source,
                pair.target,
                int(preregistration["reaction_context_radius"]),
            )
        except Exception:
            counts["template_extraction_error"] += 1
            continue
        if not templates:
            counts["template_empty"] += 1
            continue
        products, _raw = reaction_probe.apply_component_tuple(
            pair.source_smiles,
            templates,
            max_frontier=int(preregistration["max_reaction_frontier"]),
        )
        if target not in products:
            counts["nonexact_self_replay"] += 1
            continue
        smarts = tuple(template.reaction_smarts for template in templates)
        key = transaction_key(task, smarts)
        by_key.setdefault(
            key,
            ClosedReactionTransaction(
                task=task,
                reaction_smarts=smarts,
                fit_source_smiles=pair.source_smiles,
                fit_target_smiles=target,
                component_count=len(smarts),
            ),
        )
        training_examples.append((pair, key))
        counts["exact_self_replay_pairs"] += 1
        by_task[task]["exact_self_replay_pairs"] += 1
        if index % 128 == 0 or index == len(fit_pairs):
            print(
                json.dumps(
                    {
                        "stage": "build_fit_closed_transaction_grammar",
                        "fit_pairs": index,
                        "exact_pairs": counts["exact_self_replay_pairs"],
                        "unique_transactions": len(by_key),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    transactions = [by_key[key] for key in sorted(by_key)]
    manifest = {
        "counts": dict(counts),
        "by_task": {task: dict(value) for task, value in sorted(by_task.items())},
        "unique_transactions": len(transactions),
        "training_examples": len(training_examples),
        "connected_target_exact_replay_rate": counts["exact_self_replay_pairs"]
        / max(
            1,
            counts["fit_pairs"] - counts["disconnected_or_invalid_target"],
        ),
    }
    return transactions, training_examples, manifest


def reaction_vector(
    transaction: ClosedReactionTransaction, bits: int
) -> np.ndarray:
    source = atomic.fingerprint(transaction.fit_source_smiles, int(bits))
    target = atomic.fingerprint(transaction.fit_target_smiles, int(bits))
    difference = target - source
    scalars = np.asarray(
        [
            min(1.0, float(transaction.component_count) / 3.0),
            float(np.abs(difference).mean()),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [np.maximum(difference, 0.0), np.maximum(-difference, 0.0), scalars]
    ).astype(np.float32)


def deterministic_kmeans(
    values: np.ndarray, clusters: int, iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if values.ndim != 2 or len(values) < int(clusters):
        raise ValueError(
            f"VQ codebook needs at least {clusters} transactions, found {len(values)}"
        )
    generator = np.random.default_rng(int(seed))
    centroids = [values[int(generator.integers(0, len(values)))].copy()]
    for _ in range(1, int(clusters)):
        distances = np.min(
            np.stack(
                [np.square(values - centroid).sum(axis=1) for centroid in centroids],
                axis=1,
            ),
            axis=1,
        )
        total = float(distances.sum())
        if total <= 0:
            candidates = [index for index in range(len(values)) if not any(np.array_equal(values[index], value) for value in centroids)]
            chosen = candidates[0] if candidates else len(centroids) % len(values)
        else:
            chosen = int(generator.choice(len(values), p=distances / total))
        centroids.append(values[chosen].copy())
    matrix = np.stack(centroids)
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(int(iterations)):
        distances = np.square(values[:, None, :] - matrix[None, :, :]).sum(axis=2)
        new_labels = distances.argmin(axis=1).astype(np.int64)
        new_matrix = matrix.copy()
        nearest = distances.min(axis=1)
        for code in range(int(clusters)):
            members = values[new_labels == code]
            if len(members):
                new_matrix[code] = members.mean(axis=0)
            else:
                replacement = int(np.argmax(nearest))
                new_matrix[code] = values[replacement]
                new_labels[replacement] = code
                nearest[replacement] = -1.0
        converged = np.array_equal(new_labels, labels) and np.allclose(new_matrix, matrix)
        labels, matrix = new_labels, new_matrix
        if converged:
            break
    usage = np.bincount(labels, minlength=int(clusters)).astype(np.float64)
    probabilities = usage / max(1.0, usage.sum())
    positive = probabilities[probabilities > 0]
    perplexity = float(np.exp(-(positive * np.log(positive)).sum()))
    manifest = {
        "codebook_size": int(clusters),
        "active_codes": int((usage > 0).sum()),
        "code_perplexity": perplexity,
        "min_code_usage": int(usage.min()),
        "max_code_usage": int(usage.max()),
        "usage": usage.astype(int).tolist(),
        "iterations": int(iterations),
        "seed": int(seed),
    }
    return matrix.astype(np.float32), labels, manifest


def pair_feature(pair: object, bits: int) -> np.ndarray:
    source = atomic.fingerprint(pair.source_smiles, int(bits))
    condition = np.asarray(pair.condition, dtype=np.float32).reshape(-1)
    return np.concatenate([source, condition]).astype(np.float32)


def orthogonal_particles(count: int, dimension: int, seed: int) -> torch.Tensor:
    return atomic.orthogonal_particles(count, dimension, seed)


def code_flow_loss(
    logits: torch.Tensor,
    target_codes: torch.Tensor,
    preregistration: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = torch.softmax(logits, dim=2)
    gather_index = target_codes[:, None, None].expand(-1, logits.shape[1], 1)
    target_mass = probabilities.gather(2, gather_index).squeeze(2).clamp(1e-8, 1.0 - 1e-8)
    any_probability = 1.0 - torch.prod(1.0 - target_mass, dim=1)
    any_nll = -torch.log(any_probability.clamp_min(1e-8)).mean()
    participation = -torch.log(target_mass).mean()
    normalized = F.normalize(probabilities, dim=2)
    cosine = normalized @ normalized.transpose(1, 2)
    off_diagonal = ~torch.eye(
        probabilities.shape[1], dtype=torch.bool, device=probabilities.device
    )
    diversity = cosine[:, off_diagonal].mean()
    entropy = -(
        probabilities * torch.log(probabilities.clamp_min(1e-8))
    ).sum(dim=2).mean()
    mean_code_mass = probabilities.mean(dim=(0, 1)).clamp_min(1e-8)
    load_balance = (
        mean_code_mass
        * torch.log(mean_code_mass * float(probabilities.shape[2]))
    ).sum()
    loss = (
        any_nll
        + float(preregistration["participation_weight"]) * participation
        + float(preregistration["particle_diversity_weight"]) * diversity
        - float(preregistration["entropy_weight"]) * entropy
        + float(preregistration["load_balance_weight"]) * load_balance
    )
    return loss, {
        "any_code_nll": any_nll,
        "participation_nll": participation,
        "particle_distribution_cosine": diversity,
        "distribution_entropy": entropy,
        "load_balance_kl": load_balance,
        "mean_target_code_mass": target_mass.mean(),
    }


def train_code_flow(
    model: ConditionalTransactionCodeFlow,
    examples: Sequence[tuple[object, str]],
    key_to_code: Mapping[str, int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    features = torch.from_numpy(
        np.stack(
            [
                pair_feature(pair, int(preregistration["source_fingerprint_bits"]))
                for pair, _key in examples
            ]
        )
    )
    targets = torch.as_tensor([key_to_code[key] for _pair, key in examples], dtype=torch.long)
    particles = orthogonal_particles(
        20, int(preregistration["latent_dim"]), int(preregistration["latent_seed"])
    ).to(device)
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
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch_features,
                particles,
                float(preregistration["latent_logit_scale"]),
            ) / float(preregistration["training_temperature"])
            loss, metrics = code_flow_loss(logits, batch_targets, preregistration)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite compositional VQ loss: {metrics}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(preregistration["grad_clip"]))
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


def templates_from_transaction(
    transaction: ClosedReactionTransaction,
) -> tuple[reaction_probe.ComponentTemplate, ...]:
    return tuple(
        reaction_probe.ComponentTemplate(
            reaction_smarts=smarts,
            changed_slots=(),
            context_slots=(),
        )
        for smarts in transaction.reaction_smarts
    )


def applicable_actions(
    source_smiles: str,
    task: str,
    transactions: Sequence[ClosedReactionTransaction],
    transaction_codes: Sequence[int],
    preregistration: Mapping[str, object],
) -> list[ApplicableAction]:
    source = graph.canonical_smiles(source_smiles)
    by_key: dict[tuple[str, int], ApplicableAction] = {}
    for transaction, code_index in zip(transactions, transaction_codes):
        if transaction.task != task:
            continue
        products, _raw = reaction_probe.apply_component_tuple(
            source,
            templates_from_transaction(transaction),
            max_frontier=int(preregistration["max_reaction_frontier"]),
        )
        key = transaction_key(transaction.task, transaction.reaction_smarts)
        for product in products:
            if not product or product == source or "." in product:
                continue
            action = ApplicableAction(
                smiles=product,
                code_index=int(code_index),
                transaction_key=key,
                component_count=int(transaction.component_count),
                fit_source_smiles=transaction.fit_source_smiles,
            )
            by_key.setdefault((product, int(code_index)), action)
    return [by_key[key] for key in sorted(by_key)]


@torch.no_grad()
def freeze_candidates(
    model: ConditionalTransactionCodeFlow,
    development_pairs: Sequence[object],
    transactions: Sequence[ClosedReactionTransaction],
    transaction_codes: Sequence[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    particles = orthogonal_particles(
        20, int(preregistration["latent_dim"]), int(preregistration["latent_seed"])
    ).to(device)
    rows: list[dict[str, object]] = []
    support_counts: list[int] = []
    unique_support_counts: list[int] = []
    sampled_codes_by_condition: list[int] = []
    sampled_code_counts: Counter[int] = Counter()
    identity_attempts = 0
    for pair_index, pair in enumerate(development_pairs):
        task = base.task_key(pair.row)
        actions = applicable_actions(
            pair.source_smiles,
            task,
            transactions,
            transaction_codes,
            preregistration,
        )
        support_counts.append(len(actions))
        unique_support_counts.append(len({action.smiles for action in actions}))
        condition_id = f"train_only_dev_{pair_index:04d}"
        if actions:
            features = torch.from_numpy(
                pair_feature(pair, int(preregistration["source_fingerprint_bits"]))
            )[None, :].to(device)
            logits = model(
                features,
                particles,
                float(preregistration["latent_logit_scale"]),
            )[0] / float(preregistration["generation_temperature"])
            by_code: defaultdict[int, list[ApplicableAction]] = defaultdict(list)
            for action in actions:
                by_code[action.code_index].append(action)
            available_codes = sorted(by_code)
            mask = torch.full_like(logits, float("-inf"))
            mask[:, available_codes] = logits[:, available_codes]
            probabilities = torch.softmax(mask, dim=1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(preregistration["seed"]) * 100000 + pair_index)
            sampled_actions: list[tuple[ApplicableAction, float]] = []
            sampled_codes: list[int] = []
            for particle_index, probability in enumerate(probabilities):
                code = int(torch.multinomial(probability.cpu(), 1, generator=generator).item())
                choices = by_code[code]
                action_index = int(
                    torch.randint(len(choices), (1,), generator=generator).item()
                )
                sampled_actions.append((choices[action_index], float(probability[code].cpu())))
                sampled_codes.append(code)
                sampled_code_counts[code] += 1
            sampled_codes_by_condition.append(len(set(sampled_codes)))
        else:
            sampled_actions = []
            sampled_codes = [-1] * 20
            sampled_codes_by_condition.append(0)
        for attempt in range(1, 21):
            if sampled_actions:
                action, code_probability = sampled_actions[attempt - 1]
                smiles = action.smiles
                transaction_id = action.transaction_key
                component_count = action.component_count
                fit_source = action.fit_source_smiles
                identity = False
                code_index = sampled_codes[attempt - 1]
            else:
                smiles = pair.source_smiles
                transaction_id = "identity_no_applicable_transaction"
                component_count = 0
                fit_source = ""
                identity = True
                code_index = -1
                code_probability = 1.0
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
                        "stage": "freeze_compositional_closed_transactions",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    probabilities = np.asarray(list(sampled_code_counts.values()), dtype=np.float64)
    probabilities = probabilities / max(1.0, probabilities.sum())
    sampled_perplexity = float(
        np.exp(-(probabilities * np.log(probabilities)).sum())
    ) if len(probabilities) else 0.0
    manifest = {
        "conditions": len(development_pairs),
        "conditions_with_nonidentity_support": sum(count > 0 for count in support_counts),
        "condition_support_rate": sum(count > 0 for count in support_counts)
        / max(1, len(support_counts)),
        "mean_applicable_transactions": float(np.mean(support_counts)),
        "mean_unique_product_support": float(np.mean(unique_support_counts)),
        "mean_sampled_codes_per_condition": float(np.mean(sampled_codes_by_condition)),
        "active_sampled_codes": len(sampled_code_counts),
        "sampled_code_perplexity": sampled_perplexity,
        "identity_attempt_rate": identity_attempts / max(1, len(rows)),
    }
    return rows, manifest


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gate_result(
    metrics: Mapping[str, object],
    codebook: Mapping[str, object],
    support: Mapping[str, object],
    atomic_baseline: Mapping[str, object],
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
        "fit_code_perplexity": {"value": codebook["code_perplexity"], "threshold": thresholds["fit_code_perplexity"]},
        "condition_support_rate": {"value": support["condition_support_rate"], "threshold": thresholds["condition_support_rate"]},
        "mean_sampled_codes_per_condition": {"value": support["mean_sampled_codes_per_condition"], "threshold": thresholds["mean_sampled_codes_per_condition"]},
        "unique_delta_vs_atomic": {"value": float(metrics["mean_unique_valid"]) - float(atomic_baseline["mean_unique_valid"]), "threshold": thresholds["unique_delta_vs_atomic"]},
        "strict_delta_vs_atomic": {"value": float(metrics["strict_any20"]) - float(atomic_baseline["strict_any20"]), "threshold": thresholds["strict_delta_vs_atomic"]},
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
        raise ValueError(f"Completed compositional VQ result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = torch.device(str(args.device))
    if device.type != "cpu":
        raise ValueError("The preregistered compositional VQ signal is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, b43_summary, atomic_summary = check_locked_inputs(
        args, preregistration
    )
    selected_pairs, reconstruction = evidence.reconstruct_locked_b36_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b43.b41.b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    transactions, training_examples, grammar_manifest = build_fit_grammar(
        fit_pairs, preregistration
    )
    vectors = np.stack(
        [
            reaction_vector(
                transaction, int(preregistration["transaction_fingerprint_bits"])
            )
            for transaction in transactions
        ]
    )
    centroids, transaction_codes, codebook_manifest = deterministic_kmeans(
        vectors,
        int(preregistration["codebook_size"]),
        int(preregistration["kmeans_iterations"]),
        int(preregistration["codebook_seed"]),
    )
    keys = [
        transaction_key(transaction.task, transaction.reaction_smarts)
        for transaction in transactions
    ]
    key_to_code = {
        key: int(code) for key, code in zip(keys, transaction_codes.tolist())
    }
    input_dim = int(preregistration["source_fingerprint_bits"]) + int(
        preregistration["condition_dim"]
    )
    model = ConditionalTransactionCodeFlow(
        input_dim=input_dim,
        hidden_dim=int(preregistration["hidden_dim"]),
        codebook_size=int(preregistration["codebook_size"]),
        latent_dim=int(preregistration["latent_dim"]),
    ).to(device)
    history = train_code_flow(
        model, training_examples, key_to_code, preregistration, device
    )
    checkpoint_path = args.output_dir / "compositional_closed_transaction_vq_flow.pt"
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
            "vq_centroids": torch.from_numpy(centroids),
            "transaction_catalog": transaction_catalog,
            "codebook_manifest": codebook_manifest,
        },
        checkpoint_path,
    )
    frozen, development_support = freeze_candidates(
        model,
        development_pairs,
        transactions,
        transaction_codes.tolist(),
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_transactions.csv"
    write_rows(frozen_path, frozen)
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = b43.b41.b38.evaluate_frozen_candidates(
        frozen, development_pairs
    )
    evaluated_path = args.output_dir / "evaluated_train_only_dev_transactions.csv"
    write_rows(evaluated_path, evaluated)
    gate = gate_result(
        metrics,
        codebook_manifest,
        development_support,
        dict(preregistration["atomic_baseline"]),
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
        "codebook": codebook_manifest,
        "development_support": development_support,
        "fit_only_closed_reaction_grammar": True,
        "connected_fit_targets_only": True,
        "reaction_context_radius": 0,
        "deterministic_vq_codebook": True,
        "source_conditioned_code_flow": True,
        "orthogonal_latent_particles": True,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
        "support_is_decoder_action_space": True,
        "only_sampled_transactions_committed": True,
        "frozen_before_target_or_property_evaluation": True,
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
            "advance_compositional_vq_to_once_only_fresh_confirmation"
            if gate["passed"]
            else "stop_and_diagnose_vq_support_or_code_flow_without_gate_changes"
        ),
        "training": history,
        "atomic_baseline": dict(preregistration["atomic_baseline"]),
        "b43_baseline": dict(preregistration["b43_baseline"]),
        "internal_gate": gate,
        "metrics": metrics,
        "manifest": manifest,
        "checkpoint": str(checkpoint_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
