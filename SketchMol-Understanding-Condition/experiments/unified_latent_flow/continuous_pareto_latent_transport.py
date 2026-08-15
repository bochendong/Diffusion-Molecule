#!/usr/bin/env python3
"""Train a continuous source-anchored Pareto energy over latent edit states.

B34 replaces B33's four hand-discretized property/structure mixtures with one
train-only conditional energy.  Its preference input specifies a continuously
interpolated source-similarity requirement from 0.15 to 0.65.  The model learns
the feasibility margin of each latent ``(attachment site, fragment token)``
state using source-disjoint B31/B32 train labels.  At generation time twenty
preregistered preferences each produce one direct categorical draw.  Every
latent state is frozen before any molecule is assembled; there is no molecule
pool, candidate ranking, oracle selection, retry, or evaluation-target access.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pareto_conditioned_joint_latent as b33  # noqa: E402


b32 = b33.b32
b31 = b33.b31
b27 = b33.b27
kernel = b33.kernel
base = b33.base
belief = b33.belief
pinned = b33.pinned
PROTOCOL = "train_only_continuous_pareto_latent_transport_v34"


@dataclass(frozen=True)
class TransportLabel:
    condition_index: int
    task: str
    assay_margin: float
    assay_logit: float
    structure_margin: float
    structure_logit: float
    property_margin: float
    property_success: float
    source_tanimoto: float


class ContinuousParetoTransport(torch.nn.Module):
    """Small residual energy field over frozen site-token latent states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.margin_head = torch.nn.Linear(hidden_dim, 1)
        self.feasible_head = torch.nn.Linear(hidden_dim, 1)

    @staticmethod
    def features(
        assay_margin: torch.Tensor,
        assay_logit: torch.Tensor,
        structure_margin: torch.Tensor,
        structure_logit: torch.Tensor,
        preference: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                assay_margin,
                torch.tanh(assay_logit),
                structure_margin,
                torch.tanh(structure_logit),
                preference,
                preference.square(),
                preference * structure_margin,
                (1.0 - preference) * assay_margin,
            ],
            dim=-1,
        )

    def forward(
        self,
        assay_margin: torch.Tensor,
        assay_logit: torch.Tensor,
        structure_margin: torch.Tensor,
        structure_logit: torch.Tensor,
        preference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network(
            self.features(
                assay_margin,
                assay_logit,
                structure_margin,
                structure_logit,
                preference,
            )
        )
        return self.margin_head(hidden).squeeze(-1), self.feasible_head(hidden).squeeze(-1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--b31-checkpoint", type=Path, required=True)
    parser.add_argument("--b31-summary", type=Path, required=True)
    parser.add_argument("--b32-checkpoint", type=Path, required=True)
    parser.add_argument("--b32-summary", type=Path, required=True)
    parser.add_argument("--gsk3b-oracle", type=Path, required=True)
    parser.add_argument("--drd2-oracle", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "b31_protocol": b31.PROTOCOL,
        "b32_protocol": b32.PROTOCOL,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "evaluation_target_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "one_joint_latent_state_one_raw_molecule": True,
        "exact_raw_attempts_per_condition": 20,
        "train_source_limit": 512,
        "actions_per_condition": 96,
        "continuous_similarity_min": 0.15,
        "continuous_similarity_max": 0.65,
        "generation_preference_min": 0.1,
        "generation_preference_max": 1.0,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B34 preregistration drift: {drift}")
    if payload.get("tasks") != list(b31.TASK_SPECS):
        raise ValueError("B34 task order drift")
    if set(dict(payload.get("oracles", {}))) != {"GSK3B", "DRD2"}:
        raise ValueError("B34 pinned oracle contract drift")
    preferences = list(payload.get("preference_calibration_values", []))
    if preferences != [0.1, 0.4, 0.7, 1.0]:
        raise ValueError("B34 preference calibration grid drift")
    return payload


def continuous_target(
    property_margin: torch.Tensor,
    property_success: torch.Tensor,
    source_tanimoto: torch.Tensor,
    preference: torch.Tensor,
    preregistration: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    low = float(preregistration["continuous_similarity_min"])
    high = float(preregistration["continuous_similarity_max"])
    threshold = low + (high - low) * preference
    structure_margin = (source_tanimoto - threshold) / max(
        float(preregistration["structure_similarity_scale"]), 1e-8
    )
    margin = torch.minimum(property_margin, structure_margin).clamp(-2.0, 2.0)
    feasible = property_success * (source_tanimoto >= threshold).float()
    return margin, feasible


@torch.no_grad()
def build_transport_labels(
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
    property_labels: Sequence[b31.JointLabel],
    structure_labels: Sequence[b32.StructureLabel],
    *,
    batch_size: int,
    device: torch.device,
) -> list[TransportLabel]:
    if len(property_labels) != len(structure_labels):
        raise ValueError("B34 property/structure label count mismatch")
    output: list[TransportLabel] = []
    assay_model.eval()
    structure_model.eval()
    for start in range(0, len(property_labels), int(batch_size)):
        property_items = property_labels[start : start + int(batch_size)]
        structure_items = structure_labels[start : start + int(batch_size)]
        for property_item, structure_item in zip(property_items, structure_items):
            if property_item.condition_index != structure_item.condition_index:
                raise ValueError("B34 aligned label condition drift")
        endpoint = torch.from_numpy(
            np.stack([row.endpoint for row in property_items])
        ).to(device)
        context = torch.from_numpy(
            np.stack([row.context for row in property_items])
        ).to(device)
        assay_margin, assay_logit = assay_model(endpoint, context)
        structure_margin, structure_logit = structure_model(endpoint, context)
        for index, (property_item, structure_item) in enumerate(
            zip(property_items, structure_items)
        ):
            output.append(
                TransportLabel(
                    condition_index=int(property_item.condition_index),
                    task=str(property_item.task),
                    assay_margin=float(assay_margin[index].cpu()),
                    assay_logit=float(assay_logit[index].cpu()),
                    structure_margin=float(structure_margin[index].cpu()),
                    structure_logit=float(structure_logit[index].cpu()),
                    property_margin=float(property_item.margin),
                    property_success=float(property_item.strict),
                    source_tanimoto=float(structure_item.similarity),
                )
            )
    return output


def label_tensors(
    labels: Sequence[TransportLabel], indices: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, ...]:
    def values(name: str) -> torch.Tensor:
        return torch.as_tensor(
            [float(getattr(labels[index], name)) for index in indices],
            dtype=torch.float32,
            device=device,
        )

    return (
        values("assay_margin"),
        values("assay_logit"),
        values("structure_margin"),
        values("structure_logit"),
        values("property_margin"),
        values("property_success"),
        values("source_tanimoto"),
    )


def training_positive_weight(
    labels: Sequence[TransportLabel],
    indices: Sequence[int],
    preregistration: Mapping[str, object],
) -> float:
    positives = 0
    examples = 0
    low = float(preregistration["continuous_similarity_min"])
    high = float(preregistration["continuous_similarity_max"])
    for preference in preregistration["preference_calibration_values"]:
        threshold = low + (high - low) * float(preference)
        positives += sum(
            labels[index].property_success > 0.5
            and labels[index].source_tanimoto >= threshold
            for index in indices
        )
        examples += len(indices)
    negatives = examples - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("B34 train labels do not contain both feasibility classes")
    return min(
        float(preregistration["maximum_positive_weight"]),
        max(1.0, negatives / positives),
    )


def train_transport(
    model: ContinuousParetoTransport,
    labels: Sequence[TransportLabel],
    fit_conditions: set[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    indices = [
        index for index, row in enumerate(labels) if row.condition_index in fit_conditions
    ]
    positive_weight = training_positive_weight(labels, indices, preregistration)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    shuffler = random.Random(int(preregistration["seed"]))
    history: list[dict[str, float]] = []
    for epoch in range(1, int(preregistration["training_epochs"]) + 1):
        order = list(indices)
        shuffler.shuffle(order)
        preference_generator = torch.Generator(device=device)
        preference_generator.manual_seed(int(preregistration["seed"]) + epoch)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(preregistration["batch_size"])):
            selected = order[start : start + int(preregistration["batch_size"])]
            (
                assay_margin,
                assay_logit,
                structure_margin,
                structure_logit,
                property_margin,
                property_success,
                source_tanimoto,
            ) = label_tensors(labels, selected, device)
            preference = torch.rand(
                len(selected), generator=preference_generator, device=device
            )
            margin_target, feasible_target = continuous_target(
                property_margin,
                property_success,
                source_tanimoto,
                preference,
                preregistration,
            )
            margin, feasible_logit = model(
                assay_margin,
                assay_logit,
                structure_margin,
                structure_logit,
                preference,
            )
            margin_loss = F.smooth_l1_loss(margin, margin_target)
            feasible_loss = F.binary_cross_entropy_with_logits(
                feasible_logit,
                feasible_target,
                pos_weight=torch.as_tensor(positive_weight, device=device),
            )
            loss = margin_loss + float(
                preregistration["feasible_loss_weight"]
            ) * feasible_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["margin_loss"] += float(margin_loss.detach())
            totals["feasible_loss"] += float(feasible_loss.detach())
            batches += 1
        row = {
            "epoch": float(epoch),
            "positive_weight": float(positive_weight),
            **{key: value / max(1, batches) for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = score[target > 0.5]
    negative = score[target <= 0.5]
    if not len(positive) or not len(negative):
        return 0.0
    combined = np.concatenate([positive, negative])
    ranks = b31.rank_values(combined)
    rank_sum = ranks[: len(positive)].sum()
    return float(
        (rank_sum - len(positive) * (len(positive) - 1) / 2)
        / (len(positive) * len(negative))
    )


@torch.no_grad()
def calibrate_transport(
    model: ContinuousParetoTransport,
    labels: Sequence[TransportLabel],
    dev_conditions: set[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    selected_indices = [
        index for index, row in enumerate(labels) if row.condition_index in dev_conditions
    ]
    preferences = [float(value) for value in preregistration["preference_calibration_values"]]
    predicted_by_preference: list[np.ndarray] = []
    all_target_margin: list[float] = []
    all_target_feasible: list[float] = []
    all_predicted_margin: list[float] = []
    all_predicted_logit: list[float] = []
    by_preference: dict[str, object] = {}
    model.eval()
    for preference_value in preferences:
        preference_targets: list[float] = []
        preference_feasible: list[float] = []
        preference_margin: list[float] = []
        preference_logits: list[float] = []
        for start in range(0, len(selected_indices), int(preregistration["batch_size"])):
            indices = selected_indices[start : start + int(preregistration["batch_size"])]
            (
                assay_margin,
                assay_logit,
                structure_margin,
                structure_logit,
                property_margin,
                property_success,
                source_tanimoto,
            ) = label_tensors(labels, indices, device)
            preference = torch.full_like(assay_margin, preference_value)
            target_margin, target_feasible = continuous_target(
                property_margin,
                property_success,
                source_tanimoto,
                preference,
                preregistration,
            )
            predicted_margin, predicted_logit = model(
                assay_margin,
                assay_logit,
                structure_margin,
                structure_logit,
                preference,
            )
            preference_targets.extend(target_margin.cpu().tolist())
            preference_feasible.extend(target_feasible.cpu().tolist())
            preference_margin.extend(predicted_margin.cpu().tolist())
            preference_logits.extend(predicted_logit.cpu().tolist())
        target_array = np.asarray(preference_targets, dtype=np.float64)
        feasible_array = np.asarray(preference_feasible, dtype=np.float64)
        margin_array = np.asarray(preference_margin, dtype=np.float64)
        logit_array = np.asarray(preference_logits, dtype=np.float64)
        predicted_by_preference.append(margin_array)
        by_preference[f"{preference_value:.2f}"] = {
            "examples": len(target_array),
            "feasible_rate": float(feasible_array.mean()),
            "feasible_auc": binary_auc(feasible_array, logit_array),
            "margin_mae": float(np.mean(np.abs(margin_array - target_array))),
        }
        all_target_margin.extend(target_array.tolist())
        all_target_feasible.extend(feasible_array.tolist())
        all_predicted_margin.extend(margin_array.tolist())
        all_predicted_logit.extend(logit_array.tolist())
    target = np.asarray(all_target_margin, dtype=np.float64)
    feasible = np.asarray(all_target_feasible, dtype=np.float64)
    predicted = np.asarray(all_predicted_margin, dtype=np.float64)
    logits = np.asarray(all_predicted_logit, dtype=np.float64)
    monotonic = np.stack(predicted_by_preference, axis=1)
    monotonic_fraction = float(np.mean(np.all(np.diff(monotonic, axis=1) <= 0.05, axis=1)))
    spearman = (
        float(np.corrcoef(b31.rank_values(target), b31.rank_values(predicted))[0, 1])
        if len(target) > 1
        else 0.0
    )
    return {
        "base_examples": len(selected_indices),
        "examples": len(target),
        "margin_mae": float(np.mean(np.abs(predicted - target))),
        "margin_spearman": spearman if math.isfinite(spearman) else 0.0,
        "continuous_feasible_auc": binary_auc(feasible, logits),
        "preference_monotonic_fraction": monotonic_fraction,
        "by_preference": by_preference,
    }


@torch.no_grad()
def continuous_transport_actions(
    fragment_model: kernel.FragmentAttachmentKernel,
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
    transport_model: ContinuousParetoTransport,
    condition: b31.b29.TransferCondition,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
    *,
    seed: int,
) -> list[dict[str, object]]:
    sites, contexts_np, site_logits_np = b31.condition_site_state(
        fragment_model, condition, source_latent, config, device
    )
    contexts = torch.from_numpy(contexts_np).to(device)
    vocabulary = torch.from_numpy(target_endpoints).to(device)
    assay_margin, assay_logit = b32.grid_predictions(
        assay_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    structure_margin, structure_logit = b32.grid_predictions(
        structure_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    site_logits = torch.from_numpy(site_logits_np).to(device)
    base_logits = site_logits[:, None] / max(float(config.site_temperature), 1e-6)
    base_logits = base_logits.expand_as(assay_margin).clone()
    token_lookup = {token: index for index, token in enumerate(target_fragments)}
    for site_index, site in enumerate(sites):
        current_index = token_lookup.get(site.variable)
        if current_index is not None:
            base_logits[site_index, current_index] = -torch.inf

    # Freeze one categorical latent state for each continuous preference before
    # constructing any product molecule.
    frozen_draws: list[tuple[float, int, float, float, float, float]] = []
    for preference_index, preference_value in enumerate(config.generation_preferences):
        preference = torch.full_like(assay_margin, float(preference_value))
        predicted_margin, predicted_logit = transport_model(
            assay_margin.reshape(-1),
            assay_logit.reshape(-1),
            structure_margin.reshape(-1),
            structure_logit.reshape(-1),
            preference.reshape(-1),
        )
        predicted_margin = predicted_margin.reshape_as(assay_margin)
        predicted_logit = predicted_logit.reshape_as(assay_margin)
        energy = predicted_margin + 0.25 * torch.tanh(predicted_logit)
        standardized = (energy - energy.mean()) / energy.std().clamp_min(
            float(config.energy_scale_floor)
        )
        logits = base_logits + float(config.transport_energy_weight) * standardized
        logits = logits + float(config.transport_feasibility_weight) * F.logsigmoid(
            predicted_logit
        )
        probabilities = torch.softmax(logits.reshape(-1).float(), dim=0)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 104729 * preference_index)
        selected = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        )
        flat_index = int(selected.item())
        entropy = float(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum().cpu()
        )
        threshold = float(config.continuous_similarity_min) + (
            float(config.continuous_similarity_max)
            - float(config.continuous_similarity_min)
        ) * float(preference_value)
        frozen_draws.append(
            (
                float(preference_value),
                flat_index,
                float(probabilities[flat_index].cpu()),
                entropy,
                float(predicted_margin.reshape(-1)[flat_index].cpu()),
                threshold,
            )
        )
    if len(frozen_draws) != int(config.num_attempts):
        raise RuntimeError("B34 did not freeze exactly 20 latent states")

    output: list[dict[str, object]] = []
    for preference, flat_index, probability, entropy, predicted_margin, threshold in frozen_draws:
        site_index, token_index = divmod(flat_index, len(target_fragments))
        site = sites[site_index]
        token = target_fragments[token_index]
        output.append(
            {
                "smiles": kernel.fragments.join_fragments(site.core, token),
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": token,
                "joint_energy": predicted_margin,
                "joint_probability": probability,
                "joint_distribution_entropy": entropy,
                "continuous_preference": preference,
                "requested_similarity_threshold": threshold,
            }
        )
    return output


def freeze_methods(
    fragment_model: kernel.FragmentAttachmentKernel,
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
    transport_model: ContinuousParetoTransport,
    conditions: Sequence[b31.b29.TransferCondition],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> dict[str, list[tuple[b31.b29.TransferCondition, list[dict[str, object]]]]]:
    frozen = {"b31_property": [], "b32_structure": [], "b34_continuous": []}
    for index, condition in enumerate(conditions):
        seed = int(config.seed) * 100000 + index
        b31_actions = b31.joint_actions(
            fragment_model,
            assay_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
            energy_weight=float(config.assay_energy_weight),
        )
        b32_actions = b32.constrained_joint_actions(
            fragment_model,
            assay_model,
            structure_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        b34_actions = continuous_transport_actions(
            fragment_model,
            assay_model,
            structure_model,
            transport_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        for name, actions in (
            ("b31_property", b31_actions),
            ("b32_structure", b32_actions),
            ("b34_continuous", b34_actions),
        ):
            if len(actions) != int(config.num_attempts):
                raise RuntimeError(f"B34 {name} did not freeze exactly 20 attempts")
            frozen[name].append((condition, actions))
    return frozen


def preference_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        preference = float(row["continuous_preference"])
        group = min(3, int(preference * 4.0))
        grouped[f"quartile_{group + 1}"].append(row)
    output: dict[str, object] = {}
    for name, items in sorted(grouped.items()):
        valid = [row for row in items if bool(row["valid"])]
        output[name] = {
            "rows": len(items),
            "preference_min": min(float(row["continuous_preference"]) for row in items),
            "preference_max": max(float(row["continuous_preference"]) for row in items),
            "validity": len(valid) / max(1, len(items)),
            "success_t0_15_per_attempt": sum(bool(row["success_t0_15"]) for row in items)
            / max(1, len(items)),
            "success_t0_65_per_attempt": sum(bool(row["success_t0_65"]) for row in items)
            / max(1, len(items)),
            "mean_source_tanimoto": float(
                np.mean([float(row["source_tanimoto"]) for row in valid])
            )
            if valid
            else 0.0,
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B34 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))

    representation, _representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        b27.load_frozen_fragment_model(args.fragment_checkpoint, device, preregistration)
    )
    assay_model, b31_summary = b32.load_b31_energy(
        args.b31_checkpoint, args.b31_summary, preregistration, device
    )
    structure_model, b32_summary = b33.load_b32_structure_energy(
        args.b32_checkpoint, args.b32_summary, preregistration, device
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
        (args.fragment_checkpoint, "fragment_checkpoint_sha256"),
    ):
        if dict(b32_summary["manifest"]).get(key) != belief.file_sha256(path):
            raise ValueError(f"B34 frozen B32 input drift: {key}")

    assay_oracles, oracle_provenance = pinned.load_pinned_oracles(
        gsk3b_path=args.gsk3b_oracle,
        drd2_path=args.drd2_oracle,
        specifications=dict(preregistration["oracles"]),
    )
    train_pairs, reconstruction = b27.reconstruct_b24_train_pairs(args, preregistration)
    action_config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
    )
    source_pairs, source_selection = b31.select_train_sources(
        train_pairs,
        limit=int(preregistration["train_source_limit"]),
        seed=int(preregistration["source_selection_seed"]),
        site_config=action_config,
    )
    fit_sources, dev_sources = b31.source_split(
        source_pairs,
        dev_fraction=float(preregistration["dev_fraction"]),
        seed=int(preregistration["split_seed"]),
    )
    conditions = b31.build_conditions(
        source_pairs, condition_dim=int(preregistration["condition_dim"])
    )
    source_latents_unique = kernel.encode_sources(
        representation,
        source_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    source_index = {pair.source_smiles: index for index, pair in enumerate(source_pairs)}
    condition_latents = np.stack(
        [source_latents_unique[source_index[row.source_smiles]] for row in conditions]
    ).astype(np.float32)
    site_contexts: list[np.ndarray] = []
    candidates: list[b31.CandidateAction] = []
    for condition_index, condition in enumerate(conditions):
        sites, contexts, _site_logits = b31.condition_site_state(
            fragment_model,
            condition,
            condition_latents[condition_index],
            action_config,
            device,
        )
        site_contexts.append(contexts)
        candidates.extend(
            b31.sample_candidate_actions(
                condition,
                condition_index,
                sites,
                target_fragments,
                limit=int(preregistration["actions_per_condition"]),
                seed=int(preregistration["label_sampling_seed"]),
            )
        )
        if (condition_index + 1) % 128 == 0:
            print(
                json.dumps(
                    {
                        "stage": "continuous_transport_labels",
                        "conditions": condition_index + 1,
                        "candidate_actions": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    property_labels, property_support = b31.build_labels(
        conditions,
        site_contexts,
        candidates,
        target_endpoints,
        assay_oracles,
        preregistration,
    )
    structure_labels, structure_support = b32.build_structure_labels(
        site_contexts, candidates, target_endpoints, preregistration
    )
    transport_labels = build_transport_labels(
        assay_model,
        structure_model,
        property_labels,
        structure_labels,
        batch_size=int(preregistration["batch_size"]),
        device=device,
    )
    fit_conditions = {
        index for index, row in enumerate(conditions) if row.source_smiles in fit_sources
    }
    dev_conditions = {
        index for index, row in enumerate(conditions) if row.source_smiles in dev_sources
    }
    transport_model = ContinuousParetoTransport(
        hidden_dim=int(preregistration["hidden_dim"])
    ).to(device)
    history = train_transport(
        transport_model, transport_labels, fit_conditions, preregistration, device
    )
    calibration = calibrate_transport(
        transport_model, transport_labels, dev_conditions, preregistration, device
    )

    dev_source_pairs = sorted(
        [pair for pair in source_pairs if pair.source_smiles in dev_sources],
        key=lambda pair: b27.stable_value(
            int(preregistration["generation_seed"]), "dev", pair.source_smiles
        ),
    )[: int(preregistration["internal_dev_source_limit"])]
    dev_generation_conditions = b31.build_conditions(
        dev_source_pairs, condition_dim=int(preregistration["condition_dim"])
    )
    dev_latents_unique = kernel.encode_sources(
        representation,
        dev_source_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    dev_latent_lookup = {
        pair.source_smiles: dev_latents_unique[index]
        for index, pair in enumerate(dev_source_pairs)
    }
    dev_latents = np.stack(
        [dev_latent_lookup[row.source_smiles] for row in dev_generation_conditions]
    ).astype(np.float32)
    generation_preferences = np.linspace(
        float(preregistration["generation_preference_min"]),
        float(preregistration["generation_preference_max"]),
        int(preregistration["exact_raw_attempts_per_condition"]),
        dtype=np.float32,
    ).tolist()
    generation_config = SimpleNamespace(
        **vars(action_config),
        num_attempts=int(preregistration["exact_raw_attempts_per_condition"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        energy_chunk_size=int(preregistration["energy_chunk_size"]),
        energy_scale_floor=float(preregistration["energy_scale_floor"]),
        assay_energy_weight=float(preregistration["assay_energy_weight"]),
        structure_dual_weight=float(preregistration["structure_dual_weight"]),
        structure_feasibility_weight=float(
            preregistration["structure_feasibility_weight"]
        ),
        structure_logit_temperature=float(
            preregistration["structure_logit_temperature"]
        ),
        structure_similarity_threshold=float(
            preregistration["structure_similarity_threshold"]
        ),
        structure_similarity_scale=float(preregistration["structure_similarity_scale"]),
        continuous_similarity_min=float(preregistration["continuous_similarity_min"]),
        continuous_similarity_max=float(preregistration["continuous_similarity_max"]),
        transport_energy_weight=float(preregistration["transport_energy_weight"]),
        transport_feasibility_weight=float(
            preregistration["transport_feasibility_weight"]
        ),
        generation_preferences=generation_preferences,
        seed=int(preregistration["generation_seed"]),
    )
    frozen = freeze_methods(
        fragment_model,
        assay_model,
        structure_model,
        transport_model,
        dev_generation_conditions,
        dev_latents,
        target_fragments,
        target_endpoints,
        generation_config,
        device,
    )
    rows_by_method: dict[str, list[dict[str, object]]] = {}
    metrics_by_method: dict[str, dict[str, object]] = {}
    for name, values in frozen.items():
        rows, metrics = b31.evaluate_frozen(values, assay_oracles, preregistration)
        if name == "b34_continuous":
            for row in rows:
                preference = generation_preferences[int(row["attempt"]) - 1]
                row["continuous_preference"] = float(preference)
                row["requested_similarity_threshold"] = float(
                    preregistration["continuous_similarity_min"]
                ) + (
                    float(preregistration["continuous_similarity_max"])
                    - float(preregistration["continuous_similarity_min"])
                ) * float(preference)
        rows_by_method[name] = rows
        metrics_by_method[name] = metrics

    candidate = metrics_by_method["b34_continuous"]
    baseline_metrics = metrics_by_method["b32_structure"]
    gates = dict(preregistration["gates"])
    checks: dict[str, dict[str, object]] = {
        "minimum_labels": {
            "value": len(transport_labels),
            "threshold": gates["minimum_labels"],
        },
        "property_label_coverage": {
            "value": property_support["oracle_coverage"],
            "threshold": 1.0,
        },
        "structure_label_coverage": {
            "value": structure_support["structure_label_coverage"],
            "threshold": 1.0,
        },
        "fit_dev_source_overlap": {
            "value": len(fit_sources & dev_sources),
            "threshold": 0,
        },
        "calibration_examples": {
            "value": calibration["examples"],
            "threshold": gates["minimum_dev_examples"],
        },
        "margin_spearman": {
            "value": calibration["margin_spearman"],
            "threshold": gates["margin_spearman"],
        },
        "continuous_feasible_auc": {
            "value": calibration["continuous_feasible_auc"],
            "threshold": gates["continuous_feasible_auc"],
        },
        "exact_attempts": {
            "value": candidate["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": candidate["validity"], "threshold": gates["validity"]},
        "oracle_coverage": {"value": candidate["oracle_coverage"], "threshold": 1.0},
        "mean_unique_valid": {
            "value": candidate["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": candidate["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "overall_any20_t0_15": {
            "value": candidate["acc_any20_t0_15"],
            "threshold": gates["overall_any20_t0_15"],
        },
        "overall_any20_t0_65": {
            "value": candidate["acc_any20_t0_65"],
            "threshold": gates["overall_any20_t0_65"],
        },
        "wide_any20_delta_vs_b32": {
            "value": candidate["acc_any20_t0_15"]
            - baseline_metrics["acc_any20_t0_15"],
            "threshold": gates["wide_any20_delta_vs_b32"],
        },
        "strict_any20_delta_vs_b32": {
            "value": candidate["acc_any20_t0_65"]
            - baseline_metrics["acc_any20_t0_65"],
            "threshold": gates["strict_any20_delta_vs_b32"],
        },
    }
    for task in b31.TASK_SPECS:
        task_metrics = dict(candidate["by_task"])[task]
        checks[f"{task}:any20_t0_15"] = {
            "value": task_metrics["acc_any20_t0_15"],
            "threshold": gates["minimum_task_any20_t0_15"],
        }
        checks[f"{task}:any20_t0_65"] = {
            "value": task_metrics["acc_any20_t0_65"],
            "threshold": gates["minimum_task_any20_t0_65"],
        }
    exact_checks = {
        "property_label_coverage",
        "structure_label_coverage",
        "fit_dev_source_overlap",
        "exact_attempts",
        "oracle_coverage",
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name in exact_checks
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    passed = not failures

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "continuous_pareto_transport.pt"
    run_manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "b31_checkpoint_sha256": belief.file_sha256(args.b31_checkpoint),
        "b31_summary_sha256": belief.file_sha256(args.b31_summary),
        "b32_checkpoint_sha256": belief.file_sha256(args.b32_checkpoint),
        "b32_summary_sha256": belief.file_sha256(args.b32_summary),
        "representation_protocol": representation_summary.get("protocol"),
        "fragment_protocol": frozen_manifest.get("protocol", kernel.PROTOCOL),
        "training_source_selection": source_selection,
        "training_sources": len(source_pairs),
        "fit_sources": len(fit_sources),
        "internal_dev_sources": len(dev_sources),
        "fit_internal_dev_source_overlap": len(fit_sources & dev_sources),
        "training_labels_from_b24_train_sources_only": True,
        "b31_assay_energy_frozen": True,
        "b32_structure_energy_frozen": True,
        "continuous_transport_only_trainable": True,
        "evaluation_target_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "train_only_property_oracle_training_access": True,
        "property_oracle_generation_access": False,
        "post_freeze_oracle_evaluation_access": True,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "one_joint_latent_state_one_raw_molecule": True,
        "failed_attachment_retry": False,
        "second_edit": False,
        "exact_raw_attempts_per_condition": 20,
        "generation_preferences": generation_preferences,
        "pinned_oracles": oracle_provenance,
        "reconstruction": reconstruction,
    }
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": transport_model.state_dict(),
            "model_config": {"hidden_dim": int(preregistration["hidden_dim"])},
            "history": history,
            "manifest": run_manifest,
        },
        checkpoint_path,
    )
    for name, rows in rows_by_method.items():
        b31.write_rows(args.output_dir / f"internal_dev_{name}_candidates.csv", rows)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": run_manifest,
        "b31_decision": b31_summary.get("decision"),
        "b32_decision": b32_summary.get("decision"),
        "label_support": {
            "transport_labels": len(transport_labels),
            "property": property_support,
            "structure": structure_support,
        },
        "training": history,
        "transport_calibration": calibration,
        "internal_dev": metrics_by_method,
        "preference_diagnostics": preference_diagnostics(
            rows_by_method["b34_continuous"]
        ),
        "internal_gate": {"passed": passed, "checks": checks, "failures": failures},
        "decision": (
            "advance_frozen_b34_to_once_only_new_fresh_confirmation"
            if passed
            else "stop_continuous_pareto_transport_after_single_internal_signal"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
