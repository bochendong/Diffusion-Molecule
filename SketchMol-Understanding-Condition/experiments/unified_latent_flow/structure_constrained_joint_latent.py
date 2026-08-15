#!/usr/bin/env python3
"""Learn a structure feasibility constraint over the frozen B31 joint latent.

B31 learned a strong train-only assay energy over ``(site, fragment token)``
but its permissive Sim>=0.15 label allowed aggressive edits.  B32 freezes that
assay energy and learns a separate train-only structure head from the same
source-disjoint B24 action support.  At inference a preregistered Lagrangian
penalty changes the categorical distribution over latent site-token states.
Exactly 20 states are sampled before any product molecule is constructed;
there is no molecule pool, ranking, oracle selection, retry, or target access.
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

import assay_joint_site_token_latent as b31  # noqa: E402


kernel = b31.kernel
b27 = b31.b27
b28 = b31.b28
base = b31.base
belief = b31.belief
pinned = b31.pinned
PROTOCOL = "train_only_structure_constrained_joint_latent_v32"


@dataclass(frozen=True)
class StructureLabel:
    condition_index: int
    context: np.ndarray
    endpoint: np.ndarray
    similarity: float
    similarity_margin: float
    feasible: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--b31-checkpoint", type=Path, required=True)
    parser.add_argument("--b31-summary", type=Path, required=True)
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
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "evaluation_target_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "one_joint_latent_state_one_raw_molecule": True,
        "exact_raw_attempts_per_condition": 20,
        "train_source_limit": 512,
        "actions_per_condition": 96,
        "structure_similarity_threshold": 0.65,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B32 preregistration drift: {drift}")
    if payload.get("tasks") != list(b31.TASK_SPECS):
        raise ValueError("B32 task order drift")
    if set(dict(payload.get("oracles", {}))) != {"GSK3B", "DRD2"}:
        raise ValueError("B32 pinned oracle contract drift")
    return payload


def load_b31_energy(
    checkpoint_path: Path,
    summary_path: Path,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[b27.LatentPropertyEnergy, dict[str, object]]:
    if belief.file_sha256(checkpoint_path) != preregistration["b31_checkpoint_sha256"]:
        raise ValueError("B32 frozen B31 checkpoint hash drift")
    if belief.file_sha256(summary_path) != preregistration["b31_summary_sha256"]:
        raise ValueError("B32 frozen B31 summary hash drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != b31.PROTOCOL:
        raise ValueError("B32 refuses a non-B31 assay energy")
    if not bool(dict(summary.get("internal_gate", {})).get("passed")):
        raise ValueError("B32 requires the passing B31 internal gate")
    manifest = dict(summary.get("manifest", {}))
    for key in (
        "generation_target_access",
        "molecular_candidate_ranking",
        "oracle_selection",
    ):
        if manifest.get(key) is not False:
            raise ValueError(f"B32 refuses B31 manifest drift: {key}")
    if manifest.get("exact_raw_attempts_per_condition") != 20:
        raise ValueError("B32 refuses a changed B31 sampling budget")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("stage") != b31.PROTOCOL:
        raise ValueError("B32 frozen B31 checkpoint protocol drift")
    config = dict(payload["model_config"])
    model = b27.LatentPropertyEnergy(
        endpoint_dim=int(config["endpoint_dim"]),
        context_dim=int(config["context_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, summary


def build_structure_labels(
    site_contexts: Sequence[np.ndarray],
    candidates: Sequence[b31.CandidateAction],
    target_endpoints: np.ndarray,
    preregistration: Mapping[str, object],
) -> tuple[list[StructureLabel], dict[str, object]]:
    threshold = float(preregistration["structure_similarity_threshold"])
    scale = float(preregistration["structure_similarity_scale"])
    labels: list[StructureLabel] = []
    positives: defaultdict[int, int] = defaultdict(int)
    for candidate in candidates:
        feasible = float(candidate.source_tanimoto >= threshold)
        positives[candidate.condition_index] += int(feasible)
        labels.append(
            StructureLabel(
                condition_index=candidate.condition_index,
                context=site_contexts[candidate.condition_index][candidate.site_index],
                endpoint=target_endpoints[candidate.token_index],
                similarity=float(candidate.source_tanimoto),
                similarity_margin=float(
                    np.clip((candidate.source_tanimoto - threshold) / scale, -3.0, 3.0)
                ),
                feasible=feasible,
            )
        )
    return labels, {
        "candidate_actions": len(candidates),
        "complete_structure_labels": len(labels),
        "structure_label_coverage": len(labels) / max(1, len(candidates)),
        "structure_positive_labels": int(sum(row.feasible for row in labels)),
        "structure_positive_rate": float(
            sum(row.feasible for row in labels) / max(1, len(labels))
        ),
        "conditions_with_structure_positive": sum(value > 0 for value in positives.values()),
    }


def train_structure_model(
    model: b27.LatentPropertyEnergy,
    labels: Sequence[StructureLabel],
    fit_conditions: set[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    indices = [
        index for index, row in enumerate(labels) if row.condition_index in fit_conditions
    ]
    positives = sum(labels[index].feasible > 0.5 for index in indices)
    negatives = len(indices) - positives
    if positives < int(preregistration["minimum_fit_structure_positives"]):
        raise ValueError(f"B32 has only {positives} fit structure positives")
    pos_weight = min(
        float(preregistration["maximum_positive_weight"]),
        max(1.0, negatives / max(1, positives)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    generator = random.Random(int(preregistration["seed"]))
    history: list[dict[str, float]] = []
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(indices)
        generator.shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(preregistration["batch_size"])):
            selected = order[start : start + int(preregistration["batch_size"])]
            endpoint = torch.from_numpy(
                np.stack([labels[index].endpoint for index in selected])
            ).to(device)
            context = torch.from_numpy(
                np.stack([labels[index].context for index in selected])
            ).to(device)
            margin_target = torch.as_tensor(
                [labels[index].similarity_margin for index in selected],
                dtype=torch.float32,
                device=device,
            )
            feasible_target = torch.as_tensor(
                [labels[index].feasible for index in selected],
                dtype=torch.float32,
                device=device,
            )
            margin, feasible_logit = model(endpoint, context)
            margin_loss = F.smooth_l1_loss(margin, margin_target)
            feasible_loss = F.binary_cross_entropy_with_logits(
                feasible_logit,
                feasible_target,
                pos_weight=torch.as_tensor(pos_weight, device=device),
            )
            loss = margin_loss + float(preregistration["feasible_loss_weight"]) * feasible_loss
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
            **{key: value / max(1, batches) for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


@torch.no_grad()
def structure_calibration(
    model: b27.LatentPropertyEnergy,
    labels: Sequence[StructureLabel],
    dev_conditions: set[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    selected = [row for row in labels if row.condition_index in dev_conditions]
    predicted_margin: list[float] = []
    logits: list[float] = []
    model.eval()
    batch_size = int(preregistration["batch_size"])
    for start in range(0, len(selected), batch_size):
        items = selected[start : start + batch_size]
        endpoint = torch.from_numpy(np.stack([row.endpoint for row in items])).to(device)
        context = torch.from_numpy(np.stack([row.context for row in items])).to(device)
        margin, feasible_logit = model(endpoint, context)
        predicted_margin.extend(margin.float().cpu().tolist())
        logits.extend(feasible_logit.float().cpu().tolist())
    actual = np.asarray([row.similarity for row in selected], dtype=np.float64)
    feasible = np.asarray([row.feasible for row in selected], dtype=np.float64)
    margin = np.asarray(predicted_margin, dtype=np.float64)
    score = np.asarray(logits, dtype=np.float64)
    predicted = float(preregistration["structure_similarity_threshold"]) + float(
        preregistration["structure_similarity_scale"]
    ) * margin
    positive_scores = score[feasible > 0.5]
    negative_scores = score[feasible <= 0.5]
    if len(positive_scores) and len(negative_scores):
        combined = np.concatenate([positive_scores, negative_scores])
        ranks = b31.rank_values(combined)
        auc = float(
            (
                ranks[: len(positive_scores)].sum()
                - len(positive_scores) * (len(positive_scores) - 1) / 2
            )
            / (len(positive_scores) * len(negative_scores))
        )
    else:
        auc = 0.0
    spearman = (
        float(np.corrcoef(b31.rank_values(actual), b31.rank_values(predicted))[0, 1])
        if len(actual) > 1
        else 0.0
    )
    return {
        "examples": len(selected),
        "structure_positive_rate": float(feasible.mean()) if len(feasible) else 0.0,
        "similarity_mae": float(np.mean(np.abs(predicted - actual))) if len(actual) else 1.0,
        "similarity_spearman": spearman if math.isfinite(spearman) else 0.0,
        "structure_feasible_auc": auc,
    }


@torch.no_grad()
def grid_predictions(
    model: b27.LatentPropertyEnergy,
    vocabulary: torch.Tensor,
    contexts: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    margin_values: list[torch.Tensor] = []
    logit_values: list[torch.Tensor] = []
    attempts = contexts.shape[0]
    for start in range(0, vocabulary.shape[0], int(chunk_size)):
        endpoints = vocabulary[start : start + int(chunk_size)]
        count = endpoints.shape[0]
        flat_endpoint = endpoints[None, :, :].expand(attempts, count, -1).reshape(
            attempts * count, -1
        )
        flat_context = contexts[:, None, :].expand(attempts, count, -1).reshape(
            attempts * count, -1
        )
        margin, logit = model(flat_endpoint, flat_context)
        margin_values.append(margin.reshape(attempts, count))
        logit_values.append(logit.reshape(attempts, count))
    return torch.cat(margin_values, dim=1), torch.cat(logit_values, dim=1)


@torch.no_grad()
def constrained_joint_actions(
    fragment_model: kernel.FragmentAttachmentKernel,
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
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
    assay_margin, assay_logit = grid_predictions(
        assay_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    structure_margin, structure_logit = grid_predictions(
        structure_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    assay_energy = assay_margin + 0.25 * torch.tanh(assay_logit)
    standardized_assay = (assay_energy - assay_energy.mean()) / assay_energy.std().clamp_min(
        float(config.energy_scale_floor)
    )
    structure_shortfall = torch.relu(-structure_margin)
    feasibility_log_probability = F.logsigmoid(
        structure_logit / max(float(config.structure_logit_temperature), 1e-6)
    )
    site_logits = torch.from_numpy(site_logits_np).to(device)
    logits = site_logits[:, None] / max(float(config.site_temperature), 1e-6)
    logits = logits.expand_as(assay_energy).clone()
    logits = logits + float(config.assay_energy_weight) * standardized_assay
    logits = logits - float(config.structure_dual_weight) * structure_shortfall
    logits = logits + float(config.structure_feasibility_weight) * feasibility_log_probability
    token_lookup = {token: index for index, token in enumerate(target_fragments)}
    for site_index, site in enumerate(sites):
        current_index = token_lookup.get(site.variable)
        if current_index is not None:
            logits[site_index, current_index] = -torch.inf
    probabilities = torch.softmax(logits.reshape(-1).float(), dim=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    selected = torch.multinomial(
        probabilities,
        int(config.num_attempts),
        replacement=True,
        generator=generator,
    )
    entropy = float(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum().cpu()
    )
    output: list[dict[str, object]] = []
    for flat_index in selected.tolist():
        site_index, token_index = divmod(int(flat_index), len(target_fragments))
        site = sites[site_index]
        token = target_fragments[token_index]
        predicted_similarity = float(config.structure_similarity_threshold) + float(
            config.structure_similarity_scale
        ) * float(structure_margin[site_index, token_index].cpu())
        output.append(
            {
                "smiles": kernel.fragments.join_fragments(site.core, token),
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": token,
                "joint_energy": float(assay_energy[site_index, token_index].cpu()),
                "joint_probability": float(probabilities[flat_index].cpu()),
                "joint_distribution_entropy": entropy,
                "predicted_source_tanimoto": predicted_similarity,
                "predicted_structure_feasible": float(
                    torch.sigmoid(structure_logit[site_index, token_index]).cpu()
                ),
            }
        )
    return output


def freeze_methods(
    fragment_model: kernel.FragmentAttachmentKernel,
    assay_model: b27.LatentPropertyEnergy,
    structure_model: b27.LatentPropertyEnergy,
    conditions: Sequence[b31.b29.TransferCondition],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> dict[str, list[tuple[b31.b29.TransferCondition, list[dict[str, object]]]]]:
    frozen = {"b31_learned_joint": [], "structure_constrained_joint": []}
    for index, condition in enumerate(conditions):
        seed = int(config.seed) * 100000 + index
        baseline = b31.joint_actions(
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
        constrained = constrained_joint_actions(
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
        for name, actions in (
            ("b31_learned_joint", baseline),
            ("structure_constrained_joint", constrained),
        ):
            if len(actions) != int(config.num_attempts):
                raise RuntimeError(f"B32 {name} did not freeze exactly 20 attempts")
            frozen[name].append((condition, actions))
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B32 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    representation, _representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        b27.load_frozen_fragment_model(args.fragment_checkpoint, device, preregistration)
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
    ):
        if frozen_manifest.get(key) != belief.file_sha256(path):
            raise ValueError(f"B32 frozen B24 input drift: {key}")
    assay_model, b31_summary = load_b31_energy(
        args.b31_checkpoint, args.b31_summary, preregistration, device
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
                        "stage": "structure_labels",
                        "conditions": condition_index + 1,
                        "candidate_actions": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    labels, label_support = build_structure_labels(
        site_contexts, candidates, target_endpoints, preregistration
    )
    fit_conditions = {
        index for index, row in enumerate(conditions) if row.source_smiles in fit_sources
    }
    dev_conditions = {
        index for index, row in enumerate(conditions) if row.source_smiles in dev_sources
    }
    structure_model = b27.LatentPropertyEnergy(
        endpoint_dim=int(preregistration["fingerprint_bits"]),
        context_dim=int(site_contexts[0].shape[1]),
        hidden_dim=int(preregistration["hidden_dim"]),
    ).to(device)
    history = train_structure_model(
        structure_model, labels, fit_conditions, preregistration, device
    )
    calibration = structure_calibration(
        structure_model, labels, dev_conditions, preregistration, device
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
        seed=int(preregistration["generation_seed"]),
    )
    frozen = freeze_methods(
        fragment_model,
        assay_model,
        structure_model,
        dev_generation_conditions,
        dev_latents,
        target_fragments,
        target_endpoints,
        generation_config,
        device,
    )
    assay_oracles, oracle_provenance = pinned.load_pinned_oracles(
        gsk3b_path=args.gsk3b_oracle,
        drd2_path=args.drd2_oracle,
        specifications=dict(preregistration["oracles"]),
    )
    dev_rows: dict[str, list[dict[str, object]]] = {}
    dev_metrics: dict[str, dict[str, object]] = {}
    for name, values in frozen.items():
        dev_rows[name], dev_metrics[name] = b31.evaluate_frozen(
            values, assay_oracles, preregistration
        )

    constrained = dev_metrics["structure_constrained_joint"]
    baseline_metrics = dev_metrics["b31_learned_joint"]
    gates = dict(preregistration["gates"])
    checks = {
        "minimum_labels": {
            "value": label_support["complete_structure_labels"],
            "threshold": gates["minimum_labels"],
        },
        "structure_label_coverage": {
            "value": label_support["structure_label_coverage"],
            "threshold": 1.0,
        },
        "minimum_structure_positives": {
            "value": label_support["structure_positive_labels"],
            "threshold": gates["minimum_structure_positives"],
        },
        "fit_dev_source_overlap": {
            "value": len(fit_sources & dev_sources),
            "threshold": 0,
        },
        "calibration_examples": {
            "value": calibration["examples"],
            "threshold": gates["minimum_dev_labels"],
        },
        "similarity_spearman": {
            "value": calibration["similarity_spearman"],
            "threshold": gates["similarity_spearman"],
        },
        "structure_feasible_auc": {
            "value": calibration["structure_feasible_auc"],
            "threshold": gates["structure_feasible_auc"],
        },
        "exact_attempts": {
            "value": constrained["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": constrained["validity"], "threshold": gates["validity"]},
        "mean_unique_valid": {
            "value": constrained["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": constrained["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "overall_any20_t0_15": {
            "value": constrained["acc_any20_t0_15"],
            "threshold": gates["overall_any20_t0_15"],
        },
        "overall_any20_t0_65": {
            "value": constrained["acc_any20_t0_65"],
            "threshold": gates["overall_any20_t0_65"],
        },
        "strict_any20_delta_vs_b31": {
            "value": constrained["acc_any20_t0_65"]
            - baseline_metrics["acc_any20_t0_65"],
            "threshold": gates["strict_any20_delta_vs_b31"],
        },
        "wide_any20_delta_vs_b31": {
            "value": constrained["acc_any20_t0_15"]
            - baseline_metrics["acc_any20_t0_15"],
            "threshold": gates["wide_any20_delta_vs_b31"],
        },
    }
    for task in b31.TASK_SPECS:
        task_metrics = dict(constrained["by_task"])[task]
        checks[f"{task}:any20_t0_15"] = {
            "value": task_metrics["acc_any20_t0_15"],
            "threshold": gates["minimum_task_any20_t0_15"],
        }
        checks[f"{task}:any20_t0_65"] = {
            "value": task_metrics["acc_any20_t0_65"],
            "threshold": gates["minimum_task_any20_t0_65"],
        }
    maximum_checks = {
        "similarity_mae": {
            "value": calibration["similarity_mae"],
            "threshold": gates["maximum_similarity_mae"],
        }
    }
    exact_checks = {"fit_dev_source_overlap", "exact_attempts"}
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name in exact_checks
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    failures.extend(
        name
        for name, item in maximum_checks.items()
        if float(item["value"]) > float(item["threshold"])
    )
    passed = not failures

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "structure_feasibility_energy.pt"
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
        "representation_protocol": representation_summary.get("protocol"),
        "training_source_selection": source_selection,
        "training_sources": len(source_pairs),
        "fit_sources": len(fit_sources),
        "internal_dev_sources": len(dev_sources),
        "fit_internal_dev_source_overlap": len(fit_sources & dev_sources),
        "training_labels_from_b24_train_sources_only": True,
        "b31_assay_energy_frozen": True,
        "structure_head_only_trainable": True,
        "evaluation_target_access": False,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_table1_access": False,
        "property_oracle_generation_access": False,
        "post_freeze_oracle_evaluation_access": True,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "one_joint_latent_state_one_raw_molecule": True,
        "failed_attachment_retry": False,
        "second_edit": False,
        "exact_raw_attempts_per_condition": 20,
        "pinned_oracles": oracle_provenance,
        "reconstruction": reconstruction,
    }
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": structure_model.state_dict(),
            "model_config": {
                "endpoint_dim": int(preregistration["fingerprint_bits"]),
                "context_dim": int(site_contexts[0].shape[1]),
                "hidden_dim": int(preregistration["hidden_dim"]),
            },
            "history": history,
            "manifest": run_manifest,
        },
        checkpoint_path,
    )
    for name, rows in dev_rows.items():
        b31.write_rows(args.output_dir / f"internal_dev_{name}_candidates.csv", rows)
    all_checks = dict(checks)
    all_checks.update(maximum_checks)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": run_manifest,
        "b31_decision": b31_summary.get("decision"),
        "label_support": label_support,
        "training": history,
        "structure_calibration": calibration,
        "internal_dev": dev_metrics,
        "internal_gate": {"passed": passed, "checks": all_checks, "failures": failures},
        "decision": (
            "advance_structure_constrained_joint_latent_to_fresh_table1_subset"
            if passed
            else "stop_structure_constrained_joint_latent_after_single_pilot"
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
