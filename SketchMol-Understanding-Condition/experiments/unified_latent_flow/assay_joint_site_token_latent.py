#!/usr/bin/env python3
"""Train a target-free assay-conditioned joint site-token latent policy.

B30-r1 established that the frozen B24 one-cut grammar contains abundant
GSK3B and DRD2/MW/SA support, but B29's factorized site then token sampler did
not place probability mass on it.  B31 therefore learns one energy over the
joint latent state ``(source, property request, attachment site, fragment
token)`` using only B24 training molecules and pinned benchmark oracles.

At inference the model forms one categorical distribution over the complete
site-token grid and samples exactly 20 states directly.  Molecules do not
exist until after those latent states are drawn: there is no molecule pool,
ranking, oracle selection, retry, finalizer, or target access.
"""

from __future__ import annotations

import argparse
import csv
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

import table1_energy_tilted_latent_transfer as b29  # noqa: E402
import audit_table1_assay_latent_action_support as b30  # noqa: E402
import pinned_table1_assay_oracles as pinned  # noqa: E402


b28 = b29.b28
b27 = b29.b27
kernel = b29.kernel
base = b29.base
belief = b29.belief
graph = b29.graph
unified = b29.unified

PROTOCOL = "train_only_assay_joint_site_token_latent_v31"
TASK_SPECS = {
    "GSK3B:increase": (("GSK3B", 1),),
    "DRD2:decrease+MW:decrease+SA:decrease": (
        ("DRD2", -1),
        ("MW", -1),
        ("SA", -1),
    ),
}


@dataclass(frozen=True)
class CandidateAction:
    condition_index: int
    site_index: int
    token_index: int
    generated_smiles: str
    source_tanimoto: float


@dataclass(frozen=True)
class JointLabel:
    condition_index: int
    task: str
    context: np.ndarray
    endpoint: np.ndarray
    margin: float
    strict: float
    generated_smiles: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--table1-eval-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--b29-summary", type=Path, required=True)
    parser.add_argument("--b30-summary", type=Path, required=True)
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
        "frozen_fragment_protocol": kernel.PROTOCOL,
        "support_protocol": b30.PROTOCOL,
        "b29_protocol": b29.PROTOCOL,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_target_access": False,
        "evaluation_source_training_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "exact_raw_attempts_per_condition": 20,
        "train_source_limit": 512,
        "actions_per_condition": 96,
        "energy_weight": 2.0,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"B31 preregistration drift: {drift}")
    if payload.get("tasks") != list(TASK_SPECS):
        raise ValueError("B31 task order drift")
    if set(dict(payload.get("oracles", {}))) != {"GSK3B", "DRD2"}:
        raise ValueError("B31 pinned oracle contract drift")
    return payload


def read_evidence_contracts(
    b29_path: Path, b30_path: Path, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    b29_summary = json.loads(b29_path.read_text(encoding="utf-8"))
    if b29_summary.get("protocol") != preregistration["b29_protocol"]:
        raise ValueError("B31 refuses a non-B29 baseline")
    b29_manifest = dict(b29_summary.get("manifest", {}))
    if b29_manifest.get("generation_target_access") is not False:
        raise ValueError("B31 refuses a target-accessed B29 baseline")
    if b29_manifest.get("exact_raw_attempts_per_condition") != 20:
        raise ValueError("B31 B29 candidate budget drift")

    b30_summary = json.loads(b30_path.read_text(encoding="utf-8"))
    if b30_summary.get("protocol") != preregistration["support_protocol"]:
        raise ValueError("B31 refuses a non-B30-r1 support result")
    if not bool(dict(b30_summary.get("gate", {})).get("passed")):
        raise ValueError("B31 requires the passing B30-r1 support gate")
    if b30_summary.get("decision") != "train_property_conditioned_joint_site_token_latent":
        raise ValueError("B31 does not match the B30-r1 decision")
    support_manifest = dict(b30_summary.get("manifest", {}))
    if support_manifest.get("oracle_preflight_passed") is not True:
        raise ValueError("B31 requires B30-r1 oracle preflight")
    if support_manifest.get("generation_target_access") is not False:
        raise ValueError("B31 refuses target-accessed B30-r1 evidence")
    return b29_summary, b30_summary


def pseudo_row(source: str, task: str, condition_id: str) -> dict[str, str]:
    specs = TASK_SPECS[task]
    direction_name = {1: "increase", -1: "decrease"}
    tasks = [
        {"property": prop, "direction": direction_name[direction]}
        for prop, direction in specs
    ]
    row = {
        "condition_id": condition_id,
        "sample_id": condition_id,
        "task_type": "edit_generation",
        "source_smiles": source,
        "instruction": "train-only assay pseudo-condition",
        "condition_properties": ",".join(prop for prop, _direction in specs),
        "instruction_tasks": json.dumps(
            tasks, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ),
    }
    for prop, direction in specs:
        row[f"{prop}_active"] = "True"
        row[f"{prop}_direction"] = direction_name[direction]
    return row


def select_train_sources(
    train_pairs: Sequence[object],
    *,
    limit: int,
    seed: int,
    site_config: SimpleNamespace,
) -> tuple[list[object], dict[str, int]]:
    by_source: dict[str, object] = {}
    for pair in train_pairs:
        by_source.setdefault(str(pair.source_smiles), pair)
    eligible = [
        pair
        for pair in by_source.values()
        if kernel.source_sites(str(pair.source_smiles), site_config)
    ]
    ordered = sorted(
        eligible,
        key=lambda pair: b27.stable_value(seed, "source", pair.source_smiles),
    )
    if len(ordered) < int(limit):
        raise ValueError(
            "B31 has only "
            f"{len(ordered)} attachment-site-eligible unique train sources"
        )
    selected = ordered[: int(limit)]
    return selected, {
        "unique_reconstructed_sources": len(by_source),
        "attachment_site_eligible_sources": len(eligible),
        "sources_without_attachment_site": len(by_source) - len(eligible),
        "selected_sources": len(selected),
    }


def build_conditions(
    source_pairs: Sequence[object], *, condition_dim: int
) -> list[b29.TransferCondition]:
    output: list[b29.TransferCondition] = []
    for source_index, pair in enumerate(source_pairs):
        for task_index, task in enumerate(TASK_SPECS):
            condition_id = f"train_{source_index:04d}_{task_index}"
            row = pseudo_row(pair.source_smiles, task, condition_id)
            output.append(
                b29.TransferCondition(
                    row=row,
                    source_smiles=pair.source_smiles,
                    source=pair.source,
                    condition=kernel.hierarchical.property_latent_slot_tokens(
                        row, int(condition_dim)
                    ),
                    property_count=len(TASK_SPECS[task]),
                    task=task,
                    condition_id=condition_id,
                )
            )
    return output


def source_split(
    source_pairs: Sequence[object], *, dev_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    ordered = sorted(
        (str(pair.source_smiles) for pair in source_pairs),
        key=lambda source: b27.stable_value(seed, "split", source),
    )
    dev_count = max(1, int(round(len(ordered) * float(dev_fraction))))
    dev_count = min(dev_count, len(ordered) - 1)
    dev = set(ordered[:dev_count])
    fit = set(ordered[dev_count:])
    if fit & dev or not fit or not dev:
        raise ValueError("B31 fit/dev source split failed")
    return fit, dev


@torch.no_grad()
def condition_site_state(
    fragment_model: kernel.FragmentAttachmentKernel,
    condition: b29.TransferCondition,
    source_latent: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> tuple[list[kernel.Site], np.ndarray, np.ndarray]:
    sites = kernel.source_sites(condition.source_smiles, config)
    if not sites:
        raise ValueError(f"B31 train source has no attachment site: {condition.condition_id}")
    source = torch.from_numpy(source_latent[None, :]).to(device)
    property_tokens = torch.from_numpy(condition.condition[None, ...]).to(device)
    site_features = torch.from_numpy(
        np.stack([site.feature for site in sites]).astype(np.float32)[None, ...]
    ).to(device)
    mask = torch.ones(1, len(sites), dtype=torch.bool, device=device)
    request = fragment_model.request_context(source, property_tokens)
    logits, hidden = fragment_model.site_logits(request, site_features, mask)
    contexts = fragment_model.context_for_site(
        request.expand(len(sites), -1), hidden[0]
    )
    return (
        sites,
        contexts.float().cpu().numpy().astype(np.float32),
        logits[0].float().cpu().numpy().astype(np.float32),
    )


def sample_candidate_actions(
    condition: b29.TransferCondition,
    condition_index: int,
    sites: Sequence[kernel.Site],
    target_fragments: Sequence[str],
    *,
    limit: int,
    seed: int,
) -> list[CandidateAction]:
    total = len(sites) * len(target_fragments)
    generator = random.Random(b27.stable_value(seed, condition.condition_id))
    selected: set[int] = set()
    output: list[CandidateAction] = []
    probes = 0
    max_probes = min(total, int(limit) * 30)
    while len(output) < min(int(limit), total) and probes < max_probes:
        flat_index = generator.randrange(total)
        probes += 1
        if flat_index in selected:
            continue
        selected.add(flat_index)
        site_index, token_index = divmod(flat_index, len(target_fragments))
        site = sites[site_index]
        token = target_fragments[token_index]
        if token == site.variable:
            continue
        product = graph.canonical_smiles(
            kernel.fragments.join_fragments(site.core, token)
        )
        if not product or product == condition.source_smiles:
            continue
        similarity = graph.morgan_tanimoto(condition.source_smiles, product)
        if similarity is None:
            continue
        output.append(
            CandidateAction(
                condition_index=condition_index,
                site_index=site_index,
                token_index=token_index,
                generated_smiles=product,
                source_tanimoto=float(similarity),
            )
        )
    if len(output) != int(limit):
        raise ValueError(
            f"B31 sampled only {len(output)}/{limit} actions for {condition.condition_id}"
        )
    return output


def descriptor_value(smiles: str, prop: str, cache: dict[tuple[str, str], float]) -> float:
    key = (smiles, prop)
    if key not in cache:
        value = unified.score_property(smiles, prop)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"B31 missing descriptor {prop} for {smiles}")
        cache[key] = float(value)
    return cache[key]


def build_labels(
    conditions: Sequence[b29.TransferCondition],
    site_contexts: Sequence[np.ndarray],
    candidates: Sequence[CandidateAction],
    target_endpoints: np.ndarray,
    assay_oracles: Mapping[str, pinned.PinnedAssayOracle],
    preregistration: Mapping[str, object],
) -> tuple[list[JointLabel], dict[str, object]]:
    by_task: defaultdict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_task[conditions[candidate.condition_index].task].append(index)
    source_scores: dict[tuple[int, str], float] = {}
    candidate_scores: dict[int, float] = {}
    for task, indices in by_task.items():
        assay_prop = TASK_SPECS[task][0][0]
        condition_indices = sorted({candidates[index].condition_index for index in indices})
        source_values = assay_oracles[assay_prop].score_many(
            [conditions[index].source_smiles for index in condition_indices],
            batch_size=int(preregistration["oracle_batch_size"]),
        )
        for condition_index, value in zip(condition_indices, source_values):
            source_scores[(condition_index, assay_prop)] = value
        values = assay_oracles[assay_prop].score_many(
            [candidates[index].generated_smiles for index in indices],
            batch_size=int(preregistration["oracle_batch_size"]),
        )
        for candidate_index, value in zip(indices, values):
            candidate_scores[candidate_index] = value

    scales = dict(preregistration["label_scales"])
    similarity_threshold = float(preregistration["similarity_threshold"])
    descriptor_cache: dict[tuple[str, str], float] = {}
    labels: list[JointLabel] = []
    condition_positive: defaultdict[int, int] = defaultdict(int)
    task_positive: defaultdict[str, int] = defaultdict(int)
    for candidate_index, candidate in enumerate(candidates):
        condition = conditions[candidate.condition_index]
        specs = TASK_SPECS[condition.task]
        margins: list[float] = []
        for prop, direction in specs:
            if prop in assay_oracles:
                source_value = source_scores[(candidate.condition_index, prop)]
                candidate_value = candidate_scores[candidate_index]
            else:
                source_value = descriptor_value(
                    condition.source_smiles, prop, descriptor_cache
                )
                candidate_value = descriptor_value(
                    candidate.generated_smiles, prop, descriptor_cache
                )
            margins.append(
                float(direction)
                * (float(candidate_value) - float(source_value))
                / max(float(scales[prop]), 1e-8)
            )
        similarity_margin = (
            candidate.source_tanimoto - similarity_threshold
        ) / max(float(preregistration["similarity_scale"]), 1e-8)
        strict = float(
            candidate.source_tanimoto >= similarity_threshold
            and all(value > 0.0 for value in margins)
        )
        minimum_margin = float(
            np.clip(min([*margins, similarity_margin]), -2.0, 2.0)
        )
        condition_positive[candidate.condition_index] += int(strict > 0.5)
        task_positive[condition.task] += int(strict > 0.5)
        labels.append(
            JointLabel(
                condition_index=candidate.condition_index,
                task=condition.task,
                context=site_contexts[candidate.condition_index][candidate.site_index],
                endpoint=target_endpoints[candidate.token_index],
                margin=minimum_margin,
                strict=strict,
                generated_smiles=candidate.generated_smiles,
            )
        )
    supported_conditions = {
        task: sum(
            condition_positive[index] > 0
            for index, condition in enumerate(conditions)
            if condition.task == task
        )
        for task in TASK_SPECS
    }
    return labels, {
        "candidate_actions": len(candidates),
        "complete_labels": len(labels),
        "oracle_coverage": len(labels) / max(1, len(candidates)),
        "strict_positive_rate": sum(row.strict for row in labels) / max(1, len(labels)),
        "strict_positive_labels_by_task": dict(task_positive),
        "supported_conditions_by_task": supported_conditions,
    }


def train_energy(
    model: b27.LatentPropertyEnergy,
    labels: Sequence[JointLabel],
    fit_conditions: set[int],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    indices = [
        index for index, row in enumerate(labels) if row.condition_index in fit_conditions
    ]
    positives = sum(labels[index].strict > 0.5 for index in indices)
    negatives = len(indices) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("B31 fit labels do not contain both classes")
    pos_weight = min(10.0, max(1.0, negatives / positives))
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
            batch_indices = order[start : start + int(preregistration["batch_size"])]
            endpoint = torch.from_numpy(
                np.stack([labels[index].endpoint for index in batch_indices])
            ).to(device)
            context = torch.from_numpy(
                np.stack([labels[index].context for index in batch_indices])
            ).to(device)
            margin_target = torch.as_tensor(
                [labels[index].margin for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            strict_target = torch.as_tensor(
                [labels[index].strict for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            margin, strict_logit = model(endpoint, context)
            margin_loss = F.smooth_l1_loss(margin, margin_target)
            strict_loss = F.binary_cross_entropy_with_logits(
                strict_logit,
                strict_target,
                pos_weight=torch.as_tensor(pos_weight, device=device),
            )
            loss = margin_loss + float(preregistration["strict_loss_weight"]) * strict_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["margin_loss"] += float(margin_loss.detach())
            totals["strict_loss"] += float(strict_loss.detach())
            batches += 1
        row = {
            "epoch": float(epoch),
            **{key: value / max(1, batches) for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


@torch.no_grad()
def calibration(
    model: b27.LatentPropertyEnergy,
    labels: Sequence[JointLabel],
    dev_conditions: set[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    selected = [row for row in labels if row.condition_index in dev_conditions]
    margins: list[float] = []
    logits: list[float] = []
    model.eval()
    for start in range(0, len(selected), int(batch_size)):
        items = selected[start : start + int(batch_size)]
        endpoint = torch.from_numpy(np.stack([row.endpoint for row in items])).to(device)
        context = torch.from_numpy(np.stack([row.context for row in items])).to(device)
        margin, strict_logit = model(endpoint, context)
        margins.extend(margin.float().cpu().tolist())
        logits.extend(strict_logit.float().cpu().tolist())
    target = np.asarray([row.margin for row in selected], dtype=np.float64)
    strict = np.asarray([row.strict for row in selected], dtype=np.float64)
    predicted = np.asarray(margins, dtype=np.float64)
    strict_score = np.asarray(logits, dtype=np.float64)
    positive_scores = strict_score[strict > 0.5]
    negative_scores = strict_score[strict <= 0.5]
    combined = np.concatenate([positive_scores, negative_scores])
    if len(positive_scores) and len(negative_scores):
        ranks = rank_values(combined)
        positive_rank_sum = ranks[: len(positive_scores)].sum()
        auc = float(
            (positive_rank_sum - len(positive_scores) * (len(positive_scores) - 1) / 2)
            / (len(positive_scores) * len(negative_scores))
        )
    else:
        auc = 0.0
    score = predicted + 0.25 * np.tanh(strict_score)
    by_condition: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(selected):
        by_condition[row.condition_index].append(index)
    comparisons = 0
    correct = 0.0
    supported = 0
    top1_strict = 0
    for indices in by_condition.values():
        positive = [index for index in indices if strict[index] > 0.5]
        negative = [index for index in indices if strict[index] <= 0.5]
        for positive_index in positive:
            for negative_index in negative:
                comparisons += 1
                correct += float(score[positive_index] > score[negative_index])
                correct += 0.5 * float(score[positive_index] == score[negative_index])
        if positive:
            supported += 1
            top1_strict += int(strict[max(indices, key=lambda index: score[index])] > 0.5)
    spearman = (
        float(np.corrcoef(rank_values(target), rank_values(predicted))[0, 1])
        if len(target) > 1
        else 0.0
    )
    by_task = {}
    for task in TASK_SPECS:
        task_indices = [index for index, row in enumerate(selected) if row.task == task]
        task_strict = strict[task_indices]
        by_task[task] = {
            "examples": len(task_indices),
            "strict_positive_rate": float(task_strict.mean()) if len(task_strict) else 0.0,
            "supported_conditions": len(
                {
                    selected[index].condition_index
                    for index in task_indices
                    if strict[index] > 0.5
                }
            ),
        }
    return {
        "examples": len(selected),
        "strict_positive_rate": float(strict.mean()) if len(strict) else 0.0,
        "margin_mae": float(np.mean(np.abs(predicted - target))) if len(target) else 0.0,
        "margin_spearman": spearman if math.isfinite(spearman) else 0.0,
        "strict_auc": auc,
        "pairwise_preference_accuracy": correct / max(1, comparisons),
        "supported_conditions": supported,
        "top1_strict_recall": top1_strict / max(1, supported),
        "by_task": by_task,
    }


@torch.no_grad()
def joint_actions(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: b27.LatentPropertyEnergy,
    condition: b29.TransferCondition,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
    *,
    seed: int,
    energy_weight: float,
) -> list[dict[str, object]]:
    sites, contexts_np, site_logits_np = condition_site_state(
        fragment_model, condition, source_latent, config, device
    )
    contexts = torch.from_numpy(contexts_np).to(device)
    vocabulary = torch.from_numpy(target_endpoints).to(device)
    energy = b28.token_energy(
        energy_model, vocabulary, contexts, chunk_size=int(config.energy_chunk_size)
    )
    standardized = (energy - energy.mean()) / energy.std().clamp_min(
        float(config.energy_scale_floor)
    )
    site_logits = torch.from_numpy(site_logits_np).to(device)
    logits = site_logits[:, None] / max(float(config.site_temperature), 1e-6)
    logits = logits.expand_as(standardized).clone()
    logits = logits + float(energy_weight) * standardized
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
    output = []
    for flat_index in selected.tolist():
        site_index, token_index = divmod(int(flat_index), len(target_fragments))
        site = sites[site_index]
        token = target_fragments[token_index]
        output.append(
            {
                "smiles": kernel.fragments.join_fragments(site.core, token),
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": token,
                "joint_energy": float(energy[site_index, token_index].cpu()),
                "joint_probability": float(probabilities[flat_index].cpu()),
                "joint_distribution_entropy": entropy,
            }
        )
    return output


def freeze_methods(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: b27.LatentPropertyEnergy,
    conditions: Sequence[b29.TransferCondition],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> dict[str, list[tuple[b29.TransferCondition, list[dict[str, object]]]]]:
    frozen = {"frozen_b24": [], "uniform_joint": [], "learned_joint": []}
    for index, condition in enumerate(conditions):
        seed = int(config.seed) * 100000 + index
        b24_actions = kernel.generate_actions(
            fragment_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        uniform = joint_actions(
            fragment_model,
            energy_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
            energy_weight=0.0,
        )
        learned = joint_actions(
            fragment_model,
            energy_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
            energy_weight=float(config.energy_weight),
        )
        for name, actions in (
            ("frozen_b24", b24_actions),
            ("uniform_joint", uniform),
            ("learned_joint", learned),
        ):
            if len(actions) != int(config.num_attempts):
                raise RuntimeError(f"B31 {name} did not freeze exactly 20 attempts")
            frozen[name].append((condition, actions))
    return frozen


def evaluate_frozen(
    values: Sequence[tuple[b29.TransferCondition, list[dict[str, object]]]],
    assay_oracles: Mapping[str, pinned.PinnedAssayOracle],
    preregistration: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    for condition, actions in values:
        for attempt, raw in enumerate(actions, start=1):
            canonical = graph.canonical_smiles(str(raw.get("smiles", "") or ""))
            similarity = (
                graph.morgan_tanimoto(condition.source_smiles, canonical)
                if canonical
                else None
            )
            raw_rows.append(
                {
                    "condition_id": condition.condition_id,
                    "task": condition.task,
                    "property_count": condition.property_count,
                    "attempt": attempt,
                    "source_smiles": condition.source_smiles,
                    "generated_smiles": canonical or "",
                    "valid": bool(canonical),
                    "source_tanimoto": float(similarity or 0.0),
                    "site_core": raw.get("site_core", ""),
                    "source_fragment": raw.get("source_fragment", ""),
                    "target_fragment_token": raw.get("target_fragment_token", ""),
                    "joint_energy": raw.get("joint_energy", ""),
                    "joint_probability": raw.get("joint_probability", ""),
                    "joint_distribution_entropy": raw.get(
                        "joint_distribution_entropy", ""
                    ),
                }
            )

    by_task_indices: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(raw_rows):
        if bool(row["valid"]):
            by_task_indices[str(row["task"])].append(index)
    source_scores: dict[tuple[str, str], float] = {}
    candidate_scores: dict[int, float] = {}
    for task, indices in by_task_indices.items():
        assay_prop = TASK_SPECS[task][0][0]
        sources = sorted({str(raw_rows[index]["source_smiles"]) for index in indices})
        values_for_sources = assay_oracles[assay_prop].score_many(
            sources, batch_size=int(preregistration["oracle_batch_size"])
        )
        for source, value in zip(sources, values_for_sources):
            source_scores[(source, assay_prop)] = value
        values_for_candidates = assay_oracles[assay_prop].score_many(
            [str(raw_rows[index]["generated_smiles"]) for index in indices],
            batch_size=int(preregistration["oracle_batch_size"]),
        )
        for index, value in zip(indices, values_for_candidates):
            candidate_scores[index] = value

    descriptor_cache: dict[tuple[str, str], float] = {}
    evaluated_properties = 0
    for index, row in enumerate(raw_rows):
        margins: dict[str, float | None] = {}
        success = bool(row["valid"])
        if success:
            for prop, direction in TASK_SPECS[str(row["task"])]:
                if prop in assay_oracles:
                    source_value = source_scores[(str(row["source_smiles"]), prop)]
                    candidate_value = candidate_scores[index]
                else:
                    source_value = descriptor_value(
                        str(row["source_smiles"]), prop, descriptor_cache
                    )
                    candidate_value = descriptor_value(
                        str(row["generated_smiles"]), prop, descriptor_cache
                    )
                margin = float(direction) * (candidate_value - source_value)
                margins[prop] = margin
                success = success and margin > 0.0
                evaluated_properties += 1
        row["property_margins"] = json.dumps(margins, sort_keys=True)
        row["property_success"] = success
        row["success_t0_15"] = bool(success and float(row["source_tanimoto"]) >= 0.15)
        row["success_t0_65"] = bool(success and float(row["source_tanimoto"]) >= 0.65)

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["condition_id"])].append(row)
    condition_rows = []
    for condition_id, rows in grouped.items():
        if len(rows) != 20:
            raise ValueError(f"B31 {condition_id} has {len(rows)} attempts")
        condition_rows.append(
            {
                "condition_id": condition_id,
                "task": rows[0]["task"],
                "unique_valid": len(
                    {str(row["generated_smiles"]) for row in rows if bool(row["valid"])}
                ),
                "any20_t0_15": any(bool(row["success_t0_15"]) for row in rows),
                "any20_t0_65": any(bool(row["success_t0_65"]) for row in rows),
            }
        )

    def metrics(rows, conditions) -> dict[str, object]:
        valid = [row for row in rows if bool(row["valid"])]
        expected = sum(int(row["property_count"]) for row in rows)
        return {
            "conditions": len(conditions),
            "candidate_rows": len(rows),
            "attempted_per_condition": 20,
            "validity": sum(bool(row["valid"]) for row in rows) / max(1, len(rows)),
            "acc_all_t0_15_per_attempt": sum(
                bool(row["success_t0_15"]) for row in rows
            ) / max(1, len(rows)),
            "acc_all_t0_65_per_attempt": sum(
                bool(row["success_t0_65"]) for row in rows
            ) / max(1, len(rows)),
            "acc_any20_t0_15": sum(bool(row["any20_t0_15"]) for row in conditions)
            / max(1, len(conditions)),
            "acc_any20_t0_65": sum(bool(row["any20_t0_65"]) for row in conditions)
            / max(1, len(conditions)),
            "mean_unique_valid": float(
                np.mean([row["unique_valid"] for row in conditions])
            )
            if conditions
            else 0.0,
            "mean_source_tanimoto": float(
                np.mean([float(row["source_tanimoto"]) for row in valid])
            )
            if valid
            else 0.0,
            "oracle_coverage": evaluated_properties / max(1, expected),
        }

    overall = metrics(raw_rows, condition_rows)
    overall["by_task"] = {}
    for task in TASK_SPECS:
        task_rows = [row for row in raw_rows if row["task"] == task]
        task_conditions = [row for row in condition_rows if row["task"] == task]
        # Coverage is global above; all valid rows are expected to be covered.
        task_metrics = metrics(task_rows, task_conditions)
        task_metrics["oracle_coverage"] = 1.0 if task_metrics["validity"] == 1.0 else 0.0
        overall["by_task"][task] = task_metrics
    return raw_rows, overall


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.output_dir / "summary.json").exists():
        raise ValueError(f"Completed B31 result exists: {args.output_dir / 'summary.json'}")
    preregistration = read_preregistration(args.protocol_manifest)
    b29_summary, b30_summary = read_evidence_contracts(
        args.b29_summary, args.b30_summary, preregistration
    )
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    representation, _representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        b27.load_frozen_fragment_model(
            args.fragment_checkpoint, device, preregistration
        )
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
    ):
        if frozen_manifest.get(key) != belief.file_sha256(path):
            raise ValueError(f"B31 frozen B24 input drift: {key}")
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
    source_pairs, source_selection = select_train_sources(
        train_pairs,
        limit=int(preregistration["train_source_limit"]),
        seed=int(preregistration["source_selection_seed"]),
        site_config=action_config,
    )
    fit_sources, dev_sources = source_split(
        source_pairs,
        dev_fraction=float(preregistration["dev_fraction"]),
        seed=int(preregistration["split_seed"]),
    )
    conditions = build_conditions(
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
    site_lists: list[list[kernel.Site]] = []
    site_contexts: list[np.ndarray] = []
    candidates: list[CandidateAction] = []
    for condition_index, condition in enumerate(conditions):
        sites, contexts, _site_logits = condition_site_state(
            fragment_model,
            condition,
            condition_latents[condition_index],
            action_config,
            device,
        )
        site_lists.append(sites)
        site_contexts.append(contexts)
        candidates.extend(
            sample_candidate_actions(
                condition,
                condition_index,
                sites,
                target_fragments,
                limit=int(preregistration["actions_per_condition"]),
                seed=int(preregistration["label_sampling_seed"]),
            )
        )
        if (condition_index + 1) % 64 == 0:
            print(
                json.dumps(
                    {
                        "stage": "label_candidates",
                        "conditions": condition_index + 1,
                        "candidate_actions": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    labels, label_support = build_labels(
        conditions,
        site_contexts,
        candidates,
        target_endpoints,
        assay_oracles,
        preregistration,
    )
    fit_conditions = {
        index for index, row in enumerate(conditions) if row.source_smiles in fit_sources
    }
    dev_conditions = {
        index for index, row in enumerate(conditions) if row.source_smiles in dev_sources
    }
    energy_model = b27.LatentPropertyEnergy(
        endpoint_dim=int(preregistration["fingerprint_bits"]),
        context_dim=int(site_contexts[0].shape[1]),
        hidden_dim=int(preregistration["hidden_dim"]),
    ).to(device)
    history = train_energy(
        energy_model, labels, fit_conditions, preregistration, device
    )
    heldout_calibration = calibration(
        energy_model,
        labels,
        dev_conditions,
        batch_size=int(preregistration["batch_size"]),
        device=device,
    )

    dev_source_pairs = sorted(
        [pair for pair in source_pairs if pair.source_smiles in dev_sources],
        key=lambda pair: b27.stable_value(
            int(preregistration["generation_seed"]), "dev", pair.source_smiles
        ),
    )[: int(preregistration["internal_dev_source_limit"])]
    dev_generation_conditions = build_conditions(
        dev_source_pairs, condition_dim=int(preregistration["condition_dim"])
    )
    dev_generation_latents_unique = kernel.encode_sources(
        representation,
        dev_source_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    dev_latent_lookup = {
        pair.source_smiles: dev_generation_latents_unique[index]
        for index, pair in enumerate(dev_source_pairs)
    }
    dev_generation_latents = np.stack(
        [dev_latent_lookup[row.source_smiles] for row in dev_generation_conditions]
    ).astype(np.float32)
    generation_config = SimpleNamespace(
        **vars(action_config),
        num_attempts=int(preregistration["exact_raw_attempts_per_condition"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        energy_chunk_size=int(preregistration["energy_chunk_size"]),
        energy_scale_floor=float(preregistration["energy_scale_floor"]),
        energy_weight=float(preregistration["energy_weight"]),
        seed=int(preregistration["generation_seed"]),
    )
    frozen_dev = freeze_methods(
        fragment_model,
        energy_model,
        dev_generation_conditions,
        dev_generation_latents,
        target_fragments,
        target_endpoints,
        generation_config,
        device,
    )
    dev_rows = {}
    dev_metrics = {}
    for name, frozen in frozen_dev.items():
        dev_rows[name], dev_metrics[name] = evaluate_frozen(
            frozen, assay_oracles, preregistration
        )

    gates = dict(preregistration["gates"])
    learned = dev_metrics["learned_joint"]
    uniform = dev_metrics["uniform_joint"]
    frozen_b24 = dev_metrics["frozen_b24"]
    checks = {
        "minimum_labels": {
            "value": label_support["complete_labels"],
            "threshold": gates["minimum_labels"],
        },
        "label_oracle_coverage": {
            "value": label_support["oracle_coverage"],
            "threshold": 1.0,
        },
        "fit_dev_source_overlap": {
            "value": len(fit_sources & dev_sources),
            "threshold": 0,
        },
        "calibration_examples": {
            "value": heldout_calibration["examples"],
            "threshold": gates["minimum_dev_labels"],
        },
        "strict_auc": {
            "value": heldout_calibration["strict_auc"],
            "threshold": gates["strict_auc"],
        },
        "pairwise_preference_accuracy": {
            "value": heldout_calibration["pairwise_preference_accuracy"],
            "threshold": gates["pairwise_preference_accuracy"],
        },
        "exact_attempts": {
            "value": learned["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": learned["validity"], "threshold": gates["validity"]},
        "mean_unique_valid": {
            "value": learned["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": learned["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "overall_any20_t0_15": {
            "value": learned["acc_any20_t0_15"],
            "threshold": gates["overall_any20_t0_15"],
        },
        "any20_delta_vs_frozen_b24": {
            "value": learned["acc_any20_t0_15"] - frozen_b24["acc_any20_t0_15"],
            "threshold": gates["any20_delta_vs_frozen_b24"],
        },
        "per_attempt_delta_vs_uniform": {
            "value": learned["acc_all_t0_15_per_attempt"]
            - uniform["acc_all_t0_15_per_attempt"],
            "threshold": gates["per_attempt_delta_vs_uniform"],
        },
    }
    for task in TASK_SPECS:
        checks[f"{task}:any20_t0_15"] = {
            "value": learned["by_task"][task]["acc_any20_t0_15"],
            "threshold": gates["minimum_task_any20_t0_15"],
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
    internal_gate_passed = not failures

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "assay_joint_site_token_energy.pt"
    run_manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "b29_summary_sha256": belief.file_sha256(args.b29_summary),
        "b30_summary_sha256": belief.file_sha256(args.b30_summary),
        "representation_protocol": representation_summary.get("protocol"),
        "training_sources": len(source_pairs),
        "training_source_selection": source_selection,
        "fit_sources": len(fit_sources),
        "internal_dev_sources": len(dev_sources),
        "fit_internal_dev_source_overlap": len(fit_sources & dev_sources),
        "training_labels_from_b24_train_sources_only": True,
        "evaluation_source_training_access": False,
        "b26_heldout_access": False,
        "official_test_access": False,
        "generation_target_access": False,
        "moledit_target_access": False,
        "property_oracle_generation_access": False,
        "pinned_oracle_label_access_train_only": True,
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
            "model_state": energy_model.state_dict(),
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
        write_rows(args.output_dir / f"internal_dev_{name}_candidates.csv", rows)

    table1_transfer: dict[str, object]
    if internal_gate_passed:
        forbidden_sources = {pair.source_smiles for pair in train_pairs}
        table1_conditions, selection = b29.select_conditions(
            args.table1_eval_csv,
            tasks=list(TASK_SPECS),
            per_task=int(preregistration["table1_per_task"]),
            seed=int(preregistration["table1_selection_seed"]),
            forbidden_sources=forbidden_sources,
            condition_dim=int(preregistration["condition_dim"]),
            graph_fingerprint_bits=int(preregistration["graph_fingerprint_bits"]),
        )
        table1_latents = kernel.encode_sources(
            representation,
            table1_conditions,
            device,
            batch_size=int(preregistration["encoding_batch_size"]),
        )
        table1_config_values = dict(vars(generation_config))
        table1_config_values["seed"] = int(preregistration["table1_generation_seed"])
        table1_config = SimpleNamespace(**table1_config_values)
        frozen_table1 = freeze_methods(
            fragment_model,
            energy_model,
            table1_conditions,
            table1_latents,
            target_fragments,
            target_endpoints,
            table1_config,
            device,
        )
        table1_metrics = {}
        for name, frozen in frozen_table1.items():
            rows, table1_metrics[name] = evaluate_frozen(
                frozen, assay_oracles, preregistration
            )
            write_rows(args.output_dir / f"table1_{name}_candidates.csv", rows)
        table1_transfer = {
            "status": "completed_once_after_internal_gate",
            "selection": selection,
            "methods": table1_metrics,
            "b29_assay_baseline": {
                task: dict(b29_summary["energy_tilted_evaluation"]["by_task"][task])
                for task in TASK_SPECS
            },
        }
        run_manifest["table1_eval_csv_sha256"] = belief.file_sha256(args.table1_eval_csv)
        run_manifest["table1_eval_rows_read"] = len(table1_conditions)
    else:
        table1_transfer = {"status": "not_run_internal_gate_failed"}
        run_manifest["table1_eval_rows_read"] = 0

    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": run_manifest,
        "b30_support_decision": b30_summary.get("decision"),
        "label_support": label_support,
        "training": history,
        "calibration": heldout_calibration,
        "internal_dev": dev_metrics,
        "internal_gate": {"passed": internal_gate_passed, "checks": checks, "failures": failures},
        "table1_transfer": table1_transfer,
        "decision": (
            "freeze_joint_latent_and_expand_table1_replay"
            if internal_gate_passed
            else "stop_joint_latent_after_single_preregistered_pilot"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
