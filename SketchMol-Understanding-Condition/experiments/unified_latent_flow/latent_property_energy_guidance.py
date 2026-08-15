#!/usr/bin/env python3
"""Train a train-only property energy and guide the frozen B24 fragment latent.

B24 supplies the frozen graph encoder, condition-to-fragment flow, attachment
site distribution, train-only fragment vocabulary, and one-shot RDKit grammar.
B27 learns only a differentiable energy over the transported fragment latent.
Its labels are source-relative property margins on molecules assembled from
B24 training sources and train-only fragment tokens.  Fit and internal-dev are
source/task grouped before any energy labels are used for fitting.

At generation time each Gaussian latent follows the frozen B24 flow and then
moves along the learned energy gradient before one nearest-token quantization.
No property oracle, validity check, candidate pool, selection, retry, repair,
or second edit is available during generation.  Exactly 20 raw attempts are
frozen before internal-dev targets and property scorers are opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import latent_fragment_attachment_kernel as kernel  # noqa: E402


base = kernel.base
belief = kernel.belief
graph = kernel.graph
hierarchical = kernel.hierarchical
unified = kernel.unified

PROTOCOL = "train_only_latent_property_energy_guidance_v27"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_model_protocol": kernel.PROTOCOL,
        "frozen_model_seed": 1761,
        "train_selection_seed": 1741,
        "historical_validation_seed": 1742,
        "retired_development_seed": 2719,
        "b26_heldout_access": False,
        "official_test_access": False,
        "num_attempts": 20,
        "candidate_ranking": False,
        "second_edit": False,
        "property_oracle_generation_access": False,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise ValueError(f"B27 preregistration drift: {drift}")
    return payload


def stable_value(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def load_frozen_fragment_model(
    checkpoint_path: Path, device: torch.device, preregistration: Mapping[str, object]
) -> tuple[kernel.FragmentAttachmentKernel, list[str], np.ndarray, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("stage") != preregistration["frozen_model_protocol"]:
        raise ValueError("B27 fragment checkpoint protocol drift")
    manifest = dict(payload.get("manifest", {}))
    contract = {
        "seed": preregistration["frozen_model_seed"],
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "molecular_candidate_ranking": False,
        "failed_attachment_retry": False,
        "exact_raw_attempts_per_condition": 20,
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in contract.items()
        if manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"Frozen B24 contract drift: {drift}")
    config = dict(payload["model_config"])
    model = kernel.FragmentAttachmentKernel(
        source_dim=int(config["source_dim"]),
        condition_dim=int(config["condition_dim"]),
        site_dim=int(config["site_dim"]),
        endpoint_dim=int(config["endpoint_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (
        model,
        list(payload["target_fragments"]),
        np.asarray(payload["target_endpoints"], dtype=np.float32),
        manifest,
    )


def reconstruct_b24_train_pairs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[list[object], dict[str, object]]:
    selection = SimpleNamespace(
        property_counts=",".join(str(value) for value in preregistration["property_counts"]),
        validation_csv=args.validation_csv,
        train_csv=args.train_csv,
        graph_fingerprint_bits=int(preregistration["graph_fingerprint_bits"]),
        condition_dim=int(preregistration["condition_dim"]),
        mcs_timeout=int(preregistration["mcs_timeout"]),
        min_common_fraction=float(preregistration["min_common_fraction"]),
        validation_limit=int(preregistration["historical_validation_limit"]),
        validation_exclusion_seed=int(preregistration["historical_validation_seed"]),
        validation_selection_seed=int(preregistration["retired_development_seed"]),
        train_limit=int(preregistration["train_limit"]),
        train_selection_seed=int(preregistration["train_selection_seed"]),
    )
    train_pairs, retired_dev, split = kernel.select_pairs(selection)
    return train_pairs, {
        "reconstructed_b24_train_pairs": len(train_pairs),
        "retired_development_pairs_read_for_split_reconstruction_only": len(retired_dev),
        "b26_heldout_rows_read": 0,
        "official_test_rows_read": 0,
        "train_filter_counts": split["train_filter_counts"],
    }


def covered_pairs_and_actions(
    pairs: Sequence[object], config: SimpleNamespace
) -> tuple[list[object], list[kernel.AttachmentAction], dict[str, int]]:
    selected_pairs: list[object] = []
    selected_actions: list[kernel.AttachmentAction] = []
    multi_action_pairs = 0
    for pair in pairs:
        actions = kernel.exact_attachment_actions(pair, len(selected_pairs), config)
        if not actions:
            continue
        if len(actions) > 1:
            multi_action_pairs += 1
        selected_pairs.append(pair)
        selected_actions.append(actions[0][0])
    return selected_pairs, selected_actions, {
        "covered_pairs": len(selected_pairs),
        "uncovered_pairs": len(pairs) - len(selected_pairs),
        "multi_action_pairs": multi_action_pairs,
    }


def grouped_fit_dev_split(
    pairs: Sequence[object], *, seed: int, dev_fraction: float
) -> tuple[set[int], set[int]]:
    by_source: defaultdict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        by_source[str(pair.source_smiles)].append(index)
    by_task_signature: defaultdict[str, list[str]] = defaultdict(list)
    for source, indices in by_source.items():
        signature = "+".join(sorted({str(pairs[index].task) for index in indices}))
        by_task_signature[signature].append(source)
    fit: set[int] = set()
    dev: set[int] = set()
    for signature, sources in sorted(by_task_signature.items()):
        ordered = sorted(
            sources,
            key=lambda source: stable_value(seed, signature, source),
        )
        if len(ordered) == 1:
            fit.update(by_source[ordered[0]])
            continue
        dev_count = max(1, int(round(len(ordered) * float(dev_fraction))))
        dev_count = min(dev_count, len(ordered) - 1)
        for source in ordered[:dev_count]:
            dev.update(by_source[source])
        for source in ordered[dev_count:]:
            fit.update(by_source[source])
    if not fit or not dev:
        raise ValueError("B27 grouped fit/internal-dev split is empty")
    return fit, dev


@torch.no_grad()
def action_contexts(
    model: kernel.FragmentAttachmentKernel,
    actions: Sequence[kernel.AttachmentAction],
    source_latents: np.ndarray,
    condition_tokens: np.ndarray,
    target_lookup: Mapping[str, int],
    target_endpoints: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(actions), int(batch_size)):
        items = actions[start : start + int(batch_size)]
        batch = kernel.collate_actions(
            items,
            source_latents,
            condition_tokens,
            target_lookup,
            target_endpoints,
            device,
        )
        request = model.request_context(batch["source"], batch["condition"])
        _logits, hidden = model.site_logits(
            request, batch["sites"], batch["site_mask"]
        )
        selected = hidden[
            torch.arange(len(items), device=device), batch["positive_site"]
        ]
        output.append(model.context_for_site(request, selected).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


@dataclass(frozen=True)
class EnergyLabel:
    pair_index: int
    context: np.ndarray
    endpoint: np.ndarray
    margin: float
    strict: float
    exact_target: bool
    generated_smiles: str


def source_relative_label(pair: object, smiles: str) -> tuple[float, float, int]:
    canonical = graph.canonical_smiles(smiles)
    specs = base.task_specs(pair.row)
    if not canonical or not specs:
        return -2.0, 0.0, 0
    margins: list[float] = []
    evaluated = 0
    for prop, direction in specs:
        source_value = unified.score_property(pair.source_smiles, prop)
        candidate_value = unified.score_property(canonical, prop)
        if source_value is None or candidate_value is None:
            margins.append(-2.0)
            continue
        if not math.isfinite(float(source_value)) or not math.isfinite(
            float(candidate_value)
        ):
            margins.append(-2.0)
            continue
        canonical_prop = unified.canonical_prop(prop)
        normalizer = max(
            float(unified.PROPERTY_NORMALIZERS.get(canonical_prop, 1.0)), 1e-8
        )
        margins.append(
            float(direction) * (float(candidate_value) - float(source_value))
            / normalizer
        )
        evaluated += 1
    similarity = graph.morgan_tanimoto(pair.source_smiles, canonical) or 0.0
    similarity_margin = (float(similarity) - 0.4) / 0.2
    all_margins = [*margins, similarity_margin]
    minimum = float(np.clip(min(all_margins, default=-2.0), -2.0, 2.0))
    strict = float(
        evaluated == len(specs)
        and all(value > 0.0 for value in margins)
        and similarity >= 0.4
    )
    return minimum, strict, evaluated


def build_energy_labels(
    pairs: Sequence[object],
    actions: Sequence[kernel.AttachmentAction],
    contexts: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    *,
    hard_negatives: int,
) -> tuple[list[EnergyLabel], dict[str, object]]:
    lookup = {fragment: index for index, fragment in enumerate(target_fragments)}
    labels: list[EnergyLabel] = []
    attempted = 0
    full_oracle = 0
    exact_strict = 0
    negative_strict = 0
    pairs_with_full_set = 0
    for pair_index, (pair, action) in enumerate(zip(pairs, actions)):
        sites = kernel.source_sites(
            pair.source_smiles,
            SimpleNamespace(
                min_core_heavy_atoms=5,
                max_variable_heavy_atoms=30,
                fingerprint_bits=target_endpoints.shape[1],
            ),
        )
        if int(action.positive_site) >= len(sites):
            continue
        target_index = lookup[action.target_fragment]
        target_endpoint = target_endpoints[target_index]
        distances = ((target_endpoints - target_endpoint[None, :]) ** 2).mean(axis=1)
        ordered = np.argsort(distances, kind="stable")
        candidates: list[tuple[int, str, bool]] = [
            (target_index, pair.target_smiles, True)
        ]
        for token_index in ordered.tolist():
            if token_index == target_index:
                continue
            # The positive site identifies one of the source-only MMPA cores.
            product = kernel.fragments.join_fragments(
                sites[int(action.positive_site)].core,
                target_fragments[int(token_index)],
            )
            canonical = graph.canonical_smiles(product)
            if not canonical:
                continue
            if canonical in {pair.source_smiles, pair.target_smiles}:
                continue
            candidates.append((int(token_index), canonical, False))
            if len(candidates) >= int(hard_negatives) + 1:
                break
        if len(candidates) < 2:
            continue
        pairs_with_full_set += 1
        for token_index, smiles, exact in candidates:
            attempted += 1
            margin, strict, evaluated = source_relative_label(pair, smiles)
            full_oracle += int(evaluated == pair.property_count)
            if evaluated != pair.property_count:
                continue
            exact_strict += int(exact and strict > 0.5)
            negative_strict += int((not exact) and strict > 0.5)
            labels.append(
                EnergyLabel(
                    pair_index=pair_index,
                    context=contexts[pair_index],
                    endpoint=target_endpoints[token_index],
                    margin=margin,
                    strict=strict,
                    exact_target=exact,
                    generated_smiles=smiles,
                )
            )
    return labels, {
        "attempted_labels": attempted,
        "complete_labels": len(labels),
        "full_oracle_label_rate": full_oracle / max(1, attempted),
        "pairs_with_candidate_set": pairs_with_full_set,
        "exact_target_strict_labels": exact_strict,
        "hard_negative_strict_labels": negative_strict,
    }


class LatentPropertyEnergy(nn.Module):
    def __init__(self, endpoint_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(endpoint_dim + context_dim),
            nn.Linear(endpoint_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, endpoint: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(torch.cat([endpoint, context], dim=-1))
        return output[..., 0], output[..., 1]


def train_energy(
    model: LatentPropertyEnergy,
    labels: Sequence[EnergyLabel],
    fit_pairs: set[int],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    indices = [index for index, row in enumerate(labels) if row.pair_index in fit_pairs]
    if len(indices) < 100:
        raise ValueError(f"Only {len(indices)} fit energy labels")
    positives = sum(labels[index].strict > 0.5 for index in indices)
    negatives = len(indices) - positives
    pos_weight = min(10.0, max(1.0, negatives / max(1, positives)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = random.Random(int(seed))
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        order = list(indices)
        generator.shuffle(order)
        totals = defaultdict(float)
        batches = 0
        model.train()
        for start in range(0, len(order), int(batch_size)):
            batch_indices = order[start : start + int(batch_size)]
            endpoint = torch.from_numpy(
                np.stack([labels[index].endpoint for index in batch_indices])
            ).to(device)
            context = torch.from_numpy(
                np.stack([labels[index].context for index in batch_indices])
            ).to(device)
            target_margin = torch.as_tensor(
                [labels[index].margin for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            target_strict = torch.as_tensor(
                [labels[index].strict for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            predicted_margin, strict_logit = model(endpoint, context)
            margin_loss = F.smooth_l1_loss(predicted_margin, target_margin)
            strict_loss = F.binary_cross_entropy_with_logits(
                strict_logit,
                target_strict,
                pos_weight=torch.as_tensor(pos_weight, device=device),
            )
            loss = margin_loss + 0.5 * strict_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["margin_loss"] += float(margin_loss.detach())
            totals["strict_loss"] += float(strict_loss.detach())
            batches += 1
        history.append(
            {
                "epoch": float(epoch),
                **{name: value / max(1, batches) for name, value in totals.items()},
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    return history


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


@torch.no_grad()
def energy_calibration(
    model: LatentPropertyEnergy,
    labels: Sequence[EnergyLabel],
    dev_pairs: set[int],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    selected = [row for row in labels if row.pair_index in dev_pairs]
    predictions: list[float] = []
    logits: list[float] = []
    for start in range(0, len(selected), int(batch_size)):
        items = selected[start : start + int(batch_size)]
        endpoint = torch.from_numpy(np.stack([row.endpoint for row in items])).to(device)
        context = torch.from_numpy(np.stack([row.context for row in items])).to(device)
        margin, strict_logit = model(endpoint, context)
        predictions.extend(margin.float().cpu().tolist())
        logits.extend(strict_logit.float().cpu().tolist())
    target = np.asarray([row.margin for row in selected], dtype=np.float64)
    strict = np.asarray([row.strict for row in selected], dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    strict_score = np.asarray(logits, dtype=np.float64)
    spearman = float(np.corrcoef(rank_values(target), rank_values(predicted))[0, 1])
    positives = strict_score[strict > 0.5]
    negatives = strict_score[strict <= 0.5]
    auc = (
        float(
            np.mean(
                (positives[:, None] > negatives[None, :]).astype(np.float64)
                + 0.5 * (positives[:, None] == negatives[None, :])
            )
        )
        if len(positives) and len(negatives)
        else 0.0
    )
    by_pair: defaultdict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(selected):
        by_pair[row.pair_index].append(index)
    comparisons = 0
    correct = 0.0
    supported = 0
    top1_strict = 0
    energy_score = predicted + 0.25 * np.tanh(strict_score)
    for indices in by_pair.values():
        positive_indices = [index for index in indices if strict[index] > 0.5]
        negative_indices = [index for index in indices if strict[index] <= 0.5]
        for positive in positive_indices:
            for negative in negative_indices:
                comparisons += 1
                correct += float(energy_score[positive] > energy_score[negative])
                correct += 0.5 * float(energy_score[positive] == energy_score[negative])
        if positive_indices:
            supported += 1
            top = max(indices, key=lambda index: energy_score[index])
            top1_strict += int(strict[top] > 0.5)
    return {
        "examples": float(len(selected)),
        "strict_positive_rate": float(strict.mean()) if len(strict) else 0.0,
        "margin_mae": float(np.mean(np.abs(predicted - target))) if len(target) else 0.0,
        "margin_spearman": spearman if math.isfinite(spearman) else 0.0,
        "strict_auc": auc,
        "pairwise_preference_accuracy": correct / max(1, comparisons),
        "supported_pairs": float(supported),
        "top1_strict_recall": top1_strict / max(1, supported),
    }


def generation_subset(
    pairs: Sequence[object], dev_pairs: set[int], *, limit: int, seed: int
) -> list[object]:
    by_count: defaultdict[int, list[int]] = defaultdict(list)
    for index in dev_pairs:
        by_count[int(pairs[index].property_count)].append(index)
    selected: list[int] = []
    quota = max(1, int(limit) // max(1, len(by_count)))
    for count, indices in sorted(by_count.items()):
        ordered = sorted(
            indices,
            key=lambda index: stable_value(seed, count, pairs[index].source_smiles),
        )
        selected.extend(ordered[:quota])
    if len(selected) < int(limit):
        remaining = sorted(
            dev_pairs - set(selected),
            key=lambda index: stable_value(seed, "fill", pairs[index].source_smiles),
        )
        selected.extend(remaining[: int(limit) - len(selected)])
    return [pairs[index] for index in selected[: int(limit)]]


def guided_actions(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: LatentPropertyEnergy,
    pair: object,
    source_latent: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    sites = kernel.source_sites(pair.source_smiles, config)
    if not sites:
        return [
            {
                "smiles": "",
                "site_core": "",
                "source_fragment": "",
                "target_fragment_token": "",
                "latent_norm": 0.0,
                "quantization_distance": float("inf"),
                "site_entropy": 0.0,
                "energy_before": 0.0,
                "energy_after": 0.0,
                "guidance_displacement": 0.0,
            }
            for _ in range(int(config.num_attempts))
        ]
    attempts = int(config.num_attempts)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    source = torch.from_numpy(np.repeat(source_latent[None, :], attempts, axis=0)).to(device)
    condition = torch.from_numpy(
        np.repeat(pair.condition[None, ...], attempts, axis=0)
    ).to(device)
    site_matrix = np.stack([site.feature for site in sites]).astype(np.float32)
    site_tensor = torch.from_numpy(np.repeat(site_matrix[None, ...], attempts, axis=0)).to(device)
    mask = torch.ones(attempts, len(sites), dtype=torch.bool, device=device)
    with torch.no_grad():
        request = fragment_model.request_context(source, condition)
        site_logits, site_hidden = fragment_model.site_logits(request, site_tensor, mask)
        probabilities = torch.softmax(
            site_logits.float() / max(float(config.site_temperature), 1e-4), dim=-1
        )
        selected_sites = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).squeeze(-1)
        selected_hidden = site_hidden[
            torch.arange(attempts, device=device), selected_sites
        ]
        context = fragment_model.context_for_site(request, selected_hidden)
        latent = torch.randn(
            attempts,
            fragment_model.endpoint_dim,
            generator=generator,
            device=device,
            dtype=context.dtype,
        )
        for step in range(int(config.flow_steps)):
            time = torch.full(
                (attempts,),
                (step + 0.5) / max(1, int(config.flow_steps)),
                device=device,
                dtype=latent.dtype,
            )
            latent = latent + fragment_model.transport_velocity(
                latent, time, context
            ) / max(1, int(config.flow_steps))
    base_latent = latent.detach()
    energy_model.eval()
    for parameter in energy_model.parameters():
        parameter.requires_grad_(False)
    with torch.enable_grad():
        guided = base_latent.clone().requires_grad_(True)
        before_margin, before_logit = energy_model(torch.tanh(guided), context.detach())
        energy_before = before_margin + 0.25 * torch.tanh(before_logit)
        for _step in range(int(config.guidance_steps)):
            margin, strict_logit = energy_model(torch.tanh(guided), context.detach())
            trust = ((guided - base_latent) ** 2).mean(dim=-1)
            objective = margin + 0.25 * torch.tanh(strict_logit) - float(
                config.guidance_trust
            ) * trust
            gradient = torch.autograd.grad(objective.sum(), guided)[0]
            gradient = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            guided = (
                guided + float(config.guidance_step_size) * gradient
            ).clamp(-4.0, 4.0).detach().requires_grad_(True)
        after_margin, after_logit = energy_model(torch.tanh(guided), context.detach())
        energy_after = after_margin + 0.25 * torch.tanh(after_logit)
    guided = guided.detach()
    vocabulary = torch.from_numpy(target_endpoints).to(device=device, dtype=guided.dtype)
    distances = ((guided[:, None, :] - vocabulary[None, :, :]) ** 2).mean(dim=-1)
    for attempt, site_index in enumerate(selected_sites.tolist()):
        current_fragment = sites[int(site_index)].variable
        for token_index, token in enumerate(target_fragments):
            if token == current_fragment:
                distances[attempt, token_index] = torch.inf
    selected_tokens = distances.argmin(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    displacement = (guided - base_latent).norm(dim=-1)
    output: list[dict[str, object]] = []
    for attempt, (site_index, token_index) in enumerate(
        zip(selected_sites.tolist(), selected_tokens.tolist())
    ):
        site = sites[int(site_index)]
        target_fragment = target_fragments[int(token_index)]
        output.append(
            {
                "smiles": kernel.fragments.join_fragments(site.core, target_fragment),
                "site_core": site.core,
                "source_fragment": site.variable,
                "target_fragment_token": target_fragment,
                "latent_norm": float(guided[attempt].norm().cpu()),
                "quantization_distance": float(
                    distances[attempt, token_index].cpu()
                ),
                "site_entropy": float(entropy[attempt].cpu()),
                "energy_before": float(energy_before[attempt].detach().cpu()),
                "energy_after": float(energy_after[attempt].detach().cpu()),
                "guidance_displacement": float(displacement[attempt].cpu()),
            }
        )
    return output


def evaluate_guided(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: LatentPropertyEnergy,
    pairs: Sequence[object],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frozen: list[tuple[object, list[dict[str, object]]]] = []
    for pair_index, pair in enumerate(pairs):
        generated = guided_actions(
            fragment_model,
            energy_model,
            pair,
            source_latents[pair_index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=int(config.seed) * 100000 + pair_index,
        )
        if len(generated) != int(config.num_attempts):
            raise RuntimeError("B27 did not freeze exactly 20 guided attempts")
        frozen.append((pair, generated))

    # Evaluation targets and property oracles are first accessed below, after
    # every condition has frozen all 20 direct latent decodes.
    rows: list[dict[str, object]] = []
    from rdkit import Chem

    for pair_index, (pair, generated) in enumerate(frozen):
        specs = base.task_specs(pair.row)
        condition_id = str(
            pair.row.get("condition_id", "")
            or pair.row.get("sample_id", "")
            or f"internal_dev_{pair_index:04d}"
        )
        source_copy_target = graph.morgan_tanimoto(
            pair.source_smiles, pair.target_smiles
        ) or 0.0
        for rank, raw in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(str(raw["smiles"] or ""))
            valid = bool(canonical)
            molecule = Chem.MolFromSmiles(canonical) if valid else None
            source_similarity = (
                graph.morgan_tanimoto(pair.source_smiles, canonical) if valid else None
            )
            target_similarity = (
                graph.morgan_tanimoto(pair.target_smiles, canonical) if valid else None
            )
            fraction, _, evaluated, property_success = (
                unified.instruction_success_and_distance(
                    pair.row, canonical or "", task_specs=specs
                )
            )
            similarity_success = bool(
                source_similarity is not None and source_similarity >= 0.4
            )
            rows.append(
                {
                    "condition_id": condition_id,
                    "attempt": rank,
                    "property_count": pair.property_count,
                    "task": pair.task,
                    "source_smiles": pair.source_smiles,
                    "target_smiles": pair.target_smiles,
                    "generated_smiles": canonical or "",
                    "site_core": raw["site_core"],
                    "source_fragment": raw["source_fragment"],
                    "target_fragment_token": raw["target_fragment_token"],
                    "latent_norm": raw["latent_norm"],
                    "quantization_distance": raw["quantization_distance"],
                    "site_entropy": raw["site_entropy"],
                    "energy_before": raw["energy_before"],
                    "energy_after": raw["energy_after"],
                    "guidance_displacement": raw["guidance_displacement"],
                    "source_atom_count": int(pair.source.node_mask.sum()),
                    "target_atom_count": int(pair.target.node_mask.sum()),
                    "predicted_atom_count": int(molecule.GetNumAtoms()) if molecule else 0,
                    "valid": valid,
                    "source_tanimoto": float(source_similarity or 0.0),
                    "target_tanimoto": float(target_similarity or 0.0),
                    "source_copy_target_tanimoto": float(source_copy_target),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(property_success),
                    "strict_success": bool(property_success and similarity_success),
                    "source_similarity_success": similarity_success,
                }
            )
    metrics = base.summarize_candidates(rows, int(config.num_attempts))
    for name in (
        "energy_before",
        "energy_after",
        "guidance_displacement",
        "quantization_distance",
    ):
        values = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
        metrics[f"mean_{name}"] = float(np.mean(values)) if values else 0.0
    return rows, metrics


def overlap_audit(
    pairs: Sequence[object], fit_pairs: set[int], dev_pairs: set[int]
) -> dict[str, int]:
    fit_sources = {pairs[index].source_smiles for index in fit_pairs}
    dev_sources = {pairs[index].source_smiles for index in dev_pairs}
    fit_keys = {
        (pairs[index].source_smiles, pairs[index].target_smiles) for index in fit_pairs
    }
    dev_keys = {
        (pairs[index].source_smiles, pairs[index].target_smiles) for index in dev_pairs
    }
    return {
        "fit_internal_dev_source_overlap": len(fit_sources & dev_sources),
        "fit_internal_dev_pair_overlap": len(fit_keys & dev_keys),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B27 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))

    representation, representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        load_frozen_fragment_model(args.fragment_checkpoint, device, preregistration)
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
    ):
        if frozen_manifest.get(key) != belief.file_sha256(path):
            raise ValueError(f"Frozen B24 input drift: {key}")

    train_pairs, reconstruction = reconstruct_b24_train_pairs(args, preregistration)
    action_config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
    )
    pairs, actions, coverage = covered_pairs_and_actions(train_pairs, action_config)
    fit_pairs, dev_pairs = grouped_fit_dev_split(
        pairs,
        seed=int(preregistration["energy_split_seed"]),
        dev_fraction=float(preregistration["energy_dev_fraction"]),
    )
    source_latents = kernel.encode_sources(
        representation,
        pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    conditions = np.stack([pair.condition for pair in pairs]).astype(np.float32)
    target_lookup = {fragment: index for index, fragment in enumerate(target_fragments)}
    contexts = action_contexts(
        fragment_model,
        actions,
        source_latents,
        conditions,
        target_lookup,
        target_endpoints,
        device,
        int(preregistration["encoding_batch_size"]),
    )
    labels, label_stats = build_energy_labels(
        pairs,
        actions,
        contexts,
        target_fragments,
        target_endpoints,
        hard_negatives=int(preregistration["hard_negatives_per_pair"]),
    )
    energy_model = LatentPropertyEnergy(
        endpoint_dim=int(preregistration["fingerprint_bits"]),
        context_dim=int(contexts.shape[1]),
        hidden_dim=int(preregistration["energy_hidden_dim"]),
    ).to(device)
    history = train_energy(
        energy_model,
        labels,
        fit_pairs,
        epochs=int(preregistration["energy_epochs"]),
        batch_size=int(preregistration["energy_batch_size"]),
        learning_rate=float(preregistration["energy_learning_rate"]),
        weight_decay=float(preregistration["energy_weight_decay"]),
        seed=int(preregistration["seed"]),
        device=device,
    )
    calibration = energy_calibration(
        energy_model,
        labels,
        dev_pairs,
        device=device,
        batch_size=int(preregistration["energy_batch_size"]),
    )

    generation_pairs = generation_subset(
        pairs,
        dev_pairs,
        limit=int(preregistration["internal_dev_generation_limit"]),
        seed=int(preregistration["internal_dev_generation_seed"]),
    )
    generation_latents = kernel.encode_sources(
        representation,
        generation_pairs,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    generation_config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        num_attempts=int(preregistration["num_attempts"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        guidance_steps=int(preregistration["guidance_steps"]),
        guidance_step_size=float(preregistration["guidance_step_size"]),
        guidance_trust=float(preregistration["guidance_trust"]),
        seed=int(preregistration["internal_dev_generation_seed"]),
    )
    baseline_rows, baseline = kernel.evaluate(
        fragment_model,
        generation_pairs,
        generation_latents,
        target_fragments,
        target_endpoints,
        generation_config,
        device,
    )
    guided_rows, guided = evaluate_guided(
        fragment_model,
        energy_model,
        generation_pairs,
        generation_latents,
        target_fragments,
        target_endpoints,
        generation_config,
        device,
    )

    baseline_two = float(
        baseline["by_property_count"].get("2", {}).get("strict_any20", 0.0)
    )
    baseline_three = float(
        baseline["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    guided_two = float(
        guided["by_property_count"].get("2", {}).get("strict_any20", 0.0)
    )
    guided_three = float(
        guided["by_property_count"].get("3", {}).get("strict_any20", 0.0)
    )
    gates = dict(preregistration["gates"])
    audit = overlap_audit(pairs, fit_pairs, dev_pairs)
    checks = {
        "full_oracle_label_rate": {
            "value": label_stats["full_oracle_label_rate"],
            "threshold": gates["full_oracle_label_rate"],
        },
        "minimum_dev_energy_examples": {
            "value": calibration["examples"],
            "threshold": gates["minimum_dev_energy_examples"],
        },
        "margin_spearman": {
            "value": calibration["margin_spearman"],
            "threshold": gates["margin_spearman"],
        },
        "pairwise_preference_accuracy": {
            "value": calibration["pairwise_preference_accuracy"],
            "threshold": gates["pairwise_preference_accuracy"],
        },
        "exact_attempts": {
            "value": guided["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": guided["validity"], "threshold": gates["validity"]},
        "strict_any20": {
            "value": guided["strict_any20"],
            "threshold": gates["strict_any20"],
        },
        "strict_any20_delta": {
            "value": guided["strict_any20"] - baseline["strict_any20"],
            "threshold": gates["strict_any20_delta"],
        },
        "three_property_strict_any20": {
            "value": guided_three,
            "threshold": gates["three_property_strict_any20"],
        },
        "three_property_strict_delta": {
            "value": guided_three - baseline_three,
            "threshold": gates["three_property_strict_delta"],
        },
        "two_property_strict_delta": {
            "value": guided_two - baseline_two,
            "threshold": gates["two_property_strict_delta"],
        },
        "mean_unique_valid": {
            "value": guided["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": guided["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "all_split_overlaps_zero": {
            "value": sum(audit.values()),
            "threshold": 0,
        },
    }
    exact_checks = {"exact_attempts", "all_split_overlaps_zero"}
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name in exact_checks
            else item["value"] < item["threshold"]
        )
    ]
    run_manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "frozen_fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "representation_protocol": representation_summary.get("protocol"),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "b26_heldout_access": False,
        "official_test_access": False,
        "energy_fit_labels_train_only": True,
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "post_freeze_internal_dev_oracle_access": True,
        "frozen_b24_fragment_grammar": True,
        "one_latent_one_token_one_raw_molecule": True,
        "candidate_library": False,
        "candidate_ranking": False,
        "selector": False,
        "finalizer": False,
        "failed_attachment_retry": False,
        "second_edit": False,
        "exact_raw_attempts_per_condition": 20,
        "selected_pairs": len(pairs),
        "energy_fit_pairs": len(fit_pairs),
        "energy_internal_dev_pairs": len(dev_pairs),
        "generation_internal_dev_pairs": len(generation_pairs),
        "property_count_breakdown": {
            str(count): sum(pair.property_count == count for pair in generation_pairs)
            for count in (2, 3)
        },
        "split_audit": audit,
        "reconstruction": reconstruction,
        "coverage": coverage,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "latent_property_energy.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": energy_model.state_dict(),
            "model_config": {
                "endpoint_dim": int(preregistration["fingerprint_bits"]),
                "context_dim": int(contexts.shape[1]),
                "hidden_dim": int(preregistration["energy_hidden_dim"]),
            },
            "history": history,
            "manifest": run_manifest,
        },
        checkpoint_path,
    )
    base.write_candidate_rows(args.output_dir / "baseline_candidates.csv", baseline_rows)
    base.write_candidate_rows(args.output_dir / "guided_candidates.csv", guided_rows)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": run_manifest,
        "label_support": label_stats,
        "training": history,
        "calibration": calibration,
        "baseline_evaluation": baseline,
        "guided_evaluation": guided,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "decision": (
            "advance_frozen_latent_energy_to_cross_task_transfer"
            if not failures
            else "reject_property_energy_hypothesis_without_b26_retuning"
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
