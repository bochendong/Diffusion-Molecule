#!/usr/bin/env python3
"""Source-conditioned property operator on a frozen continuous graph flow.

The universal continuous graph-delta flow and its twenty latent particles are
frozen exactly.  The only learned change is the decoder's transition effect:
instead of assigning each reaction transaction one global mean property delta,
an operator predicts the delta from both the current source graph latent and
the transaction graph latent.  This tests the failure mechanism observed in
the universal flow: structurally valid cross-task transactions had almost no
property transfer when their effect was treated as source independent.

The operator is trained only on fit source/target pairs.  During generation it
receives a source latent and fit-derived transaction latent, never a generated
molecule, development target, or property oracle.  Each of the twenty frozen
continuous states samples one complete source-applicable transaction exactly
once.  There is no molecular candidate ranking, target/oracle selection,
retry, repair, task-vocabulary fallback, or second edit.
"""

from __future__ import annotations

import argparse
import csv
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
import property_aligned_balanced_transaction_transport as balanced  # noqa: E402
import universal_continuous_graph_delta_flow as universal  # noqa: E402


atomic = vq.atomic
base = vq.base
belief = vq.belief
evidence = vq.evidence

PROTOCOL = "train_only_contextual_source_conditioned_transition_operator_v1"


class SourceConditionedEffectOperator(nn.Module):
    """Predict transaction property effects from source and action latents."""

    def __init__(
        self,
        source_dim: int,
        action_dim: int,
        property_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.source = nn.Sequential(
            nn.LayerNorm(int(source_dim)),
            nn.Linear(int(source_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.action = nn.Sequential(
            nn.LayerNorm(int(action_dim)),
            nn.Linear(int(action_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.effect = nn.Sequential(
            nn.LayerNorm(4 * int(hidden_dim)),
            nn.Linear(4 * int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(property_dim)),
        )

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        source_state = self.source(source.float())
        action_state = self.action(action.float())
        interaction = torch.cat(
            [
                source_state,
                action_state,
                source_state * action_state,
                torch.abs(source_state - action_state),
            ],
            dim=1,
        )
        return self.effect(interaction)


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
    parser.add_argument("--balanced-summary", type=Path, required=True)
    parser.add_argument("--universal-checkpoint", type=Path, required=True)
    parser.add_argument("--universal-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "single_mechanism_change_after_universal_flow": True,
        "frozen_universal_continuous_flow": True,
        "flow_training": False,
        "trainable_source_conditioned_effect_operator": True,
        "source_independent_transaction_effect": False,
        "continuous_graph_delta_latent": True,
        "discrete_vq_codebook": False,
        "universal_cross_task_transaction_grammar": True,
        "task_partitioned_transaction_support": False,
        "fit_only_effect_labels": True,
        "development_target_latent_access": False,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
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
        "primary_evaluator_semantics_match_b38": True,
        "transaction_native_diagnostics": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Contextual transition preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Contextual transition property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Contextual transition implementation drift: "
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
        "balanced_summary_sha256",
        "radius_one_support_probe_sha256",
        "radius_zero_support_probe_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "set_evidence_records_sha256",
        "set_evidence_summary_sha256",
        "train_csv_sha256",
        "universal_checkpoint_sha256",
        "universal_summary_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Contextual transition locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object], device: torch.device
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    b22_summary, b22_checkpoint, balanced_summary = universal.check_locked_inputs(
        args, preregistration
    )
    locked = dict(preregistration["locked_inputs"])
    drift = {}
    for name, path in (
        ("universal_checkpoint_sha256", args.universal_checkpoint),
        ("universal_summary_sha256", args.universal_summary),
    ):
        actual = belief.file_sha256(path)
        if actual != locked[name]:
            drift[name] = {"expected": locked[name], "actual": actual}
    if drift:
        raise ValueError(f"Contextual transition universal-input drift: {drift}")
    universal_summary = json.loads(args.universal_summary.read_text(encoding="utf-8"))
    if universal_summary.get("protocol") != universal.PROTOCOL:
        raise ValueError("Contextual transition requires the universal-flow protocol")
    if universal_summary.get("decision") != (
        "stop_universal_continuous_graph_delta_flow_without_gate_changes"
    ):
        raise ValueError("Contextual transition refuses a universal-flow decision drift")
    metrics = dict(universal_summary.get("metrics", {}))
    baseline_drift = {
        key: {"expected": expected, "actual": metrics.get(key)}
        for key, expected in dict(preregistration["universal_baseline"]).items()
        if not math.isclose(
            float(metrics.get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if baseline_drift:
        raise ValueError(f"Contextual transition baseline drift: {baseline_drift}")
    checkpoint = torch.load(
        args.universal_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("stage") != universal.PROTOCOL:
        raise ValueError("Contextual transition refuses a non-universal checkpoint")
    config = dict(checkpoint["model_config"])
    flow = universal.ContinuousGraphDeltaFlow(
        source_dim=int(config["source_dim"]),
        request_dim=int(config["request_dim"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    flow.load_state_dict(checkpoint["model_state"])
    flow.eval().requires_grad_(False)
    checkpoint["loaded_flow"] = flow
    return b22_summary, b22_checkpoint, balanced_summary, checkpoint


def checkpoint_transactions(
    checkpoint: Mapping[str, object], preregistration: Mapping[str, object]
) -> tuple[list[universal.UniversalClosedTransaction], dict[str, np.ndarray], list[str]]:
    transactions = [
        universal.UniversalClosedTransaction(
            reaction_smarts=tuple(row["reaction_smarts"]),
            fit_source_smiles=str(row["fit_source_smiles"]),
            fit_target_smiles=str(row["fit_target_smiles"]),
            component_count=int(row["component_count"]),
            origin_tasks=tuple(row["origin_tasks"]),
        )
        for row in checkpoint["transaction_catalog"]
    ]
    embeddings = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in dict(checkpoint["transaction_embeddings"]).items()
    }
    vocabulary = [str(value) for value in checkpoint["property_vocabulary"]]
    graph_dimensions = int(preregistration["graph_delta_dimensions"])
    expected = graph_dimensions + len(vocabulary)
    if not embeddings or any(value.shape != (expected,) for value in embeddings.values()):
        raise ValueError("Contextual transition embedding dimension drift")
    catalog_keys = {
        universal.universal_transaction_key(transaction.reaction_smarts)
        for transaction in transactions
    }
    if catalog_keys != set(embeddings):
        raise ValueError("Contextual transition catalog/embedding key drift")
    return transactions, embeddings, vocabulary


def effect_training_data(
    examples: Sequence[tuple[object, str]],
    source_latents: np.ndarray,
    transaction_embeddings: Mapping[str, np.ndarray],
    property_vocabulary: Sequence[str],
    preregistration: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    graph_dimensions = int(preregistration["graph_delta_dimensions"])
    index = {prop: position for position, prop in enumerate(property_vocabulary)}
    actions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    requests: list[np.ndarray] = []
    attempted = 0
    evaluated_count = 0
    for pair, key in examples:
        effect, observed, pair_attempted, pair_evaluated = balanced.fit_property_effect(
            pair,
            property_vocabulary,
            float(preregistration["property_delta_clip"]),
        )
        task_mask = np.zeros(len(property_vocabulary), dtype=np.float32)
        for prop, _direction in base.task_specs(pair.row):
            task_mask[index[prop]] = 1.0
        actions.append(transaction_embeddings[key][:graph_dimensions])
        labels.append(effect)
        masks.append(observed * task_mask)
        requests.append(balanced.property_request_vector(pair, property_vocabulary))
        attempted += pair_attempted
        evaluated_count += pair_evaluated
    manifest = {
        "training_examples": len(examples),
        "attempted_fit_effect_labels": attempted,
        "evaluated_fit_effect_labels": evaluated_count,
        "fit_effect_label_coverage": evaluated_count / max(1, attempted),
        "source_conditioned_effect_labels": True,
        "development_property_access": False,
    }
    return (
        source_latents.astype(np.float32),
        np.stack(actions).astype(np.float32),
        np.stack(labels).astype(np.float32),
        np.stack(masks).astype(np.float32),
        np.stack(requests).astype(np.float32),
        manifest,
    )


def masked_effect_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    request: torch.Tensor,
    preregistration: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    squared = (predicted - target).square() * mask
    regression = squared.sum() / mask.sum().clamp_min(1.0)
    predicted_margin = (predicted * request).sum(dim=1)
    target_margin = (target * request).sum(dim=1)
    direction = F.mse_loss(predicted_margin, target_margin)
    loss = regression + float(preregistration["effect_direction_weight"]) * direction
    return loss, {
        "masked_effect_mse": regression,
        "requested_margin_mse": direction,
    }


def train_effect_operator(
    model: SourceConditionedEffectOperator,
    sources: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    requests: np.ndarray,
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    source_tensor = torch.from_numpy(sources)
    action_tensor = torch.from_numpy(actions)
    label_tensor = torch.from_numpy(labels)
    mask_tensor = torch.from_numpy(masks)
    request_tensor = torch.from_numpy(requests)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    batch_size = int(preregistration["batch_size"])
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(labels)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.train()
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            source = source_tensor[indices].to(device)
            action = action_tensor[indices].to(device)
            target = label_tensor[indices].to(device)
            mask = mask_tensor[indices].to(device)
            request = request_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(source, action)
            loss, metrics = masked_effect_loss(
                predicted, target, mask, request, preregistration
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite contextual effect loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(preregistration["grad_clip"]))
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in metrics.items():
                totals[name] += float(value.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "training_examples": len(labels),
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    model.eval().requires_grad_(False)
    with torch.no_grad():
        predicted = model(source_tensor.to(device), action_tensor.to(device)).cpu()
    observed = mask_tensor.bool()
    absolute = torch.abs(predicted - label_tensor)
    sign_equal = torch.sign(predicted).eq(torch.sign(label_tensor)) & observed
    requested_margin = (predicted * request_tensor).sum(dim=1)
    target_margin = (label_tensor * request_tensor).sum(dim=1)
    calibration = {
        "fit_masked_mae": float(absolute[observed].mean()),
        "fit_effect_sign_accuracy": float(sign_equal.sum() / observed.sum().clamp_min(1)),
        "fit_requested_direction_accuracy": float(
            (requested_margin > 0).float().mean()
        ),
        "fit_target_requested_direction_accuracy": float(
            (target_margin > 0).float().mean()
        ),
    }
    return history, calibration


@torch.no_grad()
def freeze_candidates(
    flow: universal.ContinuousGraphDeltaFlow,
    operator: SourceConditionedEffectOperator,
    development_pairs: Sequence[object],
    development_source_latents: np.ndarray,
    transactions: Sequence[universal.UniversalClosedTransaction],
    transaction_embeddings: Mapping[str, np.ndarray],
    property_vocabulary: Sequence[str],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    support_counts: list[int] = []
    family_counts: list[int] = []
    sampled_family_counts: list[int] = []
    latent_pairwise: list[float] = []
    predicted_margins: list[float] = []
    cross_task_attempts = 0
    identity_attempts = 0
    graph_dimensions = int(preregistration["graph_delta_dimensions"])
    for pair_index, pair in enumerate(development_pairs):
        task = base.task_key(pair.row)
        actions = universal.universal_applicable_actions(
            pair.source_smiles, transactions, preregistration
        )
        support_counts.append(len(actions))
        family_counts.append(len({action.transaction_key for action in actions}))
        condition_id = f"train_only_dev_{pair_index:04d}"
        request = balanced.property_request_vector(pair, property_vocabulary)
        latent = universal.transport_particles(
            flow,
            development_source_latents[pair_index],
            request,
            preregistration,
            device,
        )
        if len(latent) > 1:
            latent_pairwise.append(float(torch.pdist(latent).mean().cpu()))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(preregistration["seed"]) * 100000 + pair_index)
        sampled: list[tuple[universal.UniversalApplicableAction, float, float]] = []
        if actions:
            unique_keys = sorted({action.transaction_key for action in actions})
            action_graph = torch.from_numpy(
                np.stack(
                    [transaction_embeddings[key][:graph_dimensions] for key in unique_keys]
                )
            ).to(device)
            source = torch.from_numpy(development_source_latents[pair_index])[None, :]
            source = source.to(device).expand(len(unique_keys), -1)
            predicted_effect = operator(source, action_graph)
            effect_by_key = {
                key: predicted_effect[index] for index, key in enumerate(unique_keys)
            }
            action_embeddings = torch.stack(
                [
                    torch.cat(
                        [
                            torch.from_numpy(
                                transaction_embeddings[action.transaction_key][
                                    :graph_dimensions
                                ]
                            ).to(device),
                            effect_by_key[action.transaction_key],
                        ]
                    )
                    for action in actions
                ]
            )
            request_tensor = torch.from_numpy(request).to(device)
            for particle in latent:
                graph_distance = (
                    particle[:graph_dimensions][None, :]
                    - action_embeddings[:, :graph_dimensions]
                ).square().mean(dim=1)
                property_distance = (
                    particle[graph_dimensions:][None, :]
                    - action_embeddings[:, graph_dimensions:]
                ).square().mean(dim=1)
                energy = (
                    float(preregistration["decoder_graph_distance_weight"])
                    * graph_distance
                    + float(preregistration["decoder_property_distance_weight"])
                    * property_distance
                )
                probability = torch.softmax(
                    -energy / float(preregistration["decoder_temperature"]), dim=0
                )
                action_index = int(
                    torch.multinomial(
                        probability.cpu(), 1, generator=generator
                    ).item()
                )
                action = actions[action_index]
                margin = float(
                    (effect_by_key[action.transaction_key] * request_tensor).sum().cpu()
                )
                sampled.append(
                    (action, float(probability[action_index].cpu()), margin)
                )
                predicted_margins.append(margin)
            sampled_family_counts.append(
                len({action.transaction_key for action, _prob, _margin in sampled})
            )
        else:
            sampled_family_counts.append(0)

        for attempt in range(1, 21):
            if sampled:
                action, decoder_probability, predicted_margin = sampled[attempt - 1]
                smiles = action.smiles
                transaction_key_value = action.transaction_key
                component_count = action.component_count
                fit_source = action.fit_source_smiles
                origin_tasks = action.origin_tasks
                identity = False
                cross_task = task not in origin_tasks
                cross_task_attempts += int(cross_task)
            else:
                smiles = pair.source_smiles
                transaction_key_value = "identity_no_applicable_transaction"
                component_count = 0
                fit_source = ""
                origin_tasks = ()
                identity = True
                cross_task = False
                decoder_probability = 1.0
                predicted_margin = 0.0
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
                    "unique_transaction_family_support": len(
                        {action.transaction_key for action in actions}
                    ),
                    "decoder_probability": decoder_probability,
                    "predicted_requested_margin": predicted_margin,
                    "transaction_key": transaction_key_value,
                    "transaction_components": component_count,
                    "fit_source_smiles": fit_source,
                    "origin_tasks": "|".join(origin_tasks),
                    "cross_task_transaction": cross_task,
                    "identity_transaction": identity,
                    "latent_norm": float(latent[attempt - 1].norm().cpu()),
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
                        "stage": "freeze_contextual_source_conditioned_operator",
                        "conditions": pair_index + 1,
                        "raw_rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    support = {
        "conditions": len(development_pairs),
        "conditions_with_nonidentity_support": sum(count > 0 for count in support_counts),
        "condition_support_rate": sum(count > 0 for count in support_counts)
        / max(1, len(support_counts)),
        "mean_applicable_actions": float(np.mean(support_counts)),
        "mean_applicable_transaction_families": float(np.mean(family_counts)),
        "mean_sampled_transaction_families": float(np.mean(sampled_family_counts)),
        "mean_continuous_latent_pairwise_distance": float(np.mean(latent_pairwise)),
        "mean_predicted_requested_margin": float(np.mean(predicted_margins)),
        "positive_predicted_requested_margin_rate": sum(
            value > 0 for value in predicted_margins
        )
        / max(1, len(predicted_margins)),
        "cross_task_attempt_rate": cross_task_attempts / max(1, len(rows)),
        "identity_attempt_rate": identity_attempts / max(1, len(rows)),
    }
    return rows, support


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gate_result(
    metrics: Mapping[str, object],
    training_manifest: Mapping[str, object],
    calibration: Mapping[str, object],
    support: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    thresholds = dict(preregistration["gates"])
    by_count = dict(metrics["by_property_count"])
    balanced_baseline = dict(preregistration["balanced_baseline"])
    universal_baseline = dict(preregistration["universal_baseline"])
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
        "fit_effect_label_coverage": {"value": training_manifest["fit_effect_label_coverage"], "threshold": thresholds["fit_effect_label_coverage"]},
        "fit_requested_direction_accuracy": {"value": calibration["fit_requested_direction_accuracy"], "threshold": thresholds["fit_requested_direction_accuracy"]},
        "condition_support_rate": {"value": support["condition_support_rate"], "threshold": thresholds["condition_support_rate"]},
        "positive_predicted_requested_margin_rate": {"value": support["positive_predicted_requested_margin_rate"], "threshold": thresholds["positive_predicted_requested_margin_rate"]},
        "strict_delta_vs_universal": {"value": float(metrics["strict_any20"]) - float(universal_baseline["strict_any20"]), "threshold": thresholds["strict_delta_vs_universal"]},
        "strict_delta_vs_balanced": {"value": float(metrics["strict_any20"]) - float(balanced_baseline["strict_any20"]), "threshold": thresholds["strict_delta_vs_balanced"]},
        "target_delta_vs_balanced": {"value": float(metrics["target_improvement_any20"]) - float(balanced_baseline["target_improvement_any20"]), "threshold": thresholds["target_delta_vs_balanced"]},
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
        raise ValueError(f"Completed contextual transition result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = torch.device(str(args.device))
    if device.type != "cpu":
        raise ValueError("The contextual transition signal is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, _balanced_summary, universal_checkpoint = (
        check_locked_inputs(args, preregistration, device)
    )
    flow = universal_checkpoint.pop("loaded_flow")
    transactions, transaction_embeddings, property_vocabulary = (
        checkpoint_transactions(universal_checkpoint, preregistration)
    )
    representation, representation_config, representation_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    selected_pairs, reconstruction = evidence.reconstruct_locked_b36_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = vq.b43.b41.b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    rebuilt_transactions, training_examples, grammar_manifest = (
        universal.build_universal_fit_grammar(fit_pairs, preregistration)
    )
    rebuilt_keys = {
        universal.universal_transaction_key(transaction.reaction_smarts)
        for transaction in rebuilt_transactions
    }
    checkpoint_keys = {
        universal.universal_transaction_key(transaction.reaction_smarts)
        for transaction in transactions
    }
    if rebuilt_keys != checkpoint_keys:
        raise ValueError("Contextual transition rebuilt grammar drift")
    fit_source_latents, _fit_graph_deltas = universal.encode_fit_graph_latents(
        representation,
        training_examples,
        int(preregistration["encoding_batch_size"]),
        device,
    )
    sources, actions, labels, masks, requests, effect_manifest = effect_training_data(
        training_examples,
        fit_source_latents,
        transaction_embeddings,
        property_vocabulary,
        preregistration,
    )
    operator = SourceConditionedEffectOperator(
        source_dim=sources.shape[1],
        action_dim=actions.shape[1],
        property_dim=labels.shape[1],
        hidden_dim=int(preregistration["operator_hidden_dim"]),
    ).to(device)
    history, calibration = train_effect_operator(
        operator,
        sources,
        actions,
        labels,
        masks,
        requests,
        preregistration,
        device,
    )
    checkpoint_path = args.output_dir / "contextual_source_conditioned_transition_operator.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "operator_state": operator.state_dict(),
            "operator_config": {
                "source_dim": sources.shape[1],
                "action_dim": actions.shape[1],
                "property_dim": labels.shape[1],
                "hidden_dim": int(preregistration["operator_hidden_dim"]),
            },
            "frozen_universal_checkpoint_sha256": belief.file_sha256(
                args.universal_checkpoint
            ),
            "property_vocabulary": property_vocabulary,
            "calibration": calibration,
        },
        checkpoint_path,
    )
    development_source_latents = universal.encode_development_sources(
        representation,
        development_pairs,
        int(preregistration["encoding_batch_size"]),
        device,
    )
    frozen, development_support = freeze_candidates(
        flow,
        operator,
        development_pairs,
        development_source_latents,
        transactions,
        transaction_embeddings,
        property_vocabulary,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_transactions.csv"
    write_rows(frozen_path, frozen)
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = vq.evaluate_frozen_transactions(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_transactions.csv"
    write_rows(evaluated_path, evaluated)
    gate = gate_result(
        metrics,
        effect_manifest,
        calibration,
        development_support,
        preregistration,
    )
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "representation_config": representation_config,
        "representation_gate_passed": bool(dict(representation_summary.get("gate", {})).get("passed")),
        "reconstruction": reconstruction,
        "split": split,
        "fit_grammar": grammar_manifest,
        "effect_training": effect_manifest,
        "effect_calibration": calibration,
        "development_support": development_support,
        "single_mechanism_change_after_universal_flow": True,
        "frozen_universal_continuous_flow": True,
        "flow_training": False,
        "trainable_source_conditioned_effect_operator": True,
        "source_independent_transaction_effect": False,
        "continuous_graph_delta_latent": True,
        "discrete_vq_codebook": False,
        "universal_cross_task_transaction_grammar": True,
        "task_partitioned_transaction_support": False,
        "fit_only_effect_labels": True,
        "development_target_latent_access": False,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "single_complete_transaction_per_attempt": True,
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
            "advance_contextual_transition_operator_to_fresh_confirmation"
            if gate["passed"]
            else "stop_contextual_transition_operator_without_gate_changes"
        ),
        "training": history,
        "universal_baseline": dict(preregistration["universal_baseline"]),
        "balanced_baseline": dict(preregistration["balanced_baseline"]),
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
