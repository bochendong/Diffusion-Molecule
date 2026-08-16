#!/usr/bin/env python3
"""Train an exact-20 set-aware transport over atomic graph transactions.

The representation gate established that successful train-only edits can be
committed as complete, valence-closed graph rewrite transactions and that their
compositional grammar covers a source-grouped meta split.  This stage tests the
remaining model hypothesis: train all twenty latent particles jointly so their
next-event distributions cover different valid parts of the same orderless
rewrite program.

The B41 transport is used only as a frozen warm start.  Its event kernel is
fine-tuned with a set loss over twenty orthogonal latent particles per source:
best-set likelihood, all-particle participation, valid-next-event coverage, and
target-supported diversity.  Generation still emits exactly twenty direct
attempts and commits each complete transaction once.  There is no larger pool,
ranking, retry, oracle selection, target access, or molecule repair.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
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

import set_closed_graph_rewrite_evidence as evidence  # noqa: E402
import viability_preserving_interacting_particle_transport as b41  # noqa: E402


base = b41.base
belief = b41.belief
delta = b41.delta
full_graph = b41.full_graph
graph = b41.graph
hierarchical = b41.hierarchical
unified = b41.unified

PROTOCOL = "train_only_set_closed_graph_transport_v1"


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
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "set_closed_representation_gate_required": True,
        "warm_start_from_frozen_b41": True,
        "atomic_transaction_commit": True,
        "joint_set_training": True,
        "single_particle_training_loss": False,
        "particle_pool_size": 20,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "official_test_access": False,
        "development_source_limit": 160,
        "set_particles": 20,
        "epochs": 1,
        "fit_pair_limit": 512,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Set-closed transport preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Set-closed transport property-count contract drift")
    implementation_sha256 = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation_sha256:
        raise ValueError(
            "Set-closed transport implementation drift: "
            f"expected {payload.get('implementation_sha256')}, "
            f"found {implementation_sha256}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_records_sha256",
        "b41_checkpoint_sha256",
        "b41_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "set_evidence_records_sha256",
        "set_evidence_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Set-closed transport locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_records_sha256": args.b36_records,
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_summary_sha256": args.b41_summary,
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
        raise ValueError(f"Set-closed transport locked input drift: {drift}")

    b22_summary, b22_checkpoint = evidence.patch_evidence.load_locked_b22(
        args, preregistration
    )
    set_summary = json.loads(args.set_evidence_summary.read_text(encoding="utf-8"))
    if set_summary.get("protocol") != evidence.PROTOCOL:
        raise ValueError("Set-closed transport requires the representation protocol")
    if not bool(dict(set_summary.get("gate", {})).get("passed")):
        raise ValueError("Set-closed transport requires a passing representation gate")
    if set_summary.get("decision") != "train_set_closed_graph_transport_exact_n20":
        raise ValueError("Set-closed transport refuses representation decision drift")

    b41_summary = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    if b41_summary.get("protocol") != b41.PROTOCOL:
        raise ValueError("Set-closed transport requires the frozen B41 protocol")
    if b41_summary.get("decision") != (
        "stop_and_diagnose_viability_or_particle_support_without_gate_changes"
    ):
        raise ValueError("Set-closed transport refuses B41 decision drift")
    baseline = dict(b41_summary.get("metrics", {}))
    baseline_drift = {
        key: {"expected": expected, "actual": baseline.get(key)}
        for key, expected in dict(preregistration["b41_baseline"]).items()
        if not math.isclose(
            float(baseline.get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if baseline_drift:
        raise ValueError(f"Set-closed transport B41 baseline drift: {baseline_drift}")
    b41_checkpoint = torch.load(
        args.b41_checkpoint, map_location="cpu", weights_only=False
    )
    if b41_checkpoint.get("stage") != b41.PROTOCOL:
        raise ValueError("Set-closed transport refuses a non-B41 checkpoint")
    return b22_summary, b22_checkpoint, b41_summary, b41_checkpoint


def repeat_batch(
    values: Mapping[str, object], repeats: int
) -> dict[str, object]:
    return {
        key: value.repeat_interleave(int(repeats), dim=0)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in values.items()
    }


def orthogonal_queries(
    attempts: int, dimension: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(dimension, attempts, generator=generator)
    queries = torch.linalg.qr(matrix, mode="reduced").Q.transpose(0, 1)
    return queries.to(device=device, dtype=torch.float32)


def per_particle_orderless_loss(
    logits: torch.Tensor, legal: torch.Tensor, target_next: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool((target_next & ~legal).any()):
        bad = torch.nonzero(target_next & ~legal, as_tuple=False)[:8].tolist()
        raise ValueError(f"Set-closed target-next events outside support: {bad}")
    legal_logits = logits.float().masked_fill(~legal, -torch.inf)
    target_logits = logits.float().masked_fill(~target_next, -torch.inf)
    log_partition = torch.logsumexp(legal_logits, dim=1)
    log_target_mass = torch.logsumexp(target_logits, dim=1)
    return log_partition - log_target_mass, torch.softmax(legal_logits, dim=1)


def target_mode_coverage_loss(
    probabilities: torch.Tensor,
    target_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward the 20-particle set for covering every valid next rewrite event."""

    target = target_next[:, 0, :]
    best_probability = probabilities.max(dim=1).values
    coverage = -torch.log(best_probability.clamp_min(1e-8))[target].mean()

    target_probabilities = probabilities * target_next.float()
    target_probabilities = target_probabilities / target_probabilities.sum(
        dim=2, keepdim=True
    ).clamp_min(1e-8)
    normalized = F.normalize(target_probabilities, dim=2)
    cosine = normalized @ normalized.transpose(1, 2)
    off_diagonal = ~torch.eye(
        probabilities.shape[1], dtype=torch.bool, device=probabilities.device
    )
    multi_target = target.sum(dim=1).gt(1)
    if bool(multi_target.any()):
        diversity = cosine[multi_target][:, off_diagonal].mean()
    else:
        diversity = cosine.sum() * 0.0
    return coverage, diversity


def train_joint_set_kernel(
    model: nn.Module,
    representation: nn.Module,
    fit_pairs: Sequence[object],
    vocabulary: Mapping[str, object],
    support: Mapping[str, object],
    support_tensors: Mapping[str, torch.Tensor],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> list[dict[str, float]]:
    model.requires_grad_(False)
    model.denoiser.requires_grad_(True)
    representation.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.denoiser.parameters(),
        lr=float(preregistration["learning_rate"]),
        weight_decay=float(preregistration["weight_decay"]),
    )
    attempts = int(preregistration["set_particles"])
    queries = orthogonal_queries(
        attempts,
        model.transport_dim,
        seed=int(preregistration["set_query_seed"]),
        device=device,
    )
    query_scale = math.sqrt(model.transport_dim) * float(
        preregistration["set_latent_std"]
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    batch_size = int(preregistration["set_batch_size"])
    selected_fit = list(fit_pairs[: int(preregistration["fit_pair_limit"])])
    history: list[dict[str, float]] = []
    global_batch = 0
    for epoch in range(1, int(preregistration["epochs"]) + 1):
        order = list(range(len(selected_fit)))
        random.Random(int(preregistration["seed"]) + epoch).shuffle(order)
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        model.denoiser.train()
        for start in range(0, len(order), batch_size):
            items = [selected_fit[index] for index in order[start : start + batch_size]]
            collated = base.pair_collate(items)
            source = base.move_graph_batch(collated["source"], device)
            target = base.move_graph_batch(collated["target"], device)
            tokens = collated["condition"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                source_node, source_edge = representation.encode(source)
                target_node, target_edge = representation.encode(target)
                condition = model.route_condition(tokens)
                endpoint = model.posterior_endpoint(
                    source,
                    target,
                    source_node,
                    source_edge,
                    target_node,
                    target_edge,
                    condition,
                ).float()
                particle_latent = endpoint[:, None, :] + query_scale * queries[None, :, :]
                particle_latent = particle_latent.flatten(0, 1)
                node_targets, edge_targets = delta.delta_action_targets(
                    source, target, vocabulary
                )
                working = full_graph.working_node_mask(
                    source["node_mask"],
                    int(preregistration["birth_capacity"]),
                    target["node_mask"],
                )
                (
                    current_node,
                    current_edge,
                    target_next,
                    jump_time,
                    target_count,
                    executed_count,
                ) = b41.build_viable_prefix_batch(
                    node_targets,
                    edge_targets,
                    model.denoiser.layout,
                    epoch=epoch,
                    global_batch=global_batch,
                    preregistration=preregistration,
                    device=device,
                )
                current_node = current_node.repeat_interleave(attempts, dim=0)
                current_edge = current_edge.repeat_interleave(attempts, dim=0)
                expanded_target_next = target_next.repeat_interleave(attempts, dim=0)
                jump_time = jump_time.repeat_interleave(attempts, dim=0)
                executed_count = executed_count.repeat_interleave(attempts, dim=0)
                expanded_source = repeat_batch(source, attempts)
                expanded_working = working.repeat_interleave(attempts, dim=0)
                expanded_source_node = source_node.repeat_interleave(attempts, dim=0)
                expanded_source_edge = source_edge.repeat_interleave(attempts, dim=0)
                expanded_condition = condition.repeat_interleave(attempts, dim=0)
                count_logits = model.cardinality_logits(
                    expanded_source_node,
                    expanded_source["node_mask"].bool(),
                    expanded_condition,
                    particle_latent,
                )
                count_values = torch.arange(
                    int(preregistration["max_jumps"]) + 1,
                    device=device,
                    dtype=torch.float32,
                )
                expected_count = (
                    count_logits.float().softmax(dim=1) * count_values[None, :]
                ).sum(dim=1)
                remaining_mass = (expected_count - executed_count) / float(
                    preregistration["max_jumps"]
                )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                logits = model.denoiser(
                    current_node,
                    current_edge,
                    expanded_source_node,
                    expanded_source_edge,
                    expanded_source["node_mask"].bool(),
                    expanded_working,
                    jump_time,
                    expanded_condition,
                    particle_latent,
                    remaining_mass,
                )
                legal, _ = b41.viability_event_mask(
                    model.denoiser,
                    expanded_source,
                    current_node,
                    current_edge,
                    expanded_working,
                    support,
                    support_tensors,
                )
                row_loss, probability = per_particle_orderless_loss(
                    logits, legal, expanded_target_next
                )
                pair_count = len(items)
                particle_loss = row_loss.view(pair_count, attempts)
                temperature = float(preregistration["set_softmin_temperature"])
                best_set_loss = (
                    -temperature
                    * torch.logsumexp(-particle_loss / temperature, dim=1)
                    + temperature * math.log(attempts)
                ).mean()
                participation_loss = particle_loss.mean()
                probability = probability.view(pair_count, attempts, -1)
                target_next_set = target_next[:, None, :].expand(-1, attempts, -1)
                mode_coverage, diversity = target_mode_coverage_loss(
                    probability, target_next_set
                )
                target_logits = torch.logsumexp(
                    logits.float().masked_fill(~expanded_target_next, -torch.inf),
                    dim=1,
                )
                incomplete = ~expanded_target_next[:, 0]
                if bool(incomplete.any()):
                    stop_margin = F.relu(
                        float(preregistration["stop_margin"])
                        + logits.float()[:, 0]
                        - target_logits
                    )[incomplete].mean()
                else:
                    stop_margin = logits.float().sum() * 0.0
                loss = (
                    best_set_loss
                    + float(preregistration["participation_weight"])
                    * participation_loss
                    + float(preregistration["target_mode_coverage_weight"])
                    * mode_coverage
                    + float(preregistration["set_diversity_weight"]) * diversity
                    + float(preregistration["stop_margin_weight"]) * stop_margin
                )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.denoiser.parameters(), float(preregistration["grad_clip"])
            )
            optimizer.step()
            target_mass = torch.exp(-row_loss.detach()).clamp(0.0, 1.0)
            totals["loss"] += float(loss.detach())
            totals["best_set_nll"] += float(best_set_loss.detach())
            totals["participation_nll"] += float(participation_loss.detach())
            totals["target_mode_coverage_loss"] += float(mode_coverage.detach())
            totals["set_diversity_cosine"] += float(diversity.detach())
            totals["stop_margin_loss"] += float(stop_margin.detach())
            totals["mean_target_events"] += float(target_count.float().mean())
            totals["mean_particle_target_mass"] += float(target_mass.mean())
            batches += 1
            global_batch += 1
        row = {
            "epoch": epoch,
            "fit_pairs": len(selected_fit),
            "set_particles": attempts,
            **{name: value / max(1, batches) for name, value in totals.items()},
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"Set-closed non-finite training metrics: {row}")
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    model.eval().requires_grad_(False)
    return history


def build_model(
    representation_config: Mapping[str, object],
    vocabulary: Mapping[str, object],
    preregistration: Mapping[str, object],
    device: torch.device,
) -> nn.Module:
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    return b41.b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(preregistration["condition_dim"]),
        transport_dim=int(preregistration["transport_dim"]),
        hidden_dim=int(preregistration["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(preregistration["max_jumps"]),
        property_count=len(unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(preregistration["message_layers"]),
    ).to(device)


def gate_with_baseline(
    metrics: Mapping[str, object],
    baseline: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    gate = b41.b38.gate(metrics, dict(preregistration["gates"]))
    delta_checks = {
        "validity_delta_vs_b41": {
            "value": float(metrics["validity"]) - float(baseline["validity"]),
            "threshold": float(preregistration["delta_gates"]["validity"]),
        },
        "unique_valid_delta_vs_b41": {
            "value": float(metrics["mean_unique_valid"])
            - float(baseline["mean_unique_valid"]),
            "threshold": float(preregistration["delta_gates"]["mean_unique_valid"]),
        },
        "strict_delta_vs_b41": {
            "value": float(metrics["strict_any20"])
            - float(baseline["strict_any20"]),
            "threshold": float(preregistration["delta_gates"]["strict_any20"]),
        },
    }
    for name, check in delta_checks.items():
        gate["checks"][name] = check
        if float(check["value"]) < float(check["threshold"]):
            gate["failures"].append(name)
    gate["failures"] = sorted(set(gate["failures"]))
    gate["passed"] = not gate["failures"]
    return gate


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed set-closed transport exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, b41_summary, b41_checkpoint = check_locked_inputs(
        args, preregistration
    )
    selected_pairs, reconstruction = evidence.reconstruct_locked_b36_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b41.b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    for pair in [*fit_pairs, *development_pairs]:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(preregistration["condition_dim"])
        )
    representation, representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    vocabulary = b41.b37.checkpoint_vocabulary(b22_checkpoint)
    support = b41.b40.build_support(fit_pairs, vocabulary)
    support_tensors = b41.b40._device_support(support, device)
    model = build_model(representation_config, vocabulary, preregistration, device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    replay = b41.support_replay_gate(
        model,
        fit_pairs,
        vocabulary,
        support,
        support_tensors,
        preregistration,
        device,
    )
    print(json.dumps({"stage": "support_replay_gate", **replay}, sort_keys=True), flush=True)
    if not bool(replay["passed"]):
        raise ValueError(f"Set-closed support replay gate failed: {replay}")
    history = train_joint_set_kernel(
        model,
        representation,
        fit_pairs,
        vocabulary,
        support,
        support_tensors,
        preregistration,
        device,
    )
    training_manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "representation_protocol": representation_summary.get("protocol"),
        "reconstruction": reconstruction,
        "split": split,
        "support_replay_gate": replay,
        "warm_start_from_frozen_b41": True,
        "atomic_transaction_commit": True,
        "joint_set_training": True,
        "single_particle_training_loss": False,
        "set_particles": 20,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "exact_raw_attempts_per_condition": 20,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
    }
    checkpoint_path = args.output_dir / "set_closed_graph_transport.pt"
    torch.save(
        {
            "stage": PROTOCOL,
            "model_state": model.state_dict(),
            "vocabulary": dict(b41_checkpoint["vocabulary"]),
            "history": history,
            "manifest": training_manifest,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = belief.file_sha256(checkpoint_path)
    print(
        json.dumps(
            {
                "stage": "checkpoint_frozen_before_generation",
                "checkpoint": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frozen = b41.freeze_candidates(
        model,
        representation,
        vocabulary,
        support,
        support_tensors,
        development_pairs,
        preregistration,
        device,
    )
    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, frozen)
    evaluated, metrics = b41.evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    baseline = dict(b41_summary["metrics"])
    internal_gate = gate_with_baseline(metrics, baseline, preregistration)
    manifest = {
        **training_manifest,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_candidates_sha256": belief.file_sha256(frozen_path),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "post_freeze_train_only_dev_target_access": True,
        "b41_decision": b41_summary.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "manifest": manifest,
        "training": history,
        "baseline_b41": baseline,
        "metrics": metrics,
        "internal_gate": internal_gate,
        "decision": (
            "advance_set_closed_transport_to_once_only_prospective_confirmation"
            if internal_gate["passed"]
            else "stop_and_diagnose_set_coverage_or_closed_support_without_gate_changes"
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
