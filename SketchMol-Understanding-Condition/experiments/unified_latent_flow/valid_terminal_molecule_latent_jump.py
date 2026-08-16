#!/usr/bin/env python3
"""Constrain a frozen source-conditioned latent jump process to molecule states.

B41 already provides the desired generative object: each of twenty continuous
latent particles conditions a direct sequence of atom/bond graph events.  Its
remaining failure is a state-space mismatch.  STOP is authorized by tensor
surrogates for valence and train support, but 10.1% of the resulting terminal
graphs cannot be materialized as RDKit molecules.

This experiment makes one structural change.  The frozen B41 event field may
STOP only when its *current generated graph state* materializes as a molecule.
The check is part of the transition support, before an event is sampled.  It is
not a post-hoc repair, retry, candidate selector, property oracle, or target
comparison.  Invalid partial edit sets remain latent intermediate states and
must take another learned graph jump.  Exact n=20 raw attempts and the frozen
B41 latent particles are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import viability_preserving_interacting_particle_transport as b41  # noqa: E402


b40 = b41.b40
b39 = b41.b39
b37 = b41.b37
b36 = b41.b36
base = b41.base
belief = b41.belief
delta = b41.delta
graph = b41.graph
hierarchical = b41.hierarchical
unified = b41.unified

PROTOCOL = "train_only_valid_terminal_molecule_latent_jump_v1"
B41_PREREGISTRATION = (
    SCRIPT_DIR / "viability_preserving_interacting_particle_transport_v41_preregistration.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-summary", type=Path, required=True)
    parser.add_argument("--b37-summary", type=Path, required=True)
    parser.add_argument("--b38-checkpoint", type=Path, required=True)
    parser.add_argument("--b38-summary", type=Path, required=True)
    parser.add_argument("--b39-checkpoint", type=Path, required=True)
    parser.add_argument("--b39-summary", type=Path, required=True)
    parser.add_argument("--b39-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--b40-summary", type=Path, required=True)
    parser.add_argument("--b40-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--b41-checkpoint", type=Path, required=True)
    parser.add_argument("--b41-summary", type=Path, required=True)
    parser.add_argument("--b41-evaluated-candidates", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "source_conditioned_continuous_latent_particles": True,
        "direct_atom_bond_graph_events": True,
        "exact_molecule_materialization_is_stop_support": True,
        "exact_materialization_used_for_event_scoring": False,
        "exact_materialization_used_for_property_scoring": False,
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
        "max_jumps": 64,
        "flow_steps": 8,
        "birth_capacity": 8,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Valid-terminal preregistration drift: {drift}")
    if payload.get("property_counts") != [2, 3]:
        raise ValueError("Valid-terminal property-count contract drift")
    actual = belief.file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != actual:
        raise ValueError(
            "Valid-terminal implementation drift: "
            f"expected {payload.get('implementation_sha256')}, found {actual}"
        )
    expected_inputs = {
        "b22_checkpoint_sha256",
        "b22_summary_sha256",
        "b36_summary_sha256",
        "b37_summary_sha256",
        "b38_checkpoint_sha256",
        "b38_summary_sha256",
        "b39_checkpoint_sha256",
        "b39_evaluated_candidates_sha256",
        "b39_summary_sha256",
        "b40_evaluated_candidates_sha256",
        "b40_summary_sha256",
        "b41_checkpoint_sha256",
        "b41_evaluated_candidates_sha256",
        "b41_preregistration_sha256",
        "b41_summary_sha256",
        "representation_checkpoint_sha256",
        "representation_summary_sha256",
        "train_csv_sha256",
        "validation_csv_sha256",
    }
    if set(dict(payload.get("locked_inputs", {}))) != expected_inputs:
        raise ValueError("Valid-terminal locked-input manifest is incomplete")
    return payload


def check_locked_inputs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    locked = dict(preregistration["locked_inputs"])
    paths = {
        "b22_checkpoint_sha256": args.b22_checkpoint,
        "b22_summary_sha256": args.b22_summary,
        "b36_summary_sha256": args.b36_summary,
        "b37_summary_sha256": args.b37_summary,
        "b38_checkpoint_sha256": args.b38_checkpoint,
        "b38_summary_sha256": args.b38_summary,
        "b39_checkpoint_sha256": args.b39_checkpoint,
        "b39_evaluated_candidates_sha256": args.b39_evaluated_candidates,
        "b39_summary_sha256": args.b39_summary,
        "b40_evaluated_candidates_sha256": args.b40_evaluated_candidates,
        "b40_summary_sha256": args.b40_summary,
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_evaluated_candidates_sha256": args.b41_evaluated_candidates,
        "b41_preregistration_sha256": B41_PREREGISTRATION,
        "b41_summary_sha256": args.b41_summary,
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
    }
    drift = {
        name: {"expected": locked[name], "actual": belief.file_sha256(path)}
        for name, path in paths.items()
        if belief.file_sha256(path) != locked[name]
    }
    if drift:
        raise ValueError(f"Valid-terminal locked-input drift: {drift}")

    b41_preregistration = b41.read_preregistration(B41_PREREGISTRATION)
    (
        b22_summary,
        b22_checkpoint,
        b36_summary,
        b37_summary,
        _b39_checkpoint,
        _b40_summary,
    ) = b41.check_locked_inputs(args, b41_preregistration)
    b41_summary = json.loads(args.b41_summary.read_text(encoding="utf-8"))
    if b41_summary.get("protocol") != b41.PROTOCOL:
        raise ValueError("Valid-terminal run requires the locked B41 protocol")
    if b41_summary.get("decision") != (
        "stop_and_diagnose_viability_or_particle_support_without_gate_changes"
    ):
        raise ValueError("Valid-terminal run refuses a B41 decision drift")
    manifest = dict(b41_summary.get("manifest", {}))
    for name in (
        "molecular_candidate_ranking",
        "oracle_selection",
        "retry_or_resampling",
        "posthoc_molecule_repair",
        "generation_target_access",
        "generation_property_oracle_access",
    ):
        if manifest.get(name) is not False:
            raise ValueError(f"Valid-terminal run refuses B41 contract drift: {name}")
    metrics = dict(b41_summary.get("metrics", {}))
    evidence_drift = {
        key: {"expected": expected, "actual": metrics.get(key)}
        for key, expected in dict(preregistration["b41_baseline"]).items()
        if not math.isclose(
            float(metrics.get(key, math.nan)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if evidence_drift:
        raise ValueError(f"Valid-terminal B41 baseline drift: {evidence_drift}")
    checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != b41.PROTOCOL:
        raise ValueError("Valid-terminal run refuses a non-B41 checkpoint")
    return b22_summary, b22_checkpoint, b36_summary, b37_summary, checkpoint


def materializable_terminal_states(
    source: Mapping[str, torch.Tensor],
    node_actions: torch.Tensor,
    edge_actions: torch.Tensor,
    vocabulary: Mapping[str, object],
) -> torch.Tensor:
    """Return exact RDKit materialization support for current graph states."""

    result = delta.apply_delta_actions(source, node_actions, edge_actions, vocabulary)
    prediction = {
        key: value.detach().cpu().numpy()
        for key, value in result.items()
        if isinstance(value, torch.Tensor)
    }
    valid: list[bool] = []
    for index in range(node_actions.shape[0]):
        try:
            smiles, _ = graph.graph_to_smiles(prediction, index)
            valid.append(bool(graph.canonical_smiles(smiles or "")))
        except Exception:
            valid.append(False)
    return torch.as_tensor(valid, dtype=torch.bool, device=node_actions.device)


class ExactMoleculeStopSupport:
    """Add molecule materialization to the frozen B41 STOP support."""

    def __init__(self, vocabulary: Mapping[str, object]) -> None:
        self.vocabulary = vocabulary
        self.counts: Counter[str] = Counter()
        self._base = b41.viability_event_mask

    def __call__(
        self,
        field,
        source: Mapping[str, torch.Tensor],
        node_actions: torch.Tensor,
        edge_actions: torch.Tensor,
        working: torch.Tensor,
        support: Mapping[str, object],
        support_tensors: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        legal, diagnostics = self._base(
            field,
            source,
            node_actions,
            edge_actions,
            working,
            support,
            support_tensors,
        )
        base_stop = legal[:, 0].clone()
        materializable = materializable_terminal_states(
            source, node_actions, edge_actions, self.vocabulary
        )
        legal[:, 0] &= materializable
        rejected = base_stop & ~materializable
        self.counts["state_checks"] += int(len(materializable))
        self.counts["materializable_states"] += int(materializable.sum().cpu())
        self.counts["base_stop_legal"] += int(base_stop.sum().cpu())
        self.counts["exact_stop_rejections"] += int(rejected.sum().cpu())
        revised = dict(diagnostics)
        revised["exact_molecule_materializable"] = materializable
        revised["exact_molecule_stop_rejected"] = rejected
        revised["stop_masked"] = ~legal[:, 0]
        return legal, revised

    def manifest(self) -> dict[str, object]:
        state_checks = self.counts["state_checks"]
        base_stop = self.counts["base_stop_legal"]
        return {
            **dict(self.counts),
            "materializable_state_rate": self.counts["materializable_states"]
            / max(1, state_checks),
            "exact_stop_rejection_rate_given_base_stop": self.counts[
                "exact_stop_rejections"
            ]
            / max(1, base_stop),
        }


def gate_result(
    metrics: Mapping[str, object],
    support_manifest: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    thresholds = dict(preregistration["gates"])
    baseline = dict(preregistration["b41_baseline"])
    by_count = dict(metrics["by_property_count"])
    checks = {
        "exact_attempts": {
            "value": metrics["attempted_per_condition"],
            "threshold": 20,
        },
        "validity": {"value": metrics["validity"], "threshold": thresholds["validity"]},
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": thresholds["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": thresholds["mean_source_tanimoto"],
        },
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": thresholds["strict_any20"],
        },
        "two_property_strict_any20": {
            "value": by_count["2"]["strict_any20"],
            "threshold": thresholds["two_property_strict_any20"],
        },
        "three_property_strict_any20": {
            "value": by_count["3"]["strict_any20"],
            "threshold": thresholds["three_property_strict_any20"],
        },
        "target_improvement_any20": {
            "value": metrics["target_improvement_any20"],
            "threshold": thresholds["target_improvement_any20"],
        },
        "max_horizon_hit_rate": {
            "value": metrics["max_horizon_hit_rate"],
            "threshold": thresholds["max_horizon_hit_rate"],
            "comparison": "at_most",
        },
        "validity_delta_vs_b41": {
            "value": float(metrics["validity"]) - float(baseline["validity"]),
            "threshold": thresholds["validity_delta_vs_b41"],
        },
        "strict_delta_vs_b41": {
            "value": float(metrics["strict_any20"])
            - float(baseline["strict_any20"]),
            "threshold": thresholds["strict_delta_vs_b41"],
        },
        "exact_stop_rejections": {
            "value": support_manifest["exact_stop_rejections"],
            "threshold": thresholds["exact_stop_rejections"],
        },
    }
    failures: list[str] = []
    for name, check in checks.items():
        value = float(check["value"])
        threshold = float(check["threshold"])
        if check.get("comparison") == "at_most":
            failed = value > threshold
        else:
            failed = value < threshold
        if failed:
            failures.append(name)
    return {"checks": checks, "failures": failures, "passed": not failures}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed valid-terminal result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b22_summary, b22_checkpoint, b36_summary, b37_summary, b41_checkpoint = (
        check_locked_inputs(args, preregistration)
    )

    selected_pairs, reconstruction = b36.reconstruct_b22_train_pairs(
        args, preregistration, b22_checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b37.strict_source_group_split(
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
            args.representation_checkpoint,
            args.representation_summary,
            device,
        )
    )
    vocabulary = b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
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
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)

    exact_support = ExactMoleculeStopSupport(vocabulary)
    original_support = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    try:
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
    finally:
        b41.viability_event_mask = original_support

    frozen_path = args.output_dir / "frozen_train_only_dev_candidates.csv"
    base.write_candidate_rows(frozen_path, frozen)
    frozen_sha256 = belief.file_sha256(frozen_path)
    evaluated, metrics = b41.evaluate_frozen_candidates(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    exact_support_manifest = exact_support.manifest()
    gate = gate_result(metrics, exact_support_manifest, preregistration)
    manifest = {
        "protocol": PROTOCOL,
        "seed": int(preregistration["seed"]),
        "device": str(device),
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "implementation_sha256": belief.file_sha256(Path(__file__).resolve()),
        "locked_inputs": dict(preregistration["locked_inputs"]),
        "reconstruction": reconstruction,
        "split": split,
        "representation_protocol": representation_summary.get("protocol"),
        "exact_molecule_stop_support": exact_support_manifest,
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "source_conditioned_continuous_latent_particles": True,
        "direct_atom_bond_graph_events": True,
        "exact_molecule_materialization_is_stop_support": True,
        "exact_materialization_used_for_event_scoring": False,
        "exact_materialization_used_for_property_scoring": False,
        "frozen_before_target_or_property_evaluation": True,
        "frozen_candidates_sha256": frozen_sha256,
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
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
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
        "b36_decision": b36_summary.get("decision"),
        "b37_decision": b37_summary.get("decision"),
    }
    summary = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_valid_terminal_latent_jump_to_fresh_confirmation"
            if gate["passed"]
            else "stop_valid_terminal_latent_jump_without_gate_changes"
        ),
        "b41_baseline": dict(preregistration["b41_baseline"]),
        "metrics": metrics,
        "internal_gate": gate,
        "manifest": manifest,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
