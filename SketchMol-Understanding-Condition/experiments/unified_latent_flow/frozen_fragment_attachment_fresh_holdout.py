#!/usr/bin/env python3
"""Evaluate the frozen B24 fragment kernel once on an untouched heldout split.

The split and all gates are read from a committed preregistration manifest.
The B24 weights, fragment vocabulary, representation, and generation policy are
frozen.  Historical validation selections and the reconstructed B24 training
selection are excluded before the fresh conditions are sampled.  Evaluation
targets and property scorers are opened only after exactly 20 raw attempts per
condition have been frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


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

PROTOCOL = "frozen_fragment_attachment_fresh_holdout_v26"


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
        "prior_validation_selection_seeds": [1742, 2719],
        "fresh_validation_selection_seed": 4099,
        "validation_limit": 30,
        "minimum_conditions": 20,
        "property_counts": [2, 3],
        "num_attempts": 20,
        "evaluation_target_access": False,
        "official_test_access": False,
        "model_training": False,
        "hyperparameter_search": False,
        "repeat_after_scientific_failure": False,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise ValueError(f"Fresh-heldout preregistration drift: {drift}")
    gates = dict(payload.get("gates", {}))
    expected_gates = {
        "validity": 0.95,
        "strict_any20": 0.65,
        "two_property_strict_any20": 0.80,
        "three_property_strict_any20": 0.50,
        "mean_unique_valid": 12.0,
        "mean_source_tanimoto": 0.40,
        "minimum_conditions_per_property_count": 5,
    }
    if gates != expected_gates:
        raise ValueError(
            f"Fresh-heldout gate drift: expected {expected_gates}, found {gates}"
        )
    return payload


def pair_sets(pairs: Sequence[object]) -> tuple[set[str], set[tuple[str, str]]]:
    return (
        {pair.source_smiles for pair in pairs},
        {(pair.source_smiles, pair.target_smiles) for pair in pairs},
    )


def build_pairs(
    rows: Sequence[Mapping[str, str]],
    *,
    config: SimpleNamespace,
    limit: int,
    seed: int,
    forbidden_sources: set[str] | None = None,
    forbidden_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[object], dict[str, int]]:
    return base.build_pairs(
        rows,
        max_atoms=64,
        fingerprint_bits=int(config.graph_fingerprint_bits),
        condition_dim=int(config.condition_dim),
        allowed_counts=set(int(value) for value in config.property_counts),
        timeout=int(config.mcs_timeout),
        min_common_fraction=float(config.min_common_fraction),
        limit=int(limit),
        seed=int(seed),
        forbidden_sources=forbidden_sources,
        forbidden_pairs=forbidden_pairs,
    )


def select_fresh_pairs(
    args: argparse.Namespace, preregistration: Mapping[str, object]
) -> tuple[list[object], dict[str, object]]:
    config = SimpleNamespace(
        graph_fingerprint_bits=int(preregistration["graph_fingerprint_bits"]),
        condition_dim=int(preregistration["condition_dim"]),
        property_counts=list(preregistration["property_counts"]),
        mcs_timeout=int(preregistration["mcs_timeout"]),
        min_common_fraction=float(preregistration["min_common_fraction"]),
    )
    validation_rows = base.read_rows(args.validation_csv)
    historical, historical_counts = build_pairs(
        validation_rows,
        config=config,
        limit=int(preregistration["prior_validation_limit"]),
        seed=int(preregistration["prior_validation_selection_seeds"][0]),
    )
    historical_sources, historical_keys = pair_sets(historical)
    reused_dev, reused_counts = build_pairs(
        validation_rows,
        config=config,
        limit=int(preregistration["prior_validation_limit"]),
        seed=int(preregistration["prior_validation_selection_seeds"][1]),
        forbidden_sources=historical_sources,
        forbidden_pairs=historical_keys,
    )
    reused_sources, reused_keys = pair_sets(reused_dev)

    # Reconstruct the exact B24 train selection.  B24 excluded the reused dev
    # split (seed 2719) when sampling the training rows.
    train_pairs, train_counts = build_pairs(
        base.read_rows(args.train_csv),
        config=config,
        limit=int(preregistration["train_limit"]),
        seed=int(preregistration["train_selection_seed"]),
        forbidden_sources=reused_sources,
        forbidden_pairs=reused_keys,
    )
    train_sources, train_keys = pair_sets(train_pairs)
    all_forbidden_sources = historical_sources | reused_sources | train_sources
    all_forbidden_keys = historical_keys | reused_keys | train_keys
    fresh, fresh_counts = build_pairs(
        validation_rows,
        config=config,
        limit=int(preregistration["validation_limit"]),
        seed=int(preregistration["fresh_validation_selection_seed"]),
        forbidden_sources=all_forbidden_sources,
        forbidden_pairs=all_forbidden_keys,
    )
    for pair in fresh:
        pair.condition = hierarchical.property_latent_slot_tokens(
            pair.row, int(config.condition_dim)
        )
    fresh_sources, fresh_keys = pair_sets(fresh)
    return fresh, {
        "historical_validation_pairs": len(historical),
        "reused_development_pairs": len(reused_dev),
        "reconstructed_train_pairs": len(train_pairs),
        "fresh_heldout_pairs": len(fresh),
        "fresh_train_source_overlap": len(fresh_sources & train_sources),
        "fresh_train_pair_overlap": len(fresh_keys & train_keys),
        "fresh_historical_source_overlap": len(fresh_sources & historical_sources),
        "fresh_historical_pair_overlap": len(fresh_keys & historical_keys),
        "fresh_reused_dev_source_overlap": len(fresh_sources & reused_sources),
        "fresh_reused_dev_pair_overlap": len(fresh_keys & reused_keys),
        "historical_filter_counts": historical_counts,
        "reused_dev_filter_counts": reused_counts,
        "train_filter_counts": train_counts,
        "fresh_filter_counts": fresh_counts,
    }


def load_frozen_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    preregistration: Mapping[str, object],
) -> tuple[kernel.FragmentAttachmentKernel, list[str], np.ndarray, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("stage") != preregistration["frozen_model_protocol"]:
        raise ValueError("Frozen checkpoint protocol does not match preregistration")
    frozen_manifest = dict(payload.get("manifest", {}))
    contract = {
        "seed": preregistration["frozen_model_seed"],
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "generation_rdkit_validity_feedback": False,
        "molecular_candidate_ranking": False,
        "failed_attachment_retry": False,
        "exact_raw_attempts_per_condition": 20,
    }
    drift = {
        key: {"expected": value, "actual": frozen_manifest.get(key)}
        for key, value in contract.items()
        if frozen_manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"Frozen B24 checkpoint contract drift: {drift}")
    model_config = dict(payload["model_config"])
    model = kernel.FragmentAttachmentKernel(
        source_dim=int(model_config["source_dim"]),
        condition_dim=int(model_config["condition_dim"]),
        site_dim=int(model_config["site_dim"]),
        endpoint_dim=int(model_config["endpoint_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return (
        model,
        list(payload["target_fragments"]),
        np.asarray(payload["target_endpoints"], dtype=np.float32),
        frozen_manifest,
    )


def task_breakdown(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_condition: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    by_task: defaultdict[str, list[list[Mapping[str, object]]]] = defaultdict(list)
    for values in by_condition.values():
        by_task[str(values[0]["task"])].append(values)
    output: dict[str, object] = {}
    for task, conditions in sorted(by_task.items()):
        flat = [row for values in conditions for row in values]
        output[task] = {
            "conditions": len(conditions),
            "validity": sum(bool(row["valid"]) for row in flat) / max(1, len(flat)),
            "property_any20": sum(
                any(bool(row["property_success"]) for row in values)
                for values in conditions
            )
            / max(1, len(conditions)),
            "strict_any20": sum(
                any(bool(row["strict_success"]) for row in values)
                for values in conditions
            )
            / max(1, len(conditions)),
            "mean_unique_valid": float(
                np.mean(
                    [
                        len(
                            {
                                str(row["generated_smiles"])
                                for row in values
                                if bool(row["valid"])
                            }
                        )
                        for values in conditions
                    ]
                )
            ),
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Fresh-heldout result is once-only and already exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["generation_seed"]))
    device = base.resolve_device(str(args.device))

    representation, _representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    model, target_fragments, target_endpoints, frozen_manifest = load_frozen_model(
        args.fragment_checkpoint,
        device=device,
        preregistration=preregistration,
    )
    if frozen_manifest.get("representation_checkpoint_sha256") != belief.file_sha256(
        args.representation_checkpoint
    ):
        raise ValueError("Representation checkpoint differs from frozen B24 manifest")
    if frozen_manifest.get("train_csv_sha256") != belief.file_sha256(args.train_csv):
        raise ValueError("Training CSV differs from frozen B24 manifest")
    if frozen_manifest.get("validation_csv_sha256") != belief.file_sha256(
        args.validation_csv
    ):
        raise ValueError("Validation CSV differs from frozen B24 manifest")

    pairs, split = select_fresh_pairs(args, preregistration)
    if not pairs:
        raise ValueError("No fresh heldout pairs remain after frozen split exclusions")
    source_latents = kernel.encode_sources(representation, pairs, device)
    generation_args = SimpleNamespace(
        num_attempts=int(preregistration["num_attempts"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        seed=int(preregistration["generation_seed"]),
    )
    candidate_rows, metrics = kernel.evaluate(
        model,
        pairs,
        source_latents,
        target_fragments,
        target_endpoints,
        generation_args,
        device,
    )
    metrics["by_task"] = task_breakdown(candidate_rows)

    by_count = metrics.get("by_property_count", {})
    two = dict(by_count.get("2", {}))
    three = dict(by_count.get("3", {}))
    gates = dict(preregistration["gates"])
    checks = {
        "exact_attempts": {
            "value": metrics["attempted_per_condition"],
            "threshold": int(preregistration["num_attempts"]),
        },
        "candidate_rows": {
            "value": metrics["candidate_rows"],
            "threshold": len(pairs) * int(preregistration["num_attempts"]),
        },
        "minimum_conditions": {
            "value": len(pairs),
            "threshold": int(preregistration["minimum_conditions"]),
        },
        "minimum_two_property_conditions": {
            "value": int(two.get("conditions", 0)),
            "threshold": int(gates["minimum_conditions_per_property_count"]),
        },
        "minimum_three_property_conditions": {
            "value": int(three.get("conditions", 0)),
            "threshold": int(gates["minimum_conditions_per_property_count"]),
        },
        "validity": {"value": metrics["validity"], "threshold": gates["validity"]},
        "strict_any20": {
            "value": metrics["strict_any20"],
            "threshold": gates["strict_any20"],
        },
        "two_property_strict_any20": {
            "value": float(two.get("strict_any20", 0.0)),
            "threshold": gates["two_property_strict_any20"],
        },
        "three_property_strict_any20": {
            "value": float(three.get("strict_any20", 0.0)),
            "threshold": gates["three_property_strict_any20"],
        },
        "mean_unique_valid": {
            "value": metrics["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "mean_source_tanimoto": {
            "value": metrics["mean_source_tanimoto"],
            "threshold": gates["mean_source_tanimoto"],
        },
        "all_split_overlaps_zero": {
            "value": sum(
                int(value)
                for key, value in split.items()
                if key.endswith("_overlap")
            ),
            "threshold": 0,
        },
    }
    exact_checks = {"exact_attempts", "candidate_rows", "all_split_overlaps_zero"}
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
        "heldout_role": "once_only_fresh_development_confirmation",
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "frozen_fragment_checkpoint_sha256": belief.file_sha256(
            args.fragment_checkpoint
        ),
        "representation_checkpoint_sha256": belief.file_sha256(
            args.representation_checkpoint
        ),
        "representation_protocol": representation_summary.get("protocol"),
        "train_csv_sha256": belief.file_sha256(args.train_csv),
        "validation_csv_sha256": belief.file_sha256(args.validation_csv),
        "model_training": False,
        "hyperparameter_search": False,
        "generation_target_access": False,
        "property_oracle_generation_access": False,
        "official_test_access": False,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "failed_attachment_retry": False,
        "repeat_after_scientific_failure": False,
        "split": split,
    }
    summary = {
        "protocol": PROTOCOL,
        "preregistration": preregistration,
        "manifest": run_manifest,
        "evaluation": metrics,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "decision": (
            "advance_frozen_b24_to_cross_task_subset_validation"
            if not failures
            else "stop_and_report_frozen_b24_generalization_failure_without_retuning"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_candidate_rows(args.output_dir / "validation_candidates.csv", candidate_rows)
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
