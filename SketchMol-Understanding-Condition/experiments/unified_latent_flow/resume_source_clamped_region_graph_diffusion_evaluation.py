#!/usr/bin/env python3
"""Resume B37 strictly from its already frozen 4,700 raw candidates.

Job 19864238 completed training and exact-n=20 generation, then failed because
the post-freeze evaluator omitted ``target_atom_count``.  This recovery process
cannot load a model, train, generate, filter, rank, retry, or modify candidates.
It locks the original preregistration, failed log, and frozen CSV hashes; rebuilds
the deterministic train-only source split; and opens targets/oracles only for
evaluation of those immutable rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
for path in (SCRIPT_DIR, PROJECT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import source_clamped_region_graph_diffusion as b37  # noqa: E402


base = b37.base
belief = b37.belief
graph = b37.graph
unified = b37.unified

PROTOCOL = "frozen_candidate_evaluation_resume_v37r1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--b22-checkpoint", type=Path, required=True)
    parser.add_argument("--b22-summary", type=Path, required=True)
    parser.add_argument("--b36-summary", type=Path, required=True)
    parser.add_argument("--original-preregistration", type=Path, required=True)
    parser.add_argument("--failed-log", type=Path, required=True)
    parser.add_argument("--frozen-candidates", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_resume_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = json.loads(args.resume_manifest.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "engineering_resume_preregistered_before_evaluation",
        "failed_job_id": 19864238,
        "failure_signature": "KeyError: target_atom_count after 4700 candidates frozen",
        "model_training": False,
        "molecular_candidate_generation": False,
        "candidate_modification": False,
        "candidate_filtering": False,
        "candidate_ranking": False,
        "oracle_selection": False,
        "post_freeze_evaluation_only": True,
        "expected_conditions": 235,
        "expected_candidate_rows": 4700,
        "exact_raw_attempts_per_condition": 20,
        "scientific_configuration_changed": False,
    }
    drift = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if drift:
        raise ValueError(f"B37r1 resume contract drift: {drift}")
    locked = dict(contract["locked_files"])
    paths = {
        "original_preregistration_sha256": args.original_preregistration,
        "failed_log_sha256": args.failed_log,
        "frozen_candidates_sha256": args.frozen_candidates,
        "b37_implementation_sha256": SCRIPT_DIR
        / "source_clamped_region_graph_diffusion.py",
        "resume_implementation_sha256": Path(__file__).resolve(),
    }
    file_drift = {
        name: {"expected": locked[name], "actual": belief.file_sha256(path)}
        for name, path in paths.items()
        if belief.file_sha256(path) != locked[name]
    }
    if file_drift:
        raise ValueError(f"B37r1 locked file drift: {file_drift}")
    preregistration = json.loads(
        args.original_preregistration.read_text(encoding="utf-8")
    )
    if preregistration.get("protocol") != b37.PROTOCOL:
        raise ValueError("B37r1 original protocol drift")
    if preregistration.get("implementation_sha256") != locked[
        "b37_implementation_sha256"
    ]:
        raise ValueError("B37r1 original implementation/preregistration mismatch")
    if dict(preregistration.get("gates", {})) != dict(contract.get("gates", {})):
        raise ValueError("B37r1 gate drift")
    return contract, preregistration


def training_history(path: Path, expected_epochs: int) -> list[dict[str, object]]:
    history = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "epoch" in payload:
            history.append(payload)
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, int(expected_epochs) + 1)):
        raise ValueError(f"B37r1 incomplete training history: {epochs}")
    for row in history:
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key != "epoch"
        ):
            raise ValueError(f"B37r1 non-finite training history: {row}")
    return history


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected serialized boolean, got {value!r}")
    return normalized == "true"


def read_and_validate_frozen(
    path: Path,
    pairs: Sequence[object],
    *,
    expected_rows: int,
    attempts: int,
) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != int(expected_rows):
        raise ValueError(f"B37r1 expected {expected_rows} frozen rows, found {len(raw)}")
    forbidden = {"target_smiles", "property_success", "strict_success", "valid"}
    leaked = forbidden & set(raw[0])
    if leaked:
        raise ValueError(f"B37r1 frozen CSV contains post-freeze fields: {sorted(leaked)}")
    integer_fields = {
        "pair_index",
        "attempt",
        "property_count",
        "predicted_atom_count",
        "region_size",
        "region_components",
        "node_edit_count",
        "edge_edit_count",
        "boundary_edge_edit_count",
        "region_incident_pair_count",
    }
    float_fields = {"latent_norm"}
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    by_condition: Counter[str] = Counter()
    for raw_row in raw:
        row: dict[str, object] = dict(raw_row)
        for field in integer_fields:
            row[field] = int(str(row[field]))
        for field in float_fields:
            row[field] = float(str(row[field]))
        row["outside_source_invariant"] = parse_bool(
            row["outside_source_invariant"]
        )
        pair_index = int(row["pair_index"])
        if not 0 <= pair_index < len(pairs):
            raise ValueError(f"B37r1 pair index out of range: {pair_index}")
        pair = pairs[pair_index]
        expected_condition = f"train_only_dev_{pair_index:04d}"
        if row["condition_id"] != expected_condition:
            raise ValueError("B37r1 condition/pair mapping drift")
        if row["source_smiles"] != pair.source_smiles:
            raise ValueError("B37r1 source mapping drift")
        if int(row["property_count"]) != int(pair.property_count):
            raise ValueError("B37r1 property-count mapping drift")
        if row["task"] != base.task_key(pair.row):
            raise ValueError("B37r1 task mapping drift")
        key = (str(row["condition_id"]), int(row["attempt"]))
        if key in seen:
            raise ValueError(f"B37r1 duplicate frozen attempt: {key}")
        seen.add(key)
        by_condition[str(row["condition_id"])] += 1
        rows.append(row)
    if len(by_condition) != len(pairs):
        raise ValueError(
            f"B37r1 expected {len(pairs)} conditions, found {len(by_condition)}"
        )
    if set(by_condition.values()) != {int(attempts)}:
        raise ValueError(f"B37r1 non-exact attempt counts: {dict(by_condition)}")
    return rows


def evaluate_frozen(
    frozen: Sequence[Mapping[str, object]], pairs: Sequence[object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for row in frozen:
        pair = pairs[int(row["pair_index"])]
        smiles = str(row["generated_smiles"] or "")
        valid = bool(smiles)
        source_tanimoto = graph.morgan_tanimoto(pair.source_smiles, smiles) if valid else None
        target_tanimoto = graph.morgan_tanimoto(pair.target_smiles, smiles) if valid else None
        fraction, _, evaluated_properties, property_success = (
            unified.instruction_success_and_distance(
                pair.row, smiles, task_specs=base.task_specs(pair.row)
            )
        )
        similarity_success = bool(
            source_tanimoto is not None and source_tanimoto >= 0.4
        )
        evaluated.append(
            {
                **dict(row),
                "target_smiles": pair.target_smiles,
                "source_atom_count": int(np.asarray(pair.source.node_mask).sum()),
                "target_atom_count": int(np.asarray(pair.target.node_mask).sum()),
                "valid": valid,
                "source_tanimoto": float(source_tanimoto or 0.0),
                "target_tanimoto": float(target_tanimoto or 0.0),
                "property_fraction": float(fraction),
                "evaluated_properties": int(evaluated_properties),
                "property_success": bool(property_success),
                "source_similarity_success": similarity_success,
                "strict_success": bool(property_success and similarity_success),
                "source_copy_target_tanimoto": float(
                    graph.morgan_tanimoto(pair.source_smiles, pair.target_smiles) or 0.0
                ),
            }
        )
    metrics = base.summarize_candidates(evaluated, 20)
    for name in (
        "latent_norm",
        "region_size",
        "region_components",
        "node_edit_count",
        "edge_edit_count",
        "boundary_edge_edit_count",
    ):
        metrics[f"mean_{name}"] = float(
            np.mean([float(row[name]) for row in evaluated])
        )
    metrics["outside_source_invariant_rate"] = sum(
        bool(row["outside_source_invariant"]) for row in evaluated
    ) / max(1, len(evaluated))
    metrics["nonempty_region_rate"] = sum(
        int(row["region_size"]) > 0 for row in evaluated
    ) / max(1, len(evaluated))
    return evaluated, metrics


def gate(metrics: Mapping[str, object], thresholds: Mapping[str, object]) -> dict[str, object]:
    by_count = dict(metrics["by_property_count"])
    checks = {
        "exact_attempts": {"value": metrics["attempted_per_condition"], "threshold": 20},
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
            "value": dict(by_count.get("2", {})).get("strict_any20", 0.0),
            "threshold": thresholds["two_property_strict_any20"],
        },
        "three_property_strict_any20": {
            "value": dict(by_count.get("3", {})).get("strict_any20", 0.0),
            "threshold": thresholds["three_property_strict_any20"],
        },
        "outside_source_invariant_rate": {
            "value": metrics["outside_source_invariant_rate"],
            "threshold": 1.0,
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name == "exact_attempts"
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    return {"passed": not failures, "checks": checks, "failures": failures}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract, preregistration = load_resume_contract(args)
    b22_summary, checkpoint, b36_evidence = b37.check_locked_inputs(
        args, preregistration
    )
    selected_pairs, reconstruction = b37.b36.reconstruct_b22_train_pairs(
        args, preregistration, checkpoint, b22_summary
    )
    fit_pairs, development_pairs, split = b37.strict_source_group_split(
        selected_pairs,
        seed=int(preregistration["development_split_seed"]),
        development_source_limit=int(preregistration["development_source_limit"]),
    )
    del fit_pairs
    history = training_history(args.failed_log, int(preregistration["epochs"]))
    frozen = read_and_validate_frozen(
        args.frozen_candidates,
        development_pairs,
        expected_rows=int(contract["expected_candidate_rows"]),
        attempts=int(contract["exact_raw_attempts_per_condition"]),
    )
    evaluated, metrics = evaluate_frozen(frozen, development_pairs)
    evaluated_path = args.output_dir / "evaluated_train_only_dev_candidates.csv"
    base.write_candidate_rows(evaluated_path, evaluated)
    internal_gate = gate(metrics, dict(contract["gates"]))
    manifest = {
        "protocol": b37.PROTOCOL,
        "evaluation_resume_protocol": PROTOCOL,
        "failed_training_generation_job": int(contract["failed_job_id"]),
        "original_preregistration_sha256": belief.file_sha256(
            args.original_preregistration
        ),
        "resume_manifest_sha256": belief.file_sha256(args.resume_manifest),
        "frozen_candidates_sha256": belief.file_sha256(args.frozen_candidates),
        "evaluated_candidates_sha256": belief.file_sha256(evaluated_path),
        "reconstruction": reconstruction,
        "split": split,
        "b36_decision": b36_evidence.get("decision"),
        "model_training_in_resume": False,
        "molecular_candidate_generation_in_resume": False,
        "candidate_modification": False,
        "candidate_filtering": False,
        "candidate_ranking": False,
        "oracle_selection": False,
        "generation_target_access": False,
        "post_freeze_train_only_dev_target_access": True,
        "exact_raw_attempts_per_condition": 20,
        "model_checkpoint_available": False,
        "checkpoint_unavailable_reason": (
            "original process saved checkpoint after the failing summary call"
        ),
        "scientific_configuration_changed": False,
        "b26_heldout_access": False,
        "b33_fresh_source_access": False,
        "moledit_table1_benchmark_access": False,
        "moledit_table1_training_lineage": True,
        "official_test_access": False,
    }
    summary = {
        "protocol": b37.PROTOCOL,
        "evaluation_resume_protocol": PROTOCOL,
        "checkpoint": None,
        "manifest": manifest,
        "training": history,
        "metrics": metrics,
        "internal_gate": internal_gate,
        "decision": (
            "advance_signal_but_reproduce_checkpoint_before_prospective_transfer"
            if internal_gate["passed"]
            else "stop_and_diagnose_region_transport_without_patch_expansion"
        ),
    }
    (args.output_dir / "evaluation_resume_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
