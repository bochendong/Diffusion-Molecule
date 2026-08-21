#!/usr/bin/env python3
"""Prospective fresh-source test of language routing with horizon closure.

The experiment freezes the V5 Common-LLM mass-conserving property router and
the V6 molecular bridge.  ``prepare`` selects new source-disjoint 2p/3p pairs
using training data and writes target-free generation conditions plus a sealed
evaluation file.  ``freeze`` accepts no target path and emits exactly twenty
raw particles per condition with the finite-horizon terminal closure defined in
``horizon_closed_graph_jump``.  ``evaluate`` and ``gate`` remain physically
separate so a scientific STOP is an artifact, not a failed Slurm execution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
for module_path in (SCRIPT_DIR, PROJECT_DIR, LATENT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import fresh_graph_jump_language_confirmation as fresh  # noqa: E402
import horizon_closed_graph_jump as horizon  # noqa: E402
import mass_conserving_router_table1_bridge_v6 as v6  # noqa: E402


PROTOCOL = "fresh_horizon_closed_mass_conserving_router_v7"
ARMS = v6.ARMS
base = v6.base
graph = v6.graph
b41 = v6.b41
b40 = v6.b40
hierarchical = v6.hierarchical
semantic = v6.semantic
property_basis = v6.property_basis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare = stages.add_parser("prepare")
    prepare.add_argument("--train-csv", required=True, type=Path)
    prepare.add_argument("--validation-csv", required=True, type=Path)
    prepare.add_argument("--b36-records", required=True, type=Path)
    prepare.add_argument("--predecessor-fit-bundle", required=True, type=Path)
    prepare.add_argument("--e1-manifest", required=True, type=Path)
    prepare.add_argument("--v6-basis-bundle", required=True, type=Path)
    prepare.add_argument("--v6-gate", required=True, type=Path)
    prepare.add_argument("--known-source", action="append", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    freeze = stages.add_parser("freeze")
    freeze.add_argument("--prepare-summary", required=True, type=Path)
    freeze.add_argument("--generation-conditions", required=True, type=Path)
    freeze.add_argument("--v6-basis-bundle", required=True, type=Path)
    freeze.add_argument("--representation-checkpoint", required=True, type=Path)
    freeze.add_argument("--representation-summary", required=True, type=Path)
    freeze.add_argument("--canonical-checkpoint", required=True, type=Path)
    freeze.add_argument("--sft-adapter-dir", required=True, type=Path)
    freeze.add_argument("--v5-lora-adapter-dir", required=True, type=Path)
    freeze.add_argument("--v5-router-checkpoint", required=True, type=Path)
    freeze.add_argument("--v5-summary", required=True, type=Path)
    freeze.add_argument("--v5-gate", required=True, type=Path)
    freeze.add_argument("--v5-unlock", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--device", default="auto")

    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--prepare-summary", required=True, type=Path)
    evaluate.add_argument("--evaluation-targets", required=True, type=Path)
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    gate = stages.add_parser("gate")
    gate.add_argument("--prepare-summary", required=True, type=Path)
    gate.add_argument("--evaluation-summary", required=True, type=Path)
    gate.add_argument("--output-dir", required=True, type=Path)
    return parser


def file_sha256(path: Path) -> str:
    return v6.file_sha256(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "amended_before_any_candidate_generation_after_source_availability_audit",
        "arms": list(ARMS),
        "training": False,
        "fresh_condition_count": 40,
        "fresh_property_count_quotas": {"2": 20, "3": 20},
        "protocol_amendment": {
            "reason": (
                "The locked exclusions and alignment filters exposed only 20 eligible "
                "unique 2p sources at the deterministic 20000-row scan limit."
            ),
            "observed_before_amendment": "prepare-stage source availability only",
            "candidate_generation_started": False,
            "property_oracle_accessed": False,
            "scientific_metrics_accessed": False,
            "amended_at": "2026-08-21",
        },
        "exact_raw_attempts_per_condition": 20,
        "candidate_pool_before_selection": 20,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "finite_horizon_closure": True,
        "last_exactly_materializable_checkpoint": True,
        "single_seed": True,
        "official_test_access": False,
        "repeat_v6_conditions": False,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"V7 preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if str(payload.get("implementation_sha256")) != actual:
        raise ValueError(
            f"V7 implementation drift: expected {payload.get('implementation_sha256')}, found {actual}"
        )
    horizon_actual = file_sha256(SCRIPT_DIR / "horizon_closed_graph_jump.py")
    if str(payload.get("horizon_closure_sha256")) != horizon_actual:
        raise ValueError("V7 horizon-closure implementation drift")
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locked = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V7 locked inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locked.get(name), "actual": digest}
        for name, digest in actual.items()
        if locked.get(name) != digest
    }
    if drift:
        raise ValueError(f"V7 locked-input drift: {drift}")
    return actual


def stable_key(seed: int, *values: object) -> str:
    text = "\0".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sources_from_path(path: Path) -> set[str]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = read_json(path)
        rows = list(payload.get("records") or [])
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    output = set()
    for raw in rows:
        value = graph.canonical_smiles(str(dict(raw).get("source_smiles", "") or ""))
        if value:
            output.add(value)
    return output


def validate_v6_signal(path: Path) -> dict[str, object]:
    payload = read_json(path)
    gate = dict(payload.get("science_gate") or {})
    checks = dict(gate.get("checks") or {})
    effects = dict(payload.get("effects") or {})
    if payload.get("protocol") != "mass_conserving_router_table1_bridge_v6_science_gate":
        raise ValueError("V7 requires the frozen V6 science gate")
    if list(gate.get("failures") or []) != ["all_arm_validity"]:
        raise ValueError("V6 predecessor did not fail only on candidate validity")
    for name in (
        "candidate_rows",
        "exact_attempts",
        "language_source_tanimoto",
        "language_unique",
        "language_property_any20",
        "language_strict_any20",
        "language_vs_numeric",
        "language_vs_reversed",
    ):
        if checks.get(name) is not True:
            raise ValueError(f"V6 predecessor signal check failed: {name}")
    return {
        "strict_delta_vs_numeric": float(effects["strict_delta_vs_numeric"]),
        "strict_delta_vs_reversed": float(effects["strict_delta_vs_reversed"]),
        "only_failure": "all_arm_validity",
    }


def select_fresh_pairs(
    train_csv: Path,
    preregistration: Mapping[str, object],
    forbidden_sources: set[str],
) -> tuple[list[object], dict[str, object]]:
    rows = base.read_rows(train_csv)
    quotas = {
        int(key): int(value)
        for key, value in dict(preregistration["fresh_property_count_quotas"]).items()
    }
    buckets: dict[int, list[object]] = {}
    filter_counts: dict[str, object] = {}
    for count in sorted(quotas):
        pairs, counts = base.build_pairs(
            rows,
            max_atoms=int(preregistration["max_atoms"]),
            fingerprint_bits=int(preregistration["fingerprint_bits"]),
            condition_dim=int(preregistration["condition_dim"]),
            allowed_counts={count},
            timeout=int(preregistration["mcs_timeout"]),
            min_common_fraction=float(preregistration["min_common_fraction"]),
            limit=int(preregistration["fresh_alignment_limit_per_property_count"]),
            seed=int(preregistration["fresh_selection_seed"]) + count,
            forbidden_sources=forbidden_sources,
        )
        pairs.sort(
            key=lambda pair: stable_key(
                int(preregistration["fresh_selection_seed"]),
                count,
                pair.source_smiles,
                pair.target_smiles,
                base.task_key(pair.row),
            )
        )
        buckets[count] = pairs
        filter_counts[str(count)] = counts
    selected: list[object] = []
    used_sources: set[str] = set()
    for count in sorted(quotas):
        for pair in buckets[count]:
            source = graph.canonical_smiles(pair.source_smiles)
            if not source or source in used_sources:
                continue
            selected.append(pair)
            used_sources.add(source)
            if sum(int(item.property_count) == count for item in selected) == quotas[count]:
                break
        found = sum(int(item.property_count) == count for item in selected)
        if found != quotas[count]:
            raise ValueError(
                f"Fresh V7 quota unavailable for {count}p: wanted {quotas[count]}, found {found}"
            )
    selected.sort(
        key=lambda pair: stable_key(
            int(preregistration["fresh_selection_seed"]),
            "final",
            pair.source_smiles,
            pair.target_smiles,
        )
    )
    return selected, {
        "selected_conditions": len(selected),
        "selected_unique_sources": len(used_sources),
        "selected_by_property_count": {
            str(count): sum(int(pair.property_count) == count for pair in selected)
            for count in sorted(quotas)
        },
        "forbidden_sources": len(forbidden_sources),
        "eligible_by_property_count": {
            str(count): len(buckets[count]) for count in sorted(quotas)
        },
        "filter_counts": filter_counts,
    }


def run_prepare(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V7 prepare exists: {summary_path}")
    known_names = sorted(
        name for name in dict(preregistration["locked_inputs"]) if name.startswith("known_source_")
    )
    if len(known_names) != len(args.known_source):
        raise ValueError("V7 known-source list does not match preregistration")
    paths = {
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
        "b36_records_sha256": args.b36_records,
        "predecessor_fit_bundle_sha256": args.predecessor_fit_bundle,
        "e1_manifest_sha256": args.e1_manifest,
        "v6_basis_bundle_sha256": args.v6_basis_bundle,
        "v6_gate_sha256": args.v6_gate,
        **{
            name: value
            for name, value in zip(known_names, args.known_source, strict=True)
        },
    }
    input_hashes = check_locked_inputs(preregistration, paths)
    v6_signal = validate_v6_signal(args.v6_gate)
    basis_bundle = torch.load(args.v6_basis_bundle, map_location="cpu", weights_only=False)
    if (
        basis_bundle.get("protocol") != v6.PROTOCOL
        or basis_bundle.get("role") != "target_free_frozen_generation_basis_and_support"
    ):
        raise ValueError("V7 requires the frozen target-free V6 basis bundle")
    predecessor = fresh.load_frozen_predecessor_bundle(args.predecessor_fit_bundle)
    predecessor_sources = {
        graph.canonical_smiles(pair.source_smiles) for pair in predecessor["pairs"]
    }
    forbidden = set(predecessor_sources)
    forbidden |= fresh.all_validation_sources(args.validation_csv)
    forbidden |= canonical_sources_from_path(args.b36_records)
    known_counts = {}
    for name, path in zip(known_names, args.known_source, strict=True):
        values = canonical_sources_from_path(path)
        known_counts[name] = len(values)
        forbidden |= values
    selected, selection = select_fresh_pairs(args.train_csv, preregistration, forbidden)
    property_names = dict(read_json(args.e1_manifest)["property_names"])
    generation_records = []
    evaluation_records = []
    for index, pair in enumerate(selected):
        safe = fresh.fresh_v3.direction_only_row(pair.row, pair.source_smiles)
        condition_id = f"v7_fresh_horizon_{index:04d}"
        instructions = semantic.instruction_variants(
            safe,
            property_names,
            seed=int(preregistration["instruction_seed"]),
            key=condition_id,
            heldout=True,
        )
        generation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": index,
                "source_smiles": pair.source_smiles,
                "property_count": int(pair.property_count),
                "task": base.task_key(safe),
                "condition_row": safe,
                "instructions": instructions,
            }
        )
        evaluation_records.append(
            {
                "condition_id": condition_id,
                "pair_index": index,
                "source_smiles": pair.source_smiles,
                "target_smiles": pair.target_smiles,
                "property_count": int(pair.property_count),
                "row": dict(pair.row),
            }
        )
    generation_path = args.output_dir / "generation_conditions.json"
    evaluation_path = args.output_dir / "sealed_evaluation_targets.json"
    generation_text = json.dumps(
        {
            "protocol": PROTOCOL,
            "role": "fresh_constraint_text_and_sources_without_targets",
            "records": generation_records,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    forbidden_terms = (
        "target_smiles",
        "target_properties",
        "generated_smiles",
        "strict_success",
        "oracle",
    )
    leaks = [term for term in forbidden_terms if term in generation_text.lower()]
    if leaks:
        raise ValueError(f"V7 target-free generation manifest leaked {leaks}")
    generation_path.write_text(generation_text, encoding="utf-8")
    evaluation_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "role": "sealed_post_freeze_fresh_targets",
                "records": evaluation_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selected_sources = {
        graph.canonical_smiles(str(record["source_smiles"]))
        for record in generation_records
    }
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare_fresh_target_isolated_manifests",
        "execution_status": "completed",
        "decision": "freeze_horizon_closed_exact_n20",
        "fresh_conditions": len(selected),
        "fresh_unique_sources": len(selected_sources),
        "fresh_forbidden_source_overlap": len(selected_sources & forbidden),
        "selection": selection,
        "known_source_counts": known_counts,
        "v6_predecessor_signal": v6_signal,
        "artifacts": {
            "generation_conditions_sha256": file_sha256(generation_path),
            "evaluation_targets_sha256": file_sha256(evaluation_path),
            "locked_inputs": input_hashes,
        },
        "contract": {
            "fresh_target_process_isolation": True,
            "generation_target_access": False,
            "repeat_v6_conditions": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def load_generation_pairs(
    path: Path, preregistration: Mapping[str, object]
) -> tuple[list[object], list[dict[str, object]]]:
    payload = read_json(path)
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("role") != "fresh_constraint_text_and_sources_without_targets"
    ):
        raise ValueError("Invalid target-free V7 generation manifest")
    records = [dict(record) for record in payload["records"]]
    pairs = []
    for expected_index, record in enumerate(records):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("V7 generation order drift")
        source = graph.molecule_example(
            str(record["source_smiles"]),
            int(preregistration["max_atoms"]),
            int(preregistration["fingerprint_bits"]),
        )
        if source is None:
            raise ValueError(f"Cannot materialize V7 source {expected_index}")
        row = {str(key): str(value) for key, value in dict(record["condition_row"]).items()}
        pairs.append(
            SimpleNamespace(
                row=row,
                source_smiles=str(record["source_smiles"]),
                source=source,
                condition=np.zeros(
                    (
                        int(preregistration["token_count"]),
                        int(preregistration["condition_dim"]),
                    ),
                    dtype=np.float32,
                ),
                property_count=int(record["property_count"]),
                task=str(record["task"]),
            )
        )
    return pairs, records


def run_freeze(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_summary = args.output_dir / "summary.json"
    if root_summary.exists():
        raise ValueError(f"Completed V7 freeze exists: {root_summary}")
    check_locked_inputs(
        preregistration,
        {
            "v6_basis_bundle_sha256": args.v6_basis_bundle,
            "representation_checkpoint_sha256": args.representation_checkpoint,
            "representation_summary_sha256": args.representation_summary,
            "canonical_checkpoint_sha256": args.canonical_checkpoint,
            "common_sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
            "common_sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
            "v5_lora_adapter_config_sha256": args.v5_lora_adapter_dir / "adapter_config.json",
            "v5_lora_adapter_model_sha256": args.v5_lora_adapter_dir / "adapter_model.safetensors",
            "v5_router_checkpoint_sha256": args.v5_router_checkpoint,
            "v5_summary_sha256": args.v5_summary,
            "v5_gate_sha256": args.v5_gate,
            "v5_unlock_sha256": args.v5_unlock,
        },
    )
    v6.validate_v5_gate(args.v5_gate, args.v5_unlock)
    prepare = read_json(args.prepare_summary)
    if prepare.get("protocol") != PROTOCOL:
        raise ValueError("V7 prepare protocol drift")
    if file_sha256(args.generation_conditions) != dict(prepare["artifacts"])["generation_conditions_sha256"]:
        raise ValueError("V7 generation manifest drift")
    bundle = torch.load(args.v6_basis_bundle, map_location="cpu", weights_only=False)
    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["generation_seed"]))
    pairs, records = load_generation_pairs(args.generation_conditions, preregistration)
    llm, tokenizer, router, property_columns = v6.load_v5_language_router(
        args, preregistration, device
    )
    if property_columns != list(bundle["property_columns"]):
        raise ValueError("V7 property vocabulary drift")
    matched_examples = v6.examples_for_records(records, property_columns, "matched")
    reversed_examples = v6.examples_for_records(records, property_columns, "reversed")
    matched_coefficients, matched_support, matched_cardinality = v6.v5.predict_examples(
        llm, router, tokenizer, matched_examples, preregistration, device
    )
    reversed_coefficients, reversed_support, reversed_cardinality = v6.v5.predict_examples(
        llm, router, tokenizer, reversed_examples, preregistration, device
    )
    routing = {
        "language_full": v6.v5.routing_metrics(
            matched_coefficients, matched_support, matched_cardinality, matched_examples
        ),
        "language_reversed": v6.v5.routing_metrics(
            reversed_coefficients,
            reversed_support,
            reversed_cardinality,
            reversed_examples,
        ),
    }
    del llm, tokenizer, router
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    token_shape = tuple(int(value) for value in bundle["token_shape"])
    language_tokens = {
        "language_full": property_basis.compose_tokens(
            matched_coefficients, bundle["basis"], token_shape
        ).numpy(),
        "language_reversed": property_basis.compose_tokens(
            reversed_coefficients, bundle["basis"], token_shape
        ).numpy(),
    }
    model, representation, _config, representation_summary = semantic.load_graph_stack(
        args, preregistration, bundle, device
    )
    vocabulary = dict(bundle["vocabulary"])
    support = dict(bundle["support"])
    support_tensors = b40._device_support(support, device)
    for arm in ARMS:
        arm_dir = args.output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = arm_dir / "frozen_candidates.csv"
        summary_path = arm_dir / "summary.json"
        if candidate_path.exists() or summary_path.exists():
            raise ValueError(f"Completed V7 arm exists: {arm_dir}")
        rows = []
        started = time.perf_counter()
        for index, pair in enumerate(pairs):
            if arm == "numeric_canonical":
                pair.condition = hierarchical.property_latent_slot_tokens(
                    pair.row, int(preregistration["condition_dim"])
                )
            else:
                pair.condition = np.asarray(language_tokens[arm][index], dtype=np.float32)
            generated = horizon.sample_from_source(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
                pair.source,
                np.asarray(pair.condition, dtype=np.float32),
                preregistration,
                device,
                int(preregistration["generation_seed"]) * 100000
                + ARMS.index(arm) * 10000
                + index,
            )
            if len(generated) != 20:
                raise RuntimeError(f"V7 {arm} did not emit exactly 20 attempts")
            for attempt, candidate in enumerate(generated, start=1):
                rows.append(
                    {
                        "condition_id": str(records[index]["condition_id"]),
                        "pair_index": index,
                        "attempt": attempt,
                        "property_count": int(pair.property_count),
                        "task": pair.task,
                        "source_smiles": pair.source_smiles,
                        "arm": arm,
                        **candidate,
                    }
                )
            if (index + 1) % 12 == 0 or index + 1 == len(pairs):
                print(
                    json.dumps(
                        {
                            "stage": "v7_freeze_progress",
                            "arm": arm,
                            "conditions": index + 1,
                            "total": len(pairs),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        expected = len(pairs) * 20
        if len(rows) != expected:
            raise RuntimeError(f"V7 {arm} expected {expected} rows, found {len(rows)}")
        if any(not str(row["generated_smiles"]) for row in rows):
            raise RuntimeError(f"V7 {arm} emitted a non-materializable candidate")
        base.write_candidate_rows(candidate_path, rows)
        arm_summary = {
            "protocol": PROTOCOL,
            "stage": "fresh_target_isolated_horizon_closed_freeze",
            "execution_status": "completed",
            "arm": arm,
            "conditions": len(pairs),
            "candidate_rows": len(rows),
            "attempts_per_condition": 20,
            "mean_unique_smiles": v6.mean_unique_smiles(rows),
            "horizon_forced_stop_rate": sum(bool(row["horizon_forced_stop"]) for row in rows) / len(rows),
            "horizon_checkpoint_restore_rate": sum(bool(row["horizon_checkpoint_restored"]) for row in rows) / len(rows),
            "elapsed_sec": round(time.perf_counter() - started, 1),
            "artifacts": {"frozen_candidates_sha256": file_sha256(candidate_path)},
            "contract": {
                "generation_target_path_accepted": False,
                "generation_property_oracle_access": False,
                "exact_raw_attempts_per_condition": 20,
                "candidate_pool_before_selection": 20,
                "molecular_candidate_ranking": False,
                "oracle_selection": False,
                "retry_or_resampling": False,
                "posthoc_molecule_repair": False,
                "finite_horizon_closure": True,
            },
        }
        summary_path.write_text(
            json.dumps(arm_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "protocol": PROTOCOL,
        "stage": "all_fresh_arms_frozen",
        "execution_status": "completed",
        "decision": "await_post_freeze_evaluation",
        "arms": list(ARMS),
        "conditions": len(pairs),
        "candidate_rows_per_arm": len(pairs) * 20,
        "routing": routing,
        "representation_protocol": representation_summary.get("protocol"),
        "contract": {
            "generation_target_access": False,
            "exact_raw_attempts_per_condition": 20,
            "candidate_pool_before_selection": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
            "posthoc_molecule_repair": False,
            "finite_horizon_closure": True,
        },
    }
    root_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def load_evaluation_pairs(
    path: Path, preregistration: Mapping[str, object]
) -> list[object]:
    payload = read_json(path)
    if (
        payload.get("protocol") != PROTOCOL
        or payload.get("role") != "sealed_post_freeze_fresh_targets"
    ):
        raise ValueError("Invalid sealed V7 evaluation targets")
    pairs = []
    for expected_index, record in enumerate(payload["records"]):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("V7 evaluation order drift")
        row = {str(key): str(value) for key, value in dict(record["row"]).items()}
        aligned = base.align_pair(
            str(record["source_smiles"]),
            str(record["target_smiles"]),
            max_atoms=int(preregistration["max_atoms"]),
            fingerprint_bits=int(preregistration["fingerprint_bits"]),
            timeout=int(preregistration["mcs_timeout"]),
            min_common_fraction=float(preregistration["min_common_fraction"]),
        )
        if aligned is None:
            raise ValueError(f"Cannot reconstruct sealed V7 pair {expected_index}")
        source, target, common = aligned
        pairs.append(
            base.EditPair(
                row=row,
                source_smiles=str(record["source_smiles"]),
                target_smiles=str(record["target_smiles"]),
                source=source,
                target=target,
                condition=np.zeros((1, int(preregistration["condition_dim"])), dtype=np.float32),
                property_count=int(record["property_count"]),
                task=base.task_key(row),
                common_atoms=int(common),
            )
        )
    return pairs


def property_count_breakdown(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output = {}
    for count in (2, 3):
        conditions = [values for values in grouped.values() if int(values[0]["property_count"]) == count]
        output[str(count)] = {
            "conditions": len(conditions),
            "property_any20": sum(any(bool(row["property_success"]) for row in values) for values in conditions) / max(1, len(conditions)),
            "strict_any20": sum(any(bool(row["strict_success"]) for row in values) for values in conditions) / max(1, len(conditions)),
        }
    return output


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V7 evaluation exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    if file_sha256(args.evaluation_targets) != dict(prepare["artifacts"])["evaluation_targets_sha256"]:
        raise ValueError("Sealed V7 evaluation targets drift")
    pairs = load_evaluation_pairs(args.evaluation_targets, preregistration)
    expected = len(pairs) * 20
    metrics_by_arm = {}
    for arm in ARMS:
        arm_dir = args.frozen_root / arm
        arm_summary = read_json(arm_dir / "summary.json")
        candidate_path = arm_dir / "frozen_candidates.csv"
        if file_sha256(candidate_path) != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]:
            raise ValueError(f"Frozen V7 candidate drift: {arm}")
        frozen = semantic.coerce_frozen_rows(candidate_path)
        if len(frozen) != expected:
            raise ValueError(f"V7 {arm} expected {expected} rows, found {len(frozen)}")
        evaluated, metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = dict(metrics)
        metrics["by_task"] = v6.task_breakdown(evaluated)
        metrics["by_property_count"] = property_count_breakdown(evaluated)
        metrics["horizon_forced_stop_rate"] = float(arm_summary["horizon_forced_stop_rate"])
        metrics["horizon_checkpoint_restore_rate"] = float(arm_summary["horizon_checkpoint_restore_rate"])
        evaluated_path = args.output_dir / f"evaluated_{arm}.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        metrics_by_arm[arm] = {
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
    summary = {
        "protocol": PROTOCOL,
        "stage": "post_freeze_fresh_evaluation_execution",
        "execution_status": "completed",
        "decision": "await_separate_science_gate",
        "arms": metrics_by_arm,
        "contract": {
            "generation_target_access": False,
            "post_freeze_target_access": True,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "science_decision_in_separate_process": True,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def run_gate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "gate_summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V7 science gate exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    evaluation = read_json(args.evaluation_summary)
    if prepare.get("protocol") != PROTOCOL or evaluation.get("protocol") != PROTOCOL:
        raise ValueError("V7 gate input protocol drift")
    by_arm = dict(evaluation["arms"])
    metrics = {arm: dict(dict(by_arm[arm])["metrics"]) for arm in ARMS}
    full = metrics["language_full"]
    numeric = metrics["numeric_canonical"]
    reversed_metrics = metrics["language_reversed"]
    effects = {
        "strict_delta_vs_numeric": float(full["strict_any20"]) - float(numeric["strict_any20"]),
        "property_delta_vs_reversed": float(full["property_any20"]) - float(reversed_metrics["property_any20"]),
        "strict_delta_vs_reversed": float(full["strict_any20"]) - float(reversed_metrics["strict_any20"]),
    }
    gates = dict(preregistration["science_gates"])
    expected_rows = int(preregistration["fresh_condition_count"]) * 20
    checks = {
        "fresh_conditions": int(prepare["fresh_conditions"]) == int(preregistration["fresh_condition_count"]),
        "fresh_unique_sources": int(prepare["fresh_unique_sources"]) == int(preregistration["fresh_condition_count"]),
        "fresh_forbidden_source_overlap": int(prepare["fresh_forbidden_source_overlap"]) == 0,
        "candidate_rows": all(int(metrics[arm]["candidate_rows"]) == expected_rows for arm in ARMS),
        "exact_attempts": all(int(metrics[arm]["attempted_per_condition"]) == 20 for arm in ARMS),
        "all_arm_validity": all(float(metrics[arm]["validity"]) >= float(gates["validity"]) for arm in ARMS),
        "language_source_tanimoto": float(full["mean_source_tanimoto"]) >= float(gates["mean_source_tanimoto"]),
        "language_unique": float(full["mean_unique_valid"]) >= float(gates["mean_unique_valid"]),
        "language_property_any20": float(full["property_any20"]) >= float(gates["property_any20"]),
        "language_strict_any20": float(full["strict_any20"]) >= float(gates["strict_any20"]),
        "language_vs_numeric": effects["strict_delta_vs_numeric"] >= float(gates["strict_delta_vs_numeric"]),
        "language_vs_reversed": effects["strict_delta_vs_reversed"] >= float(gates["strict_delta_vs_reversed"]),
    }
    failures = [name for name, passed in checks.items() if not bool(passed)]
    passed = not failures
    summary = {
        "protocol": "fresh_horizon_closed_mass_conserving_router_v7_science_gate",
        "execution_status": "completed",
        "science_gate": {
            "passed": passed,
            "checks": checks,
            "thresholds": gates,
            "failures": failures,
        },
        "decision": (
            "advance_frozen_router_to_denovo_table1_mumo_replays"
            if passed
            else "stop_before_three_task_replays_fresh_confirmation_failed"
        ),
        "metrics": metrics,
        "effects": effects,
        "contract": {
            "scientific_stop_exits_zero": True,
            "portal_engineering_state_independent_of_science_decision": True,
            "repeat_on_same_fresh_sources_for_retuning": False,
            "lower_gate_after_result": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = read_preregistration(args.protocol_manifest)
    if args.stage == "prepare":
        return run_prepare(args, preregistration)
    if args.stage == "freeze":
        return run_freeze(args, preregistration)
    if args.stage == "evaluate":
        return run_evaluate(args, preregistration)
    if args.stage == "gate":
        return run_gate(args, preregistration)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
