#!/usr/bin/env python3
"""Prospective molecular test of language-to-graph semantic interventions.

V9 selects source-disjoint train-only compositional edit conditions, freezes exactly
twenty paired particles per condition and arm without target/oracle access, and
evaluates the frozen particles in a separate process.  Candidate-level success
is primary because any@20 was saturated in V8.  The explicit numeric condition
is a privileged upper-bound control; language is tested for non-inferiority to
it and for superiority to semantic reversal and three frozen router ablations.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
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
import mass_conserving_property_set_router_v5 as v5  # noqa: E402
import mass_conserving_router_fresh_horizon_v7 as v7  # noqa: E402
import mass_conserving_router_table1_bridge_v6 as v6  # noqa: E402


PROTOCOL = "prospective_semantic_intervention_fresh_v9"
GATE_PROTOCOL = "prospective_semantic_intervention_fresh_v9_science_gate"
ARMS = (
    "numeric_canonical",
    "language_full",
    "language_reversed",
    "language_no_lora",
    "language_no_token_slots",
    "language_no_composition",
)
ROUTER_ARMS = {
    "language_full": "full",
    "language_reversed": "full",
    "language_no_lora": "no_lora",
    "language_no_token_slots": "no_token_slots",
    "language_no_composition": "no_composition",
}
CONTROLS = (
    "numeric_canonical",
    "language_reversed",
    "language_no_lora",
    "language_no_token_slots",
    "language_no_composition",
)

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
    prepare.add_argument("--known-source", action="append", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    freeze = stages.add_parser("freeze")
    freeze.add_argument("--replicate-index", required=True, type=int)
    freeze.add_argument("--prepare-summary", required=True, type=Path)
    freeze.add_argument("--generation-conditions", required=True, type=Path)
    freeze.add_argument("--v6-basis-bundle", required=True, type=Path)
    freeze.add_argument("--representation-checkpoint", required=True, type=Path)
    freeze.add_argument("--representation-summary", required=True, type=Path)
    freeze.add_argument("--canonical-checkpoint", required=True, type=Path)
    freeze.add_argument("--sft-adapter-dir", required=True, type=Path)
    freeze.add_argument("--v5-root", required=True, type=Path)
    freeze.add_argument("--v5-gate", required=True, type=Path)
    freeze.add_argument("--v5-unlock", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--device", default="auto")

    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--replicate-index", required=True, type=int)
    evaluate.add_argument("--prepare-summary", required=True, type=Path)
    evaluate.add_argument("--evaluation-targets", required=True, type=Path)
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    gate = stages.add_parser("gate")
    gate.add_argument("--prepare-summary", required=True, type=Path)
    gate.add_argument("--evaluation-root", required=True, type=Path)
    gate.add_argument("--output-dir", required=True, type=Path)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "amended_after_source_availability_only_before_candidate_generation",
        "arms": list(ARMS),
        "training": False,
        "fresh_condition_count": 20,
        "fresh_property_count_quotas": {"3": 20},
        "protocol_amendment": {
            "trigger": "strict_source_disjoint_2p_quota_unavailable",
            "observed_eligible_history": {"v7_2p": 21, "v7_3p": 43, "v7_used_2p": 20, "v7_used_3p": 20},
            "failed_prepare_job": 20229890,
            "candidate_generation_started": False,
            "property_oracle_accessed": False,
            "scientific_metrics_accessed": False,
            "change": "replace_12x2p_plus_12x3p_with_20x3p",
            "science_gates_changed": False,
        },
        "paired_common_random_numbers": True,
        "exact_raw_attempts_per_condition": 20,
        "candidate_pool_before_selection": 20,
        "primary_metric_family": "candidate_level_distributional_success",
        "any20_role": "secondary_capability_metric",
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "official_test_access": False,
        "single_frozen_method": True,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"V9 preregistration drift: {drift}")
    actual = file_sha256(Path(__file__).resolve())
    if str(payload.get("implementation_sha256")) != actual:
        raise ValueError(
            f"V9 implementation drift: expected {payload.get('implementation_sha256')}, found {actual}"
        )
    horizon_actual = file_sha256(SCRIPT_DIR / "horizon_closed_graph_jump.py")
    if str(payload.get("horizon_closure_sha256")) != horizon_actual:
        raise ValueError("V9 horizon-closure implementation drift")
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locked = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V9 locked inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locked.get(name), "actual": digest}
        for name, digest in actual.items()
        if locked.get(name) != digest
    }
    if drift:
        raise ValueError(f"V9 locked-input drift: {drift}")
    return actual


def stable_key(seed: int, *values: object) -> str:
    text = "\0".join([str(seed), *[str(value) for value in values]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sources_from_path(path: Path) -> set[str]:
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        payload = read_json(path)
        rows = list(payload.get("records") or [])
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    values = set()
    for raw in rows:
        source = graph.canonical_smiles(str(dict(raw).get("source_smiles", "") or ""))
        if source:
            values.add(source)
    return values


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
                f"Fresh V9 quota unavailable for {count}p: wanted {quotas[count]}, found {found}"
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
        raise ValueError(f"Completed V9 prepare exists: {summary_path}")
    known_names = sorted(
        name for name in dict(preregistration["locked_inputs"]) if name.startswith("known_source_")
    )
    if len(known_names) != len(args.known_source):
        raise ValueError("V9 known-source list does not match preregistration")
    paths = {
        "train_csv_sha256": args.train_csv,
        "validation_csv_sha256": args.validation_csv,
        "b36_records_sha256": args.b36_records,
        "predecessor_fit_bundle_sha256": args.predecessor_fit_bundle,
        "e1_manifest_sha256": args.e1_manifest,
        "v6_basis_bundle_sha256": args.v6_basis_bundle,
        **{
            name: value
            for name, value in zip(known_names, args.known_source, strict=True)
        },
    }
    hashes = check_locked_inputs(preregistration, paths)
    bundle = torch.load(args.v6_basis_bundle, map_location="cpu", weights_only=False)
    if (
        bundle.get("protocol") != v6.PROTOCOL
        or bundle.get("role") != "target_free_frozen_generation_basis_and_support"
    ):
        raise ValueError("V9 requires the frozen target-free V6 basis bundle")
    predecessor = fresh.load_frozen_predecessor_bundle(args.predecessor_fit_bundle)
    forbidden = {
        graph.canonical_smiles(pair.source_smiles) for pair in predecessor["pairs"]
    }
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
        condition_id = f"v9_semantic_fresh_{index:04d}"
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
    target_path = args.output_dir / "sealed_evaluation_targets.json"
    generation_text = json.dumps(
        {
            "protocol": PROTOCOL,
            "role": "fresh_constraint_text_and_sources_without_targets",
            "records": generation_records,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    forbidden_terms = ("target_smiles", "target_properties", "strict_success", "oracle")
    leaks = [term for term in forbidden_terms if term in generation_text.lower()]
    if leaks:
        raise ValueError(f"V9 target-free generation manifest leaked {leaks}")
    generation_path.write_text(generation_text, encoding="utf-8")
    target_path.write_text(
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
        graph.canonical_smiles(str(record["source_smiles"])) for record in generation_records
    }
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare_prospective_target_isolated_manifests",
        "execution_status": "completed",
        "decision": "freeze_paired_semantic_interventions",
        "fresh_conditions": len(selected),
        "fresh_unique_sources": len(selected_sources),
        "fresh_forbidden_source_overlap": len(selected_sources & forbidden),
        "selection": selection,
        "known_source_counts": known_counts,
        "artifacts": {
            "generation_conditions_sha256": file_sha256(generation_path),
            "evaluation_targets_sha256": file_sha256(target_path),
            "locked_inputs": hashes,
        },
        "contract": {
            "fresh_target_process_isolation": True,
            "generation_target_access": False,
            "official_test_access": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        raise ValueError("Invalid target-free V9 generation manifest")
    records = [dict(record) for record in payload["records"]]
    pairs = []
    for expected_index, record in enumerate(records):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("V9 generation order drift")
        source = graph.molecule_example(
            str(record["source_smiles"]),
            int(preregistration["max_atoms"]),
            int(preregistration["fingerprint_bits"]),
        )
        if source is None:
            raise ValueError(f"Cannot materialize V9 source {expected_index}")
        row = {str(key): str(value) for key, value in dict(record["condition_row"]).items()}
        pairs.append(
            SimpleNamespace(
                row=row,
                source_smiles=str(record["source_smiles"]),
                source=source,
                condition=np.zeros(
                    (int(preregistration["token_count"]), int(preregistration["condition_dim"])),
                    dtype=np.float32,
                ),
                property_count=int(record["property_count"]),
                task=str(record["task"]),
            )
        )
    return pairs, records


def v5_static_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "representation_checkpoint_sha256": args.representation_checkpoint,
        "representation_summary_sha256": args.representation_summary,
        "canonical_checkpoint_sha256": args.canonical_checkpoint,
        "common_sft_adapter_config_sha256": args.sft_adapter_dir / "adapter_config.json",
        "common_sft_adapter_model_sha256": args.sft_adapter_dir / "adapter_model.safetensors",
        "v5_gate_sha256": args.v5_gate,
        "v5_unlock_sha256": args.v5_unlock,
    }
    for arm in ("full", "no_lora", "no_token_slots", "no_composition"):
        root = args.v5_root / arm
        paths[f"v5_{arm}_router_sha256"] = root / "structured_sparse_router.pt"
        paths[f"v5_{arm}_summary_sha256"] = root / "summary.json"
        if arm != "no_lora":
            paths[f"v5_{arm}_lora_config_sha256"] = root / "lora_adapter" / "adapter_config.json"
            paths[f"v5_{arm}_lora_model_sha256"] = root / "lora_adapter" / "adapter_model.safetensors"
    return paths


def load_language_router(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
    arm: str,
):
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError(f"Missing PEFT for V9 inference: {exc}") from exc
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm, tokenizer = semantic.operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=False
    )
    llm = llm.merge_and_unload()
    arm_root = args.v5_root / arm
    if arm != "no_lora":
        llm = peft.PeftModel.from_pretrained(
            llm,
            arm_root / "lora_adapter",
            is_trainable=False,
            adapter_name=f"v9_{arm}",
        ).to(device)
    checkpoint = torch.load(
        arm_root / "structured_sparse_router.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("protocol") != v5.PROTOCOL or checkpoint.get("arm") != arm:
        raise ValueError(f"V9 V5 checkpoint drift for {arm}")
    router = v5.MassConservingPropertySetRouter(
        int(checkpoint["llm_hidden_dim"]),
        int(checkpoint["slot_dim"]),
        len(checkpoint["property_columns"]),
        int(checkpoint["max_instruction_cardinality"]),
        bool(checkpoint["use_token_slots"]),
    ).to(device)
    router.load_state_dict(dict(checkpoint["state_dict"]), strict=True)
    llm.eval().requires_grad_(False)
    router.eval().requires_grad_(False)
    return llm, tokenizer, router, [str(name) for name in checkpoint["property_columns"]]


def replicate_seed(preregistration: Mapping[str, object], index: int) -> int:
    seeds = [int(seed) for seed in preregistration["replicate_seeds"]]
    if index < 0 or index >= len(seeds):
        raise ValueError(f"Invalid V9 replicate index: {index}")
    return seeds[index]


def run_freeze(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    seed = replicate_seed(preregistration, int(args.replicate_index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_summary = args.output_dir / "summary.json"
    if root_summary.exists():
        raise ValueError(f"Completed V9 freeze exists: {root_summary}")
    hashes = check_locked_inputs(
        preregistration,
        {"v6_basis_bundle_sha256": args.v6_basis_bundle, **v5_static_paths(args)},
    )
    v6.validate_v5_gate(args.v5_gate, args.v5_unlock)
    prepare = read_json(args.prepare_summary)
    if prepare.get("protocol") != PROTOCOL:
        raise ValueError("V9 prepare protocol drift")
    if file_sha256(args.generation_conditions) != dict(prepare["artifacts"])["generation_conditions_sha256"]:
        raise ValueError("V9 generation manifest drift")
    bundle = torch.load(args.v6_basis_bundle, map_location="cpu", weights_only=False)
    if bundle.get("protocol") != v6.PROTOCOL:
        raise ValueError("V9 V6 basis protocol drift")
    device = base.resolve_device(str(args.device))
    base.seed_everything(seed)
    pairs, records = load_generation_pairs(args.generation_conditions, preregistration)
    expected_conditions = int(preregistration["fresh_condition_count"])
    if len(pairs) != expected_conditions:
        raise ValueError(f"V9 expected {expected_conditions} generation conditions")

    language_tokens: dict[str, np.ndarray] = {}
    routing: dict[str, object] = {}
    property_columns_reference: list[str] | None = None
    token_shape = tuple(int(value) for value in bundle["token_shape"])
    for output_arm, router_arm in ROUTER_ARMS.items():
        llm, tokenizer, router, property_columns = load_language_router(
            args, preregistration, device, router_arm
        )
        if property_columns_reference is None:
            property_columns_reference = property_columns
        if property_columns != property_columns_reference or property_columns != list(bundle["property_columns"]):
            raise ValueError(f"V9 property vocabulary drift for {output_arm}")
        variant = "reversed" if output_arm == "language_reversed" else "matched"
        examples = v6.examples_for_records(records, property_columns, variant)
        coefficients, support_mask, cardinality = v5.predict_examples(
            llm, router, tokenizer, examples, preregistration, device
        )
        routing[output_arm] = v5.routing_metrics(
            coefficients, support_mask, cardinality, examples
        )
        language_tokens[output_arm] = property_basis.compose_tokens(
            coefficients, bundle["basis"], token_shape
        ).numpy()
        del llm, tokenizer, router, coefficients, support_mask, cardinality
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
            raise ValueError(f"Completed V9 arm exists: {arm_dir}")
        rows: list[dict[str, object]] = []
        started = time.perf_counter()
        for index, pair in enumerate(pairs):
            if arm == "numeric_canonical":
                pair.condition = hierarchical.property_latent_slot_tokens(
                    pair.row, int(preregistration["condition_dim"])
                )
            else:
                pair.condition = np.asarray(language_tokens[arm][index], dtype=np.float32)
            particle_seed = seed * 100000 + index
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
                particle_seed,
            )
            if len(generated) != 20:
                raise RuntimeError(f"V9 {arm} did not emit exactly 20 attempts")
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
                        "replicate_index": int(args.replicate_index),
                        "replicate_seed": seed,
                        "paired_particle_seed": particle_seed,
                        **candidate,
                    }
                )
            if (index + 1) % 8 == 0 or index + 1 == len(pairs):
                print(
                    json.dumps(
                        {
                            "stage": "v9_freeze_progress",
                            "replicate_index": int(args.replicate_index),
                            "arm": arm,
                            "conditions": index + 1,
                            "total": len(pairs),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        expected = len(pairs) * 20
        if len(rows) != expected or any(not str(row["generated_smiles"]) for row in rows):
            raise RuntimeError(f"V9 {arm} frozen candidate contract failed")
        base.write_candidate_rows(candidate_path, rows)
        arm_summary = {
            "protocol": PROTOCOL,
            "stage": "paired_fresh_target_isolated_freeze",
            "execution_status": "completed",
            "replicate_index": int(args.replicate_index),
            "replicate_seed": seed,
            "arm": arm,
            "conditions": len(pairs),
            "candidate_rows": len(rows),
            "attempts_per_condition": 20,
            "mean_unique_smiles": v6.mean_unique_smiles(rows),
            "elapsed_sec": round(time.perf_counter() - started, 1),
            "artifacts": {"frozen_candidates_sha256": file_sha256(candidate_path)},
            "contract": {
                "generation_target_path_accepted": False,
                "generation_property_oracle_access": False,
                "paired_common_random_numbers": True,
                "exact_raw_attempts_per_condition": 20,
                "candidate_pool_before_selection": 20,
                "molecular_candidate_ranking": False,
                "oracle_selection": False,
                "retry_or_resampling": False,
                "posthoc_molecule_repair": False,
            },
        }
        summary_path.write_text(json.dumps(arm_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "protocol": PROTOCOL,
        "stage": "paired_all_semantic_interventions_frozen",
        "execution_status": "completed",
        "decision": "await_post_freeze_evaluation",
        "replicate_index": int(args.replicate_index),
        "replicate_seed": seed,
        "arms": list(ARMS),
        "conditions": len(pairs),
        "candidate_rows_per_arm": len(pairs) * 20,
        "routing": routing,
        "representation_protocol": representation_summary.get("protocol"),
        "artifacts": {"locked_inputs": hashes},
        "contract": {
            "generation_target_access": False,
            "paired_common_random_numbers": True,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
        },
    }
    root_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def load_evaluation_pairs(path: Path, preregistration: Mapping[str, object]) -> list[object]:
    payload = read_json(path)
    if payload.get("protocol") != PROTOCOL or payload.get("role") != "sealed_post_freeze_fresh_targets":
        raise ValueError("Invalid sealed V9 evaluation targets")
    pairs = []
    for expected_index, record in enumerate(payload["records"]):
        if int(record["pair_index"]) != expected_index:
            raise ValueError("V9 evaluation order drift")
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
            raise ValueError(f"Cannot reconstruct sealed V9 pair {expected_index}")
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


def condition_metrics(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    output = []
    for condition_id in sorted(grouped):
        values = grouped[condition_id]
        output.append(
            {
                "condition_id": condition_id,
                "property_count": int(values[0]["property_count"]),
                "candidate_property_success": float(np.mean([bool(row["property_success"]) for row in values])),
                "candidate_strict_success": float(np.mean([bool(row["strict_success"]) for row in values])),
                "mean_property_fraction": float(np.mean([float(row["property_fraction"]) for row in values])),
                "property_any20": any(bool(row["property_success"]) for row in values),
                "strict_any20": any(bool(row["strict_success"]) for row in values),
            }
        )
    return output


def augment_metrics(
    metrics: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    output = dict(metrics)
    output["candidate_property_success"] = float(np.mean([bool(row["property_success"]) for row in rows]))
    output["candidate_strict_success"] = float(np.mean([bool(row["strict_success"]) for row in rows]))
    output["mean_property_fraction"] = float(np.mean([float(row["property_fraction"]) for row in rows]))
    output["by_property_count"] = {}
    for count in sorted({int(row["property_count"]) for row in rows}):
        subset = [row for row in rows if int(row["property_count"]) == count]
        conditions = condition_metrics(subset)
        output["by_property_count"][str(count)] = {
            "candidate_rows": len(subset),
            "candidate_property_success": float(np.mean([bool(row["property_success"]) for row in subset])),
            "candidate_strict_success": float(np.mean([bool(row["strict_success"]) for row in subset])),
            "mean_property_fraction": float(np.mean([float(row["property_fraction"]) for row in subset])),
            "property_any20": float(np.mean([bool(row["property_any20"]) for row in conditions])),
            "strict_any20": float(np.mean([bool(row["strict_any20"]) for row in conditions])),
        }
    output["condition_metrics"] = condition_metrics(rows)
    return output


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    seed = replicate_seed(preregistration, int(args.replicate_index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V9 evaluation exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    if prepare.get("protocol") != PROTOCOL:
        raise ValueError("V9 evaluation prepare drift")
    if file_sha256(args.evaluation_targets) != dict(prepare["artifacts"])["evaluation_targets_sha256"]:
        raise ValueError("V9 sealed target drift")
    pairs = load_evaluation_pairs(args.evaluation_targets, preregistration)
    expected = int(preregistration["fresh_condition_count"]) * 20
    metrics_by_arm: dict[str, object] = {}
    for arm in ARMS:
        arm_dir = args.frozen_root / arm
        arm_summary = read_json(arm_dir / "summary.json")
        candidate_path = arm_dir / "frozen_candidates.csv"
        if (
            arm_summary.get("protocol") != PROTOCOL
            or int(arm_summary.get("replicate_seed", -1)) != seed
            or file_sha256(candidate_path) != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]
        ):
            raise ValueError(f"V9 frozen artifact drift: {arm}")
        frozen = semantic.coerce_frozen_rows(candidate_path)
        if len(frozen) != expected:
            raise ValueError(f"V9 {arm} expected {expected} frozen rows")
        evaluated, raw_metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = augment_metrics(raw_metrics, evaluated)
        metrics["by_task"] = v6.task_breakdown(evaluated)
        evaluated_path = args.output_dir / f"evaluated_{arm}.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        metrics_by_arm[arm] = {
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
    summary = {
        "protocol": PROTOCOL,
        "stage": "paired_fresh_post_freeze_evaluation",
        "execution_status": "completed",
        "decision": "await_separate_science_gate",
        "replicate_index": int(args.replicate_index),
        "replicate_seed": seed,
        "arms": metrics_by_arm,
        "contract": {
            "generation_target_access": False,
            "post_freeze_target_access": True,
            "paired_common_random_numbers": True,
            "exact_raw_attempts_per_condition": 20,
            "science_decision_in_separate_process": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


METRIC_NAMES = (
    "candidate_rows",
    "attempted_per_condition",
    "validity",
    "mean_source_tanimoto",
    "mean_unique_valid",
    "candidate_property_success",
    "candidate_strict_success",
    "mean_property_fraction",
    "property_any20",
    "strict_any20",
)


def mean_metrics(evaluations: Sequence[Mapping[str, object]], arm: str) -> dict[str, float]:
    rows = [dict(dict(dict(item["arms"])[arm])["metrics"]) for item in evaluations]
    return {
        name: float(sum(float(row[name]) for row in rows) / len(rows))
        for name in METRIC_NAMES
    }


def paired_differences(
    evaluations: Sequence[Mapping[str, object]], control: str, metric: str
) -> tuple[list[float], list[float]]:
    cluster_differences = []
    replicate_differences = []
    for payload in evaluations:
        full_rows = list(dict(dict(dict(payload["arms"])["language_full"])["metrics"])["condition_metrics"])
        control_rows = list(dict(dict(dict(payload["arms"])[control])["metrics"])["condition_metrics"])
        full_lookup = {str(row["condition_id"]): float(row[metric]) for row in full_rows}
        control_lookup = {str(row["condition_id"]): float(row[metric]) for row in control_rows}
        if set(full_lookup) != set(control_lookup):
            raise ValueError(f"V9 paired condition drift for {control}")
        values = [full_lookup[key] - control_lookup[key] for key in sorted(full_lookup)]
        cluster_differences.extend(values)
        replicate_differences.append(float(np.mean(values)))
    return cluster_differences, replicate_differences


def bootstrap_interval(values: Sequence[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def run_gate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "gate_summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V9 science gate exists: {summary_path}")
    prepare = read_json(args.prepare_summary)
    if (
        prepare.get("protocol") != PROTOCOL
        or int(prepare.get("fresh_conditions", -1)) != int(preregistration["fresh_condition_count"])
        or int(prepare.get("fresh_forbidden_source_overlap", -1)) != 0
    ):
        raise ValueError("Incomplete V9 prospective prepare artifact")
    evaluations = []
    for index, seed in enumerate(preregistration["replicate_seeds"]):
        path = args.evaluation_root / f"replicate_{index}" / "evaluation" / "summary.json"
        payload = read_json(path)
        if (
            payload.get("protocol") != PROTOCOL
            or payload.get("execution_status") != "completed"
            or int(payload.get("replicate_seed", -1)) != int(seed)
        ):
            raise ValueError(f"Incomplete V9 evaluation: {path}")
        evaluations.append(payload)
    metrics = {arm: mean_metrics(evaluations, arm) for arm in ARMS}
    effects: dict[str, object] = {}
    replicate_effects: dict[str, object] = {}
    for control_index, control in enumerate(CONTROLS):
        effects[control] = {}
        replicate_effects[control] = {}
        for metric in ("candidate_property_success", "candidate_strict_success", "mean_property_fraction"):
            clusters, replicates = paired_differences(evaluations, control, metric)
            effects[control][metric] = bootstrap_interval(
                clusters, int(preregistration["bootstrap_seed"]) + control_index * 10 + len(metric)
            )
            replicate_effects[control][metric] = replicates
    gates = dict(preregistration["science_gates"])
    expected_rows = int(preregistration["fresh_condition_count"]) * 20
    full = metrics["language_full"]
    checks = {
        "fresh_source_contract": int(prepare["fresh_forbidden_source_overlap"]) == 0,
        "replicate_count": len(evaluations) == len(preregistration["replicate_seeds"]),
        "candidate_rows": all(math.isclose(metrics[arm]["candidate_rows"], expected_rows) for arm in ARMS),
        "exact_attempts": all(math.isclose(metrics[arm]["attempted_per_condition"], 20.0) for arm in ARMS),
        "all_arm_validity": all(metrics[arm]["validity"] >= float(gates["validity"]) for arm in ARMS),
        "language_source_tanimoto": full["mean_source_tanimoto"] >= float(gates["mean_source_tanimoto"]),
        "language_unique": full["mean_unique_valid"] >= float(gates["mean_unique_valid"]),
        "language_candidate_property": full["candidate_property_success"] >= float(gates["candidate_property_success"]),
        "language_candidate_strict": full["candidate_strict_success"] >= float(gates["candidate_strict_success"]),
        "numeric_noninferior_property": float(effects["numeric_canonical"]["candidate_property_success"]["ci95_low"]) >= float(gates["numeric_noninferiority_margin"]),
        "numeric_noninferior_strict": float(effects["numeric_canonical"]["candidate_strict_success"]["ci95_low"]) >= float(gates["numeric_noninferiority_margin"]),
    }
    for control in CONTROLS[1:]:
        short = control.removeprefix("language_")
        property_effect = dict(effects[control]["candidate_property_success"])
        strict_effect = dict(effects[control]["candidate_strict_success"])
        checks[f"{short}_property_mean"] = float(property_effect["mean"]) >= float(gates["causal_property_delta"])
        checks[f"{short}_strict_mean"] = float(strict_effect["mean"]) >= float(gates["causal_strict_delta"])
        checks[f"{short}_property_ci"] = float(property_effect["ci95_low"]) > 0.0
        checks[f"{short}_strict_ci"] = float(strict_effect["ci95_low"]) > 0.0
        checks[f"{short}_replicate_consistency"] = sum(
            value > 0.0 for value in replicate_effects[control]["candidate_property_success"]
        ) >= int(gates["positive_replicates"])
    failures = [name for name, passed in checks.items() if not bool(passed)]
    passed = not failures
    summary = {
        "protocol": GATE_PROTOCOL,
        "execution_status": "completed",
        "science_gate": {
            "passed": passed,
            "checks": checks,
            "thresholds": gates,
            "failures": failures,
        },
        "decision": (
            "unlock_cross_benchmark_language_graph_latent_evaluation"
            if passed
            else "stop_semantic_intervention_line_before_benchmark_expansion"
        ),
        "metrics": metrics,
        "paired_effects": effects,
        "replicate_effects": replicate_effects,
        "secondary_metric_note": "any@20 is reported but is not a causal gate because V8 showed saturation",
        "contract": {
            "scientific_stop_exits_zero": True,
            "portal_engineering_state_independent_of_science_decision": True,
            "candidate_level_metrics_primary": True,
            "official_test_access": False,
            "automatic_benchmark_submission": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    raise ValueError(f"Unsupported V9 stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
