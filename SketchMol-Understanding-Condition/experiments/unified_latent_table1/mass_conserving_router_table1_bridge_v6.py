#!/usr/bin/env python3
"""Bridge the frozen V5 Common-LLM router into real graph generation.

The prepare process fits the already-defined train-only property-token basis and
writes a target-free generation bundle.  The freeze process accepts no target
path: it compiles matched or direction-reversed constraint language with the
frozen V5 LoRA/router, injects those coefficients into the frozen B41 graph
jump, and emits exactly 20 raw attempts per condition without ranking or retry.
The evaluation process opens the sealed targets only after all candidates have
been hashed.  A separate gate process turns metrics into a scientific decision;
all successfully executed stages exit zero even when that decision is STOP.
"""

from __future__ import annotations

import argparse
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

import mass_conserving_property_set_router_v5 as v5  # noqa: E402
import semantic_energy_graph_jump_v1 as semantic  # noqa: E402
from dead_end_safe_support import DeadEndSafeSupport  # noqa: E402


PROTOCOL = "target_isolated_mass_conserving_router_table1_bridge_v6"
SOURCE_PROTOCOL = "train_only_semantic_energy_graph_jump_v1"
ARMS = ("numeric_canonical", "language_full", "language_reversed")

base = semantic.base
graph = semantic.graph
b41 = semantic.b41
b40 = semantic.b40
hierarchical = semantic.hierarchical
valid_terminal = semantic.valid_terminal
v3 = v5.v4.v3
property_basis = v3.v1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare = stages.add_parser("prepare")
    prepare.add_argument("--fit-probe-bundle", required=True, type=Path)
    prepare.add_argument("--generation-conditions", required=True, type=Path)
    prepare.add_argument("--v5-summary", required=True, type=Path)
    prepare.add_argument("--v5-gate", required=True, type=Path)
    prepare.add_argument("--v5-unlock", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    freeze = stages.add_parser("freeze")
    freeze.add_argument("--prepare-summary", required=True, type=Path)
    freeze.add_argument("--generation-bundle", required=True, type=Path)
    freeze.add_argument("--generation-conditions", required=True, type=Path)
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
    evaluate.add_argument("--evaluation-targets", required=True, type=Path)
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    gate = stages.add_parser("gate")
    gate.add_argument("--evaluation-summary", required=True, type=Path)
    gate.add_argument("--output-dir", required=True, type=Path)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "task_replay": "moledit_table1_internal_target_isolated_bridge",
        "arms": list(ARMS),
        "training": False,
        "exact_raw_attempts_per_condition": 20,
        "candidate_pool_before_selection": 20,
        "generation_target_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "single_seed": True,
        "official_test_access": False,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"V6 preregistration drift: {drift}")
    implementation = file_sha256(Path(__file__).resolve())
    expected = str(payload.get("implementation_sha256", ""))
    if expected and implementation != expected:
        raise ValueError(
            f"V6 implementation drift: expected {expected}, found {implementation}"
        )
    return payload


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locked = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V6 locked inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locked.get(name), "actual": digest}
        for name, digest in actual.items()
        if locked.get(name) != digest
    }
    if drift:
        raise ValueError(f"V6 locked-input drift: {drift}")
    return actual


def validate_v5_gate(gate_path: Path, unlock_path: Path) -> None:
    gate = read_json(gate_path)
    unlock = read_json(unlock_path)
    if gate.get("decision") != "unlock_target_isolated_exact_n20_generation":
        raise ValueError("V5 science gate did not unlock generation")
    expected = {
        "protocol": "target_isolated_exact_n20_generation_unlock_v5",
        "status": "unlocked_not_executed",
        "exact_raw_attempts_per_condition": 20,
        "candidate_pool_before_selection": 20,
        "generation_target_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "single_seed": True,
    }
    drift = {
        key: {"expected": value, "actual": unlock.get(key)}
        for key, value in expected.items()
        if unlock.get(key) != value
    }
    if drift:
        raise ValueError(f"V5 generation unlock drift: {drift}")
    if str(unlock.get("source_gate_sha256")) != file_sha256(gate_path):
        raise ValueError("V5 unlock source-gate digest drift")


def run_prepare(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    bundle_path = args.output_dir / "target_free_generation_bundle.pt"
    if summary_path.exists() or bundle_path.exists():
        raise ValueError(f"Completed V6 prepare artifact exists: {args.output_dir}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "fit_probe_bundle_sha256": args.fit_probe_bundle,
            "generation_conditions_sha256": args.generation_conditions,
            "v5_summary_sha256": args.v5_summary,
            "v5_gate_sha256": args.v5_gate,
            "v5_unlock_sha256": args.v5_unlock,
        },
    )
    validate_v5_gate(args.v5_gate, args.v5_unlock)
    source = read_json(args.generation_conditions)
    if (
        source.get("protocol") != SOURCE_PROTOCOL
        or source.get("role") != "constraint_text_and_sources_without_targets"
    ):
        raise ValueError("Invalid target-free V6 generation-condition source")
    records = list(source["records"])
    expected_conditions = int(preregistration["conditions"])
    if len(records) != expected_conditions:
        raise ValueError(
            f"V6 expected {expected_conditions} generation records, found {len(records)}"
        )
    fit_bundle = torch.load(args.fit_probe_bundle, map_location="cpu", weights_only=False)
    pairs = list(fit_bundle["pairs"])
    train_indices = list(fit_bundle["train_indices"])
    property_columns = [str(name) for name in semantic.unified.PROPERTY_COLUMNS]
    matched_targets = property_basis.coefficient_targets(pairs, property_columns)["matched"]
    basis = property_basis.fit_property_token_basis(
        pairs,
        train_indices,
        matched_targets,
        float(preregistration["basis_ridge"]),
    )
    token_shape = tuple(int(value) for value in np.asarray(pairs[0].condition).shape)
    expected_shape = (
        int(preregistration["token_count"]),
        int(preregistration["condition_dim"]),
    )
    if token_shape != expected_shape:
        raise ValueError(f"V6 token shape drift: {token_shape} != {expected_shape}")
    torch.save(
        {
            "protocol": PROTOCOL,
            "role": "target_free_frozen_generation_basis_and_support",
            "basis": basis,
            "property_columns": property_columns,
            "token_shape": token_shape,
            "vocabulary": dict(fit_bundle["vocabulary"]),
            "support": dict(fit_bundle["support"]),
            "fit_indices_used": len(train_indices),
            "generation_condition_ids": [str(record["condition_id"]) for record in records],
        },
        bundle_path,
    )
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare_target_free_generation_bundle",
        "execution_status": "completed",
        "conditions": len(records),
        "fit_indices_used": len(train_indices),
        "artifacts": {
            "generation_bundle_sha256": file_sha256(bundle_path),
            "locked_inputs": input_hashes,
        },
        "contract": {
            "generation_target_access": False,
            "evaluation_target_path_accepted": False,
            "training": False,
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
        payload.get("protocol") != SOURCE_PROTOCOL
        or payload.get("role") != "constraint_text_and_sources_without_targets"
    ):
        raise ValueError("Invalid target-free generation manifest")
    records = list(payload["records"])
    pairs = []
    for expected_index, raw in enumerate(records):
        record = dict(raw)
        if int(record["pair_index"]) != expected_index:
            raise ValueError("V6 generation-condition order drift")
        source = graph.molecule_example(
            str(record["source_smiles"]),
            int(preregistration["max_atoms"]),
            int(preregistration["fingerprint_bits"]),
        )
        if source is None:
            raise ValueError(f"Cannot materialize V6 source {expected_index}")
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


def examples_for_records(
    records: Sequence[Mapping[str, object]],
    property_columns: Sequence[str],
    variant: str,
) -> list[object]:
    lookup = {name: index for index, name in enumerate(property_columns)}
    examples = []
    for index, record in enumerate(records):
        row = dict(record["condition_row"])
        target = torch.zeros(len(property_columns), dtype=torch.float32)
        for name, direction in semantic.specs_for_row(row):
            target[lookup[str(name)]] = float(direction)
        if variant == "reversed":
            target = -target
        text = str(dict(record["instructions"])["reversed" if variant == "reversed" else "matched"])
        examples.append(
            v3.TextExample(
                text=text,
                target=target,
                phrases={},
                key=f"v6_{variant}_{index:04d}",
            )
        )
    return examples


def load_v5_language_router(
    args: argparse.Namespace,
    preregistration: Mapping[str, object],
    device: torch.device,
):
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError(f"Missing PEFT for V6 inference: {exc}") from exc
    llm_args = SimpleNamespace(sft_adapter_dir=args.sft_adapter_dir)
    llm, tokenizer = semantic.operator.load_common_llm(
        llm_args, preregistration, device, sft=True, latent_lora=False
    )
    llm = llm.merge_and_unload()
    llm = peft.PeftModel.from_pretrained(
        llm,
        args.v5_lora_adapter_dir,
        is_trainable=False,
        adapter_name="v5_property_router",
    ).to(device)
    checkpoint = torch.load(
        args.v5_router_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("protocol") != v5.PROTOCOL or checkpoint.get("arm") != "full":
        raise ValueError("V5 full-router checkpoint protocol drift")
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


def mean_unique_smiles(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = str(row.get("generated_smiles", "") or "")
        if value:
            grouped[str(row["condition_id"])].add(value)
    return float(np.mean([len(values) for values in grouped.values()])) if grouped else 0.0


def run_freeze(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_summary = args.output_dir / "summary.json"
    if root_summary.exists():
        raise ValueError(f"Completed V6 freeze exists: {root_summary}")
    input_hashes = check_locked_inputs(
        preregistration,
        {
            "generation_conditions_sha256": args.generation_conditions,
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
    validate_v5_gate(args.v5_gate, args.v5_unlock)
    prepare = read_json(args.prepare_summary)
    if prepare.get("protocol") != PROTOCOL:
        raise ValueError("V6 prepare-summary protocol drift")
    if file_sha256(args.generation_bundle) != dict(prepare["artifacts"])["generation_bundle_sha256"]:
        raise ValueError("V6 target-free generation bundle drift")
    bundle = torch.load(args.generation_bundle, map_location="cpu", weights_only=False)
    if (
        bundle.get("protocol") != PROTOCOL
        or bundle.get("role") != "target_free_frozen_generation_basis_and_support"
    ):
        raise ValueError("Invalid V6 target-free generation bundle")
    device = base.resolve_device(str(args.device))
    base.seed_everything(int(preregistration["generation_seed"]))
    pairs, records = load_generation_pairs(args.generation_conditions, preregistration)
    llm, tokenizer, router, property_columns = load_v5_language_router(
        args, preregistration, device
    )
    if property_columns != list(bundle["property_columns"]):
        raise ValueError("V6 property vocabulary drift")
    matched_examples = examples_for_records(records, property_columns, "matched")
    reversed_examples = examples_for_records(records, property_columns, "reversed")
    matched_coefficients, matched_support, matched_cardinality = v5.predict_examples(
        llm, router, tokenizer, matched_examples, preregistration, device
    )
    reversed_coefficients, reversed_support, reversed_cardinality = v5.predict_examples(
        llm, router, tokenizer, reversed_examples, preregistration, device
    )
    routing = {
        "language_full": v5.routing_metrics(
            matched_coefficients, matched_support, matched_cardinality, matched_examples
        ),
        "language_reversed": v5.routing_metrics(
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
    basis = bundle["basis"]
    language_tokens = {
        "language_full": property_basis.compose_tokens(
            matched_coefficients, basis, token_shape
        ).numpy(),
        "language_reversed": property_basis.compose_tokens(
            reversed_coefficients, basis, token_shape
        ).numpy(),
    }
    model, representation, _config, representation_summary = semantic.load_graph_stack(
        args, preregistration, bundle, device
    )
    vocabulary = dict(bundle["vocabulary"])
    support = dict(bundle["support"])
    support_tensors = b40._device_support(support, device)
    safe_support = DeadEndSafeSupport(valid_terminal.ExactMoleculeStopSupport(vocabulary))
    original_support = b41.viability_event_mask
    try:
        b41.viability_event_mask = safe_support
        for arm in ARMS:
            arm_dir = args.output_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = arm_dir / "frozen_candidates.csv"
            summary_path = arm_dir / "summary.json"
            if candidate_path.exists() or summary_path.exists():
                raise ValueError(f"Completed V6 arm exists: {arm_dir}")
            rows = []
            started = time.perf_counter()
            for index, pair in enumerate(pairs):
                if arm == "numeric_canonical":
                    pair.condition = hierarchical.property_latent_slot_tokens(
                        pair.row, int(preregistration["condition_dim"])
                    )
                else:
                    pair.condition = np.asarray(language_tokens[arm][index], dtype=np.float32)
                generated = b41.sample_from_source(
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
                if len(generated) != int(preregistration["exact_raw_attempts_per_condition"]):
                    raise RuntimeError(f"{arm} did not emit exactly 20 attempts")
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
                                "stage": "v6_freeze_progress",
                                "arm": arm,
                                "conditions": index + 1,
                                "total": len(pairs),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
            if len(rows) != expected:
                raise RuntimeError(f"{arm} expected {expected} rows, found {len(rows)}")
            base.write_candidate_rows(candidate_path, rows)
            arm_summary = {
                "protocol": PROTOCOL,
                "stage": "target_isolated_exact_n20_freeze",
                "execution_status": "completed",
                "arm": arm,
                "conditions": len(pairs),
                "candidate_rows": len(rows),
                "attempts_per_condition": 20,
                "mean_unique_smiles": mean_unique_smiles(rows),
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
                },
            }
            summary_path.write_text(
                json.dumps(arm_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        b41.viability_event_mask = original_support
    summary = {
        "protocol": PROTOCOL,
        "stage": "all_arms_frozen",
        "execution_status": "completed",
        "decision": "await_post_freeze_evaluation",
        "arms": list(ARMS),
        "conditions": len(pairs),
        "candidate_rows_per_arm": len(pairs) * 20,
        "routing": routing,
        "representation_protocol": representation_summary.get("protocol"),
        "artifacts": {"locked_inputs": input_hashes},
        "contract": {
            "generation_target_access": False,
            "exact_raw_attempts_per_condition": 20,
            "candidate_pool_before_selection": 20,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "retry_or_resampling": False,
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
        payload.get("protocol") != SOURCE_PROTOCOL
        or payload.get("role") != "sealed_post_freeze_targets"
    ):
        raise ValueError("Invalid sealed V6 evaluation targets")
    pairs = []
    for expected_index, raw in enumerate(payload["records"]):
        record = dict(raw)
        if int(record["pair_index"]) != expected_index:
            raise ValueError("V6 evaluation target order drift")
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
            raise ValueError(f"Cannot reconstruct sealed V6 pair {expected_index}")
        source, target, common = aligned
        pairs.append(
            base.EditPair(
                row=row,
                source_smiles=str(record["source_smiles"]),
                target_smiles=str(record["target_smiles"]),
                source=source,
                target=target,
                condition=np.zeros(
                    (1, int(preregistration["condition_dim"])), dtype=np.float32
                ),
                property_count=int(record["property_count"]),
                task=base.task_key(row),
                common_atoms=int(common),
            )
        )
    return pairs


def task_breakdown(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_condition: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    by_task: defaultdict[str, list[list[Mapping[str, object]]]] = defaultdict(list)
    for values in by_condition.values():
        by_task[str(values[0]["task"])].append(values)
    return {
        task: {
            "conditions": len(conditions),
            "property_any20": sum(
                any(bool(row["property_success"]) for row in values)
                for values in conditions
            )
            / len(conditions),
            "strict_any20": sum(
                any(bool(row["strict_success"]) for row in values)
                for values in conditions
            )
            / len(conditions),
        }
        for task, conditions in sorted(by_task.items())
    }


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V6 evaluation exists: {summary_path}")
    check_locked_inputs(
        preregistration,
        {"evaluation_targets_sha256": args.evaluation_targets},
    )
    pairs = load_evaluation_pairs(args.evaluation_targets, preregistration)
    expected = len(pairs) * int(preregistration["exact_raw_attempts_per_condition"])
    metrics_by_arm = {}
    for arm in ARMS:
        arm_dir = args.frozen_root / arm
        arm_summary = read_json(arm_dir / "summary.json")
        candidate_path = arm_dir / "frozen_candidates.csv"
        if file_sha256(candidate_path) != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]:
            raise ValueError(f"Frozen V6 candidate drift: {arm}")
        frozen = semantic.coerce_frozen_rows(candidate_path)
        if len(frozen) != expected:
            raise ValueError(f"{arm} expected {expected} frozen rows, found {len(frozen)}")
        evaluated, metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = dict(metrics)
        metrics["by_task"] = task_breakdown(evaluated)
        evaluated_path = args.output_dir / f"evaluated_{arm}.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        metrics_by_arm[arm] = {
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
    summary = {
        "protocol": PROTOCOL,
        "stage": "post_freeze_evaluation_execution",
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
        raise ValueError(f"Completed V6 science gate exists: {summary_path}")
    evaluation = read_json(args.evaluation_summary)
    if (
        evaluation.get("protocol") != PROTOCOL
        or evaluation.get("execution_status") != "completed"
    ):
        raise ValueError("V6 evaluation execution is incomplete")
    by_arm = dict(evaluation["arms"])
    metrics = {arm: dict(dict(by_arm[arm])["metrics"]) for arm in ARMS}
    gates = dict(preregistration["science_gates"])
    full = metrics["language_full"]
    numeric = metrics["numeric_canonical"]
    reversed_metrics = metrics["language_reversed"]
    effects = {
        "strict_delta_vs_numeric": float(full["strict_any20"])
        - float(numeric["strict_any20"]),
        "property_delta_vs_reversed": float(full["property_any20"])
        - float(reversed_metrics["property_any20"]),
        "strict_delta_vs_reversed": float(full["strict_any20"])
        - float(reversed_metrics["strict_any20"]),
    }
    expected_rows = int(preregistration["conditions"]) * 20
    checks = {
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
        "protocol": "mass_conserving_router_table1_bridge_v6_science_gate",
        "execution_status": "completed",
        "science_gate": {
            "passed": passed,
            "checks": checks,
            "thresholds": gates,
            "failures": failures,
        },
        "decision": (
            "advance_v5_router_to_denovo_table1_mumo_replays"
            if passed
            else "stop_before_three_task_replays_no_molecular_bridge_signal"
        ),
        "metrics": metrics,
        "effects": effects,
        "contract": {
            "scientific_stop_exits_zero": True,
            "portal_engineering_state_independent_of_science_decision": True,
            "repeat_on_same_conditions_for_retuning": False,
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
