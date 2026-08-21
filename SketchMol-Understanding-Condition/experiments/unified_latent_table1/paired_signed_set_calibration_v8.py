#!/usr/bin/env python3
"""Paired-seed development calibration of Common-LLM signed property sets.

V7 showed that matched language strongly beats direction-reversed language but
trails numeric canonical control on a sealed fresh confirmation.  This
train-only development study asks whether that residual gap is sampling noise
or confidence-amplitude mismatch.  All arms share identical per-condition
particle seeds.  The language arms differ only by a preregistered deterministic
projection of the frozen V5 router coefficients; no model is trained and no
target or property oracle is accepted by the freeze process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
for module_path in (SCRIPT_DIR, PROJECT_DIR, LATENT_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import horizon_closed_graph_jump as horizon  # noqa: E402
import mass_conserving_router_table1_bridge_v6 as v6  # noqa: E402


PROTOCOL = "train_only_paired_signed_set_calibration_v8"
GATE_PROTOCOL = "paired_signed_set_calibration_v8_science_gate"
ARMS = (
    "numeric_canonical",
    "language_raw",
    "language_signed_vertex",
    "language_sqrt_sharpened",
    "language_reversed_signed_vertex",
)
CALIBRATION_ARMS = ("language_signed_vertex", "language_sqrt_sharpened")
base = v6.base
semantic = v6.semantic
property_basis = v6.property_basis
hierarchical = v6.hierarchical
b40 = v6.b40
b41 = v6.b41


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", required=True, type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)

    validate = stages.add_parser("validate")
    add_shared_inputs(validate, include_targets=True)
    validate.add_argument("--output-root", required=True, type=Path)

    freeze = stages.add_parser("freeze")
    freeze.add_argument("--replicate-index", required=True, type=int)
    add_shared_inputs(freeze, include_targets=False)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--device", default="auto")

    evaluate = stages.add_parser("evaluate")
    evaluate.add_argument("--replicate-index", required=True, type=int)
    evaluate.add_argument("--evaluation-targets", required=True, type=Path)
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    gate = stages.add_parser("gate")
    gate.add_argument("--evaluation-root", required=True, type=Path)
    gate.add_argument("--output-dir", required=True, type=Path)
    return parser


def add_shared_inputs(parser: argparse.ArgumentParser, *, include_targets: bool) -> None:
    parser.add_argument("--v6-prepare-summary", required=True, type=Path)
    parser.add_argument("--v6-basis-bundle", required=True, type=Path)
    parser.add_argument("--generation-conditions", required=True, type=Path)
    if include_targets:
        parser.add_argument("--evaluation-targets", required=True, type=Path)
    parser.add_argument("--representation-checkpoint", required=True, type=Path)
    parser.add_argument("--representation-summary", required=True, type=Path)
    parser.add_argument("--canonical-checkpoint", required=True, type=Path)
    parser.add_argument("--sft-adapter-dir", required=True, type=Path)
    parser.add_argument("--v5-lora-adapter-dir", required=True, type=Path)
    parser.add_argument("--v5-router-checkpoint", required=True, type=Path)
    parser.add_argument("--v5-summary", required=True, type=Path)
    parser.add_argument("--v5-gate", required=True, type=Path)
    parser.add_argument("--v5-unlock", required=True, type=Path)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = read_json(path)
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "task_role": "train_only_development_calibration_not_fresh_confirmation",
        "arms": list(ARMS),
        "calibration_arms": list(CALIBRATION_ARMS),
        "training": False,
        "conditions": 48,
        "replicate_seeds": [2111, 2112, 2113],
        "paired_common_random_numbers": True,
        "exact_raw_attempts_per_condition": 20,
        "candidate_pool_before_selection": 20,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "v7_fresh_source_access": False,
        "official_test_access": False,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"V8 preregistration drift: {drift}")
    implementation = file_sha256(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation:
        raise ValueError(
            f"V8 implementation drift: expected {payload.get('implementation_sha256')}, "
            f"found {implementation}"
        )
    horizon_digest = file_sha256(SCRIPT_DIR / "horizon_closed_graph_jump.py")
    if payload.get("horizon_closure_sha256") != horizon_digest:
        raise ValueError("V8 horizon-closure implementation drift")
    v6_digest = file_sha256(Path(v6.__file__).resolve())
    if payload.get("v6_implementation_sha256") != v6_digest:
        raise ValueError("V8 V6 implementation drift")
    return payload


def input_paths(args: argparse.Namespace, *, include_targets: bool) -> dict[str, Path]:
    paths = {
        "v6_prepare_summary_sha256": args.v6_prepare_summary,
        "v6_basis_bundle_sha256": args.v6_basis_bundle,
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
    }
    if include_targets:
        paths["evaluation_targets_sha256"] = args.evaluation_targets
    return paths


def check_locked_inputs(
    preregistration: Mapping[str, object], paths: Mapping[str, Path]
) -> dict[str, str]:
    locked = dict(preregistration["locked_inputs"])
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V8 inputs: {missing}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    drift = {
        name: {"expected": locked.get(name), "actual": digest}
        for name, digest in actual.items()
        if locked.get(name) != digest
    }
    if drift:
        raise ValueError(f"V8 locked-input drift: {drift}")
    return actual


def transform_coefficients(
    name: str, coefficients: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    support_float = support.to(coefficients.dtype)
    signed = torch.where(
        coefficients >= 0,
        torch.ones_like(coefficients),
        -torch.ones_like(coefficients),
    ) * support_float
    if name in {"language_signed_vertex", "language_reversed_signed_vertex"}:
        return signed
    if name == "language_sqrt_sharpened":
        magnitude = coefficients.abs().clamp_min(1e-8).sqrt()
        return signed * magnitude
    if name == "language_raw":
        return coefficients * support_float
    raise ValueError(f"Unsupported coefficient transform: {name}")


def run_validate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    hashes = check_locked_inputs(
        preregistration, input_paths(args, include_targets=True)
    )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"V8 output root is not empty: {args.output_root}")
    prepare = read_json(args.v6_prepare_summary)
    if prepare.get("protocol") != v6.PROTOCOL:
        raise ValueError("V8 requires the locked V6 prepare summary")
    bundle = torch.load(args.v6_basis_bundle, map_location="cpu", weights_only=False)
    if (
        bundle.get("protocol") != v6.PROTOCOL
        or bundle.get("role") != "target_free_frozen_generation_basis_and_support"
    ):
        raise ValueError("V8 requires the locked target-free V6 basis bundle")
    conditions = read_json(args.generation_conditions)
    targets = read_json(args.evaluation_targets)
    if (
        conditions.get("protocol") != v6.SOURCE_PROTOCOL
        or conditions.get("role") != "constraint_text_and_sources_without_targets"
        or len(conditions.get("records", [])) != int(preregistration["conditions"])
    ):
        raise ValueError("V8 generation-condition contract drift")
    if (
        targets.get("protocol") != v6.SOURCE_PROTOCOL
        or targets.get("role") != "sealed_post_freeze_targets"
        or len(targets.get("records", [])) != int(preregistration["conditions"])
    ):
        raise ValueError("V8 evaluation-target contract drift")
    v6.validate_v5_gate(args.v5_gate, args.v5_unlock)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "stage": "preflight_validation",
                "execution_status": "completed",
                "conditions": len(conditions["records"]),
                "replicates": len(preregistration["replicate_seeds"]),
                "arms": list(ARMS),
                "candidate_rows_planned": (
                    int(preregistration["conditions"])
                    * int(preregistration["exact_raw_attempts_per_condition"])
                    * len(ARMS)
                    * len(preregistration["replicate_seeds"])
                ),
                "locked_inputs": hashes,
                "contract": {
                    "v7_fresh_source_access": False,
                    "generation_target_access": False,
                    "paired_common_random_numbers": True,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def replicate_seed(
    preregistration: Mapping[str, object], replicate_index: int
) -> int:
    seeds = [int(value) for value in preregistration["replicate_seeds"]]
    if replicate_index < 0 or replicate_index >= len(seeds):
        raise ValueError(f"Invalid V8 replicate index: {replicate_index}")
    return seeds[replicate_index]


def run_freeze(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    seed = replicate_seed(preregistration, int(args.replicate_index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_summary = args.output_dir / "summary.json"
    if root_summary.exists():
        raise ValueError(f"Completed V8 freeze exists: {root_summary}")
    hashes = check_locked_inputs(
        preregistration, input_paths(args, include_targets=False)
    )
    v6.validate_v5_gate(args.v5_gate, args.v5_unlock)
    prepare = read_json(args.v6_prepare_summary)
    if prepare.get("protocol") != v6.PROTOCOL:
        raise ValueError("V8 V6-prepare protocol drift")
    bundle = torch.load(args.v6_basis_bundle, map_location="cpu", weights_only=False)
    if (
        bundle.get("protocol") != v6.PROTOCOL
        or bundle.get("role") != "target_free_frozen_generation_basis_and_support"
    ):
        raise ValueError("V8 basis bundle drift")
    device = base.resolve_device(str(args.device))
    base.seed_everything(seed)
    pairs, records = v6.load_generation_pairs(args.generation_conditions, preregistration)
    llm, tokenizer, router, property_columns = v6.load_v5_language_router(
        args, preregistration, device
    )
    if property_columns != list(bundle["property_columns"]):
        raise ValueError("V8 property vocabulary drift")
    matched_examples = v6.examples_for_records(records, property_columns, "matched")
    reversed_examples = v6.examples_for_records(records, property_columns, "reversed")
    matched_coefficients, matched_support, matched_cardinality = v6.v5.predict_examples(
        llm, router, tokenizer, matched_examples, preregistration, device
    )
    reversed_coefficients, reversed_support, reversed_cardinality = v6.v5.predict_examples(
        llm, router, tokenizer, reversed_examples, preregistration, device
    )
    coefficient_rows = {
        "language_raw": transform_coefficients(
            "language_raw", matched_coefficients, matched_support
        ),
        "language_signed_vertex": transform_coefficients(
            "language_signed_vertex", matched_coefficients, matched_support
        ),
        "language_sqrt_sharpened": transform_coefficients(
            "language_sqrt_sharpened", matched_coefficients, matched_support
        ),
        "language_reversed_signed_vertex": transform_coefficients(
            "language_reversed_signed_vertex",
            reversed_coefficients,
            reversed_support,
        ),
    }
    routing = {
        "language_raw": v6.v5.routing_metrics(
            coefficient_rows["language_raw"],
            matched_support,
            matched_cardinality,
            matched_examples,
        ),
        "language_signed_vertex": v6.v5.routing_metrics(
            coefficient_rows["language_signed_vertex"],
            matched_support,
            matched_cardinality,
            matched_examples,
        ),
        "language_sqrt_sharpened": v6.v5.routing_metrics(
            coefficient_rows["language_sqrt_sharpened"],
            matched_support,
            matched_cardinality,
            matched_examples,
        ),
        "language_reversed_signed_vertex": v6.v5.routing_metrics(
            coefficient_rows["language_reversed_signed_vertex"],
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
        arm: property_basis.compose_tokens(values, bundle["basis"], token_shape).numpy()
        for arm, values in coefficient_rows.items()
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
            raise ValueError(f"Completed V8 arm exists: {arm_dir}")
        rows: list[dict[str, object]] = []
        started = time.perf_counter()
        for index, pair in enumerate(pairs):
            if arm == "numeric_canonical":
                pair.condition = hierarchical.property_latent_slot_tokens(
                    pair.row, int(preregistration["condition_dim"])
                )
            else:
                pair.condition = np.asarray(language_tokens[arm][index], dtype=np.float32)
            paired_particle_seed = seed * 100000 + index
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
                paired_particle_seed,
            )
            if len(generated) != 20:
                raise RuntimeError(f"V8 {arm} did not emit exactly 20 attempts")
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
                        "paired_particle_seed": paired_particle_seed,
                        **candidate,
                    }
                )
            if (index + 1) % 12 == 0 or index + 1 == len(pairs):
                print(
                    json.dumps(
                        {
                            "stage": "v8_freeze_progress",
                            "replicate_index": int(args.replicate_index),
                            "arm": arm,
                            "conditions": index + 1,
                            "total": len(pairs),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        expected = int(preregistration["conditions"]) * 20
        if len(rows) != expected:
            raise RuntimeError(f"V8 {arm} expected {expected} rows, found {len(rows)}")
        if any(not str(row["generated_smiles"]) for row in rows):
            raise RuntimeError(f"V8 {arm} emitted a non-materializable candidate")
        base.write_candidate_rows(candidate_path, rows)
        arm_summary = {
            "protocol": PROTOCOL,
            "stage": "paired_target_isolated_horizon_closed_freeze",
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
                "exact_raw_attempts_per_condition": 20,
                "candidate_pool_before_selection": 20,
                "molecular_candidate_ranking": False,
                "oracle_selection": False,
                "retry_or_resampling": False,
                "posthoc_molecule_repair": False,
                "paired_common_random_numbers": True,
                "v7_fresh_source_access": False,
            },
        }
        summary_path.write_text(
            json.dumps(arm_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    root_summary.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "stage": "paired_all_arms_frozen",
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
                    "exact_raw_attempts_per_condition": 20,
                    "candidate_pool_before_selection": 20,
                    "paired_common_random_numbers": True,
                    "molecular_candidate_ranking": False,
                    "oracle_selection": False,
                    "retry_or_resampling": False,
                    "v7_fresh_source_access": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def run_evaluate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    seed = replicate_seed(preregistration, int(args.replicate_index))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V8 evaluation exists: {summary_path}")
    expected_target_hash = dict(preregistration["locked_inputs"])[
        "evaluation_targets_sha256"
    ]
    actual_target_hash = file_sha256(args.evaluation_targets)
    if actual_target_hash != expected_target_hash:
        raise ValueError("V8 evaluation-target drift")
    pairs = v6.load_evaluation_pairs(args.evaluation_targets, preregistration)
    expected = int(preregistration["conditions"]) * 20
    metrics_by_arm: dict[str, object] = {}
    for arm in ARMS:
        arm_dir = args.frozen_root / arm
        arm_summary = read_json(arm_dir / "summary.json")
        candidate_path = arm_dir / "frozen_candidates.csv"
        if (
            arm_summary.get("protocol") != PROTOCOL
            or int(arm_summary.get("replicate_seed", -1)) != seed
            or file_sha256(candidate_path)
            != dict(arm_summary["artifacts"])["frozen_candidates_sha256"]
        ):
            raise ValueError(f"V8 frozen artifact drift: {arm}")
        frozen = semantic.coerce_frozen_rows(candidate_path)
        if len(frozen) != expected:
            raise ValueError(f"V8 {arm} expected {expected} frozen rows")
        evaluated, metrics = b41.evaluate_frozen_candidates(frozen, pairs)
        metrics = dict(metrics)
        metrics["by_task"] = v6.task_breakdown(evaluated)
        evaluated_path = args.output_dir / f"evaluated_{arm}.csv"
        base.write_candidate_rows(evaluated_path, evaluated)
        metrics_by_arm[arm] = {
            "metrics": metrics,
            "evaluated_candidates_sha256": file_sha256(evaluated_path),
        }
    summary = {
        "protocol": PROTOCOL,
        "stage": "paired_post_freeze_evaluation",
        "execution_status": "completed",
        "decision": "await_separate_science_gate",
        "replicate_index": int(args.replicate_index),
        "replicate_seed": seed,
        "arms": metrics_by_arm,
        "contract": {
            "generation_target_access": False,
            "post_freeze_target_access": True,
            "exact_raw_attempts_per_condition": 20,
            "paired_common_random_numbers": True,
            "molecular_candidate_ranking": False,
            "oracle_selection": False,
            "science_decision_in_separate_process": True,
            "v7_fresh_source_access": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def mean_metrics(
    evaluations: Sequence[Mapping[str, object]], arm: str
) -> dict[str, float]:
    rows = [dict(dict(dict(item["arms"])[arm])["metrics"]) for item in evaluations]
    names = (
        "candidate_rows",
        "attempted_per_condition",
        "validity",
        "mean_source_tanimoto",
        "mean_unique_valid",
        "property_any20",
        "strict_any20",
    )
    return {
        name: float(sum(float(row[name]) for row in rows) / len(rows))
        for name in names
    }


def run_gate(args: argparse.Namespace, preregistration: Mapping[str, object]) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "gate_summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed V8 science gate exists: {summary_path}")
    evaluations = []
    for replicate_index, seed in enumerate(preregistration["replicate_seeds"]):
        path = args.evaluation_root / f"replicate_{replicate_index}" / "summary.json"
        payload = read_json(path)
        if (
            payload.get("protocol") != PROTOCOL
            or payload.get("execution_status") != "completed"
            or int(payload.get("replicate_seed", -1)) != int(seed)
        ):
            raise ValueError(f"Incomplete V8 evaluation: {path}")
        evaluations.append(payload)
    metrics = {arm: mean_metrics(evaluations, arm) for arm in ARMS}
    replicate_metrics = {
        str(index): {
            arm: dict(dict(dict(payload["arms"])[arm])["metrics"])
            for arm in ARMS
        }
        for index, payload in enumerate(evaluations)
    }
    winner = sorted(
        CALIBRATION_ARMS,
        key=lambda arm: (
            -float(metrics[arm]["strict_any20"]),
            -float(metrics[arm]["property_any20"]),
            arm,
        ),
    )[0]
    numeric = metrics["numeric_canonical"]
    raw = metrics["language_raw"]
    reversed_metrics = metrics["language_reversed_signed_vertex"]
    chosen = metrics[winner]
    replicate_wins = sum(
        float(replicate_metrics[str(index)][winner]["strict_any20"])
        >= float(replicate_metrics[str(index)]["language_raw"]["strict_any20"])
        for index in range(len(evaluations))
    )
    effects = {
        "strict_delta_vs_numeric": float(chosen["strict_any20"])
        - float(numeric["strict_any20"]),
        "strict_delta_vs_raw": float(chosen["strict_any20"])
        - float(raw["strict_any20"]),
        "property_delta_vs_raw": float(chosen["property_any20"])
        - float(raw["property_any20"]),
        "strict_delta_vs_reversed": float(chosen["strict_any20"])
        - float(reversed_metrics["strict_any20"]),
        "replicate_win_rate_vs_raw": replicate_wins / len(evaluations),
    }
    gates = dict(preregistration["science_gates"])
    expected_rows = int(preregistration["conditions"]) * 20
    checks = {
        "replicate_count": len(evaluations) == len(preregistration["replicate_seeds"]),
        "candidate_rows": all(
            math.isclose(float(metrics[arm]["candidate_rows"]), expected_rows)
            for arm in ARMS
        ),
        "exact_attempts": all(
            math.isclose(float(metrics[arm]["attempted_per_condition"]), 20.0)
            for arm in ARMS
        ),
        "all_arm_validity": all(
            float(metrics[arm]["validity"]) >= float(gates["validity"])
            for arm in ARMS
        ),
        "chosen_source_tanimoto": float(chosen["mean_source_tanimoto"])
        >= float(gates["mean_source_tanimoto"]),
        "chosen_unique": float(chosen["mean_unique_valid"])
        >= float(gates["mean_unique_valid"]),
        "chosen_property": float(chosen["property_any20"])
        >= float(gates["property_any20"]),
        "chosen_strict": float(chosen["strict_any20"])
        >= float(gates["strict_any20"]),
        "chosen_vs_numeric": effects["strict_delta_vs_numeric"]
        >= float(gates["strict_delta_vs_numeric"]),
        "chosen_vs_raw_strict": effects["strict_delta_vs_raw"]
        >= float(gates["strict_delta_vs_raw"]),
        "chosen_vs_raw_property": effects["property_delta_vs_raw"]
        >= float(gates["property_delta_vs_raw"]),
        "chosen_vs_reversed": effects["strict_delta_vs_reversed"]
        >= float(gates["strict_delta_vs_reversed"]),
        "replicate_win_rate": effects["replicate_win_rate_vs_raw"]
        >= float(gates["replicate_win_rate_vs_raw"]),
    }
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
            "lock_calibration_for_one_new_fresh_confirmation"
            if passed
            else "stop_calibration_before_new_fresh_confirmation"
        ),
        "selected_calibration": winner,
        "metrics": metrics,
        "replicate_metrics": replicate_metrics,
        "effects": effects,
        "contract": {
            "scientific_stop_exits_zero": True,
            "portal_engineering_state_independent_of_science_decision": True,
            "v7_fresh_source_access": False,
            "official_test_access": False,
            "automatic_fresh_submission": False,
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
    if args.stage == "validate":
        return run_validate(args, preregistration)
    if args.stage == "freeze":
        return run_freeze(args, preregistration)
    if args.stage == "evaluate":
        return run_evaluate(args, preregistration)
    if args.stage == "gate":
        return run_gate(args, preregistration)
    raise ValueError(f"Unsupported V8 stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
