#!/usr/bin/env python3
"""Target-free MolEdit Table1 transfer of the frozen B24/B27/B28 latent editor.

The transfer changes no model weight or sampling hyperparameter.  It projects a
fixed Table1 task subset into the same property-slot latent used by B24, then
compares the frozen nearest-token decoder with the frozen B28 energy-tilted
categorical decoder.  Each latent attempt emits one fragment token and one raw
molecule; there is no molecular candidate ranking, retry, repair, or second
edit.  Table1 success is source-relative, so target molecules are never used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
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

import energy_tilted_vq_fragment_sampling as b28  # noqa: E402


b27 = b28.b27
kernel = b28.kernel
base = b28.base
belief = b28.belief
graph = b28.graph
unified = b28.unified

PROTOCOL = "target_free_moledit_table1_latent_transfer_v29"
TARGET_FIELDS = {
    "target_smiles",
    "target_canonical_smiles",
    "target_scaffold_smiles",
    "source_target_tanimoto",
}


@dataclass
class TransferCondition:
    row: dict[str, str]
    source_smiles: str
    source: object
    condition: np.ndarray
    property_count: int
    task: str
    condition_id: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--table1-eval-csv", type=Path, required=True)
    parser.add_argument("--representation-checkpoint", type=Path, required=True)
    parser.add_argument("--representation-summary", type=Path, required=True)
    parser.add_argument("--fragment-checkpoint", type=Path, required=True)
    parser.add_argument("--energy-checkpoint", type=Path, required=True)
    parser.add_argument("--energy-summary", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_first_run",
        "frozen_fragment_protocol": kernel.PROTOCOL,
        "frozen_energy_protocol": b27.PROTOCOL,
        "frozen_quantizer_protocol": b28.PROTOCOL,
        "b26_heldout_access": False,
        "official_test_access": False,
        "moledit_target_access": False,
        "model_training": False,
        "num_attempts": 20,
        "per_task": 4,
        "distance_temperature": 0.03,
        "energy_weight": 1.25,
        "molecular_candidate_ranking": False,
        "second_edit": False,
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise ValueError(f"B29 preregistration drift: {drift}")
    tasks = payload.get("table1_tasks")
    if not isinstance(tasks, list) or len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("B29 requires five unique preregistered Table1 tasks")
    return payload


def table1_task_key(specs: Sequence[tuple[str, int]]) -> str:
    direction_name = {1: "increase", -1: "decrease"}
    if not specs or any(direction not in direction_name for _prop, direction in specs):
        return "unknown"
    return "+".join(
        f"{prop}:{direction_name[direction]}"
        for prop, direction in sorted(specs)
    )


def source_only_row(raw: Mapping[str, str], specs: Sequence[tuple[str, int]]) -> dict[str, str]:
    """Project a MolEdit row to the only fields permitted before generation."""
    condition_id = str(raw.get("example_id", "") or raw.get("condition_id", "")).strip()
    source = str(raw.get("source_smiles", "") or "").strip()
    instruction = str(raw.get("instruction", "") or "").strip()
    direction_name = {1: "increase", -1: "decrease"}
    tasks = [
        {"property": prop, "direction": direction_name[direction]}
        for prop, direction in specs
    ]
    row = {
        "condition_id": condition_id,
        "sample_id": condition_id,
        "task_type": "edit_generation",
        "source_smiles": source,
        "instruction": instruction,
        "condition_properties": ",".join(prop for prop, _direction in specs),
        "instruction_tasks": json.dumps(
            tasks, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ),
    }
    for prop, direction in specs:
        row[f"{prop}_active"] = "True"
        row[f"{prop}_direction"] = direction_name[direction]
    if any(key in row for key in TARGET_FIELDS) or any(
        key.startswith("target_") or key.startswith("delta_") for key in row
    ):
        raise AssertionError("B29 source-only projection retained a target field")
    return row


def select_conditions(
    path: Path,
    *,
    tasks: Sequence[str],
    per_task: int,
    seed: int,
    forbidden_sources: set[str],
    condition_dim: int,
    graph_fingerprint_bits: int,
) -> tuple[list[TransferCondition], dict[str, object]]:
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {
        task: [] for task in tasks
    }
    counts: defaultdict[str, int] = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"example_id", "instruction", "source_smiles", "instruction_tasks"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Table1 eval split missing fields: {sorted(missing)}")
        for raw in reader:
            specs = base.task_specs(raw)
            task = table1_task_key(specs)
            if task not in grouped:
                continue
            counts[f"available_{task}"] += 1
            row = source_only_row(raw, specs)
            source = graph.canonical_smiles(row["source_smiles"])
            condition_id = row["condition_id"]
            if not source or not condition_id:
                counts["invalid_source_or_id"] += 1
                continue
            if source in forbidden_sources:
                counts["b24_train_source_overlap_excluded"] += 1
                continue
            key = b27.stable_value(seed, task, condition_id, source)
            grouped[task].append((key, row))

    selected: list[TransferCondition] = []
    selected_sources: set[str] = set()
    per_task_selected: dict[str, int] = {}
    for task in tasks:
        for _key, row in sorted(grouped[task], key=lambda item: item[0]):
            source = graph.canonical_smiles(row["source_smiles"])
            if source in selected_sources:
                counts["duplicate_source_excluded"] += 1
                continue
            source_graph = graph.molecule_example(
                source, max_atoms=64, fingerprint_bits=int(graph_fingerprint_bits)
            )
            if source_graph is None:
                counts["source_graph_rejected"] += 1
                continue
            specs = base.task_specs(row)
            selected.append(
                TransferCondition(
                    row=row,
                    source_smiles=source,
                    source=source_graph,
                    condition=kernel.hierarchical.property_latent_slot_tokens(
                        row, int(condition_dim)
                    ),
                    property_count=len(specs),
                    task=task,
                    condition_id=row["condition_id"],
                )
            )
            selected_sources.add(source)
            if sum(item.task == task for item in selected) >= int(per_task):
                break
        per_task_selected[task] = sum(item.task == task for item in selected)
    short = {task: count for task, count in per_task_selected.items() if count != int(per_task)}
    if short:
        raise ValueError(f"B29 fixed Table1 subset is incomplete: {short}")
    selected.sort(key=lambda item: (tasks.index(item.task), item.condition_id))
    return selected, {
        "rows_scanned": sum(count for key, count in counts.items() if key.startswith("available_")),
        "filter_counts": dict(counts),
        "selected_conditions": len(selected),
        "per_task_selected": per_task_selected,
        "selected_source_count": len(selected_sources),
        "b24_train_source_overlap_after_filter": len(selected_sources & forbidden_sources),
        "target_columns_used": 0,
    }


def freeze_candidates(
    fragment_model: kernel.FragmentAttachmentKernel,
    energy_model: b27.LatentPropertyEnergy,
    conditions: Sequence[TransferCondition],
    source_latents: np.ndarray,
    target_fragments: Sequence[str],
    target_endpoints: np.ndarray,
    config: SimpleNamespace,
    device: torch.device,
) -> dict[str, list[tuple[TransferCondition, list[dict[str, object]]]]]:
    frozen: dict[str, list[tuple[TransferCondition, list[dict[str, object]]]]] = {
        "nearest_token": [],
        "energy_tilted": [],
    }
    for index, condition in enumerate(conditions):
        seed = int(config.seed) * 100000 + index
        nearest = kernel.generate_actions(
            fragment_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        tilted = b28.tilted_actions(
            fragment_model,
            energy_model,
            condition,
            source_latents[index],
            target_fragments,
            target_endpoints,
            config,
            device,
            seed=seed,
        )
        if len(nearest) != int(config.num_attempts) or len(tilted) != int(config.num_attempts):
            raise RuntimeError("B29 did not freeze exactly 20 attempts per method")
        frozen["nearest_token"].append((condition, nearest))
        frozen["energy_tilted"].append((condition, tilted))
    return frozen


def evaluate_frozen(
    values: Sequence[tuple[TransferCondition, list[dict[str, object]]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition, generated in values:
        specs = base.task_specs(condition.row)
        for attempt, raw in enumerate(generated, start=1):
            canonical = graph.canonical_smiles(str(raw.get("smiles", "") or ""))
            valid = bool(canonical)
            similarity = (
                graph.morgan_tanimoto(condition.source_smiles, canonical)
                if valid
                else None
            )
            fraction, _distance, evaluated, property_success = (
                unified.instruction_success_and_distance(
                    condition.row, canonical or "", task_specs=specs
                )
            )
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "task": condition.task,
                    "property_count": condition.property_count,
                    "attempt": attempt,
                    "source_smiles": condition.source_smiles,
                    "generated_smiles": canonical or "",
                    "valid": valid,
                    "source_tanimoto": float(similarity or 0.0),
                    "property_fraction": float(fraction),
                    "evaluated_properties": int(evaluated),
                    "property_success": bool(property_success),
                    "success_t0_15": bool(
                        property_success and similarity is not None and similarity >= 0.15
                    ),
                    "success_t0_65": bool(
                        property_success and similarity is not None and similarity >= 0.65
                    ),
                    "site_core": raw.get("site_core", ""),
                    "source_fragment": raw.get("source_fragment", ""),
                    "target_fragment_token": raw.get("target_fragment_token", ""),
                    "token_energy": raw.get("token_energy", ""),
                    "token_probability": raw.get("token_probability", ""),
                }
            )
    return rows, summarize(rows)


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_id"])].append(row)
    condition_rows = []
    for condition_id, values in grouped.items():
        if len(values) != 20:
            raise ValueError(f"{condition_id}: expected 20 attempts, found {len(values)}")
        condition_rows.append(
            {
                "condition_id": condition_id,
                "task": values[0]["task"],
                "unique_valid": len(
                    {str(row["generated_smiles"]) for row in values if bool(row["valid"])}
                ),
                "property_any20": any(bool(row["property_success"]) for row in values),
                "acc_any20_t0_15": any(bool(row["success_t0_15"]) for row in values),
                "acc_any20_t0_65": any(bool(row["success_t0_65"]) for row in values),
            }
        )

    def metrics(candidate: Sequence[Mapping[str, object]], conditions) -> dict[str, object]:
        expected_properties = sum(int(row["property_count"]) for row in candidate)
        evaluated_properties = sum(int(row["evaluated_properties"]) for row in candidate)
        valid = [row for row in candidate if bool(row["valid"])]
        return {
            "conditions": len(conditions),
            "candidate_rows": len(candidate),
            "attempted_per_condition": 20,
            "validity": sum(bool(row["valid"]) for row in candidate) / max(1, len(candidate)),
            "property_success_per_attempt": sum(
                bool(row["property_success"]) for row in candidate
            ) / max(1, len(candidate)),
            "acc_all_t0_15_per_attempt": sum(
                bool(row["success_t0_15"]) for row in candidate
            ) / max(1, len(candidate)),
            "acc_all_t0_65_per_attempt": sum(
                bool(row["success_t0_65"]) for row in candidate
            ) / max(1, len(candidate)),
            "property_any20": sum(bool(row["property_any20"]) for row in conditions)
            / max(1, len(conditions)),
            "acc_any20_t0_15": sum(bool(row["acc_any20_t0_15"]) for row in conditions)
            / max(1, len(conditions)),
            "acc_any20_t0_65": sum(bool(row["acc_any20_t0_65"]) for row in conditions)
            / max(1, len(conditions)),
            "mean_unique_valid": float(
                np.mean([float(row["unique_valid"]) for row in conditions])
            ) if conditions else 0.0,
            "mean_source_tanimoto": float(
                np.mean([float(row["source_tanimoto"]) for row in valid])
            ) if valid else 0.0,
            "oracle_coverage": evaluated_properties / max(1, expected_properties),
        }

    summary = metrics(rows, condition_rows)
    by_task: dict[str, object] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_candidates = [row for row in rows if str(row["task"]) == task]
        task_conditions = [row for row in condition_rows if str(row["task"]) == task]
        by_task[task] = metrics(task_candidates, task_conditions)
    summary["by_task"] = by_task
    return summary


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Completed B29 result exists: {summary_path}")
    preregistration = read_preregistration(args.protocol_manifest)
    base.seed_everything(int(preregistration["seed"]))
    device = base.resolve_device(str(args.device))
    representation, _representation_config, representation_summary = (
        base.load_representation(
            args.representation_checkpoint, args.representation_summary, device
        )
    )
    fragment_model, target_fragments, target_endpoints, frozen_manifest = (
        b27.load_frozen_fragment_model(
            args.fragment_checkpoint, device, preregistration
        )
    )
    energy_model, energy_summary = b28.load_energy(
        args.energy_checkpoint, args.energy_summary, preregistration, device
    )
    for path, key in (
        (args.representation_checkpoint, "representation_checkpoint_sha256"),
        (args.train_csv, "train_csv_sha256"),
        (args.validation_csv, "validation_csv_sha256"),
    ):
        if frozen_manifest.get(key) != belief.file_sha256(path):
            raise ValueError(f"Frozen B24 input drift: {key}")

    train_pairs, reconstruction = b27.reconstruct_b24_train_pairs(
        args, preregistration
    )
    forbidden_sources = {pair.source_smiles for pair in train_pairs}
    conditions, selection = select_conditions(
        args.table1_eval_csv,
        tasks=list(preregistration["table1_tasks"]),
        per_task=int(preregistration["per_task"]),
        seed=int(preregistration["selection_seed"]),
        forbidden_sources=forbidden_sources,
        condition_dim=int(preregistration["condition_dim"]),
        graph_fingerprint_bits=int(preregistration["graph_fingerprint_bits"]),
    )
    source_latents = kernel.encode_sources(
        representation,
        conditions,
        device,
        batch_size=int(preregistration["encoding_batch_size"]),
    )
    config = SimpleNamespace(
        min_core_heavy_atoms=int(preregistration["min_core_heavy_atoms"]),
        max_variable_heavy_atoms=int(preregistration["max_variable_heavy_atoms"]),
        fingerprint_bits=int(preregistration["fingerprint_bits"]),
        num_attempts=int(preregistration["num_attempts"]),
        flow_steps=int(preregistration["flow_steps"]),
        site_temperature=float(preregistration["site_temperature"]),
        distance_temperature=float(preregistration["distance_temperature"]),
        energy_weight=float(preregistration["energy_weight"]),
        energy_scale_floor=float(preregistration["energy_scale_floor"]),
        energy_chunk_size=int(preregistration["energy_chunk_size"]),
        seed=int(preregistration["seed"]),
    )

    # The property oracles occur only after both methods' exact 20 raw
    # molecules have been frozen.  No target molecule is used even here.
    frozen = freeze_candidates(
        fragment_model,
        energy_model,
        conditions,
        source_latents,
        target_fragments,
        target_endpoints,
        config,
        device,
    )
    nearest_rows, nearest = evaluate_frozen(frozen["nearest_token"])
    tilted_rows, tilted = evaluate_frozen(frozen["energy_tilted"])
    deltas = {
        key: float(tilted[key]) - float(nearest[key])
        for key in (
            "property_success_per_attempt",
            "acc_all_t0_15_per_attempt",
            "acc_all_t0_65_per_attempt",
            "property_any20",
            "acc_any20_t0_15",
            "acc_any20_t0_65",
        )
    }
    task_nonzero = sum(
        float(item["acc_any20_t0_15"]) > 0.0
        for item in tilted["by_task"].values()
    )
    gates = dict(preregistration["gates"])
    checks = {
        "selected_conditions": {"value": selection["selected_conditions"], "threshold": 20},
        "selected_tasks": {"value": len(selection["per_task_selected"]), "threshold": 5},
        "b24_train_source_overlap": {
            "value": selection["b24_train_source_overlap_after_filter"],
            "threshold": 0,
        },
        "target_columns_used": {"value": selection["target_columns_used"], "threshold": 0},
        "attempted_per_condition": {"value": tilted["attempted_per_condition"], "threshold": 20},
        "validity": {"value": tilted["validity"], "threshold": gates["validity"]},
        "oracle_coverage": {
            "value": tilted["oracle_coverage"],
            "threshold": gates["oracle_coverage"],
        },
        "mean_unique_valid": {
            "value": tilted["mean_unique_valid"],
            "threshold": gates["mean_unique_valid"],
        },
        "acc_all_t0_15_delta": {
            "value": deltas["acc_all_t0_15_per_attempt"],
            "threshold": gates["acc_all_t0_15_delta"],
        },
        "acc_any20_t0_65_delta": {
            "value": deltas["acc_any20_t0_65"],
            "threshold": gates["acc_any20_t0_65_delta"],
        },
        "tasks_with_any20_t0_15": {
            "value": task_nonzero,
            "threshold": gates["tasks_with_any20_t0_15"],
        },
    }
    exact = {
        "selected_conditions",
        "selected_tasks",
        "b24_train_source_overlap",
        "target_columns_used",
        "attempted_per_condition",
    }
    failures = [
        name
        for name, item in checks.items()
        if (
            item["value"] != item["threshold"]
            if name in exact
            else float(item["value"]) < float(item["threshold"])
        )
    ]
    run_manifest = {
        "protocol": PROTOCOL,
        "preregistration_sha256": belief.file_sha256(args.protocol_manifest),
        "table1_eval_csv_sha256": belief.file_sha256(args.table1_eval_csv),
        "representation_checkpoint_sha256": belief.file_sha256(args.representation_checkpoint),
        "fragment_checkpoint_sha256": belief.file_sha256(args.fragment_checkpoint),
        "energy_checkpoint_sha256": belief.file_sha256(args.energy_checkpoint),
        "energy_summary_sha256": belief.file_sha256(args.energy_summary),
        "representation_protocol": representation_summary.get("protocol"),
        "model_training": False,
        "models_frozen": True,
        "generation_target_access": False,
        "moledit_target_access": False,
        "property_oracle_generation_access": False,
        "post_freeze_property_oracle_access": True,
        "molecular_candidate_ranking": False,
        "one_latent_one_sampled_token_one_raw_molecule": True,
        "failed_attachment_retry": False,
        "second_edit": False,
        "exact_raw_attempts_per_condition": 20,
        "selection": selection,
        "reconstruction": reconstruction,
        "energy_calibration": energy_summary.get("calibration"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "nearest_token_candidates.csv", nearest_rows)
    write_rows(args.output_dir / "energy_tilted_candidates.csv", tilted_rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "protocol": PROTOCOL,
        "manifest": run_manifest,
        "nearest_token_evaluation": nearest,
        "energy_tilted_evaluation": tilted,
        "energy_tilted_minus_nearest": deltas,
        "gate": {"passed": not failures, "checks": checks, "failures": failures},
        "decision": (
            "advance_frozen_latent_editor_to_full_table1_transfer"
            if not failures
            else "stop_and_classify_table1_transfer_failure_before_any_retraining"
        ),
        "metric_note": (
            "acc_all_t*_per_attempt is the Table1-style per-generated-molecule rate; "
            "acc_any20_t* is a separate exact-n=20 support diagnostic and is not paper Acc_all."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
