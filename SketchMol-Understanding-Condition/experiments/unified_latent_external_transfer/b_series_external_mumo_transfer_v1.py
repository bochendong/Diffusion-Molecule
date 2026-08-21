#!/usr/bin/env python3
"""Frozen B-series valid-terminal transfer to a balanced MuMO OOD dev gate.

The B backbone only accepts its eight native numeric properties.  This runner
therefore fits one small train-only linear adapter from an explicit signed MuMO
property set and the source's native descriptors to a native B-property delta.
The B checkpoint and graph-event dynamics remain frozen.  Candidate generation
is isolated from MuMO targets and property oracles and emits exactly twenty raw
attempts per condition without ranking, retries, or resampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
UCA_DIR = PROJECT_DIR / "experiments" / "unified_constraint_agent"
LATENT_DIR = PROJECT_DIR / "experiments" / "unified_latent_flow"
TABLE1_DIR = PROJECT_DIR / "experiments" / "unified_latent_table1"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, UCA_DIR, LATENT_DIR, TABLE1_DIR, SCRIPTS_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import export_external_multiproperty_benchmark_rows as external  # noqa: E402
import mumo_parallel_protocol as mumo  # noqa: E402
from sketchmol_understanding_condition.chem import (  # noqa: E402
    canonical_smiles,
    molecular_properties,
)
from sketchmol_understanding_condition.direct_condition_tokens import (  # noqa: E402
    PROPERTY_NORMALIZERS,
)


PROTOCOL = "b_series_external_mumo_transfer_v1"
OOD_TASKS = ("MPQ", "BDMQ", "BHMQ", "BMPQ", "HMPQ")
EXTERNAL_PROPERTIES = ("bbbp", "drd2", "hia", "mutagenicity", "plogp", "qed")
EXTERNAL_SIGNS = {
    "bbbp": 1.0,
    "drd2": 1.0,
    "hia": 1.0,
    "mutagenicity": -1.0,
    "plogp": 1.0,
    "qed": 1.0,
}
NATIVE_PROPERTIES = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA")
NATIVE_KEYS = {
    "MW": "MolWt",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "rotatable",
    "SA": "SA",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_shard_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def stable_fraction(value: str, seed: int) -> float:
    raw = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64)


def read_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL,
        "status": "preregistered_before_external_transfer_generation",
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "train_only_numeric_adapter": True,
        "external_task_split": "ood",
        "conditions_per_ood_task": 10,
        "condition_count": 50,
        "exact_raw_attempts_per_condition": 20,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "posthoc_molecule_repair": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "official_test_access": False,
        "language_conditioning": False,
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"preregistration drift: {drift}")
    if tuple(payload.get("ood_task_ids", ())) != OOD_TASKS:
        raise ValueError("OOD task contract drift")
    implementation = sha256_file(Path(__file__).resolve())
    if payload.get("implementation_sha256") != implementation:
        raise ValueError(
            f"implementation drift: expected {payload.get('implementation_sha256')}, found {implementation}"
        )
    return payload


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"non-object row in {path}")
                rows.append(dict(value))
    return rows


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def native_properties(smiles: str) -> dict[str, float] | None:
    values = molecular_properties(smiles)
    if not values:
        return None
    output: dict[str, float] = {}
    for prop in NATIVE_PROPERTIES:
        value = values.get(NATIVE_KEYS[prop])
        if value is None or not math.isfinite(float(value)):
            return None
        output[prop] = float(value)
    return output


def task_spec(task_id: str):
    return next(
        spec
        for spec in external.TASK_SPECS
        if spec.suite == "mumo" and spec.task_id == str(task_id)
    )


def signed_property_vectors(task_id: str) -> tuple[np.ndarray, np.ndarray]:
    active_set = set(task_spec(task_id).properties)
    active = np.asarray(
        [1.0 if prop in active_set else 0.0 for prop in EXTERNAL_PROPERTIES],
        dtype=np.float64,
    )
    signed = np.asarray(
        [EXTERNAL_SIGNS[prop] if prop in active_set else 0.0 for prop in EXTERNAL_PROPERTIES],
        dtype=np.float64,
    )
    return signed, active


def adapter_features(task_id: str, source_native: Mapping[str, float]) -> np.ndarray:
    """Canonical set features; independent of property mention order."""

    signed, active = signed_property_vectors(task_id)
    native = np.asarray(
        [float(source_native[prop]) / float(PROPERTY_NORMALIZERS[prop]) for prop in NATIVE_PROPERTIES],
        dtype=np.float64,
    )
    pairwise = np.asarray(
        [signed[i] * signed[j] for i in range(len(signed)) for j in range(i + 1, len(signed))],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            np.ones(1, dtype=np.float64),
            signed,
            active,
            pairwise,
            native,
            np.outer(signed, native).reshape(-1),
            np.outer(active, native).reshape(-1),
        ]
    )


def normalized_native_delta(
    source_native: Mapping[str, float], target_native: Mapping[str, float]
) -> np.ndarray:
    return np.asarray(
        [
            (float(target_native[prop]) - float(source_native[prop]))
            / float(PROPERTY_NORMALIZERS[prop])
            for prop in NATIVE_PROPERTIES
        ],
        dtype=np.float64,
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return math.nan
    left = rankdata(a)
    right = rankdata(b)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def fit_adapter(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    task_ids: Sequence[str],
    *,
    ridge_alpha: float,
) -> dict[str, np.ndarray]:
    mean = x_fit[:, 1:].mean(axis=0)
    scale = x_fit[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_scaled = x_fit.copy()
    x_scaled[:, 1:] = (x_scaled[:, 1:] - mean) / scale
    counts = Counter(task_ids)
    weights = np.asarray([1.0 / counts[task] for task in task_ids], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    root = np.sqrt(weights)[:, None]
    gram = (x_scaled * root).T @ (x_scaled * root)
    penalty = np.eye(x_scaled.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(gram + penalty, (x_scaled * root).T @ (y_fit * root))
    clip_low = np.quantile(y_fit, 0.02, axis=0)
    clip_high = np.quantile(y_fit, 0.98, axis=0)
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "coef": coef,
        "clip_low": clip_low,
        "clip_high": clip_high,
    }


def adapter_predict(model: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    scaled = np.asarray(features, dtype=np.float64).copy()
    scaled[1:] = (scaled[1:] - model["feature_mean"]) / model["feature_scale"]
    prediction = scaled @ model["coef"]
    return np.clip(prediction, model["clip_low"], model["clip_high"])


def proxy_condition_row(
    source_smiles: str,
    source_native: Mapping[str, float],
    predicted_normalized_delta: np.ndarray,
    *,
    active_count: int,
    delta_scale: float,
) -> tuple[dict[str, str], list[str]]:
    magnitude = np.abs(np.asarray(predicted_normalized_delta, dtype=np.float64))
    chosen_indices = sorted(
        np.argsort(-magnitude, kind="stable")[: int(active_count)].tolist()
    )
    chosen = [NATIVE_PROPERTIES[index] for index in chosen_indices]
    row: dict[str, str] = {
        "source_smiles": str(source_smiles),
        "condition_properties": ",".join(chosen),
        "property_count": str(len(chosen)),
    }
    for index, prop in enumerate(NATIVE_PROPERTIES):
        delta = float(predicted_normalized_delta[index]) * float(PROPERTY_NORMALIZERS[prop]) * float(delta_scale)
        target = float(source_native[prop]) + delta
        row[f"source_{prop}"] = f"{float(source_native[prop]):.10g}"
        row[f"target_{prop}"] = f"{target:.10g}"
        row[f"{prop}_active"] = "1" if prop in chosen else "0"
        row[f"{prop}_direction"] = "increase" if delta >= 0.0 else "decrease"
    return row, chosen


def generation_condition(raw: Mapping[str, object]) -> dict[str, object]:
    task_id = str(raw["_uca_task_id"])
    spec = task_spec(task_id)
    pair_digest = str(raw["_uca_pair_digest"])
    condition_id = f"mumo_dev_{task_id.lower()}_{pair_digest}"
    return {
        "condition_id": condition_id,
        "sample_id": condition_id,
        "source_smiles": str(raw["source_smiles"]),
        "external_suite": "mumo",
        "external_task_id": task_id,
        "external_task_key": spec.task_key,
        "external_task_split": "ood",
        "external_task_properties": ",".join(spec.properties),
        "external_property_directions_json": json.dumps(spec.directions, sort_keys=True),
        "external_property_objectives_json": json.dumps(spec.objectives, sort_keys=True),
        "external_property_thresholds_json": json.dumps(dict(spec.thresholds), sort_keys=True),
        "source_group": str(raw["_uca_source_group"]),
        "pair_digest": pair_digest,
        "data_partition": "external_transfer_dev",
    }


def prepare(args: argparse.Namespace, prereg: Mapping[str, object]) -> int:
    manifest_path = args.data_dir / "manifest.json"
    shard_paths = sorted(args.data_dir.glob("train_shard_*.jsonl"))
    locked = dict(prereg["locked_inputs"])
    if sha256_file(manifest_path) != locked["mumo_v8_manifest_sha256"]:
        raise ValueError("MuMO v8 manifest drift")
    if len(shard_paths) != 32:
        raise ValueError(f"expected 32 MuMO shards, found {len(shard_paths)}")
    if aggregate_shard_digest(shard_paths) != locked["mumo_v8_shards_aggregate_sha256"]:
        raise ValueError("MuMO v8 shard aggregate drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_target_access") is not False:
        raise ValueError("target-exposed MuMO evidence is forbidden")
    if manifest.get("official_test_content_access") is not False:
        raise ValueError("official test content access is forbidden")

    rows = [row for path in shard_paths for row in read_jsonl(path)]
    per_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw in rows:
        task_id = str(raw.get("_uca_task_id", ""))
        if raw.get("_uca_partition") == "dev" and task_id in OOD_TASKS:
            per_task[task_id].append(raw)

    selected: list[dict[str, object]] = []
    selected_sources: set[str] = set()
    quota = int(prereg["conditions_per_ood_task"])
    seed = int(prereg["dev_selection_seed"])
    for task_id in OOD_TASKS:
        candidates = sorted(
            per_task[task_id],
            key=lambda raw: stable_fraction(
                f"{task_id}:{canonical_smiles(str(raw.get('source_smiles', ''))) or ''}", seed
            ),
        )
        for raw in candidates:
            source = canonical_smiles(str(raw.get("source_smiles", "")))
            if not source or source in selected_sources:
                continue
            props = native_properties(source)
            if props is None:
                continue
            selected.append(raw)
            selected_sources.add(source)
            if sum(str(item["_uca_task_id"]) == task_id for item in selected) >= quota:
                break
        found = sum(str(item["_uca_task_id"]) == task_id for item in selected)
        if found != quota:
            raise ValueError(f"fresh balanced quota unavailable for {task_id}: {found}/{quota}")
    if len(selected) != int(prereg["condition_count"]):
        raise ValueError("balanced dev condition count drift")

    generation_rows = [generation_condition(raw) for raw in selected]
    leaked = sorted(
        key
        for row in generation_rows
        for key in row
        if key.lower() == "target_smiles"
        or key.lower().startswith("external_target_")
        or key.lower().startswith("external_source_")
        or "oracle" in key.lower()
    )
    if leaked:
        raise ValueError(f"generation condition leakage: {leaked}")

    fit_x: list[np.ndarray] = []
    fit_y: list[np.ndarray] = []
    fit_tasks: list[str] = []
    calibration_x: list[np.ndarray] = []
    calibration_y: list[np.ndarray] = []
    calibration_tasks: list[str] = []
    source_sets = {"fit": set(), "calibration": set()}
    skip_counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str, str]] = set()
    for raw in rows:
        if raw.get("_uca_partition") != "fit":
            continue
        task_id = str(raw.get("_uca_task_id", ""))
        if task_id not in mumo.TASK_IDS:
            skip_counts["unknown_task"] += 1
            continue
        source = canonical_smiles(str(raw.get("source_smiles", "")))
        target = canonical_smiles(str(raw.get("target_smiles", "")))
        if not source or not target:
            skip_counts["invalid_pair"] += 1
            continue
        if source in selected_sources:
            skip_counts["dev_source_excluded"] += 1
            continue
        pair_key = (task_id, source, target)
        if pair_key in seen_pairs:
            skip_counts["duplicate_pair"] += 1
            continue
        seen_pairs.add(pair_key)
        source_native = native_properties(source)
        target_native = native_properties(target)
        if source_native is None or target_native is None:
            skip_counts["missing_native_properties"] += 1
            continue
        x = adapter_features(task_id, source_native)
        y = normalized_native_delta(source_native, target_native)
        partition = (
            "calibration"
            if stable_fraction(source, int(prereg["adapter_split_seed"]))
            < float(prereg["adapter_calibration_fraction"])
            else "fit"
        )
        if partition == "fit":
            fit_x.append(x)
            fit_y.append(y)
            fit_tasks.append(task_id)
        else:
            calibration_x.append(x)
            calibration_y.append(y)
            calibration_tasks.append(task_id)
        source_sets[partition].add(source)
    if source_sets["fit"] & source_sets["calibration"]:
        raise ValueError("adapter fit/calibration source overlap")
    if (source_sets["fit"] | source_sets["calibration"]) & selected_sources:
        raise ValueError("adapter/dev source overlap")
    if len(fit_x) < 1000 or len(calibration_x) < 100:
        raise ValueError("insufficient adapter fit/calibration rows")

    x_fit = np.stack(fit_x)
    y_fit = np.stack(fit_y)
    x_cal = np.stack(calibration_x)
    y_cal = np.stack(calibration_y)
    model = fit_adapter(
        x_fit,
        y_fit,
        fit_tasks,
        ridge_alpha=float(prereg["adapter_ridge_alpha"]),
    )
    cal_prediction = np.stack([adapter_predict(model, row) for row in x_cal])
    calibration_metrics = {}
    for index, prop in enumerate(NATIVE_PROPERTIES):
        calibration_metrics[prop] = {
            "mae_normalized": float(np.abs(cal_prediction[:, index] - y_cal[:, index]).mean()),
            "spearman": spearman(cal_prediction[:, index], y_cal[:, index]),
            "direction_accuracy": float(
                (np.sign(cal_prediction[:, index]) == np.sign(y_cal[:, index])).mean()
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_dir / "numeric_adapter.npz"
    np.savez_compressed(
        adapter_path,
        **model,
        external_properties=np.asarray(EXTERNAL_PROPERTIES),
        native_properties=np.asarray(NATIVE_PROPERTIES),
    )
    dev_path = args.output_dir / "generation_conditions.jsonl"
    write_jsonl(dev_path, generation_rows)
    summary = {
        "protocol": PROTOCOL,
        "stage": "prepare",
        "condition_count": len(generation_rows),
        "conditions_by_task": dict(Counter(row["external_task_id"] for row in generation_rows)),
        "unique_dev_sources": len(selected_sources),
        "fit_rows": len(fit_x),
        "calibration_rows": len(calibration_x),
        "fit_sources": len(source_sets["fit"]),
        "calibration_sources": len(source_sets["calibration"]),
        "fit_calibration_source_overlap": len(source_sets["fit"] & source_sets["calibration"]),
        "fit_dev_source_overlap": len((source_sets["fit"] | source_sets["calibration"]) & selected_sources),
        "skip_counts": dict(skip_counts),
        "calibration_metrics": calibration_metrics,
        "adapter_sha256": sha256_file(adapter_path),
        "generation_conditions_sha256": sha256_file(dev_path),
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "official_test_access": False,
    }
    write_json(args.output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def load_adapter(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: np.asarray(payload[key])
            for key in ("feature_mean", "feature_scale", "coef", "clip_low", "clip_high")
        }


def freeze(args: argparse.Namespace, prereg: Mapping[str, object]) -> int:
    torch = importlib.import_module("torch")
    d0 = importlib.import_module("eval_d0_b41_table1")
    b41 = d0.b41
    base = d0.base
    graph = d0.graph
    hierarchical = d0.hierarchical
    b40 = d0.b40
    b39 = d0.b39
    delta = d0.delta
    valid_terminal = d0.valid_terminal

    prepare_summary = json.loads(args.prepare_summary.read_text(encoding="utf-8"))
    if prepare_summary.get("protocol") != PROTOCOL:
        raise ValueError("prepare protocol drift")
    if sha256_file(args.adapter_npz) != prepare_summary.get("adapter_sha256"):
        raise ValueError("adapter artifact drift")
    if sha256_file(args.dev_conditions) != prepare_summary.get("generation_conditions_sha256"):
        raise ValueError("generation condition artifact drift")
    locked = dict(prereg["locked_inputs"])
    direct_locks = {
        "b41_checkpoint_sha256": args.b41_checkpoint,
        "b41_summary_sha256": args.b41_summary,
        "valid_terminal_summary_sha256": args.valid_terminal_summary,
    }
    for name, path in direct_locks.items():
        actual = sha256_file(path)
        if actual != locked[name]:
            raise ValueError(f"locked B artifact drift for {name}: {actual}")
    valid_summary = json.loads(args.valid_terminal_summary.read_text(encoding="utf-8"))
    if valid_summary.get("protocol") != "train_only_valid_terminal_molecule_latent_jump_v1":
        raise ValueError("valid-terminal protocol drift")
    if valid_summary.get("manifest", {}).get("frozen_b41_checkpoint") is not True:
        raise ValueError("valid-terminal B41 checkpoint was not frozen")

    b41_prereg = b41.read_preregistration(args.b41_protocol_manifest)
    device = base.resolve_device(str(args.device))
    b41_namespace = SimpleNamespace(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        representation_checkpoint=args.representation_checkpoint,
        representation_summary=args.representation_summary,
        b22_checkpoint=args.b22_checkpoint,
        b22_summary=args.b22_summary,
        b36_summary=args.b36_summary,
        b37_summary=args.b37_summary,
        b38_checkpoint=args.b38_checkpoint,
        b38_summary=args.b38_summary,
        b39_checkpoint=args.b39_checkpoint,
        b39_summary=args.b39_summary,
        b39_evaluated_candidates=args.b39_evaluated_candidates,
        b40_summary=args.b40_summary,
        b40_evaluated_candidates=args.b40_evaluated_candidates,
    )
    (_b22_summary, b22_checkpoint, _b36, _b37, _b39, _b40) = b41.check_locked_inputs(
        b41_namespace, b41_prereg
    )
    b41_checkpoint = torch.load(args.b41_checkpoint, map_location="cpu", weights_only=False)
    selected_pairs = d0.reconstruct_support_pairs(b41_namespace, b41_prereg, b22_checkpoint)
    fit_pairs, _dev_pairs, _split = d0.b37.strict_source_group_split(
        selected_pairs,
        seed=int(b41_prereg["development_split_seed"]),
        development_source_limit=int(b41_prereg["development_source_limit"]),
    )
    representation, representation_config, _rep_summary = base.load_representation(
        args.representation_checkpoint, args.representation_summary, device
    )
    vocabulary = d0.b37.checkpoint_vocabulary(b22_checkpoint)
    support = b40.build_support(fit_pairs, vocabulary)
    support_tensors = b40._device_support(support, device)
    node_action_count, edge_action_count = delta.action_space_sizes(vocabulary)
    model = b39.LatentCardinalityGraphJumpBridge(
        node_dim=int(representation_config["node_dim"]),
        edge_dim=int(representation_config["edge_dim"]),
        condition_dim=int(b41_prereg["condition_dim"]),
        transport_dim=int(b41_prereg["transport_dim"]),
        hidden_dim=int(b41_prereg["hidden_dim"]),
        max_atoms=int(representation_config["max_atoms"]),
        max_jumps=int(b41_prereg["max_jumps"]),
        property_count=len(d0.unified.PROPERTY_COLUMNS),
        node_state_count=node_action_count,
        edge_state_count=edge_action_count,
        message_layers=int(b41_prereg["message_layers"]),
    ).to(device)
    model.load_state_dict(dict(b41_checkpoint["model_state"]), strict=True)
    model.eval().requires_grad_(False)

    adapter = load_adapter(args.adapter_npz)
    conditions = read_jsonl(args.dev_conditions)
    if len(conditions) != int(prereg["condition_count"]):
        raise ValueError("generation condition count drift")
    exact_support = valid_terminal.ExactMoleculeStopSupport(vocabulary)
    original_support = b41.viability_event_mask
    b41.viability_event_mask = exact_support
    rows: list[dict[str, object]] = []
    proxy_audit: list[dict[str, object]] = []
    try:
        for index, condition in enumerate(conditions):
            source = canonical_smiles(str(condition["source_smiles"]))
            if not source:
                raise ValueError(f"invalid source at condition {index}")
            source_native = native_properties(source)
            if source_native is None:
                raise ValueError(f"missing native source properties at condition {index}")
            features = adapter_features(str(condition["external_task_id"]), source_native)
            predicted = adapter_predict(adapter, features)
            proxy_row, chosen = proxy_condition_row(
                source,
                source_native,
                predicted,
                active_count=int(prereg["adapter_native_property_count"]),
                delta_scale=float(prereg["adapter_delta_scale"]),
            )
            source_graph = graph.molecule_example(
                source,
                max_atoms=int(representation_config["max_atoms"]),
                fingerprint_bits=int(b41_prereg["fingerprint_bits"]),
            )
            if source_graph is None:
                raise ValueError(f"source not representable by B graph at condition {index}")
            tokens = hierarchical.property_latent_slot_tokens(
                proxy_row, int(b41_prereg["condition_dim"])
            )
            generated = b41.sample_from_source(
                model,
                representation,
                vocabulary,
                support,
                support_tensors,
                source_graph,
                tokens,
                b41_prereg,
                device,
                int(prereg["generation_seed"]) * 100000 + index,
            )
            attempts = int(prereg["exact_raw_attempts_per_condition"])
            if len(generated) != attempts:
                raise ValueError(
                    f"condition {condition['condition_id']} emitted {len(generated)} attempts, expected {attempts}"
                )
            proxy_audit.append(
                {
                    "condition_id": condition["condition_id"],
                    "external_task_id": condition["external_task_id"],
                    "selected_native_properties": chosen,
                    "predicted_normalized_native_delta": {
                        prop: float(predicted[prop_index])
                        for prop_index, prop in enumerate(NATIVE_PROPERTIES)
                    },
                    "proxy_row": proxy_row,
                }
            )
            for attempt, candidate in enumerate(generated, start=1):
                rows.append(
                    {
                        **dict(condition),
                        "generated_smiles": str(candidate.get("generated_smiles", "")),
                        "sample_index": attempt,
                        "candidate_index": attempt,
                        "candidate_rank": attempt,
                        "candidate_selected": True,
                        "method": PROTOCOL,
                        "family": "frozen_b41_valid_terminal",
                        "numeric_adapter": "signed_property_set_ridge_v1",
                    }
                )
            if (index + 1) % 5 == 0 or index + 1 == len(conditions):
                print(
                    json.dumps(
                        {"stage": "frozen_generation", "done": index + 1, "total": len(conditions)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        b41.viability_event_mask = original_support

    expected_rows = int(prereg["condition_count"]) * int(prereg["exact_raw_attempts_per_condition"])
    if len(rows) != expected_rows:
        raise ValueError(f"candidate row contract drift: {len(rows)} != {expected_rows}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "frozen_candidates.csv"
    proxy_path = args.output_dir / "numeric_proxy_audit.jsonl"
    write_csv(candidates_path, rows)
    write_jsonl(proxy_path, proxy_audit)
    summary = {
        "protocol": PROTOCOL,
        "stage": "freeze",
        "device": str(device),
        "condition_count": len(conditions),
        "candidate_rows": len(rows),
        "attempts_per_condition": int(prereg["exact_raw_attempts_per_condition"]),
        "candidates_sha256": sha256_file(candidates_path),
        "numeric_proxy_audit_sha256": sha256_file(proxy_path),
        "frozen_b41_checkpoint": True,
        "b41_training": False,
        "molecular_candidate_ranking": False,
        "oracle_selection": False,
        "retry_or_resampling": False,
        "generation_target_access": False,
        "generation_property_oracle_access": False,
        "official_test_access": False,
        "exact_molecule_stop_support": exact_support.manifest(),
    }
    write_json(args.output_dir / "freeze_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def aggregate_evaluation(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    by_condition: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition_id"])].append(row)
    condition_records = []
    candidate_validity = []
    similarities = []
    unique_counts = []
    for condition_id, candidates in sorted(by_condition.items()):
        if len(candidates) != 20:
            raise ValueError(f"{condition_id} has {len(candidates)} evaluated attempts")
        valid = [row for row in candidates if truthy(row.get("external_valid"))]
        candidate_validity.extend(truthy(row.get("external_valid")) for row in candidates)
        similarities.extend(
            value
            for value in (safe_float(row.get("external_source_tanimoto")) for row in valid)
            if value is not None
        )
        unique = {
            str(row.get("external_generated_canonical_smiles", ""))
            for row in valid
            if str(row.get("external_generated_canonical_smiles", ""))
        }
        unique_counts.append(len(unique))
        requested = set(
            part.strip()
            for part in str(candidates[0].get("external_task_properties", "")).split(",")
            if part.strip()
        )
        supported: set[str] = set()
        for row in candidates:
            payload = json.loads(str(row.get("external_property_success_json", "{}") or "{}"))
            supported.update(prop for prop, passed in payload.items() if passed is True)
        condition_records.append(
            {
                "condition_id": condition_id,
                "task": str(candidates[0].get("external_task_id", "")),
                "property_success": any(truthy(row.get("external_official_success")) for row in candidates),
                "strict_success": any(truthy(row.get("external_strict_success")) for row in candidates),
                "support_ceiling": bool(requested) and requested <= supported,
            }
        )
    by_task = {}
    for task in OOD_TASKS:
        task_rows = [row for row in condition_records if row["task"] == task]
        by_task[task] = {
            "conditions": len(task_rows),
            "property_any20": float(np.mean([row["property_success"] for row in task_rows])),
            "strict_any20": float(np.mean([row["strict_success"] for row in task_rows])),
            "support_ceiling": float(np.mean([row["support_ceiling"] for row in task_rows])),
        }
    return {
        "conditions": len(condition_records),
        "candidate_rows": len(rows),
        "validity": float(np.mean(candidate_validity)),
        "property_any20": float(np.mean([row["property_success"] for row in condition_records])),
        "strict_any20": float(np.mean([row["strict_success"] for row in condition_records])),
        "support_ceiling": float(np.mean([row["support_ceiling"] for row in condition_records])),
        "mean_source_tanimoto": float(np.mean(similarities)) if similarities else 0.0,
        "mean_unique_valid": float(np.mean(unique_counts)),
        "full_oracle_candidate_rate": float(
            np.mean([truthy(row.get("external_full_property_coverage")) for row in rows])
        ),
        "by_task": by_task,
        "condition_records": condition_records,
    }


def gate(args: argparse.Namespace, prereg: Mapping[str, object]) -> int:
    locked = dict(prereg["locked_inputs"])
    if sha256_file(args.baseline_detail) != locked["baseline_detail_sha256"]:
        raise ValueError("baseline detail drift")
    prepare_summary = json.loads(args.prepare_summary.read_text(encoding="utf-8"))
    freeze_summary = json.loads(args.freeze_summary.read_text(encoding="utf-8"))
    if freeze_summary.get("protocol") != PROTOCOL:
        raise ValueError("freeze protocol drift")
    if sha256_file(args.candidates) != freeze_summary.get("candidates_sha256"):
        raise ValueError("frozen candidate artifact drift")
    detail_rows = read_csv(args.evaluation_detail)
    candidate_ids = {str(row["condition_id"]) for row in detail_rows}
    baseline_rows = [
        row for row in read_csv(args.baseline_detail) if str(row.get("condition_id", "")) in candidate_ids
    ]
    baseline_ids = {str(row["condition_id"]) for row in baseline_rows}
    if candidate_ids != baseline_ids:
        raise ValueError(
            f"paired baseline condition mismatch: missing={sorted(candidate_ids - baseline_ids)}"
        )
    metrics = aggregate_evaluation(detail_rows)
    baseline = aggregate_evaluation(baseline_rows)
    metrics.pop("condition_records", None)
    baseline.pop("condition_records", None)
    property_gain = float(metrics["property_any20"]) - float(baseline["property_any20"])
    strict_gain = float(metrics["strict_any20"]) - float(baseline["strict_any20"])
    gates = dict(prereg["gates"])
    checks = {
        "condition_count": int(metrics["conditions"]) == int(prereg["condition_count"]),
        "candidate_rows": int(metrics["candidate_rows"])
        == int(prereg["condition_count"]) * int(prereg["exact_raw_attempts_per_condition"]),
        "paired_baseline_conditions": int(baseline["conditions"]) == int(prereg["condition_count"]),
        "fit_dev_source_overlap": int(prepare_summary["fit_dev_source_overlap"]) == 0,
        "validity": float(metrics["validity"]) >= float(gates["validity"]),
        "property_any20": float(metrics["property_any20"]) >= float(gates["property_any20"]),
        "property_gain_vs_paired_baseline": property_gain
        >= float(gates["property_gain_vs_paired_baseline"]),
        "strict_any20": float(metrics["strict_any20"]) >= float(gates["strict_any20"]),
        "mean_source_tanimoto": float(metrics["mean_source_tanimoto"])
        >= float(gates["mean_source_tanimoto"]),
        "mean_unique_valid": float(metrics["mean_unique_valid"])
        >= float(gates["mean_unique_valid"]),
        "full_oracle_candidate_rate": float(metrics["full_oracle_candidate_rate"]) == 1.0,
        "frozen_b41_checkpoint": freeze_summary.get("frozen_b41_checkpoint") is True,
        "exact_n20": int(freeze_summary.get("attempts_per_condition", 0)) == 20,
        "no_ranking": freeze_summary.get("molecular_candidate_ranking") is False,
        "no_retry": freeze_summary.get("retry_or_resampling") is False,
        "no_oracle_selection": freeze_summary.get("oracle_selection") is False,
        "no_generation_target_access": freeze_summary.get("generation_target_access") is False,
        "no_generation_oracle_access": freeze_summary.get("generation_property_oracle_access") is False,
        "no_official_test_access": freeze_summary.get("official_test_access") is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    passed = not failures
    summary = {
        "protocol": PROTOCOL,
        "decision": (
            "advance_frozen_b_series_to_table1_and_denovo_external_pilots"
            if passed
            else "stop_b_series_model_iteration_and_report_external_transfer_diagnostic"
        ),
        "scientific_gate": {"passed": passed, "checks": checks, "failures": failures},
        "metrics": metrics,
        "paired_graph_edit_baseline": baseline,
        "deltas": {
            "property_any20": property_gain,
            "strict_any20": strict_gain,
            "validity": float(metrics["validity"]) - float(baseline["validity"]),
            "mean_source_tanimoto": float(metrics["mean_source_tanimoto"])
            - float(baseline["mean_source_tanimoto"]),
            "mean_unique_valid": float(metrics["mean_unique_valid"])
            - float(baseline["mean_unique_valid"]),
        },
        "reference_full_ood_baseline": prereg["reference_full_ood_baseline"],
        "manifest": {
            "frozen_b41_checkpoint": True,
            "train_only_numeric_adapter": True,
            "exact_raw_attempts_per_condition": 20,
            "molecular_candidate_ranking": False,
            "retry_or_resampling": False,
            "oracle_selection": False,
            "generation_target_access": False,
            "generation_property_oracle_access": False,
            "official_test_access": False,
            "language_conditioning": False,
            "adapter_lineage": "MuMO v8 train fit rows only; selected dev canonical sources globally excluded",
        },
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    # A scientific STOP remains a valid completed artifact.  Contract failures
    # above raise and fail the execution job; metric failure exits zero.
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "gate"))
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-summary", type=Path)
    parser.add_argument("--adapter-npz", type=Path)
    parser.add_argument("--dev-conditions", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--representation-checkpoint", type=Path)
    parser.add_argument("--representation-summary", type=Path)
    parser.add_argument("--b22-checkpoint", type=Path)
    parser.add_argument("--b22-summary", type=Path)
    parser.add_argument("--b36-summary", type=Path)
    parser.add_argument("--b37-summary", type=Path)
    parser.add_argument("--b38-checkpoint", type=Path)
    parser.add_argument("--b38-summary", type=Path)
    parser.add_argument("--b39-checkpoint", type=Path)
    parser.add_argument("--b39-summary", type=Path)
    parser.add_argument("--b39-evaluated-candidates", type=Path)
    parser.add_argument("--b40-summary", type=Path)
    parser.add_argument("--b40-evaluated-candidates", type=Path)
    parser.add_argument("--b41-checkpoint", type=Path)
    parser.add_argument("--b41-summary", type=Path)
    parser.add_argument("--b41-protocol-manifest", type=Path)
    parser.add_argument("--valid-terminal-summary", type=Path)
    parser.add_argument("--freeze-summary", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--evaluation-detail", type=Path)
    parser.add_argument("--baseline-detail", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def require_paths(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(args, name, None) is None]
    if missing:
        raise ValueError(f"missing required arguments for {args.stage}: {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg = read_preregistration(args.preregistration)
    if args.stage == "prepare":
        require_paths(args, ("data_dir", "output_dir"))
        return prepare(args, prereg)
    if args.stage == "freeze":
        require_paths(
            args,
            (
                "output_dir",
                "prepare_summary",
                "adapter_npz",
                "dev_conditions",
                "train_csv",
                "validation_csv",
                "representation_checkpoint",
                "representation_summary",
                "b22_checkpoint",
                "b22_summary",
                "b36_summary",
                "b37_summary",
                "b38_checkpoint",
                "b38_summary",
                "b39_checkpoint",
                "b39_summary",
                "b39_evaluated_candidates",
                "b40_summary",
                "b40_evaluated_candidates",
                "b41_checkpoint",
                "b41_summary",
                "b41_protocol_manifest",
                "valid_terminal_summary",
            ),
        )
        return freeze(args, prereg)
    require_paths(
        args,
        (
            "prepare_summary",
            "freeze_summary",
            "candidates",
            "evaluation_detail",
            "baseline_detail",
            "output_json",
        ),
    )
    return gate(args, prereg)


if __name__ == "__main__":
    raise SystemExit(main())
