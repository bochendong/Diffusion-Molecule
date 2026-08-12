#!/usr/bin/env python3
"""Freeze exactly n=20 verifier-ranked candidates for disjoint MuMO dev rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_mumo_feature_shard as feature_builder  # noqa: E402
import build_retrieved_delta_edit_candidates as delta  # noqa: E402
import export_external_multiproperty_benchmark_rows as export  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402
import train_mumo_property_verifier as verifier  # noqa: E402
from sketchmol_understanding_condition.chem import canonical_smiles  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dev-sources-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--max-transforms-per-fragment", type=int, default=96)
    parser.add_argument("--min-retrieval-similarity", type=float, default=0.15)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--min-core-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-variable-heavy-atoms", type=int, default=30)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1711)
    return parser.parse_args(argv)


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_transforms(path: Path) -> dict[str, list[dict[str, object]]]:
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in protocol.read_jsonl(path):
        by_task[str(row["task_key"])].append(row)
    for task in by_task:
        by_task[task].sort(key=lambda row: (-int(row["frequency"]), str(row["source_variable"])))
    return dict(by_task)


def load_models(run_root: Path) -> dict[str, dict[str, object]]:
    output = {}
    available_jobs = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    for prop in protocol.PROPERTIES:
        path = run_root / "verifiers" / prop / "model.joblib"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = joblib.load(path)
        if payload.get("protocol") != protocol.PROTOCOL_VERSION:
            raise ValueError(f"Verifier protocol mismatch for {prop}")
        payload["pair_classifier"].n_jobs = available_jobs
        output[prop] = payload
    return output


def load_fit_analog_library(run_root: Path) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """Load deduplicated fit-target fingerprints/descriptors, grouped by task."""

    rows: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for path in sorted((run_root / "features").glob("features_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            roles = np.asarray(payload["role"], dtype=str)
            partitions = np.asarray(payload["partition"], dtype=str)
            task_ids = np.asarray(payload["task_id"], dtype=str)
            smiles = np.asarray(payload["smiles"], dtype=str)
            fingerprints = np.asarray(payload["fingerprint"], dtype=np.uint8)
            descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
            keep = np.flatnonzero((roles == "target") & (partitions == "fit"))
            for index in keep:
                task = str(task_ids[index])
                canonical = str(smiles[index])
                rows[task].setdefault(canonical, (fingerprints[index], descriptors[index]))
    output = {}
    for task, values in rows.items():
        ordered = sorted(values.items())
        output[task] = (
            [smiles for smiles, _features in ordered],
            np.stack([features[0] for _smiles, features in ordered]),
            np.stack([features[1] for _smiles, features in ordered]),
        )
    return output


def retrieve_fit_analogs(
    source_feature: np.ndarray,
    library: tuple[list[str], np.ndarray, np.ndarray] | None,
    *,
    min_tanimoto: float,
    limit: int,
) -> list[tuple[str, float, np.ndarray]]:
    if library is None:
        return []
    smiles, fingerprints, descriptors = library
    source_bits = np.asarray(source_feature[: fingerprints.shape[1]] > 0, dtype=np.uint8)
    intersections = np.bitwise_and(fingerprints, source_bits).sum(axis=1, dtype=np.int32)
    unions = np.bitwise_or(fingerprints, source_bits).sum(axis=1, dtype=np.int32)
    similarities = intersections / np.maximum(unions, 1)
    eligible = np.flatnonzero(similarities >= float(min_tanimoto))
    ranked = eligible[np.argsort(-similarities[eligible], kind="stable")][: int(limit)]
    return [
        (
            smiles[index],
            float(similarities[index]),
            np.concatenate(
                [fingerprints[index].astype(np.float32), descriptors[index].astype(np.float32)]
            ),
        )
        for index in ranked
    ]


def freeze_attempts(
    ranked_candidates: Sequence[Mapping[str, object]],
    *,
    budget: int,
) -> list[tuple[dict[str, object], bool, int]]:
    """Freeze exactly ``budget`` attempts without inventing extra molecules."""

    if not ranked_candidates:
        raise ValueError("Cannot freeze attempts from an empty candidate pool")
    frozen = []
    for attempt_index in range(int(budget)):
        unique_index = attempt_index % len(ranked_candidates)
        frozen.append(
            (
                dict(ranked_candidates[unique_index]),
                attempt_index >= len(ranked_candidates),
                unique_index + 1,
            )
        )
    return frozen


def source_noop_candidate(
    source_smiles: str,
    source_feature: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    """Represent an explicit valid no-op when the constrained support is empty."""

    return (
        {
            "generated_smiles": source_smiles,
            "method": "source_noop_empty_support_fallback",
            "source_tanimoto": 1.0,
            "retrieval_similarity": 1.0,
            "transform_frequency": 0,
            "source_variable": "",
            "target_variable": "",
            "candidate_is_noop": True,
        },
        source_feature.copy(),
    )


def condition_row(raw: Mapping[str, object], _index: int) -> dict[str, object]:
    task_id = str(raw["_uca_task_id"])
    spec = next(spec for spec in export.TASK_SPECS if spec.suite == "mumo" and spec.task_id == task_id)
    pair_digest = str(raw["_uca_pair_digest"])
    condition_id = f"mumo_dev_{task_id.lower()}_{pair_digest}"
    # Deliberately omit target_smiles and every external_target_* field. Source
    # labels are not used for candidate generation, only emitted for the
    # post-freeze evaluator's source baseline.
    row: dict[str, object] = {
        "condition_id": condition_id,
        "sample_id": condition_id,
        "source_smiles": str(raw["source_smiles"]),
        "external_suite": "mumo",
        "external_task_id": task_id,
        "external_task_key": spec.task_key,
        "external_task_split": "ind" if task_id in protocol.IND_TASK_IDS else "ood",
        "external_task_properties": ",".join(spec.properties),
        "external_property_directions_json": json.dumps(spec.directions, sort_keys=True),
        "external_property_objectives_json": json.dumps(spec.objectives, sort_keys=True),
        "external_property_thresholds_json": json.dumps(dict(spec.thresholds), sort_keys=True),
        "source_group": str(raw["_uca_source_group"]),
        "data_partition": "dev",
    }
    for prop in spec.properties:
        source_value = export.read_property_value(raw, prop, prefix="source")
        if source_value is not None:
            row[f"external_source_{prop}"] = float(source_value)
    return row


def candidate_feature(smiles: str) -> np.ndarray | None:
    value = feature_builder.molecule_features(smiles, radius=2, n_bits=2048)
    if value is None:
        return None
    return np.concatenate([value[0].astype(np.float32), value[1].astype(np.float32)])


def exact_qed_margin(source_feature: np.ndarray, candidate_features: np.ndarray) -> np.ndarray:
    """Return exact RDKit QED improvement beyond the official MuMO threshold."""

    descriptor_start = int(source_feature.shape[0]) - len(feature_builder.DESCRIPTOR_NAMES)
    qed_index = descriptor_start + feature_builder.DESCRIPTOR_NAMES.index("QED")
    return np.asarray(
        candidate_features[:, qed_index]
        - float(source_feature[qed_index])
        - float(export.MUMO_THRESHOLDS["qed"]),
        dtype=np.float32,
    )


def score_candidate(
    source_feature: np.ndarray,
    candidate_feature_value: np.ndarray,
    *,
    properties: Sequence[str],
    models: Mapping[str, Mapping[str, object]],
    source_tanimoto: float,
    retrieval_similarity: float,
    frequency: int,
) -> tuple[float, dict[str, float]]:
    pair_features = np.concatenate(
        [source_feature, candidate_feature_value, candidate_feature_value - source_feature]
    ).reshape(1, -1)
    margins = {}
    for prop in properties:
        if prop == "qed":
            margins[prop] = float(
                exact_qed_margin(source_feature, candidate_feature_value.reshape(1, -1))[0]
            )
            continue
        payload = models[prop]
        classifier = payload["pair_classifier"]
        positive_column = list(classifier.classes_).index(True)
        probability = float(classifier.predict_proba(pair_features)[0, positive_column])
        threshold = float(payload.get("pair_decision_threshold", 0.5))
        margins[prop] = probability - threshold
    minimum = min(margins.values()) if margins else -1.0
    mean = sum(margins.values()) / max(len(margins), 1)
    passed = sum(value >= 0.0 for value in margins.values())
    score = (
        4.0 * passed
        + 2.0 * minimum
        + mean
        + 0.35 * source_tanimoto
        + 0.20 * retrieval_similarity
        + 0.02 * math.log1p(max(frequency, 0))
    )
    return score, margins


def score_candidates_batch(
    source_feature: np.ndarray,
    candidate_features: Sequence[np.ndarray],
    *,
    properties: Sequence[str],
    models: Mapping[str, Mapping[str, object]],
    source_tanimotos: Sequence[float],
    retrieval_similarities: Sequence[float],
    frequencies: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    candidates = np.stack(candidate_features).astype(np.float32, copy=False)
    sources = np.repeat(source_feature.reshape(1, -1), len(candidates), axis=0)
    pair_features = np.concatenate([sources, candidates, candidates - sources], axis=1)
    margin_columns: dict[str, np.ndarray] = {}
    for prop in properties:
        if prop == "qed":
            margin_columns[prop] = exact_qed_margin(source_feature, candidates)
            continue
        payload = models[prop]
        classifier = payload["pair_classifier"]
        positive_column = list(classifier.classes_).index(True)
        probability = classifier.predict_proba(pair_features)[:, positive_column]
        margin_columns[prop] = np.asarray(
            probability - float(payload.get("pair_decision_threshold", 0.5)),
            dtype=np.float32,
        )
    margins = [
        {prop: float(margin_columns[prop][index]) for prop in properties}
        for index in range(len(candidates))
    ]
    scores = []
    for index, item in enumerate(margins):
        values = list(item.values())
        scores.append(
            4.0 * sum(value >= 0.0 for value in values)
            + 2.0 * min(values)
            + sum(values) / max(len(values), 1)
            + 0.35 * float(source_tanimotos[index])
            + 0.20 * float(retrieval_similarities[index])
            + 0.02 * math.log1p(max(int(frequencies[index]), 0))
        )
    return np.asarray(scores, dtype=np.float32), margins


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("MuMO dev protocol is fixed at exactly n=20")
    evidence = json.loads((args.run_root / "merged" / "summary.json").read_text(encoding="utf-8"))
    if evidence.get("passed") is not True or evidence.get("next_transition") != "closed_loop_dev_n20":
        raise ValueError("Evidence gate did not authorize closed_loop_dev_n20")
    models = load_models(args.run_root)
    transforms = load_transforms(args.run_root / "merged" / "transforms.jsonl")
    analog_library = load_fit_analog_library(args.run_root)
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("Invalid shard index/count")
    raw_rows = [
        row
        for row in protocol.read_jsonl(args.dev_sources_jsonl)
        if protocol.stable_shard(
            str(row["_uca_source_group"]),
            seed=int(args.seed),
            shard_count=int(args.shard_count),
        ) == int(args.shard_index)
    ]
    raw_rows.sort(key=lambda row: (str(row["_uca_task_id"]), str(row["_uca_source_group"])))

    output = []
    task_counts: Counter[str] = Counter()
    candidate_sources: Counter[str] = Counter()
    proposal_counts: list[int] = []
    unique_attempt_counts: list[int] = []
    repeated_attempt_rows = 0
    noop_fallback_conditions = 0
    for index, raw in enumerate(raw_rows):
        row = condition_row(raw, index)
        source = str(row["source_smiles"])
        source_canonical = canonical_smiles(source)
        source_feature = candidate_feature(source)
        if not source_canonical or source_feature is None:
            raise ValueError(f"Invalid dev source: {row['condition_id']}")
        properties = tuple(str(row["external_task_properties"]).split(","))
        task_transforms = transforms.get(str(row["external_task_key"]), [])
        candidates: dict[str, tuple[dict[str, object], np.ndarray]] = {}
        for split in delta.fragment_splits(source, int(args.min_core_heavy_atoms), int(args.max_variable_heavy_atoms)):
            ranked = []
            for transform in task_transforms:
                similarity = delta.variable_similarity(split.variable, str(transform["source_variable"]))
                if similarity >= float(args.min_retrieval_similarity):
                    ranked.append((similarity, transform))
            ranked.sort(key=lambda item: (item[0], int(item[1]["frequency"])), reverse=True)
            for similarity, transform in ranked[: int(args.max_transforms_per_fragment)]:
                generated = delta.canonical_smiles(delta.join_fragments(split.core, str(transform["target_variable"])))
                if not generated or generated == source_canonical:
                    continue
                tanimoto = float(delta.graph.revise.morgan_tanimoto(source, generated))
                if tanimoto < float(args.min_source_tanimoto):
                    continue
                generated_feature = candidate_feature(generated)
                if generated_feature is None:
                    continue
                record = {
                    "generated_smiles": generated,
                    "method": "fit_only_pair_verifier_delta_closed_loop",
                    "source_tanimoto": tanimoto,
                    "retrieval_similarity": float(similarity),
                    "transform_frequency": int(transform["frequency"]),
                    "source_variable": str(transform["source_variable"]),
                    "target_variable": str(transform["target_variable"]),
                }
                previous = candidates.get(generated)
                if previous is None or (
                    float(record["retrieval_similarity"]), int(record["transform_frequency"])
                ) > (
                    float(previous[0]["retrieval_similarity"]), int(previous[0]["transform_frequency"])
                ):
                    candidates[generated] = (record, generated_feature)
        for generated, tanimoto, generated_feature in retrieve_fit_analogs(
            source_feature,
            analog_library.get(str(row["external_task_id"])),
            min_tanimoto=float(args.min_source_tanimoto),
            limit=int(args.max_transforms_per_fragment),
        ):
            if generated == source_canonical or generated in candidates:
                continue
            candidates[generated] = (
                {
                    "generated_smiles": generated,
                    "method": "fit_only_target_analog_retrieval",
                    "source_tanimoto": tanimoto,
                    "retrieval_similarity": tanimoto,
                    "transform_frequency": 1,
                    "source_variable": "",
                    "target_variable": "",
                },
                generated_feature,
            )
        candidate_items = list(candidates.values())
        if not candidate_items:
            candidate_items = [source_noop_candidate(source_canonical, source_feature)]
            noop_fallback_conditions += 1
        proposal_counts.append(len(candidate_items))
        scores, candidate_margins = score_candidates_batch(
            source_feature,
            [item[1] for item in candidate_items],
            properties=properties,
            models=models,
            source_tanimotos=[float(item[0]["source_tanimoto"]) for item in candidate_items],
            retrieval_similarities=[float(item[0]["retrieval_similarity"]) for item in candidate_items],
            frequencies=[int(item[0]["transform_frequency"]) for item in candidate_items],
        )
        scored = []
        for candidate_index, (record, _features) in enumerate(candidate_items):
            margins = candidate_margins[candidate_index]
            scored.append(
                {
                    **record,
                    "verifier_margins_json": json.dumps(margins, sort_keys=True),
                    "verifier_min_margin": min(margins.values()),
                    "verifier_mean_margin": sum(margins.values()) / max(len(margins), 1),
                    "selection_score": float(scores[candidate_index]),
                }
            )
        ranked_candidates = sorted(scored, key=lambda item: float(item["selection_score"]), reverse=True)
        unique_attempt_count = min(len(ranked_candidates), int(args.candidate_budget))
        unique_attempt_counts.append(unique_attempt_count)
        frozen_attempts = freeze_attempts(ranked_candidates, budget=int(args.candidate_budget))
        for rank, (candidate, is_repeat, unique_rank) in enumerate(frozen_attempts, start=1):
            output.append(
                {
                    **row,
                    **candidate,
                    "candidate_rank": rank,
                    "candidate_selected": rank == 1,
                    "candidate_attempt_is_repeat": is_repeat,
                    "candidate_unique_rank": unique_rank,
                    "candidate_valid": True,
                    "candidate_source_similarity_pass": True,
                    "candidate_is_noop": bool(candidate.get("candidate_is_noop", False)),
                }
            )
            repeated_attempt_rows += int(is_repeat)
            candidate_sources[str(candidate["method"])] += 1
        task_counts[str(row["external_task_id"])] += 1
        if (index + 1) % 100 == 0:
            print(f"[closed-loop-dev] {index + 1}/{len(raw_rows)}", flush=True)

    write_rows(args.output_csv, output)
    manifest = {
        "protocol": "mumo_fit_only_pair_verifier_closed_loop_dev_v1",
        "data_role": "fit_models_to_disjoint_train_dev_sources",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "candidate_budget": 20,
        "attempted_candidates_per_condition": 20,
        "conditions": len(raw_rows),
        "candidate_rows": len(output),
        "unique_candidates_total": sum(unique_attempt_counts),
        "unique_valid_candidates_total": sum(unique_attempt_counts),
        "mean_unique_candidates_per_condition": (
            sum(unique_attempt_counts) / max(len(unique_attempt_counts), 1)
        ),
        "min_unique_candidates_per_condition": min(unique_attempt_counts, default=0),
        "repeated_attempt_rows": repeated_attempt_rows,
        "repeat_policy": "cycle_ranked_valid_candidates_only_when_unique_support_below_20",
        "noop_fallback_conditions": noop_fallback_conditions,
        "noop_policy": "repeat_source_only_when_constrained_unique_support_is_empty",
        "task_conditions": dict(sorted(task_counts.items())),
        "candidate_sources": dict(sorted(candidate_sources.items())),
        "selection": "hybrid_fit_delta_analog_then_pair_margin_similarity_frequency",
        "fit_only_transform_count": sum(len(rows) for rows in transforms.values()),
        "internal_proposals_total": sum(proposal_counts),
        "mean_internal_proposals_per_condition": (
            sum(proposal_counts) / max(len(proposal_counts), 1)
        ),
        "max_internal_proposals_per_condition": max(proposal_counts, default=0),
        "max_transforms_per_fragment": int(args.max_transforms_per_fragment),
        "fit_only_analog_limit_per_condition": int(args.max_transforms_per_fragment),
        "verifier_properties": list(protocol.PROPERTIES),
        "exact_property_margins": ["qed"],
        "source_similarity_floor": float(args.min_source_tanimoto),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "dev_source_view": str(args.dev_sources_jsonl),
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
