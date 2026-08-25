#!/usr/bin/env python3
"""Aggregate paired P17/P18 frozen expanded metrics and preregistered uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_moledit_table1_anyk import (  # noqa: E402
    Chemistry,
    evaluate_prediction,
    load_candidates,
    load_references,
    task_key,
    task_specs_for_reference,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref_id(row: Mapping[str, object]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or row.get("pair_id") or "")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"successes": successes, "n": n, "estimate": p, "low": center - spread, "high": center + spread}


def table_outcomes(reference: Path, candidates: Path, k: int) -> dict[str, dict[str, object]]:
    refs = load_references(reference)
    pools = load_candidates(candidates, method_filter=None, candidate_limit=k)
    chem = Chemistry()
    output = {}
    for key, row in refs.items():
        specs = task_specs_for_reference(row)
        evaluated = [evaluate_prediction(row, smi, specs, chem=chem, thresholds=(0.65, 0.15)) for smi in pools[key]]
        similarities = [float(item["source_tanimoto"]) for item in evaluated if item.get("source_tanimoto") is not None]
        output[key] = {
            "task": task_key(specs),
            "validity": float(any(bool(item.get("valid")) for item in evaluated)),
            "property_anyk": float(any(bool(item.get("property_success")) for item in evaluated)),
            "acc_0.15": float(any(bool(item.get("success_t0.15")) for item in evaluated)),
            "strict_acc_0.65": float(any(bool(item.get("success_t0.65")) for item in evaluated)),
            "mean_best_source_similarity": max(similarities) if similarities else 0.0,
        }
    return output


def table_summary(outcomes: dict[str, dict[str, object]]) -> dict:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in outcomes.values():
        grouped[str(row["task"])].append(row)
    metrics = ("validity", "property_anyk", "acc_0.15", "strict_acc_0.65", "mean_best_source_similarity")
    by_task = {
        task: {metric: mean([float(row[metric]) for row in rows]) for metric in metrics}
        for task, rows in sorted(grouped.items())
    }
    return {
        "macro": {metric: mean([values[metric] for values in by_task.values()]) for metric in metrics},
        "by_task": by_task,
        "n_by_task": {task: len(rows) for task, rows in sorted(grouped.items())},
    }


def paired_bootstrap(
    left: dict[str, dict[str, object]], right: dict[str, dict[str, object]], *, seed: int, replicates: int
) -> dict[str, dict[str, float | int]]:
    if set(left) != set(right):
        raise ValueError("paired Table1 condition ids differ")
    by_task: dict[str, list[str]] = defaultdict(list)
    for key, row in left.items():
        if row["task"] != right[key]["task"]:
            raise ValueError(f"task mismatch for {key}")
        by_task[str(row["task"])].append(key)
    rng = random.Random(seed)
    metrics = ("validity", "property_anyk", "acc_0.15", "strict_acc_0.65", "mean_best_source_similarity")
    draws = {metric: [] for metric in metrics}
    for _ in range(replicates):
        task_deltas = {metric: [] for metric in metrics}
        for keys in by_task.values():
            sampled = [keys[rng.randrange(len(keys))] for _ in keys]
            for metric in metrics:
                task_deltas[metric].append(mean([float(right[key][metric]) - float(left[key][metric]) for key in sampled]))
        for metric in metrics:
            draws[metric].append(mean(task_deltas[metric]))
    return {
        metric: {
            "seed": seed,
            "replicates": replicates,
            "delta": mean([float(right[key][metric]) - float(left[key][metric]) for key in left]),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
        }
        for metric, values in draws.items()
    }


def denovo_fraction(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[ref_id(row)].append(row)
    result = {}
    for stratum in ("all", "6p", "7p"):
        selected = {
            key: values for key, values in grouped.items()
            if stratum == "all" or str(values[0].get("property_count")) == stratum[:-1]
        }
        result[stratum] = {}
        for k in (1, 4, 8):
            best = []
            for values in selected.values():
                prefix = sorted(values, key=lambda row: int(float(row["candidate_rank"])))[:k]
                best.append(max(float(row["direct_candidate_strict_fraction"]) for row in prefix))
            result[stratum][str(k)] = mean(best)
    return result


def denovo_block(metrics_path: Path, annotated_path: Path) -> dict:
    metrics = load(metrics_path)
    return {
        "records": {f"{row['stratum']}@{row['k']}": row for row in metrics["records"]},
        "mean_best_property_fraction": denovo_fraction(annotated_path),
    }


def numeric_delta(right: dict, left: dict) -> dict:
    return {
        key: float(right[key]) - float(left[key])
        for key in right
        if key in left and isinstance(right[key], (int, float)) and isinstance(left[key], (int, float))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--adapter-lock", required=True, type=Path)
    parser.add_argument("--validation-summary", required=True, type=Path)
    parser.add_argument("--p18-small-pilot", required=True, type=Path)
    parser.add_argument("--table-reference", required=True, type=Path)
    parser.add_argument("--p17-table-candidates", required=True, type=Path)
    parser.add_argument("--p18-table-candidates", required=True, type=Path)
    parser.add_argument("--p17-denovo-metrics", required=True, type=Path)
    parser.add_argument("--p18-denovo-metrics", required=True, type=Path)
    parser.add_argument("--p17-denovo-annotated", required=True, type=Path)
    parser.add_argument("--p18-denovo-annotated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    locked = manifest["locked_sha256"]
    current = {
        "table1_reference": sha_file(args.table_reference),
        "table1_prompts": sha_file(args.table_reference.parent / "table1_expanded.prompts.jsonl"),
        "denovo_reference": sha_file(args.table_reference.parent / "denovo_expanded.reference.csv"),
        "denovo_prompts": sha_file(args.table_reference.parent / "denovo_expanded.prompts.jsonl"),
    }
    if current != locked:
        raise AssertionError(f"locked subset hashes changed: expected={locked}, current={current}")

    table = {"p17": {}, "p18": {}}
    uncertainty = {}
    drd2 = {}
    drd2_key = "DRD2:decrease+MW:decrease+SA:decrease"
    for k in (1, 4, 8):
        left = table_outcomes(args.table_reference, args.p17_table_candidates, k)
        right = table_outcomes(args.table_reference, args.p18_table_candidates, k)
        table["p17"][str(k)] = table_summary(left)
        table["p18"][str(k)] = table_summary(right)
        table["p18"][str(k)]["delta_p18_minus_p17"] = numeric_delta(
            table["p18"][str(k)]["macro"], table["p17"][str(k)]["macro"]
        )
        uncertainty[str(k)] = paired_bootstrap(left, right, seed=1919, replicates=10000)
        left_drd2 = [row for row in left.values() if row["task"] == drd2_key]
        right_drd2 = [row for row in right.values() if row["task"] == drd2_key]
        if len(left_drd2) != 10 or len(right_drd2) != 10:
            raise AssertionError("DRD2 stratum must have exact n=10 per model")
        lsuccess = sum(int(row["strict_acc_0.65"]) for row in left_drd2)
        rsuccess = sum(int(row["strict_acc_0.65"]) for row in right_drd2)
        drd2[str(k)] = {
            "p17_wilson95": wilson(lsuccess, 10),
            "p18_wilson95": wilson(rsuccess, 10),
            "delta_p18_minus_p17": (rsuccess - lsuccess) / 10,
            "step_size": 0.1,
        }

    d17 = denovo_block(args.p17_denovo_metrics, args.p17_denovo_annotated)
    d18 = denovo_block(args.p18_denovo_metrics, args.p18_denovo_annotated)
    denovo_delta = {key: numeric_delta(d18["records"][key], d17["records"][key]) for key in d18["records"]}
    fraction_delta = {
        stratum: {k: d18["mean_best_property_fraction"][stratum][k] - d17["mean_best_property_fraction"][stratum][k] for k in ("1", "4", "8")}
        for stratum in ("all", "6p", "7p")
    }

    small = load(args.p18_small_pilot)
    direction = {}
    for k in ("1", "4", "8"):
        direction[k] = {}
        for metric, expanded_delta in table["p18"][k]["delta_p18_minus_p17"].items():
            small_delta = small["table1"]["p18"][k]["delta_p18_minus_p17"].get(metric)
            if small_delta is not None:
                direction[k][metric] = {
                    "small_delta": small_delta,
                    "expanded_delta": expanded_delta,
                    "same_direction": (small_delta == 0 and expanded_delta == 0) or small_delta * expanded_delta > 0,
                }

    payload = {
        "protocol": "p19_frozen_expanded_paired_estimate_v1",
        "status_label": "expanded paired pilot estimate; not full benchmarks",
        "validation_reused_not_rerun": load(args.validation_summary),
        "validation_summary_sha256": sha_file(args.validation_summary),
        "locked_subset_hashes_verified": current,
        "frozen_adapter_lock_sha256": sha_file(args.adapter_lock),
        "frozen_adapter_lock_lines": [line for line in args.adapter_lock.read_text().splitlines() if line.strip()],
        "table1": table,
        "table1_paired_bootstrap95": uncertainty,
        "drd2_strict": drd2,
        "hard_denovo": {
            "p17": d17,
            "p18": d18,
            "record_delta_p18_minus_p17": denovo_delta,
            "mean_best_property_fraction_delta_p18_minus_p17": fraction_delta,
        },
        "direction_vs_p18_20_row_pilot": direction,
        "tuned_on_expanded_benchmark": False,
        "training_or_parameter_updates": False,
        "candidate_order": "greedy first then seven raw samples; K uses prefix",
        "static_candidate_pool": False,
        "property_reranking": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "protocol": payload["protocol"],
        "table1_rows": manifest["table1_rows"],
        "denovo_rows": manifest["denovo_rows"],
        "locked_hashes_verified": current == locked,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
