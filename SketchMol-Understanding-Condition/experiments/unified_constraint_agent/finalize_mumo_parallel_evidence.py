#!/usr/bin/env python3
"""Merge parallel delta/verifier artifacts and enforce the v8 evidence gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=32)
    parser.add_argument("--min-label-coverage", type=float, default=0.95)
    parser.add_argument("--min-threshold-recall", type=float, default=0.85)
    parser.add_argument("--min-dev-pairs", type=int, default=100)
    parser.add_argument("--min-unique-transforms", type=int, default=10000)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_manifest = load_json(args.run_root / "data" / "manifest.json")
    failures: list[str] = []
    if prepare_manifest.get("protocol") != protocol.PROTOCOL_VERSION:
        failures.append("prepare protocol mismatch")
    if prepare_manifest.get("evaluation_target_access") is not False:
        failures.append("evaluation_target_access must be false")
    if int(prepare_manifest.get("candidate_budget", 0)) != 20:
        failures.append("candidate budget drifted from n=20")
    if int(prepare_manifest.get("selected_audit_canonical_source_overlap", -1)) != 0:
        failures.append("selected train/audit canonical-source overlap is nonzero")

    counts: Counter[tuple[str, str, str]] = Counter()
    effect_sums: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    effect_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    first_condition: dict[tuple[str, str, str], str] = {}
    delta_rows = 0
    delta_observations = 0
    for shard in range(int(args.shard_count)):
        manifest_path = args.run_root / "delta" / f"manifest_{shard:03d}.json"
        rows_path = args.run_root / "delta" / f"transforms_{shard:03d}.jsonl"
        if not manifest_path.is_file() or not rows_path.is_file():
            failures.append(f"missing delta shard {shard:03d}")
            continue
        manifest = load_json(manifest_path)
        if manifest.get("evaluation_target_access") is not False or manifest.get("evaluation_oracle_access") is not False:
            failures.append(f"delta shard {shard:03d} violated target/oracle contract")
        delta_rows += int(manifest.get("input_rows", 0))
        delta_observations += int(manifest.get("transform_observations", 0))
        for row in protocol.read_jsonl(rows_path):
            key = (
                str(row["task_key"]),
                str(row["source_variable"]),
                str(row["target_variable"]),
            )
            frequency = int(row["frequency"])
            counts[key] += frequency
            first_condition.setdefault(key, str(row.get("first_train_condition_id", "")))
            for prop, value in dict(row.get("effect_sums", {})).items():
                effect_sums[key][str(prop)] += float(value)
            for prop, value in dict(row.get("effect_counts", {})).items():
                effect_counts[key][str(prop)] += int(value)

    merged_transforms = []
    for key in sorted(counts, key=lambda item: (-counts[item], item)):
        task_key, source_variable, target_variable = key
        averages = {
            prop: float(effect_sums[key][prop] / count)
            for prop, count in effect_counts[key].items()
            if count > 0
        }
        merged_transforms.append(
            {
                "task_key": task_key,
                "source_variable": source_variable,
                "target_variable": target_variable,
                "frequency": int(counts[key]),
                "first_train_condition_id": first_condition[key],
                "effects": dict(sorted(averages.items())),
                "effect_counts": dict(sorted(effect_counts[key].items())),
            }
        )
    protocol.write_jsonl(args.run_root / "merged" / "transforms.jsonl", merged_transforms)
    if len(merged_transforms) < int(args.min_unique_transforms):
        failures.append(
            f"unique transforms {len(merged_transforms)} < {int(args.min_unique_transforms)}"
        )

    feature_rows = 0
    for shard in range(int(args.shard_count)):
        path = args.run_root / "features" / f"manifest_{shard:03d}.json"
        npz_path = args.run_root / "features" / f"features_{shard:03d}.npz"
        if not path.is_file() or not npz_path.is_file():
            failures.append(f"missing feature shard {shard:03d}")
            continue
        manifest = load_json(path)
        if manifest.get("evaluation_target_access") is not False or manifest.get("evaluation_oracle_access") is not False:
            failures.append(f"feature shard {shard:03d} violated target/oracle contract")
        feature_rows += int(manifest.get("output_rows", 0))

    coverage = dict(prepare_manifest.get("raw_label_coverage", {}))
    coverage_rates = {}
    for prop in protocol.PROPERTIES:
        item = dict(coverage.get(prop, {}))
        rate = float(item.get("rate", 0.0))
        coverage_rates[prop] = rate
        if rate < float(args.min_label_coverage):
            failures.append(f"{prop} raw label coverage {rate:.3f} < {float(args.min_label_coverage):.3f}")

    verifier_metrics = {}
    for prop in protocol.PROPERTIES:
        metrics_path = args.run_root / "verifiers" / prop / "metrics.json"
        model_path = args.run_root / "verifiers" / prop / "model.joblib"
        if not metrics_path.is_file() or not model_path.is_file():
            failures.append(f"missing verifier artifact for {prop}")
            continue
        metrics = load_json(metrics_path)
        verifier_metrics[prop] = metrics
        if metrics.get("evaluation_target_access") is not False or metrics.get("evaluation_oracle_access") is not False:
            failures.append(f"{prop} verifier violated target/oracle contract")
        pairwise = dict(metrics.get("pairwise", {}))
        eligible = int(pairwise.get("eligible_dev_pairs", 0))
        recall = float(pairwise.get("threshold_recall", 0.0))
        if eligible < int(args.min_dev_pairs):
            failures.append(f"{prop} eligible dev pairs {eligible} < {int(args.min_dev_pairs)}")
        if recall < float(args.min_threshold_recall):
            failures.append(
                f"{prop} threshold recall {recall:.3f} < {float(args.min_threshold_recall):.3f}"
            )

    passed = not failures
    summary = {
        "protocol": protocol.PROTOCOL_VERSION,
        "stage": "parallel_evidence_gate",
        "passed": passed,
        "next_transition": "closed_loop_dev_n20" if passed else "STOP",
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "candidate_budget": 20,
        "prepare_manifest": str(args.run_root / "data" / "manifest.json"),
        "selected_train_rows": int(prepare_manifest.get("selected_rows", 0)),
        "fit_rows": int(prepare_manifest.get("fit_rows", 0)),
        "dev_rows": int(prepare_manifest.get("dev_rows", 0)),
        "feature_rows": feature_rows,
        "delta_input_rows_across_shards": delta_rows,
        "delta_transform_observations": delta_observations,
        "unique_merged_transforms": len(merged_transforms),
        "raw_label_coverage_rates": coverage_rates,
        "verifier_metrics": {
            prop: {
                "mae": metrics.get("mae"),
                "spearman": metrics.get("spearman"),
                "fit_unique_molecules": metrics.get("fit_unique_molecules"),
                "dev_unique_molecules": metrics.get("dev_unique_molecules"),
                "pairwise": metrics.get("pairwise"),
            }
            for prop, metrics in verifier_metrics.items()
        },
        "gates": {
            "min_label_coverage": float(args.min_label_coverage),
            "min_threshold_recall": float(args.min_threshold_recall),
            "min_dev_pairs": int(args.min_dev_pairs),
            "min_unique_transforms": int(args.min_unique_transforms),
        },
        "failures": failures,
    }
    protocol.write_json(args.run_root / "merged" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed or not args.fail_on_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
