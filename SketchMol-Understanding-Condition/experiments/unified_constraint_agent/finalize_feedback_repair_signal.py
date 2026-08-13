#!/usr/bin/env python3
"""Gate the v14a LLM signal against its matched deterministic controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-manifest", required=True, type=Path)
    parser.add_argument("--deterministic-manifest", required=True, type=Path)
    parser.add_argument("--llm-summary", required=True, type=Path)
    parser.add_argument("--deterministic-summary", required=True, type=Path)
    parser.add_argument("--controller-validation", required=True, type=Path)
    parser.add_argument("--baseline-controller-validation", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--llm-oracle-summary", required=True, type=Path)
    parser.add_argument("--deterministic-oracle-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-overall-sr", type=float, default=0.65)
    parser.add_argument("--min-ood-sr", type=float, default=0.60)
    parser.add_argument("--min-llm-gain", type=float, default=0.02)
    parser.add_argument("--max-noop-rate", type=float, default=0.20)
    parser.add_argument("--min-mean-unique", type=float, default=14.0)
    parser.add_argument("--min-feedback-accuracy", type=float, default=0.70)
    parser.add_argument("--max-common-retention-drop", type=float, default=0.05)
    return parser.parse_args(argv)


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return dict(value)


def rates(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    overall = next(row for row in rows if row["external_suite"] == "all")
    counts = {"ind": 0, "ood": 0}
    successes = {"ind": 0.0, "ood": 0.0}
    for row in rows:
        if row["external_suite"] != "mumo":
            continue
        split = row["external_task_split"]
        count = int(row["input_groups"])
        counts[split] += count
        successes[split] += count * float(row["success_rate"])
    return {
        "conditions": int(overall["input_groups"]),
        "success_rate": float(overall["success_rate"]),
        "ind_success_rate": successes["ind"] / max(counts["ind"], 1),
        "ood_success_rate": successes["ood"] / max(counts["ood"], 1),
        "validity": float(overall["validity"]),
        "official_oracle_coverage": float(overall["official_evaluable_rate"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    llm_manifest = load(args.llm_manifest)
    deterministic_manifest = load(args.deterministic_manifest)
    llm = rates(args.llm_summary)
    deterministic = rates(args.deterministic_summary)
    validation = load(args.controller_validation)
    baseline_validation = load(args.baseline_controller_validation)
    data = load(args.data_manifest)
    training = load(args.training_summary)
    llm_oracle = load(args.llm_oracle_summary)
    deterministic_oracle = load(args.deterministic_oracle_summary)
    llm_noop_rate = int(llm_manifest["noop_attempt_rows"]) / max(
        int(llm_manifest["candidate_rows"]), 1
    )
    missing = {
        "llm": dict(llm_oracle.get("missing_counts", {})),
        "deterministic": dict(deterministic_oracle.get("missing_counts", {})),
    }
    checks = {
        "exact_n20_matched_200_conditions": int(llm_manifest.get("candidate_rows", 0))
        == int(deterministic_manifest.get("candidate_rows", -1))
        == 4000
        and int(llm_manifest.get("conditions", 0))
        == int(deterministic_manifest.get("conditions", -1))
        == 200,
        "balanced_ind_ood_signal": int(llm_manifest.get("ind_conditions", 0)) == 100
        and int(llm_manifest.get("ood_conditions", 0)) == 100,
        "target_oracle_hidden_during_generation": all(
            item.get("evaluation_target_access") is False
            and item.get("evaluation_oracle_access") is False
            and item.get("output_selection") == "none"
            for item in (llm_manifest, deterministic_manifest)
        ),
        "fit_only_feedback_data_disjoint": data.get("evaluation_target_access") is False
        and data.get("evaluation_oracle_access") is False
        and int(data.get("source_group_overlap", -1)) == 0,
        "adapter_finite": int(training.get("adapter_nonfinite_parameters", -1)) == 0,
        "controller_feedback_accuracy": float(
            dict(validation.get("feedback", {})).get("top1_action_accuracy", 0.0)
        )
        >= float(args.min_feedback_accuracy),
        "common_task_retention": float(
            dict(validation.get("common_retention", {})).get(
                "mean_canonical_action_log_probability", -math.inf
            )
        )
        >= float(
            dict(baseline_validation.get("common_retention", {})).get(
                "mean_canonical_action_log_probability", math.inf
            )
        )
        - float(args.max_common_retention_drop),
        "official_oracle_complete": float(llm["official_oracle_coverage"]) == 1.0
        and float(deterministic["official_oracle_coverage"]) == 1.0
        and not any(int(value) for table in missing.values() for value in table.values()),
        "validity_complete": float(llm["validity"]) >= 0.95,
        "minimum_overall_signal": float(llm["success_rate"]) >= float(args.min_overall_sr),
        "minimum_ood_signal": float(llm["ood_success_rate"]) >= float(args.min_ood_sr),
        "llm_causal_gain": float(llm["success_rate"])
        >= float(deterministic["success_rate"]) + float(args.min_llm_gain),
        "noop_budget": llm_noop_rate <= float(args.max_noop_rate),
        "candidate_diversity": float(llm_manifest.get("mean_unique_candidates_per_condition", 0.0))
        >= float(args.min_mean_unique),
    }
    passed = all(checks.values())
    result = {
        "protocol": "common_llm_feedback_repair_signal_v14a_gate_v1",
        "passed": passed,
        "decision": "scale_to_full_dev" if passed else "STOP",
        "next_transition": "feedback_repair_full_dev_n20" if passed else "STOP",
        "candidate_budget": 20,
        "signal_conditions": 200,
        "llm": llm,
        "deterministic": deterministic,
        "llm_gains": {
            key: float(llm[key]) - float(deterministic[key])
            for key in ("success_rate", "ind_success_rate", "ood_success_rate", "validity")
        },
        "llm_noop_rate": llm_noop_rate,
        "llm_mean_unique_candidates": float(
            llm_manifest.get("mean_unique_candidates_per_condition", 0.0)
        ),
        "controller_deterministic_divergence_rate": float(
            llm_manifest.get("controller_deterministic_divergence_rate", 0.0)
        ),
        "controller_validation": validation,
        "baseline_controller_validation": baseline_validation,
        "checks": checks,
        "failures": [name for name, value in checks.items() if not value],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
