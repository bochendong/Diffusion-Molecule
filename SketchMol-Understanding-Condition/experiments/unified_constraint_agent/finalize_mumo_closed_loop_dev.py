#!/usr/bin/env python3
"""Gate fixed-n MuMO dev results before any common-LLM GPU signal."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--oracle-summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--min-sr", type=float, default=0.65)
    parser.add_argument("--min-ood-sr", type=float, default=0.60)
    parser.add_argument("--min-validity", type=float, default=0.95)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_json(args.candidate_manifest)
    oracle = load_json(args.oracle_summary)
    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    overall = next(row for row in rows if row["external_suite"] == "all")
    split_rows = [row for row in rows if row["external_suite"] == "mumo"]
    split_counts = {split: 0 for split in ("ind", "ood")}
    split_successes = {split: 0.0 for split in ("ind", "ood")}
    for row in split_rows:
        split = row["external_task_split"]
        count = int(row["input_groups"])
        split_counts[split] += count
        split_successes[split] += count * float(row["success_rate"])
    split_rates = {split: split_successes[split] / max(split_counts[split], 1) for split in split_counts}
    failures = []
    if manifest.get("evaluation_target_access") is not False or manifest.get("evaluation_oracle_access") is not False:
        failures.append("candidate generation accessed evaluation target/oracle")
    if int(manifest.get("candidate_budget", 0)) != 20:
        failures.append("candidate budget is not n=20")
    if int(manifest.get("candidate_rows", 0)) != 20 * int(manifest.get("conditions", 0)):
        failures.append("candidate row contract mismatch")
    missing = {key: int(value) for key, value in dict(oracle.get("missing_counts", {})).items() if int(value) > 0}
    if missing:
        failures.append(f"oracle missing values: {missing}")
    official_coverage = float(overall["official_evaluable_rate"])
    if official_coverage < 1.0:
        failures.append(f"official oracle coverage {official_coverage:.3f} < 1.000")
    sr = float(overall["success_rate"])
    validity = float(overall["validity"])
    if sr < float(args.min_sr):
        failures.append(f"overall SR {sr:.3f} < {float(args.min_sr):.3f}")
    if split_rates["ood"] < float(args.min_ood_sr):
        failures.append(f"OOD SR {split_rates['ood']:.3f} < {float(args.min_ood_sr):.3f}")
    if validity < float(args.min_validity):
        failures.append(f"validity {validity:.3f} < {float(args.min_validity):.3f}")
    summary = {
        "protocol": "mumo_closed_loop_dev_n20_gate_v1",
        "passed": not failures,
        "next_transition": "common_llm_1p5b_residual" if not failures else "STOP",
        "candidate_budget": 20,
        "conditions": int(manifest["conditions"]),
        "candidate_rows": int(manifest["candidate_rows"]),
        "attempted_candidates_total": int(manifest["attempted_candidates_total"]),
        "unique_candidates_total": int(manifest["unique_candidates_total"]),
        "unique_valid_candidates_total": int(manifest["unique_valid_candidates_total"]),
        "mean_unique_candidates_per_condition": float(
            manifest["mean_unique_candidates_per_condition"]
        ),
        "min_unique_candidates_per_condition": int(manifest["min_unique_candidates_per_condition"]),
        "repeated_attempt_rows": int(manifest["repeated_attempt_rows"]),
        "success_rate": sr,
        "ind_success_rate": split_rates["ind"],
        "ood_success_rate": split_rates["ood"],
        "validity": validity,
        "oracle_missing_counts": missing,
        "official_oracle_coverage": official_coverage,
        "evaluation_target_access": False,
        "failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
