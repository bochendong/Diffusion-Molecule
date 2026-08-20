#!/usr/bin/env python3
"""Write the C1 go/no-go summary from sampling stats and honest any@20 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REAL_TASK_KEYS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "MW:increase",
    "RB:decrease",
    "SA:decrease",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-summary", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--official-gsk3b-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sampling = json.loads(args.sampling_summary.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    by_task = {
        str(row.get("task_key", "")): row
        for row in metrics
        if isinstance(row, dict)
    }
    gsk3b = by_task.get("GSK3B:increase", {})
    measured = [
        row
        for row in metrics
        if isinstance(row, dict) and row.get("Acc_all(0.65)") not in ("", None)
    ]
    overall_065 = mean_metric(measured, "Acc_all(0.65)")
    overall_015 = mean_metric(measured, "Acc_all(0.15)")
    validity = mean_metric(measured, "Validity")
    gsk3b_065 = float(gsk3b.get("Acc_all(0.65)") or 0.0)
    rb = by_task.get("RB:decrease", {})
    drd2 = by_task.get("DRD2:decrease+MW:decrease+SA:decrease", {})
    real_rows = [
        by_task[key]
        for key in REAL_TASK_KEYS
        if key in by_task and by_task[key].get("Acc_all(0.65)") not in ("", None)
    ]
    synthetic_rows = [
        row
        for key, row in by_task.items()
        if key and key not in REAL_TASK_KEYS and row.get("Acc_all(0.65)") not in ("", None)
    ]
    real5_065 = mean_metric(real_rows, "Acc_all(0.65)")
    synthetic_065 = mean_metric(synthetic_rows, "Acc_all(0.65)")
    gates = dict(prereg["gates"])
    checks = {
        "gsk3b_any20_t0_65": gsk3b_065 >= float(gates["gsk3b_any20_t0_65"]),
        "validity": float(validity or 0.0) >= float(gates["validity"]),
        "exact_attempts": int(sampling.get("attempts_per_condition") or 0)
        == int(gates["exact_attempts"]),
        "no_ranking": sampling.get("molecular_candidate_ranking") is False,
        "no_task_router": sampling.get("task_router") is False,
    }
    if "overall_any20_t0_65" in gates:
        checks["overall_any20_t0_65"] = float(overall_065 or 0.0) >= float(
            gates["overall_any20_t0_65"]
        )
    if "stop_not_degenerate" in gates:
        lo, hi = [float(item) for item in gates["stop_not_degenerate"]]
        frac = float(sampling.get("stop_fraction") or 0.0)
        checks["stop_not_degenerate"] = lo <= frac <= hi
    if "family_not_uniform" in gates:
        delta = float(sampling.get("mean_abs_graph_prior_delta") or 0.0)
        checks["family_not_uniform"] = delta >= float(gates["family_not_uniform"])
    if "real5_any20_t0_65" in gates:
        checks["real5_any20_t0_65"] = float(real5_065 or 0.0) >= float(
            gates["real5_any20_t0_65"]
        )
    if "rb_any20_t0_65" in gates:
        checks["rb_any20_t0_65"] = float(rb.get("Acc_all(0.65)") or 0.0) >= float(
            gates["rb_any20_t0_65"]
        )
    if "drd2_any20_t0_65" in gates:
        checks["drd2_any20_t0_65"] = float(drd2.get("Acc_all(0.65)") or 0.0) >= float(
            gates["drd2_any20_t0_65"]
        )
    if prereg.get("oracle_in_environment") is False:
        checks["no_oracle_in_environment"] = sampling.get("oracle_in_environment") is False
    summary = {
        "protocol": prereg["protocol"],
        "series": "C",
        "decision": "go" if all(checks.values()) else "stop",
        "checks": checks,
        "gsk3b_any20_t0_65": gsk3b.get("Acc_all(0.65)"),
        "gsk3b_any20_t0_15": gsk3b.get("Acc_all(0.15)"),
        "overall_any20_t0_65": overall_065,
        "overall_any20_t0_15": overall_015,
        "real5_any20_t0_65": real5_065,
        "synthetic_any20_t0_65": synthetic_065,
        "real_task_keys": list(REAL_TASK_KEYS),
        "rb_any20_t0_65": rb.get("Acc_all(0.65)"),
        "drd2_any20_t0_65": drd2.get("Acc_all(0.65)"),
        "validity": validity,
        "family_fraction_fragment": sampling.get("family_fraction_fragment"),
        "family_counts": sampling.get("family_counts"),
        "step_counts": sampling.get("step_counts"),
        "decision_counts": sampling.get("decision_counts"),
        "stop_fraction": sampling.get("stop_fraction"),
        "mean_stop_prob": sampling.get("mean_stop_prob"),
        "mean_graph_prior": sampling.get("mean_graph_prior"),
        "mean_abs_graph_prior_delta": sampling.get("mean_abs_graph_prior_delta"),
        "graph_prior_by_task": sampling.get("graph_prior_by_task"),
        "c1_overall_any20_t0_65": prereg.get("c1_overall_any20_t0_65"),
        "loaded_conditions": sampling.get("loaded_conditions"),
        "candidate_rows": sampling.get("candidate_rows"),
        "by_task": {
            key: {
                "n": row.get("n"),
                "validity": row.get("Validity"),
                "acc_all_0_65": row.get("Acc_all(0.65)"),
                "acc_all_0_15": row.get("Acc_all(0.15)"),
                "mean_best_source_tanimoto": row.get("mean_best_source_tanimoto"),
            }
            for key, row in by_task.items()
            if key
        },
        "sampling_summary": str(args.sampling_summary),
        "metrics_json": str(args.metrics_json),
    }
    if args.official_gsk3b_json is not None and args.official_gsk3b_json.exists():
        official = json.loads(args.official_gsk3b_json.read_text(encoding="utf-8"))
        if isinstance(official, list):
            gsk_rows = [
                row
                for row in official
                if isinstance(row, dict) and row.get("task_key") == "GSK3B:increase"
            ]
            official = gsk_rows[0] if gsk_rows else {}
        summary["official_gsk3b_any20_t0_65"] = official.get("Acc_all(0.65)")
        summary["official_gsk3b_any20_t0_15"] = official.get("Acc_all(0.15)")
        summary["official_gsk3b_n"] = official.get("n")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def mean_metric(rows: list[dict], key: str) -> float | None:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw in ("", None):
            continue
        values.append(float(raw))
    if not values:
        return None
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
