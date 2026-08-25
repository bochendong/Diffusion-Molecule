#!/usr/bin/env python3
"""Compare frozen P21 with P18 on the exact P19 expanded subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P19_DIR = SCRIPT_DIR.parent / "p19_frozen_expanded_unified_benchmark"
if str(P19_DIR) not in sys.path:
    sys.path.insert(0, str(P19_DIR))
import aggregate_expanded as expanded  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-reference", required=True, type=Path)
    parser.add_argument("--p18-table", required=True, type=Path)
    parser.add_argument("--p21-table", required=True, type=Path)
    parser.add_argument("--p18-denovo-metrics", required=True, type=Path)
    parser.add_argument("--p21-denovo-metrics", required=True, type=Path)
    parser.add_argument("--p18-denovo-annotated", required=True, type=Path)
    parser.add_argument("--p21-denovo-annotated", required=True, type=Path)
    parser.add_argument("--p21-id", required=True, type=Path)
    parser.add_argument("--p21-ood", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    table = {"p18": {}, "p21": {}}
    uncertainty = {}
    for k in (1, 4, 8):
        left = expanded.table_outcomes(args.table_reference, args.p18_table, k)
        right = expanded.table_outcomes(args.table_reference, args.p21_table, k)
        table["p18"][str(k)] = expanded.table_summary(left)
        table["p21"][str(k)] = expanded.table_summary(right)
        table["p21"][str(k)]["delta_p21_minus_p18"] = expanded.numeric_delta(
            table["p21"][str(k)]["macro"], table["p18"][str(k)]["macro"]
        )
        uncertainty[str(k)] = expanded.paired_bootstrap(left, right, seed=2121, replicates=10000)

    d18 = expanded.denovo_block(args.p18_denovo_metrics, args.p18_denovo_annotated)
    d21 = expanded.denovo_block(args.p21_denovo_metrics, args.p21_denovo_annotated)
    id_metrics = load(args.p21_id)["metrics"]
    gates = {
        "id_denovo_greedy_validity": id_metrics["de_novo"]["greedy"]["valid_rate"] >= 0.95,
        "id_edit_greedy_validity": id_metrics["edit"]["greedy"]["valid_rate"] >= 0.84,
        "id_denovo_any3_validity": id_metrics["de_novo"]["any_at_3"]["valid_rate"] >= 0.9375,
        "id_edit_any3_validity": id_metrics["edit"]["any_at_3"]["valid_rate"] >= 0.9375,
        "table1_strict_k8_not_below_p18": table["p21"]["8"]["macro"]["strict_acc_0.65"] >= table["p18"]["8"]["macro"]["strict_acc_0.65"],
    }
    strict_delta_k1 = table["p21"]["1"]["delta_p21_minus_p18"]["strict_acc_0.65"]
    strict_delta_k8 = table["p21"]["8"]["delta_p21_minus_p18"]["strict_acc_0.65"]
    payload = {
        "protocol": "p21_frozen_expanded_estimate_v1",
        "status_label": "single-seed frozen expanded pilot estimate; not full benchmarks",
        "training": load(args.training_summary),
        "development": {"id": load(args.p21_id), "ood": load(args.p21_ood)},
        "table1": table,
        "table1_paired_bootstrap95": uncertainty,
        "hard_denovo": {
            "p18": d18,
            "p21": d21,
            "record_delta_p21_minus_p18": {
                key: expanded.numeric_delta(d21["records"][key], d18["records"][key])
                for key in d21["records"]
            },
        },
        "gates": gates,
        "all_retention_gates_passed": all(gates.values()),
        "primary_success": all(gates.values()) and (strict_delta_k1 > 0 or strict_delta_k8 > 0),
        "candidate_order": "greedy first then seven raw samples; no selection",
        "target_access_during_inference": False,
        "property_reranking": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_retention_gates_passed": payload["all_retention_gates_passed"],
        "primary_success": payload["primary_success"],
        "p18_table1_strict_k1": table["p18"]["1"]["macro"]["strict_acc_0.65"],
        "p21_table1_strict_k1": table["p21"]["1"]["macro"]["strict_acc_0.65"],
        "p18_table1_strict_k8": table["p18"]["8"]["macro"]["strict_acc_0.65"],
        "p21_table1_strict_k8": table["p21"]["8"]["macro"]["strict_acc_0.65"],
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
