#!/usr/bin/env python3
"""Gate v11 on train-only preferences, frozen MuMO dev, and untouched Table1 smoke."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table1-summary", required=True, type=Path)
    parser.add_argument("--table1-official-root", required=True, type=Path)
    parser.add_argument("--mumo-gate", required=True, type=Path)
    parser.add_argument("--preference-manifest", required=True, type=Path)
    parser.add_argument("--reference-annotation-summary", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-table1-raw1", type=float, default=0.35)
    parser.add_argument("--min-table1-top5", type=float, default=0.65)
    parser.add_argument("--min-mumo-sr", type=float, default=0.739752)
    return parser.parse_args(argv)


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return dict(value)


def table_metrics(root: Path, budget: int, selection: str) -> dict[str, float]:
    path = (
        root
        / "moledit_table1"
        / f"n{budget}"
        / selection
        / "metrics"
        / "moledit_table_summary.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "measured"]
    if len(rows) != 10:
        raise ValueError(f"Expected 10 measured Table1 tasks: {path}")
    by_task = {row["task_key"]: row for row in rows}
    return {
        "validity": mean(float(row["Validity"]) for row in rows),
        "strict": mean(float(row["Acc_all(0.65)"]) for row in rows),
        "relaxed": mean(float(row["Acc_all(0.15)"]) for row in rows),
        "gsk3b_strict": float(by_task["GSK3B:increase"]["Acc_all(0.65)"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    table = load(args.table1_summary)
    mumo = load(args.mumo_gate)
    pref = load(args.preference_manifest)
    reference = load(args.reference_annotation_summary)
    training = load(args.training_summary)
    all_group = dict(dict(table["groups"])["all"])
    official = {
        "n1_raw": table_metrics(args.table1_official_root, 1, "raw"),
        "n5_finalizer": table_metrics(args.table1_official_root, 5, "finalizer"),
        "n20_finalizer": table_metrics(args.table1_official_root, 20, "finalizer"),
    }
    mumo_candidate = dict(mumo["candidate"])
    checks = {
        "fixed_n20": int(table.get("candidate_budget", 0)) == 20,
        "target_hidden": pref.get("evaluation_target_access") is False
        and pref.get("prompt_target_access") is False,
        "train_only_preferences": pref.get("data_role") == "train_only",
        "preference_split_disjoint": int(pref.get("pair_id_overlap", -1)) == 0
        and int(pref.get("mumo_source_group_overlap", -1)) == 0,
        "reference_anchored_dpo": training.get("reference_anchored_dpo") is True
        and reference.get("reference_margin_field") == "stable_reference_margin",
        "adapter_finite": int(training.get("adapter_nonfinite_parameters", -1)) == 0,
        "table1_reference_top5_preserved": float(
            all_group.get("reference_top_k_preserved_rate", 0.0)
        ) == 1.0,
        "table1_raw1_signal": official["n1_raw"]["strict"] >= float(args.min_table1_raw1),
        "table1_top5_signal": official["n5_finalizer"]["strict"]
        >= float(args.min_table1_top5),
        "table1_n20_ceiling_preserved": official["n20_finalizer"]["strict"] >= 0.76,
        "table1_validity": all(item["validity"] == 1.0 for item in official.values()),
        "table1_full_pool": float(all_group["full_pool_rate"]) == 1.0,
        "mumo_not_below_v9": float(mumo_candidate["success_rate"]) + 1e-12
        >= float(args.min_mumo_sr),
        "mumo_ood_not_below_v9": float(mumo_candidate["ood_success_rate"]) + 1e-12
        >= 0.7010310103092784,
        "mumo_validity": float(mumo_candidate["validity"]) == 1.0,
    }
    result = {
        "protocol": "unified_anchor_residual_v11_signal_gate_v1",
        "passed": all(checks.values()),
        "decision": "advance" if all(checks.values()) else "stop",
        "candidate_budget": 20,
        "evaluation_target_access": False,
        "table1": {
            "conditions": int(all_group["rows"]),
            "raw1_strict": float(all_group["llm_at_1_strict"]),
            "top5_strict": float(all_group["verifier_at_k_strict"]),
            "any20_strict": float(all_group["any_strict_at_20"]),
            "reference_top5_preserved_rate": float(
                all_group["reference_top_k_preserved_rate"]
            ),
            "official": official,
        },
        "mumo": mumo_candidate,
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
