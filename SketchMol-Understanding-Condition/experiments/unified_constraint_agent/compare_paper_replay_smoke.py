#!/usr/bin/env python3
"""Compare small paired De novo 2p-7p and Table1 paper replays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-manifest", required=True, type=Path)
    parser.add_argument("--denovo-summary", required=True, type=Path)
    parser.add_argument("--stable-ranking-summary", required=True, type=Path)
    parser.add_argument("--residual-ranking-summary", required=True, type=Path)
    parser.add_argument("--stable-table-root", required=True, type=Path)
    parser.add_argument("--residual-table-root", required=True, type=Path)
    parser.add_argument("--stable-candidates", required=True, type=Path)
    parser.add_argument("--residual-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-table1-drop", type=float, default=0.02)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(path)
    return dict(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def table_metrics(path: Path) -> dict[str, object]:
    rows = read_csv(path)
    measured = [row for row in rows if row.get("status") == "measured"]
    if len(rows) != 10 or len(measured) != 10:
        raise ValueError(f"Table1 requires 10/10 measured tasks: {path}")
    def avg(key: str) -> float:
        return mean(float(row[key]) for row in measured)
    by_task = {row["task_key"]: row for row in measured}
    return {
        "tasks": 10,
        "validity": avg("Validity"),
        "acc_all_0_65": avg("Acc_all(0.65)"),
        "acc_all_0_15": avg("Acc_all(0.15)"),
        "gsk3b_acc_all_0_65": float(by_task["GSK3B:increase"]["Acc_all(0.65)"]),
        "gsk3b_acc_all_0_15": float(by_task["GSK3B:increase"]["Acc_all(0.15)"]),
    }


def table_summary_path(root: Path, budget: int, selection: str) -> Path:
    return (
        root
        / "moledit_table1"
        / f"n{budget}"
        / selection
        / "metrics"
        / "moledit_table_summary.csv"
    )


def pool_identity(path: Path) -> dict[str, set[tuple[str, str]]]:
    output: dict[str, set[tuple[str, str]]] = {}
    for row in read_csv(path):
        identity = str(row.get("condition_id") or row.get("sample_id") or row.get("example_id") or "")
        output.setdefault(identity, set()).add(
            (str(row.get("generated_smiles", "")), str(row.get("graph_action_json", "")))
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_json(args.smoke_manifest)
    denovo_rows = read_csv(args.denovo_summary)
    denovo = {
        str(row["property_count"]): {
            "conditions": int(row["conditions"]),
            "validity": float(row["validity"]),
            "strict_success_rate": float(row["strict_success_rate"]),
        }
        for row in denovo_rows
        if row["setting"] == "best_of_20"
    }
    stable_ranking = load_json(args.stable_ranking_summary)
    residual_ranking = load_json(args.residual_ranking_summary)
    table_settings = ((1, "raw"), (5, "finalizer"), (20, "finalizer"))
    stable_table = {
        f"n{budget}_{selection}": table_metrics(
            table_summary_path(args.stable_table_root, budget, selection)
        )
        for budget, selection in table_settings
    }
    residual_table = {
        f"n{budget}_{selection}": table_metrics(
            table_summary_path(args.residual_table_root, budget, selection)
        )
        for budget, selection in table_settings
    }
    stable_group = dict(dict(stable_ranking["groups"])["all"])
    residual_group = dict(dict(residual_ranking["groups"])["all"])
    same_pool = pool_identity(args.stable_candidates) == pool_identity(args.residual_candidates)
    stable_primary = stable_table["n20_finalizer"]
    residual_primary = residual_table["n20_finalizer"]
    checks = {
        "exact_n20": int(manifest.get("candidate_budget", 0)) == 20,
        "target_hidden": manifest.get("evaluation_target_access") is False,
        "denovo_all_buckets": set(str(count) for count in range(2, 8)) <= set(denovo),
        "denovo_adapter_independent": manifest.get("common_llm_adapter_in_denovo_execution_graph") is False,
        "table1_all_tasks": all(
            stable_table[key]["tasks"] == residual_table[key]["tasks"] == 10
            for key in stable_table
        ),
        "immutable_table1_pool": same_pool,
        "table1_n1_strict_preserved": float(residual_table["n1_raw"]["acc_all_0_65"])
        + float(args.max_table1_drop)
        >= float(stable_table["n1_raw"]["acc_all_0_65"]),
        "table1_n1_relaxed_preserved": float(residual_table["n1_raw"]["acc_all_0_15"])
        + float(args.max_table1_drop)
        >= float(stable_table["n1_raw"]["acc_all_0_15"]),
        "table1_n5_strict_preserved": float(residual_table["n5_finalizer"]["acc_all_0_65"])
        + float(args.max_table1_drop)
        >= float(stable_table["n5_finalizer"]["acc_all_0_65"]),
        "table1_n5_relaxed_preserved": float(residual_table["n5_finalizer"]["acc_all_0_15"])
        + float(args.max_table1_drop)
        >= float(stable_table["n5_finalizer"]["acc_all_0_15"]),
        "table1_n20_support_ceiling_equal": all(
            math.isclose(
                float(residual_primary[key]),
                float(stable_primary[key]),
                abs_tol=1e-12,
            )
            for key in ("validity", "acc_all_0_65", "acc_all_0_15")
        ),
    }
    result = {
        "protocol": "paper_replay_smoke_gate_v1",
        "passed": all(checks.values()),
        "scope": "single_seed_smoke_not_paper_final",
        "candidate_budget": 20,
        "denovo_2p7p": denovo,
        "table1": {
            "stable": stable_table,
            "residual": residual_table,
            "gains_by_setting": {
                setting: {
                    key: float(residual_table[setting][key]) - float(stable_table[setting][key])
                    for key in ("validity", "acc_all_0_65", "acc_all_0_15")
                }
                for setting in stable_table
            },
            "ranking_signal": {
                "stable_llm_at_1_strict": stable_group["llm_at_1_strict"],
                "residual_llm_at_1_strict": residual_group["llm_at_1_strict"],
                "stable_any_strict_at_20": stable_group["any_strict_at_20"],
                "residual_any_strict_at_20": residual_group["any_strict_at_20"],
            },
        },
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
