#!/usr/bin/env python3
"""Collect validation-only metrics for instruction-aligned GraphEditDSL v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_umtp_v1_rl_pilot as paired  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--v1-checkpoint", required=True, type=Path)
    parser.add_argument("--v2-checkpoint", required=True, type=Path)
    parser.add_argument("--oracle-manifest", required=True, type=Path)
    parser.add_argument("--oracle-gate", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8,20")
    return parser.parse_args(argv)


def parse_budgets(raw: str) -> tuple[int, ...]:
    budgets = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if 1 not in budgets or 20 not in budgets:
        raise ValueError("Instruction pilot requires n=1 and n=20")
    return budgets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_task_metric(path: Path, task_key: str, metric: str) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if str(row.get("task_key", "")).strip() == task_key:
            return paired.parse_float(row.get(metric, ""))
    return math.nan


def collect(root: Path, budgets: Sequence[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for variant in ("v1", "v2"):
        for budget in budgets:
            for selection in ("raw", "finalizer"):
                path = paired.summary_path(root, variant, "table1", int(budget), selection)
                metrics = paired.table1_summary(path)
                records.append(
                    {
                        "variant": variant,
                        "task": "table1",
                        "budget": int(budget),
                        "selection": selection,
                        **metrics,
                        "gsk3b_acc_all_0_65": table_task_metric(path, "GSK3B:increase", "Acc_all(0.65)"),
                    }
                )
    for variant in ("base", "v2"):
        for budget in budgets:
            for selection in ("raw", "finalizer"):
                path = paired.summary_path(root, variant, "retention", int(budget), selection)
                records.append(
                    {
                        "variant": variant,
                        "task": "retention",
                        "budget": int(budget),
                        "selection": selection,
                        **paired.denovo_summary(path),
                    }
                )
    return records


def decision(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    index = {
        (str(row["variant"]), str(row["task"]), int(row["budget"]), str(row["selection"])): row
        for row in records
    }

    def value(variant: str, task: str, budget: int, selection: str, metric: str) -> float:
        return paired.parse_float(index[(variant, task, budget, selection)].get(metric, ""))

    raw_gain = value("v2", "table1", 1, "raw", "acc_all_0_65") - value(
        "v1", "table1", 1, "raw", "acc_all_0_65"
    )
    n20_gain = value("v2", "table1", 20, "finalizer", "acc_all_0_65") - value(
        "v1", "table1", 20, "finalizer", "acc_all_0_65"
    )
    gsk3b_n20 = value("v2", "table1", 20, "finalizer", "gsk3b_acc_all_0_65")
    retention_raw = value("v2", "retention", 1, "raw", "strict_success_rate") - value(
        "base", "retention", 1, "raw", "strict_success_rate"
    )
    retention_n20 = value("v2", "retention", 20, "finalizer", "strict_success_rate") - value(
        "base", "retention", 20, "finalizer", "strict_success_rate"
    )
    checks = {
        "raw_edit_gain": math.isfinite(raw_gain) and raw_gain >= 0.05,
        "n20_edit_non_regression": math.isfinite(n20_gain) and n20_gain >= 0.0,
        "gsk3b_recovery": math.isfinite(gsk3b_n20) and gsk3b_n20 >= 0.10,
        "raw_retention": math.isfinite(retention_raw) and retention_raw >= -0.01,
        "n20_retention": math.isfinite(retention_n20) and retention_n20 >= -0.01,
    }
    return {
        "decision": "go" if all(checks.values()) else "stop",
        "criteria": {
            "raw_edit_gain": "Table1 raw n=1 Acc@0.65 improves by at least 5pp",
            "n20_edit_non_regression": "Table1 finalizer n=20 Acc@0.65 does not decrease",
            "gsk3b_recovery": "GSK3B finalizer n=20 Acc@0.65 reaches at least 10%",
            "retention": "de novo raw and finalizer n=20 strict drops are no worse than 1pp",
        },
        "checks": checks,
        "observed": {
            "table1_raw_n1_delta": raw_gain,
            "table1_finalizer_n20_delta": n20_gain,
            "gsk3b_finalizer_n20": gsk3b_n20,
            "retention_raw_n1_delta": retention_raw,
            "retention_finalizer_n20_delta": retention_n20,
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, records: Sequence[Mapping[str, object]], result: Mapping[str, object]) -> None:
    lines = [
        "# Instruction-aligned GraphEditDSL v2 Pilot",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "Validation-only comparison; formal Table1 test rows are not used for training or selection.",
        "",
        "| Variant | Task | n | Selection | Validity | Strict/Acc@0.65 | GSK3B@0.65 | Acc@0.15 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        strict = row.get("acc_all_0_65", row.get("strict_success_rate", ""))
        lines.append(
            "| {variant} | {task} | {budget} | {selection} | {validity} | {strict} | {gsk} | {relaxed} |".format(
                variant=row["variant"],
                task=row["task"],
                budget=row["budget"],
                selection=row["selection"],
                validity=paired.format_value(row.get("validity", "")),
                strict=paired.format_value(strict),
                gsk=paired.format_value(row.get("gsk3b_acc_all_0_65", "")),
                relaxed=paired.format_value(row.get("acc_all_0_15", "")),
            )
        )
    lines.extend(["", "## Go/no-go", ""])
    for key, observed in dict(result["observed"]).items():
        lines.append(f"- {key}: {paired.format_value(observed)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(str(args.budgets))
    records = collect(args.pilot_root, budgets)
    pilot_decision = decision(records)
    oracle = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    oracle_gate = json.loads(args.oracle_gate.read_text(encoding="utf-8"))
    metrics_csv = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    summary_json = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    report_md = args.output_prefix.with_name(args.output_prefix.name + "_report.md")
    write_csv(metrics_csv, records)
    payload = {
        "protocol": "umtp_graph_action_instruction_distillation_v2_validation",
        "pilot_root": str(args.pilot_root),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "v1_checkpoint": str(args.v1_checkpoint),
        "v1_checkpoint_sha256": sha256(args.v1_checkpoint),
        "v2_checkpoint": str(args.v2_checkpoint),
        "v2_checkpoint_sha256": sha256(args.v2_checkpoint),
        "oracle": oracle,
        "oracle_gate": oracle_gate,
        "records": records,
        "decision": pilot_decision,
        "metrics_csv": str(metrics_csv),
        "report": str(report_md),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_md, records, pilot_decision)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
