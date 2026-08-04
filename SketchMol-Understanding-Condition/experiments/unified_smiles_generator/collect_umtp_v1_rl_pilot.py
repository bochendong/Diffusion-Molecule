#!/usr/bin/env python3
"""Collect paired raw/finalizer metrics for the UMTP v1 short RL pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


VARIANTS = ("baseline", "rl")
TASKS = ("table1", "retention")
BUDGETS = (1, 8)
SELECTIONS = ("raw", "finalizer")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--rl-checkpoint", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8")
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_float(value: object) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def numeric_mean(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = [parse_float(row.get(key, "")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else math.nan


def table1_summary(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    complete_groups = sum(str(row.get("status", "")).strip() == "measured" for row in rows)
    if len(rows) != 10 or complete_groups != 10:
        raise ValueError(
            f"Table1 pilot requires 10/10 measured tasks; found {complete_groups}/{len(rows)} in {path}"
        )
    return {
        "groups": len(rows),
        "complete_groups": complete_groups,
        "validity": numeric_mean(rows, "Validity"),
        "acc_all_0_65": numeric_mean(rows, "Acc_all(0.65)"),
        "acc_all_0_15": numeric_mean(rows, "Acc_all(0.15)"),
        "source_summary": str(path),
    }


def weighted_metric(rows: Sequence[Mapping[str, str]], key: str) -> float:
    all_rows = [row for row in rows if str(row.get("benchmark_label", "")).strip().lower() == "all"]
    if all_rows:
        return parse_float(all_rows[0].get(key, ""))
    pairs = []
    for row in rows:
        value = parse_float(row.get(key, ""))
        weight = parse_float(row.get("n", ""))
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return math.nan
    return sum(value * weight for value, weight in pairs) / sum(weight for _, weight in pairs)


def denovo_summary(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    return {
        "groups": len(rows),
        "validity": weighted_metric(rows, "validity"),
        "strict_success_rate": weighted_metric(rows, "strict_success_rate"),
        "source_summary": str(path),
    }


def summary_path(root: Path, variant: str, task: str, budget: int, selection: str) -> Path:
    task_root = root / "eval" / variant / task
    if task == "table1":
        return (
            task_root
            / "moledit_table1"
            / f"n{budget}"
            / selection
            / "metrics"
            / "moledit_table_summary.csv"
        )
    return task_root / "denovo_2p7p" / f"n{budget}" / selection / "benchmark_summary.csv"


def parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if 1 not in values:
        raise ValueError("The pilot requires candidate budget 1 for the raw policy decision.")
    return values


def collect(root: Path, *, budgets: Sequence[int] = BUDGETS) -> list[dict[str, object]]:
    records = []
    for variant in VARIANTS:
        for task in TASKS:
            for budget in budgets:
                for selection in SELECTIONS:
                    path = summary_path(root, variant, task, budget, selection)
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing pilot summary: {path}")
                    metrics = table1_summary(path) if task == "table1" else denovo_summary(path)
                    records.append(
                        {
                            "variant": variant,
                            "task": task,
                            "budget": budget,
                            "selection": selection,
                            **metrics,
                        }
                    )
    return records


def record_index(records: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int, str], Mapping[str, object]]:
    return {
        (
            str(row["variant"]),
            str(row["task"]),
            int(row["budget"]),
            str(row["selection"]),
        ): row
        for row in records
    }


def delta(index: Mapping[tuple[str, str, int, str], Mapping[str, object]], task: str, budget: int, selection: str, metric: str) -> float:
    before = parse_float(index[("baseline", task, budget, selection)].get(metric, ""))
    after = parse_float(index[("rl", task, budget, selection)].get(metric, ""))
    if not math.isfinite(before) or not math.isfinite(after):
        return math.nan
    return after - before


def pilot_decision(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    index = record_index(records)
    raw_strict_delta = delta(index, "table1", 1, "raw", "acc_all_0_65")
    raw_relaxed_delta = delta(index, "table1", 1, "raw", "acc_all_0_15")
    retention_delta = delta(index, "retention", 1, "raw", "strict_success_rate")
    edit_gain = (
        (math.isfinite(raw_strict_delta) and raw_strict_delta >= 0.02)
        or (math.isfinite(raw_relaxed_delta) and raw_relaxed_delta >= 0.05)
    )
    retention_ok = math.isfinite(retention_delta) and retention_delta >= -0.02
    return {
        "decision": "go" if edit_gain and retention_ok else "stop",
        "criteria": {
            "edit_gain": "raw n=1 Acc@0.65 +2pp or Acc@0.15 +5pp",
            "retention": "raw n=1 held-out de novo strict drop no worse than 2pp",
        },
        "observed": {
            "table1_raw_n1_acc_0_65_delta": raw_strict_delta,
            "table1_raw_n1_acc_0_15_delta": raw_relaxed_delta,
            "denovo_retention_raw_n1_strict_delta": retention_delta,
        },
        "edit_gain": edit_gain,
        "retention_ok": retention_ok,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object) -> str:
    parsed = parse_float(value)
    return f"{parsed:.4f}" if math.isfinite(parsed) else ""


def write_report(path: Path, records: Sequence[Mapping[str, object]], decision: Mapping[str, object]) -> None:
    lines = [
        "# UMTP v1 Short RL Pilot",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "| Variant | Task | n | Selection | Validity | Strict/Acc@0.65 | Acc@0.15 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in records:
        strict = row.get("acc_all_0_65", row.get("strict_success_rate", ""))
        lines.append(
            "| {variant} | {task} | {budget} | {selection} | {validity} | {strict} | {relaxed} |".format(
                variant=row["variant"],
                task=row["task"],
                budget=row["budget"],
                selection=row["selection"],
                validity=format_value(row.get("validity", "")),
                strict=format_value(strict),
                relaxed=format_value(row.get("acc_all_0_15", "")),
            )
        )
    observed = decision["observed"]
    lines.extend(
        [
            "",
            "## Paired raw n=1 deltas",
            "",
            f"- Table1 Acc@0.65: {format_value(observed['table1_raw_n1_acc_0_65_delta'])}",
            f"- Table1 Acc@0.15: {format_value(observed['table1_raw_n1_acc_0_15_delta'])}",
            f"- De novo retention strict: {format_value(observed['denovo_retention_raw_n1_strict_delta'])}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = collect(args.pilot_root, budgets=parse_budgets(str(args.budgets)))
    decision = pilot_decision(records)
    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    json_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_report.md")
    write_csv(csv_path, records)
    payload = {
        "protocol": "umtp_v1_short_rl_pilot",
        "pilot_root": str(args.pilot_root),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "rl_checkpoint": str(args.rl_checkpoint),
        "rl_checkpoint_sha256": sha256(args.rl_checkpoint),
        "records": records,
        "decision": decision,
        "metrics_csv": str(csv_path),
        "report": str(report_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, records, decision)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
