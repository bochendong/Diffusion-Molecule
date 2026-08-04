#!/usr/bin/env python3
"""Collect the paired common-decoder GraphEditDSL pilot metrics."""

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


VARIANTS = ("baseline", "action")
TASKS = ("table1", "retention")
SELECTIONS = ("raw", "finalizer")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--oracle-manifest", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--budgets", default="1,8")
    return parser.parse_args(argv)


def parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",") if value.strip()))
    if 1 not in values:
        raise ValueError("Graph-action pilot requires n=1")
    return values


def collect(root: Path, budgets: Sequence[int]) -> list[dict[str, object]]:
    records = []
    for variant in VARIANTS:
        for task in TASKS:
            for budget in budgets:
                for selection in SELECTIONS:
                    path = paired.summary_path(root, variant, task, int(budget), selection)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    metrics = paired.table1_summary(path) if task == "table1" else paired.denovo_summary(path)
                    records.append(
                        {
                            "variant": variant,
                            "task": task,
                            "budget": int(budget),
                            "selection": selection,
                            **metrics,
                        }
                    )
    return records


def index_records(records: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int, str], Mapping[str, object]]:
    return {
        (str(row["variant"]), str(row["task"]), int(row["budget"]), str(row["selection"])): row
        for row in records
    }


def metric_delta(
    index: Mapping[tuple[str, str, int, str], Mapping[str, object]],
    task: str,
    metric: str,
) -> float:
    before = paired.parse_float(index[("baseline", task, 1, "raw")].get(metric, ""))
    after = paired.parse_float(index[("action", task, 1, "raw")].get(metric, ""))
    return after - before if math.isfinite(before) and math.isfinite(after) else math.nan


def decision(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    index = index_records(records)
    strict_delta = metric_delta(index, "table1", "acc_all_0_65")
    relaxed_delta = metric_delta(index, "table1", "acc_all_0_15")
    retention_delta = metric_delta(index, "retention", "strict_success_rate")
    edit_gain = (
        (math.isfinite(strict_delta) and strict_delta >= 0.02)
        or (math.isfinite(relaxed_delta) and relaxed_delta >= 0.05)
    )
    retention_ok = math.isfinite(retention_delta) and retention_delta >= -0.02
    return {
        "decision": "go" if edit_gain and retention_ok else "stop",
        "edit_gain": edit_gain,
        "retention_ok": retention_ok,
        "criteria": {
            "edit_gain": "raw n=1 Acc@0.65 +2pp or Acc@0.15 +5pp",
            "retention": "raw n=1 held-out de novo strict drop no worse than 2pp",
        },
        "observed": {
            "table1_raw_n1_acc_0_65_delta": strict_delta,
            "table1_raw_n1_acc_0_15_delta": relaxed_delta,
            "denovo_retention_raw_n1_strict_delta": retention_delta,
        },
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


def write_report(
    path: Path,
    records: Sequence[Mapping[str, object]],
    pilot_decision: Mapping[str, object],
    oracle: Mapping[str, object],
) -> None:
    lines = [
        "# UMTP Common-Decoder GraphEditDSL Pilot",
        "",
        f"Decision: **{pilot_decision['decision']}**",
        "",
        "The same decoder emits SMILES for de novo design and ranks executable GraphEditDSL programs for editing.",
        "",
        "## Action-space oracle",
        "",
        f"- Executable edit coverage: {paired.format_value(oracle.get('edit_action_coverage', ''))}",
        f"- Exact paired-target reconstruction: {paired.format_value(oracle.get('exact_reconstruction_rate', ''))}",
        f"- Mean best target similarity: {paired.format_value(oracle.get('mean_best_target_similarity', ''))}",
        f"- Best target similarity >= 0.65: {paired.format_value(oracle.get('best_target_similarity_at_0_65', ''))}",
        "",
        "## Paired benchmark",
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
                validity=paired.format_value(row.get("validity", "")),
                strict=paired.format_value(strict),
                relaxed=paired.format_value(row.get("acc_all_0_15", "")),
            )
        )
    observed = pilot_decision["observed"]
    lines.extend(
        [
            "",
            "## Paired raw n=1 deltas",
            "",
            f"- Table1 Acc@0.65: {paired.format_value(observed['table1_raw_n1_acc_0_65_delta'])}",
            f"- Table1 Acc@0.15: {paired.format_value(observed['table1_raw_n1_acc_0_15_delta'])}",
            f"- De novo retention strict: {paired.format_value(observed['denovo_retention_raw_n1_strict_delta'])}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budgets = parse_budgets(str(args.budgets))
    records = collect(args.pilot_root, budgets)
    pilot_decision = decision(records)
    oracle = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
    metrics_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_report.md")
    write_csv(metrics_path, records)
    payload = {
        "protocol": "umtp_common_decoder_graph_action_pilot",
        "pilot_root": str(args.pilot_root),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "action_checkpoint": str(args.action_checkpoint),
        "action_checkpoint_sha256": sha256(args.action_checkpoint),
        "oracle_manifest": str(args.oracle_manifest),
        "oracle": oracle,
        "records": records,
        "decision": pilot_decision,
        "metrics_csv": str(metrics_path),
        "report": str(report_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_path, records, pilot_decision, oracle)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
