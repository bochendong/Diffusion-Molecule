#!/usr/bin/env python3
"""Collect Joint v2 runs, aggregates, paired U1/U2 deltas, and gate status."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence


RUN_FIELDS = (
    "stage",
    "train_seed",
    "eval_seed",
    "benchmark",
    "budget",
    "selection",
    "group",
    "metric",
    "value",
    "gate_status",
    "selected_epoch",
    "input_modality",
    "checkpoint_sha256",
    "eval_csv_sha256",
    "candidate_csv_sha256",
    "max_candidates",
    "source_summary",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    # Legacy compatibility for the original v2 wrapper/tests.
    parser.add_argument("--stage")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def metadata(path: Path, stage_root: Path) -> dict[str, str]:
    """Parse both legacy task/atN/mode and v2 task/name/nN/mode layouts."""
    parts = path.relative_to(stage_root).parts
    task = parts[0] if parts else ""
    budget = ""
    selection = ""
    for index, part in enumerate(parts):
        match = re.fullmatch(r"(?:at|n)(\d+)", part)
        if match:
            budget = match.group(1)
            if index + 1 < len(parts):
                selection = parts[index + 1]
            break
    return {"task": task, "budget": budget, "selection": selection, "source_summary": str(path)}


def path_run_metadata(path: Path, eval_root: Path) -> dict[str, str]:
    parts = path.relative_to(eval_root).parts
    if len(parts) < 4:
        return {}
    stage, train_part, eval_part, benchmark = parts[:4]
    if not train_part.startswith("train_seed_") or not eval_part.startswith("eval_seed_"):
        return {}
    tail_root = eval_root / stage / train_part / eval_part
    local = metadata(path, tail_root)
    return {
        "stage": stage,
        "train_seed": train_part.removeprefix("train_seed_"),
        "eval_seed": eval_part.removeprefix("eval_seed_"),
        "benchmark": benchmark,
        "budget": local["budget"],
        "selection": local["selection"],
    }


def reproducibility_metadata(path: Path, eval_root: Path) -> dict[str, object]:
    parts = path.relative_to(eval_root).parts
    if len(parts) < 4:
        return {}
    metadata_path = eval_root.joinpath(*parts[:4], "joint_v2_run_metadata.json")
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "input_modality": payload.get("input_modality", ""),
        "checkpoint_sha256": payload.get("checkpoint_sha256", ""),
        "eval_csv_sha256": payload.get("eval_csv_sha256", ""),
        "candidate_csv_sha256": payload.get("candidate_csv_sha256", ""),
        "max_candidates": payload.get("max_candidates", ""),
    }


def numeric_items(row: Mapping[str, str], ignored: Iterable[str]) -> Iterable[tuple[str, float]]:
    ignored_set = set(ignored)
    for key, raw in row.items():
        if key in ignored_set or raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            yield key, value


def load_gate_status(selection_root: Path | None) -> dict[tuple[str, str], tuple[str, str]]:
    result: dict[tuple[str, str], tuple[str, str]] = {("u0", "base"): ("baseline", "")}
    if selection_root is None:
        return result
    patterns = {
        "u1": "u1_joint_sft/seed_*/checkpoint_selection.json",
        "u2": "u2_joint_protected_sft/seed_*/checkpoint_selection.json",
    }
    for stage, pattern in patterns.items():
        for path in selection_root.glob(pattern):
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed = path.parent.name.removeprefix("seed_")
            result[(stage, seed)] = (
                str(payload.get("status", "unknown")),
                str(payload.get("selected_epoch", "") or ""),
            )
    return result


def summary_rows(
    path: Path,
    *,
    eval_root: Path,
    gates: Mapping[tuple[str, str], tuple[str, str]],
) -> list[dict[str, object]]:
    meta = path_run_metadata(path, eval_root)
    if not meta:
        return []
    gate, epoch = gates.get((meta["stage"], meta["train_seed"]), ("missing", ""))
    reproducibility = reproducibility_metadata(path, eval_root)
    output = []
    is_table1 = path.name == "moledit_table_summary.csv"
    for row in read_rows(path):
        if is_table1:
            group = str(row.get("task_key", row.get("task", "")))
            ignored = {"model", "task", "task_key", "status"}
        else:
            group = str(row.get("property_count", row.get("benchmark_label", "")))
            ignored = {"method", "benchmark_label", "property_count"}
        for metric, value in numeric_items(row, ignored):
            output.append(
                {
                    **meta,
                    **reproducibility,
                    "group": group,
                    "metric": metric,
                    "value": value,
                    "gate_status": gate,
                    "selected_epoch": epoch,
                    "source_summary": str(path),
                }
            )
    return output


def ood_spec_rows(
    path: Path,
    *,
    eval_root: Path,
    gates: Mapping[tuple[str, str], tuple[str, str]],
) -> list[dict[str, object]]:
    meta = path_run_metadata(path, eval_root)
    if not meta or meta.get("benchmark") != "ood":
        return []
    gate, epoch = gates.get((meta["stage"], meta["train_seed"]), ("missing", ""))
    reproducibility = reproducibility_metadata(path, eval_root)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(path):
        spec = str(row.get("ood_spec_id", "")).strip()
        if spec:
            grouped[spec].append(row)
    result = []
    for spec, rows in sorted(grouped.items()):
        for metric, field in (("strict_success_rate", "strict_success"), ("validity", "valid")):
            values = [str(row.get(field, "")).strip().lower() in {"1", "true", "yes"} for row in rows]
            result.append(
                {
                    **meta,
                    **reproducibility,
                    "group": f"ood_spec:{spec}",
                    "metric": metric,
                    "value": sum(values) / len(values),
                    "gate_status": gate,
                    "selected_epoch": epoch,
                    "source_summary": str(path),
                }
            )
    return result


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    keys = ("stage", "benchmark", "budget", "selection", "group", "metric", "gate_status")
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(float(row["value"]))
    output = []
    for key, values in sorted(grouped.items()):
        output.append(
            {
                **dict(zip(keys, key)),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return output


def paired_delta_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    identity = ("eval_seed", "benchmark", "budget", "selection", "group", "metric")
    baseline = {
        tuple(str(row.get(key, "")) for key in identity): float(row["value"])
        for row in rows
        if row.get("stage") == "u0"
    }
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    keys = ("stage", "benchmark", "budget", "selection", "group", "metric", "gate_status")
    for row in rows:
        if row.get("stage") not in {"u1", "u2"}:
            continue
        match = tuple(str(row.get(key, "")) for key in identity)
        if match not in baseline:
            continue
        aggregate_key = tuple(str(row.get(key, "")) for key in keys)
        grouped[aggregate_key].append(float(row["value"]) - baseline[match])
    return [
        {
            **dict(zip(keys, key)),
            "mean_paired_delta_vs_u0": mean(values),
            "std_paired_delta_vs_u0": stdev(values) if len(values) > 1 else 0.0,
            "n_pairs": len(values),
        }
        for key, values in sorted(grouped.items())
    ]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]], preferred: Sequence[str] = ()) -> None:
    fields: list[str] = []
    seen = set()
    for key in preferred:
        if key not in seen:
            fields.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_prefix is None and args.output_csv is None:
        raise SystemExit("Set --output-prefix (v2) or --output-csv (legacy)")
    if args.stage and args.output_csv:
        stage_root = args.eval_root / args.stage
        summaries = sorted(stage_root.glob("**/benchmark_summary.csv")) + sorted(
            stage_root.glob("**/moledit_table_summary.csv")
        )
        legacy = []
        for path in summaries:
            meta = metadata(path, stage_root)
            legacy.extend({"stage": args.stage, **meta, **row} for row in read_rows(path))
        if not legacy:
            raise SystemExit(f"No benchmark summaries found under {stage_root}")
        write_rows(args.output_csv, legacy, ("stage", "task", "budget", "selection", "source_summary"))
        return 0

    gates = load_gate_status(args.selection_root)
    summaries = sorted(args.eval_root.glob("**/benchmark_summary.csv")) + sorted(
        args.eval_root.glob("**/moledit_table_summary.csv")
    )
    runs = []
    for path in summaries:
        runs.extend(summary_rows(path, eval_root=args.eval_root, gates=gates))
    for path in sorted(args.eval_root.glob("**/benchmark_decoded.csv")):
        runs.extend(ood_spec_rows(path, eval_root=args.eval_root, gates=gates))
    if not runs:
        raise SystemExit(f"No Joint v2 benchmark summaries found under {args.eval_root}")
    prefix = args.output_prefix
    assert prefix is not None
    runs_path = prefix.with_name(prefix.name + "_runs.csv")
    aggregate_path = prefix.with_name(prefix.name + "_aggregate.csv")
    paired_path = prefix.with_name(prefix.name + "_paired_deltas.csv")
    write_rows(runs_path, runs, RUN_FIELDS)
    aggregates = aggregate_rows(runs)
    paired = paired_delta_rows(runs)
    write_rows(aggregate_path, aggregates)
    write_rows(paired_path, paired)
    report = {
        "protocol": "unified_joint_fair_v2",
        "scope": ["u0", "u1", "u2"],
        "primary_table_selection": "raw",
        "auxiliary_table_selection": "finalizer",
        "reference_only_methods": ["SketchMol", "Direct", "UniVideo", "legacy Phase1/Fair v1"],
        "summary_files": len(summaries),
        "run_metadata_files": len(list(args.eval_root.glob("**/joint_v2_run_metadata.json"))),
        "run_metric_rows": len(runs),
        "aggregate_rows": len(aggregates),
        "paired_delta_rows": len(paired),
        "outputs": {
            "runs": str(runs_path),
            "aggregate": str(aggregate_path),
            "paired_deltas": str(paired_path),
        },
    }
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
