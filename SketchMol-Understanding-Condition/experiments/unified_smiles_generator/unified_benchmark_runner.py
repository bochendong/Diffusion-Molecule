#!/usr/bin/env python3
"""Connect unified SMILES predictions to existing benchmark evaluators.

The runner is intentionally thin: it adapts unified selected/candidate CSVs to
the repo's existing evaluator contracts and records the generated report paths.
It does not redefine benchmark metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = PROJECT_DIR.parent
UNIFIED_3M_DIR = REPO_DIR / "SketchMol-Unified-3MDiffusion"

SUPPORTED_TASKS = {
    "denovo_2p7p",
    "denovo_ood",
    "external_multiproperty",
    "moledit_table1",
}
ID_COLUMNS = ("example_id", "condition_id", "sample_id", "pair_hash", "variant_id", "pair_id")
SMILES_COLUMNS = ("generated_smiles", "predicted_smiles", "candidate_smiles", "smiles")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-csv", type=Path, default=None)
    parser.add_argument("--candidate-prediction-csv", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tasks", default="denovo_2p7p,external_multiproperty,moledit_table1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--method-name", default="unified_smiles_generator")
    parser.add_argument("--accept-direct-smiles", action="store_true", default=True)
    parser.add_argument(
        "--candidate-budgets",
        default="",
        help="Optional shared candidate-prefix budgets, e.g. 1,20,128,256. Enables fair offline selection for all tasks.",
    )
    parser.add_argument(
        "--selection-modes",
        default="raw,finalizer",
        help="Offline prefix selection modes used with --candidate-budgets.",
    )

    parser.add_argument("--denovo-2p7p-family", default="unified_smiles_denovo_property_design")
    parser.add_argument("--denovo-2p7p-task", default="unified_smiles_denovo_2p7p_property_design")
    parser.add_argument("--denovo-ood-family", default="unified_smiles_denovo_ood_property_design")
    parser.add_argument("--denovo-ood-task", default="unified_smiles_denovo_ood_property_design")

    parser.add_argument("--external-generated-properties-csv", type=Path, default=None)
    parser.add_argument("--external-source-properties-csv", type=Path, default=None)
    parser.add_argument("--external-group-column", default="condition_id")
    parser.add_argument("--external-min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--external-report-title", default="Unified SMILES External Multi-property Benchmark")

    parser.add_argument("--moledit-reference-csv", type=Path, default=None)
    parser.add_argument("--moledit-budgets", default="20,256")
    parser.add_argument("--moledit-selection-mode", choices=("unified_score", "rank"), default="unified_score")
    parser.add_argument("--moledit-thresholds", default="0.65,0.15")
    parser.add_argument("--moledit-model-name", default="UnifiedSMILES")
    parser.add_argument("--moledit-missing-oracle-policy", choices=("fail", "skip-task", "mark-false"), default="fail")
    parser.add_argument("--moledit-require-table1-coverage", action="store_true")
    parser.add_argument(
        "--moledit-rdkit-modules",
        default=os.environ.get("SUCC_UNIFIED_MOLEDIT_RDKIT_MODULES", "gcc/12.3 rdkit/2024.09.6"),
        help="EasyBuild modules to load before running evaluate_moledit_table_metrics.py",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)
    records: list[dict[str, object]] = []
    shared_budgets = parse_int_list(str(args.candidate_budgets)) if str(args.candidate_budgets).strip() else []
    selection_modes = parse_selection_modes(str(args.selection_modes))

    if "denovo_2p7p" in tasks:
        if shared_budgets:
            require_file(args.candidate_prediction_csv, "--candidate-prediction-csv")
            records.extend(
                run_denovo_candidate_benchmarks(
                    args,
                    task_name="denovo_2p7p",
                    family=str(args.denovo_2p7p_family),
                    benchmark_task=str(args.denovo_2p7p_task),
                    title="Unified SMILES de novo 2p-7p Benchmark",
                    budgets=shared_budgets,
                    selection_modes=selection_modes,
                )
            )
        else:
            require_file(args.prediction_csv, "--prediction-csv")
            records.append(run_denovo_benchmark(
                args,
                task_name="denovo_2p7p",
                family=str(args.denovo_2p7p_family),
                benchmark_task=str(args.denovo_2p7p_task),
                title="Unified SMILES de novo 2p-7p Benchmark",
            ))
    if "denovo_ood" in tasks:
        if shared_budgets:
            require_file(args.candidate_prediction_csv, "--candidate-prediction-csv")
            records.extend(
                run_denovo_candidate_benchmarks(
                    args,
                    task_name="denovo_ood",
                    family=str(args.denovo_ood_family),
                    benchmark_task=str(args.denovo_ood_task),
                    title="Unified SMILES de novo OOD Benchmark",
                    budgets=shared_budgets,
                    selection_modes=selection_modes,
                )
            )
        else:
            require_file(args.prediction_csv, "--prediction-csv")
            records.append(run_denovo_benchmark(
                args,
                task_name="denovo_ood",
                family=str(args.denovo_ood_family),
                benchmark_task=str(args.denovo_ood_task),
                title="Unified SMILES de novo OOD Benchmark",
            ))
    if "external_multiproperty" in tasks:
        require_file(args.candidate_prediction_csv, "--candidate-prediction-csv")
        records.append(run_external_benchmark(args))
    if "moledit_table1" in tasks:
        require_file(args.candidate_prediction_csv, "--candidate-prediction-csv")
        require_file(args.moledit_reference_csv, "--moledit-reference-csv")
        records.extend(
            run_moledit_table1_benchmarks(
                args,
                budgets=shared_budgets or None,
                selection_modes=selection_modes if shared_budgets else None,
            )
        )

    summary = {
        "tasks": list(tasks),
        "output_dir": str(args.output_dir),
        "records": records,
    }
    (args.output_dir / "benchmark_suite_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_suite_report(args.output_dir / "benchmark_suite_report.md", records)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_denovo_benchmark(
    args: argparse.Namespace,
    *,
    task_name: str,
    family: str,
    benchmark_task: str,
    title: str,
) -> dict[str, object]:
    assert args.prediction_csv is not None
    output_dir = args.output_dir / task_name
    input_csv = filter_rows_to_csv(
        args.prediction_csv,
        output_dir / "unified_denovo_input.csv",
        lambda row: include_denovo_row(row, task_name=task_name),
        label=task_name,
    )
    cmd = [
        str(args.python_bin),
        str(PROJECT_DIR / "scripts" / "evaluate_univideo_image_benchmark.py"),
        "--image-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--method",
        str(args.method_name),
        "--smiles-column",
        "generated_smiles",
        "--report-title",
        title,
        "--benchmark-family",
        family,
        "--benchmark-task",
        benchmark_task,
        "--accept-direct-smiles",
        "--hide-source-similarity-section",
    ]
    run_cmd(cmd)
    return {
        "task": task_name,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "benchmark_summary.csv"),
        "report": str(output_dir / "benchmark_report.md"),
    }


def run_denovo_candidate_benchmarks(
    args: argparse.Namespace,
    *,
    task_name: str,
    family: str,
    benchmark_task: str,
    title: str,
    budgets: Sequence[int],
    selection_modes: Sequence[str],
) -> list[dict[str, object]]:
    assert args.candidate_prediction_csv is not None
    candidate_rows = [
        row for row in read_rows(args.candidate_prediction_csv) if include_denovo_row(row, task_name=task_name)
    ]
    references = unique_references_from_candidates(candidate_rows)
    records = []
    for budget in budgets:
        for selection_mode in selection_modes:
            output_dir = args.output_dir / task_name / f"n{budget}" / selection_mode
            selected_csv = output_dir / f"unified_{task_name}_selected_n{budget}_{selection_mode}.csv"
            selection_summary = select_candidate_prefixes(
                references=references,
                candidate_rows=candidate_rows,
                output_csv=selected_csv,
                budget=int(budget),
                method_name=f"{args.method_name}_{task_name}_n{budget}_{selection_mode}",
                selection_mode=selection_mode,
            )
            cmd = [
                str(args.python_bin),
                str(PROJECT_DIR / "scripts" / "evaluate_univideo_image_benchmark.py"),
                "--image-csv",
                str(selected_csv),
                "--output-dir",
                str(output_dir),
                "--method",
                f"{args.method_name}_{task_name}_n{budget}_{selection_mode}",
                "--smiles-column",
                "generated_smiles",
                "--report-title",
                f"{title} n={budget} {selection_mode}",
                "--benchmark-family",
                family,
                "--benchmark-task",
                benchmark_task,
                "--accept-direct-smiles",
                "--hide-source-similarity-section",
            ]
            run_cmd(cmd)
            records.append(
                {
                    "task": task_name,
                    "budget": int(budget),
                    "selection_mode": selection_mode,
                    "input_csv": str(args.candidate_prediction_csv),
                    "selected_csv": str(selected_csv),
                    "selection_summary": selection_summary,
                    "output_dir": str(output_dir),
                    "summary_csv": str(output_dir / "benchmark_summary.csv"),
                    "report": str(output_dir / "benchmark_report.md"),
                }
            )
    return records


def run_external_benchmark(args: argparse.Namespace) -> dict[str, object]:
    assert args.candidate_prediction_csv is not None
    output_dir = args.output_dir / "external_multiproperty"
    input_csv = filter_rows_to_csv(
        args.candidate_prediction_csv,
        output_dir / "unified_external_candidate_input.csv",
        include_external_row,
        label="external_multiproperty",
    )
    cmd = [
        str(args.python_bin),
        str(PROJECT_DIR / "scripts" / "evaluate_external_multiproperty_predictions.py"),
        "--prediction-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--smiles-column",
        "generated_smiles",
        "--source-smiles-column",
        "source_smiles",
        "--group-column",
        str(args.external_group_column),
        "--min-source-tanimoto",
        str(args.external_min_source_tanimoto),
        "--report-title",
        str(args.external_report_title),
    ]
    if args.external_generated_properties_csv:
        cmd.extend(["--generated-properties-csv", str(args.external_generated_properties_csv)])
    if args.external_source_properties_csv:
        cmd.extend(["--source-properties-csv", str(args.external_source_properties_csv)])
    run_cmd(cmd)
    return {
        "task": "external_multiproperty",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "external_multiproperty_summary.csv"),
        "report": str(output_dir / "external_multiproperty_report.md"),
    }


def run_moledit_table1_benchmarks(
    args: argparse.Namespace,
    *,
    budgets: Sequence[int] | None = None,
    selection_modes: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    assert args.candidate_prediction_csv is not None
    assert args.moledit_reference_csv is not None
    records = []
    shared_prefix_protocol = budgets is not None
    effective_budgets = list(budgets) if budgets is not None else parse_int_list(str(args.moledit_budgets))
    effective_modes = list(selection_modes) if selection_modes is not None else [str(args.moledit_selection_mode)]
    for budget in effective_budgets:
        for selection_mode in effective_modes:
            task_dir = args.output_dir / "moledit_table1" / f"n{budget}"
            if shared_prefix_protocol:
                task_dir = task_dir / selection_mode
            task_dir.mkdir(parents=True, exist_ok=True)
            method_suffix = f"_{selection_mode}" if shared_prefix_protocol else ""
            method_name = f"{args.method_name}_moledit_table1_n{budget}{method_suffix}"
            selected_csv = task_dir / f"unified_moledit_table1_selected_n{budget}{method_suffix}.csv"
            selection_summary = select_moledit_candidates(
                reference_csv=args.moledit_reference_csv,
                candidate_csv=args.candidate_prediction_csv,
                output_csv=selected_csv,
                budget=budget,
                method_name=method_name,
                selection_mode=selection_mode,
            )
            eval_dir = task_dir / "metrics"
            cmd = [
                str(args.python_bin),
                str(UNIFIED_3M_DIR / "scripts" / "evaluate_moledit_table_metrics.py"),
                "--reference",
                str(args.moledit_reference_csv),
                "--predictions",
                str(selected_csv),
                "--method",
                method_name,
                "--output-dir",
                str(eval_dir),
                "--model-name",
                str(args.moledit_model_name),
                "--thresholds",
                str(args.moledit_thresholds),
                "--task-filter",
                "table1",
                "--include-empty-table1",
                "--missing-oracle-policy",
                str(args.moledit_missing_oracle_policy),
            ]
            if bool(args.moledit_require_table1_coverage):
                cmd.append("--require-table1-coverage")
            run_cmd(cmd, rdkit_modules=str(args.moledit_rdkit_modules))
            records.append(
                {
                    "task": "moledit_table1",
                    "budget": budget,
                    "selection_mode": selection_mode,
                    "input_csv": str(args.candidate_prediction_csv),
                    "selected_csv": str(selected_csv),
                    "selection_summary": selection_summary,
                    "output_dir": str(eval_dir),
                    "summary_csv": str(eval_dir / "moledit_table_summary.csv"),
                    "report": str(eval_dir / "moledit_table_summary.md"),
                }
            )
    return records


def select_moledit_candidates(
    *,
    reference_csv: Path,
    candidate_csv: Path,
    output_csv: Path,
    budget: int,
    method_name: str,
    selection_mode: str,
) -> dict[str, object]:
    references = {normalized_row_id(row): row for row in read_rows(reference_csv) if normalized_row_id(row)}
    return select_candidate_prefixes(
        references=references,
        candidate_rows=read_rows(candidate_csv),
        output_csv=output_csv,
        budget=budget,
        method_name=method_name,
        selection_mode=selection_mode,
    )


def unique_references_from_candidates(rows: Sequence[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    references: dict[str, Mapping[str, str]] = {}
    for row in rows:
        row_id = normalized_row_id(row)
        if row_id and row_id not in references:
            references[row_id] = row
    return references


def select_candidate_prefixes(
    *,
    references: Mapping[str, Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    output_csv: Path,
    budget: int,
    method_name: str,
    selection_mode: str,
) -> dict[str, object]:
    candidates_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        row_id = normalized_row_id(row)
        if row_id:
            candidates_by_id[row_id].append(dict(row))
    for rows in candidates_by_id.values():
        rows.sort(key=candidate_sort_key)

    selected_rows = []
    missing = 0
    for ref_id, ref in references.items():
        pool = candidates_by_id.get(ref_id, [])[: max(1, int(budget))]
        if not pool:
            missing += 1
            continue
        selected = select_candidate_row(pool, selection_mode=selection_mode)
        out = dict(ref)
        generated = first_value(selected, SMILES_COLUMNS)
        out.update(
            {
                "generated_smiles": generated,
                "method": method_name,
                "unified_selected_candidate_rank": selected.get("candidate_rank", ""),
                "unified_selected_generation_rank": selected.get("generation_rank", ""),
                "unified_selected_candidate_score": selected.get("unified_finalizer_score", ""),
                "unified_candidate_limit": int(budget),
                "unified_selection_mode": selection_mode,
                "candidate_budget": int(budget),
                "selection_mode": normalized_selection_mode(selection_mode),
                "oracle_assisted": "True" if normalized_selection_mode(selection_mode) == "finalizer" else "False",
                "oracle_call_type": (
                    "rdkit_tdc_property_score"
                    if normalized_selection_mode(selection_mode) == "finalizer"
                    else "none"
                ),
                "candidate_pool_id": selected.get("candidate_pool_id", ""),
                "candidate_pool_hash": selected.get("candidate_pool_hash", ""),
            }
        )
        selected_rows.append(out)
    write_rows(output_csv, selected_rows)
    summary = {
        "reference_rows": len(references),
        "candidate_rows": len(candidate_rows),
        "selected_rows": len(selected_rows),
        "missing_candidate_rows": missing,
        "candidate_limit": int(budget),
        "selection_mode": selection_mode,
        "oracle_assisted": normalized_selection_mode(selection_mode) == "finalizer",
        "oracle_call_type": (
            "rdkit_tdc_property_score"
            if normalized_selection_mode(selection_mode) == "finalizer"
            else "none"
        ),
        "candidate_pool_ids": len(
            {
                str(row.get("candidate_pool_id", "") or "")
                for row in candidate_rows
                if str(row.get("candidate_pool_id", "") or "")
            }
        ),
        "output_csv": str(output_csv),
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def select_candidate_row(rows: Sequence[Mapping[str, str]], *, selection_mode: str) -> Mapping[str, str]:
    mode = normalized_selection_mode(selection_mode)
    if mode == "raw":
        return rows[0]
    return max(
        rows,
        key=lambda row: (
            parse_float(row.get("unified_finalizer_score"), default=-math.inf),
            parse_float(row.get("unified_property_success_fraction"), default=0.0),
            parse_float(row.get("source_tanimoto"), default=-1.0),
            -parse_float(row.get("candidate_rank"), default=1_000_000.0),
        ),
    )


def normalized_selection_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"raw", "rank"}:
        return "raw"
    if mode in {"finalizer", "unified_score"}:
        return "finalizer"
    raise ValueError(f"Unsupported selection mode: {value!r}")


def parse_selection_modes(raw: str) -> tuple[str, ...]:
    modes = []
    for item in str(raw or "").replace(";", ",").split(","):
        if item.strip():
            mode = normalized_selection_mode(item)
            if mode not in modes:
                modes.append(mode)
    return tuple(modes or ["raw", "finalizer"])


def parse_tasks(raw: str) -> tuple[str, ...]:
    out = []
    for item in str(raw or "").replace(";", ",").split(","):
        task = item.strip()
        if not task:
            continue
        if task == "all":
            return tuple(sorted(SUPPORTED_TASKS))
        if task not in SUPPORTED_TASKS:
            raise SystemExit(f"Unsupported benchmark task: {task}. Supported: {', '.join(sorted(SUPPORTED_TASKS))}")
        out.append(task)
    if not out:
        raise SystemExit("--tasks is empty")
    return tuple(dict.fromkeys(out))


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in str(raw or "").replace(";", ",").replace(" ", ",").split(","):
        if item.strip():
            values.append(int(item))
    return values or [20]


def filter_rows_to_csv(
    source_csv: Path,
    output_csv: Path,
    predicate,
    *,
    label: str,
) -> Path:
    rows = [row for row in read_rows(source_csv) if predicate(row)]
    if not rows:
        raise SystemExit(f"No rows matched benchmark filter `{label}` in {source_csv}")
    write_rows(output_csv, rows)
    return output_csv


def include_denovo_row(row: Mapping[str, object], *, task_name: str) -> bool:
    benchmark_task = str(row.get("benchmark_task", "") or "").strip().lower()
    if benchmark_task:
        if task_name == "denovo_2p7p":
            return "2p7p" in benchmark_task
        if task_name == "denovo_ood":
            return "ood" in benchmark_task
    return task_mode_for_row(row) == "de_novo"


def include_external_row(row: Mapping[str, object]) -> bool:
    benchmark_task = str(row.get("benchmark_task", "") or "").strip().lower()
    if benchmark_task:
        return benchmark_task.startswith("external_multiproperty")
    if str(row.get("external_task_properties", "") or "").strip():
        return True
    if str(row.get("external_property_directions_json", "") or "").strip():
        return True
    return task_mode_for_row(row) == "edit"


def task_mode_for_row(row: Mapping[str, object]) -> str:
    raw = str(row.get("task_mode", "") or row.get("unified_task_mode", "") or "").strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in {"de_novo", "denovo", "generate", "generation"}:
        return "de_novo"
    if normalized in {"edit", "conditional_edit", "source_edit", "edit_generation"}:
        return "edit"
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    return "edit" if source else "de_novo"


def require_file(path: Path | None, flag_name: str) -> None:
    if path is None or not path.is_file():
        raise SystemExit(f"{flag_name} is required and must exist: {path}")


def run_cmd(cmd: Sequence[str], *, rdkit_modules: str = "") -> None:
    modules = str(rdkit_modules).strip()
    if modules:
        quoted = " ".join(shlex.quote(str(part)) for part in cmd)
        shell_cmd = f"module load {modules} && {quoted}"
        print(f"+ {shell_cmd}", flush=True)
        subprocess.run(["bash", "-lc", shell_cmd], check=True)
        return
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    rows.append({str(key): "" if value is None else str(value) for key, value in payload.items()})
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = infer_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    preferred = [
        "example_id",
        "condition_id",
        "sample_id",
        "source_smiles",
        "target_smiles",
        "instruction",
        "instruction_tasks",
        "generated_smiles",
        "method",
    ]
    out = []
    seen = set()
    for key in preferred:
        if key not in seen:
            out.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                out.append(str(key))
                seen.add(str(key))
    return out


def normalized_row_id(row: Mapping[str, object]) -> str:
    text = first_value(row, ID_COLUMNS)
    if text.startswith("edit:"):
        text = text.split(":")[-1]
    return text


def first_value(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = str(row.get(column, "") or "").strip()
        if value:
            return value
    return ""


def candidate_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    rank = parse_float(
        row.get("generation_rank"),
        default=parse_float(row.get("candidate_rank"), default=1_000_000.0),
    )
    return int(rank), first_value(row, SMILES_COLUMNS)


def parse_float(value: object, *, default: float = math.nan) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def write_suite_report(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    lines = ["# Unified SMILES Benchmark Suite", ""]
    if not records:
        lines.append("No benchmark records were produced.")
    else:
        lines.extend(["| Task | Budget | Report | Summary |", "| --- | ---: | --- | --- |"])
        for record in records:
            task = record.get("task", "")
            budget = record.get("budget", "")
            report = record.get("report", "")
            summary = record.get("summary_csv", "")
            lines.append(f"| {task} | {budget} | `{report}` | `{summary}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
