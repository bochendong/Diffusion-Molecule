#!/usr/bin/env python3
"""Export a Table1-balanced MolEdit benchmark pack for extended table metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
UNIFIED_DIR = REPO_DIR / "SketchMol-Unified-3MDiffusion"
UNIFIED_SCRIPTS_DIR = UNIFIED_DIR / "scripts"
for path in (DATASET_DIR, UNIFIED_DIR, UNIFIED_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_moledit_benchmark_condition_rows import _moledit_to_condition_row  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_condition_dataset import (  # noqa: E402
    TABLE1_TASK_KEYS,
    _parse_instruction_tasks,
    _task_key,
    _task_specs_from_instruction,
    read_moledit_generation_samples,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moledit-train-split", required=True, type=Path)
    parser.add_argument("--moledit-eval-split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-first", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = _sample_table1_balanced(
        eval_path=args.moledit_eval_split,
        train_path=args.moledit_train_split,
        per_task=args.per_task,
        eval_first=args.eval_first,
    )
    selected = []
    for task_key in sorted(TABLE1_TASK_KEYS):
        for item in groups.get(task_key, []):
            selected.append(item)

    if not selected:
        raise SystemExit("No Table1 rows sampled; check MolEdit splits and --per-task.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    moledit_csv = args.output_dir / "table1_moledit_rows.csv"
    condition_csv = args.output_dir / "table1_benchmark_condition_rows.csv"
    eval_jsonl = args.output_dir / "table1_eval.jsonl"
    example_ids_path = args.output_dir / "table1_example_ids.txt"

    _write_moledit_rows(moledit_csv, selected)
    _write_condition_rows(condition_csv, selected)
    samples = read_moledit_generation_samples(
        moledit_csv,
        split="table1_benchmark",
        dataset_name="moledit_instruct",
    )
    write_jsonl(eval_jsonl, samples)
    example_ids_path.write_text(
        "\n".join(item["row"].get("example_id", "") for item in selected) + "\n",
        encoding="utf-8",
    )

    per_task_counts = {key: len(groups.get(key, [])) for key in sorted(TABLE1_TASK_KEYS)}
    missing_tasks = [key for key, count in per_task_counts.items() if count == 0]
    summary = {
        "moledit_train_split": str(args.moledit_train_split),
        "moledit_eval_split": str(args.moledit_eval_split),
        "output_dir": str(args.output_dir),
        "per_task": args.per_task,
        "eval_first": args.eval_first,
        "rows": len(selected),
        "tasks_with_rows": sum(1 for count in per_task_counts.values() if count > 0),
        "table1_task_count": len(TABLE1_TASK_KEYS),
        "per_task_counts": per_task_counts,
        "missing_tasks": missing_tasks,
        "moledit_rows_csv": str(moledit_csv),
        "condition_rows_csv": str(condition_csv),
        "eval_jsonl": str(eval_jsonl),
        "example_ids_txt": str(example_ids_path),
    }
    summary_path = args.output_dir / "table1_benchmark_pack.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _sample_table1_balanced(
    *,
    eval_path: Path,
    train_path: Path,
    per_task: int,
    eval_first: bool,
) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {key: [] for key in sorted(TABLE1_TASK_KEYS)}
    seen_ids: set[str] = set()
    split_order = ["eval", "train"] if eval_first else ["train", "eval"]
    sources = {"eval": eval_path, "train": train_path}

    for split in split_order:
        path = sources[split]
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                example_id = str(row.get("example_id", "")).strip()
                if not example_id or example_id in seen_ids:
                    continue
                task_key = _moledit_task_key_from_row(row)
                if task_key not in TABLE1_TASK_KEYS:
                    continue
                if len(groups[task_key]) >= per_task:
                    continue
                groups[task_key].append({"split": split, "row": row, "task_key": task_key})
                seen_ids.add(example_id)
    return groups


def _moledit_task_key_from_row(row: dict[str, str]) -> str:
    instruction_tasks = _parse_instruction_tasks(row.get("instruction_tasks", ""))
    task_specs = _task_specs_from_instruction(row, instruction_tasks)
    return _task_key(task_specs)


def _write_moledit_rows(path: Path, selected: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for item in selected:
        row = item["row"]
        assert isinstance(row, dict)
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            writer.writerow(item["row"])


def _write_condition_rows(path: Path, selected: list[dict[str, object]]) -> None:
    rows = []
    for item in selected:
        raw = item["row"]
        assert isinstance(raw, dict)
        rows.append(_moledit_to_condition_row(raw, split=str(item["split"])))
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
