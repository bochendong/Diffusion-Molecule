#!/usr/bin/env python3
"""Export a Table1-balanced MolEdit benchmark pack for extended table metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_DIR / "SketchMol-MultiProperty-EditDataset"
UNIFIED_DIR = REPO_DIR / "SketchMol-Unified-3MDiffusion"
UNIFIED_SCRIPTS_DIR = UNIFIED_DIR / "scripts"
SUCC_DIR = REPO_DIR / "SketchMol-Understanding-Condition"
for path in (DATASET_DIR, UNIFIED_DIR, UNIFIED_SCRIPTS_DIR, SUCC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_moledit_benchmark_condition_rows import _moledit_to_condition_row  # noqa: E402
from sketchmol_understanding_condition.chem import molecular_properties, morgan_tanimoto  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_condition_dataset import (  # noqa: E402
    TABLE1_TASK_SPECS,
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
    parser.add_argument(
        "--synthesize-missing-tasks",
        action="store_true",
        help="Fill missing RDKit-only Table1 tasks by pairing sources with satisfying target molecules from the split pool.",
    )
    parser.add_argument("--synthetic-min-source-tanimoto", type=float, default=0.4)
    parser.add_argument("--synthetic-candidate-limit", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = _sample_table1_balanced(
        eval_path=args.moledit_eval_split,
        train_path=args.moledit_train_split,
        per_task=args.per_task,
        eval_first=args.eval_first,
    )
    synthesized_counts: dict[str, int] = {}
    if args.synthesize_missing_tasks:
        synthesized_counts = _synthesize_missing_table1_rows(
            groups,
            eval_path=args.moledit_eval_split,
            train_path=args.moledit_train_split,
            per_task=args.per_task,
            min_source_tanimoto=args.synthetic_min_source_tanimoto,
            candidate_limit=args.synthetic_candidate_limit,
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
        "synthesize_missing_tasks": bool(args.synthesize_missing_tasks),
        "synthetic_min_source_tanimoto": args.synthetic_min_source_tanimoto,
        "synthetic_candidate_limit": args.synthetic_candidate_limit,
        "synthesized_counts": synthesized_counts,
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


def _synthesize_missing_table1_rows(
    groups: dict[str, list[dict[str, object]]],
    *,
    eval_path: Path,
    train_path: Path,
    per_task: int,
    min_source_tanimoto: float,
    candidate_limit: int,
    eval_first: bool,
) -> dict[str, int]:
    rows = _load_rows_for_synthesis(eval_path=eval_path, train_path=train_path, eval_first=eval_first)
    candidate_rows = rows[: max(1, int(candidate_limit))]
    candidate_targets = _candidate_targets(candidate_rows)
    synthesized_counts: dict[str, int] = {}
    for task_key in sorted(TABLE1_TASK_KEYS):
        if len(groups.get(task_key, [])) >= per_task:
            continue
        task_specs = _specs_from_table1_key(task_key)
        if not task_specs or not _rdkit_only_task(task_specs):
            continue
        needed = per_task - len(groups.get(task_key, []))
        made = []
        used_pairs: set[tuple[str, str]] = set()
        for source_item in rows:
            if len(made) >= needed:
                break
            source_row = source_item["row"]
            assert isinstance(source_row, dict)
            source_smiles = str(source_row.get("source_smiles", "") or "").strip()
            if not source_smiles:
                continue
            source_props = _props_for_smiles(source_smiles)
            if not source_props:
                continue
            best_target = _best_synthetic_target(
                source_smiles=source_smiles,
                source_props=source_props,
                task_specs=task_specs,
                candidates=candidate_targets,
                min_source_tanimoto=min_source_tanimoto,
                used_pairs=used_pairs,
            )
            if best_target is None:
                continue
            target_smiles, target_props, source_tanimoto = best_target
            synthetic_row = _make_synthetic_row(
                source_row,
                task_key=task_key,
                task_specs=task_specs,
                target_smiles=target_smiles,
                source_props=source_props,
                target_props=target_props,
                source_tanimoto=source_tanimoto,
                index=len(made),
            )
            made.append({"split": source_item["split"], "row": synthetic_row, "task_key": task_key})
            used_pairs.add((source_smiles, target_smiles))
        if made:
            groups.setdefault(task_key, []).extend(made)
            synthesized_counts[task_key] = len(made)
    return synthesized_counts


def _load_rows_for_synthesis(*, eval_path: Path, train_path: Path, eval_first: bool) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    split_order = ["eval", "train"] if eval_first else ["train", "eval"]
    sources = {"eval": eval_path, "train": train_path}
    for split in split_order:
        with sources[split].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("source_smiles") and row.get("target_smiles"):
                    out.append({"split": split, "row": row})
    return out


def _candidate_targets(rows: list[dict[str, object]]) -> list[tuple[str, dict[str, float]]]:
    out = []
    seen: set[str] = set()
    for item in rows:
        row = item["row"]
        assert isinstance(row, dict)
        smiles = str(row.get("target_smiles", "") or "").strip()
        if not smiles or smiles in seen:
            continue
        props = _props_for_smiles(smiles)
        if not props:
            continue
        out.append((smiles, props))
        seen.add(smiles)
    return out


def _specs_from_table1_key(task_key: str) -> list[dict[str, str]]:
    for spec_set, key in TABLE1_TASK_SPECS.items():
        if key != task_key:
            continue
        return [{"property": prop, "direction": direction} for prop, direction in sorted(spec_set)]
    return []


def _rdkit_only_task(task_specs: list[dict[str, str]]) -> bool:
    return all(spec.get("property") in {"MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA"} for spec in task_specs)


def _best_synthetic_target(
    *,
    source_smiles: str,
    source_props: dict[str, float],
    task_specs: list[dict[str, str]],
    candidates: list[tuple[str, dict[str, float]]],
    min_source_tanimoto: float,
    used_pairs: set[tuple[str, str]],
) -> tuple[str, dict[str, float], float] | None:
    best: tuple[float, str, dict[str, float], float] | None = None
    for target_smiles, target_props in candidates:
        if target_smiles == source_smiles or (source_smiles, target_smiles) in used_pairs:
            continue
        if not _satisfies_task(source_props, target_props, task_specs):
            continue
        tani = morgan_tanimoto(source_smiles, target_smiles)
        if tani is None or float(tani) < min_source_tanimoto:
            continue
        score = float(tani) + 0.01 * _task_margin(source_props, target_props, task_specs)
        if best is None or score > best[0]:
            best = (score, target_smiles, target_props, float(tani))
    if best is None:
        return None
    _, target_smiles, target_props, tani = best
    return target_smiles, target_props, tani


def _make_synthetic_row(
    source_row: dict[str, str],
    *,
    task_key: str,
    task_specs: list[dict[str, str]],
    target_smiles: str,
    source_props: dict[str, float],
    target_props: dict[str, float],
    source_tanimoto: float,
    index: int,
) -> dict[str, str]:
    row = dict(source_row)
    base_id = row.get("example_id") or row.get("pair_hash") or f"row{index}"
    row["example_id"] = f"synthetic-table1:{task_key}:{base_id}:{index:04d}"
    row["pair_hash"] = row["example_id"]
    row["target_smiles"] = target_smiles
    row["target_canonical_smiles"] = target_smiles
    row["source_target_tanimoto"] = f"{source_tanimoto:.8g}"
    row["difficulty_bucket"] = _similarity_bucket(source_tanimoto)
    row["pair_quality"] = "synthetic_table1_source_target"
    row["instruction"] = _instruction_for_task(task_specs)
    row["instruction_tasks"] = json.dumps(task_specs, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    row["instruction_task_properties"] = "|".join(spec["property"] for spec in task_specs)
    row["instruction_task_directions"] = json.dumps(
        {spec["property"]: spec["direction"] for spec in task_specs},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    row["computed_active_properties"] = "|".join(spec["property"] for spec in task_specs)
    row["computed_active_count"] = str(len(task_specs))
    active_props = {spec["property"] for spec in task_specs}
    directions = {spec["property"]: spec["direction"] for spec in task_specs}
    for prop in ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA"):
        if prop in source_props:
            row[f"source_{prop}"] = _format_float(source_props[prop])
        if prop in target_props:
            row[f"target_{prop}"] = _format_float(target_props[prop])
        if prop in source_props and prop in target_props:
            row[f"delta_{prop}"] = _format_float(target_props[prop] - source_props[prop])
        row[f"{prop}_active"] = "1" if prop in active_props else "0"
        row[f"{prop}_direction"] = directions.get(prop, "")
    return row


def _props_for_smiles(smiles: str) -> dict[str, float]:
    props = molecular_properties(smiles) or {}
    out = {
        "MW": props.get("MolWt", math.nan),
        "LogP": props.get("LogP", math.nan),
        "QED": props.get("QED", math.nan),
        "TPSA": props.get("TPSA", math.nan),
        "HBD": props.get("HBD", math.nan),
        "HBA": props.get("HBA", math.nan),
        "RB": props.get("rotatable", math.nan),
        "SA": props.get("SA", math.nan),
    }
    return {key: float(value) for key, value in out.items() if value is not None and not math.isnan(float(value))}


def _satisfies_task(
    source_props: dict[str, float],
    target_props: dict[str, float],
    task_specs: list[dict[str, str]],
) -> bool:
    for spec in task_specs:
        prop = spec["property"]
        direction = spec["direction"]
        if prop not in source_props or prop not in target_props:
            return False
        if direction == "increase" and not (target_props[prop] > source_props[prop]):
            return False
        if direction == "decrease" and not (target_props[prop] < source_props[prop]):
            return False
    return True


def _task_margin(
    source_props: dict[str, float],
    target_props: dict[str, float],
    task_specs: list[dict[str, str]],
) -> float:
    margin = 0.0
    for spec in task_specs:
        prop = spec["property"]
        sign = 1.0 if spec["direction"] == "increase" else -1.0
        margin += sign * (target_props[prop] - source_props[prop])
    return margin


def _instruction_for_task(task_specs: list[dict[str, str]]) -> str:
    parts = []
    for spec in task_specs:
        verb = "increase" if spec["direction"] == "increase" else "decrease"
        parts.append(f"{verb} {spec['property']}")
    return "Edit the source molecule to " + ", ".join(parts) + " while preserving molecular similarity."


def _similarity_bucket(value: float) -> str:
    if value >= 0.7:
        return "easy_high_similarity"
    if value >= 0.5:
        return "medium_similarity"
    if value >= 0.4:
        return "hard_similarity"
    return "exploratory_low_similarity"


def _format_float(value: float) -> str:
    return "" if value is None or math.isnan(float(value)) else f"{float(value):.8g}"


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
