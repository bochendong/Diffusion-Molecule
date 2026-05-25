"""Create train/eval splits that control train-set nearest-neighbor shortcuts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .benchmark_audit import audit_split
from .chem import scaffold_key
from .dataset import load_examples_csv, write_examples_csv
from .schema import BenchmarkExample


def build_hard_split(
    examples: Iterable[BenchmarkExample],
    eval_fraction: float = 0.2,
    seed: int = 7,
    max_train_target_tanimoto: float = 0.55,
    scaffold_holdout: bool = True,
) -> tuple[list[BenchmarkExample], list[BenchmarkExample], dict[str, object]]:
    examples = list(examples)
    if len(examples) < 2:
        return examples, examples, {"warning": "split requires at least two examples"}
    rng = random.Random(seed)
    eval_target_size = max(1, int(round(len(examples) * min(0.9, max(0.05, eval_fraction)))))
    groups = _target_groups(examples, scaffold_holdout=scaffold_holdout)
    group_keys = list(groups)
    rng.shuffle(group_keys)

    selected_groups: set[str] = set()
    eval_candidates: list[BenchmarkExample] = []
    for key in group_keys:
        selected_groups.add(key)
        eval_candidates.extend(groups[key])
        if len(eval_candidates) >= eval_target_size:
            break

    train = [example for example in examples if _group_key(example, scaffold_holdout=scaffold_holdout) not in selected_groups]
    if not train:
        train = [example for example in examples if example not in eval_candidates]

    audited_rows, _ = audit_split(train, eval_candidates)
    keep_ids = {
        str(row["task_id"])
        for row in audited_rows
        if float(row["nearest_train_target_tanimoto"]) <= max_train_target_tanimoto and not bool(row["target_scaffold_in_train_targets"])
    }
    eval_examples = [example for example in eval_candidates if example.task_id in keep_ids]
    dropped = [example for example in eval_candidates if example.task_id not in keep_ids]
    if not eval_examples:
        eval_examples = eval_candidates[: max(1, min(len(eval_candidates), eval_target_size))]
        dropped = eval_candidates[len(eval_examples) :]
    train_ids = {example.task_id for example in train}
    train = [example for example in examples if example.task_id in train_ids]
    summary = _summary(train, eval_examples, dropped, seed, eval_fraction, max_train_target_tanimoto, scaffold_holdout)
    return train, eval_examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard SketchImage-JEPA train/eval splits from a task CSV.")
    parser.add_argument("--task-csv", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-target-tanimoto", type=float, default=0.55)
    parser.add_argument("--disable-scaffold-holdout", action="store_true")
    args = parser.parse_args()

    train, eval_examples, summary = build_hard_split(
        load_examples_csv(args.task_csv),
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_train_target_tanimoto=args.max_train_target_tanimoto,
        scaffold_holdout=not args.disable_scaffold_holdout,
    )
    write_examples_csv(args.train_out, train)
    write_examples_csv(args.eval_out, eval_examples)
    summary_out = Path(args.summary_out) if args.summary_out else Path(args.eval_out).with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _target_groups(examples: list[BenchmarkExample], scaffold_holdout: bool) -> dict[str, list[BenchmarkExample]]:
    groups: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        groups[_group_key(example, scaffold_holdout=scaffold_holdout)].append(example)
    return groups


def _group_key(example: BenchmarkExample, scaffold_holdout: bool) -> str:
    if scaffold_holdout:
        scaffold = scaffold_key(example.target_smiles)
        if scaffold:
            return f"scaffold:{scaffold}"
    return f"target:{example.target_smiles}"


def _summary(
    train: list[BenchmarkExample],
    eval_examples: list[BenchmarkExample],
    dropped: list[BenchmarkExample],
    seed: int,
    eval_fraction: float,
    max_train_target_tanimoto: float,
    scaffold_holdout: bool,
) -> dict[str, object]:
    audit_rows, audit_summary = audit_split(train, eval_examples)
    return {
        "train_tasks": len(train),
        "eval_tasks": len(eval_examples),
        "dropped_eval_candidates": len(dropped),
        "seed": seed,
        "eval_fraction": eval_fraction,
        "max_train_target_tanimoto": max_train_target_tanimoto,
        "scaffold_holdout": scaffold_holdout,
        "train_task_counts": dict(sorted(Counter(example.task_type.value for example in train).items())),
        "eval_task_counts": dict(sorted(Counter(example.task_type.value for example in eval_examples).items())),
        "audit_summary": audit_summary,
        "max_observed_nearest_train_target_tanimoto": max((float(row["nearest_train_target_tanimoto"]) for row in audit_rows), default=0.0),
    }


if __name__ == "__main__":
    main()
