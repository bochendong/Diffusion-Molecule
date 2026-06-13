#!/usr/bin/env python3
"""Export MolEdit edit + de novo 2p-7p + OOD rows into one UniVideo train/eval pack."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from export_denovo_2p7p_eval_jsonl import read_rows, sample_from_row, write_jsonl  # noqa: E402
from sketchmol_understanding_condition.unified_condition_dataset import (  # noqa: E402
    UnifiedConditionSample,
    read_jsonl,
    summarize_samples,
    write_jsonl as write_samples_jsonl,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--molecule-db-csv", required=True, type=Path)
    parser.add_argument("--moledit-train-split", required=True, type=Path)
    parser.add_argument("--moledit-eval-split", required=True, type=Path)
    parser.add_argument("--moledit-table1-tasks-only", action="store_true")
    parser.add_argument("--moledit-balanced-train-per-task", type=int, default=None)
    parser.add_argument("--moledit-balanced-eval-per-task", type=int, default=None)
    parser.add_argument("--moledit-train-limit", type=int, default=None)
    parser.add_argument("--moledit-eval-limit", type=int, default=None)
    parser.add_argument("--min-source-tanimoto", type=float, default=0.0)
    parser.add_argument("--denovo-eval-rows-per-property-count", type=int, default=1000)
    parser.add_argument("--denovo-train-rows-per-property-count", type=int, default=500)
    parser.add_argument("--ood-eval-rows-per-spec", type=int, default=100)
    parser.add_argument("--ood-train-rows-per-spec", type=int, default=200)
    parser.add_argument("--ood-spec-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dual_dir = args.output_dir / "dualmode_pack"
    moledit_dir = dual_dir / "moledit_only"
    denovo_dir = dual_dir / "denovo"
    ood_dir = dual_dir / "ood"
    for path in (moledit_dir, denovo_dir, ood_dir):
        path.mkdir(parents=True, exist_ok=True)

    export_moledit(args, moledit_dir)
    denovo_eval_csv, denovo_train_csv = export_denovo(args, denovo_dir)
    ood_eval_csv = export_ood(args, ood_dir)

    moledit_train = read_jsonl(moledit_dir / "univideo_edit_train.jsonl")
    moledit_eval = read_jsonl(moledit_dir / "univideo_edit_eval.jsonl")
    denovo_train = csv_to_samples(denovo_train_csv, split_override="train")
    denovo_eval = csv_to_samples(denovo_eval_csv, split_override="eval")
    ood_train = csv_to_samples(ood_dir / "denovo_ood_train_rows.csv", split_override="train")
    ood_eval = csv_to_samples(ood_eval_csv, split_override="eval")

    train = [*moledit_train, *denovo_train, *ood_train]
    eval_rows = [*moledit_eval, *denovo_eval, *ood_eval]
    if not train or not eval_rows:
        raise SystemExit("Dual-mode export produced empty train or eval JSONL.")

    train_jsonl = args.output_dir / "univideo_edit_train.jsonl"
    eval_jsonl = args.output_dir / "univideo_edit_eval.jsonl"
    write_samples_jsonl(train_jsonl, train)
    write_samples_jsonl(eval_jsonl, eval_rows)
    _copy_if_exists(moledit_dir / "baseline_variants.csv", args.output_dir / "baseline_variants.csv")

    summary = summarize_samples([*train, *eval_rows], train_rows=len(train), eval_rows=len(eval_rows))
    summary.update(
        {
            "output_dir": str(args.output_dir),
            "train_jsonl": str(train_jsonl),
            "eval_jsonl": str(eval_jsonl),
            "moledit_train_rows": len(moledit_train),
            "moledit_eval_rows": len(moledit_eval),
            "denovo_train_rows": len(denovo_train),
            "denovo_eval_rows": len(denovo_eval),
            "ood_train_rows": len(ood_train),
            "ood_eval_rows": len(ood_eval),
            "denovo_eval_rows_per_property_count": args.denovo_eval_rows_per_property_count,
            "denovo_train_rows_per_property_count": args.denovo_train_rows_per_property_count,
            "ood_eval_rows_per_spec": args.ood_eval_rows_per_spec,
            "ood_train_rows_per_spec": args.ood_train_rows_per_spec,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def export_moledit(args: argparse.Namespace, output_dir: Path) -> None:
    cmd = [
        str(args.python_bin),
        str(PROJECT_DIR / "scripts/export_univideo_edit_dataset.py"),
        "--moledit-train-split",
        str(args.moledit_train_split),
        "--moledit-eval-split",
        str(args.moledit_eval_split),
        "--output-dir",
        str(output_dir),
        "--variants",
        "full",
        "--min-source-tanimoto",
        str(args.min_source_tanimoto),
    ]
    if args.moledit_table1_tasks_only:
        cmd.append("--moledit-table1-tasks-only")
    if args.moledit_balanced_train_per_task:
        cmd.extend(["--moledit-balanced-train-per-task", str(args.moledit_balanced_train_per_task)])
    if args.moledit_balanced_eval_per_task:
        cmd.extend(["--moledit-balanced-eval-per-task", str(args.moledit_balanced_eval_per_task)])
    if args.moledit_train_limit:
        cmd.extend(["--moledit-train-limit", str(args.moledit_train_limit)])
    if args.moledit_eval_limit:
        cmd.extend(["--moledit-eval-limit", str(args.moledit_eval_limit)])
    subprocess.run(cmd, check=True)


def export_denovo(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    eval_csv = output_dir / "denovo_2p7p_rows.csv"
    train_csv = output_dir / "denovo_2p7p_train_rows.csv"
    cmd = [
        str(args.python_bin),
        str(PROJECT_DIR / "scripts/export_denovo_2p7p_benchmark_rows.py"),
        "--molecule-db-csv",
        str(args.molecule_db_csv),
        "--output-csv",
        str(eval_csv),
        "--rows-per-property-count",
        str(args.denovo_eval_rows_per_property_count),
        "--train-rows-per-property-count",
        str(args.denovo_train_rows_per_property_count),
        "--train-output-csv",
        str(train_csv),
    ]
    subprocess.run(cmd, check=True)
    return eval_csv, train_csv


def export_ood(args: argparse.Namespace, output_dir: Path) -> Path:
    eval_csv = output_dir / "denovo_ood_rows.csv"
    train_csv = output_dir / "denovo_ood_train_rows.csv"
    cmd = [
        str(args.python_bin),
        str(PROJECT_DIR / "scripts/export_denovo_ood_benchmark_rows.py"),
        "--molecule-db-csv",
        str(args.molecule_db_csv),
        "--output-csv",
        str(eval_csv),
        "--rows-per-spec",
        str(args.ood_eval_rows_per_spec),
        "--train-rows-per-spec",
        str(args.ood_train_rows_per_spec),
        "--train-output-csv",
        str(train_csv),
    ]
    if args.ood_spec_json is not None:
        cmd.extend(["--spec-json", str(args.ood_spec_json)])
    subprocess.run(cmd, check=True)
    return eval_csv


def csv_to_samples(path: Path, *, split_override: str) -> list[UnifiedConditionSample]:
    if not path.exists():
        return []
    return [sample_from_row(row, split_override=split_override) for row in read_rows(path)]


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
