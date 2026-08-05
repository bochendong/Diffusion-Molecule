#!/usr/bin/env python3
"""Sample SketchMol + MolScribe OCR for one shard of de novo 2p-7p eval rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
IMAGE_PATH_FROM_LOG = re.compile(r"^path save to (.+/image_path\.csv)\s*$", re.MULTILINE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--start-index",
        type=int,
        default=-1,
        help="Optional absolute row offset (overrides shard math).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max rows for this shard (0 = all assigned rows).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sketchmol-repo", type=Path, required=True)
    parser.add_argument("--sketchmol-python", required=True)
    parser.add_argument("--molscribe-python", required=True)
    parser.add_argument("--molscribe-script", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--molscribe-model", type=Path, required=True)
    parser.add_argument("--conditional-count", type=int, default=40)
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=0,
        help=(
            "SketchMol diffusion micro-batch size. 0 uses --conditional-count "
            "as one batch; otherwise the total candidate count must be divisible "
            "by this value."
        ),
    )
    parser.add_argument("--custom-steps", type=int, default=250)
    parser.add_argument("--scale", type=float, default=1.2)
    parser.add_argument("--scale-pro", type=float, default=6.3)
    parser.add_argument("--molscribe-batch-size", type=int, default=8)
    parser.add_argument("--molscribe-backend", default="custom")
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def shard_rows(rows: Sequence[Mapping[str, str]], shard_index: int, shard_count: int) -> list[dict[str, str]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    return [dict(row) for idx, row in enumerate(rows) if idx % shard_count == shard_index]


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return token or "condition"


def preset_for_sketchmol(row: Mapping[str, str]) -> str:
    preset = str(row.get("sketchmol_preset_str") or "").strip()
    if preset:
        return preset.replace(",", " ")
    raise ValueError(
        "missing sketchmol_preset_str for "
        f"{row.get('condition_id') or row.get('sample_id')}"
    )


def resolve_image_csv(sample_stdout: str, log_dir: Path, sketchmol_cwd: Path) -> Path:
    match = IMAGE_PATH_FROM_LOG.search(sample_stdout)
    if match:
        raw = match.group(1).strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (sketchmol_cwd / candidate).resolve()
        if candidate.is_file():
            return candidate

    resolved_log_dir = log_dir.resolve()
    if resolved_log_dir.is_dir():
        matches = sorted(resolved_log_dir.rglob("image_path.csv"))
        if matches:
            return matches[-1]

    legacy_root = sketchmol_cwd / "SketchMolBenchmark"
    if legacy_root.is_dir():
        matches = sorted(legacy_root.rglob("image_path.csv"))
        if matches:
            return matches[-1]

    raise FileNotFoundError(f"image_path.csv not found under {resolved_log_dir}")


def resolve_image_path(raw: str, sketchmol_cwd: Path, log_dir: Path) -> str:
    path = Path(str(raw or "").strip())
    if path.is_absolute() and path.is_file():
        return str(path)
    for base in (log_dir, log_dir.parent, sketchmol_cwd, REPO_ROOT):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return str(candidate)
    return str((sketchmol_cwd / path).resolve())


def normalize_image_csv_paths(image_csv: Path, sketchmol_cwd: Path, log_dir: Path) -> Path:
    rows = read_rows(image_csv)
    if not rows or "image_path" not in rows[0]:
        return image_csv

    normalized_rows: list[dict[str, str]] = []
    changed = False
    for row in rows:
        fixed = dict(row)
        raw = str(row.get("image_path") or "")
        resolved = resolve_image_path(raw, sketchmol_cwd, log_dir)
        if resolved != raw:
            changed = True
        fixed["image_path"] = resolved
        normalized_rows.append(fixed)

    if not changed:
        return image_csv

    out_csv = image_csv.with_name("image_path_abs.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(normalized_rows)
    return out_csv


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
) -> str:
    print("RUN " + " ".join(cmd), flush=True)
    result = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    return result.stdout or ""


def append_candidate_rows(
    out_csv: Path,
    eval_row: Mapping[str, str],
    image_csv: Path,
    fieldnames: list[str],
    shard_index: int,
) -> None:
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        with image_csv.open("r", encoding="utf-8", newline="") as image_handle:
            for candidate_index, image_row in enumerate(csv.DictReader(image_handle)):
                merged = dict(eval_row)
                merged.update(image_row)
                merged["candidate_index"] = str(candidate_index)
                merged["shard_index"] = str(shard_index)
                writer.writerow(merged)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    args.eval_csv = resolve_repo_path(args.eval_csv)
    args.output_dir = resolve_repo_path(args.output_dir)
    args.sketchmol_repo = resolve_repo_path(args.sketchmol_repo)
    args.molscribe_script = resolve_repo_path(args.molscribe_script)
    args.ckpt = resolve_repo_path(args.ckpt)
    args.molscribe_model = resolve_repo_path(args.molscribe_model)

    if args.conditional_count < 1:
        raise ValueError("conditional-count must be >= 1")
    sample_batch_size = int(args.sample_batch_size or args.conditional_count)
    if sample_batch_size < 1 or sample_batch_size > args.conditional_count:
        raise ValueError("sample-batch-size must be in [1, conditional-count]")
    if args.conditional_count % sample_batch_size != 0:
        raise ValueError("conditional-count must be divisible by sample-batch-size")
    sample_rounds = args.conditional_count // sample_batch_size

    eval_rows = read_rows(args.eval_csv)
    if args.start_index >= 0:
        end = len(eval_rows) if args.limit <= 0 else min(len(eval_rows), args.start_index + args.limit)
        assigned = eval_rows[args.start_index : end]
    else:
        assigned = shard_rows(eval_rows, args.shard_index, args.shard_count)
        if args.limit > 0:
            assigned = assigned[: args.limit]

    shard_dir = args.output_dir / "shards" / f"shard_{args.shard_index:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = shard_dir / "shard_candidates.csv"
    progress_json = shard_dir / "progress.json"

    base_fieldnames = list(eval_rows[0].keys()) if eval_rows else []
    extra_fields = [
        "candidate_index",
        "shard_index",
        "image_path",
        "SMILES",
        "molscribe_score",
        "molscribe_decode_source",
    ]
    fieldnames = base_fieldnames + [name for name in extra_fields if name not in base_fieldnames]

    completed = 0
    skipped = 0
    failed: list[str] = []

    env = os.environ.copy()
    sketchmol_cwd = args.sketchmol_repo.resolve()
    env["PYTHONPATH"] = f"{sketchmol_cwd}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)

    for row in assigned:
        condition_id = str(row.get("condition_id") or row.get("sample_id") or "").strip()
        if not condition_id:
            failed.append("<missing condition_id>")
            continue

        token = sanitize_token(condition_id)
        done_marker = shard_dir / "conditions" / f"{token}.done"
        if args.resume and done_marker.exists():
            skipped += 1
            continue

        cond_dir = shard_dir / "conditions" / token
        cond_dir.mkdir(parents=True, exist_ok=True)
        preset = preset_for_sketchmol(row)
        log_dir = (cond_dir / "sketchmol_logs").resolve()

        try:
            image_csv: Path | None = None
            if args.resume:
                try:
                    reusable = resolve_image_csv("", log_dir, sketchmol_cwd)
                    reusable_rows = read_rows(reusable)
                    if len(reusable_rows) == args.conditional_count:
                        image_csv = reusable
                        print(
                            f"REUSE {condition_id}: {len(reusable_rows)} existing images from {reusable}",
                            flush=True,
                        )
                except FileNotFoundError:
                    pass

            if image_csv is None:
                sample_stdout = run_command(
                    [
                        args.sketchmol_python,
                        "scripts/sample_diffusion_condition_continuousV2.py",
                        "-r",
                        str(args.ckpt),
                        "--logdir",
                        str(log_dir),
                        "--post",
                        token,
                        "-p",
                        preset,
                        "--conditional_count",
                        str(sample_batch_size),
                        "-n",
                        str(sample_rounds),
                        "-c",
                        str(args.custom_steps),
                        "--scale",
                        str(args.scale),
                        "--scale_pro",
                        str(args.scale_pro),
                        "--property_num",
                        "1",
                        "--tri",
                        "true",
                    ],
                    cwd=sketchmol_cwd,
                    env=env,
                    capture_output=True,
                )
                image_csv = resolve_image_csv(sample_stdout, log_dir, sketchmol_cwd)
            image_csv = normalize_image_csv_paths(image_csv, sketchmol_cwd, log_dir)
            run_command(
                [
                    args.molscribe_python,
                    str(args.molscribe_script),
                    "--model-path",
                    str(args.molscribe_model),
                    "--image-csv",
                    str(image_csv),
                    "--batch-size",
                    str(args.molscribe_batch_size),
                    "--backend",
                    args.molscribe_backend,
                    "--no-preprocess-images",
                    "--no-raw-smiles-fallback",
                ],
                env=env,
            )
            append_candidate_rows(
                candidates_csv,
                row,
                image_csv,
                fieldnames,
                args.shard_index,
            )
            done_marker.parent.mkdir(parents=True, exist_ok=True)
            done_marker.write_text(
                json.dumps(
                    {
                        "condition_id": condition_id,
                        "preset": preset,
                        "image_csv": str(image_csv),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            completed += 1
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            failed.append(f"{condition_id}: {exc}")
            print(f"ERROR condition {condition_id}: {exc}", file=sys.stderr, flush=True)

    progress = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_rows": len(assigned),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "candidates_csv": str(candidates_csv),
    }
    progress_json.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(progress, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
