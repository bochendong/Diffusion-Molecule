#!/usr/bin/env python3
"""Export Unified 3M eval latents for benchmark materialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_unified_3m_diffusion.benchmark_export import write_edit_latent_benchmark_inputs  # noqa: E402
from sketchmol_unified_3m_diffusion.unified_condition_dataset import EDIT_GENERATION, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True, type=Path)
    parser.add_argument("--latents-npy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fingerprint-dim", type=int, default=None)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = [sample for sample in read_jsonl(args.eval_jsonl) if sample.task_type == EDIT_GENERATION]
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"No edit_generation rows found in {args.eval_jsonl}")
    latents = np.load(args.latents_npy)
    if args.limit is not None and args.limit > 0:
        latents = latents[: args.limit]
    fingerprint_dim = args.fingerprint_dim or _fingerprint_dim_from_metrics(args.metrics_json) or 512
    summary = write_edit_latent_benchmark_inputs(
        samples,
        latents,
        args.output_dir,
        fingerprint_dim=fingerprint_dim,
        variant=args.variant,
    )
    summary.update(
        {
            "eval_jsonl": str(args.eval_jsonl),
            "latents_npy": str(args.latents_npy),
            "metrics_json": str(args.metrics_json) if args.metrics_json else None,
        }
    )
    (args.output_dir / "benchmark_export_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _fingerprint_dim_from_metrics(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("fingerprint_dim")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
