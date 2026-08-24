#!/usr/bin/env python3
"""Fail closed unless a reused P8.1.2 R2 representation gate is exact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    args = parser.parse_args()
    if not args.rows.is_file() or args.rows.stat().st_size <= 0:
        raise FileNotFoundError(args.rows)
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    if payload.get("variant") != "r2_source_aligned":
        raise ValueError("prepared rows are not the preregistered R2 variant")
    if float(payload.get("coverage", 0.0)) < 1.0:
        raise ValueError("R2 representation coverage is incomplete")
    for mode, metrics in payload.get("by_mode", {}).items():
        if float(metrics.get("exact_reconstruction", 0.0)) < 1.0:
            raise ValueError(f"R2 {mode} reconstruction is not exact")
        if float(metrics.get("fit_budget_fraction", 0.0)) < 1.0:
            raise ValueError(f"R2 {mode} does not fit the frozen budget")
    print(f"P8.1.2 prepared R2 verified: {args.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
