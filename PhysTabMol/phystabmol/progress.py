"""Small stdout progress helpers for Slurm-friendly logs."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def progress_enabled() -> bool:
    return os.environ.get("PHYSTABMOL_PROGRESS", "1").strip().lower() not in {"0", "false", "no", "off"}


def progress_step(default: int = 5) -> int:
    raw = os.environ.get("PHYSTABMOL_PROGRESS_STEP", str(default))
    try:
        return max(1, min(100, int(float(raw))))
    except Exception:
        return default


def iter_progress(items: Iterable[T], total: int, label: str, step: int | None = None) -> Iterator[T]:
    if total <= 0 or not progress_enabled():
        yield from items
        return

    step = progress_step() if step is None else max(1, min(100, int(step)))
    next_pct = step
    start = time.time()
    print(f"{label}: 0% (0/{total})", flush=True)
    idx = 0
    completed = False
    try:
        for idx, item in enumerate(items, start=1):
            yield item
            pct = int(idx * 100 / total)
            if pct >= next_pct or idx == total:
                elapsed = time.time() - start
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(f"{label}: {pct}% ({idx}/{total}, {rate:.1f}/s)", flush=True)
                next_pct = ((pct // step) + 1) * step
        completed = True
    finally:
        if idx and not completed:
            elapsed = time.time() - start
            rate = idx / elapsed if elapsed > 0 else 0.0
            pct = int(idx * 100 / total)
            print(f"{label}: stopped at {pct}% ({idx}/{total}, {rate:.1f}/s)", flush=True)
