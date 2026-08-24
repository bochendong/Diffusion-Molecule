#!/usr/bin/env python3
"""Static CPU contract checks for P8.1.7."""

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    entry = (root / "source_clamped_entrypoint.py").read_text(encoding="utf-8")
    run = (root / "run_round.sh").read_text(encoding="utf-8")
    submit = (root / "submit_queue.sh").read_text(encoding="utf-8")
    assert 'P817_SOURCE_CLAMP_SCALE' in entry
    assert 'core.task_mode_for_row(row) != core.EDIT_MODE' in entry
    assert 'np.concatenate([base, program]' in entry
    assert '--disable-finalizer' in run
    assert '--budgets 1,8,20' in run
    assert 'afterany:$r1' in submit
    assert 'r2' in submit
    print("P8.1.7 contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
