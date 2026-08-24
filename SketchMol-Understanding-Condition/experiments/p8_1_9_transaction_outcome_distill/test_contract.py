#!/usr/bin/env python3
"""Static contract tests for P8.1.9."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    run = (ROOT / "run_round.sh").read_text()
    submit = (ROOT / "submit_queue.sh").read_text()
    builder = (ROOT / "build_teacher_pseudopairs.py").read_text()
    assert "source_clamped_entrypoint.py\" train" in run
    assert "--trainable-scope source_only" in run
    assert "--resume-checkpoint \"$BASE\"" in run
    assert "--teacher-checkpoint" not in run.split("source_clamped_entrypoint.py\" train", 1)[1].split("STUDENT=", 1)[0]
    assert "--disable-finalizer" in run
    assert "--num-samples 20" in run
    assert "property-rerank" not in run.lower()
    assert "afterany:$r1" in submit
    assert '"eval_candidates_used_for_training": False' in builder
    assert "np.argmax" in builder
    assert "outcome in eval_molecules" in builder
    print("P8.1.9 static contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
