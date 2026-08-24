#!/usr/bin/env python3
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    rl = (root / "group_relative_reinforce.py").read_text()
    run = (root / "run_round.sh").read_text(); submit = (root / "submit_queue.sh").read_text()
    assert "group_relative_REINFORCE_not_GRPO" in rl
    assert "advantages = (reward_tensor - reward_tensor.mean())" in rl
    assert 'clean["target_smiles"] = ""' in rl
    assert "joint_bottleneck" in run and "dense_softmin" in run
    assert "--budgets 1,8,20" in run and "afterany:$r1" in submit
    print("P8.1.11 contract OK"); return 0


if __name__ == "__main__": raise SystemExit(main())
