# P26: reward-decoupled, conflict-aware joint RL

P26 tests whether the unified P23 Qwen construction/editing policy can be
improved without allowing the denser editing reward to dominate the sparse
5p--7p construction reward.

The intervention has three coupled parts. Each reward channel is normalized
inside its own mode and rollout group before aggregation; property rewards use
both aligned mean satisfaction and a smooth bottleneck term; and de-novo and
editing LoRA gradients are measured separately and passed through symmetric
two-task PCGrad whenever their cosine is negative. Training targets are used
only by the two mode-specific SFT anchors and never by rollout reward.

Before interpreting P26, the same submission also evaluates P25.1 checkpoints
10 and 20 on the frozen dev gate to detect endpoint overshoot.
The GPU chain is gated by a CPU preflight that compiles the trainer and runs
the experiment contract tests in the exact Nibi RDKit environment.

```bash
./experiments/p26_decoupled_joint_rl/submit_p251_trajectory.sh
./experiments/p26_decoupled_joint_rl/submit_dev_gate.sh
```
