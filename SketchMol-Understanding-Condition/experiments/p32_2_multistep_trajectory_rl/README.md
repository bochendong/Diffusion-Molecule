# P32.2: multi-step terminal-return graph RL

P32.2 isolates the credit-assignment failure observed in P32.1. It retains the
same pinned train/gate records and verifier-routed residual inference contract,
but replaces immediate per-action rewards with a terminal return shared by
both actions in a two-step trajectory.

For every direct-failed training proposal, the current policy constructs eight
trajectories with distinct first actions sampled without replacement. Final
strict success dominates the return. A stopped failed proposal receives zero,
so dense property rewards cannot turn `stop` into the universal optimum. The
group-relative terminal advantage is applied to every selected action in the
trajectory, and de-novo/editing gradients are merged with paired PCGrad.

The frozen P32.1 gate and pinned oracle labels are reused. Checkpoint 0 is P32
checkpoint 30 under the residual routing protocol; checkpoints 10, 20, and 30
measure only the contribution of P32.2 trajectory RL.

```bash
./experiments/p32_2_multistep_trajectory_rl/submit_p32_2.sh
```
