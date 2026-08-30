# P32.3: strict-absorbing exploration RL

P32.3 targets the remaining editing failure without discarding the de-novo gain
from P32.2. A completed P32.2 diagnostic showed that a successful first graph
edit was not terminal: the policy always applied a second edit, destroying ten
of eighteen first-step de-novo rescues at checkpoint 30. P32.3 makes verified
strict success absorbing in both training and inference.

Editing additionally had zero strict trajectories in 240 P32.2 training
rollouts. P32.3 therefore runs a training-only support audit and trains only on
direct-failed editing records with at least one strict two-step path. This is a
curriculum, not reward-ranked imitation: every editing update still samples
actions from the current policy, uses terminal verifier returns, and applies a
group-relative policy-gradient objective. Exploration covers every first action
and four distinct second actions per state. De-novo updates remain paired and
PCGrad protects the shared policy.

The frozen gate is unchanged. Promotion requires editing strict success to rise
above checkpoint 0 while de-novo strict success and both validity rates do not
fall.

```bash
./experiments/p32_3_strict_absorbing_exploration_rl/submit_p32_3.sh
```
