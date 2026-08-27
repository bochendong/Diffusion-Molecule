# P27: joint RL from a P24 broad-training checkpoint

P27 measures the incremental effect of the frozen P26 conflict-aware joint-RL
recipe when the initial policy is a later P24 balanced-scale checkpoint rather
than the aligned-24k policy. The first locked run uses P24 checkpoint 7500,
which was the newest complete checkpoint available before submission.

The intervention changes only the initial adapter. It reuses the P26 training
rows, paired rollout schedule, rewards, random seed, and frozen P25.1 dev gate.
The baseline and RL policy are evaluated with identical prompts and four
sampling repeats. Outputs are isolated under
`outputs/p27_p24_checkpoint_joint_rl/checkpoint_7500_seed_26001` and do not
modify the running P24 full-training directory.

Submit with:

```bash
./experiments/p27_p24_checkpoint_joint_rl/submit_dev_gate.sh
```
