# P30: balanced shared-policy group-relative RL

P30 is the confirmatory shared-policy RL experiment for the balanced P24
supervised policy.  It starts from the exact completed P24 balanced adapter,
not from an intermediate checkpoint, and keeps one LoRA policy shared by de
novo construction and source-conditioned editing.

The rollout schedule is balanced at both levels.  Every optimizer step pairs
one construction group with one editing group.  Across 60 paired steps, the
six construction arities (2p--7p) contribute ten groups each and the ten exact
Table 2 editing families contribute six groups each.  Every group contains 16
samples and its reward channels are normalized only within that prompt.

The two mode losses are differentiated separately.  Before the optimizer
update, their gradients are normalized to equal norm and averaged.  This
bisector update prevents either mode from dominating through gradient scale;
when the gradients are not exactly opposite, it is a descent direction for
both local mode losses.  A frozen-reference KL term and equal SFT anchors keep
the update close to the balanced supervised policy.

Checkpoints 10, 20, 30, 40, 50, and 60 are evaluated on the frozen development
gate.  The checkpoint is selected without reading the final gate: first prefer
checkpoints satisfying every joint promotion condition, then maximize the
smaller of the construction and editing strict deltas.  The selected checkpoint
is evaluated once on the disjoint final gate.  Native Table 1 and Table 2 runs
are authorized only if both development and final gates pass.

Submit on Nibi with:

```bash
./experiments/p30_balanced_shared_policy_rl/submit_balanced_rl.sh
```
