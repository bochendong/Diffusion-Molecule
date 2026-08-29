# P30.3: joint-gated multi-negative refinement after RL

P30.3 starts from the completed P30.1 step-30 RL adapter, the only checkpoint
that improved the frozen de-novo greedy Raw@1 screen. It applies a small,
train-only invalid-completion contrastive refinement using the existing P23
multi-negative implementation. This is an RL checkpoint followed by
contrastive syntax refinement; the refinement itself is not presented as RL.

The refinement set contains 1,200 invalid-corruption pairs: 100 for each
de-novo arity 2p--7p and 60 for each of the ten exact editing task families.
Every row comes from the held-out-disjoint P23 training set. Chosen completion
CE is normalized to one per row, and only the invalid-corruption margin is
retained. Training uses 0.5 epoch at `2e-6` to avoid overwriting the RL policy.

Every shared-policy continuation is evaluated on both modes:

- de novo: the exact frozen 120-condition greedy Raw@1 screen;
- editing: the frozen P25.1 final gate, 20 rows for each of ten tasks and four
  independent sampled Raw@1 trials per row, without reranking.

The editing baseline is the same P24 alignment-refresh adapter and is cached in
this output root for later shared-policy experiments. Full budget evaluation is
not submitted automatically.

```bash
./experiments/p30_3_joint_multinegative_refinement/submit_joint_refinement.sh
```
