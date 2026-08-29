# P31: P24 reward-support audit before further RL

P31 freezes the P24 alignment-refresh policy and audits whether its current
candidate distribution already contains strict successes that can be distilled
into Raw@1. It performs no training and never reads a frozen evaluation split.

The audit uses 60 training prompts from each of the six de-novo arities and ten
exact editing task families in the P24 alignment-refresh training set. For each
prompt it generates one greedy output plus 16 target-blind sampled outputs.
All candidates are scored with the same validity, property, source-similarity,
and strict-success implementation used by P30.

The report separates three cases per task:

- ranking opportunity: sampled strict successes exist but greedy fails;
- reward misalignment: a strict success exists but the P30 advantage ranks a
  failure first;
- support limitation: even Any@16 strict success is low.

Distillable pairs are exported only as an audited preview. No preference or RL
training is launched automatically.

```bash
./experiments/p31_reward_support_audit/submit_support_audit.sh
```
