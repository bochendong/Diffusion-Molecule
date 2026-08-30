# P32.4: editing-specialist source-constrained online RL

P32.4 follows the asymmetric result of P29: the editing specialist is stronger than the shared
adapter, whereas a construction specialist is not. The construction policy is therefore frozen.
Only the P29 editing-specialist LoRA receives online sequence-RLOO updates.

The reward gives dense credit for remaining in the source molecule's neighborhood, then for
property progress, with a terminal strict-success bonus that no non-strict candidate can exceed.
This directly tests whether earlier RL failures came from optimizing properties before learning
source preservation.

The first run is intentionally small. Checkpoints 5, 10, and 20 are compared with the unchanged
P29 editing specialist on the fixed 200-example P25.1 editing gate using sampled Raw@1. A
checkpoint is promoted only if strict macro improves by at least two points, relaxed and validity
drop by no more than one point, and at least seven of ten task buckets do not regress. Full Table 2
evaluation is a separate follow-up after promotion.

```bash
bash experiments/p32_4_edit_specialist_source_rl/submit_p32_4.sh
```
