# P34: hard-boundary editing RL

P34 follows the stopped P32.4 pilot. P32.4 softly rewarded movement toward the source-similarity
threshold, but its best checkpoint improved strict Raw@1 by only one point and did not increase
mean source similarity. P34 instead treats source feasibility as a hard eligibility condition.

An output must be valid, non-copy, and have Morgan similarity at least 0.65 before property
progress can raise its reward above the common ineligible floor. Because online RLOO updates only
mixed-strict groups, every updated group contains a strict eligible candidate; the common floor is
therefore below the group mean, so an ineligible candidate cannot receive positive advantage.

The P29 editing specialist remains the initialization and frozen reference. P34 trains 20
informative online-RLOO updates and evaluates checkpoints 5, 10, and 20 on the same 200-example
Raw@1 gate. The exact P32.4 baseline evaluation is reused. No full Table 2 job is launched unless
strict macro improves by at least two points without material relaxed/validity regression.

```bash
bash experiments/p34_hard_boundary_edit_rl/submit_p34.sh
```
