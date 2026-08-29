# P30.2: validity-stable Raw@1 RL

P30.2 is the preregistered repair of the completed P30.1 screen. P30.1 step 30
improved the 120-condition greedy Raw@1 macro by 2.5 points but reduced
validity from 100.0% to 98.3%. Earlier checkpoints did not improve the macro.

This run starts again from the frozen P24 alignment-refresh adapter. It does
not continue from any P30.1 checkpoint. Relative to P30.1 it changes only the
stability controls:

- learning rate: `1.5e-7 -> 1.0e-7`;
- de-novo validity weight: `0.5 -> 1.5`;
- de-novo canonical-output weight: `0.1 -> 0.25`;
- de-novo SFT anchor: `1.5 -> 2.0`;
- frozen-reference KL: `0.1 -> 0.2`.

The training duration remains fixed at 30 paired steps with group size 16.
The completed P30.1 gate prompts and cached P24 baseline are reused exactly.
Only one greedy candidate is generated for each of 120 conditions. Full
1/4/8/20/40 evaluation remains a manual follow-up after promotion.

```bash
./experiments/p30_2_validity_stable_raw1_rl/submit_validity_stable_raw1_rl.sh
```

