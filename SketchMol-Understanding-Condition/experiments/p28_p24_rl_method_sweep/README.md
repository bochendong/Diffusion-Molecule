# P28: alternative RL methods from P24 checkpoint 11000

P28 compares three RL update rules from the same frozen P24 checkpoint and on
the same four-repeat joint gate:

1. `vanilla_paired_grpo`: scalar reward, mode-paired groups, no gradient
   surgery;
2. `decoupled_no_pcgrad`: per-mode, per-channel normalized advantages with a
   direct sum of the two mode gradients;
3. `editing_protected_pcgrad`: decoupled advantages, PCGrad, half learning
   rate, doubled editing SFT anchor, and stronger reference KL.

All trainers run a deterministic 30-pair schedule so that the existing
balanced selector remains unchanged. The preregistered comparison endpoint is
the saved step-20 adapter, motivated by the earlier checkpoint-7500 diagnostic
in which step 20 was closest to a joint improvement and step 30 overshot toward
de novo construction.

```bash
./experiments/p28_p24_rl_method_sweep/submit_dev_sweep.sh
```
