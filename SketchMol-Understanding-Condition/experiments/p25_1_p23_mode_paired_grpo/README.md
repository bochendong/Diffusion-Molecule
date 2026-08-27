# P25.1: mode-paired P23 group-relative RL

P25.1 repairs the two main problems exposed by the completed P25 gate. P25
performed 30 edit updates but only nine high-order de-novo updates, and its
parameter-space trust penalty remained around `1e-11`. P25.1 therefore pairs
one de-novo group with one edit group in every optimizer step and replaces the
ineffective penalty with token-level KL to a second, frozen copy of the P23
adapter.

The 30 paired steps contain ten groups each for 5p, 6p, and 7p, plus three
groups for each exact Table 2 edit task. Each prompt samples eight candidates.
The supervised anchor stays at 1.0 and the frozen-reference KL coefficient is
0.05.

Evaluation uses two newly constructed 260-row gates. Both exclude the old P25
gate and are mutually disjoint. Only the dev gate is submitted initially. The
final gate cannot be submitted unless the dev comparison promotes, and paper
tables cannot be submitted unless the final comparison also promotes.

```bash
./experiments/p25_1_p23_mode_paired_grpo/submit_dev_gate.sh
```
