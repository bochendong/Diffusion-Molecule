# P30.1: alignment-refresh Raw@1 RL screen

P30.1 is a low-cost continuation of P30 aimed specifically at the frozen P24
de-novo Raw@1 endpoint. It starts from the exact P24 alignment-refresh adapter
used for the current Table 1 budget curve. The earlier P30 run is preserved.

The RL update is fixed at 30 paired construction/editing steps because the
original P30 development gate selected step 30 and later checkpoints degraded
de-novo strict success. Training still uses 16 candidates per prompt to obtain
group-relative advantages, but the screening endpoint is one greedy candidate
per condition. No Any@K or property-aware reranking is used in the screen.

The screen deterministically freezes 20 P24 conditions per arity (2p--7p), for
120 candidates total. Its baseline is read from the completed P24
`budget_sweep_condition_detail.csv` files; the baseline model is not generated
again. Full 1/4/8/20/40 evaluation is deliberately not submitted by this
pipeline. It is authorized only when the screen improves macro Raw@1 by at
least two percentage points, loses no more than one point of validity, and no
arity drops by more than ten points.

Submit on Nibi with:

```bash
./experiments/p30_1_alignment_raw1_rl/submit_alignment_raw1_rl.sh
```

