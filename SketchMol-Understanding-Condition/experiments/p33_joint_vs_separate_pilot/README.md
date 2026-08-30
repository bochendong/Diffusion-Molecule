# P33: clean joint-vs-separate pilot

P33 is the smallest matched experiment that directly tests whether a unified molecular policy is
useful. Three fresh rank-16 LoRA adapters start from the same unadapted Qwen2.5-VL-7B checkpoint:
one sees 3,000 de-novo plus 3,000 editing examples, while two specialists see the exact same 3,000
examples for their respective task. The joint arm therefore uses one adapter where the separate
system uses two, and its total data exposure is matched to the sum of the specialists.

The pilot uses one seed and small target-blind Raw@1 gates: 20 construction requests per arity and
20 editing requests per benchmark task. It is a screening experiment, not the final paper table.
Joint training supports the unified claim only when it is within two percentage points of both
specialists; a two-point gain on either task is treated as preliminary positive transfer.

```bash
bash experiments/p33_joint_vs_separate_pilot/submit_p33.sh
```
