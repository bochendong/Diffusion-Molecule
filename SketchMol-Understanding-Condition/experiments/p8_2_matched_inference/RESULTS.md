# P8.2 matched-inference Table1 results

All numbers below are percentages from the original generation order. Each of the 200 conditions has 20 raw candidates; no property reranking is used.

## Aggregate

| Selection | Validity | Acc_all (0.65) | Acc_valid (0.65) | Acc_all (0.15) | Acc_valid (0.15) |
|---|---:|---:|---:|---:|---:|
| Any@1 | 100.000 | 23.500 | 23.500 | 24.000 | 24.000 |
| Any@8 | 100.000 | 53.000 | 53.000 | 53.500 | 53.500 |
| Any@20 | 100.000 | 70.500 | 70.500 | 71.000 | 71.000 |
| Candidate-level | 100.000 | 17.075 | 17.075 | 18.025 | 18.025 |

Candidate identity: 0.000%. Candidate uniqueness: 100.000%.

## Per-task

| Task | Cand. Validity | Cand. Acc_all .65 | Cand. Acc_valid .65 | Cand. Acc_all .15 | Cand. Acc_valid .15 | Strict Any@1 | Strict Any@8 | Strict Any@20 | Relaxed Any@1 | Relaxed Any@8 | Relaxed Any@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSK3B↑ | 100.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Rotbonds↓ | 100.000 | 3.000 | 3.000 | 3.000 | 3.000 | 0.000 | 25.000 | 55.000 | 0.000 | 25.000 | 55.000 |
| MW↑ | 100.000 | 71.000 | 71.000 | 77.500 | 77.500 | 85.000 | 100.000 | 100.000 | 90.000 | 100.000 | 100.000 |
| SA↓ | 100.000 | 12.000 | 12.000 | 13.000 | 13.000 | 20.000 | 45.000 | 85.000 | 20.000 | 50.000 | 90.000 |
| Haccept↓ SA↓ | 100.000 | 2.250 | 2.250 | 2.250 | 2.250 | 5.000 | 20.000 | 40.000 | 5.000 | 20.000 | 40.000 |
| QED↑ SA↓ | 100.000 | 8.750 | 8.750 | 9.000 | 9.000 | 10.000 | 55.000 | 75.000 | 10.000 | 55.000 | 75.000 |
| Haccept↓ LogP↑ | 100.000 | 8.500 | 8.500 | 9.250 | 9.250 | 5.000 | 45.000 | 70.000 | 5.000 | 45.000 | 70.000 |
| Haccept↓ MW↓ | 100.000 | 10.000 | 10.000 | 10.000 | 10.000 | 5.000 | 60.000 | 85.000 | 5.000 | 60.000 | 85.000 |
| DRD2↓ MW↓ SA↓ | 100.000 | 11.000 | 11.000 | 11.000 | 11.000 | 45.000 | 80.000 | 95.000 | 45.000 | 80.000 | 95.000 |
| Haccept↑ MW↑ QED↓ | 100.000 | 44.250 | 44.250 | 45.250 | 45.250 | 60.000 | 100.000 | 100.000 | 60.000 | 100.000 | 100.000 |

Checkpoint SHA-256: `9a95fb9dc4056bbca04f9a9681b9b0d1c11b5cf5d73f5c01643a95ad6f22ad6f`.

The JSON companion preserves all five metrics for candidate-level and Any@1/8/20 for every task.
