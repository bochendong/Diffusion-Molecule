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

## Complete 2p--7p de novo complexity curve

The same checkpoint was sampled in generation order for 1,000 held-out conditions at each property count and 20 candidates per condition. Structural evaluation targets were removed before inference. Raw success is the fraction of individual candidates satisfying every active property; Pass@$k$ is the fraction of conditions with at least one success in the first $k$ candidates.

### Pass@$k$ (%)

| Properties | Conditions | Pass@1 | Pass@4 | Pass@8 | Pass@20 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1,000 | 15.9 | 47.4 | 66.0 | 85.7 |
| 3 | 1,000 | 11.5 | 37.0 | 54.9 | 74.3 |
| 4 | 1,000 | 8.2 | 27.1 | 42.5 | 66.7 |
| 5 | 1,000 | 7.5 | 23.0 | 36.5 | 57.2 |
| 6 | 1,000 | 5.7 | 17.1 | 29.8 | 46.6 |
| 7 | 1,000 | 5.3 | 15.6 | 26.7 | 44.6 |
| **All** | **6,000** | **9.017** | **27.867** | **42.733** | **62.517** |

### Raw candidate success (%)

| Properties | Raw@1 | Raw@4 | Raw@8 | Raw@20 |
|---:|---:|---:|---:|---:|
| 2 | 15.900 | 17.425 | 17.850 | 17.615 |
| 3 | 11.500 | 11.750 | 11.913 | 12.095 |
| 4 | 8.200 | 8.550 | 8.525 | 9.005 |
| 5 | 7.500 | 7.175 | 7.113 | 7.460 |
| 6 | 5.700 | 5.425 | 5.475 | 5.310 |
| 7 | 5.300 | 4.675 | 4.713 | 4.545 |
| **All** | **9.017** | **9.167** | **9.265** | **9.338** |

### Candidate validity (%)

| Properties | Valid@1 | Valid@4 | Valid@8 | Valid@20 |
|---:|---:|---:|---:|---:|
| 2 | 34.500 | 34.875 | 35.225 | 34.825 |
| 3 | 33.000 | 33.100 | 33.663 | 34.185 |
| 4 | 32.000 | 33.300 | 32.788 | 33.510 |
| 5 | 34.000 | 33.250 | 33.800 | 34.050 |
| 6 | 33.600 | 33.725 | 33.038 | 32.665 |
| 7 | 32.200 | 33.700 | 33.513 | 33.945 |
| **All** | **33.217** | **33.658** | **33.671** | **33.863** |

The 6p/7p hard aggregate contains 2,000 conditions and reaches 5.5% Strict@1, 28.25% Pass@8, and 45.6% Pass@20 at 33.305% candidate validity over all 20 draws. This is the paper-facing hard-de-novo row for the unified checkpoint.

### Alignment with the legacy and published matched tables

| Method | Budget | Avg. | 2p | 3p | 4p | 5p | 6p | 7p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P8.2 state-adaptive checkpoint | 20 | 62.5 | 85.7 | 74.3 | 66.7 | 57.2 | 46.6 | 44.6 |
| P1 direct Group-RL legacy curve | 20 | 62.5 | 85.6 | 73.7 | 66.3 | 60.5 | 47.5 | 41.5 |
| P1 direct Group-RL matched rerun | 40 | 68.1 | 91.7 | 83.2 | 72.7 | 63.6 | 51.0 | 46.5 |
| SketchMol published reference | 40 | 73.1 | 80.4 | 76.8 | 73.6 | 71.6 | 67.8 | 68.5 |

The fresh raw20 curve reproduces the legacy P1 aggregate Pass@20 (62.5%) and closely tracks each complexity bucket, as expected from the protected de novo path. The raw20 row is not used as a direct rank comparison against the 40-candidate SketchMol table; both budgets are shown explicitly.

The final audit passes with checkpoint SHA-256 `9a95fb9dc4056bbca04f9a9681b9b0d1c11b5cf5d73f5c01643a95ad6f22ad6f`: both evaluation arms name the exact same checkpoint, all 120,000 de novo candidates are present, every property count contributes 20,000 candidates, property reranking is disabled, and no structural evaluation target column reaches de novo inference.
