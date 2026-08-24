# D3 candidate-level MolEdit Table1 results

Final post-hoc job: `20401081` (`COMPLETED`, exit 0, 8m32s). The job reused the frozen seed-2005 raw streams: 997 evaluable conditions and exactly 20 candidates per condition (19,940 candidate rows) for each model. It did not train, rank, select, or regenerate molecules.

The candidate-level rows below are directly comparable to the candidate-level `Acc_all` values reported by MolEditRL. Any@20 remains a separate condition-level diagnostic.

| Model | Candidate validity | Candidate Acc_all(.65) | Candidate Acc_valid(.65) | Candidate Acc_all(.15) | Candidate Acc_valid(.15) | Any@20 Acc_all(.65) | Any@20 Acc_all(.15) |
|---|---:|---:|---:|---:|---:|---:|---:|
| D3 supervised | 97.379 | 43.350 | 44.105 | 57.600 | 58.740 | 76.522 | 92.506 |
| D3 + GRPO | **97.510** | **46.118** | **46.878** | **61.897** | **63.033** | **78.831** | **93.313** |
| MolEditRL reported macro | 96.620 | 45.000 | -- | 72.700 | -- | -- | -- |

GRPO improves candidate-level strict accuracy by 2.768 percentage points and relaxed accuracy by 4.298 points over D3 supervised. Its strict `Acc_all(.65)` exceeds the reported MolEditRL macro by 1.118 points; its relaxed `Acc_all(.15)` remains 10.803 points lower. This supports a strict-threshold editing claim, not an across-the-board editing state-of-the-art claim.

## D3 + GRPO per task

| Task | Candidates | Validity | Acc_all(.65) | Acc_valid(.65) | Acc_all(.15) | Acc_valid(.15) |
|---|---:|---:|---:|---:|---:|---:|
| GSK3B up | 1,980 | 93.889 | 15.455 | 16.460 | 36.162 | 38.515 |
| Rotbonds down | 1,980 | 99.545 | 86.919 | 87.316 | 93.889 | 94.318 |
| MW up | 2,000 | 96.650 | 7.450 | 7.708 | 14.900 | 15.416 |
| SA down | 2,000 | 98.850 | 63.100 | 63.834 | 84.400 | 85.382 |
| Haccept down / SA down | 2,000 | 98.050 | 51.100 | 52.116 | 72.650 | 74.095 |
| QED up / SA down | 2,000 | 97.950 | 59.600 | 60.847 | 76.300 | 77.897 |
| Haccept down / LogP up | 1,820 | 98.626 | 61.868 | 62.730 | 71.813 | 72.813 |
| Haccept down / MW down | 2,000 | 98.500 | 56.050 | 56.904 | 81.950 | 83.198 |
| DRD2 down / MW down / SA down | 2,000 | 99.450 | 45.450 | 45.701 | 66.100 | 66.466 |
| Haccept up / MW up / QED down | 1,980 | 93.586 | 14.192 | 15.165 | 20.808 | 22.234 |

## Integrity and oracle handling

- Empty raw outputs remain in the denominator and count as invalid.
- Evaluation fails unless every matched condition has exactly 20 raw rows.
- The final run used the original pinned DRD2 pickle, not a repickled derivative.
- The legacy sklearn module path is shimmed at load time; the learned arrays remain unchanged.
- DRD2 uses the Graph2Graph radius-3 feature-count fingerprint, while GSK3B retains its ECFP4 bit-vector fingerprint.
- The known DRD2 active probe scored `0.9999993745656254` before submission.

Nibi audits:

- `outputs/d3_event_kernel_energy_table1_n20/d3_table1_aggregation_audit.json`
- `outputs/d3_event_kernel_energy_grpo_table1_n20/d3_table1_aggregation_audit.json`
- `logs/d3_candidate_level_posthoc/d3-candidate-metrics-20401081.log`
