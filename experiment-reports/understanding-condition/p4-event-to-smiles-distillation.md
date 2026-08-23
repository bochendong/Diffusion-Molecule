# P4 Event-to-SMILES distillation

| Field | Value |
| --- | --- |
| Job | `20338120` (`p4-event-distill-s7`) |
| Status | Complete, exit `0:0` |
| Runtime | 36m20s |
| Seed | 7 |
| Evaluation size | 20 inputs per task, 10 tasks |
| Decision | **stop** |
| Output | `outputs/p4_event_to_smiles_distillation_v1_h100/seed_7/` on Nibi |

P4 distills strict train-only candidates from the frozen D3 event teacher into
the source-conditioned modules of one direct-SMILES MolProgram checkpoint.
Inference uses the student only: there is no D3 teacher, router, materializer,
or property-aware candidate selection. The source-free de novo path is frozen.

## Preregistered result

| Variant | k | Validity | Acc_all(0.65) | Acc_all(0.15) |
| --- | ---: | ---: | ---: | ---: |
| Base | 1 | 39.5% | 0.0% | 15.0% |
| Base | 8 | 88.5% | 0.0% | 49.0% |
| Base | 20 | 97.5% | 0.0% | 62.5% |
| SFT | 1 | 32.0% | 0.0% | 10.0% |
| SFT | 8 | 83.0% | 0.0% | 34.0% |
| SFT | 20 | 94.5% | 0.0% | 50.0% |
| Group-RL | 1 | 33.0% | 0.0% | 9.5% |
| Group-RL | 8 | 85.5% | 0.0% | 31.0% |
| Group-RL | 20 | 96.5% | 0.0% | 48.0% |

These are any@k budget diagnostics. The matched MolEdit Table 1 comparison
instead uses candidate-level means over the complete unranked pool of 20.

| Variant | Candidates/input | Validity | Acc_all(0.65) | Acc_all(0.15) |
| --- | ---: | ---: | ---: | ---: |
| Base | 20 | **36.8%** | 0.0% | **11.8%** |
| SFT | 20 | 32.0% | 0.0% | 7.8% |
| Group-RL | 20 | 32.5% | 0.0% | 7.5% |

## Interpretation

- The preregistered P4 decision is **stop**, and the strong raw gate fails.
- The D3 teacher supplies strict targets for 154/200 conditions (77.0%), above
  the 45% coverage gate, but has zero coverage on GSK3B increase.
- SFT and Group-RL both preserve the source-free de novo path bit-for-bit.
- The transfer nevertheless fails as an editor: all ten tasks have zero
  candidate-level `Acc_all(0.65)`, and the final Group-RL student is worse than
  the protected base on both validity and relaxed accuracy.
- Any@20 must not be used to fill the MolEdit comparison table; it answers an
  easier any-hit question than the benchmark's candidate-wise aggregation.
- The pilot has only 20 inputs per task and is smaller than the published
  MolEdit Table 1 evaluation, so its candidate-level values also stay out of
  the main external table. The next eligible step is a full-scale frozen
  checkpoint evaluation, not more training.
- This job evaluates MolEdit Table 1 only. It does not produce the 30
  task/metric cells required for the MuMO IND/OOD tables.
