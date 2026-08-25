# P20-R2 frozen de-novo 2p-4p fair-budget results

## Corrected fill-table result

**Correction:** the earlier P20 R1 statement that K1 was the primary fill-table
metric is withdrawn. SketchMol paper-style evaluation uses a budget of 40
candidates per condition (`n_samples=1 x conditional_count=40`). K1 is only an
ablation. The primary P20 result is the repository evaluator's property-aware
`best_of_40` setting.

These remain pilot estimates with **n=100 locked conditions per property-count
stratum**, rather than the full official n=1000 per stratum.

| frozen model | setting | selected validity | 2p strict | 3p strict | 4p strict | overall strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P17 | best_of_40 | 1.000 | **0.820 (82/100)** | **0.710 (71/100)** | 0.430 (43/100) | **0.653** |
| **P18 current** | **best_of_40** | **1.000** | **0.790 (79/100)** | **0.650 (65/100)** | **0.460 (46/100)** | **0.633** |
| SketchMol paper reference | — | — | 0.804 | 0.768 | 0.736 | 0.769 for 2p-4p |

For the current P18 model, the corrected pilot row to enter is therefore:

```text
validity = 1.000; 2p = 0.790; 3p = 0.650; 4p = 0.460
```

P17 is stronger on 2p and 3p, while P18 is stronger on 4p. Both frozen rows are
reported rather than selecting one metric from each model.

## Candidate-budget sweep

`best_of_K` uses the same ordered candidate prefix and the repository's
property-aware finalizer. Target properties are not available during generation;
they are used only in this post-generation selection step.

| model | setting | validity | 2p | 3p | 4p | overall strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P17 | raw_at_1 | 1.000 | 0.100 | 0.040 | 0.000 | 0.047 |
| P17 | best_of_4 | 1.000 | 0.360 | 0.150 | 0.100 | 0.203 |
| P17 | best_of_8 | 1.000 | 0.510 | 0.250 | 0.150 | 0.303 |
| P17 | best_of_20 | 1.000 | 0.740 | 0.520 | 0.240 | 0.500 |
| **P17** | **best_of_40** | **1.000** | **0.820** | **0.710** | **0.430** | **0.653** |
| P18 | raw_at_1 | 1.000 | 0.110 | 0.020 | 0.020 | 0.050 |
| P18 | best_of_4 | 1.000 | 0.330 | 0.130 | 0.080 | 0.180 |
| P18 | best_of_8 | 1.000 | 0.500 | 0.290 | 0.150 | 0.313 |
| P18 | best_of_20 | 1.000 | 0.680 | 0.470 | 0.300 | 0.483 |
| **P18** | **best_of_40** | **1.000** | **0.790** | **0.650** | **0.460** | **0.633** |

The overall 95% normal-approximation intervals are [0.599, 0.707] for P17
best-of-40 and [0.579, 0.688] for P18 best-of-40.

## Average-of-40 diagnostic

`average_of_40` scores every generated candidate independently; it is not the
fair-budget final selected output.

| model | candidate validity | 2p strict | 3p strict | 4p strict | overall candidate strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| P17 | 0.854 | 0.117 | 0.048 | 0.022 | 0.062 |
| P18 | **0.884** | **0.118** | **0.051** | 0.022 | **0.064** |

This distinction explains why selected validity is 1.000 at best-of-40 while
the raw P18 candidate validity averaged across all 12,000 candidates is 0.884.

## Frozen-data and prefix audit

- Official source CSV: 6000 rows, exactly 1000 rows each for 2p through 7p.
- Pilot subset: the same locked 300 conditions as R1, with 100 each for 2p/3p/4p.
- Reference SHA256: `ce8732266e156a59efc97a1be2466037343661188f838f6938340e3d20398739`.
- Prompt SHA256: `afb6947ddba9ef559abc1056c5c379edc2b5520168f548aca3c91842945a4302`.
- P16/P17 canonical training-target overlap: 0.
- P17 R1 ranks 1-8 SHA256, verified unchanged:
  `931de0c47cccdd476101917daeb7715c19565b484fcde5d49ac3462d7ff24aed`.
- P18 R1 ranks 1-8 SHA256, verified unchanged:
  `13e7d62be1ca6903424ea259da25ab30193a94d3a87866a129d0adef2862d6d6`.
- R2 generated only ranks 9-40: four deterministic batches of eight samples,
  seed 9040 with a fixed batch stride, no target access.
- Final merged pools: exactly 12,000 rows per model, 300 conditions x 40 ranks.
- No training, parameter update, benchmark tuning, or external static candidate library.
- Official evaluator:
  `SketchMolBenchmark/scripts/evaluate_denovo_2p7p_budget_sweep.py` with budgets
  `1,4,8,20,40`.
- Absolute strict tolerances: MW 35, LogP 1, QED 0.10, TPSA 20, HBD 1, HBA 1,
  RB 1, and SA 1. Invalid candidates fail strict.

## Jobs

- P17 ranks 9-40 generation: `20463422`, COMPLETED 0:0 in 32m32s.
- P18 ranks 9-40 generation: `20463425`, COMPLETED 0:0 in 31m23s.
- Merge and official budget sweep: `20464118`, COMPLETED 0:0 in 42s.
- Redundant 20GB/full-H100 races were cancelled after both 40GB MIG jobs started.

## Local artifacts

- R2 preregistration: `experiments/p20_frozen_denovo_2p4p_table/r2_preregistration.json`
- Pre-extension audit: `outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/audit_before_extension.json`
- Corrected aggregate JSON: `outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/results/aggregate_r2.json`
- Direct corrected table CSV: `outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/results/fair_table_rows.csv`
- P17 official sweep: `outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/results/budget_sweep/p17/`
- P18 official sweep: `outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/results/budget_sweep/p18/`
- Merged raw and evaluator CSVs:
  `outputs/p20_frozen_denovo_2p4p_table/seed_2020/r2/generated/{p17,p18}/denovo.{raw40,eval40}.csv`

All R2 artifacts were synchronized from Nibi to the local workspace.
