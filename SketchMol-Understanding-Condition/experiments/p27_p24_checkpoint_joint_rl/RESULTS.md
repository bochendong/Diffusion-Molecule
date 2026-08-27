# P27 results: joint RL from P24 checkpoint 7500

## Outcome

The newest complete P24 checkpoint at submission time was checkpoint 7500.
The frozen P26 conflict-aware joint-RL recipe was applied without changing its
training rows, rollout schedule, rewards, random seed, or gate. All three saved
RL endpoints were evaluated on the same four-repeat P25.1 dev gate.

No endpoint satisfies every preregistered promotion condition. Step 20 is the
closest Pareto improvement: de novo strict increases by 0.833 percentage
points and editing strict by 0.125 points, but their summed gain is 0.958
points, just below the required 1.0 point. Step 30 produces a larger de novo
gain while reducing editing strict by 0.250 points. Full paper-table evaluation
is therefore not authorized from this gate.

## Aggregate gate results

All entries are percentages. `Hit@repeats` is the de novo condition-level hit
macro across the four sampling repeats.

| Policy | De novo hit@repeats | De novo strict | De novo valid | Edit strict .65 | Edit relaxed .15 | Edit valid | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P24 checkpoint 7500 | 45.000 | 17.917 | 95.833 | 48.625 | 71.000 | 93.375 | baseline |
| + RL step 10 | 41.667 | 15.833 | 96.250 | 48.875 | 71.375 | 92.750 | stop |
| + RL step 20 | 48.333 | 18.750 | 97.083 | 48.750 | 70.875 | 92.250 | stop |
| + RL step 30 | 53.333 | 20.833 | 96.250 | 48.375 | 70.125 | 92.500 | stop |

## Deltas from checkpoint 7500

| Endpoint | De novo hit@repeats | De novo strict | Edit strict .65 | Edit relaxed .15 | Joint strict gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL step 10 | -3.333 | -2.083 | +0.250 | +0.375 | -1.833 |
| RL step 20 | +3.333 | +0.833 | +0.125 | -0.125 | +0.958 |
| RL step 30 | +8.333 | +2.917 | -0.250 | -0.875 | +2.667 |

At step 30, the de novo strict changes are +6.25/-2.50/+5.00 points for
5p/6p/7p respectively. Editing changes are heterogeneous: MW increase gains
7.50 points, while GSK3B increase loses 6.25 points and SA decrease loses 3.75
points. The DRD2/MW/SA three-property task remains at zero strict success.

## Training diagnostics

- paired optimizer steps: 30
- rollout group size: 16 per mode
- de novo zero-signal groups: 0
- editing zero-signal groups: 2
- gradient conflicts: 10
- PCGrad applications: 10
- target-SMILES reward access: false

## Completed Slurm jobs

- preflight: `20678704`
- checkpoint-7500 baseline gate: `20678705`
- joint-RL training: `20678706`
- step-30 gate and comparison: `20678707`, `20678708`
- step-10 gate and comparison: `20679164`, `20679166`
- step-20 gate and comparison: `20679165`, `20679167`

Machine-readable comparisons are under
`outputs/p27_p24_checkpoint_joint_rl/checkpoint_7500_seed_26001/gate/dev/`.
