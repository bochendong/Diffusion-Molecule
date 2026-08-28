# P28 results: alternative RL methods from P24 checkpoints

## Outcome

Three RL methods were trained from the exact P24 checkpoint-11000 adapter and
evaluated at the preregistered step-20 endpoint. Vanilla paired GRPO and
decoupled GRPO without PCGrad both fail the frozen dev gate. The conservative
editing-protected PCGrad variant passes every dev condition, but the gain does
not replicate on the disjoint final gate. Full Table 1 and Table 2 evaluation
is not authorized.

The stable result is therefore negative: none of the tested RL methods provides
a reproducible joint improvement over P24 checkpoints 11000 and 11500.

The same three-method sweep was subsequently repeated from the newly completed
checkpoint 11500. All three variants fail its frozen dev gate, confirming the
same construction--editing trade-off at a later supervised checkpoint.

## Frozen dev gate

All entries are percentages from the same four-repeat, target-blind,
no-reranking gate.

| Policy | De novo hit@repeats | De novo strict | De novo valid | Edit strict .65 | Edit relaxed .15 | Edit valid | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P24 checkpoint 11000 | 65.000 | 30.000 | 95.833 | 55.375 | 74.625 | 96.625 | baseline |
| Vanilla paired GRPO, step 20 | 71.667 | 27.917 | 96.250 | 55.250 | 74.500 | 96.000 | stop |
| Decoupled, no PCGrad, step 20 | 66.667 | 25.000 | 95.833 | 55.000 | 74.625 | 96.750 | stop |
| Editing-protected PCGrad, step 20 | 70.000 | 30.417 | 96.667 | 56.375 | 75.125 | 96.875 | promote to final gate |

The editing-protected variant changes the dev metrics by +0.417 points for de
novo strict, +1.000 for editing strict, +0.500 for editing relaxed, and +5.000
for de novo hit@repeats. Its bucket effects are still heterogeneous: de novo
5p improves by 5.00 points, 6p is unchanged, and 7p decreases by 3.75 points.

## Disjoint final gate

| Policy | De novo hit@repeats | De novo strict | De novo valid | Edit strict .65 | Edit relaxed .15 | Edit valid | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P24 checkpoint 11000 | 76.667 | 33.333 | 96.667 | 52.125 | 75.125 | 96.000 | baseline |
| Editing-protected PCGrad, step 20 | 76.667 | 32.083 | 96.667 | 50.750 | 74.375 | 95.375 | stop |
| Delta | 0.000 | -1.250 | 0.000 | -1.375 | -0.750 | -0.625 | no promotion |

Every validity-drop tolerance remains satisfied, but strict non-regression
fails in both modes and the required joint strict gain is absent. The dev
promotion was therefore not robust to a disjoint condition split.

## Checkpoint 11500 replication

All entries below use the same seed, step-20 endpoint, four-repeat frozen dev
gate, and no-reranking protocol as the checkpoint-11000 sweep.

| Policy | De novo hit@repeats | De novo strict | De novo valid | Edit strict .65 | Edit relaxed .15 | Edit valid | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P24 checkpoint 11500 | 66.667 | 30.000 | 94.167 | 51.000 | 73.375 | 95.750 | baseline |
| Vanilla paired GRPO, step 20 | 63.333 | 24.167 | 95.417 | 53.500 | 74.125 | 96.125 | stop |
| Decoupled, no PCGrad, step 20 | 63.333 | 22.500 | 97.083 | 52.500 | 74.250 | 95.500 | stop |
| Editing-protected PCGrad, step 20 | 71.667 | 28.750 | 96.250 | 52.375 | 74.000 | 96.000 | stop |

All variants improve editing strict success, but reduce de novo strict success.
The respective strict deltas are -5.833/+2.500 points for vanilla,
-7.500/+1.500 for decoupled without PCGrad, and -1.250/+1.375 for the
editing-protected variant. No checkpoint-11500 variant is authorized for the
disjoint final gate or full-table evaluation.

## Interpretation

The P24 checkpoint itself improves substantially as broad supervised training
continues. On the identical dev gate, checkpoint 11000 reaches 30.000% de novo
strict and 55.375% editing strict, compared with 17.917% and 48.625% at
checkpoint 7500. This supervised-training gain is much larger and more stable
than any RL delta observed here.

Vanilla scalar-reward GRPO and removing PCGrad are actively harmful to de novo
Raw@1 strict success. Stronger editing anchoring, reference KL, lower learning
rate, and PCGrad can suppress that damage on the development split, but do not
yield a repeatable final-split improvement. The current evidence favors
finishing P24 broad training and its alignment refresh rather than promoting an
RL-refined checkpoint.

## Completed Slurm jobs

- successful preflight: `20705986`
- checkpoint-11000 dev baseline: `20705987`
- vanilla train/eval/compare: `20705988`, `20705989`, `20705990`
- decoupled-no-PCGrad train/eval/compare: `20705991`, `20705992`, `20705993`
- editing-protected train/eval/compare: `20705994`, `20705995`, `20705996`
- disjoint final baseline/RL/compare: `20706534`, `20706535`, `20706536`
- checkpoint-11500 preflight/baseline: `20706930`, `20706931`
- checkpoint-11500 vanilla train/eval/compare: `20706932`, `20706944`, `20706945`
- checkpoint-11500 decoupled train/eval/compare: `20706969`, `20706981`, `20706985`
- checkpoint-11500 protected train/eval/compare: `20706986`, `20706987`, `20706988`

Machine-readable artifacts are under
`outputs/p28_p24_rl_method_sweep/checkpoint_{11000,11500}_seed_28001/gate/`.
