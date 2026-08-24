# P8.1.6 paired result

Both mandatory rounds completed with seed 7 and exit code 0:

- R1 `joint_bottleneck`: Nibi job `20383503`, 10m51s.
- R2 `dense_softmin`: Nibi job `20383504`, 10m51s.

R1 and R2 restart from the same P8.1.4-R1 full-SMILES checkpoint. The
scientific difference is only terminal trajectory-reward aggregation. Both
retain one checkpoint, decoder, output softmax and 47-token SMILES vocabulary;
neither adds a router, pointer-copy path, interpreter, materializer, output
token, or property reranker.

## Failure inherited from the base

The P8.1.4-R1 base already has zero validity and zero success at k=1/8/20 on
the P6 hard de-novo subset. Consequently P8.1.6 does not isolate whether either
reward can preserve a working de-novo policy: both rounds inherit and retain
zero de-novo validity. On editing, the base has 16.65% candidate validity and
0.50% relaxed candidate accuracy, but zero strict 0.65 accuracy.

## Paired edit result

| Arm | Any@1 valid / relaxed | Any@8 valid / relaxed | Any@20 valid / relaxed | Candidate valid / relaxed | Strict 0.65 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P8.1.4 base | 18.5 / 1.0 | 71.0 / 3.5 | 92.0 / 7.5 | 16.65 / 0.50 | 0.0 |
| R1 joint bottleneck | 12.5 / 0.5 | 61.0 / 2.0 | 85.0 / 6.0 | 13.05 / 0.325 | 0.0 |
| R2 dense softmin | 10.5 / 0.5 | 61.0 / 3.0 | 86.0 / 7.5 | 12.65 / 0.45 | 0.0 |

All values are percentages. Relative to R1, dense softmin recovers 1.0 point
of relaxed Any@8 and 1.5 points of relaxed Any@20, while losing 0.4 points of
candidate validity. It does not produce any strict source-preserving edit.
Thus R2 is a small relaxed-hit tradeoff, not evidence that distributionally
robust reward aggregation solves the unified task.

## Audit correction

The original R1 audit counted every nonempty string as valid and incorrectly
reported 100% candidate validity. Audit v2 canonicalizes and sanitizes every
candidate with RDKit, matching the official evaluator:

- R1: 522/4000 valid = 13.05%, identity-copy rate 0.
- R2: 506/4000 valid = 12.65%, identity-copy rate 0.

This correction only recomputes diagnostic metadata. Checkpoints, raw candidate
files, generation order and official evaluation outputs are unchanged.
