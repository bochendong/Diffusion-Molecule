# P22 MolEdit Table1 protocol sweep results

## Decision

MolEditRL Table 1 is an output-level, candidate-pooled evaluation. Its ten task rows
resolve exactly to 500 outputs per task. The table alone cannot distinguish
`500 sources x 1` from fewer sources with multiple unselected outputs, but it is not
an Any@K metric.

The closest protocol-compatible result is the frozen D3 + GRPO raw-candidate stream.
Its candidate-level result is stable from K=1 to K=20. At K=20 it reaches 97.51%
validity, 46.12% strict `Acc_all(.65)`, and 61.90% relaxed `Acc_all(.15)`, versus
MolEditRL's 96.62%, 45.02%, and 72.66%. Strict accuracy is close (+1.10 points), but
relaxed accuracy remains 10.76 points lower and the task-level vector is not matched.

The P19/P18 unified LLM is not close under the candidate-level protocol. Its first
candidate reaches 83.0% / 19.0% / 39.0%; pooling its first eight raw candidates gives
79.13% / 8.0% / 34.38%. Its Any@8 result rises to 100% / 41% / 92%, confirming that
Any@K measures a different best-of-budget event.

## MolEditRL integer reconstruction

| Task | Valid outputs | Strict successes | Relaxed successes |
|---|---:|---:|---:|
| GSK3B up | 476/500 | 171/500 | 257/500 |
| Rotbonds down | 492/500 | 317/500 | 415/500 |
| MW up | 480/500 | 202/500 | 428/500 |
| SA down | 494/500 | 314/500 | 414/500 |
| Haccept down + SA down | 486/500 | 173/500 | 255/500 |
| QED up + SA down | 487/500 | 316/500 | 394/500 |
| Haccept down + LogP up | 473/500 | 158/500 | 400/500 |
| Haccept down + MW down | 471/500 | 126/500 | 330/500 |
| DRD2 down + MW down + SA down | 493/500 | 259/500 | 362/500 |
| Haccept up + MW up + QED down | 479/500 | 215/500 | 378/500 |
| **Macro / total** | **4831/5000 = .9662** | **2251/5000 = .4502** | **3633/5000 = .7266** |

The displayed `Acc_valid` entries are also recovered exactly by dividing each success
count by the corresponding valid-output count.

## D3 frozen raw-stream sweep

All candidate rows retain every raw output in the denominator. Any rows collapse each
source to one success event.

| Model | K | Candidate validity | Candidate strict | Candidate relaxed | Any strict | Any relaxed |
|---|---:|---:|---:|---:|---:|---:|
| D3 supervised | 1 | .9758 | .4236 | .5779 | .4236 | .5779 |
| D3 supervised | 2 | .9768 | .4287 | .5783 | .5054 | .6696 |
| D3 supervised | 4 | .9758 | .4357 | .5792 | .5895 | .7656 |
| D3 supervised | 5 | .9758 | .4348 | .5773 | .6117 | .7857 |
| D3 supervised | 10 | .9744 | .4359 | .5751 | .6893 | .8551 |
| D3 supervised | 20 | .9738 | .4335 | .5760 | .7652 | .9251 |
| D3 + GRPO | 1 | .9807 | .4566 | .6229 | .4566 | .6229 |
| D3 + GRPO | 2 | .9777 | .4576 | .6222 | .5405 | .7116 |
| D3 + GRPO | 4 | .9752 | .4593 | .6203 | .6235 | .7968 |
| D3 + GRPO | 5 | .9741 | .4592 | .6160 | .6487 | .8241 |
| D3 + GRPO | 10 | .9754 | .4610 | .6179 | .7235 | .8957 |
| D3 + GRPO | 20 | .9751 | .4612 | .6190 | .7883 | .9331 |

Ignoring protocol semantics, the closest three-macro vector is D3 supervised Any@2
(`.9960/.5054/.6696`, macro L2 distance `.0847`). It is not a valid replacement for
MolEditRL candidate-level `Acc_all`. The closest compatible macro row is D3 + GRPO
candidate-level K=1 (`.9807/.4566/.6229`, distance `.1049`); K=20 has distance `.1086`.
The small difference across candidate prefixes is evidence that candidate pooling is
stable rather than a hidden best-of-K score.

## P19 frozen unified-LLM sweep

| Model | K | Candidate validity | Candidate strict | Candidate relaxed | Any strict | Any relaxed |
|---|---:|---:|---:|---:|---:|---:|
| P17 | 1 | .8400 | .1800 | .3800 | .1800 | .3800 |
| P17 | 2 | .8300 | .0950 | .3400 | .1900 | .5300 |
| P17 | 4 | .7950 | .0600 | .3100 | .2200 | .7600 |
| P17 | 5 | .7860 | .0540 | .3100 | .2200 | .8100 |
| P17 | 8 | .7613 | .0400 | .2913 | .2400 | .8800 |
| P18 | 1 | .8300 | .1900 | .3900 | .1900 | .3900 |
| P18 | 2 | .8050 | .1500 | .4100 | .2700 | .6400 |
| P18 | 4 | .7750 | .1025 | .3500 | .3300 | .7600 |
| P18 | 5 | .7840 | .0940 | .3500 | .3600 | .8100 |
| P18 | 8 | .7913 | .0800 | .3438 | .4100 | .9200 |

P19 contains only 100 conditions and eight outputs per condition, so it is a protocol
diagnostic rather than a matched 500-output-per-task reproduction.

## Nibi provenance

- P19 sweep job `20469228`: completed, exit `0:0`, 1m39s, CPU node c252.
- D3 supervised sweep job `20469226`: completed, exit `0:0`, 9m08s, CPU node c246.
- D3 + GRPO sweep job `20469227`: completed, exit `0:0`, 10m21s, CPU node c252.
- Pending short-walltime racers `20469243`--`20469245` were cancelled after the main
  jobs acquired nodes; they wrote no benchmark results.
- Full machine-readable comparison:
  `outputs/p22_moledit_table1_protocol_sweep/P22_RESULT.json` on Nibi.

No model training, molecule generation, target access, selection, reranking,
deduplication, repair, or threshold tuning was performed by P22.
