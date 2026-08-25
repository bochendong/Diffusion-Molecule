# P23 fast gate results

## Aligned 24k follow-up

Status: repaired preparation chain submitted on Nibi; no result is claimed until
the manifest, adapter, and matched evaluation are frozen.

- output: `outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned`;
- first data preparation: job `20496020`, failed before writing data because
  held-out exclusion left 1,999 rather than 2,000 eligible 2p rows;
- first dependent SFT/contrastive jobs `20496057`/`20496086`: automatically
  canceled without running;
- repaired data preparation: job `20500215`;
- repaired positive SFT: job `20500236`, dependent on successful preparation;
- repaired contrastive refinement: job `20500251`, dependent on successful SFT;
- positives: 12,000 de novo plus 12,000 edit;
- de novo: 2,000 rows per 2p--7p count;
- de novo repair: one seeded, heldout-disjoint candidate-pool donor is projected
  from 7p to an MW+QED 2p program; the excluded benchmark-overlap row remains
  excluded;
- edit: ten exact train-only oracle-verified paper-task quotas of 720 rows plus
  4,800 broad explicit-instruction rows;
- fail-closed gate: any missing paper quota, held-out source/target overlap, or
  missing pinned assay oracle stops the chain before GPU training.

The aligned follow-up is an amended Stage-1 data protocol, not a GRPO run. Its
paper cells remain dashes while this status is pending.

Status: completed fast Stage-1 v2 gate on Nibi, seed `2323_fast`.

## Frozen adapter

- positive SFT job: `20473611`, `COMPLETED 0:0`, 6000 rows, 0.5 epoch,
  loss 0.6293, non-finite parameters 0;
- contrastive job: `20473612`, `COMPLETED 0:0`, 19,643 logical pairs,
  0.15 epoch, loss 0.1900, non-finite parameters 0;
- adapter:
  `/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/p23_explicit_task_stage1_v2/seed_2323_fast/model/stage1_v2/adapter`;
- adapter SHA-256:
  `fdb93c510ba29c68ec2fa7b0013bc9ecaf656104a3097457e6ecace8e1e29852`.

The selected positives are 3000 de-novo plus 3000 edit rows. De-novo has 500
rows per 2p--7p count. Canonical held-out source and target overlap is zero.

## Corrected-prompt edit gate

Reference: the same frozen P19 100-row, 10-task edit subset. P23 prompts were
rebuilt with `p23_protocol.py`; the historical P19 prompt JSONL was not reused.
All numbers below are raw candidate-level K=1 macro averages.

| Model | Validity | Acc@0.65 | Acc@0.15 | Property success | Mean source similarity |
|---|---:|---:|---:|---:|---:|
| P18 historical | 83% | 18% | 35% | 36% | 0.553 |
| P23 fast | **86%** | **25%** | **46%** | **46%** | **0.620** |
| Delta | +3 pp | +7 pp | +11 pp | +10 pp | +0.067 |

Strict Acc@0.65 by task changed as follows:

| Task | P18 | P23 | Delta |
|---|---:|---:|---:|
| GSK3B increase | 0% | 0% | 0 pp |
| RB decrease | 0% | 30% | +30 pp |
| MW increase | 40% | 50% | +10 pp |
| SA decrease | 20% | 20% | 0 pp |
| HBA decrease + SA decrease | 30% | 50% | +20 pp |
| QED increase + SA decrease | 20% | 20% | 0 pp |
| HBA decrease + LogP increase | 20% | 10% | -10 pp |
| HBA decrease + MW decrease | 20% | 40% | +20 pp |
| DRD2 decrease + MW decrease + SA decrease | 10% | 10% | 0 pp |
| HBA increase + MW increase + QED decrease | 20% | 20% | 0 pp |

Any@8 strict is 49%, but it is diagnostic only and must not replace K=1 in the
paper table. The paired change combines the corrected prompt contract and P23
training; it is not an estimate of training-only causality.

## Corrected-prompt de-novo gate

On the same frozen P20 300-row 2p--4p reference, raw K=1 is:

| Model | Validity | 2p strict | 3p strict | 4p strict |
|---|---:|---:|---:|---:|
| P18 | 100% | 11% | 2% | 2% |
| P23 fast | 98% | 12% | 5% | 2% |
| Delta | -2 pp | +1 pp | +3 pp | 0 pp |

At K=4, P23 reaches 45% / 21% / 8% for 2p / 3p / 4p versus P18's
33% / 13% / 8%, but candidate validity is 91.4%. At K=8, P23 reaches
62% / 30% / 12% versus 50% / 29% / 15%, with 89.2% candidate validity.

On the frozen P19 20+20 hard 6p/7p reference, K=1 strict remains 0% for both
counts. P23 raises 6p pass@4 and pass@8 from 5% to 10%, but 7p remains 0%.

## Decision

P23 is a usable checkpoint and a positive edit result, but it is not enough to
fill the entire de-novo table competitively. The next data iteration should retain
this explicit-task protocol, add direct GSK3B positives and missing paper-task
families, and add clause-level de-novo hard negatives focused on 4p--7p. Do not
run another GRPO stage until that supervised gate improves raw K=1.
