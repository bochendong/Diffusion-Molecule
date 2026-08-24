# P8.1.12 paired results (seed 7)

## Execution and support audit

The only submitted chain was PRE `20392190` -> R1 `20392191` -> R2
`20392192`.  All three jobs completed with exit code 0.  PRE selected 280
verified strict-success rows from 459 eligible train-only rows (61.00% row
coverage); property-count coverage was 96/191 (1p), 99/175 (2p), and 85/93
(3p).  The audit found no evaluation identifier, source, or pseudo-target
overlap.

## Formal raw-order evaluation

All values are fractions.  `Validity@k` is condition-level availability of a
valid molecule within the first `k` candidates.  Candidate metrics use all
4,000 raw candidates.  No property reranking is used.

| Metric | R1: uniform verified SFT | R2: confidence weighted | R2 - R1 |
|---|---:|---:|---:|
| De novo validity@1 | 0.53125 | 0.53125 | 0.00000 |
| De novo validity@8 | 0.40625 | 0.40625 | 0.00000 |
| De novo validity@20 | 0.40000 | 0.40000 | 0.00000 |
| De novo pass@1 | 0.06250 | 0.06250 | 0.00000 |
| De novo pass@8 | 0.18750 | 0.18750 | 0.00000 |
| De novo pass@20 | 0.34375 | 0.34375 | 0.00000 |
| Edit validity@1 | 0.28000 | 0.27500 | -0.00500 |
| Edit validity@8 | 0.85000 | 0.84500 | -0.00500 |
| Edit validity@20 | 0.93500 | 0.94500 | +0.01000 |
| Edit candidate validity | 0.30000 | 0.31625 | +0.01625 |
| Strict Any@1 (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Strict Any@8 (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Strict Any@20 (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Strict candidate success (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Relaxed Any@1 (0.15) | 0.03000 | 0.02000 | -0.01000 |
| Relaxed Any@8 (0.15) | 0.17500 | 0.18000 | +0.00500 |
| Relaxed Any@20 (0.15) | 0.29500 | 0.32500 | +0.03000 |
| Relaxed candidate success (0.15) | 0.02600 | 0.02825 | +0.00225 |
| Identity fraction | 0.00000 | 0.00000 | 0.00000 |
| Raw candidate source Tanimoto | 0.12940 | 0.12925 | -0.00015 |
| Best@20 macro source Tanimoto | 0.16602 | 0.16830 | +0.00227 |

## Scientific diagnosis

Verified teacher support is not the missing variable: 280 training rows contain
an officially verified strict-success outcome, including 91.40% coverage of the
3-property training rows.  Nevertheless, neither student reaches one strict
success among 4,000 candidates.  Both students have low raw source similarity
(about 0.129), while confidence weighting primarily improves validity and the
relaxed 0.15 threshold.  The evidence therefore indicates a representation and
credit-assignment bottleneck in full-SMILES source-conditioned decoding: teacher
confidence selects outcomes that are somewhat easier to decode, but does not
teach the student to preserve enough source structure for the 0.65 gate.

The protection contract holds in both rounds: only 38 source-conditioned
parameters change; the de novo parameter path is bitwise frozen; vocabulary and
model configuration are unchanged; and inference uses one full-SMILES checkpoint
and output head with no router, materializer, property reranker, or teacher.
