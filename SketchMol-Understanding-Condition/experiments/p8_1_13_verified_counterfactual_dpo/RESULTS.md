# P8.1.13 paired results (seed 7)

## Execution and preference audit

After the zero-byte completion-marker check was repaired in `f55ac63`, the
successful chain was PRE `20393441` -> R1 `20393442` -> R2 `20393443`.  All
three jobs completed with exit code 0.  PRE formed 277 same-prompt preference
pairs from 280 upstream verified positives.  It enumerated 67,444 candidates,
including 50,960 valid strict-failure negatives, and found no evaluation ID or
molecule overlap.

## Formal raw-order evaluation

All values are fractions.  Candidate metrics use every emitted candidate
(3,999 in R1 and 3,994 in R2); condition metrics use the first `k` candidates
in generation order.  No property reranking is used.

| Metric | R1: uniform DPO | R2: confidence-weighted DPO | R2 - R1 |
|---|---:|---:|---:|
| De novo validity@1 | 0.40625 | 0.40625 | 0.00000 |
| De novo validity@8 | 0.38477 | 0.38477 | 0.00000 |
| De novo validity@20 | 0.40703 | 0.40703 | 0.00000 |
| De novo pass@1 | 0.03125 | 0.03125 | 0.00000 |
| De novo pass@8 | 0.26563 | 0.26563 | 0.00000 |
| De novo pass@20 | 0.40625 | 0.40625 | 0.00000 |
| De novo raw success@1 | 0.03125 | 0.03125 | 0.00000 |
| De novo raw success@8 | 0.03711 | 0.03711 | 0.00000 |
| De novo raw success@20 | 0.03828 | 0.03828 | 0.00000 |
| Edit validity@1 | 0.06500 | 0.17000 | +0.10500 |
| Edit validity@8 | 0.35000 | 0.63000 | +0.28000 |
| Edit validity@20 | 0.64000 | 0.84500 | +0.20500 |
| Edit candidate validity | 0.05151 | 0.12143 | +0.06992 |
| Strict Any@1 (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Strict Any@8 (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Strict Any@20 (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Strict candidate success (0.65) | 0.00000 | 0.00000 | 0.00000 |
| Relaxed Any@1 (0.15) | 0.00000 | 0.00000 | 0.00000 |
| Relaxed Any@8 (0.15) | 0.00000 | 0.00000 | 0.00000 |
| Relaxed Any@20 (0.15) | 0.00000 | 0.00000 | 0.00000 |
| Relaxed candidate success (0.15) | 0.00000 | 0.00000 | 0.00000 |
| Identity fraction | 0.00000 | 0.00000 | 0.00000 |
| Unique candidate fraction | 0.04926 | 0.09740 | +0.04813 |
| Raw candidate source Tanimoto | 0.07878 | 0.07280 | -0.00598 |

## Scientific diagnosis

The preference construction is not support-limited: almost every upstream
verified positive has a same-prompt valid strict-failure negative.  During
training, the fraction of pairs on which the policy scores the positive above
the negative reaches 0.652 in R1 and 0.664 in R2.  This likelihood separation
does not survive autoregressive sampling.  Confidence weighting more than
doubles candidate validity and substantially raises Any@k validity, while mean
source similarity decreases and every relaxed and strict success metric stays
at zero.

The result isolates a sequence-level objective/generation mismatch.  On this
full-SMILES source-adapter policy, relative likelihood of a verified positive
against a hard negative is insufficient to assign token-level credit for source
retention.  The audit otherwise passes: only source-conditioned parameters
change; the de novo path is bitwise protected; vocabulary and model
configuration remain fixed; and inference uses one full-SMILES checkpoint and
decoder without a router, materializer, property reranker, or teacher.
