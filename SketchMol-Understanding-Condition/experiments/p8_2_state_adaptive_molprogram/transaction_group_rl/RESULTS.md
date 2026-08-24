# P8.2 transaction group-relative RL results

Seed 7 uses the P8.1.1 transaction checkpoint as a shared start for both mandatory rounds. R1 uses joint-bottleneck reward aggregation; R2 changes only that aggregation to dense soft-min. Evaluation uses 20 candidates in generation order, no property reranking, and no target molecule at inference.

## Matched MolEdit Table1

All values are percentages. The matched P8.1.1 base is the same checkpoint evaluated at temperature 0.8, matching both RL rounds.

| Method | Validity @1/@8/@20 | Strict Any@1 | Strict Any@8 | Strict Any@20 | Relaxed Any@1 | Relaxed Any@8 | Relaxed Any@20 | Candidate strict | Candidate relaxed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8.1.1 matched base | 100/100/100 | 23.5 | 53.0 | 70.5 | 24.0 | 53.5 | 71.0 | 17.075 | 18.025 |
| R1 joint bottleneck | 100/100/100 | 18.0 | 46.0 | 61.5 | 19.5 | 46.5 | 62.0 | 16.750 | 17.775 |
| R2 dense soft-min | 100/100/100 | 18.0 | 45.5 | 62.0 | 19.5 | 46.0 | 62.5 | 16.775 | 17.800 |
| R2 minus R1 (pp) | 0/0/0 | 0.0 | -0.5 | +0.5 | 0.0 | -0.5 | +0.5 | +0.025 | +0.025 |
| R2 minus matched base (pp) | 0/0/0 | -5.5 | -7.5 | -8.5 | -4.5 | -7.5 | -8.5 | -0.300 | -0.225 |

Both rounds have 0% identity candidates and 100% candidate uniqueness. Thus the loss is not caused by identity-copy collapse or duplicate sampling: the update shifts probability away from successful source-conditioned transactions.

## Empty-source functional protection

Job `20397317` replays the unmodified P8.1.1 base checkpoint with exactly the same hard de novo rows, seed 1982, sampling arguments, and 20-candidate order used for R1 and R2.

| Check | Base | R1 | R2 | Result |
|---|---:|---:|---:|---|
| Candidate rows | 1280 | 1280 | 1280 | exact |
| Ordered candidate signature | `0517a422...` | `0517a422...` | `0517a422...` | exact |
| Metric records | same | same | same | exact |

The functional replay audit passes: source-only RL preserves empty-source de novo behavior exactly, not merely legacy parameter tensors. The earlier comparison against historical P8.1.1 candidates used seed 1907 rather than seed 1982 and is invalid; all performance differences inferred from that mismatched comparison are withdrawn.

## Diagnosis

Dense soft-min is a clean single-factor R2, but it is effectively neutral relative to R1 and does not recover the matched P8.1.1 edit baseline. The negative result is localized: architecture and masking successfully protect de novo generation, while group-relative source-transaction updates degrade Table1 Any@k despite unchanged validity, identity, and diversity. The next attempt should therefore change rollout support or credit assignment rather than further tune the reward aggregator.

Audits for both rounds pass: one checkpoint serves both arms; legacy de novo parameters are bit-exact; only source-memory or transaction-token parameters change; training uses no evaluation rows or paired targets; reward has no target-structure access; and inference applies no property reranking.
