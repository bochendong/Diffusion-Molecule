# P2: Syntax-Safe De Novo + Source-Anchored Edit Repair

P2 follows the P1 single-seed signal with one paired repair gate for the two
failure modes that the raw-candidate audit exposed.

- De novo uses the strongest frozen Group-RL checkpoints. It compares legacy
  decoding with a syntax-safe autoregressive decoder that masks locally
  impossible SMILES continuations, permits repeated chemistry motifs, and
  uses the historically successful conservative OOD temperature/top-k profile.
- Edit reuses the protected common-decoder GraphEditDSL validation pool from
  P1. The property program states what should change; fingerprint similarity,
  scaffold retention, and edit magnitude preserve the source frame. No target
  molecule or output property oracle enters raw ranking.

The de novo subset contains 64 conditions for every 2p--7p count and 100 for
every OOD bucket. Both decode arms use the same checkpoint, rows, seed, and
candidate budgets `k=1,8,20`. Property scoring is retained only to write
per-candidate strict diagnostics; the property-reranked selected CSV is never
used as one-shot output.

The gate requires at least +20 percentage points raw de novo validity on both
benchmarks without losing more than one point of raw strict success or two
points of pass@8. The edit gate requires +2 points raw Acc@0.65 or +3 points
raw Acc@0.15 over the protected policy ranking.

On Nibi, while the P1 edit job is queued or running:

```bash
P2_EDIT_DEPENDENCY_JOB=<p1-job-id> \
  bash SketchMol-Understanding-Condition/experiments/p2_validity_edit_repair/submit_p2_validity_edit_repair.sh
```

The final report is written to
`outputs/p2_validity_edit_repair_seed7/final/p2_report.md`.
