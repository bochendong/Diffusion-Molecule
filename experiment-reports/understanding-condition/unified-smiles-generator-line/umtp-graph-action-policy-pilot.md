# UMTP Common-Decoder GraphEditDSL Pilot

Date: 2026-08-05

## Outcome

The protected common-decoder GraphEditDSL pilot passed the preregistered go/no-go gate.

- The same `ConditionedSmilesDecoder` emits SMILES for de novo design and scores executable GraphEditDSL programs for source-conditioned editing.
- Edit candidates are grammar constrained and executed deterministically with RDKit.
- The legacy de novo path is frozen; only source-conditioned modules and appended action-token embeddings are trained.
- Normal SMILES sampling masks edit-only tokens.

This removes the two failure modes seen in the previous reward-only pilot: strict edit candidates are now reachable, and de novo retention no longer drops.

## Runs

| Run | Job | GPU | Elapsed | Result |
| --- | ---: | --- | ---: | --- |
| Full shared-parameter action SFT | `19078642` | 1x H100 20 GB MIG | 14m36s | stop: edit improved, de novo regressed |
| Protected source/action SFT | `19079748` | 1x H100 20 GB MIG | 6m20s | **go** |

Both jobs completed with exit code `0:0`.

## Action-space oracle

The audited training pilot contained 480 edit rows and 288 de novo replay rows.

| Metric | Value |
| --- | ---: |
| Executable edit coverage | 100.0% |
| Mean best target Tanimoto | 0.7016 |
| Best target Tanimoto >= 0.65 | 90.2% |
| Mean best source Tanimoto | 0.8305 |
| Best source Tanimoto >= 0.65 | 100.0% |
| Exact one-step target reconstruction | 0.0% |

The zero exact-reconstruction rate is expected: the current DSL is a one-step local policy, while the paired MolEdit target can contain a larger transformation. The important result is that the strict source-similarity region is now reachable for nearly all rows.

## Paired validation result

The evaluation uses the same 200 Table1 validation rows and 120 held-out de novo validation rows for baseline and action variants.

| Variant | Task | n | Selection | Validity | Acc@0.65 / strict | Acc@0.15 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| UMTP baseline | Table1 | 1 | raw | 31.0% | 0.0% | 14.0% |
| Protected action policy | Table1 | 1 | raw | **100.0%** | **20.0%** | **20.0%** |
| UMTP baseline | Table1 | 8 | finalizer | 75.5% | 0.0% | 43.5% |
| Protected action policy | Table1 | 8 | finalizer | **100.0%** | **43.5%** | **43.5%** |
| UMTP baseline | de novo retention | 1 | raw | 27.5% | 7.5% | - |
| Protected action policy | de novo retention | 1 | raw | 27.5% | 7.5% | - |
| UMTP baseline | de novo retention | 8 | finalizer | 83.3% | 33.3% | - |
| Protected action policy | de novo retention | 8 | finalizer | **90.0%** | **35.8%** | - |

Paired raw n=1 deltas:

- Table1 Acc@0.65: **+20.0 percentage points**
- Table1 Acc@0.15: **+6.0 percentage points**
- Held-out de novo strict success: **0.0 percentage points**

The unprotected action SFT reached 17.5% raw Acc@0.65 but lost 5 points of de novo strict success. Freezing the legacy path improved raw Acc@0.65 to 20.0% and exactly restored retention.

## Table1 task detail: protected n=8 finalizer

| Task | Acc@0.65 |
| --- | ---: |
| GSK3B up | 0% |
| Rotatable bonds down | 0% |
| MW up | 90% |
| SA down | 15% |
| HBA down + SA down | 15% |
| QED up + SA down | 30% |
| HBA down + LogP up | 45% |
| HBA down + MW down | 70% |
| DRD2 down + MW down + SA down | 75% |
| HBA up + MW up + QED down | 95% |

GSK3B remains the clearest unsupported task. The rotatable-bond result also shows that the current generic finalizer is not instruction-faithful: raw n=1 reaches 15%, but finalizer n=8 selects 0%. The next full evaluation should use a Table1-instruction-aware finalizer.

## Decision

**Go with the protected common-decoder action policy as the main edit direction.**

The next paper-facing run should keep the architecture fixed and scale evaluation, not reopen reward-weight tuning:

1. Add an instruction-task-aware finalizer and rerun the same frozen checkpoint at shared candidate budgets `n=1,8,64,256`.
2. Evaluate the full Table1 test split and at least three evaluation seeds; retain the frozen de novo checkpoint comparison.
3. Extend the same DSL to two-step programs only after the one-step full test is recorded.
4. Treat GSK3B as a separate oracle/action-prior problem; do not contaminate the common local-property policy with an ad hoc reward patch.

## Artifacts

- Protected result root: `/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/umtp_graph_action_protected_pilot_v1/seed_7`
- Protected report: `/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/umtp_graph_action_protected_pilot_v1/seed_7/umtp_graph_action_pilot_report.md`
- Protected checkpoint: `/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/umtp_graph_action_protected_pilot_v1/seed_7/action_policy/umtp_graph_action_policy.pt`
- First-pass stop root: `/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/outputs/umtp_graph_action_pilot_v1/seed_7`
- Logs: `/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/umtp_graph_action_pilot_v1/umtp-action-s7-19078642.log` and `umtp-action-s7-19079748.log`
