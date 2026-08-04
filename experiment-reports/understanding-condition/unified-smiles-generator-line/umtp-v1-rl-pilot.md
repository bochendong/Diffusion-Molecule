# UMTP v1 Source-aware RL Pilot

| Field | Value |
| --- | --- |
| Status | completed; automatic decision **stop** |
| Nibi job | `19073785` (`COMPLETED`, exit `0:0`) |
| Runtime | **9m25s** |
| Resource | one H100 20 GB MIG, 8 CPU, 48 GB RAM |
| Code | `fb25a2d` |
| Warm-start | UMTP seed-7 search-distilled policy |

## Question

Can one short source-aware GRPO stage improve the current common molecular
language model's raw strict editing ability without hurting held-out de novo
generation?

This is a validation-only go/no-go experiment. Formal Table1 and OOD test rows
were not used for training, checkpoint selection, or the stop decision.

## Protocol

- Train: 384 rows, balanced across 6 de novo groups and 10 edit tasks.
- RL: one epoch, 8 rollouts per prompt, clipped GRPO, `lr=1e-6`.
- Edit reward: property success/distance plus continuous source similarity at
  threshold `0.65`; source-copy penalty remains enabled.
- Retention: SFT weight `0.5` plus frozen-reference penalty `0.1` on the same
  common checkpoint.
- Paired validation: 200 edit rows and 120 de novo rows, fixed eval seed `919`.
- Candidate protocol: one shared `n=8` pool per row; report `n=1,8` with both
  raw and finalizer selection.

## Paired results

| Variant | Task | n | Selection | Validity | Strict / Acc@0.65 | Acc@0.15 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| baseline | edit | 1 | raw | 23.50% | 0% | 10.00% |
| RL | edit | 1 | raw | 25.00% | 0% | 12.50% |
| baseline | edit | 8 | finalizer | 78.50% | 0% | 39.50% |
| RL | edit | 8 | finalizer | 71.50% | 0% | 36.00% |
| baseline | de novo retention | 1 | raw | 24.17% | 5.83% | — |
| RL | de novo retention | 1 | raw | 25.00% | 7.50% | — |
| baseline | de novo retention | 8 | finalizer | 86.67% | 35.83% | — |
| RL | de novo retention | 8 | finalizer | 89.17% | 41.67% | — |

Automatic decision inputs:

- edit raw `n=1` Acc@0.65 delta: **0.0pp**;
- edit raw `n=1` Acc@0.15 delta: **+2.5pp**;
- de novo raw `n=1` strict delta: **+1.67pp**.

The edit gain did not reach the predeclared `+2pp` strict or `+5pp` relaxed
threshold, so the collector returned **stop**.

## Reachability diagnostic

RL slightly changed the first sample, but did not put source-similar molecules
inside the policy's reachable support:

| Diagnostic | Baseline | RL |
| --- | ---: | ---: |
| raw valid | 23.5% | 25.0% |
| raw mean source Tanimoto among scored rows | 0.1748 | 0.1771 |
| raw source Tanimoto >=0.15 | 15.5% | 19.0% |
| raw source Tanimoto >=0.40 | 0% | 0% |
| raw source Tanimoto >=0.65 | 0% | 0% |
| any of n=8 reaches Tanimoto >=0.40 | 0.5% | 0.5% |
| any of n=8 reaches Tanimoto >=0.65 | 0% | 0% |

Training completed normally (`96` batches). The validation reward remained much
lower for edit than de novo (`-1.108` vs `-0.388`), consistent with an action
space/support problem rather than catastrophic forgetting.

## Decision

Do not scale this reward-only GRPO configuration to full H100 runs. The pilot
provides no strict editing gain, while the small-candidate finalizer result
worsens. It does provide positive evidence that the current UMTP checkpoint can
take a small RL update without immediate de novo forgetting.

The next main-line experiment should change how the same common model expresses
an edit—structured copy/graph-edit actions with a deterministic executor—rather
than increasing the similarity-reward weight again.

## Audit artifacts

- Output root: `outputs/unified_molecular_transformation_policy_rl_pilot_v1/seed_7/`
- Base checkpoint SHA-256: `618ef304d7286782056252befe793a0c0b8f2c88dbe7ae303e0ce1cb0361e9f4`
- RL checkpoint SHA-256: `36b97b3d7bd0bea370306cc43c9ef29df8e7f76e04307a6e8582b2b041c7c7f2`
- Train-pool SHA-256: `534a9741467ac7d5e4c0d324c91d2e410176c785aae5eddc885a576d3cf9cad1`
- Edit-validation SHA-256: `93595de1b3672563acc8b5f8bf0786e53afbab34dac8ae2dc7a82fa21df5b4ca`
- De-novo-validation SHA-256: `82e1edb50f29a08aa190507137362a60090c07ea7a9d7f89af9e23227cb549ca`
