# P18 results: validity-aware multi-negative unified continuation

## Decision

**`freeze_and_pilot` — all preregistered operational-gate checks passed.**

P18 recovers part of P17's edit-validity loss while retaining and slightly improving
its noncopy behavior. On the frozen ID view, edit greedy validity rises from P17's
`0.8125` to `0.8750` and edit noncopy rises from `0.8125` to `0.8750`. Relative to
P16, validity is down `0.09375`, inside the frozen `-0.10` tolerance. De-novo greedy
and Any@3 validity are both `1.0`; edit Any@3 validity is `0.96875`. The gate passes
exactly at the absolute edit-validity threshold (`28/32`), so this is a positive but
small and still fragile pilot result.

## Unified model, data, and run audit

- one cached Qwen2.5-VL-7B base, one tokenizer, and one LoRA adapter initialized from
  frozen P16 mixed; unchanged prompt/schema, no router or mode-specific head
- exact frozen P17 160+160 train rows, 32+32 ID rows, 32+32 strict-family-OOD rows,
  20-row Table1 subset, and 20-row hard de-novo subset; all seven input hashes match
- 320 unique train rows expanded into 800 logical pairs: 160 weak source-copy, 320
  invalid corruption, and 320 same-mode condition-mismatch negatives
- invalid audit: 320/320 remain strict JSON and 0/320 are RDKit-valid; mismatch audit:
  zero same-target and zero edit-source-copy collisions
- chosen CE total weight is balanced at 160 per mode; edit chosen weight is not
  increased; one rejected forward per step; this is contrastive SFT, not true ORPO
- no inference target access, static candidate pool, property reranking, or benchmark
  tuning; candidate budgets are raw generation-order prefixes
- prepare job `20455299`: `COMPLETED 0:0`, 16s; Nibi tests `6 passed`
- winner train job `20456952`: `COMPLETED 0:0`, 7m12s, H100 g18; train loss
  `0.269382`, finite adapter
- final validation/pilot job `20456953`: `COMPLETED 0:0`, 19m21s, H100 g2
- frozen adapter SHA256:
  `7b3e1736bac49b7b2e35eceeed11fde199e0403a948e4dae08fd4fb9b89a0827`

## Expanded P16 / P17 / P18 validation

### ID-condition, source-isolated gate view

| Mode / metric | P16 | P17 | P18 | P18 − P17 |
| --- | ---: | ---: | ---: | ---: |
| de-novo chosen NLL | 0.9683 | 0.9551 | **0.9539** | -0.0012 |
| de-novo greedy validity | 0.9375 | 1.0000 | **1.0000** | 0.0000 |
| de-novo Any@3 validity | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| edit chosen NLL | 0.4820 | 0.4788 | **0.4321** | -0.0468 |
| edit greedy validity | 0.9688 | 0.8125 | **0.8750** | +0.0625 |
| edit greedy noncopy | 0.5625 | 0.8125 | **0.8750** | +0.0625 |
| edit greedy source similarity | 0.8746 | 0.5764 | **0.6620** | +0.0855 |
| edit Any@3 validity | 0.9688 | 0.9375 | **0.9688** | +0.0313 |
| edit Any@3 noncopy | 0.9375 | 0.9375 | **0.9688** | +0.0313 |
| chosen preferred to source-copy | 0.0000 | 0.6875 | **0.5000** | -0.1875 |

All nine gate/audit checks pass. The lower chosen-versus-copy preference compared with
P17 is intentional: P18 restores a more moderate edit pressure instead of maximizing
distance from the source.

### Strict condition-family and source OOD diagnostic

All 64 OOD rows use unseen condition families; exact-condition, source, and target
overlap are zero.

| Mode / metric | P16 | P17 | P18 | P18 − P17 |
| --- | ---: | ---: | ---: | ---: |
| de-novo chosen NLL | 0.9264 | 0.9139 | **0.9033** | -0.0106 |
| de-novo greedy validity | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| de-novo Any@3 validity | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| edit chosen NLL | 0.5623 | 0.5849 | **0.5420** | -0.0429 |
| edit greedy validity | 0.9375 | 0.8125 | **0.8438** | +0.0313 |
| edit greedy noncopy | 0.5625 | 0.8125 | **0.8438** | +0.0313 |
| edit greedy source similarity | 0.8256 | 0.6082 | **0.6526** | +0.0444 |
| edit Any@3 validity | 1.0000 | 0.9688 | **0.9688** | 0.0000 |
| edit Any@3 noncopy | 1.0000 | 0.9688 | **0.9688** | 0.0000 |
| chosen preferred to source-copy | 0.0000 | 0.5938 | **0.4688** | -0.1250 |

## MolEdit Table1 paired pilot estimate

The same twenty frozen rows (two per task) and raw seeds are used for P17 and P18.
These values are pilot estimates, not a full Table1 benchmark.

### Macro across ten tasks

| K | Model | Validity | Property Any@K | Acc@0.15 | Strict Acc@0.65 | Best source sim |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | P17 | 0.900 | 0.500 | 0.450 | 0.300 | 0.646 |
| 1 | P18 | 0.850 | 0.250 | 0.250 | 0.200 | 0.715 |
| 1 | P18 − P17 | -0.050 | -0.250 | -0.200 | -0.100 | +0.069 |
| 4 | P17 | 1.000 | 0.700 | 0.550 | 0.300 | 0.690 |
| 4 | P18 | 1.000 | 0.650 | 0.650 | 0.350 | 0.742 |
| 4 | P18 − P17 | 0.000 | -0.050 | +0.100 | +0.050 | +0.052 |
| 8 | P17 | 1.000 | 0.750 | 0.700 | 0.350 | 0.703 |
| 8 | P18 | **1.000** | **0.750** | **0.750** | **0.450** | **0.805** |
| 8 | P18 − P17 | 0.000 | 0.000 | +0.050 | **+0.100** | +0.102 |

P18 is worse at greedy K=1 but better once a small raw sampling budget is available.
At K=8, strict Acc@0.65 improves from `0.35` to `0.45`, Acc@0.15 improves from
`0.70` to `0.75`, and Property Any@K is retained at `0.75`.

### Strict Acc@0.65 by task

Each value is based on two rows and therefore changes in increments of `0.5`.

| Task | P17 K1 | P18 K1 | P17 K4 | P18 K4 | P17 K8 | P18 K8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DRD2↓ + MW↓ + SA↓ | 1.0 | **0.0** | 1.0 | **0.0** | 1.0 | **0.0** |
| GSK3B↑ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| HBA↓ + LogP↑ | 0.5 | 0.5 | 0.5 | **1.0** | 1.0 | 1.0 |
| HBA↓ + MW↓ | 0.5 | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| HBA↓ + SA↓ | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| HBA↑ + MW↑ + QED↓ | 0.5 | 0.5 | 0.5 | **1.0** | 0.5 | **1.0** |
| MW↑ | 0.0 | **0.5** | 0.0 | **0.5** | 0.0 | **1.0** |
| QED↑ + SA↓ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.5** |
| RB↓ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| SA↓ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The macro gain hides a real task transfer: MW-focused and HBA/MW/QED tasks improve,
but the previously perfect DRD2↓ + MW↓ + SA↓ stratum falls from `1.0` to `0.0` at
all three budgets. That tradeoff must be retested on more than two rows per task before
claiming an overall Table1 improvement.

## Hard de-novo paired pilot estimate

The same ten 6-property and ten 7-property conditions are used. “Strict raw” is the
fraction of all raw candidates satisfying every condition; Pass@K is the fraction of
conditions with at least one strict success. Mean-best is a read-only diagnostic and
was not used for selection.

| Stratum | K | P17 strict / pass | P18 strict / pass | P17 valid / unique | P18 valid / unique | P17 mean-best | P18 mean-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1 | 0.000 / 0.000 | **0.050 / 0.050** | 1.000 / 1.000 | 1.000 / 1.000 | 0.324 | **0.379** |
| all | 4 | 0.000 / 0.000 | **0.0125 / 0.050** | 0.900 / 0.900 | 0.925 / 0.925 | 0.523 | **0.562** |
| all | 8 | 0.000 / 0.000 | **0.0063 / 0.050** | 0.869 / 0.869 | 0.900 / 0.900 | 0.631 | **0.657** |
| 6p | 1 | 0.000 / 0.000 | **0.100 / 0.100** | 1.000 / 1.000 | 1.000 / 1.000 | 0.333 | **0.400** |
| 6p | 4 | 0.000 / 0.000 | **0.025 / 0.100** | 0.900 / 0.900 | 0.925 / 0.925 | 0.517 | **0.567** |
| 6p | 8 | 0.000 / 0.000 | **0.0125 / 0.100** | 0.888 / 0.888 | 0.900 / 0.900 | 0.633 | **0.700** |
| 7p | 1 | 0.000 / 0.000 | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.314 | **0.357** |
| 7p | 4 | 0.000 / 0.000 | 0.000 / 0.000 | 0.900 / 0.900 | 0.925 / 0.925 | 0.529 | **0.557** |
| 7p | 8 | 0.000 / 0.000 | 0.000 / 0.000 | 0.850 / 0.850 | 0.900 / 0.900 | **0.629** | 0.614 |

P18 produces the first strict success in this paired pilot: one of ten 6-property
conditions succeeds, so 6p Pass@K is `0.10` at every budget versus `0.00` for P17.
No 7-property condition succeeds. Mean-best property satisfaction improves overall at
all budgets, although the 7-property K=8 diagnostic slips from `0.629` to `0.614`.

## Interpretation

The weaker copy loss plus invalid and condition-mismatch negatives is a better balance
than P17 for this small paired experiment: it passes the unified operational gate,
improves P17 edit validity and noncopy together, improves Table1 strict K=8 macro, and
finds a hard 6-property de-novo success. It does not yet justify a full-benchmark claim.
The next useful expansion is more rows per Table1 task, especially the lost DRD2 task,
and a larger 6p/7p set using this frozen checkpoint before any additional tuning.
