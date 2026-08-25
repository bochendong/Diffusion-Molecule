# P17 results

## Decision

**`freeze_negative_diagnostic_pilot`**

Copy-contrastive continuation did what it was designed to do: on both development
views, greedy edit noncopy rose from 0.5625 to 0.8125 and the model changed from always
preferring the source-copy completion to preferring the chosen edit on most rows.
De-novo validity and NLL did not regress. However, ID edit greedy validity fell from
0.96875 to 0.8125, a delta of `-0.15625`, just below the preregistered relaxed bound of
`-0.15`. The threshold was not changed. The checkpoint was frozen and the explicitly
requested benchmark run is therefore reported only as a **negative-diagnostic pilot
estimate**, not as a full Table1 or full de-novo result.

## Unified model and run audit

- one cached Qwen2.5-VL-7B base, one tokenizer, one continued LoRA adapter
- unchanged P16 prompt and JSON/SMILES schema; no router or mode-specific head
- 160 de-novo rehearsal rows + 160 edit chosen/source-copy pairs
- chosen completion CE + edit-only pairwise margin (`0.20`, weight `0.35`)
- no DPO/ORPO, inference target access, static pool, or property reranking
- training job `20450242`: `COMPLETED 0:0`, 3m38s, 40GB H100 MIG, finite adapter
- P16 baseline job `20450241`: `COMPLETED 0:0`, 12m41s
- final validation/pilot job `20450243`: `COMPLETED 0:0`, 15m34s
- train loss `0.745593`; frozen adapter SHA256
  `5944bbc44bf74735d6acb40ba8cd42ec8af7aeadb83c4ee468670dc12c7b448e`

The two expanded views each contain 32 de-novo and 32 edit rows. Train/development
source and target overlap is zero. The OOD view contains 64/64 strict unseen
condition-family rows; exact-condition, source, and target overlap are all zero.

## Expanded P16 versus P17 validation

### ID-condition, source-isolated gate view

| Mode / metric | Frozen P16 | Frozen P17 | Delta |
| --- | ---: | ---: | ---: |
| de-novo chosen NLL | 0.9683 | 0.9551 | -0.0132 |
| de-novo greedy validity | 0.9375 | 1.0000 | +0.0625 |
| de-novo Any@3 validity | 1.0000 | 1.0000 | 0.0000 |
| edit chosen NLL | 0.4820 | 0.4788 | -0.0031 |
| edit greedy validity | 0.9688 | 0.8125 | **-0.1563** |
| edit greedy noncopy | 0.5625 | 0.8125 | +0.2500 |
| edit greedy source similarity | 0.8746 | 0.5764 | -0.2981 |
| edit Any@3 validity | 0.9688 | 0.9375 | -0.0313 |
| edit Any@3 noncopy | 0.9375 | 0.9375 | 0.0000 |
| chosen preferred to source-copy | 0.0000 | 0.6875 | +0.6875 |

### Strict condition-family + source OOD diagnostic

| Mode / metric | Frozen P16 | Frozen P17 | Delta |
| --- | ---: | ---: | ---: |
| de-novo chosen NLL | 0.9264 | 0.9139 | -0.0125 |
| de-novo greedy validity | 1.0000 | 1.0000 | 0.0000 |
| edit chosen NLL | 0.5623 | 0.5849 | +0.0226 |
| edit greedy validity | 0.9375 | 0.8125 | -0.1250 |
| edit greedy noncopy | 0.5625 | 0.8125 | +0.2500 |
| edit greedy source similarity | 0.8256 | 0.6082 | -0.2175 |
| edit Any@3 validity | 1.0000 | 0.9688 | -0.0313 |
| edit Any@3 noncopy | 1.0000 | 0.9688 | -0.0313 |
| chosen preferred to source-copy | 0.0000 | 0.5938 | +0.5938 |

This identifies a real tradeoff rather than a failed objective: source-copy preference
was reversed, but the current margin weight moves too much probability toward malformed
or noncanonical edits.

## MolEdit Table1 pilot estimate

The frozen pilot uses two deterministic, leak-free rows from each of ten Table1 task
strata (20 rows total). Candidate 0 is greedy and candidates 1-7 are one raw sampling
pass. Budgets 1/4/8 are prefixes in generation order. Four source-overlapping rows were
removed from the sampling pool before deterministic selection. Condition overlap is
reported but was not used to tune the checkpoint.

### Macro pilot estimate across ten tasks

| Raw budget | Validity | Strict Acc@0.65 | Acc@0.15 | Property Any@K | Best source similarity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.900 | 0.300 | 0.450 | 0.500 | 0.646 |
| 4 | 1.000 | 0.300 | 0.550 | 0.700 | 0.690 |
| 8 | 1.000 | 0.350 | 0.700 | 0.750 | 0.703 |

### Strict Acc@0.65 by task

Each task estimate has only two rows, so values move in increments of 0.5.

| Table1 task | K=1 | K=4 | K=8 |
| --- | ---: | ---: | ---: |
| DRD2 decrease + MW decrease + SA decrease | 1.0 | 1.0 | 1.0 |
| GSK3B increase | 0.0 | 0.0 | 0.0 |
| HBA decrease + LogP increase | 0.5 | 0.5 | 1.0 |
| HBA decrease + MW decrease | 0.5 | 0.5 | 0.5 |
| HBA decrease + SA decrease | 0.5 | 0.5 | 0.5 |
| HBA increase + MW increase + QED decrease | 0.5 | 0.5 | 0.5 |
| MW increase | 0.0 | 0.0 | 0.0 |
| QED increase + SA decrease | 0.0 | 0.0 | 0.0 |
| RB decrease | 0.0 | 0.0 | 0.0 |
| SA decrease | 0.0 | 0.0 | 0.0 |

## Hard de-novo pilot estimate

The frozen pilot uses ten 6-property and ten 7-property conditions from P6's hard
gate. It produces valid and mostly unique molecules, but **no candidate satisfies all
six or seven requested properties** at any raw budget.

| Stratum | K | Strict raw success | Pass@K | Validity | Unique |
| --- | ---: | ---: | ---: | ---: | ---: |
| all (20) | 1 | 0.000 | 0.000 | 1.000 | 1.000 |
| all (20) | 4 | 0.000 | 0.000 | 0.900 | 0.900 |
| all (20) | 8 | 0.000 | 0.000 | 0.869 | 0.869 |
| 6-property (10) | 1 | 0.000 | 0.000 | 1.000 | 1.000 |
| 6-property (10) | 4 | 0.000 | 0.000 | 0.900 | 0.900 |
| 6-property (10) | 8 | 0.000 | 0.000 | 0.887 | 0.887 |
| 7-property (10) | 1 | 0.000 | 0.000 | 1.000 | 1.000 |
| 7-property (10) | 4 | 0.000 | 0.000 | 0.900 | 0.900 |
| 7-property (10) | 8 | 0.000 | 0.000 | 0.850 | 0.850 |

As a read-only diagnostic on the same frozen candidate prefixes, the mean best
fraction of requested properties satisfied rises from `0.324` at K=1 to `0.523` at
K=4 and `0.631` at K=8 (6-property: `0.333/0.517/0.633`; 7-property:
`0.314/0.529/0.629`). This was not used for selection or tuning: it shows partial
multi-property control despite zero all-constraint success.

## Interpretation

P17 is still a genuinely unified checkpoint and is better at avoiding identity edits,
but source-copy contrast alone is too blunt. The Table1 pilot shows useful edit signal,
especially on the DRD2/MW/SA task and at K=8, while the hard de-novo result shows that
valid SMILES generation is not the same as multi-property control. A next round should
retain a weaker copy margin and add validity-aware chosen/rejected construction or a
small syntax-validity auxiliary loss; it should not increase copy pressure further.
