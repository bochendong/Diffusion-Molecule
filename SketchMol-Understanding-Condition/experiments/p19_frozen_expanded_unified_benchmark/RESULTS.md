# P19 frozen expanded unified benchmark results

## Decision

The expanded paired estimate **supports P18 as the current unified edit checkpoint**, but
it does not support a hard de-novo improvement claim. On 100 MolEdit rows, P18 improves
strict Acc@0.65 by `+0.10` at K=4 and `+0.15` at K=8; both preregistered paired-bootstrap
95% intervals exclude zero. The apparent two-row DRD2 collapse is budget-dependent:
P18 remains worse at K=1/K=4 but matches P17 at K=8 on the expanded `n=10` stratum.

On 40 hard de-novo conditions, P18 and P17 each obtain one six-property success by K=8,
neither succeeds on seven-property conditions, and P18's overall mean-best property
fraction is lower. These are expanded paired pilot estimates, not full benchmarks.

## Frozen protocol and audit

- no training, parameter update, threshold change, or benchmark-driven tuning
- frozen P17 adapter SHA256:
  `5944bbc44bf74735d6acb40ba8cd42ec8af7aeadb83c4ee468670dc12c7b448e`
- frozen P18 adapter SHA256:
  `7b3e1736bac49b7b2e35eceeed11fde199e0403a948e4dae08fd4fb9b89a0827`
- Table1: 100 rows, exactly ten rows from each of ten task keys; all original P17
  20-row pilot conditions retained
- hard de-novo: 20 six-property + 20 seven-property conditions; all original P17
  20-row pilot conditions retained
- training source/target overlap: `0 / 0`; four Table1 source-overlap pool rows excluded
- frozen reference/prompt hashes:
  - Table1 reference: `dd6f6d4145247eb90891c05c434b95b4b511abb553192cb850901f2a514cf6a7`
  - Table1 prompts: `8ec67319d6e05acaf9cae1b4daafcff0fe3c4c47ac16f86f71a8bddc475d06b6`
  - de-novo reference: `e840ab367e447f63f3b94b7208f173b67549d2cde6f69ac5122d6f7c99d6c650`
  - de-novo prompts: `3f5ef86a957b42048dfd0537878c3202ba9b69c1c14fb409eef340988e670bc6`
- byte-identical prompts and matching seeds for P17/P18; one greedy candidate followed
  by seven sampled candidates; K=1/4/8 uses the raw candidate prefix
- no target access, property reranking, or static candidate pool
- preregistered uncertainty: task-stratified paired condition bootstrap, seed `1919`,
  10,000 replicates, 95% percentile interval; DRD2 uses 95% Wilson intervals at exact
  `n=10` per model

## Reused frozen expanded validation

These P16/P17/P18 ID/OOD values are read from the existing frozen P18 validation
summary; P19 did not rerun or tune against them. The original P18 operational gate
remains 9/9 passed.

### ID condition, source-isolated

| Metric | P16 | P17 | P18 | P18 - P17 |
| --- | ---: | ---: | ---: | ---: |
| de-novo chosen NLL | 0.9683 | 0.9551 | **0.9539** | -0.0012 |
| de-novo greedy validity | 0.9375 | 1.0000 | **1.0000** | 0.0000 |
| de-novo Any@3 validity | 1.0000 | 1.0000 | **1.0000** | 0.0000 |
| edit chosen NLL | 0.4820 | 0.4788 | **0.4321** | -0.0468 |
| edit greedy validity | 0.9688 | 0.8125 | **0.8750** | +0.0625 |
| edit greedy noncopy | 0.5625 | 0.8125 | **0.8750** | +0.0625 |
| edit Any@3 validity | 0.9688 | 0.9375 | **0.9688** | +0.0313 |
| edit Any@3 noncopy | 0.9375 | 0.9375 | **0.9688** | +0.0313 |

### Strict condition-family and source OOD

| Metric | P16 | P17 | P18 | P18 - P17 |
| --- | ---: | ---: | ---: | ---: |
| de-novo chosen NLL | 0.9264 | 0.9139 | **0.9033** | -0.0106 |
| de-novo greedy / Any@3 validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 | **1.0000 / 1.0000** | 0.0000 / 0.0000 |
| edit chosen NLL | 0.5623 | 0.5849 | **0.5420** | -0.0429 |
| edit greedy validity | 0.9375 | 0.8125 | **0.8438** | +0.0313 |
| edit greedy noncopy | 0.5625 | 0.8125 | **0.8438** | +0.0313 |
| edit Any@3 validity / noncopy | 1.0000 / 1.0000 | 0.9688 / 0.9688 | **0.9688 / 0.9688** | 0.0000 / 0.0000 |

## MolEdit Table1 expanded paired estimate

### Macro across ten tasks

| K | Model | Validity | Property Any@K | Acc@0.15 | Strict Acc@0.65 | Best source sim |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | P17 | 0.840 | 0.390 | 0.360 | 0.170 | 0.513 |
| 1 | P18 | 0.830 | 0.360 | 0.350 | **0.180** | **0.553** |
| 1 | P18 - P17 | -0.010 | -0.030 | -0.010 | +0.010 | +0.040 |
| 4 | P17 | 1.000 | **0.750** | 0.680 | 0.190 | 0.625 |
| 4 | P18 | 1.000 | 0.710 | 0.680 | **0.290** | **0.692** |
| 4 | P18 - P17 | 0.000 | -0.040 | 0.000 | **+0.100** | **+0.067** |
| 8 | P17 | 1.000 | 0.840 | 0.790 | 0.210 | 0.671 |
| 8 | P18 | 1.000 | 0.840 | **0.820** | **0.360** | **0.731** |
| 8 | P18 - P17 | 0.000 | 0.000 | +0.030 | **+0.150** | **+0.060** |

### Paired-bootstrap P18 - P17 uncertainty

| K | Metric | Delta | 95% percentile CI |
| ---: | --- | ---: | ---: |
| 1 | Strict Acc@0.65 | +0.010 | [-0.060, +0.080] |
| 1 | Best source sim | +0.040 | [-0.021, +0.099] |
| 4 | Strict Acc@0.65 | **+0.100** | **[+0.020, +0.180]** |
| 4 | Best source sim | **+0.067** | **[+0.034, +0.101]** |
| 8 | Strict Acc@0.65 | **+0.150** | **[+0.070, +0.240]** |
| 8 | Best source sim | **+0.060** | **[+0.034, +0.085]** |
| 8 | Acc@0.15 | +0.030 | [-0.040, +0.100] |
| 8 | Property Any@K | 0.000 | [-0.060, +0.060] |

The expanded result agrees with the 20-row pilot on the main K=4/K=8 strict direction.
At K=8, the strict delta grows from `+0.10` in the small pilot to `+0.15` here. The
strict K=4 and K=8 intervals exclude zero; the K=1 difference does not.

### Strict Acc@0.65 by task (`n=10` each)

| Task | P17 K1 | P18 K1 | P17 K4 | P18 K4 | P17 K8 | P18 K8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DRD2 down + MW down + SA down | **0.4** | 0.1 | **0.4** | 0.2 | 0.4 | **0.4** |
| GSK3B up | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| HBA down + LogP up | 0.2 | 0.2 | 0.2 | **0.3** | 0.2 | **0.4** |
| HBA down + MW down | 0.2 | 0.2 | 0.2 | **0.3** | 0.2 | **0.3** |
| HBA down + SA down | 0.1 | **0.3** | 0.2 | **0.6** | 0.3 | **0.6** |
| HBA up + MW up + QED down | 0.2 | 0.2 | 0.2 | 0.2 | 0.3 | 0.3 |
| MW up | 0.3 | **0.4** | 0.4 | **0.7** | 0.4 | **0.9** |
| QED up + SA down | 0.2 | 0.2 | 0.2 | **0.3** | 0.2 | **0.3** |
| RB down | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.1** |
| SA down | 0.1 | **0.2** | 0.1 | **0.3** | 0.1 | **0.3** |

### DRD2 focused audit

Each estimate changes in steps of `0.1`.

| K | P17 strict, Wilson 95% | P18 strict, Wilson 95% | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.4 [0.168, 0.687] | 0.1 [0.018, 0.404] | -0.3 |
| 4 | 0.4 [0.168, 0.687] | 0.2 [0.057, 0.510] | -0.2 |
| 8 | 0.4 [0.168, 0.687] | 0.4 [0.168, 0.687] | **0.0** |

The original two-row P18 result (`0/2` versus P17 `2/2`) was too coarse to support an
all-budget collapse claim. The expanded result still shows weaker P18 greedy/small-K
behavior, but the deficit disappears at K=8. This is a real budget sensitivity rather
than either a clean regression or a clean recovery.

## Hard de-novo expanded paired estimate

“Raw/pass” is strict raw-candidate success fraction / condition Pass@K. Mean-best is
the best property-satisfaction fraction in the raw prefix and was not used for selection.

| Stratum | K | P17 raw / pass | P18 raw / pass | P17 valid / unique | P18 valid / unique | P17 mean-best | P18 mean-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1 | 0 / 0 | 0 / 0 | 1.000 / 1.000 | 1.000 / 1.000 | **0.369** | 0.356 |
| all | 4 | 0 / 0 | **0.0063 / 0.025** | 0.888 / 0.888 | 0.888 / 0.888 | **0.534** | 0.511 |
| all | 8 | 0.0031 / 0.025 | 0.0031 / 0.025 | 0.856 / 0.856 | **0.894 / 0.894** | **0.616** | 0.586 |
| 6p | 1 | 0 / 0 | 0 / 0 | 1.000 / 1.000 | 1.000 / 1.000 | 0.367 | **0.383** |
| 6p | 4 | 0 / 0 | **0.0125 / 0.050** | **0.900 / 0.900** | 0.888 / 0.888 | 0.525 | **0.550** |
| 6p | 8 | 0.0063 / 0.050 | 0.0063 / 0.050 | 0.875 / 0.875 | **0.894 / 0.894** | **0.625** | 0.600 |
| 7p | 1 | 0 / 0 | 0 / 0 | 1.000 / 1.000 | 1.000 / 1.000 | **0.371** | 0.329 |
| 7p | 4 | 0 / 0 | 0 / 0 | 0.875 / 0.875 | **0.888 / 0.888** | **0.543** | 0.471 |
| 7p | 8 | 0 / 0 | 0 / 0 | 0.838 / 0.838 | **0.894 / 0.894** | **0.607** | 0.571 |

The 20-row pilot's P18 hard de-novo advantage does not persist as an overall effect.
P18 gets a six-property success earlier at K=4, but by K=8 both models pass exactly
one of 20 six-property conditions. Neither passes any seven-property condition. P18
has better K=8 validity/uniqueness, while its all-row mean-best property fraction is
lower by `0.030` (`0.586` versus `0.616`).

## Operational history

- prepare/freeze job `20459245`: `COMPLETED 0:0`, 12s; Nibi CPU tests `3 passed`
- P17 generation job `20459246`: `COMPLETED 0:0`, 11m19s, H100 g35
- P18 generation job `20459247`: both raw files completed, but Slurm ended `FAILED 1:0`
  after 8m36s because the old runner used physical `wc -l` on CSV. One raw response
  contained an embedded newline, producing 802 physical lines while the CSV correctly
  contained 800 Table1 records. This was a post-generation false failure, not a model,
  adapter, or generation failure.
- no P18 GPU rerun: the CPU CSV-aware validator confirmed exact `100 x 8` Table1 and
  `40 x 8` de-novo records, ranks 1..8, expected labels, and both adapter hashes
- final evaluation/aggregation job `20461088`: `COMPLETED 0:0`, 1m21s
- short scheduling racers were cancelled after winners started; the one 20GB P18 racer
  exited before generation because its updated audit expected a not-yet-materialized
  copied lock file; its output directory was empty and it caused no corruption
- final `EXPANDED_ESTIMATE.json` reports all four subset hashes verified

No files were committed or pushed.
