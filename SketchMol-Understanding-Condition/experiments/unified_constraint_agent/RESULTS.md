# Unified Constraint Agent Results

## Common-LLM 1.5B method-signal pilot

Date: 2026-08-09

### Protocol

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adaptation: LoRA rank 16, alpha 32, one epoch
- Stable compute: FP32/TF32, learning rate `5e-5`
- Training rows after tokenization: 4,485
- Held-out rows: 144, separated during train-only data preparation
- Decoding: deterministic first action for this format/control diagnostic
- Training job: `19372638` (`COMPLETED`, 12 minutes 25 seconds)
- Evaluation job: `19373690` (`COMPLETED`, 3 minutes 1 second)

The saved adapter contains 18,464,768 parameters. Both the training gate and
an independent safetensors scan reported zero non-finite parameters.

### Held-out action-control results

| Scope | Model | JSON parse | Correct action type | Executable action | Exact action |
|---|---:|---:|---:|---:|---:|
| All (144) | Base | 66.0% | 27.1% | 0.7% | 0.0% |
| All (144) | LoRA | **97.9%** | **97.2%** | **37.5%** | 0.7% |
| De novo (95) | Base | 85.3% | 40.0% | 1.1% | 0.0% |
| De novo (95) | LoRA | **100.0%** | **100.0%** | 9.5% | 0.0% |
| Table1 edit (23) | Base | 30.4% | 4.3% | 0.0% | 0.0% |
| Table1 edit (23) | LoRA | **87.0%** | **82.6%** | **82.6%** | 4.3% |
| MuMO edit (26) | Base | 26.9% | 0.0% | 0.0% | 0.0% |
| MuMO edit (26) | LoRA | **100.0%** | **100.0%** | **100.0%** | 0.0% |

Artifacts:

```text
/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/
  outputs/unified_constraint_agent_common_llm_pilot_v1/
    model/seed_1703/training_summary.json
    eval/seed_1703/base/summary.json
    eval/seed_1703/tuned/summary.json
```

### Interpretation

The pilot passes the method-signal gate for shared constraint parsing and typed
action control: the LoRA model reliably selects the de novo versus edit action
space and makes most edit actions executable. It does **not** establish a
paper-level molecular-design improvement.

Greedy de novo output collapsed to five unique strings across 95 conditions;
one invalid string was emitted for 83 conditions. Edit outputs were more
diverse, but executable GraphEditDSL does not imply that the requested
properties improve or that the similarity constraint is met. Exact-action
matching is also too strict for underdetermined molecular tasks, so final
claims must use molecular property, similarity, validity, and strict-success
metrics.

### Method decision

Do not scale this direct action-SFT recipe to 7B yet. Keep the common LLM as the
shared constraint and revision controller, while retaining typed chemical
executors:

1. De novo: let the chemical generator produce candidates; let the common LLM
   produce a structured design/revision plan and select verifier-guided actions.
2. Editing: constrain GraphEditDSL generation to source-valid sites and use the
   executor plus vector verifier feedback during revision.
3. Evaluate both branches with sampled `n=20`, deterministic verifier selection,
   and the official strict property/similarity metrics.

This preserves one common reasoning model without requiring it to serve as an
unconstrained low-level molecular decoder for every task.

### Invalid earlier attempts

- Job `19371280` completed at the scheduler level but every saved adapter
  parameter was non-finite. It must not be used.
- Job `19372184` used the finite-value gate and failed at step 50 after BF16
  instability. It is audit evidence only.

## Executable GraphEditDSL ranking diagnostic

Date: 2026-08-09

Job `19387681` completed in 2 minutes 38 seconds. This diagnostic replaced
free-form edit generation with LLM likelihood ranking over at most 20
RDKit-executable one-step actions. It used the same 49 train-only held-out edit
rows as the format diagnostic: 23 Table1 and 26 MuMO.

| Scope | Model | Strict selected@1 | Any strict@20 | Mean property success | Mean source similarity |
|---|---:|---:|---:|---:|---:|
| All | Base | 2.0% | 18.4% | 20.4% | 0.793 |
| All | LoRA | **4.1%** | 18.4% | **28.0%** | 0.795 |
| Table1 | Base | 4.3% | 26.1% | 27.8% | 0.775 |
| Table1 | LoRA | 4.3% | 26.1% | **38.6%** | **0.810** |
| MuMO | Base | 0.0% | 11.5% | 13.8% | **0.810** |
| MuMO | LoRA | **3.8%** | 11.5% | **18.6%** | 0.782 |

All rows had at least one executable action, but post-filter pools averaged
only 11.6 unique molecules. Only 9/49 rows contained any strict-success action,
and the target teacher action appeared in the pool for 8/49 rows.

The LoRA signal is clearer below top-1. Among the nine reachable strict cases,
the best strict action's median rank improved from 9 to 4. LoRA placed five
strict actions in the global top-4 (10.2% of all rows), eight in top-7 (16.3%),
and all nine in top-10 (18.4%). Base placed only one in top-7 (2.0%). Thus the
common LLM learned useful action ordering, but the current exact-action SFT loss
is not a strong enough ranking objective.

### Decision after constrained ranking

Do not launch the official benchmark with this checkpoint. The next pilot must
address both independent limits:

1. Candidate support: produce 20 actual valid, diverse actions and add compact
   multi-step revisions so `any-hit@20` and teacher-action recall rise.
2. Selection: train directly on train-only strict-positive versus hard-negative
   action preferences, rather than relying on free-generation cross-entropy.
3. Keep the paper contract fixed: Common LLM ranks typed actions, RDKit executes
   them, and the vector verifier returns feedback for one bounded revision.

The failed predecessor job `19377452` stopped safely because four prompts were
longer than 512 tokens. Commit `de9e23b` raised the limit to 1024 and preserves
the complete action target under truncation; it did not change the evaluation
rows or candidate budget.

## Action preference pilot v1

Date: 2026-08-09

Job `19389677` completed in 20 minutes 54 seconds on one H100 20 GB MIG. It
built 1,840 train-only preference pairs and performed one reference-free
pairwise epoch from the stable `seed_1703` adapter. All adapter parameters were
finite. Training-pair ranking accuracy reached 88.7%.

The evaluation first expanded typed enumeration to 64 internal action attempts,
then retained exactly 20 valid unique output candidates. This fixed the prior
under-filled pool: mean candidate count rose from 11.6 to 20.0.

| Scope | Model | Strict selected@1 | Any strict@20 | Mean property success | Mean source similarity |
|---|---:|---:|---:|---:|---:|
| All | SFT | **12.2%** | 26.5% | **46.2%** | 0.768 |
| All | Preference v1 | 8.2% | 26.5% | 41.3% | **0.770** |
| Table1 | SFT | **17.4%** | 39.1% | **52.7%** | **0.800** |
| Table1 | Preference v1 | 13.0% | 39.1% | 47.4% | 0.797 |
| MuMO | SFT | **7.7%** | 15.4% | **40.4%** | 0.739 |
| MuMO | Preference v1 | 3.8% | 15.4% | 35.9% | **0.747** |

The candidate-support change is useful: relative to the prior under-filled SFT
diagnostic, `any-hit@20` increased from 18.4% to 26.5% and SFT selected@1 from
4.1% to 12.2%. Preference v1 itself failed the method gate.

The cause is label misalignment, not numerical instability. Only 3 of the 13
reachable strict-success rows had a teacher action that was also strict. The
pairwise objective successfully favored teacher actions, but teacher matching
was the wrong proxy for molecular strict success. It also increased the
top-ranked `delete_terminal_atom` bias from 32/49 to 37/49 rows. Among the 13
reachable rows, strict top-1 recovery fell from 6 to 4.

### Decision after preference v1

- Keep the full 20-candidate typed pool and the original SFT adapter as the new
  diagnostic baseline.
- Reject preference-v1 checkpoint `seed_1704` for benchmark use.
- Build preference positives directly from train-only verifier strict success;
  use valid non-strict candidates as hard negatives and keep property/similarity
  feedback in each record.
- Do not treat projected teacher-action identity as a strict-success label.

## Verifier-aligned preference v2

Date: 2026-08-09

Job `19390295` completed in 11 minutes 48 seconds. The data builder found 255
strict-positive train-only conditions out of 920 unique edit conditions and
created 510 strict-positive versus valid-hard-negative pairs. The conservative
one-epoch update used learning rate `5e-6`, produced 84.1% training-pair ranking
accuracy, and saved zero non-finite adapter parameters.

| Scope | Model | Strict selected@1 | Any strict@20 | Mean property success | Mean source similarity |
|---|---:|---:|---:|---:|---:|
| All | SFT | 12.2% | 26.5% | 46.2% | 0.768 |
| All | Verifier preference v2 | **14.3%** | 26.5% | **47.4%** | **0.775** |
| Table1 | SFT | **17.4%** | 39.1% | 52.7% | **0.800** |
| Table1 | Verifier preference v2 | **17.4%** | 39.1% | **55.0%** | 0.778 |
| MuMO | SFT | 7.7% | 15.4% | 40.4% | 0.739 |
| MuMO | Verifier preference v2 | **11.5%** | 15.4% | **40.7%** | **0.772** |

Top-1 gains are small: v2 recovers 7/13 reachable strict rows versus 6/13 for
SFT. The ranking distribution provides the stronger method signal. The median
best-strict rank improves from 2 to 1; strict coverage within top-2 improves
from 7/13 to 10/13, within top-5 from 8/13 to 12/13, and within top-10 from
11/13 to 13/13. A bounded vector-verifier check of v2's top five would therefore
recover 12/49 = 24.5% strict success, close to the pool ceiling of 13/49 =
26.5%, while checking only one quarter of the output pool.

Preference v2 passes the controller/ranker method-signal gate. It does not yet
support a paper claim because the diagnostic contains only 49 train-only
held-out edit rows. The next experiment should run the frozen `seed_1705`
ranker on the official Table1 and MuMO splits and report LLM@1, LLM plus bounded
verifier@5, and any-hit@20 separately.

## Existing MuMO two-step rerank

Date: 2026-08-09

Job `19400534` completed in 1 hour 53 minutes on one H100 20 GB MIG. It
reconstructed all 39,840 candidates for 1,992 official MuMO conditions and
ranked the immutable two-step top-20 pool without target-molecule access.

| Selection | IND SR | OOD SR | Overall SR | Overall strict | Sim>=0.4 |
|---|---:|---:|---:|---:|---:|
| Original heuristic@1 | 18.2% | 47.5% | 32.8% | 5.87% | 45.63% |
| Common LLM@1 | 18.6% | 49.5% | 34.0% | 4.17% | 38.25% |
| Original heuristic + verifier@5 | 23.7% | 61.7% | 42.6% | 7.58% | 42.67% |
| Common LLM + verifier@5 | 24.1% | 63.2% | 43.6% | 6.98% | 40.16% |
| Any@20 | 28.1% | 68.8% | 48.3% | 8.38% | 40.41% |

The common LLM recovered 90.1% of reachable property successes in its first
five plans, versus 88.2% for the original heuristic prefix. The net property
gain was small and not significant (`+1.21pp`, paired `p=0.104` at top-1;
`+0.95pp`, `p=0.073` with verifier@5). More importantly, top-1 strict success
dropped by 1.71 points and source-similarity success dropped by 7.38 points.

This run rejects further inference-only tuning of `seed_1705`. The cause is a
method mismatch: v2 learned single-action preferences and exposed heuristic
`policy_score`, whereas 37,812/39,840 formal candidates contain two actions.
The replacement v3 protocol trains complete plan payloads on a separately
generated MuMO train-only pool, uses explicit property/similarity tradeoff hard
negatives, hides planner scores, and applies balanced SFT replay before a
held-out source-preservation gate.

## Two-step plan preference v3

Date: 2026-08-09

Job `19409321` was submitted on one H100 20 GB MIG with seed `1706`, four CPU
cores, 96 GB host memory, and a 10-hour ceiling. It first generates and scores
a fresh MuMO train-only top-20 two-step pool, trains joint plan preferences, and
runs the frozen held-out gate. The immutable official test pool is opened only
if the new controller improves property success without regressing strict
success or source preservation beyond the declared tolerance.

The monolithic job remained pending without consuming compute time and was
cancelled. Replacement job `19426250` prepared the train-only pool on CPU in 3
hours 36 minutes and exited successfully. Its dependent 20 GB MIG job
`19426251` then remained pending for more than nine hours due solely to Slurm
priority and was cancelled without consuming compute. Job `19450295` reuses
the completed pool and requests one 40 GB H100 MIG for the six-hour GPU-only
stage. This preserves the exact experiment contract while avoiding an H100
reservation during graph enumeration and providing more eligible MIG slots.

Job `19450295` started on the 40 GB MIG but failed after 22 seconds because the
train-pool detail CSV contained only local oracle properties. All 10,000
candidates were valid and 9,597 passed the source-similarity threshold, but
missing ADMET-AI values for BBBP, HIA, and mutagenicity forced full property
coverage, official success, and strict success to zero. The repaired prepare
stage now runs the complete ADMET-AI + TDC oracle, produces a separately named
`benchmark_with_oracle_v1`, and materializes non-empty preference data before
the dependent GPU job becomes eligible.

The repaired retry uses CPU prepare job `19456333` with a two-hour limit and 32
GB RAM, followed by 40 GB H100 MIG job `19456334`. Historical runs of the same
oracle stack took at most 44 minutes 39 seconds and about 1.6 GB peak RAM, so
the tighter request improves backfill eligibility without reducing the actual
oracle or evaluation workload.

The completed repair established an oracle-complete train pool of 500
conditions and 10,000 candidates. Property any-hit@20 was only 7/500 (1.4%)
and strict any-hit@20 was 6/500 (1.2%), leaving 15 preference pairs and one
held-out validation condition. Training-pair ranking accuracy was 53.3%, and
the held-out primary gain was zero. The gate therefore stopped and the formal
test remained untouched. This is a support failure, not evidence that another
ranker loss or larger reranking budget is needed.

## Closed-loop common-LLM tool policy v1

Date: 2026-08-11

The replacement method trains the common 1.5B LLM on its own executable
one/two-step GraphEditDSL trajectories. Universal grammar plus RDKit defines
the action support; constraint predictors return per-property margins and
source-similarity observations after execution. Group-relative policy
gradients optimize the trajectory reward, and balanced de novo/Table1/MuMO SFT
anchors preserve the shared policy contract. No target molecule, heuristic
property planner, ranked pool, or formal test row is used by the rollout loop.
The official ADMET-AI predictors run as one persistent CPU sidecar so BBBP,
HIA, and mutagenicity are present in every MuMO observation; the trainer aborts
on any incomplete hard-constraint feedback.

The first signal gate uses seed `1707`, eight train conditions per edit suite,
four rollouts per condition, a two-edit ceiling, and 16 grammar actions per
state. Held-out train conditions are sampled with identical seeds before and
after the update. Only a positive reward/strict/property signal without more
than two points of property or source-similarity regression advances this
method to a larger pilot.

Initial job `19484190` failed after 3 minutes 1 second during the first policy
gradient update. The rollout, model, pinned TDC oracle, and persistent ADMET-AI
bridge all initialized successfully, but the differentiable 16-action
normalizer retained every candidate sequence graph and exhausted the 20 GB MIG
(18.02 GB already resident; the loss requested another 1.71 GB). The corrected
two-pass estimator still samples from all executable actions, then recomputes
gradients only for actions actually executed in the trajectory. This preserves
the grammar support and group-relative signal while bounding the update graph
to one action sequence at a time.

Job `19540548` completed successfully on one H100 40 GB MIG in 6 minutes 36
seconds, but the method gate stopped. Sixteen train conditions produced only
four optimizer updates. Held-out strict any-hit and property-all any-hit were
zero both before and after training; mean best reward remained `1.4623`, and
all 48 candidate trajectories were byte-identical to baseline. The saved
adapter differs from the input and contains no non-finite parameters, so this
was an underpowered sampled-policy update rather than a failed training run.
The MuMO train rollouts contained no property-all or strict success, while
Table1 found strict trajectories for two of eight train conditions.

## Exact-action-value common-LLM tool policy v2

Date: 2026-08-11

V2 replaces the four-rollout Monte-Carlo advantage with the exact expectation
over the complete executable GraphEditDSL support at every visited state. Each
action is executed and scored by the same official property and source
similarity environment; policy probabilities and normalized advantages are
computed over the full support, then action log probabilities are recomputed
serially for a bounded-memory exact categorical update. The transition remains
on-policy and target-free.

The new held-out gate supplements fixed-seed two-step rollouts with a
deterministic action-distribution comparison. It reports expected/top-1/oracle
reward, strict/property/similarity probability mass, and oracle-best action
probability. Held-out canonical action likelihood is measured independently on
de novo, Table1, and MuMO rows; regression beyond `0.05` mean token log
probability stops the run even when edit reward improves. The first v2 signal
run uses seed `1708`, 16 train and eight validation conditions per edit suite,
one epoch, up to 16 edit actions plus STOP per state, and 16 optimizer updates
expected from two-condition gradient accumulation.

Job `19543795` completed on one H100 40 GB MIG in 19 minutes 3 seconds. The
adapter now changed the actual policy: 41/128 held-out trajectories changed,
19 improved in reward, and 22 regressed. MuMO exact expected reward improved
from `1.1004` to `1.1284`, but Table1 decreased from `1.4979` to `1.4727` and
its top-1 reward dropped from `1.6082` to `1.2221`. Joint expected reward gained
only `0.0014`, strict any-hit remained `12.5%`, and the gate stopped. Canonical
action likelihood regressed by `0.0171` for de novo, `0.0110` for Table1, and
`0.0112` for MuMO, all inside the `0.05` safety bound. This establishes a real
but conflicting multi-task policy signal rather than another no-op update.

## Paired-PCGrad common-LLM tool policy v3

Date: 2026-08-11

V3 preserves the common adapter and exact typed-action environment while
changing only how the two edit-task gradients are combined. Each update pairs
one Table1 and one MuMO train condition. Their exact action-value gradients are
captured separately; when their dot product is negative, each is projected off
the conflicting component of the other before the two are averaged. Balanced
three-origin SFT anchoring is added after projection, so conflict handling does
not remove the explicit de novo/Table1/MuMO retention constraint.

The first signal run keeps the v2 budget fixed at 16 train and eight validation
conditions per edit suite, one two-step path, one epoch, and 16 paired updates.
It reuses every existing model, dataset, and oracle artifact. The one-hour
40 GB MIG submission is expected to take roughly 20--30 minutes and records
per-update task-gradient cosine, projection decisions, and gradient norms. Seed
`1708` is held fixed to preserve the v2 condition split and rollout randomness.

Job `19544779` completed in 18 minutes 33 seconds. Seven of 16 paired updates
had negative Table1/MuMO gradient cosine and were projected. The update changed
47/128 held-out trajectories, with 23 reward improvements and 24 regressions.
Strict any-hit stayed at 12.5%. Best-of-eight reward increased by `0.0406`, but
mean rollout reward decreased by `0.0041`, deterministic expected reward by
`0.00036`, top-1 reward fell by `0.1918`, and property-all probability fell
from 1.63% to 0.64%. Relative to v2, neither task improved expected reward. The
apparent legacy gate advance was therefore a best-of-k false positive; v3 is
retained as evidence of cross-task gradient conflict, not as the next main-line
checkpoint.

## Hierarchical common-agent support gate v4

Date: 2026-08-11

The global method decision is to retain the common LLM as the shared
ConstraintIR planner and revision controller, but stop treating flat token-level
RL over 16 generic edits as the main molecular generator. Existing train-only
source-copy pools reached only 1.2% strict any-hit@20, while the official
source-conditioned proposal plus two-step GraphEditDSL pool reached 48.3%
property any-hit@20. The missing variable is proposal/action support, not a
larger policy-gradient budget.

The v4 signal job uses 1,000 MuMO train-only proposer rows and 50 disjoint
train-only audit conditions. It emits exactly one raw proposal, performs bounded
two-step GraphEditDSL search, and sends exactly 20 final unique molecules per
condition to the official ADMET-AI + TDC oracle. It requests one 20 GB H100 MIG
for two hours; expected runtime is 45--70 minutes. Planner training is launched
only if property any@20 is at least 20% and strict any@20 is at least 5%.

Initial job `19547513` failed safely after 50 seconds, before model training.
Two independently seeded exports still shared 10 source/target pairs, and the
new contamination guard rejected the split. The retry no longer relies on seed
luck: it exports a larger audit candidate set, excludes every proposer-train
source, and then deterministically backfills exactly five rows for each of the
10 MuMO tasks. The failed job consumed no useful training or oracle budget.
