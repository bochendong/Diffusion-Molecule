# Unified Constraint-to-Action Molecular Agent

This experiment line treats unification as a shared constraint-understanding
and revision policy, not as one unconstrained low-level decoder for every task.

The contract is:

```text
(instruction, source_or_null)
  -> MolecularConstraintIR
  -> DESIGN: constrained SMILES | EDIT: GraphEditDSL
  -> vector verifier feedback
  -> fixed-budget revision
```

The first stage is CPU-only and reuses immutable candidate pools. It reports:

- `raw@1`: first-shot policy success;
- `any-hit@20`: reachable strict-success support;
- `selected@20`: current system selection success;
- strict-positive preferences and revision-needed failures for later LLM
  post-training.

Run locally or inside a CPU Slurm job:

```bash
SUCC_UCA_CONFIG_JSON=/path/to/teacher_pools.json \
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/run_teacher_pool_audit.sh

SUCC_UCA_CONFIG_JSON=/path/to/teacher_pools.json \
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_teacher_pool_audit.sh
```

The config contains immutable evaluated candidate CSVs:

```json
{
  "runs": [
    {
      "name": "denovo_direct_v2",
      "suite": "denovo_2p7p",
      "candidate_csv": "/absolute/path/to/candidates.csv",
      "budget": 20
    }
  ]
}
```

Do not mix candidate budgets in one paper comparison. Generate one maximum
pool and evaluate deterministic prefixes.

## Common-LLM pilot

Benchmark candidate pools are audit-only and must never be used for
post-training. The first common-LLM pilot builds supervision exclusively from:

- de novo training conditions and their target SMILES;
- official-instruction-aligned Table1 training actions;
- MuMO training source/target pairs projected into target-aligned GraphEditDSL.

The pilot deliberately starts with a 1.5B instruction LLM plus LoRA. This is a
method-signal gate that fits one H100 20 GB MIG; scale the same contract to 7B
only after first-shot control improves.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_data_prep.sh

# After the CPU data job succeeds:
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_pilot.sh

# After the adapter succeeds and passes its finite-weight gate:
SUCC_UCA_SEED=1703 \
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_eval.sh
```

The LLM sees a canonical `MolecularConstraintIR` and returns exactly one JSON
action. De novo rows use `{"action_type":"smiles",...}`; edit rows use
`{"action_type":"graph_edit_dsl",...}`. Evaluation remains fixed at the
paper-comparable candidate budget `n=20`.

The first 1.5B pilot outcome and its method decision are recorded in
[`RESULTS.md`](RESULTS.md). The held-out action-control evaluation is a
training-data diagnostic, not a replacement for fixed-`n=20` benchmark
evaluation.

After the stable SFT gate, the preference pilot keeps the same 1.5B adapter and
trains directly on train-only teacher actions versus structurally hard,
executable negatives:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_preference_pilot.sh
```

Its evaluation scores no more than 20 valid output actions per condition. A
larger internal attempt budget is used only for typed-grammar/RDKit validation
and is recorded separately in every manifest.

Teacher-action identity is not a reliable proxy for strict molecular success.
The verifier-aligned v2 pilot rebuilds preferences from train-only property and
source-similarity outcomes:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_verifier_preference_pilot.sh
```

After the v2 method gate passes, freeze `seed_1705` and submit the two official
edit suites in parallel:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_official_edit_eval.sh
```

Both suites rank exactly 20 executable, unique GraphEditDSL outputs per input.
The paper-facing comparison separates the LLM's first choice (`LLM@1`), the
target-free property/similarity verifier restricted to the first five
(`verifier@5`), and oracle reachability over all 20 (`any-hit@20`). Table1 uses
the existing MolEdit evaluator. MuMO candidates are scored with the official
ADMET-AI plus TDC property stack before bounded verifier selection. Progress is
checkpointed after every input, so a pre-empted cluster job can resume safely.

The official one-step MuMO pool is intentionally diagnostic. To isolate the
common LLM ranker from candidate support, rerank the already evaluated,
oracle-complete heuristic 2-step top-20 pool without regenerating molecules:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_existing_2step_rerank.sh
```

The runner streams the condition-grouped 2-step trace once, reconstructs each
candidate's executable plan, and uses mean per-action LLM log probability. The
official ADMET-AI/TDC outcomes are consulted only after LLM ordering for
`verifier@5` and metric aggregation.

## Two-step plan preference v3

The official rerank exposed a training/inference mismatch: verifier preference
v2 learned one action at a time, while the useful MuMO support consists almost
entirely of two-action plans. V3 fixes the contract instead of tuning inference:

1. build a fresh fixed `n=20` two-step pool from MuMO **train** rows only;
2. run the complete ADMET-AI + TDC oracle and score that pool with the same
   official property and source-similarity stack;
3. train on complete one/two-action plan preferences, including separate hard
   negatives for property success with similarity failure and the reverse;
4. remove heuristic `policy_score` from every model-visible action;
5. replay balanced de novo/Table1/MuMO SFT examples during preference tuning;
6. gate on held-out train conditions before the formal test pool is opened.

The stable 1.5B SFT adapter and the v3 adapter are both evaluated with joint
plan likelihood on the same held-out prefixes. The gate requires a positive
property/strict gain, no strict regression, and at most one point of source
similarity loss. Only a `go` decision launches the immutable formal MuMO pool.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_two_step_plan_preference_split.sh
```

The preferred Slurm entrypoint builds and officially scores the train-only pool
and materializes non-empty preference data in a CPU job. Only then does it
launch the H100 MIG training/evaluation job through an `afterok` dependency.
It sends begin/end/fail mail to the configured address, checkpoints train-pool
generation, and safely reuses completed artifacts on resubmission. The
monolithic submit script remains available for clusters where separate CPU/GPU
queues provide no scheduling advantage.

## Closed-loop common-LLM tool policy

V3 showed that static-pool preference tuning is support-limited: only 1.2% of
the train conditions had a strict success in the property-agnostic pool, so a
ranker cannot learn the missing molecular operation. The next method line no
longer treats the LLM as a reranker:

```text
ConstraintIR + current molecule + prior feedback
  -> common LLM tool policy
  -> typed GraphEditDSL executor
  -> per-constraint margins + source similarity
  -> revise or STOP (maximum two edits)
```

The action support is generated only by the universal GraphEditDSL grammar and
RDKit validity. It does not use a target molecule, property heuristic, oracle
ranking, or an existing top-20 pool. Properties are evaluated only after an
action executes. A persistent sidecar serves the official ADMET-AI predictors
from their isolated cluster environment, while RDKit and pinned TDC predictors
remain in the policy process; incomplete property feedback aborts the run.
Group-relative policy gradients optimize complete sampled trajectories, while
a balanced de novo/Table1/MuMO SFT term keeps the shared constraint language
and both action schemas anchored. Rollout selection scores the complete typed
action support without gradients; the update pass recomputes only executed
autoregressive action log probabilities, so the 16-action support does not
require all candidate computation graphs to remain resident on a 20 GB MIG.

The first single-seed pilot uses eight train conditions per edit suite, four
rollouts per condition, at most two edits, and a 16-action grammar slice. A
fixed held-out train split is evaluated before and after the update with the
same rollout seeds. The gate advances only on a real reward/strict/property
signal and rejects more than two points of property or similarity regression.
It does not open any formal benchmark test set.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_tool_policy_grpo.sh
```

The default Slurm request is one H100 20 GB MIG slice, four hours, with
begin/end/fail email notifications. Scale the number of train conditions and
the grammar support only after this closed-loop method gate advances.

### Exact-action-value policy iteration v2

The sampled v1 gate changed the adapter but reproduced all 48 held-out
trajectories exactly. V2 keeps the same target-free typed executor and official
feedback contract, but evaluates every executable action at each visited state.
It applies the exact categorical score-function update with serial backward
passes, so the complete 16-edit-action support supplies learning signal without
retaining every action graph in GPU memory.

The deterministic held-out gate now measures expected reward, top-1 reward,
strict/property/similarity probability mass, and the probability assigned to
the oracle-best action. A separate canonical-action likelihood check covers
held-out de novo, Table1, and MuMO rows before and after training; an edit gain
cannot advance if the shared policy regresses beyond the declared retention
tolerance.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_tool_policy_exact_v2.sh
```

The default signal run uses 16 train and eight validation conditions per edit
suite, one two-step path per condition, one epoch, up to 16 edit action values
plus STOP per state, and balanced three-task SFT anchoring. It requests one
H100 40 GB MIG for at most two hours and does not open a formal benchmark test
split.

### Paired conflict-aware policy iteration v3

The v2 exact-action update moved 41/128 held-out trajectories and raised MuMO
expected reward, but Table1 top-1 reward regressed enough to cancel the joint
gain. V3 treats this as shared-policy gradient interference rather than a need
for more sampling. Every optimizer step pairs one Table1 condition with one
MuMO condition, computes their exact action-value gradients independently,
projects only negative gradient conflicts, averages the projected gradients,
and then adds the balanced de novo/Table1/MuMO SFT anchor.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_llm_tool_policy_pcgrad_v3.sh
```

The signal run reuses the v2 data, evaluator, support size, train/validation
counts, and one-epoch budget for a direct comparison. It requests one H100
40 GB MIG, four CPUs, 32 GB host memory, and a one-hour ceiling. The output
records task-gradient cosine, projection frequency, and per-task gradient norm
for every paired update in addition to the unchanged effect and retention gate.
Seed `1708` is intentionally retained so the condition split and held-out
rollout randomness match v2; the versioned output directory prevents overwrite.

## Hierarchical common-agent support gate v4

The exact policy runs establish that a flat 16-action GraphEditDSL policy has a
real gradient signal, but cannot select a strict MuMO molecule that its support
never produces. V4 changes the method boundary rather than adding another loss
patch:

```text
ConstraintIR
  -> common LLM planner/controller
  -> raw-1 source-conditioned proposal tool
  -> typed GraphEditDSL revisions
  -> vector verifier feedback and bounded stop/replan
```

The first v4 job is deliberately a support gate, not planner training. It fits
the proposal tool on MuMO train rows, selects five audit rows per task only
after excluding every proposer-train source/target pair, emits one raw proposal
per condition, performs two-step graph search, and evaluates exactly 20 final
molecules with the full ADMET-AI + TDC oracle. Internal RDKit enumeration is
reported as search
compute; it is not represented as additional oracle candidates. The gate
requires property any@20 of at least 20% and strict any@20 of at least 5%.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_agent_hierarchical_support_v4.sh
```

If support advances, the common LLM is trained on complete typed tool plans,
including the proposal root and subsequent edit steps. If it stops, the next
change belongs in the proposal/action tool; flat policy RL remains an ablation
showing gradient conflict and support limitation.

## RetrievedDeltaEdit support gate v5

V4 stopped because property-success descendants of the unconstrained proposal
lost the source scaffold, while source-root GraphEditDSL descendants preserved
similarity but never satisfied all properties. V5 changes the action
representation instead of the loss or candidate budget. It extracts one-cut
matched-pair side-chain substitutions from the 1,000 proposer-training pairs,
retrieves same-task transformations from each held-out source fragment, and
rejoins the retrieved target-side fragment to the untouched held-out core. The
builder never reads a held-out target molecule.

```text
ConstraintIR
  -> common LLM planner/controller
  -> RetrievedDeltaEdit(source core, train-only matched-pair delta)
  -> vector verifier feedback
  -> bounded replan/stop
```

The signal run reuses the zero-overlap 50-condition v4 audit split and its
oracle-blind GraphEditDSL pool only as a completeness fallback. Candidate
selection uses source similarity, train-fragment retrieval similarity, and
descriptor-only ADMET priors. The complete official ADMET-AI + TDC oracle sees
exactly 20 final molecules per condition. Advance still requires property
any@20 of at least 20% and strict any@20 of at least 5%.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_common_agent_retrieved_delta_support_v5.sh
```

The completed full-oracle gate advanced. On the fixed zero-overlap 50-condition
audit with exactly 20 final candidates, property any@20 and strict any@20 were
both 40.0% overall and separately 40.0% on IND and OOD; validity and full-oracle
coverage were 100%. Main job `19556545` ran for 2 minutes 4 seconds. CPU repair
job `19558277` reused the same candidates and completed missing TDC DRD2 scores
in 54 seconds; controller `19558278` reconciled the complete result. The earlier
18% figure is retained only as an incomplete-oracle lower bound and must not be
used as the v5 result. See `RESULTS.md` for lineage and task-level details.

## Common-LLM RetrievedDelta planner v6

V6 keeps `RetrievedDeltaEdit` as the executable source-preserving tool and
trains the shared 1.5B common LLM to select its typed side-chain substitution.
The training target identifies a positive delta only on paired train rows; it
is absent from prompts. Balanced De novo/Table1/MuMO SFT replay and a frozen
format/action validation provide the explicit anti-forgetting gate.

For the 50-condition zero-overlap audit, the LLM may score at most 96 actions
from v5's existing internal retrieval space, then emits exactly 20 candidates
for the complete oracle. The comparison is paired against v5's 40% strict
any@20 result. Advance requires at least 46% property and strict any@20, no IND
or OOD strict regression, 100% validity/oracle coverage, and a passing unified
format-retention check.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/submit_experiment_loop.sh retrieved_delta_planner_v6
```

V6 completed but stopped its scientific gate: property and strict any@20 fell
from 40% to 22%, with OOD falling to 8%. The anti-forgetting check passed, and
the v5 heuristic prefix was byte-identical when reconstructed inside v6. The
failure therefore belongs to LLM-only molecular selection, not unified action
control or candidate-space drift.

## V7 support-ceiling decision audit

Before training another planner, the CPU-only v7 decision audit freezes the
pre-oracle heuristic prefix of the exact v6 enumerated pool and evaluates at
most 96 candidates per held-out condition. This is an explicitly diagnostic
support ceiling: it never changes the paper-facing n=20 contract and cannot be
reported as an n=96 method result. Oracle values are computed only after the
prefix has been materialized and are never used for selection.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_retrieved_delta_support_ceiling_v7.sh
```

If both IND and OOD strict support reach 70%, v7 may train a verifier-observed
n=20 residual planner over this support. Otherwise the next method change must
expand the source-preserving generator, using the common LLM as a typed
GraphEditDSL controller rather than launching another ranker on the same pool.

CPU job `19565544` completed this audit in 3 minutes 4 seconds. The oracle-blind
prefix averaged 94.82 candidates and reached only 58% property support and 50%
strict support; OOD was 48% property and 40% strict. The existing one-step
support therefore cannot reach the requested regime even with a perfect
selector. The next support gate composes at most two train-observed deltas,
keeps all v5 n=20 candidates as an immutable prefix, and uses normalized
train-pair property effects only to expand the pre-oracle support:

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/submit_composed_retrieved_delta_support_v7.sh
```

Job `19567054` completed in 12 minutes 6 seconds. The composed prefix retained
all 25 strict-success conditions from the one-step diagnostic and added four
OOD strict successes, raising overall strict support from 50% to 58% and OOD
strict support from 40% to 56%. Property support stayed at 58%, below the gate,
so another search-depth increase is not justified. The next reviewed method
step is a train-only learned property verifier that supplies calibrated numeric
margins to the common LLM; no formal test or GPU planner run has been opened.

## Bounded Slurm experiment automation

The experiment ladder now has a deterministic controller under
`automation/`. It treats Slurm and versioned JSON gates as the control plane;
email is notification only. Each runnable round, transition, protocol check,
retry, and accelerator-hour reservation is declared in
`automation/experiment_plan.json`. The controller uses an `afterany`
dependency job to reconcile the completed experiment and can submit only the
next allowlisted entrypoint.

```bash
bash SketchMol-Understanding-Condition/experiments/unified_constraint_agent/automation/submit_experiment_loop.sh
```

The checked-in plan now declares only the one-seed v6 planner signal and writes
a new versioned state; completed v4/v5 states remain immutable. V6 is terminal
after its support and anti-forgetting gate, so formal benchmark expansion still
requires an explicit reviewed manifest change. See `automation/README.md` for
the state, retry, budget, and recovery contracts.
