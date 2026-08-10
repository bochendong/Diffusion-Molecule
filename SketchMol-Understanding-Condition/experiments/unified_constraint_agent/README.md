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
action executes. Group-relative policy gradients optimize complete sampled
trajectories, while a balanced de novo/Table1/MuMO SFT term keeps the shared
constraint language and both action schemas anchored.

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
