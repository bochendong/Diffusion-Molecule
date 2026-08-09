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
