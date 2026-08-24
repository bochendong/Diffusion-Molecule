# P7: State-Constrained MolProgram

P7 is a fast, single-seed validity repair of P6. It changes no checkpoint and
does not introduce a task router, task-specific head, property oracle, or
reranker. The same autoregressive policy is decoded through one state-dependent
transition grammar from either an empty or source molecular graph.

The gate reuses the exact 64-condition P6 hard de-novo subset and the exact
200-condition MolEdit-Instruct Table 1 subset, with 20 raw samples per input.
It reports raw candidate validity plus strict `k=1,8,20` results.

```bash
bash SketchMol-Understanding-Condition/experiments/p7_state_constrained_molprogram/submit_p7_state_constrained_gate.sh
```
