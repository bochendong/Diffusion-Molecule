# P6: Unified Molecular Transition Policy

P6 removes the remaining representation switch in the earlier UMTP pilot.
Both tasks use one property-program conditioner, one autoregressive decoder,
one checkpoint, one typed graph-action vocabulary, and one deterministic
interpreter. De-novo generation starts from the empty graph; molecular editing
starts from the supplied source graph. There is no task router, task-specific
head, direct-SMILES branch, candidate materializer, or property-aware finalizer.
An automatically derived initial-state token (`INIT_EMPTY` or `INIT_SOURCE` in
the method description) exposes graph occupancy to the common policy; it is an
input-state observation and never selects a separate module.

The first experiment is deliberately a bounded single-seed falsification gate:
32 conditions each for 6p/7p de novo and 20 examples from every Table 1 editing
task, with 20 raw policy samples per condition. It reports raw success,
pass@k, validity, uniqueness, and the official editing predicate.

```bash
bash SketchMol-Understanding-Condition/experiments/p6_unified_molecular_transition_policy/submit_p6_unified_transition_gate.sh
```
