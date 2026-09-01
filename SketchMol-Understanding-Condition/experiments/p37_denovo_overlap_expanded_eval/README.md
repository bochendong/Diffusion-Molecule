# P37 expanded de novo overlap evaluation

This evaluation reuses the completed fresh 10k and 100k Unified and de novo
specialist adapters. It does not train or modify a checkpoint.

The frozen gate excludes every condition in the earlier 120-row gate and every
target molecule in the 100k de novo training subset. It contains 100 conditions
per overlap group and arity at 2p--4p, plus 40 per group at 5p:

- `shared_only`: every requested property is in `{MW, LogP, QED, HBA, RB}`;
- `contains_denovo_only`: the program contains `TPSA` or `HBD`.

The primary diagnostic is the paired, unweighted 2p--4p arity macro. The
2p--5p macro is secondary because the official 6,000-condition pool contains
only 48 eligible 5p shared-only programs. Every output is Raw@1 with no
property-aware selection or reranking.

Submit with:

```bash
bash experiments/p37_denovo_overlap_expanded_eval/submit_expanded_overlap_eval.sh
```
