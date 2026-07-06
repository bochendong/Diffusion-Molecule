# GeLLMO official baseline

Official code: https://github.com/ninglab/GeLLMO

This baseline covers MuMOInstruct source-conditioned multi-property editing.
The sync script clones or fast-forwards the official repository into `repo/`.
Task inference outputs and converted candidate CSVs are written under
`results/` or the configured SUCC output directory.

Server entrypoint:

```bash
bash SketchMol-Understanding-Condition/scripts/submit_external_gellmo_official_suite.sh
```

The converted candidate CSV is evaluated with candidate-level `SR`,
`Similarity(success)`, and `RI(success)` using the existing external
multi-property evaluator.
