# GeLLMO-C official baseline

Official code: https://github.com/ninglab/GeLLMO-C

This baseline covers C-MuMOInstruct source-conditioned multi-property editing
with property-specific improve/maintain objectives. The sync script clones or
fast-forwards the official repository into `repo/`.

Server entrypoint:

```bash
bash SketchMol-Understanding-Condition/scripts/submit_external_gellmoc_official_suite.sh
```

The wrapper calls the official `inference.sh` by default and then converts
`<SMILES>...</SMILES>` responses into the same candidate-level evaluator format
used by MuMO.
