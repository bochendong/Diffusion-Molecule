# P5: Source-Anchored MolProgram

P5 tests whether one direct-SMILES MolProgram decoder can cover both de novo
generation and molecular editing without a router, graph-action executor,
materializer, external molecule library, or property-based inference reranking.

The edit path adds two source-gated components to the shared decoder:

1. zero-initialized residual adapters conditioned on the encoded source; and
2. a pointer-generator distribution that can copy individual source SMILES
   tokens while still generating mutations from the shared vocabulary.

The null-source path is an exact identity, so the protected de novo checkpoint
is unchanged by construction. The first run is a single-seed, 200-condition
gate using the same held-out MolEdit pool and train-only D3 targets as P4.

Run on Nibi with:

```bash
bash SketchMol-Understanding-Condition/experiments/p5_source_anchored_molprogram/submit_p5_source_anchored_molprogram.sh
```

