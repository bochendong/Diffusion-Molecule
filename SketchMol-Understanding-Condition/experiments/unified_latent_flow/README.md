# Unified Language-Conditioned Molecular Latent Flow

This is a bounded kill test for the new main direction:

```text
Common-LLM constraint tokens + source molecular latent (optional)
  -> continuous source-to-target latent flow
  -> direct SMILES decoder
  -> exactly 20 independent attempts
```

It intentionally forbids candidate-library materialization, retrieval, a
finalizer, and oracle reranking. De novo starts from a learned noisy prior;
Table1-style edit starts from the encoded source molecule. Both routes share
the same molecular encoder, latent vector field, constraint interface, and
SMILES decoder.

The first run is a single-seed feasibility test on the already frozen Unified
Joint train/validation split. It reuses the Common-LLM condition feature cache
and the stable direct-SMILES vocabulary/decoder checkpoint, while training only
the new molecular encoder and latent dynamics together with a short decoder
adaptation.

```bash
cd /scratch/bdong/projects/Diffusion-Molecule-automation
bash SketchMol-Understanding-Condition/experiments/unified_latent_flow/submit_unified_latent_flow_pilot.sh
```

Primary artifact:

```text
/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/
  outputs/unified_latent_flow_pilot_v1/seed_1715/summary.json
```

Read direct validity and unique-valid before property success. The experiment
is a no-go if the decoder repeats the historical token/image diffusion failure
(near-zero direct validity) or if it only copies the edit source. A positive
signal requires non-trivial direct validity/diversity and property any@20 above
an unconditioned/source-copy diagnostic; it does not by itself authorize an
official test run.
