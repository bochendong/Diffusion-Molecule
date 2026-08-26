# P24: MolProgramInstruct-Balanced

This experiment builds and trains on a release-oriented unified molecular
design dataset with separately audited generation and editing modes:

- **2,000,000 de novo program-to-molecule examples** derived from PubChem bulk
  SDF records;
- **569,919 source-conditioned edit examples** selected from the isolated
  MolEdit-Instruct training split.

The original 2M-edit target is not supportable without repeating pairs. After
excluding the frozen evaluation molecule set and deduplicating source-target
pairs, the limiting 7p bucket contains 81,417 assignable pairs. The release
therefore freezes 81,417 examples in each edit bucket (1p--7p), for 569,919
unique edit pairs total. This capacity-derived number is preferred over a
nominal 4M label that would silently count duplicates.

The release manifest counts instruction examples and unique chemical structures.
The manifest reports unique molecules, unique source-target pairs, and
instruction views separately. A future 4M/4M view release may reuse a molecule
or edit pair under several verified programs, but must not describe those views
as four million unique edit pairs.

## Data contract

Every row uses the same optional-source property program:

```json
{
  "source": "<EMPTY> or canonical source SMILES",
  "condition_program": [{"property": "MW", "goal": {"around": 320.1}}],
  "target_smiles": "canonical target SMILES",
  "messages": ["system", "user", "assistant"]
}
```

De novo programs contain two to seven numerical clauses drawn from MW, LogP,
QED, TPSA, HBD, HBA, and RB. Edit programs contain one to seven directional
clauses whose direction is verified from the cached source-target descriptor
difference. Original MolEdit-Instruct text and inferred task metadata are kept
as provenance, but unverified bioactivity clauses are not silently promoted to
training labels.

The release is task-balanced by property-program arity. The six de novo buckets
(2p--7p) and seven edit buckets (1p--7p) receive equal quotas, with at most one
extra row in the first buckets when a total is not divisible. Edit pairs remain
unique: high-arity scarcity causes the build to fail visibly instead of filling
a bucket by copying lower-arity examples.

## Leakage boundary

- PubChem molecules overlapping any frozen de novo target are excluded during
  finalization.
- Editing uses `enhanced_v1/splits/train.csv`; the 50,000-row balanced
  evaluation split is never scanned as a training source.
- Canonical source and target hashes are checked against all frozen editing
  references before release.
- Exact counts, SHA256 hashes, selection seeds, duplicate counts, and property
  distributions are written to the release manifest.

## Build and training sequence

1. `submit_build.sh` downloads eight official PubChem SDF chunks, extracts valid
   molecule records, selects exactly 2M de novo and 569,919 edit rows, and builds
   byte-offset indices.
2. `submit_train.sh gate` continues the frozen aligned-24k adapter in a fresh
   `gate_13k` directory for exactly 1,000 examples per broad task. Submission
   pins 500 steps and accumulation 26 so inherited shell variables cannot alter
   the frozen gate contract.
3. Training uses a deterministic round-robin sampler over all 13 broad task
   buckets. `submit_gate_validation.sh` freezes ten target-blind prompts per
   bucket and requires overall validity, per-bucket validity, and edit non-copy
   thresholds before full training can start. After that gate passes,
   `submit_train.sh full` consumes 81,415 examples per task and supports Slurm
   checkpoint resume.
4. `submit_alignment_refresh.sh FULL_JOB_ID` applies one low-rate epoch over
   720 rows for each of six de novo arities and ten frozen Table 2 edit tasks.
   This preserves scarce GSK3B, DRD2, and SA supervision after broad scaling.
5. `submit_table1_eval.sh REFRESH_JOB_ID` evaluates the refreshed P24 adapter on
   the same frozen 2p--7p conditions and best-of-40 finalizer used by the paper.
   `submit_table2_eval.sh REFRESH_JOB_ID` independently runs the frozen ten-task,
   500-output-per-task MolEdit sampled-once protocol with pinned assay oracles.
   Paper prose is updated only after both results are frozen; the existing table
   structures are not changed by this experiment.
