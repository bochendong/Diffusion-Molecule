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
2. `submit_train.sh gate` continues the frozen aligned-24k adapter for a short
   token-budgeted gate.
3. Training uses a deterministic round-robin sampler over all 13 task buckets,
   so every prefix differs by at most one example per task. After the gate
   passes validity and non-copy checks, `submit_train.sh full` continues the
   balanced schedule and supports Slurm checkpoint resume.
4. The frozen Table 1 and Table 2 protocols are rerun. Paper prose is updated
   only after those results are frozen; the existing table structures are not
   changed by this experiment.
