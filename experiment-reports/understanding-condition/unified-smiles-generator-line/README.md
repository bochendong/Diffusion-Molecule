# Unified SMILES Generator Experiment Line

| Field | Value |
| --- | --- |
| Status | Phase1 de novo + Table1 edit benchmarks completed (reports below) |
| Scope | One generator for de novo generation and conditional/source-preserving edit |
| Code base | `SketchMol-Understanding-Condition` |
| Code folder | `SketchMol-Understanding-Condition/experiments/unified_smiles_generator/` |
| Primary output | Final/generated SMILES, plus optional candidate CSV for benchmark aggregation |
| Benchmark contract | Must stay compatible with existing de novo, MolEditRL/Table1, MuMO, C-MuMO, and SketchMol-style property benchmarks |

## Core idea

The new line should stop treating de novo generation and conditional edit as two unrelated pipelines. Instead, both tasks are routed through one instruction-conditioned SMILES generator:

```text
instruction / property program
        |
Frozen HF/VLM condition encoder
        |
shared condition tokens
        |
mode token: <DE_NOVO> or <EDIT>
        |
Unified SMILES generator
        |
candidate SMILES
        |
SMILES finalizer
        |
final SMILES
```

The two task modes differ only in the condition pack:

```text
de novo:
  [VLM query tokens; <DE_NOVO>; property tokens]

conditional edit:
  [VLM query tokens; <EDIT>; source tokens; property/edit tokens]
```

## Why this line exists

Current results show a split:

- De novo property generation can work well with direct SMILES sampling/reranking under the existing property-program setup.
- Source-conditioned editing is not solved by zero-source direct generation alone; current MuMO gains come from GraphEditDSL search and source-aware ranking.
- The paper story is stronger if we define a unified generator that can learn both modes, then benchmark it under the same candidate budgets and official evaluators already used in the repo.

This folder is the experiment-line home for that unified-generator idea.

## Completed reports

| Report | Summary |
| --- | --- |
| [`phase1-denovo-direct-compat-v1.md`](phase1-denovo-direct-compat-v1.md) | De novo 2p7p **97%** / OOD **74%** 2p · **45%** 7p |
| [`table1-edit-v1.md`](table1-edit-v1.md) | Direct Table1 RL **57%** / **81%** Acc@0.15; Unified gap documented |
| [`umtp-v1.md`](umtp-v1.md) | Unified Molecular Transformation Policy v1: source-aware shared decoder, adaptive retention, and train-only search distillation |
| [`umtp-v1.md#fast-source-aware-rl-go-no-go`](umtp-v1.md#fast-source-aware-rl-go-no-go) | One-MIG paired raw@1 RL pilot with an automatic retention-aware stop rule |

## Current-code grounding

The repository already provides the upstream building blocks:

- `HfVlmConditionEncoder` can use `source_image`; if it is missing but `source_smiles` exists, it renders the source molecule image on the fly.
- `ConditionedSmilesDecoder` is the current autoregressive Transformer SMILES decoder.
- Existing benchmark exporters/evaluators already expect selected prediction CSVs and candidate CSVs.

The new standalone experiment line makes the mode token, source-token path, task-aware group-RL reward router, and output contract explicit across all benchmarks.

## Files

| File | Purpose |
| --- | --- |
| `architecture.md` | Network/input-output definition for the unified generator |
| `benchmark_alignment.md` | Benchmark-by-benchmark alignment matrix |
| `runbook.md` | Proposed rollout stages, output schema, and script-entrypoint plan |
| `phase1-denovo-direct-compat-v1.md` | Phase1 de novo direct_compat benchmark results |
| `table1-edit-v1.md` | Table1 edit metrics/oracle fixes and RL benchmark results |
