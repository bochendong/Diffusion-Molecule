# Benchmark Alignment Matrix

The unified generator line must produce the same externally visible artifacts as the current benchmark scripts: selected prediction CSVs for single-output evaluation and candidate CSVs for candidate-level aggregation.

## Summary

| Benchmark family | Mode | Source input | Main output | Candidate budget | Existing metric contract |
| --- | --- | --- | --- | ---: | --- |
| SketchMol / de novo 2p-7p | `<DE_NOVO>` | none | generated SMILES | @1, @40, @128, @256 | strict success by property-count bucket |
| De novo OOD | `<DE_NOVO>` | none | generated SMILES | @1, @40, @128, @256 | strict success by OOD property task |
| MolEditRL / MolEdit-Instruct Table1 | `<EDIT>` | source SMILES/image | edited SMILES | n=20, n=256, n=2048 | Acc@0.65, Acc@0.15, validity, source similarity |
| MuMOInstruct / GeLLM3O style | `<EDIT>` | source SMILES/image | top-k edited candidates | top-20 or top-40 | SR, Sim(success), RI(success), IND/OOD split |
| C-MuMOInstruct / GeLLMO-C style | `<EDIT>` | source SMILES/image | top-k edited candidates | top-20 or top-40 | SR, Sim(success), RI(success), maintain/improve objective |

## De novo 2p-7p

Input contract:

```text
source_smiles = ""
task_mode = <DE_NOVO>
condition_properties = subset of LogP, QED, MW, TPSA, HBD, HBA, RB, SA
target_* columns = requested property targets
```

Unified condition pack:

```text
[VLM query tokens; <DE_NOVO>; property-program tokens]
```

Output contract:

```text
condition_id, generated_smiles, candidate_rank, candidate_selected,
condition_properties, property_count, method, task_mode
```

Alignment notes:

- Report @40 as the fair-budget row when comparing to SketchMol-like candidate budgets.
- Keep @128/@256 as scaling rows, not silently mixed with @40.
- No source similarity metric applies.

## De novo OOD

Input contract is the same as 2p-7p, but task distribution comes from the OOD property-program benchmark.

Alignment notes:

- Use the same `<DE_NOVO>` mode and output schema as 2p-7p.
- Report OOD strict separately from 2p-7p strict.
- Do not use source-preserving edit diagnostics here.

## MolEditRL / MolEdit-Instruct Table1

Input contract:

```text
source_smiles != ""
task_mode = <EDIT>
instruction/property direction = single-property or multi-property edit objective
```

Unified condition pack:

```text
[VLM query tokens; <EDIT>; source tokens; property/edit tokens]
```

Output contract:

```text
condition_id, source_smiles, generated_smiles, candidate_rank,
candidate_selected, condition_properties, property directions,
method, task_mode
```

Alignment notes:

- The fair row must not use Table1-success output-side reranking.
- If the finalizer uses Table1 objective success directly, label the row as assisted/selection.
- Keep candidate budget explicit: n=20, n=256, n=2048 are not interchangeable.
- Primary table metrics remain Acc@0.65 and Acc@0.15.

## MuMO / GeLLM3O-style benchmark

Input contract:

```text
source_smiles != ""
task_mode = <EDIT>
external_task_properties = task property set
external_property_directions_json = increase/decrease objectives
condition_id = input-level grouping key
```

Unified condition pack:

```text
[VLM query tokens; <EDIT>; source tokens; property/edit tokens]
```

Output contract:

```text
selected prediction CSV:
  one selected generated_smiles per condition_id

candidate prediction CSV:
  top-k generated_smiles per condition_id
  candidate_rank
  candidate_selected
```

Alignment notes:

- Candidate-level SR means one input succeeds if any candidate in the group satisfies every requested property.
- Keep top-20 and top-40 budgets separate.
- Current assisted edit baseline to beat/compare against: ADMET-prior GraphEditDSL v1 top-40 SR 54.3%.
- Oracle properties are for final evaluation only; they must not be used for test-time selection.

## C-MuMO / GeLLMO-C-style benchmark

Input contract:

```text
source_smiles != ""
task_mode = <EDIT>
external_task_properties = maintain/improve property set
external_property_directions_json = C-MuMO objectives
condition_id = input-level grouping key
```

Alignment notes:

- Use the same candidate CSV contract as MuMO.
- Maintain/improve needs source-side property values; source oracle coverage must be checked before claiming official SR.
- Current heuristic GraphEditDSL C-MuMO official SR is low; treat this as a separate hard benchmark, not as a MuMO proxy.

## Required columns for all outputs

Minimum selected-output columns:

```text
condition_id
task_mode
source_smiles
generated_smiles
method
condition_properties
property_count
```

Minimum candidate-output columns:

```text
condition_id
task_mode
source_smiles
generated_smiles
candidate_rank
candidate_selected
method
condition_properties
property_count
```

External edit benchmarks should additionally preserve:

```text
external_suite
external_task_properties
external_property_directions_json
source_property_* when present
target_property_* when present
```

