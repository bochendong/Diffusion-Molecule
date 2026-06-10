# SUCC MolEdit-Instruct v3 Attack Plan

| Field | Value |
| --- | --- |
| Status | code ready; cluster run pending |
| Last update | 2026-06-10 |
| Goal | beat MolEditRL across Table1 task success, validity, source similarity, and FCD |
| Entry point | `SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_v3_attack_pipeline.sh` |

## Why v2 was not enough

The v2 repair proved the LLM-understanding path works, but the stricter Table1 extension exposed three gaps:

1. The enhanced_v1 data only had native rows for 7/10 Table1 tasks.
2. The default materializer optimized latent/source similarity, not the exact Table1 property-direction objective.
3. GSK3B remained at 0, and RB/MW/SA dropped on full 100-row sampling.

## v3 attack changes

### 1. Table1-complete benchmark pack

`export_moledit_table1_benchmark_pack.py` now supports:

```bash
--synthesize-missing-tasks
```

For missing RDKit-only tasks, it pairs existing source molecules with target molecules from the train/eval pool that satisfy the requested directions:

- `Haccept↓ SA↓`
- `Haccept↓ MW↓`
- `Haccept↑ MW↑ QED↓`

This should move the extension from 7/10 tasks toward 10/10 tasks when enough satisfying pairs exist.

### 2. Table-success candidate reranking

`materialize_univideo_target_molecules.py` now has two aggressive methods:

- `edit_latent_table_success_rerank`
- `source_tanimoto_table_success_oracle`

The main method first takes many latent candidates, then reranks by:

1. Table1 property-direction success fraction
2. source Tanimoto
3. latent score

This directly optimizes the table metric instead of hoping latent distance aligns with it.

### 3. Stronger training pressure

The v3 submit script uses:

- Table1-only training
- balanced train/eval sampling
- stronger Table1 sample weight
- stronger MW/SA/RB/HBA auxiliary weights
- `aux_all_properties=1`
- lower condition/source dropout
- more sampling steps
- `table_attack` materialized profile

## Run command

```bash
bash SketchMol-Understanding-Condition/scripts/submit_univideo_moledit_v3_attack_pipeline.sh
```

Expected primary outputs:

```text
SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v3_attack/
  univideo_molecule/benchmark_materialized_table_attack/
  univideo_molecule/moledit_table_metrics_attack/
  dataset/table1_benchmark_synthetic/
  univideo_molecule/benchmark_materialized_table1_attack/
  univideo_molecule/moledit_table_metrics_table1_attack/
```

## Success criteria

The run should be judged against MolEditRL on the same row set:

- Acc_all(0.65) per task
- Acc_all(0.15) per task
- Validity
- FCD
- coverage of all 10 Table1 tasks

The strongest claim is only allowed if `moledit_table_metrics_table1_attack/moledit_table_summary.md` covers all 10 tasks and beats MolEditRL on the mean and most individual tasks.
