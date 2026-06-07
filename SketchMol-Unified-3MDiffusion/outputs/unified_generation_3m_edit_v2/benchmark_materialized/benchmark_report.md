# Multi-Property Direct Benchmark

This report compares source-image/property-condition retrieval baselines against the SketchMol multi-property strict-success reference, then adds source-similarity metrics for source-conditioned editing.

- condition rows: `SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/condition_rows.csv`
- eval split: `eval`
- candidate source: `molecule_database`
- candidate molecule DB: `SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/molecule_database.csv`
- max eval rows per property count: `5000`
- eval target candidates excluded from retrieval pool: `True`
- scaffold fallback mode: `source_identity`
- source Tanimoto thresholds: `0.4,0.6,0.8`
- restrict eval to edit-latent index: `True`
- edit latent predictions: `SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2/eval_latent`
- edit latent scorer weights: property=1.0, delta=0.35, direction=0.1, fingerprint=1.0, source_similarity=0.25

## Strict Property Success

This table is directly comparable to the SketchMol structured multi-property reference.

| method | 2p | 3p | 4p | 5p | 6p | 7p | scaffold all | joint all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.520 | 0.208 | 0.040 | 0.020 | 0.008 | 0.000 | 1.000 | 0.152 |
| scaffold_property_retrieval | 0.656 | 0.460 | 0.272 | 0.172 | 0.088 | 0.035 | 1.000 | 0.317 |
| edit_latent_global_retrieval | 0.792 | 0.608 | 0.536 | 0.384 | 0.388 | 0.404 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.636 | 0.428 | 0.264 | 0.160 | 0.072 | 0.018 | 1.000 | 0.299 |
| edit_latent_scaffold_source_rerank | 0.640 | 0.420 | 0.248 | 0.168 | 0.072 | 0.018 | 1.000 | 0.297 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |  |  |

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success AND Morgan fingerprint Tanimoto(source, generated) >= t. This is the main source-preservation metric for source-conditioned edit.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_identity | 1.000 | 1.000 | 0.152 | 0.152 | 0.152 |
| scaffold_property_retrieval | 0.808 | 0.857 | 0.295 | 0.239 | 0.086 |
| edit_latent_global_retrieval | 0.145 | 0.135 | 0.009 | 0.002 | 0.000 |
| edit_latent_scaffold_retrieval | 0.810 | 1.000 | 0.276 | 0.223 | 0.083 |
| edit_latent_scaffold_source_rerank | 0.820 | 1.000 | 0.277 | 0.230 | 0.092 |
| target_oracle | 0.632 | 0.634 | 1.000 | 0.607 | 0.086 |

## Scaffold-Match Diagnostics

`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.

| method | 2p | 3p | 4p | 5p | 6p | 7p | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.520 | 0.208 | 0.040 | 0.020 | 0.008 | 0.000 | 0.152 |
| scaffold_property_retrieval | 0.656 | 0.460 | 0.272 | 0.172 | 0.088 | 0.035 | 0.317 |
| edit_latent_global_retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.636 | 0.428 | 0.264 | 0.160 | 0.072 | 0.018 | 0.299 |
| edit_latent_scaffold_source_rerank | 0.640 | 0.420 | 0.248 | 0.168 | 0.072 | 0.018 | 0.297 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Fallback Diagnostics

| method | global fallback | source fallback | empty fallback |
| --- | ---: | ---: | ---: |
| source_identity | 0.000 | 0.000 | 0.000 |
| scaffold_property_retrieval | 0.000 | 0.366 | 0.000 |
| edit_latent_global_retrieval | 0.000 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.000 | 0.366 | 0.000 |
| edit_latent_scaffold_source_rerank | 0.000 | 0.366 | 0.000 |
| target_oracle | 0.000 | 0.000 | 0.000 |

Notes:
- `target_oracle` is an upper bound because it returns the known target molecule.
- `scaffold_property_retrieval` is the main strong non-generative baseline: it retrieves a candidate with the same scaffold and closest active-property values from the configured candidate library.
- `edit_latent_global_retrieval` ranks candidates using predicted target/delta properties from the learned source-conditioned edit latent.
- `edit_latent_scaffold_retrieval` applies that edit-latent scorer inside the same-scaffold candidate pool.
- `edit_latent_scaffold_source_rerank` is the main proposed retrieval-style method: it uses the learned edit latent plus optional fingerprint/source-similarity reranking inside the scaffold-preserving pool.
- `vlm_scaffold_feature_retrieval` retrieves same-scaffold train targets by nearest frozen VLM condition feature.
- `vlm_feature_retrieval` retrieves train targets by nearest frozen VLM condition feature without scaffold filtering.
- `global_property_vlm_rerank` first keeps the top property-matched candidates, then reranks them with Understanding-Condition features.
- `scaffold_property_vlm_rerank` applies the same rerank after same-scaffold property filtering. Unseen-scaffold fallback mode is `source_identity`.
- `global_property_retrieval` ignores source structure and can satisfy properties while failing source preservation.
- High strict success with low source Tanimoto should be interpreted as property retrieval, not source-conditioned molecular editing.
- A learned Understanding-Condition model should eventually improve strict property success at useful source-Tanimoto thresholds.
