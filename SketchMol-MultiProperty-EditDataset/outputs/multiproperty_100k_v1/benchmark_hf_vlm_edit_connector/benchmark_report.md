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
- condition features: `SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_edit_connector`
- condition feature array: `pooled`
- condition feature variant: `full`
- edit latent predictions: `SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm_edit_connector`
- edit latent scorer weights: property=1.0, delta=0.35, direction=0.1, source_similarity=0.25

## Strict Property Success

This table is directly comparable to the SketchMol structured multi-property reference.

| method | 2p | 3p | 4p | 5p | 6p | 7p | scaffold all | joint all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.425 | 0.201 | 0.054 | 0.013 | 0.005 | 0.000 | 1.000 | 0.155 |
| global_property_retrieval | 1.000 | 1.000 | 0.993 | 0.968 | 0.898 | 0.811 | 0.003 | 0.003 |
| scaffold_property_retrieval | 0.474 | 0.247 | 0.100 | 0.037 | 0.025 | 0.024 | 1.000 | 0.194 |
| edit_latent_global_retrieval | 0.316 | 0.190 | 0.124 | 0.082 | 0.050 | 0.029 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.464 | 0.240 | 0.094 | 0.034 | 0.020 | 0.020 | 1.000 | 0.188 |
| edit_latent_scaffold_source_rerank | 0.465 | 0.240 | 0.094 | 0.034 | 0.020 | 0.020 | 1.000 | 0.187 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |  |  |

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success AND Morgan fingerprint Tanimoto(source, generated) >= t. This is the main source-preservation metric for source-conditioned edit.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_identity | 1.000 | 1.000 | 0.155 | 0.155 | 0.155 |
| global_property_retrieval | 0.141 | 0.122 | 0.031 | 0.009 | 0.001 |
| scaffold_property_retrieval | 0.941 | 1.000 | 0.181 | 0.165 | 0.135 |
| edit_latent_global_retrieval | 0.119 | 0.114 | 0.001 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.942 | 1.000 | 0.178 | 0.163 | 0.136 |
| edit_latent_scaffold_source_rerank | 0.943 | 1.000 | 0.178 | 0.164 | 0.137 |
| target_oracle | 0.579 | 0.604 | 0.824 | 0.512 | 0.076 |

## Scaffold-Match Diagnostics

`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.

| method | 2p | 3p | 4p | 5p | 6p | 7p | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.425 | 0.201 | 0.054 | 0.013 | 0.005 | 0.000 | 0.155 |
| global_property_retrieval | 0.004 | 0.002 | 0.003 | 0.002 | 0.003 | 0.002 | 0.003 |
| scaffold_property_retrieval | 0.474 | 0.247 | 0.100 | 0.037 | 0.025 | 0.024 | 0.194 |
| edit_latent_global_retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.464 | 0.240 | 0.094 | 0.034 | 0.020 | 0.020 | 0.188 |
| edit_latent_scaffold_source_rerank | 0.465 | 0.240 | 0.094 | 0.034 | 0.020 | 0.020 | 0.187 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Fallback Diagnostics

| method | global fallback | source fallback | empty fallback |
| --- | ---: | ---: | ---: |
| source_identity | 0.000 | 0.000 | 0.000 |
| global_property_retrieval | 0.000 | 0.000 | 0.000 |
| scaffold_property_retrieval | 0.000 | 0.841 | 0.000 |
| edit_latent_global_retrieval | 0.000 | 0.000 | 0.000 |
| edit_latent_scaffold_retrieval | 0.000 | 0.841 | 0.000 |
| edit_latent_scaffold_source_rerank | 0.000 | 0.841 | 0.000 |
| target_oracle | 0.000 | 0.000 | 0.000 |

Notes:
- `target_oracle` is an upper bound because it returns the known target molecule.
- `scaffold_property_retrieval` is the main strong non-generative baseline: it retrieves a candidate with the same scaffold and closest active-property values from the configured candidate library.
- `edit_latent_global_retrieval` ranks candidates using predicted target/delta properties from the learned source-conditioned edit latent.
- `edit_latent_scaffold_retrieval` applies that edit-latent scorer inside the same-scaffold candidate pool.
- `edit_latent_scaffold_source_rerank` is the main proposed retrieval-style method: it uses the learned edit latent plus source similarity inside the scaffold-preserving pool.
- `vlm_scaffold_feature_retrieval` retrieves same-scaffold train targets by nearest frozen VLM condition feature.
- `vlm_feature_retrieval` retrieves train targets by nearest frozen VLM condition feature without scaffold filtering.
- `global_property_vlm_rerank` first keeps the top property-matched candidates, then reranks them with Understanding-Condition features.
- `scaffold_property_vlm_rerank` applies the same rerank after same-scaffold property filtering. Unseen-scaffold fallback mode is `source_identity`.
- `global_property_retrieval` ignores source structure and can satisfy properties while failing source preservation.
- High strict success with low source Tanimoto should be interpreted as property retrieval, not source-conditioned molecular editing.
- A learned Understanding-Condition model should eventually improve strict property success at useful source-Tanimoto thresholds.
