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
- condition features: `SketchMol-Understanding-Condition/outputs/condition_features_multiproperty_hf_vlm`
- condition feature array: `pooled`
- condition feature variant: `full`

## Strict Property Success

This table is directly comparable to the SketchMol structured multi-property reference.

| method | 2p | 3p | 4p | 5p | 6p | 7p | scaffold all | joint all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.425 | 0.201 | 0.054 | 0.013 | 0.005 | 0.000 | 1.000 | 0.155 |
| global_property_retrieval | 1.000 | 1.000 | 0.993 | 0.968 | 0.898 | 0.811 | 0.003 | 0.003 |
| scaffold_property_retrieval | 0.474 | 0.247 | 0.100 | 0.037 | 0.025 | 0.024 | 1.000 | 0.194 |
| vlm_feature_retrieval | 0.236 | 0.119 | 0.075 | 0.040 | 0.028 | 0.020 | 0.001 | 0.000 |
| vlm_scaffold_feature_retrieval | 0.425 | 0.201 | 0.054 | 0.013 | 0.005 | 0.000 | 1.000 | 0.155 |
| global_property_vlm_rerank | 0.999 | 0.986 | 0.922 | 0.806 | 0.687 | 0.535 | 0.001 | 0.001 |
| scaffold_property_vlm_rerank | 0.474 | 0.247 | 0.099 | 0.036 | 0.023 | 0.024 | 1.000 | 0.193 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |  |  |

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success AND Morgan fingerprint Tanimoto(source, generated) >= t. This is the main source-preservation metric for source-conditioned edit.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_identity | 1.000 | 1.000 | 0.155 | 0.155 | 0.155 |
| global_property_retrieval | 0.141 | 0.122 | 0.031 | 0.009 | 0.001 |
| scaffold_property_retrieval | 0.941 | 1.000 | 0.181 | 0.165 | 0.135 |
| vlm_feature_retrieval | 0.185 | 0.159 | 0.012 | 0.003 | 0.000 |
| vlm_scaffold_feature_retrieval | 0.997 | 1.000 | 0.155 | 0.154 | 0.154 |
| global_property_vlm_rerank | 0.131 | 0.121 | 0.009 | 0.001 | 0.000 |
| scaffold_property_vlm_rerank | 0.941 | 1.000 | 0.181 | 0.165 | 0.135 |
| target_oracle | 0.579 | 0.604 | 0.824 | 0.512 | 0.076 |

## Scaffold-Match Diagnostics

`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.

| method | 2p | 3p | 4p | 5p | 6p | 7p | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.425 | 0.201 | 0.054 | 0.013 | 0.005 | 0.000 | 0.155 |
| global_property_retrieval | 0.004 | 0.002 | 0.003 | 0.002 | 0.003 | 0.002 | 0.003 |
| scaffold_property_retrieval | 0.474 | 0.247 | 0.100 | 0.037 | 0.025 | 0.024 | 0.194 |
| vlm_feature_retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| vlm_scaffold_feature_retrieval | 0.425 | 0.201 | 0.054 | 0.013 | 0.005 | 0.000 | 0.155 |
| global_property_vlm_rerank | 0.000 | 0.000 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |
| scaffold_property_vlm_rerank | 0.474 | 0.247 | 0.099 | 0.036 | 0.023 | 0.024 | 0.193 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Fallback Diagnostics

| method | global fallback | source fallback | empty fallback |
| --- | ---: | ---: | ---: |
| source_identity | 0.000 | 0.000 | 0.000 |
| global_property_retrieval | 0.000 | 0.000 | 0.000 |
| scaffold_property_retrieval | 0.000 | 0.841 | 0.000 |
| vlm_feature_retrieval | 0.000 | 0.000 | 0.000 |
| vlm_scaffold_feature_retrieval | 0.000 | 0.996 | 0.000 |
| global_property_vlm_rerank | 0.000 | 0.000 | 0.000 |
| scaffold_property_vlm_rerank | 0.000 | 0.841 | 0.000 |
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
