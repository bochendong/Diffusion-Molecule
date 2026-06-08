# Multi-Property Direct Benchmark

This report compares source-image/property-condition retrieval baselines against the SketchMol multi-property strict-success reference, then adds source-similarity metrics for source-conditioned editing.

- condition rows: `SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/condition_rows.csv`
- eval split: `eval`
- candidate source: `molecule_database`
- candidate molecule DB: `SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/molecule_database.csv`
- max eval rows per property count: `250`
- eval target candidates excluded from retrieval pool: `True`
- scaffold fallback mode: `source_identity`
- source Tanimoto thresholds: `0.4,0.6,0.8`
- restrict eval to edit-latent index: `True`
- edit latent predictions: `SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceanchor_sweep_v1/blend095_guard050_p005/eval_latent_prior_only`
- edit latent scorer weights: property=1.0, delta=0.35, direction=0.1, fingerprint=1.0, source_similarity=1.0, source_similarity_rerank_candidates=256

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success AND Morgan fingerprint Tanimoto(source, generated) >= t. This is the primary source-conditioned editing metric; scaffold match is diagnostic only.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_identity | 1.000 | 1.000 | 0.170 | 0.170 | 0.170 |
| source_tanimoto_property_oracle | 0.488 | 0.500 | 0.515 | 0.148 | 0.013 |
| edit_latent_source_first_rerank | 0.499 | 0.500 | 0.463 | 0.148 | 0.013 |
| edit_latent_source_similarity_rerank | 0.278 | 0.221 | 0.118 | 0.031 | 0.004 |
| target_oracle | 0.640 | 0.649 | 1.000 | 0.655 | 0.070 |

## Strict Property Success

This table is directly comparable to the SketchMol structured multi-property reference, but it does not measure source similarity by itself.

| method | 2p | 3p | 4p | 5p | 6p | 7p | mean source Tani all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.480 | 0.260 | 0.020 | 0.020 | 0.000 | 0.000 | 1.000 |
| source_tanimoto_property_oracle | 0.740 | 0.700 | 0.560 | 0.340 | 0.042 | 0.000 | 0.488 |
| edit_latent_source_first_rerank | 0.640 | 0.660 | 0.480 | 0.320 | 0.042 | 0.000 | 0.499 |
| edit_latent_source_similarity_rerank | 0.840 | 0.740 | 0.480 | 0.480 | 0.458 | 0.000 | 0.278 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.640 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |  |

## Scaffold-Match Diagnostics

`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.

| method | 2p | 3p | 4p | 5p | 6p | 7p | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.480 | 0.260 | 0.020 | 0.020 | 0.000 | 0.000 | 0.170 |
| source_tanimoto_property_oracle | 0.140 | 0.120 | 0.080 | 0.060 | 0.000 | 0.000 | 0.087 |
| edit_latent_source_first_rerank | 0.120 | 0.140 | 0.060 | 0.060 | 0.000 | 0.000 | 0.083 |
| edit_latent_source_similarity_rerank | 0.060 | 0.040 | 0.000 | 0.040 | 0.000 | 0.000 | 0.031 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Fallback Diagnostics

| method | global fallback | source fallback | empty fallback |
| --- | ---: | ---: | ---: |
| source_identity | 0.000 | 0.000 | 0.000 |
| source_tanimoto_property_oracle | 0.000 | 0.000 | 0.000 |
| edit_latent_source_first_rerank | 0.000 | 0.000 | 0.000 |
| edit_latent_source_similarity_rerank | 0.000 | 0.000 | 0.000 |
| target_oracle | 0.000 | 0.000 | 0.000 |

Notes:
- `target_oracle` is an upper bound because it returns the known target molecule.
- `scaffold_property_retrieval` is the main strong non-generative baseline: it retrieves a candidate with the same scaffold and closest active-property values from the configured candidate library.
- `edit_latent_global_retrieval` ranks candidates using predicted target/delta properties from the learned source-conditioned edit latent.
- `source_tanimoto_property_oracle` is a candidate-library upper bound: it filters candidates by source Tanimoto first, then picks the closest active-property match.
- `edit_latent_source_first_rerank` filters candidates by source Tanimoto first, then applies the learned edit-latent scorer inside that source-similar pool.
- `edit_latent_source_similarity_rerank` is the main proposed retrieval-style method: it uses the learned edit latent plus source Tanimoto reranking without requiring scaffold identity.
- `edit_latent_scaffold_retrieval` applies that edit-latent scorer inside the same-scaffold candidate pool.
- `edit_latent_scaffold_source_rerank` is a scaffold-restricted diagnostic version of source-similarity reranking.
- `vlm_scaffold_feature_retrieval` retrieves same-scaffold train targets by nearest frozen VLM condition feature.
- `vlm_feature_retrieval` retrieves train targets by nearest frozen VLM condition feature without scaffold filtering.
- `global_property_vlm_rerank` first keeps the top property-matched candidates, then reranks them with Understanding-Condition features.
- `scaffold_property_vlm_rerank` applies the same rerank after same-scaffold property filtering. Unseen-scaffold fallback mode is `source_identity`.
- `global_property_retrieval` ignores source structure and can satisfy properties while failing source preservation.
- High strict success with low source Tanimoto should be interpreted as property retrieval, not source-conditioned molecular editing.
- A learned Understanding-Condition model should eventually improve strict property success at useful source-Tanimoto thresholds.
