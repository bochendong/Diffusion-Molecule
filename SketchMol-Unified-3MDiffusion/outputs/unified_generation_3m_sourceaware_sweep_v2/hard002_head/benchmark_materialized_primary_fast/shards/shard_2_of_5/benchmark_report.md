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
- edit latent predictions: `SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_sweep_v2/hard002_head/eval_latent`
- edit latent scorer weights: property=1.0, delta=0.35, direction=0.1, fingerprint=1.0, source_similarity=1.0, source_similarity_rerank_candidates=256

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success AND Morgan fingerprint Tanimoto(source, generated) >= t. This is the primary source-conditioned editing metric; scaffold match is diagnostic only.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_identity | 1.000 | 1.000 | 0.350 | 0.350 | 0.350 |
| scaffold_property_retrieval | 0.914 | 1.000 | 0.330 | 0.330 | 0.305 |
| edit_latent_source_similarity_rerank | 0.199 | 0.169 | 0.050 | 0.010 | 0.000 |
| edit_latent_scaffold_source_rerank | 0.915 | 1.000 | 0.330 | 0.330 | 0.305 |
| target_oracle | 0.657 | 0.681 | 1.000 | 0.690 | 0.085 |

## Strict Property Success

This table is directly comparable to the SketchMol structured multi-property reference, but it does not measure source similarity by itself.

| method | 2p | 3p | 4p | 5p | 6p | 7p | mean source Tani all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.540 | 0.280 | 0.000 | 0.143 | 0.000 | 0.000 | 1.000 |
| scaffold_property_retrieval | 0.560 | 0.300 | 0.034 | 0.000 | 0.200 | 0.000 | 0.914 |
| edit_latent_source_similarity_rerank | 0.610 | 0.520 | 0.241 | 0.143 | 0.600 | 0.500 | 0.199 |
| edit_latent_scaffold_source_rerank | 0.560 | 0.300 | 0.034 | 0.000 | 0.200 | 0.000 | 0.915 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.657 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |  |

## Scaffold-Match Diagnostics

`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.

| method | 2p | 3p | 4p | 5p | 6p | 7p | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.540 | 0.280 | 0.000 | 0.143 | 0.000 | 0.000 | 0.350 |
| scaffold_property_retrieval | 0.560 | 0.300 | 0.034 | 0.000 | 0.200 | 0.000 | 0.365 |
| edit_latent_source_similarity_rerank | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 |
| edit_latent_scaffold_source_rerank | 0.560 | 0.300 | 0.034 | 0.000 | 0.200 | 0.000 | 0.365 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Fallback Diagnostics

| method | global fallback | source fallback | empty fallback |
| --- | ---: | ---: | ---: |
| source_identity | 0.000 | 0.000 | 0.000 |
| scaffold_property_retrieval | 0.000 | 0.855 | 0.000 |
| edit_latent_source_similarity_rerank | 0.000 | 0.000 | 0.000 |
| edit_latent_scaffold_source_rerank | 0.000 | 0.855 | 0.000 |
| target_oracle | 0.000 | 0.000 | 0.000 |

Notes:
- `target_oracle` is an upper bound because it returns the known target molecule.
- `scaffold_property_retrieval` is the main strong non-generative baseline: it retrieves a candidate with the same scaffold and closest active-property values from the configured candidate library.
- `edit_latent_global_retrieval` ranks candidates using predicted target/delta properties from the learned source-conditioned edit latent.
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
