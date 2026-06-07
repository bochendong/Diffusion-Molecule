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
- edit latent predictions: `SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1/sharedtiny_full_s11/eval_latent`
- edit latent scorer weights: property=1.0, delta=0.35, direction=0.1, fingerprint=1.0, source_similarity=1.0, source_similarity_rerank_candidates=256

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success AND Morgan fingerprint Tanimoto(source, generated) >= t. This is the primary source-conditioned editing metric; scaffold match is diagnostic only.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_identity | 1.000 | 1.000 | 0.295 | 0.295 | 0.295 |
| scaffold_property_retrieval | 0.941 | 1.000 | 0.279 | 0.278 | 0.270 |
| edit_latent_source_similarity_rerank | 0.200 | 0.171 | 0.045 | 0.014 | 0.002 |
| edit_latent_scaffold_source_rerank | 0.941 | 1.000 | 0.279 | 0.278 | 0.270 |
| target_oracle | 0.654 | 0.660 | 1.000 | 0.685 | 0.104 |

## Strict Property Success

This table is directly comparable to the SketchMol structured multi-property reference, but it does not measure source similarity by itself.

| method | 2p | 3p | 4p | 5p | 6p | 7p | mean source Tani all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.472 | 0.214 | 0.056 | 0.027 | 0.000 | 0.000 | 1.000 |
| scaffold_property_retrieval | 0.476 | 0.220 | 0.059 | 0.027 | 0.000 | 0.000 | 0.941 |
| edit_latent_source_similarity_rerank | 0.614 | 0.491 | 0.404 | 0.320 | 0.259 | 0.333 | 0.200 |
| edit_latent_scaffold_source_rerank | 0.476 | 0.220 | 0.063 | 0.027 | 0.000 | 0.000 | 0.941 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.654 |
| SketchMol structured reference | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |  |

## Scaffold-Match Diagnostics

`joint success = strict property success AND scaffold match`. Keep this as a hard source-preservation diagnostic, not the only edit metric.

| method | 2p | 3p | 4p | 5p | 6p | 7p | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_identity | 0.472 | 0.214 | 0.056 | 0.027 | 0.000 | 0.000 | 0.295 |
| scaffold_property_retrieval | 0.476 | 0.220 | 0.059 | 0.027 | 0.000 | 0.000 | 0.299 |
| edit_latent_source_similarity_rerank | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| edit_latent_scaffold_source_rerank | 0.476 | 0.220 | 0.063 | 0.027 | 0.000 | 0.000 | 0.299 |
| target_oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Fallback Diagnostics

| method | global fallback | source fallback | empty fallback |
| --- | ---: | ---: | ---: |
| source_identity | 0.000 | 0.000 | 0.000 |
| scaffold_property_retrieval | 0.000 | 0.909 | 0.000 |
| edit_latent_source_similarity_rerank | 0.000 | 0.000 | 0.000 |
| edit_latent_scaffold_source_rerank | 0.000 | 0.909 | 0.000 |
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
