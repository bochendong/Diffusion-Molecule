# Unified 3M Source-Anchor Sweep Summary

- output root: `SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceanchor_sweep_v1`
- tasks completed: `6`

| label | fp blend | fp guard | prior w | target fp | source-target fp | prior fp | fp gain vs source | fp beats source | property MAE | prop gain vs source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blend070_guard025_p005 | 0.7 | 0.25 | 0.05 | 0.6785 | 0.7878 | 0.6982 | -0.1093 | 0.01048 | 6.012 | 3.101 |
| blend085_guard050_p005 | 0.85 | 0.5 | 0.05 | 0.7073 | 0.7878 | 0.7338 | -0.08051 | 0.03581 | 5.701 | 3.412 |
| blend085_guard050_p025 | 0.85 | 0.5 | 0.25 | 0.7051 | 0.7878 | 0.7313 | -0.08273 | 0.03319 | 5.707 | 3.407 |
| blend085_guard100_p005 | 0.85 | 1 | 0.05 | 0.7066 | 0.7878 | 0.7329 | -0.08119 | 0.03231 | 5.82 | 3.293 |
| blend095_guard050_p005 | 0.95 | 0.5 | 0.05 | 0.7184 | 0.7878 | 0.7505 | -0.06943 | 0.05502 | 5.682 | 3.431 |
| blend095_guard100_p025 | 0.95 | 1 | 0.25 | 0.7169 | 0.7878 | 0.7488 | -0.07093 | 0.04105 | 6.424 | 2.689 |
