# Unified 3M Source-Aware Follow-up Summary

- output root: `SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_sourceaware_followup_v1`
- tasks completed: `12`

| label | kind | rows | target fp | source fp | prior target fp | gen-prior fp | property MAE | delta MAE | move from prior |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced005001_full_s11 | full | 9455 | 0.4436 | 0.4295 | 0.4926 | -0.04906 | 9.472 | 2.158 | 0.1104 |
| baseline_full_s11 | full | 9455 | 0.437 | 0.4228 | 0.4953 | -0.05822 | 9.581 | 1.951 | 0.1309 |
| hard002_freeze_p000_steps1 | diffusion | 1000 | 0.4195 | 0.405 | 0.4625 | -0.04298 | 8.394 | 2.102 | 0.0954 |
| hard002_freeze_p000_steps20 | diffusion | 1000 | 0.4374 | 0.4217 | 0.4625 | -0.02513 | 8.602 | 2.032 | 0.07777 |
| hard002_freeze_p000_steps5 | diffusion | 1000 | 0.4368 | 0.4213 | 0.4625 | -0.02573 | 8.552 | 2.029 | 0.07715 |
| hard002_full_s11 | full | 9455 | 0.4338 | 0.4231 | 0.4894 | -0.05565 | 10.23 | 2.163 | 0.1431 |
| hard002_full_s23 | full | 9455 | 0.426 | 0.4111 | 0.4879 | -0.06192 | 10.25 | 2.125 | 0.1106 |
| hard002_full_s37 | full | 9455 | 0.4337 | 0.4216 | 0.4897 | -0.05599 | 10.73 | 2.265 | 0.1429 |
| hard002_joint_p025_steps20 | diffusion | 1000 | 0.4026 | 0.3891 | 0.411 | -0.008452 | 8.06 | 4.026 | 0.02506 |
| hard002_joint_p050_steps20 | diffusion | 1000 | 0.3899 | 0.3761 | 0.3938 | -0.003977 | 7.985 | 4.04 | 0.02485 |
| hard002_joint_p100_steps20 | diffusion | 1000 | 0.3854 | 0.3706 | 0.3887 | -0.003241 | 8.333 | 4.396 | 0.02277 |
| sharedtiny_full_s11 | full | 9455 | 0.4448 | 0.4317 | 0.4921 | -0.0473 | 8.926 | 2.059 | 0.1329 |
