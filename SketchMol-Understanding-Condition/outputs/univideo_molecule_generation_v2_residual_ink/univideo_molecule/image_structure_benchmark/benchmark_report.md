# UniVideo Image-to-Structure Benchmark

This report evaluates generated molecule images after MolScribe OCR and RDKit validation.

- image CSV: `SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/univideo_molecule/image_structure_benchmark/image_path.csv`
- method label: `univideo_image_vae`
- source Tanimoto thresholds: `0.4,0.6,0.8`

## Main 2p-7p Table

`strict` is computed over all generated rows; invalid/OCR-failed molecules count as failures.

| method | validity all | 2p strict | 3p strict | 4p strict | 5p strict | 6p strict | 7p strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| univideo_image_vae | 0.385 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SketchMol structured reference |  | 0.804 | 0.768 | 0.736 | 0.716 | 0.678 | 0.685 |

## Source-Similarity-Constrained Success

`strict@Tanimoto>=t` means strict property success and Morgan Tanimoto(source, generated) >= t.

| method | mean source Tani | median source Tani | strict@0.4 | strict@0.6 | strict@0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| univideo_image_vae | 0.042 | 0.033 | 0.000 | 0.000 | 0.000 |

## Diagnostics

| method | OCR present | valid | source scaffold | unique valid | exact target | source identity | decode source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| univideo_image_vae | 0.385 | 0.385 | 0.000 | 0.003 | 0.000 | 0.000 | empty:615;raw_token_fallback:385 |
