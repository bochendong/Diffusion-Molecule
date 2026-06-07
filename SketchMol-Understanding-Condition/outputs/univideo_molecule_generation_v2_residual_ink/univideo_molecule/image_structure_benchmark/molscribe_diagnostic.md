# UniVideo MolScribe OCR Diagnostic

- run_dir: `SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink`
- samples per cohort: `128`
- device: `cuda`
- backend: `sketchmol`
- preprocess_images: `True`

## Summary

| cohort | n | graph usable | raw usable | final valid | exact match | mean Tanimoto | decode sources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rdkit_source | 128 | 0.000 | 0.000 | 0.000 | 0.000 |  | sketchmol_graph:128 |
| rdkit_target | 128 | 0.000 | 0.000 | 0.000 | 0.000 |  | sketchmol_graph:128 |
| source_oracle | 128 | 0.000 | 0.000 | 0.000 | 0.000 |  | sketchmol_graph:128 |
| target_oracle | 128 | 0.000 | 0.000 | 0.000 | 0.000 |  | sketchmol_graph:128 |
| generated | 128 | 0.000 | 0.000 | 0.000 | 0.000 |  | sketchmol_graph:128 |

## Image Stats (mean)

- **rdkit_source**: nonwhite=0.0417, dark=0.0233, unique_rgb=4
- **rdkit_target**: nonwhite=0.0416, dark=0.0232, unique_rgb=4
- **source_oracle**: nonwhite=0.0433, dark=0.0230, unique_rgb=8
- **target_oracle**: nonwhite=0.0432, dark=0.0230, unique_rgb=8
- **generated**: nonwhite=0.0447, dark=0.0232, unique_rgb=8

## Examples

### rdkit_source

- idx=0 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
- idx=1 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
- idx=2 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``

### rdkit_target

- idx=0 source=sketchmol_graph valid=False
  expected: `COc1cccc(C(=O)N2C[C@H](CN3CCC(c4ccccc4)CC3)[C@@H](c3ccccc3)C2)c1`
  final: ``
  graph: ``
- idx=1 source=sketchmol_graph valid=False
  expected: `COc1cccc(C(=O)N2C[C@H](CN3CCC(c4ccccc4)CC3)[C@@H](c3ccccc3)C2)c1`
  final: ``
  graph: ``
- idx=2 source=sketchmol_graph valid=False
  expected: `COc1cccc(C(=O)N2C[C@H](CN3CCC(c4ccccc4)CC3)[C@@H](c3ccccc3)C2)c1`
  final: ``
  graph: ``

### source_oracle

- idx=0 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
- idx=1 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
- idx=2 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``

### target_oracle

- idx=0 source=sketchmol_graph valid=False
  expected: `COc1cccc(C(=O)N2C[C@H](CN3CCC(c4ccccc4)CC3)[C@@H](c3ccccc3)C2)c1`
  final: ``
  graph: ``
- idx=1 source=sketchmol_graph valid=False
  expected: `COc1cccc(C(=O)N2C[C@H](CN3CCC(c4ccccc4)CC3)[C@@H](c3ccccc3)C2)c1`
  final: ``
  graph: ``
- idx=2 source=sketchmol_graph valid=False
  expected: `COc1cccc(C(=O)N2C[C@H](CN3CCC(c4ccccc4)CC3)[C@@H](c3ccccc3)C2)c1`
  final: ``
  graph: ``

### generated

- idx=0 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
- idx=1 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
- idx=2 source=sketchmol_graph valid=False
  expected: `O=C(c1cccc(Cl)c1)N1C[C@H](CN2CCC(c3ccccc3)CC2)[C@@H](c2ccccc2)C1`
  final: ``
  graph: ``
