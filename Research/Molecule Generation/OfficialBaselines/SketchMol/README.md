# SketchMol official baseline

Official code: https://github.com/WangZiXubiubi/SketchMol-v1

The repo already contains a vendored SketchMol checkout used by
`SketchMolBenchmark` for the confirmed OCR path:

```text
Research/Molecule Generation/SketchMol/SketchMol-v1-main
```

The official sync path under this folder is still useful for provenance and
future clean-room reruns, while existing OCR scripts continue to default to the
vendored path unless `SKETCHMOL_REPO` is overridden.
