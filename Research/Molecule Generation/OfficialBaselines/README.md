# Official baseline workspace

This directory keeps paper-facing baselines separate from SUCC run outputs.

Tracked files in each method folder are only metadata and paper contracts.
The actual official repositories live under `repo/`, and generated results live
under `results/`; both are ignored by git and should be populated on the server.

Use:

```bash
bash SketchMol-Understanding-Condition/scripts/sync_official_code.sh --dry-run
bash SketchMol-Understanding-Condition/scripts/sync_official_code.sh
```

For methods without public official code, such as MolEditRL at the time this
contract was written, the folder records the paper-faithful settings instead of
pretending an official implementation exists.
