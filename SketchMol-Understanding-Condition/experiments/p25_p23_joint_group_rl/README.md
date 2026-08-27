# P25: P23 joint group-relative RL gate

P25 is the first RL experiment that actually starts from the frozen aligned-24k
P23 policy. It targets the weak high-order de-novo cells (5p--7p) and the ten
exact Table 2 edit families without mixing their raw reward scales: advantages
are normalized only among candidates sampled for the same prompt.

The first run is deliberately a promotion gate rather than a paper-table run.
It uses 39 training prompts (13 task buckets times three rounds), saves a
resumable checkpoint after every complete 13-bucket round, and compares the
original P23 policy with the RL policy on the same 260 frozen prompts and four
sampling repeats. No generated candidate is selected or reranked by properties.

The reward reads only the property program, optional source molecule, and model
response. The supervised anchor uses the training positive, but the RL reward
never reads `target_smiles`. Full Table 1 and Table 2 evaluation is authorized
only when `gate/comparison.json` says `PROMOTE_FULL_EVAL`.

Submit on Nibi with:

```bash
./experiments/p25_p23_joint_group_rl/submit_p25.sh
```
