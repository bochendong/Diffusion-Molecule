# P8.2 matched inference evaluation

P8.2 evaluates the completed P8.1.1-R2 policy under two paper-facing raw-order
protocols with one exact checkpoint SHA. Arm A reuses the already sampled full
MolEdit Table1 candidate file (200 conditions, ten tasks, raw20) and reruns the
official evaluator at k=1/8/20 plus candidate level. Arm B samples the complete
6,000-condition 2p--7p suite at raw20 and reports k=1/4/8/20 for every property
count.

No property reranking is permitted. Table1 sampling already records that the
target molecule was not used at inference. For the new de novo arm, structural
target columns are physically removed before sampling. The final audit requires
the Table1 summary and all six de novo shards to name the exact P8.1.1-R2
checkpoint SHA, complete raw-candidate coverage, and one source-aware decoder.

```bash
bash SketchMol-Understanding-Condition/experiments/p8_2_matched_inference/submit_queue.sh
```
