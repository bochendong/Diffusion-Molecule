#!/usr/bin/env python3
"""Static scientific contract for P8.1.12."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main()->int:
    builder=(ROOT/"build_verified_pseudopairs.py").read_text(); run=(ROOT/"run_round.sh").read_text(); submit=(ROOT/"submit_queue.sh").read_text()
    filter_pos=builder.index('metrics.get("table1_strict_success") == "True"')
    score_pos=builder.index("scores = policy.score_programs")
    assert filter_pos < score_pos
    assert "if strict and not identity" in builder
    assert "source_similarity_threshold=0.65" in builder
    assert "outcome in eval_molecules" in builder
    assert "coverage_by_task" in builder and "coverage_by_property_count" in builder
    assert "eval_candidates_used_for_training\": False" in builder
    assert "--trainable-scope source_only" in run and "--disable-finalizer" in run
    assert "--num-samples 20" in run and "property-rerank" not in run.lower()
    assert "afterany:$pre" in submit and "afterany:$r1" in submit
    print("P8.1.12 static contract: PASS"); return 0
if __name__=="__main__": raise SystemExit(main())
