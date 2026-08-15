from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    ROOT / "SketchMol-Understanding-Condition" / "experiments" / "unified_latent_flow"
)
MODEL_PATH = EXPERIMENT_DIR / "latent_edit_center_rewrite_decode.py"
RUN_PATH = EXPERIMENT_DIR / "run_latent_edit_center_rewrite_decode.sh"
SUBMIT_PATH = EXPERIMENT_DIR / "submit_latent_edit_center_rewrite_decode.sh"


def test_edit_region_is_latent_conditioned_and_applied_inside_diffusion() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    ast.parse(source)
    assert "def latent_edit_region" in source
    assert "node_nonkeep = torch.logsumexp(node_logits" in source
    assert "edge_nonkeep = torch.logsumexp(edge_logits" in source
    assert "if local_working is None:" in source
    assert "restrict_node_actions(node_legal, local_working)" in source
    assert "restrict_edge_actions(edge_legal, local_pairs)" in source
    assert '"latent_conditioned_edit_center": True' in source
    assert '"property_cardinality_center_count": True' in source


def test_generation_contract_freezes_exact_raw_n20_without_oracle_repair() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    assert '"generation_target_access": False' in source
    assert '"property_oracle_generation_access": False' in source
    assert '"generation_rdkit_validity_access": False' in source
    assert '"selector": False' in source
    assert '"finalizer": False' in source
    assert '"oracle_reranking": False' in source
    assert '"posthoc_molecule_repair": False' in source
    assert '"exact_raw_attempts_per_condition": 20' in source
    assert "if int(args.num_attempts) != 20:" in source


def test_runner_reuses_frozen_b22_and_is_bounded() -> None:
    run_source = RUN_PATH.read_text(encoding="utf-8")
    submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
    assert "valid_early_stop_delta_diffusion.pt" in run_source
    assert '--rewrite-radius "${SUCC_EDIT_CENTER_RADIUS:-1}"' in run_source
    assert "--num-attempts 20" in run_source
    assert "--gate-validity-improvement 0.10" in run_source
    assert "--cpus-per-task=1" in submit_source
    assert "--mem=4G" in submit_source
    assert "00:08:00" in submit_source
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in submit_source
    assert "dongbochen1218@gmail.com" in submit_source
