import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_multiproperty_retrieval.py"
SPEC = importlib.util.spec_from_file_location("benchmark_multiproperty_retrieval", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_summary_tracks_joint_success_and_fallbacks():
    rows = [
        {
            "method": "scaffold_property_retrieval",
            "property_count": 2,
            "strict_success": True,
            "scaffold_match": True,
            "joint_success": True,
            "fallback": "",
        },
        {
            "method": "scaffold_property_retrieval",
            "property_count": 2,
            "strict_success": True,
            "scaffold_match": False,
            "joint_success": False,
            "fallback": "global",
        },
        {
            "method": "scaffold_property_retrieval",
            "property_count": 2,
            "strict_success": False,
            "scaffold_match": True,
            "joint_success": False,
            "fallback": "source_identity",
        },
    ]

    summary = benchmark._summarize(rows)
    by_label = {row["benchmark_label"]: row for row in summary}

    assert by_label["2_properties"]["success_rate_strict_in_valid_mols"] == 2 / 3
    assert by_label["2_properties"]["scaffold_match_rate"] == 2 / 3
    assert by_label["2_properties"]["joint_success_rate"] == 1 / 3
    assert by_label["2_properties"]["global_fallback_fraction"] == 1 / 3
    assert by_label["2_properties"]["source_identity_fallback_fraction"] == 1 / 3
    assert by_label["all"]["joint_success_rate"] == 1 / 3


def test_edit_latent_candidate_uses_predicted_target_and_delta():
    prop_count = len(benchmark.PROPERTY_COLUMNS)
    latent = np.zeros(prop_count * 4, dtype=np.float32)
    mw_idx = benchmark.PROPERTY_COLUMNS.index("MW")
    logp_idx = benchmark.PROPERTY_COLUMNS.index("LogP")
    latent[mw_idx] = 200.0
    latent[logp_idx] = 2.0
    latent[prop_count + mw_idx] = 100.0
    latent[prop_count + logp_idx] = 1.0
    latent[2 * prop_count + mw_idx] = 1.0
    latent[2 * prop_count + logp_idx] = 1.0
    latent[3 * prop_count + mw_idx] = 1.0
    latent[3 * prop_count + logp_idx] = 1.0

    row = {
        "condition_id": "cond_1",
        "source_smiles": "",
    }
    source_props = {prop: 0.0 for prop in benchmark.PROPERTY_COLUMNS}
    source_props["MW"] = 100.0
    source_props["LogP"] = 1.0
    pool = [
        {
            "smiles": "bad",
            "scaffold": "s",
            "props": {"MW": 350.0, "LogP": -1.0},
        },
        {
            "smiles": "good",
            "scaffold": "s",
            "props": {"MW": 205.0, "LogP": 2.1},
        },
    ]

    candidate, fallback = benchmark._best_edit_latent_candidate(
        row,
        pool,
        edit_latent_context={"predictions_by_condition_id": {"cond_1": latent}},
        selected_props=["MW", "LogP"],
        source_props=source_props,
        property_weight=1.0,
        delta_weight=0.35,
        direction_weight=0.1,
        source_similarity_weight=0.0,
    )

    assert candidate["smiles"] == "good"
    assert fallback == ""
