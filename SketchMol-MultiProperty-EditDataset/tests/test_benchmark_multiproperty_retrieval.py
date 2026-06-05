import importlib.util
from pathlib import Path


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
