import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "unified_latent_external_transfer"
RUNNER = EXPERIMENT / "b_series_external_mumo_transfer_v1.py"
PREREG = EXPERIMENT / "b_series_external_mumo_transfer_v1_preregistration.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("b_external", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preregistration_locks_exact_n20_and_isolation_contract():
    payload = json.loads(PREREG.read_text())
    assert payload["condition_count"] == 50
    assert payload["conditions_per_ood_task"] == 10
    assert payload["exact_raw_attempts_per_condition"] == 20
    assert payload["candidate_pool_size"] == 20
    assert payload["molecular_candidate_ranking"] is False
    assert payload["retry_or_resampling"] is False
    assert payload["oracle_selection"] is False
    assert payload["generation_target_access"] is False
    assert payload["generation_property_oracle_access"] is False
    assert payload["official_test_access"] is False
    assert payload["frozen_b41_checkpoint"] is True
    assert payload["b41_training"] is False


def test_adapter_features_are_canonical_signed_set_features():
    module = load_runner()
    native = {prop: float(index + 1) for index, prop in enumerate(module.NATIVE_PROPERTIES)}
    first = module.adapter_features("BDMQ", native)
    second = module.adapter_features("BDMQ", dict(reversed(list(native.items()))))
    assert first.shape == second.shape
    np.testing.assert_allclose(first, second)
    signed, active = module.signed_property_vectors("BDMQ")
    assert signed[module.EXTERNAL_PROPERTIES.index("mutagenicity")] == -1.0
    assert active.sum() == 4.0


def test_proxy_condition_is_three_native_numeric_constraints():
    module = load_runner()
    native = {prop: float(index + 1) for index, prop in enumerate(module.NATIVE_PROPERTIES)}
    delta = np.asarray([0.1, -0.9, 0.8, 0.2, 0.7, 0.3, 0.4, 0.5])
    row, chosen = module.proxy_condition_row(
        "CC", native, delta, active_count=3, delta_scale=1.0
    )
    assert len(chosen) == 3
    assert row["property_count"] == "3"
    assert sum(row[f"{prop}_active"] == "1" for prop in module.NATIVE_PROPERTIES) == 3
    assert "target_smiles" not in row


def test_science_stop_is_not_an_execution_failure():
    source = RUNNER.read_text()
    assert "A scientific STOP remains a valid completed artifact" in source
    assert "return 0" in source
    submit = (EXPERIMENT / "submit_b_series_external_mumo_transfer_v1.sh").read_text()
    assert "afterok:$prepare_id" in submit
    assert "afterok:$freeze_id" in submit
    assert "afterok:$oracle_id" in submit
