from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_validity_stability_contract():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    trainer = (HERE / "train_validity_stable_rl.py").read_text()
    run = (HERE / "run_train.sh").read_text()
    submit = (HERE / "submit_validity_stable_raw1_rl.sh").read_text()
    assert prereg["training"]["paired_optimizer_steps"] == 30
    assert prereg["training"]["denovo_validity_weight"] == 1.5
    assert prereg["training"]["reference_kl_weight"] == 0.2
    assert 'CHANNEL_WEIGHTS["de_novo"]["validity"] = 1.50' in trainer
    assert "--learning-rate 1.0e-7" in run
    assert "--denovo-anchor-weight 2.0" in run
    assert "--reference-kl-weight 0.20" in run
    assert "alignment_refresh/model/adapter" in run
    assert "best_of_40" not in submit

