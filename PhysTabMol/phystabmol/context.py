"""UniVideo-inspired in-context conditioning for PhysTabMol.

UniVideo's useful idea for this project is not video synthesis itself; it is
the unified in-context interface: reference media + query media + text intent
become one conditioning stream. This module implements a lightweight analogue
for tabular molecular generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import IMAGE_FEATURE_COLUMNS, extract_image_features, table_row_from_smiles
from .schema import TARGET_COLUMNS


INTENT_DELTAS = {
    "default": {},
    "increase_logp": {"LogP": 0.8, "TPSA": -8.0},
    "decrease_logp": {"LogP": -0.8, "TPSA": 8.0},
    "increase_qed": {"QED": 0.12, "SA": -0.25},
    "lower_sa": {"SA": -0.6, "MW": -20.0},
    "more_polar": {"TPSA": 20.0, "HBA": 1.0, "LogP": -0.5},
    "less_polar": {"TPSA": -20.0, "HBA": -1.0, "LogP": 0.5},
}


@dataclass
class InContextConditioner:
    reference_weight: float = 0.55
    instruction_strength: float = 1.0
    feature_columns: list[str] = field(default_factory=lambda: list(IMAGE_FEATURE_COLUMNS))
    target_columns: list[str] = field(default_factory=lambda: list(TARGET_COLUMNS))

    def build(
        self,
        query_image_features: dict[str, float],
        default_targets: dict[str, float],
        reference_image_features: dict[str, float] | None = None,
        reference_smiles: str | None = None,
        intent: str = "default",
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Return image+target condition values and a readable target dict."""

        image_features = dict(query_image_features)
        if reference_image_features:
            image_features = self._blend_dicts(query_image_features, reference_image_features)

        targets = dict(default_targets)
        if reference_smiles:
            ref_row = table_row_from_smiles(reference_smiles)
            ref_targets = {col: ref_row[col] for col in self.target_columns}
            targets = self._blend_dicts(default_targets, ref_targets)

        targets = self._apply_intent(targets, intent)
        condition = np.asarray(
            [[image_features[col] for col in self.feature_columns] + [targets[col] for col in self.target_columns]],
            dtype=float,
        )
        return condition, targets

    def _blend_dicts(self, base: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
        out = {}
        keys = set(base) | set(reference)
        for key in keys:
            base_value = float(base.get(key, reference.get(key, 0.0)))
            ref_value = float(reference.get(key, base_value))
            out[key] = (1.0 - self.reference_weight) * base_value + self.reference_weight * ref_value
        return out

    def _apply_intent(self, targets: dict[str, float], intent: str) -> dict[str, float]:
        out = dict(targets)
        deltas = INTENT_DELTAS.get(intent, INTENT_DELTAS["default"])
        for key, delta in deltas.items():
            if key in out:
                out[key] = float(out[key] + self.instruction_strength * delta)
        out["QED"] = float(np.clip(out.get("QED", 0.5), 0.0, 1.0))
        out["LogP"] = float(np.clip(out.get("LogP", 0.0), -3.0, 7.0))
        out["TPSA"] = float(np.clip(out.get("TPSA", 0.0), 0.0, 180.0))
        out["HBD"] = float(np.clip(out.get("HBD", 0.0), 0.0, 6.0))
        out["HBA"] = float(np.clip(out.get("HBA", 0.0), 0.0, 12.0))
        out["RB"] = float(np.clip(out.get("RB", 0.0), 0.0, 15.0))
        out["SA"] = float(np.clip(out.get("SA", 2.0), 1.0, 8.0))
        out["MW"] = float(np.clip(out.get("MW", 250.0), 80.0, 650.0))
        return out


def features_from_image_or_default(image_path: str | None, default_row) -> dict[str, float]:
    if image_path:
        return extract_image_features(image_path)
    return {col: float(default_row[col]) for col in IMAGE_FEATURE_COLUMNS}

