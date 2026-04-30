"""UniVideo-style understanding stream for PhysTabMol.

The stream converts query media, reference molecules, and text intent into an
explicit semantic layer before generation. It is intentionally independent from
UniVideo code: the point is to preserve the idea of an understanding branch,
then feed its structured output into tabular diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .context import INTENT_DELTAS
from .features import IMAGE_FEATURE_COLUMNS, table_row_from_smiles
from .schema import TARGET_COLUMNS

INTENT_COLUMNS = [f"u_intent_{name}" for name in sorted(INTENT_DELTAS)]

UNDERSTANDING_COLUMNS = [
    "u_visual_brightness",
    "u_visual_contrast",
    "u_visual_edge_density",
    "u_visual_texture",
    "u_size_small",
    "u_size_medium",
    "u_size_large",
    "u_lipophilic",
    "u_polar",
    "u_hbond_rich",
    "u_druglike_window",
    "u_synthetic_easy",
    "u_reference_present",
] + INTENT_COLUMNS


@dataclass
class UnderstandingStream:
    enabled: bool = True
    intent_names: list[str] = field(default_factory=lambda: sorted(INTENT_DELTAS))

    def describe_dataframe(
        self,
        df: pd.DataFrame,
        targets_df: pd.DataFrame | None = None,
        intent: str = "default",
        reference_smiles: str | None = None,
    ) -> pd.DataFrame:
        if targets_df is None:
            targets_df = df[TARGET_COLUMNS]
        reference_targets = table_row_from_smiles(reference_smiles) if reference_smiles else None
        rows = []
        for idx, source in df.reset_index(drop=True).iterrows():
            image_features = {col: float(source[col]) for col in IMAGE_FEATURE_COLUMNS}
            targets = {col: float(targets_df.iloc[idx][col]) for col in TARGET_COLUMNS}
            rows.append(self.describe_row(image_features, targets, intent, reference_targets))
        return pd.DataFrame(rows)

    def describe_row(
        self,
        image_features: dict[str, float],
        targets: dict[str, float],
        intent: str = "default",
        reference_targets: dict[str, float] | None = None,
    ) -> dict[str, float | str]:
        numeric = {
            "u_visual_brightness": float(np.clip(image_features["img_mean"], 0.0, 1.0)),
            "u_visual_contrast": float(np.clip(image_features["img_contrast"], 0.0, 1.0)),
            "u_visual_edge_density": float(np.clip(image_features["img_edge_density"], 0.0, 1.0)),
            "u_visual_texture": float(np.clip(image_features["img_texture"] * 6.0, 0.0, 1.0)),
            "u_size_small": float(targets["MW"] < 250),
            "u_size_medium": float(250 <= targets["MW"] <= 500),
            "u_size_large": float(targets["MW"] > 500),
            "u_lipophilic": float(targets["LogP"] >= 3.0),
            "u_polar": float(targets["TPSA"] >= 75.0),
            "u_hbond_rich": float(targets["HBD"] + targets["HBA"] >= 6.0),
            "u_druglike_window": float(_druglike_window(targets)),
            "u_synthetic_easy": float(targets["SA"] <= 4.0),
            "u_reference_present": float(reference_targets is not None),
        }
        for name in self.intent_names:
            numeric[f"u_intent_{name}"] = float(intent == name)

        summary = self._summary(image_features, targets, intent, reference_targets)
        return {
            **numeric,
            "understanding_summary": summary,
            "understanding_tags": ";".join(self._tags(numeric)),
            "intent": intent,
        }

    def _summary(
        self,
        image_features: dict[str, float],
        targets: dict[str, float],
        intent: str,
        reference_targets: dict[str, float] | None,
    ) -> str:
        brightness = "bright" if image_features["img_mean"] >= 0.55 else "dark"
        contrast = "high-contrast" if image_features["img_contrast"] >= 0.55 else "low-contrast"
        size = "small" if targets["MW"] < 250 else "large" if targets["MW"] > 500 else "medium"
        polarity = "polar" if targets["TPSA"] >= 75 else "less-polar"
        ref = "with reference molecule" if reference_targets is not None else "without reference molecule"
        return (
            f"{brightness}, {contrast} query; target is {size}, {polarity}, "
            f"LogP={targets['LogP']:.2f}, QED={targets['QED']:.2f}; "
            f"intent={intent}; {ref}."
        )

    def _tags(self, numeric: dict[str, float]) -> list[str]:
        tags = []
        if numeric["u_visual_contrast"] >= 0.55:
            tags.append("high_visual_contrast")
        if numeric["u_lipophilic"]:
            tags.append("lipophilic")
        if numeric["u_polar"]:
            tags.append("polar")
        if numeric["u_druglike_window"]:
            tags.append("druglike_window")
        if numeric["u_reference_present"]:
            tags.append("reference_guided")
        active_intents = [name for name in self.intent_names if numeric[f"u_intent_{name}"]]
        tags.extend(active_intents)
        return tags or ["neutral"]


def understanding_matrix(understanding_df: pd.DataFrame) -> np.ndarray:
    return understanding_df[UNDERSTANDING_COLUMNS].to_numpy(dtype=float)


def _druglike_window(targets: dict[str, float]) -> bool:
    return (
        targets["MW"] <= 500
        and targets["LogP"] <= 5
        and targets["HBD"] <= 5
        and targets["HBA"] <= 10
        and targets["TPSA"] <= 140
        and targets["SA"] <= 6
    )
