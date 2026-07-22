"""Turn calibrated model evidence into a cautious user-facing analysis."""

from __future__ import annotations

import sys
from pathlib import Path

from config import PROJECT_ROOT

ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from calibration import classify_reliability  # noqa: E402


def _normalize_crop(value: str | None) -> str:
    return (value or "").lower().replace("_", "").replace(",", "").replace("(", "").replace(")", "")


class AnalysisService:
    @staticmethod
    def build(
        prediction: dict,
        image_validation: dict,
        selected_crop: str | None,
        readings: list[dict],
        contract: dict,
    ) -> dict:
        reliability = classify_reliability(
            prediction["probability_array"], contract["decision_policy"]
        )
        crop_probabilities = prediction.get("crop_probabilities", {})
        inferred_crop = max(crop_probabilities, key=crop_probabilities.get) if crop_probabilities else image_validation.get("inferred_crop")
        crop_confidence = float(crop_probabilities.get(inferred_crop, 0.0)) if inferred_crop else 0.0
        matches = _normalize_crop(selected_crop) == _normalize_crop(inferred_crop)
        crop_consistency = {
            "selected": selected_crop,
            "inferred": inferred_crop,
            "matches": matches,
            "confidence": crop_confidence,
        }
        observations = []
        if selected_crop and inferred_crop and not matches:
            observations.append(
                f"The selected crop ({selected_crop}) differs from the image evidence ({inferred_crop})."
            )
        if reliability["analysis_status"] == "inconclusive":
            observations.append("The stress classes are too close for a reliable conclusion.")
            recommendations = [
                "Retake a clear centered leaf photo in even light.",
                "Verify the seven sensor readings before trying again.",
            ]
        else:
            observations.append(
                f"The strongest model evidence is {prediction['predicted_class']} stress."
            )
            recommendations = [
                "Monitor the plant and compare it with nearby plants.",
                "Confirm unusual sensor readings with a second measurement.",
                "Consult a qualified agronomy professional before treatment decisions.",
            ]
        return {
            "analysis_status": reliability["analysis_status"],
            "reliability": reliability,
            "image_validation": image_validation,
            "crop_consistency": crop_consistency,
            "observations": observations,
            "recommendations": recommendations,
        }
