"""Explainability utilities for DrugOracle."""

from typing import Dict, List


def explain_prediction(admet_preds: Dict[str, float]) -> List[str]:
    explanations = []
    for name, val in admet_preds.items():
        if name in {"herg", "ames", "dili"} and val > 0.5:
            explanations.append(f"High risk predicted for {name.upper()}")
    return explanations
