"""
severity_calculator.py
======================
Computes a numeric severity score from a list of symptom strings and
classifies the result as "mild", "moderate", or "severe".

How the scoring works (explain this in your viva)
--------------------------------------------------
1. Each symptom provided by the user is looked up in the severity_weights
   dictionary (loaded from symptom_severity.csv).
2. The numeric weight for every *known* symptom is summed.
3. Symptoms that are NOT found in the dictionary are ignored (a warning is
   logged) and returned in the "skipped_symptoms" list so the caller can
   surface that information.
4. The total score is compared against two configurable thresholds:
   - If score <= MILD_MAX      → "mild"
   - If score <= MODERATE_MAX  → "moderate"
   - Otherwise                 → "severe"

Threshold constants
-------------------
The two constants below were chosen by inspecting the weight distribution in
symptom_severity.csv (typical weights range from 1–7):
- MILD_MAX     = 12  → a handful of low-weight symptoms like runny nose / cough
- MODERATE_MAX = 25  → multiple medium-weight symptoms, or a few high-weight ones

Adjust these values if you use a dataset with a different weight scale.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable severity thresholds  ← viva talking point
# ---------------------------------------------------------------------------
# MILD_MAX:     Maximum total score still classified as "mild".
#               Scores from 0 up to and including MILD_MAX are "mild".
MILD_MAX: float = 12.0

# MODERATE_MAX: Maximum total score still classified as "moderate".
#               Scores from (MILD_MAX + 1) up to MODERATE_MAX are "moderate".
#               Scores above MODERATE_MAX are classified as "severe".
MODERATE_MAX: float = 25.0


def calculate_severity(
    symptoms: list[str],
    severity_weights: dict[str, float],
) -> dict[str, Any]:
    """
    Compute a severity classification from a list of symptom strings.

    Parameters
    ----------
    symptoms         : List of symptom names as submitted by the user.
                       Case-insensitive; whitespace is stripped before look-up.
    severity_weights : Mapping of symptom (lower-case) -> numeric weight.
                       Typically loaded via dataset_manager.load_severity_weights().

    Returns
    -------
    dict with three keys:
        "severity"         : str  — one of "mild", "moderate", "severe"
        "score"            : float — raw sum of matched symptom weights
        "skipped_symptoms" : list[str] — symptoms not found in severity_weights

    Examples
    --------
    >>> weights = {"fever": 5.0, "cough": 3.0, "fatigue": 4.0}
    >>> calculate_severity(["fever", "cough"], weights)
    {'severity': 'mild', 'score': 8.0, 'skipped_symptoms': []}

    >>> calculate_severity(["fever", "chest_pain", "breathlessness"], weights)
    # "chest_pain" and "breathlessness" not in weights → skipped
    {'severity': 'mild', 'score': 5.0, 'skipped_symptoms': ['chest_pain', 'breathlessness']}
    """
    if not symptoms:
        logger.warning("calculate_severity called with an empty symptoms list.")
        return {"severity": "mild", "score": 0.0, "skipped_symptoms": []}

    total_score: float = 0.0
    skipped: list[str] = []

    for raw_symptom in symptoms:
        # Normalise: strip whitespace and lower-case for consistent look-up
        normalised = raw_symptom.strip().lower().replace(" ", "_")

        if normalised in severity_weights:
            weight = severity_weights[normalised]
            total_score += weight
            logger.debug("Symptom '%s' matched → weight %.2f (running total: %.2f)",
                         normalised, weight, total_score)
        else:
            # Try with spaces instead of underscores as a fallback
            alt = normalised.replace("_", " ")
            if alt in severity_weights:
                weight = severity_weights[alt]
                total_score += weight
                logger.debug(
                    "Symptom '%s' matched via alt key '%s' → weight %.2f",
                    raw_symptom, alt, weight,
                )
            else:
                skipped.append(raw_symptom)
                logger.warning(
                    "Symptom '%s' (normalised: '%s') not found in severity weights — skipped.",
                    raw_symptom, normalised,
                )

    # -----------------------------------------------------------------------
    # Classification logic  ← viva talking point
    # -----------------------------------------------------------------------
    # We use two boundary checks so the thresholds can be changed in one place
    # (the module-level constants) without touching this function.
    if total_score <= MILD_MAX:
        severity_label = "mild"
    elif total_score <= MODERATE_MAX:
        severity_label = "moderate"
    else:
        severity_label = "severe"

    logger.info(
        "Severity calculation: score=%.2f, label='%s', skipped=%s",
        total_score, severity_label, skipped,
    )

    return {
        "severity": severity_label,
        "score": round(total_score, 2),
        "skipped_symptoms": skipped,
    }
