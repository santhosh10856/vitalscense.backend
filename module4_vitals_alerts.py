"""
vitals_alerts.py
================
VitalSense AI - Heart Rate & SpO2 Monitoring Module
Alert-generation logic for abnormal vitals readings.

Clinical justification for thresholds (cite in viva)
-----------------------------------------------------
Heart rate
  Normal resting adult HR : 60-100 bpm  (American Heart Association)
  Moderate alert  HR > 120 : sustained tachycardia at rest - warrants
                             monitoring (Mayo Clinic: >100 bpm is
                             tachycardia; 120 bpm is a conservative
                             general-purpose alarm threshold).
  Moderate alert  HR < 50  : mild bradycardia; symptomatic concern
                             (AHA: <60 bpm is bradycardia, <50 bpm
                             requires clinical evaluation).
  Severe alert    HR > 150 : rapid ventricular response / SVT risk
                             (ACC/AHA guidelines flag > 150 bpm as
                             clinically significant tachycardia).
  Severe alert    HR < 40  : profound bradycardia / potential
                             haemodynamic compromise (AHA).

SpO2
  Normal adult range : 95-100 %  (WHO, British Thoracic Society)
  Moderate alert < 95 % : mild hypoxaemia - supplemental O2 may be
                          indicated (BTS SpO2 target for at-risk
                          adults is 94-98 %).
  Severe alert   < 90 % : significant hypoxaemia - ABCDE approach
                          and urgent medical review required
                          (WHO defines SpO2 < 90 % as severe
                          hypoxaemia).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heart rate alert thresholds (bpm)
# ---------------------------------------------------------------------------
HR_MODERATE_HIGH: int = 120   # > this -> moderate tachycardia alert
HR_MODERATE_LOW: int = 50     # < this -> moderate bradycardia alert
HR_SEVERE_HIGH: int = 150     # > this -> severe tachycardia alert
HR_SEVERE_LOW: int = 40       # < this -> severe bradycardia alert

# ---------------------------------------------------------------------------
# SpO2 alert thresholds (%)
# ---------------------------------------------------------------------------
SPO2_MODERATE_LOW: float = 95.0  # < this -> moderate hypoxaemia alert
SPO2_SEVERE_LOW: float = 90.0    # < this -> severe hypoxaemia alert


def _make_alert(alert_type: str, level: str, message: str) -> dict[str, Any]:
    """
    Construct a standardised alert dict.

    Parameters
    ----------
    alert_type : str
        Category of the alert, e.g. "heart_rate" or "spo2".
    level : str
        Severity level: "moderate" or "severe".
    message : str
        Human-readable description surfaced to the caller / clinician.

    Returns
    -------
    dict[str, Any]
        Alert object with keys: type, level, message.
    """
    return {"type": alert_type, "level": level, "message": message}


def check_alerts(reading: dict) -> dict[str, Any]:
    """
    Evaluate a vitals reading against clinical alert thresholds and
    return a structured response object.

    The function checks heart rate and SpO2 independently.  Severe
    thresholds take precedence over moderate ones for the same vital.

    Parameters
    ----------
    reading : dict
        A vitals reading dict (as returned by storage or parsed from
        the API request).  Keys: heart_rate_bpm (int | None),
        spo2_percent (float | None).

    Returns
    -------
    dict[str, Any]
        {
            "alerts": list[dict],        # zero or more alert objects
            "requires_attention": bool   # True if any alert is "severe"
        }
    """
    alerts: list[dict[str, Any]] = []

    hr: int | None = reading.get("heart_rate_bpm")
    spo2: float | None = reading.get("spo2_percent")

    # ------------------------------------------------------------------
    # Heart rate checks
    # Severe thresholds supersede moderate ones for the same direction.
    # ------------------------------------------------------------------
    if hr is not None:
        if hr > HR_SEVERE_HIGH:
            # Severe tachycardia (> 150 bpm) - overrides the moderate threshold
            alerts.append(_make_alert(
                "heart_rate", "severe",
                "Elevated heart rate detected"
            ))
            logger.warning("Severe tachycardia: HR=%d bpm user=%s",
                           hr, reading.get("user_id"))
        elif hr > HR_MODERATE_HIGH:
            # Moderate tachycardia (120 < HR <= 150)
            alerts.append(_make_alert(
                "heart_rate", "moderate",
                "Elevated heart rate detected"
            ))
            logger.info("Moderate tachycardia: HR=%d bpm user=%s",
                        hr, reading.get("user_id"))

        if hr < HR_SEVERE_LOW:
            # Severe bradycardia (< 40 bpm) - overrides the moderate threshold
            alerts.append(_make_alert(
                "heart_rate", "severe",
                "Low heart rate detected"
            ))
            logger.warning("Severe bradycardia: HR=%d bpm user=%s",
                           hr, reading.get("user_id"))
        elif hr < HR_MODERATE_LOW:
            # Moderate bradycardia (40 <= HR < 50)
            alerts.append(_make_alert(
                "heart_rate", "moderate",
                "Low heart rate detected"
            ))
            logger.info("Moderate bradycardia: HR=%d bpm user=%s",
                        hr, reading.get("user_id"))

    # ------------------------------------------------------------------
    # SpO2 checks
    # Severe threshold (< 90 %) supersedes moderate (< 95 %).
    # ------------------------------------------------------------------
    if spo2 is not None:
        if spo2 < SPO2_SEVERE_LOW:
            # Critical hypoxaemia (< 90 %) - emergency level
            alerts.append(_make_alert(
                "spo2", "severe",
                "Low blood oxygen - seek medical attention"
            ))
            logger.warning("Critical hypoxaemia: SpO2=%.1f%% user=%s",
                           spo2, reading.get("user_id"))
        elif spo2 < SPO2_MODERATE_LOW:
            # Mild-to-moderate hypoxaemia (90 % <= SpO2 < 95 %)
            alerts.append(_make_alert(
                "spo2", "moderate",
                "Blood oxygen below normal range"
            ))
            logger.info("Mild hypoxaemia: SpO2=%.1f%% user=%s",
                        spo2, reading.get("user_id"))

    # ------------------------------------------------------------------
    # Overall severity flag
    # If ANY alert is "severe", the caller must trigger doctor-referral
    # logic immediately.
    # ------------------------------------------------------------------
    requires_attention: bool = any(a["level"] == "severe" for a in alerts)

    return {
        "alerts": alerts,
        "requires_attention": requires_attention,
    }
