"""
vitals_model.py
===============
VitalSense AI - Heart Rate & SpO2 Monitoring Module
Dataclass and validation logic for a single vitals reading pushed
from a Bluetooth-connected smartwatch / fitness band.

Normal adult reference ranges used throughout this module:
  * Resting heart rate : 60-100 bpm  (AHA / Mayo Clinic standard)
  * SpO2              : 95-100 %     (WHO / pulse-oximetry guidelines)

Alert thresholds are intentionally wider than the "normal" band so
that the system captures borderline and emergency cases:
  * HR < 50  (bradycardia risk)  / HR > 120 (tachycardia risk at rest)
  * HR < 40  (severe bradycardia)/ HR > 150 (severe tachycardia)
  * SpO2 < 95 % (hypoxaemia warning) / SpO2 < 90 % (critical hypoxia)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class VitalsReading:
    """
    Represents a single vitals snapshot pushed from a wearable device.

    Attributes
    ----------
    user_id : str
        Unique identifier of the patient / app user.
    device_id : str
        Bluetooth MAC address or internal ID of the wearable.
    heart_rate_bpm : Optional[int]
        Heart rate in beats per minute (optional; at least one vital
        must be present).
    spo2_percent : Optional[float]
        Blood-oxygen saturation as a percentage (optional; at least
        one vital must be present).
    recorded_at : str
        ISO-8601 timestamp string indicating when the reading was
        captured on the device (e.g. "2026-09-01T05:30:00Z").
    """

    user_id: str
    device_id: str
    recorded_at: str
    heart_rate_bpm: Optional[int] = field(default=None)
    spo2_percent: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Physiological bounds for heart rate (bpm).
# Absolute hardware minimum / maximum a real sensor can produce:
HR_MIN: int = 30   # below this is a sensor artefact in almost all cases
HR_MAX: int = 220  # above this is physiologically implausible for adults

# SpO2 bounds (%).
# A reading below 50 % is considered a sensor error; above 100 is impossible.
SPO2_MIN: float = 50.0
SPO2_MAX: float = 100.0


def validate_reading(reading: VitalsReading) -> list[str]:
    """
    Validate a VitalsReading and return a list of error messages.

    Returns an empty list when the reading is completely valid.

    Validation rules
    ----------------
    1. At least one of heart_rate_bpm or spo2_percent must be present.
    2. If heart_rate_bpm is provided it must be in [HR_MIN, HR_MAX].
    3. If spo2_percent is provided it must be in [SPO2_MIN, SPO2_MAX].

    Parameters
    ----------
    reading : VitalsReading
        The reading object to validate.

    Returns
    -------
    list[str]
        A (possibly empty) list of human-readable error messages.
    """
    errors: list[str] = []

    # Rule 1 - at least one vital must be present
    if reading.heart_rate_bpm is None and reading.spo2_percent is None:
        errors.append(
            "At least one of 'heart_rate_bpm' or 'spo2_percent' must be provided."
        )

    # Rule 2 - heart rate range check
    if reading.heart_rate_bpm is not None:
        if not (HR_MIN <= reading.heart_rate_bpm <= HR_MAX):
            errors.append(
                f"heart_rate_bpm must be between {HR_MIN} and {HR_MAX} bpm "
                f"(received {reading.heart_rate_bpm})."
            )

    # Rule 3 - SpO2 range check
    if reading.spo2_percent is not None:
        if not (SPO2_MIN <= reading.spo2_percent <= SPO2_MAX):
            errors.append(
                f"spo2_percent must be between {SPO2_MIN} and {SPO2_MAX} % "
                f"(received {reading.spo2_percent})."
            )

    # Field-presence checks
    if not reading.user_id or not isinstance(reading.user_id, str):
        errors.append("'user_id' must be a non-empty string.")

    if not reading.device_id or not isinstance(reading.device_id, str):
        errors.append("'device_id' must be a non-empty string.")

    if not reading.recorded_at or not isinstance(reading.recorded_at, str):
        errors.append("'recorded_at' must be a non-empty ISO-8601 timestamp string.")

    return errors
