"""
report_analyzer.py — VitalSense AI · Module 3
==============================================
Rule-based medical report analysis engine.

Classification logic (for viva explanation):
─────────────────────────────────────────────
Given a numeric *value* and a reference row that contains four numeric
thresholds (critical_low, min_normal, max_normal, critical_high):

    value < critical_low            → "critical"  (dangerously low)
    critical_low ≤ value < min_normal → "low"      (below normal range)
    min_normal ≤ value ≤ max_normal → "normal"    (within expected range)
    max_normal < value ≤ critical_high → "high"   (above normal range)
    value > critical_high           → "critical"  (dangerously high)

"overall_status" rolls up the worst individual status:
    any "critical"        → "urgent"
    any "high" or "low"   → "attention_needed"
    all "normal"          → "normal"

No AI / ML is involved anywhere in this file — every decision is a simple
numeric comparison against CSV-sourced thresholds.
"""

import logging
from typing import Any

from reference_data import get_cached_data, _normalise_param

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status constants & severity ordering (higher index = more severe)
# ---------------------------------------------------------------------------
STATUS_NORMAL = "normal"
STATUS_LOW = "low"
STATUS_HIGH = "high"
STATUS_CRITICAL = "critical"
STATUS_UNKNOWN = "unknown_parameter"

# Maps each status to a severity score for roll-up logic
_SEVERITY: dict[str, int] = {
    STATUS_NORMAL: 0,
    STATUS_LOW: 1,
    STATUS_HIGH: 1,
    STATUS_CRITICAL: 2,
    STATUS_UNKNOWN: 0,  # unknown params are noted but don't drive urgency
}

# Maps worst severity score → overall_status string
_OVERALL_MAP: dict[int, str] = {
    0: "normal",
    1: "attention_needed",
    2: "urgent",
}

# ---------------------------------------------------------------------------
# Note template (fully deterministic — no generative model)
# ---------------------------------------------------------------------------

def _build_note(
    parameter: str,
    value: float,
    unit: str,
    status: str,
    min_normal: float | None,
    max_normal: float | None,
) -> str:
    """
    Compose a short, human-readable explanation using an f-string template.

    The text is 100% deterministic — the same inputs always produce the
    exact same output string. No AI/LLM is involved.

    Parameters
    ----------
    parameter   : Display name of the lab parameter.
    value       : The submitted numeric value.
    unit        : Unit string from the reference data.
    status      : One of "normal", "low", "high", "critical".
    min_normal  : Lower bound of the normal range.
    max_normal  : Upper bound of the normal range.
    """
    range_str: str
    if min_normal is not None and max_normal is not None:
        range_str = f"{min_normal}–{max_normal} {unit}"
    elif min_normal is not None:
        range_str = f"≥{min_normal} {unit}"
    elif max_normal is not None:
        range_str = f"≤{max_normal} {unit}"
    else:
        range_str = "not specified"

    # Per-status message fragments
    status_phrases: dict[str, str] = {
        STATUS_NORMAL: (
            f"{parameter} is within the normal range. "
            f"Measured value {value} {unit} falls within the expected range of {range_str}."
        ),
        STATUS_LOW: (
            f"{parameter} is LOW. "
            f"Measured value {value} {unit} is below the normal range of {range_str}. "
            f"Please consult a healthcare professional."
        ),
        STATUS_HIGH: (
            f"{parameter} is HIGH. "
            f"Measured value {value} {unit} is above the normal range of {range_str}. "
            f"Please consult a healthcare professional."
        ),
        STATUS_CRITICAL: (
            f"CRITICAL: {parameter} requires immediate medical attention. "
            f"Measured value {value} {unit} is outside the critical threshold "
            f"(normal range: {range_str})."
        ),
    }
    return status_phrases.get(
        status,
        f"{parameter}: {value} {unit}. Status: {status}. Normal range: {range_str}.",
    )


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------

def _classify_value(
    value: float,
    ref: dict[str, Any],
) -> str:
    """
    Classify a single numeric *value* against one reference-range record.

    Comparison order (matches clinical convention):
        1. Check critical boundaries first — most important to catch.
        2. Then check normal range boundaries.

    Parameters
    ----------
    value : float
        The patient's measured lab value.
    ref   : dict
        One gender-specific reference record with keys:
        unit, min_normal, max_normal, critical_low, critical_high.

    Returns
    -------
    str
        One of "critical", "low", "normal", "high".
    """
    crit_low: float | None = ref.get("critical_low")
    crit_high: float | None = ref.get("critical_high")
    min_normal: float | None = ref.get("min_normal")
    max_normal: float | None = ref.get("max_normal")

    # ── Step 1: Critical check ────────────────────────────────────────────
    # A value below the critical_low threshold is dangerously low → "critical"
    if crit_low is not None and value < crit_low:
        return STATUS_CRITICAL

    # A value above the critical_high threshold is dangerously high → "critical"
    if crit_high is not None and value > crit_high:
        return STATUS_CRITICAL

    # ── Step 2: Normal-range check ────────────────────────────────────────
    # Below min_normal (but above critical_low) → "low"
    if min_normal is not None and value < min_normal:
        return STATUS_LOW

    # Above max_normal (but below critical_high) → "high"
    if max_normal is not None and value > max_normal:
        return STATUS_HIGH

    # Falls within [min_normal, max_normal] (or thresholds undefined) → "normal"
    return STATUS_NORMAL


# ---------------------------------------------------------------------------
# Gender-aware reference lookup
# ---------------------------------------------------------------------------

def _lookup_reference(
    param_key: str,
    gender: str,
    ref_data: dict,
) -> tuple[dict | None, str]:
    """
    Retrieve the most-specific reference record for a parameter + gender pair.

    Lookup strategy (gender fallback):
        1. Try the exact gender string (e.g., "Male", "Female").
        2. Fall back to "Any" if no gender-specific row exists.
        3. Return (None, reason_string) if the parameter isn't in the dataset
           at all — caller will mark it as STATUS_UNKNOWN.

    Parameters
    ----------
    param_key : str
        Normalised parameter key (from _normalise_param()).
    gender    : str
        Caller-supplied gender ("Male" / "Female" / "Any").
    ref_data  : dict
        The full nested reference dict from get_cached_data().

    Returns
    -------
    tuple[dict | None, str]
        (reference_record | None, applied_gender_label)
    """
    gender_map: dict | None = ref_data.get(param_key)
    if gender_map is None:
        # Parameter not found in dataset at all
        return None, "not_found"

    # Normalise gender capitalisation for lookup
    gender_title = gender.strip().title()  # "male" → "Male"

    if gender_title in gender_map:
        return gender_map[gender_title], gender_title

    # Fall back to "Any" if gender-specific range absent
    if "Any" in gender_map:
        return gender_map["Any"], "Any"

    # Dataset has rows for this param but none match — return first available
    first_gender = next(iter(gender_map))
    logger.warning(
        "Parameter '%s': no record for gender '%s' or 'Any'; "
        "using first available gender '%s'.",
        param_key,
        gender,
        first_gender,
    )
    return gender_map[first_gender], first_gender


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze_report(
    lab_values: dict[str, float | int | str],
    gender: str = "Any",
) -> dict:
    """
    Analyse a set of lab results against the cached reference dataset.

    Parameters
    ----------
    lab_values : dict[str, float | int | str]
        Mapping of parameter name → numeric value.
        Keys are matched to the dataset using _normalise_param(), so
        "Hemoglobin", "hemoglobin", and "HEMOGLOBIN" all resolve the same way.
    gender : str, optional
        Patient gender ("Male", "Female", or "Any"). Default "Any".
        Gender-specific ranges are preferred; "Any" is the fallback.

    Returns
    -------
    dict with keys:
        overall_status : "normal" | "attention_needed" | "urgent"
        results        : list of per-parameter dicts
        warnings       : list of warning strings (unknown params, parse errors)

    Raises
    ------
    RuntimeError
        If no reference dataset is loaded in memory yet.
    ValueError
        If lab_values is empty or not a mapping type.
    """
    # ── Pre-flight checks ─────────────────────────────────────────────────
    ref_data = get_cached_data()
    if ref_data is None:
        raise RuntimeError(
            "No reference dataset is loaded. "
            "Upload a CSV first via POST /module3/upload-reference."
        )

    if not isinstance(lab_values, dict):
        raise ValueError(
            f"lab_values must be a dict mapping parameter names to values, "
            f"got {type(lab_values).__name__}."
        )

    if not lab_values:
        raise ValueError("lab_values is empty — nothing to analyse.")

    # ── Initialise accumulators ───────────────────────────────────────────
    results: list[dict] = []
    warnings: list[str] = []
    worst_severity: int = 0  # tracks the worst status across all parameters

    logger.info(
        "Analysing report: %d parameters, gender=%s",
        len(lab_values),
        gender,
    )

    for raw_param, raw_value in lab_values.items():
        param_key: str = _normalise_param(str(raw_param))
        display_name: str = str(raw_param).strip()

        # ── Parse the submitted value ─────────────────────────────────────
        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            msg = (
                f"Could not parse '{raw_param}' value '{raw_value}' as a number; "
                "skipping."
            )
            warnings.append(msg)
            logger.warning(msg)
            continue

        # ── Lookup reference range ────────────────────────────────────────
        ref, applied_gender = _lookup_reference(param_key, gender, ref_data)

        if ref is None:
            # Parameter is not in the dataset — flag as unknown
            results.append(
                {
                    "parameter": display_name,
                    "value": value,
                    "unit": "unknown",
                    "status": STATUS_UNKNOWN,
                    "note": (
                        f"'{display_name}' was not found in the reference dataset. "
                        "No classification is possible without reference thresholds."
                    ),
                }
            )
            warnings.append(
                f"Parameter '{display_name}' is not in the reference dataset."
            )
            logger.warning("Unknown parameter: '%s' (key='%s')", display_name, param_key)
            continue

        # ── Classify the value (pure numeric comparison — no AI) ──────────
        status: str = _classify_value(value, ref)

        # Update worst-severity tracker for overall_status roll-up
        worst_severity = max(worst_severity, _SEVERITY.get(status, 0))

        # ── Build deterministic note ──────────────────────────────────────
        note: str = _build_note(
            parameter=display_name,
            value=value,
            unit=ref["unit"],
            status=status,
            min_normal=ref.get("min_normal"),
            max_normal=ref.get("max_normal"),
        )

        # Annotate if we fell back from the requested gender to "Any"
        if applied_gender != gender and applied_gender == "Any":
            note += f" (Gender-specific range not available; 'Any' range applied.)"

        results.append(
            {
                "parameter": display_name,
                "value": value,
                "unit": ref["unit"],
                "status": status,
                "note": note,
            }
        )

        logger.debug(
            "  %s = %s %s → %s (gender used: %s)",
            display_name,
            value,
            ref["unit"],
            status,
            applied_gender,
        )

    # ── Compute overall_status from worst individual status ───────────────
    overall_status: str = _OVERALL_MAP.get(worst_severity, "normal")

    logger.info("Analysis complete. Overall status: %s", overall_status)

    return {
        "overall_status": overall_status,
        "results": results,
        "warnings": warnings,
    }
