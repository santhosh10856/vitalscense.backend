"""
report_text_analyzer.py — VitalSense AI · Module 3
===================================================
**Dataset-based** diagnostic report text analysis engine.

NO AI / LLM / API is used anywhere in this file — every decision is made by:
  1. Regex-based extraction of parameter names and numeric values from the
     pasted report text.
  2. Fuzzy matching of extracted parameter names against the reference CSV
     (reference_ranges.csv) already loaded by reference_data.py.
  3. Rule-based classification of each extracted value using the same
     _classify_value() logic in report_analyzer.py.

This approach is fully offline, deterministic, and gives report-specific
results because it actually reads and interprets the numbers in the text.

Design
------
  • _PARAM_ALIASES maps common report variations (e.g. "Hb", "SGPT",
    "Blood Sugar Fasting") to the canonical key used in reference_ranges.csv.
  • Extraction tries multiple regex patterns to handle the wide variety of
    lab report formats (colon-separated, tab-separated, etc.).
  • The public function summarize_report_text() returns the same JSON shape
    expected by the frontend's renderDiagnosticResult() function.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from reference_data import get_cached_data, _normalise_param
from report_analyzer import (
    _classify_value,
    _lookup_reference,
    _build_note,
    STATUS_NORMAL,
    STATUS_LOW,
    STATUS_HIGH,
    STATUS_CRITICAL,
    _SEVERITY,
)

logger = logging.getLogger("module3.report_text_analyzer")

# ---------------------------------------------------------------------------
# Minimum text length to attempt analysis
# ---------------------------------------------------------------------------
MIN_TEXT_LENGTH: int = 20

# ---------------------------------------------------------------------------
# Alias map: common lab report names → canonical reference_ranges.csv key
# ---------------------------------------------------------------------------
# Keys must be lowercase.  Values must be the _normalise_param() form of the
# Parameter column in reference_ranges.csv.
_PARAM_ALIASES: dict[str, str] = {
    # Haemoglobin
    "hb":                       "hemoglobin",
    "haemoglobin":              "hemoglobin",
    "hemoglobin":               "hemoglobin",
    "hgb":                      "hemoglobin",
    # Hematocrit
    "hct":                      "hematocrit",
    "hematocrit":               "hematocrit",
    "haematocrit":              "hematocrit",
    "pcv":                      "hematocrit",
    "packed cell volume":       "hematocrit",
    # WBC
    "wbc":                      "wbc_count",
    "wbc count":                "wbc_count",
    "white blood cells":        "wbc_count",
    "white blood cell count":   "wbc_count",
    "total wbc":                "wbc_count",
    "total leucocyte count":    "wbc_count",
    "tlc":                      "wbc_count",
    # RBC
    "rbc":                      "rbc_count",
    "rbc count":                "rbc_count",
    "red blood cells":          "rbc_count",
    "red blood cell count":     "rbc_count",
    # Platelets
    "platelets":                "platelet_count",
    "platelet count":           "platelet_count",
    "plt":                      "platelet_count",
    # MCV / MCH / MCHC
    "mcv":                      "mcv",
    "mch":                      "mch",
    "mchc":                     "mchc",
    # Differentials
    "neutrophils":              "neutrophils",
    "lymphocytes":              "lymphocytes",
    "monocytes":                "monocytes",
    "eosinophils":              "eosinophils",
    "basophils":                "basophils",
    # Glucose
    "fasting glucose":          "fasting_glucose",
    "fasting blood glucose":    "fasting_glucose",
    "fasting blood sugar":      "fasting_glucose",
    "fbs":                      "fasting_glucose",
    "fbg":                      "fasting_glucose",
    "random glucose":           "random_glucose",
    "random blood sugar":       "random_glucose",
    "rbs":                      "random_glucose",
    "blood sugar":              "fasting_glucose",
    "blood glucose":            "fasting_glucose",
    # HbA1c
    "hba1c":                    "hba1c",
    "a1c":                      "hba1c",
    "glycated hemoglobin":      "hba1c",
    "glycosylated hemoglobin":  "hba1c",
    # Electrolytes
    "sodium":                   "sodium",
    "na":                       "sodium",
    "na+":                      "sodium",
    "potassium":                "potassium",
    "k":                        "potassium",
    "k+":                       "potassium",
    "chloride":                 "chloride",
    "cl":                       "chloride",
    "bicarbonate":              "bicarbonate",
    "hco3":                     "bicarbonate",
    # Renal
    "bun":                      "blood_urea_nitrogen",
    "blood urea nitrogen":      "blood_urea_nitrogen",
    "blood urea":               "blood_urea_nitrogen",
    "urea":                     "blood_urea_nitrogen",
    "creatinine":               "creatinine",
    "serum creatinine":         "creatinine",
    "s. creatinine":            "creatinine",
    "egfr":                     "egfr",
    "gfr":                      "egfr",
    "uric acid":                "uric_acid",
    "serum uric acid":          "uric_acid",
    # Lipid Profile
    "total cholesterol":        "total_cholesterol",
    "cholesterol":              "total_cholesterol",
    "t. cholesterol":           "total_cholesterol",
    "ldl":                      "ldl_cholesterol",
    "ldl cholesterol":          "ldl_cholesterol",
    "ldl-c":                    "ldl_cholesterol",
    "hdl":                      "hdl_cholesterol",
    "hdl cholesterol":          "hdl_cholesterol",
    "hdl-c":                    "hdl_cholesterol",
    "triglycerides":            "triglycerides",
    "tg":                       "triglycerides",
    "trigs":                    "triglycerides",
    "vldl":                     "vldl_cholesterol",
    "vldl cholesterol":         "vldl_cholesterol",
    # Liver
    "total bilirubin":          "total_bilirubin",
    "t. bilirubin":             "total_bilirubin",
    "bilirubin total":          "total_bilirubin",
    "bilirubin":                "total_bilirubin",
    "direct bilirubin":         "direct_bilirubin",
    "d. bilirubin":             "direct_bilirubin",
    "indirect bilirubin":       "indirect_bilirubin",
    "alt":                      "alt",
    "sgpt":                     "alt",
    "sgpt/alt":                 "alt",
    "alanine aminotransferase": "alt",
    "alanine transaminase":     "alt",
    "ast":                      "ast",
    "sgot":                     "ast",
    "sgot/ast":                 "ast",
    "aspartate aminotransferase": "ast",
    "aspartate transaminase":   "ast",
    "alp":                      "alp",
    "alkaline phosphatase":     "alp",
    "ggt":                      "ggt",
    "gamma gt":                 "ggt",
    "total protein":            "total_protein",
    "albumin":                  "albumin",
    "serum albumin":            "albumin",
    "globulin":                 "globulin",
    # Thyroid
    "tsh":                      "tsh",
    "thyroid stimulating hormone": "tsh",
    "free t3":                  "free_t3",
    "ft3":                      "free_t3",
    "t3":                       "free_t3",
    "free t4":                  "free_t4",
    "ft4":                      "free_t4",
    "t4":                       "free_t4",
    # Minerals
    "calcium":                  "calcium",
    "serum calcium":            "calcium",
    "ca":                       "calcium",
    "phosphorus":               "phosphorus",
    "phosphate":                "phosphorus",
    "magnesium":                "magnesium",
    "mg":                       "magnesium",
    # Iron
    "iron":                     "iron",
    "serum iron":               "iron",
    "tibc":                     "tibc",
    "ferritin":                 "ferritin",
    "serum ferritin":           "ferritin",
    # Vitamins
    "vitamin d":                "vitamin_d",
    "vit d":                    "vitamin_d",
    "25-oh vitamin d":          "vitamin_d",
    "vitamin b12":              "vitamin_b12",
    "vit b12":                  "vitamin_b12",
    "b12":                      "vitamin_b12",
    "folate":                   "folate",
    "folic acid":               "folate",
    # Inflammation
    "crp":                      "crp",
    "c-reactive protein":       "crp",
    "esr":                      "esr",
    "sed rate":                 "esr",
    "erythrocyte sedimentation rate": "esr",
    # PSA
    "psa":                      "psa",
    # Hormones
    "cortisol":                 "cortisol",
    "insulin":                  "insulin",
    "fasting insulin":          "insulin",
    # Coagulation
    "pt":                       "pt",
    "prothrombin time":         "pt",
    "inr":                      "inr",
    "aptt":                     "aptt",
    "ptt":                      "aptt",
    # ABG
    "ph":                       "ph",
    "blood ph":                 "ph",
    "pco2":                     "pco2",
    "po2":                      "po2",
}

# ---------------------------------------------------------------------------
# Severity → plain-language mapping for the frontend
# ---------------------------------------------------------------------------
_SEVERITY_LABELS: dict[str, str] = {
    STATUS_NORMAL:   "normal",
    STATUS_LOW:      "moderate",
    STATUS_HIGH:     "moderate",
    STATUS_CRITICAL: "severe",
}

_OVERALL_MAP: dict[int, str] = {
    0: "normal",
    1: "attention_needed",
    2: "urgent",
}

DISCLAIMER = (
    "This analysis is based on reference ranges from the loaded dataset. "
    "It does not replace your doctor's interpretation. "
    "Please discuss these results with your physician."
)


# ---------------------------------------------------------------------------
# Text parsing — extract parameter:value pairs from free-text reports
# ---------------------------------------------------------------------------

def _extract_lab_values(text: str) -> list[tuple[str, float, str]]:
    """
    Extract (parameter_name, numeric_value, original_line) triples from
    free-form diagnostic report text.

    Handles common formats:
      • "Hemoglobin: 10.2 g/dL"
      • "Hemoglobin    10.2   g/dL   (Ref: 13.5-17.5)"
      • "Hemoglobin - 10.2 g/dL"
      • "Hemoglobin = 10.2"
      • "HbA1c: 6.8 %"
      • Lines with multiple separators (tabs, pipes, etc.)

    Returns a list of (param_name, value, original_line_text).
    """
    results: list[tuple[str, float, str]] = []
    seen_params: set[str] = set()

    lines = text.strip().splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        # Skip header/title lines (no numbers at all)
        if not re.search(r'\d', stripped):
            continue

        # Try to match: PARAM_NAME <separator> NUMBER
        # Separators: colon, equals, dash, tab, multiple spaces, pipe
        match = re.match(
            r'^(.+?)'                         # parameter name (non-greedy)
            r'[\s]*[:=\-–—|\t][\s]*'          # separator
            r'(\d+[,.]?\d*)'                  # numeric value
            r'(.*)$',                          # rest of line
            stripped,
        )

        if not match:
            # Try format: PARAM_NAME  <spaces>  NUMBER (no explicit separator)
            match = re.match(
                r'^([a-zA-Z][a-zA-Z0-9\s\.\-/\(\)]+?)'  # param name starting with letter
                r'\s{2,}'                                  # 2+ spaces as separator
                r'(\d+[,.]?\d*)'                          # numeric value
                r'(.*)$',
                stripped,
            )

        if not match:
            continue

        raw_name = match.group(1).strip().rstrip(':=-–— \t')
        raw_value = match.group(2).replace(',', '')

        # Skip if the "name" is just a number or too short
        if len(raw_name) < 2 or re.match(r'^\d+$', raw_name):
            continue

        try:
            value = float(raw_value)
        except ValueError:
            continue

        # Deduplicate: only take the first occurrence of each parameter
        norm_name = raw_name.lower().strip()
        if norm_name in seen_params:
            continue
        seen_params.add(norm_name)

        results.append((raw_name, value, stripped))

    return results


def _resolve_param_key(raw_name: str) -> str | None:
    """
    Try to match a raw parameter name to a canonical key in the reference
    dataset, using the alias table and fuzzy normalisation.

    Returns the canonical key, or None if no match is found.
    """
    lower = raw_name.strip().lower()

    # 1. Direct alias lookup
    if lower in _PARAM_ALIASES:
        return _PARAM_ALIASES[lower]

    # 2. Normalised form (same as reference_data._normalise_param)
    normalised = _normalise_param(raw_name)

    # Check if normalised key exists directly in the reference data
    ref_data = get_cached_data()
    if ref_data and normalised in ref_data:
        return normalised

    # 3. Try partial matches in alias table
    for alias, canonical in _PARAM_ALIASES.items():
        if alias in lower or lower in alias:
            return canonical

    # 4. Try partial matches against reference data keys
    if ref_data:
        for ref_key in ref_data:
            # Check if the normalised name is a substring of the ref key or vice versa
            if normalised in ref_key or ref_key in normalised:
                return ref_key

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_report_text(raw_text: str) -> dict[str, Any]:
    """
    Analyse diagnostic report text by extracting lab values and classifying
    them against the reference dataset. NO AI/API is used.

    Parameters
    ----------
    raw_text : str
        The raw text of a diagnostic / lab report (pasted by the user).

    Returns
    -------
    dict
        Response matching the frontend's expected shape:
            findings         : list of finding dicts
            overall_summary  : str
            overall_severity : "normal" | "attention_needed" | "urgent"
            disclaimer       : str

    Raises
    ------
    ValueError
        If raw_text is empty, too short, or no lab values could be extracted.
    RuntimeError
        If no reference dataset is loaded.
    """
    # ── Input validation ─────────────────────────────────────────────────────
    if not raw_text or not raw_text.strip():
        raise ValueError("Report text cannot be empty.")

    cleaned = raw_text.strip()
    if len(cleaned) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"Report text is too short ({len(cleaned)} characters). "
            f"Please provide at least {MIN_TEXT_LENGTH} characters."
        )

    # ── Check that reference data is loaded ──────────────────────────────────
    ref_data = get_cached_data()
    if ref_data is None:
        raise RuntimeError(
            "No reference dataset is loaded. "
            "Upload a CSV first via POST /module3/upload-reference."
        )

    logger.info("Analysing diagnostic report text (%d chars).", len(cleaned))

    # ── Try to detect gender from text ───────────────────────────────────────
    gender = _detect_gender(cleaned)

    # ── Extract lab values from text ─────────────────────────────────────────
    extracted = _extract_lab_values(cleaned)

    if not extracted:
        raise ValueError(
            "Could not extract any lab values from the report text. "
            "Please ensure your report contains parameter names with numeric values "
            "(e.g., 'Hemoglobin: 12.5 g/dL')."
        )

    logger.info("Extracted %d lab values from report text.", len(extracted))

    # ── Classify each extracted value ────────────────────────────────────────
    findings: list[dict] = []
    worst_severity: int = 0
    matched_count: int = 0
    unmatched_params: list[str] = []
    extracted_param_values: dict[str, float] = {}  # for condition prediction

    for raw_name, value, original_line in extracted:
        param_key = _resolve_param_key(raw_name)

        if param_key is None:
            # Parameter not in our dataset — still report it
            unmatched_params.append(raw_name)
            findings.append({
                "original_text": original_line,
                "plain_language": (
                    f"{raw_name} has a value of {value}. "
                    f"This parameter is not in our reference database, "
                    f"so we cannot determine if it's normal or abnormal."
                ),
                "severity": "normal",
            })
            continue

        # Store for condition prediction
        extracted_param_values[param_key] = value

        # Look up the reference range
        ref, applied_gender = _lookup_reference(param_key, gender, ref_data)
        if ref is None:
            unmatched_params.append(raw_name)
            continue

        # Classify
        status = _classify_value(value, ref)
        worst_severity = max(worst_severity, _SEVERITY.get(status, 0))
        matched_count += 1

        # Build finding
        display_name = raw_name.strip()
        unit = ref.get("unit", "")
        min_n = ref.get("min_normal")
        max_n = ref.get("max_normal")
        range_str = f"{min_n}–{max_n} {unit}" if min_n is not None and max_n is not None else "N/A"

        plain = _build_plain_language(display_name, value, unit, status, range_str)

        findings.append({
            "original_text": original_line,
            "plain_language": plain,
            "severity": _SEVERITY_LABELS.get(status, "normal"),
        })

    # ── Condition prediction using diagnostic & laboratory datasets ───────
    condition_predictions: list[dict] = []
    try:
        from condition_predictor import predict_conditions, is_available
        if is_available() and extracted_param_values:
            condition_predictions = predict_conditions(extracted_param_values, gender)
            logger.info(
                "Condition prediction returned %d result(s).", len(condition_predictions)
            )

            # Add condition predictions as additional findings
            for pred in condition_predictions:
                confidence_pct = round(pred["confidence"] * 100)
                source_label = (
                    "Diagnostic Dataset Analysis" if pred["source"] == "diagnostic_dataset"
                    else "Laboratory Dataset Analysis"
                )

                # Build top predictions string
                top_str = ", ".join(
                    f"{t['condition']} ({round(t['probability'] * 100)}%)"
                    for t in pred["top_3"]
                )

                # Upgrade overall severity if prediction is worse
                sev_map = {"normal": 0, "mild": 1, "moderate": 1, "severe": 2}
                pred_sev_rank = sev_map.get(pred["severity"], 0)
                worst_severity = max(worst_severity, pred_sev_rank)

                findings.append({
                    "original_text": f"[{source_label}] Predicted condition based on {pred['features_used']}/{pred['features_total']} matched parameters",
                    "plain_language": (
                        f"Based on your lab values, the {source_label.lower()} predicts: "
                        f"{pred['condition']} (confidence: {confidence_pct}%). "
                        f"{pred['description']} "
                        f"Top predictions: {top_str}. "
                        f"{pred['recommendation']}"
                    ),
                    "severity": pred["severity"],
                })
    except ImportError:
        logger.warning("condition_predictor module not available; skipping prediction.")
    except Exception as exc:
        logger.warning("Condition prediction failed: %s", exc)

    # ── Build overall summary ────────────────────────────────────────────────
    overall_severity = _OVERALL_MAP.get(worst_severity, "normal")
    overall_summary = _build_overall_summary(
        findings, matched_count, unmatched_params, overall_severity,
        condition_predictions,
    )

    logger.info(
        "Report analysis complete: %d findings, %d matched, %d conditions, severity='%s'.",
        len(findings), matched_count, len(condition_predictions), overall_severity,
    )

    return {
        "findings": findings,
        "overall_summary": overall_summary,
        "overall_severity": overall_severity,
        "condition_predictions": condition_predictions,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_gender(text: str) -> str:
    """Try to detect patient gender from report text. Default to 'Any'."""
    lower = text.lower()
    # Look for gender indicators
    male_patterns = [
        r'\bmale\b', r'\bgender\s*:\s*m\b', r'\bsex\s*:\s*m\b',
        r'\bgender\s*:\s*male\b', r'\bsex\s*:\s*male\b',
    ]
    female_patterns = [
        r'\bfemale\b', r'\bgender\s*:\s*f\b', r'\bsex\s*:\s*f\b',
        r'\bgender\s*:\s*female\b', r'\bsex\s*:\s*female\b',
    ]

    # Check female FIRST (because "female" contains "male")
    for pat in female_patterns:
        if re.search(pat, lower):
            return "Female"
    for pat in male_patterns:
        if re.search(pat, lower):
            return "Male"

    return "Any"


def _build_plain_language(
    name: str, value: float, unit: str, status: str, range_str: str
) -> str:
    """Build a patient-friendly explanation of a single finding."""
    if status == STATUS_NORMAL:
        return (
            f"Your {name} is {value} {unit}, which is within the normal range "
            f"({range_str}). This is a healthy result."
        )
    elif status == STATUS_LOW:
        return (
            f"Your {name} is {value} {unit}, which is below the normal range "
            f"({range_str}). This may need attention — consult your doctor."
        )
    elif status == STATUS_HIGH:
        return (
            f"Your {name} is {value} {unit}, which is above the normal range "
            f"({range_str}). This may need attention — consult your doctor."
        )
    elif status == STATUS_CRITICAL:
        return (
            f"Your {name} is {value} {unit}, which is significantly outside "
            f"the normal range ({range_str}). This requires immediate medical "
            f"attention. Please consult your doctor urgently."
        )
    return f"Your {name} is {value} {unit}. Normal range: {range_str}."


def _build_overall_summary(
    findings: list[dict],
    matched: int,
    unmatched: list[str],
    severity: str,
    condition_predictions: list[dict] | None = None,
) -> str:
    """Build a one-line overall summary of the report analysis."""
    total = len(findings)

    # Don't count condition prediction findings as "parameters"
    param_findings = [
        f for f in findings
        if not f.get("original_text", "").startswith("[")
    ]
    param_total = len(param_findings)
    normal_count = sum(1 for f in param_findings if f["severity"] == "normal")
    abnormal_count = param_total - normal_count

    parts: list[str] = []

    if param_total == 0:
        return "No lab values could be identified in the report text."

    parts.append(f"Analysed {param_total} parameter(s) from your report.")

    if abnormal_count == 0:
        parts.append("All values are within normal ranges.")
    elif abnormal_count == 1:
        parts.append("1 parameter is outside the normal range.")
    else:
        parts.append(f"{abnormal_count} parameters are outside normal ranges.")

    # Add condition prediction summary
    if condition_predictions:
        conditions = [
            p["condition"] for p in condition_predictions
            if p["condition"] not in ("Fit",)
        ]
        if conditions:
            parts.append(
                f"Possible condition(s) detected: {', '.join(conditions)}."
            )
        else:
            parts.append("No significant conditions predicted.")

    if severity == "urgent":
        parts.append("Some values require immediate medical attention.")
    elif severity == "attention_needed":
        parts.append("Please discuss these results with your doctor.")

    if unmatched:
        parts.append(
            f"{len(unmatched)} parameter(s) could not be matched to our reference database."
        )

    return " ".join(parts)
