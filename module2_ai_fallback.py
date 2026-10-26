"""
ai_fallback.py
==============
Google Gemini AI fallback for VitalSense AI — Module 2.

When a disease is NOT found in the local CSV datasets, this module calls the
Gemini API and returns a structured recommendation containing:
  - precautions
  - medications (OTC category only — no dosages/brand names)
  - specialist type
  - diet recommendations
  - a disclaimer

How to set your API key
-----------------------
Option 1 (recommended): set the GEMINI_API_KEY environment variable:
    Windows PowerShell:
        $env:GEMINI_API_KEY = "AIza..."
    .env file (auto-loaded if python-dotenv is installed):
        GEMINI_API_KEY=AIza...

Option 2: pass the key directly to get_ai_recommendation(api_key="AIza...").

Get a free Gemini API key at: https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Gemini model to use — gemini-1.5-flash is free-tier and fast
GEMINI_MODEL = "gemini-2.5-flash"

# Disclaimer always appended to AI responses (same as dataset-based responses)
DISCLAIMER = (
    "This is an AI-generated suggestion, not a medical prescription. "
    "Please consult a licensed doctor before taking any medication."
)

# The structured JSON prompt sent to Gemini
_PROMPT_TEMPLATE = """
You are a medical information assistant for the VitalSense AI system.
A patient has been diagnosed with: "{disease}"
Their symptom severity level is: {severity} (mild / moderate / severe)

Return ONLY a valid JSON object with exactly these keys — no extra text, no markdown:

{{
  "precautions": ["**Action Title**: Detailed clinical guideline...", ...],
  "medications": ["OTC category or drug class (no brand names, no dosages)", ...],
  "specialist_type": "name of the appropriate medical specialist",
  "diet": {{
    "eat":   ["specific food or dietary habit to follow 1", "... 2", "... 3", "... 4", "... 5", "... 6"],
    "avoid": ["specific food or habit to strictly avoid 1", "... 2", "... 3", "... 4", "... 5"]
  }},
  "urgency": "none" | "same-day" | "emergency"
}}

Rules:
- precautions: 6 to 8 comprehensive, practical self-care steps with bold action directives (e.g. "**Symptom Relief**: ...", "**Emergency Warning**: ...").
- medications: list OTC/general drug categories only (e.g. "Antihistamines", 
  "NSAIDs") — NEVER include brand names or dosage amounts.
  If severity is moderate or severe, return an empty list [] for medications
  (patient must see a doctor).
- specialist_type: single most relevant specialist (e.g. "Pulmonologist").
- diet.eat: 6 specific foods, food groups, or dietary patterns that support recovery.
- diet.avoid: 5 foods or habits that worsen or interfere with recovery.
- urgency: "none" if mild, "same-day" if moderate, "emergency" if severe.

Respond with ONLY the JSON — no explanation, no markdown fences.
"""


def _extract_json(raw: str) -> dict:
    """
    Extract a JSON object from a Gemini response string.
    Handles cases where the model wraps the JSON in markdown fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract {...} block via regex as last resort
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse Gemini response as JSON.\nRaw: {raw[:300]}")


def get_ai_recommendation(
    disease: str,
    severity: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Call the Gemini API to get a structured medical recommendation for a disease
    that is not present in the local CSV datasets.

    Parameters
    ----------
    disease  : The predicted disease name (any string — not limited to datasets).
    severity : "mild", "moderate", or "severe".
    api_key  : Gemini API key.  If None, reads from GEMINI_API_KEY env variable.

    Returns
    -------
    dict with keys matching the standard Module 2 response shape:
        {
          "disease":             str,
          "severity":            str,
          "precautions":         list[str],
          "medications":         list[str] | null,
          "doctor_referral":     {"specialist": str, "urgency": str} | null,
          "diet_recommendation": list[str],
          "disclaimer":          str,
          "source":              "gemini_ai"   ← marks AI-generated response
        }

    Raises
    ------
    EnvironmentError  : If no API key is found.
    RuntimeError      : If the Gemini call fails or returns unparseable output.
    """
    import google.generativeai as genai

    # Resolve API key
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not resolved_key:
        raise EnvironmentError(
            "Gemini API key not found. Set the GEMINI_API_KEY environment variable "
            "or pass api_key= to get_ai_recommendation(). "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=resolved_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = _PROMPT_TEMPLATE.format(disease=disease, severity=severity)
    logger.info("Calling Gemini (%s) for disease='%s', severity='%s'.", GEMINI_MODEL, disease, severity)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    logger.debug("Gemini raw response: %s", raw_text[:500])

    try:
        parsed = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to parse Gemini response: {exc}") from exc

    # Build standard response shape
    medications  = parsed.get("medications") or []
    specialist   = parsed.get("specialist_type", "General Practitioner")
    urgency      = parsed.get("urgency", "none")
    precautions  = parsed.get("precautions", [])

    # Diet: accept {eat, avoid} object (new prompt) or flat list (legacy)
    raw_diet = parsed.get("diet") or parsed.get("diet_recommendation") or {}
    if isinstance(raw_diet, dict):
        diet_recommendation = {
            "eat":   raw_diet.get("eat", []),
            "avoid": raw_diet.get("avoid", []),
        }
    elif isinstance(raw_diet, list):
        diet_recommendation = {"eat": raw_diet, "avoid": []}
    else:
        diet_recommendation = {"eat": [], "avoid": []}

    # Apply the same severity gate as dataset-based logic:
    # moderate/severe → suppress medications, activate referral
    if severity == "mild":
        final_medications  = medications if medications else None
        doctor_referral    = None
    else:
        final_medications  = None
        doctor_referral    = {
            "specialist": specialist,
            "urgency":    urgency if urgency != "none" else (
                "same-day" if severity == "moderate" else "emergency"
            ),
        }

    result = {
        "disease":             disease,
        "severity":            severity,
        "precautions":         precautions or None,
        "medications":         final_medications,
        "doctor_referral":     doctor_referral,
        "diet_recommendation": diet_recommendation or None,
        "disclaimer":          DISCLAIMER,
        "source":              "gemini_ai",   # clearly marks AI-generated response
    }

    logger.info(
        "Gemini recommendation built for disease='%s': "
        "meds=%s, referral=%s.",
        disease, bool(final_medications), bool(doctor_referral),
    )
    return result


def is_gemini_available() -> bool:
    """
    Return True if the google-generativeai package is installed AND
    a GEMINI_API_KEY environment variable is set.
    """
    try:
        import google.generativeai  # noqa: F401
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())
    except ImportError:
        return False
