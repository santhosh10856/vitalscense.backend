"""
module1_ai_fallback.py — VitalSense AI | Module 1: AI Disease Prediction Fallback
===================================================================================
When a user enters symptoms that are NOT in the training dataset (zero recognised
symptoms), this module calls the Gemini API to predict the most likely disease.

The response shape is intentionally identical to predict_disease() in predict.py:
    {
        "disease":    str,      # predicted disease name
        "confidence": float,    # AI-estimated confidence (0.0-1.0)
        "warnings":   list,     # symptoms passed (all were unrecognised)
        "source":     "ai",     # marks this as AI-generated (not ML model)
    }

How to set your Gemini API key
-------------------------------
Option 1 (recommended): set the GEMINI_API_KEY environment variable
    Windows PowerShell:
        $env:GEMINI_API_KEY = "AIza..."
    .env file (auto-loaded if python-dotenv is installed):
        GEMINI_API_KEY=AIza...

Option 2: pass the key directly when calling ai_predict_disease(api_key="AIza...").

Get a free key at: https://aistudio.google.com/app/apikey

Author : VitalSense AI Team
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Gemini model - gemini-1.5-flash is free-tier and very fast
GEMINI_MODEL = "gemini-2.5-flash"

# --------------------------------------------------------------------------- #
# Structured prompt sent to Gemini                                             #
# --------------------------------------------------------------------------- #
_PROMPT_TEMPLATE = """\
You are a medical AI assistant for the VitalSense AI system.

A patient has described the following symptoms:
{symptoms_list}

Based ONLY on these symptoms, predict the single most likely disease or medical condition.

Return ONLY a valid JSON object - no markdown, no extra text - with exactly these keys:

{{
  "disease":    "Name of the most likely disease or condition",
  "confidence": 0.82,
  "reasoning":  "One-sentence plain-English explanation of why these symptoms suggest this disease."
}}

Rules:
- "disease": use a clear, common clinical name (e.g. "Common Cold", "Migraine", "Type 2 Diabetes").
- "confidence": a float between 0.50 and 0.95 reflecting how strongly the symptoms point to one condition.
  Use lower values (0.50-0.65) when symptoms are generic, higher (0.80-0.95) when the pattern is specific.
- "reasoning": plain English, one sentence, no jargon.
- Respond with ONLY the JSON - no explanation, no markdown fences.
"""


# --------------------------------------------------------------------------- #
# Internal helper                                                               #
# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> dict[str, Any]:
    """
    Parse a JSON object from a Gemini response string.
    Handles markdown code fences the model sometimes adds despite instructions.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: pull the first {...} block with regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(
            f"Could not parse Gemini response as JSON.\nRaw: {raw[:400]}"
        )


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #
def ai_predict_disease(
    symptoms: list[str],
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Ask Gemini to predict a disease from a free-form symptom list.

    This is called by predict.py when ZERO symptoms are recognised by the
    trained ML model - i.e., the user described symptoms that don't exist in
    the training dataset.

    Parameters
    ----------
    symptoms : list[str]
        Raw symptom strings as entered by the user (the "warnings" from
        predict.py that would otherwise cause a ValueError).
    api_key  : str | None
        Gemini API key.  If None, reads GEMINI_API_KEY from the environment.

    Returns
    -------
    dict with keys:
        "disease"    : str   - predicted disease name
        "confidence" : float - AI-estimated confidence (0.0-1.0)
        "warnings"   : list  - the original unrecognised symptom strings
        "source"     : "ai"  - tells the frontend to show the AI badge

    Raises
    ------
    EnvironmentError
        If no API key is available.
    RuntimeError
        If the Gemini API call fails or returns unparseable output.
    """
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        ) from exc

    # ---- Resolve API key -------------------------------------------------- #
    resolved_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not resolved_key:
        raise EnvironmentError(
            "Gemini API key not found. Set the GEMINI_API_KEY environment variable "
            "or pass api_key= to ai_predict_disease(). "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=resolved_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # ---- Format symptom list for the prompt -------------------------------- #
    symptoms_text = "\n".join(
        f"  - {s.strip().replace('_', ' ').title()}" for s in symptoms
    )
    prompt = _PROMPT_TEMPLATE.format(symptoms_list=symptoms_text)

    logger.info(
        "Calling Gemini (%s) AI fallback for %d unrecognised symptom(s): %s",
        GEMINI_MODEL, len(symptoms), symptoms,
    )

    # ---- Call Gemini ------------------------------------------------------- #
    try:
        response = model.generate_content(prompt)
        raw_text = response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    logger.debug("Gemini raw response: %.300s", raw_text)

    # ---- Parse response ---------------------------------------------------- #
    try:
        parsed = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to parse Gemini response: {exc}") from exc

    disease    = str(parsed.get("disease", "Unknown condition")).strip()
    confidence = float(parsed.get("confidence", 0.65))
    # Clamp to a sensible range
    confidence = round(max(0.0, min(confidence, 0.95)), 4)

    logger.info(
        "Gemini AI predicted disease='%s' with confidence=%.4f",
        disease, confidence,
    )

    return {
        "disease":    disease,
        "confidence": confidence,
        "warnings":   symptoms,   # all input symptoms were unrecognised by ML
        "source":     "ai",       # frontend uses this to show the AI badge
    }


def is_gemini_available() -> bool:
    """
    Return True if google-generativeai is installed AND GEMINI_API_KEY is set.
    """
    try:
        import google.generativeai  # noqa: F401
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())
    except ImportError:
        return False
