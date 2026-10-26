"""
herbal_ai.py — VitalSense AI | Module 7: Gemini AI Fallback for Herbal Remedies
=================================================================================
When a disease is NOT in the local herbal_kb, this module queries Gemini AI
for natural/herbal remedy suggestions.

Follows the same pattern as module2/ai_fallback.py.

Author: VitalSense AI Team
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Gemini API configuration
# ──────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.0-flash"
GEMINI_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

DISCLAIMER: str = (
    "These AI-generated herbal suggestions are for informational purposes only. "
    "They are NOT a substitute for professional medical advice. Always consult "
    "a qualified healthcare practitioner before using any herbal remedy."
)


def is_available() -> bool:
    """Check if Gemini AI is available (API key is set)."""
    return bool(GEMINI_API_KEY.strip())


def suggest_herbal_remedies(disease: str, symptoms: list[str] | None = None) -> Optional[dict[str, Any]]:
    """
    Query Gemini AI for herbal/natural remedies for a disease.

    Returns a dict with:
      - remedies: list of remedy objects
      - ai_generated: True
      - disclaimer: str
    Or None if the API call fails.
    """
    if not is_available():
        logger.warning("[Module 7 AI] GEMINI_API_KEY not set — AI fallback unavailable.")
        return None

    symptom_context = ""
    if symptoms:
        symptom_context = f"\nThe patient's symptoms include: {', '.join(symptoms)}."

    prompt = f"""You are a knowledgeable herbalist and naturopath. A patient has been diagnosed with or is experiencing: "{disease}".{symptom_context}

Provide 3-5 natural/herbal remedies that may help with this condition. Focus on well-known medicinal plants and traditional remedies that have some scientific backing.

For each remedy, provide:
1. herb_name: Common name of the herb/plant (include Indian name if applicable)
2. scientific_name: Scientific/botanical name
3. preparation: Detailed step-by-step preparation method
4. dosage: Recommended dosage and frequency
5. benefits: How it helps with this specific condition
6. precautions: Safety warnings, contraindications, drug interactions
7. evidence_level: One of "Well-Studied", "Preliminary Studies", or "Traditional"

IMPORTANT RULES:
- Only suggest remedies for MILD conditions
- If the disease is serious/life-threatening, respond with an empty remedies array and set "serious_condition" to true
- Always include safety precautions
- Never suggest replacing prescribed medication with herbs
- Focus on Indian/Ayurvedic herbs where applicable, but include global herbs too

Respond ONLY with valid JSON in this exact format:
{{
    "disease": "{disease}",
    "serious_condition": false,
    "remedies": [
        {{
            "herb_name": "...",
            "scientific_name": "...",
            "emoji": "🌿",
            "preparation": "...",
            "dosage": "...",
            "benefits": "...",
            "precautions": "...",
            "evidence_level": "..."
        }}
    ]
}}"""

    try:
        import urllib.request

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048,
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        # Extract text from Gemini response
        text = raw["candidates"][0]["content"]["parts"][0]["text"]

        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        parsed = json.loads(text)

        # Validate structure
        if not isinstance(parsed.get("remedies"), list):
            logger.warning("[Module 7 AI] Invalid response structure from Gemini.")
            return None

        return {
            "disease": disease,
            "remedies": parsed["remedies"],
            "serious_condition": parsed.get("serious_condition", False),
            "ai_generated": True,
            "disclaimer": DISCLAIMER,
        }

    except Exception as exc:
        logger.error("[Module 7 AI] Gemini API call failed: %s", exc)
        return None
