"""
recommend.py
============
Core recommendation engine for VitalSense AI — Module 2.

Lookup priority (for each field independently):
  1. Local CSV datasets (fast, no network call, curated data).
  2. Gemini AI fallback — called ONLY when ALL dataset fields are missing
     for a disease (i.e. disease is completely unknown to the local data).
     If GEMINI_API_KEY is not set, uses the local OTC symptom-based fallback.

Severity rules (unchanged):
  - mild     → medications shown, no doctor referral.
  - moderate → medications suppressed, doctor referral (same-day).
  - severe   → medications suppressed, doctor referral (emergency).
  Diet + precautions are always returned.
"""

from __future__ import annotations

import logging
from typing import Any

import dataset_manager
import ai_fallback
import diet_knowledge

logger = logging.getLogger(__name__)

DISCLAIMER: str = (
    "This is an AI-generated suggestion, not a medical prescription. "
    "Please consult a licensed doctor before taking any medication."
)


# ---------------------------------------------------------------------------
# Local OTC fallback — no network call, no API key needed
# ---------------------------------------------------------------------------
# Used when: severity == "mild", disease NOT in medications.csv, Gemini unavailable.
# Infers likely OTC drug categories from symptom names so mild cases always
# get a meaningful medication suggestion rather than null.
# ---------------------------------------------------------------------------

_SYMPTOM_OTC: dict[str, list[str]] = {
    "fever":                ["Paracetamol (Acetaminophen)", "NSAIDs (Ibuprofen)"],
    "cough":                ["Antitussives (Cough suppressants)", "Expectorants (Guaifenesin)"],
    "runny_nose":           ["Antihistamines (Loratadine / Cetirizine)", "Nasal decongestant sprays"],
    "continuous_sneezing":  ["Antihistamines (Cetirizine / Loratadine)", "Nasal decongestant sprays"],
    "congestion":           ["Decongestants (Pseudoephedrine)", "Saline nasal rinse"],
    "sore_throat":          ["Throat lozenges (Benzocaine)", "Antiseptic gargles"],
    "headache":             ["Paracetamol", "NSAIDs (Ibuprofen / Naproxen)"],
    "nausea":               ["Antiemetics (Domperidone / Ondansetron)"],
    "vomiting":             ["Oral rehydration salts (ORS)", "Antiemetics (Domperidone)"],
    "diarrhoea":            ["Oral rehydration salts (ORS)", "Antidiarrhoeals (Loperamide)"],
    "itching":              ["Antihistamines (Cetirizine / Loratadine)", "Calamine lotion (topical)"],
    "skin_rash":            ["Antihistamines (Cetirizine)", "Topical hydrocortisone cream"],
    "fatigue":              ["Multivitamins", "Iron supplements (if deficiency suspected)"],
    "muscle_pain":          ["NSAIDs (Ibuprofen)", "Topical analgesic gels"],
    "joint_pain":           ["NSAIDs (Ibuprofen / Naproxen)", "Topical analgesic gels"],
    "swelling":             ["NSAIDs (Ibuprofen)", "Cold compress"],
    "constipation":         ["Laxatives (Bisacodyl / Lactulose)", "Fibre supplements (Psyllium husk)"],
    "acidity":              ["Antacids (Aluminium hydroxide)", "H2-blockers (Famotidine)"],
    "burning_micturition":  ["Urinary alkalinisers (Potassium citrate)", "Cranberry extract supplements"],
    "anxiety":              ["Chamomile / Valerian supplements"],
    "insomnia":             ["Melatonin supplements", "Valerian root supplements"],
    "high_fever":           ["Paracetamol (Acetaminophen)", "NSAIDs (Ibuprofen)"],
    "mild_fever":           ["Paracetamol (Acetaminophen)"],
    "throat_irritation":    ["Throat lozenges", "Warm saline gargles"],
    "chills":               ["Paracetamol", "Warm fluids"],
    "sweating":             ["ORS (Oral Rehydration Salts)", "Paracetamol"],
    "loss_of_appetite":     ["Multivitamins", "Zinc supplements"],
    "stomach_pain":         ["Antacids", "Antispasmodics (Hyoscine)"],
    "abdominal_pain":       ["Antacids", "Antispasmodics (Hyoscine)"],
    "phlegm":               ["Expectorants (Guaifenesin)", "Steam inhalation"],
    "breathlessness":       ["Bronchodilators (consult pharmacist)", "Steam inhalation"],
}

_DISEASE_OTC: dict[str, list[str]] = {
    "cold":        ["Antihistamines", "Decongestants", "Paracetamol", "Cough suppressants"],
    "flu":         ["NSAIDs (Ibuprofen)", "Antihistamines", "Decongestants", "Paracetamol"],
    "allergy":     ["Antihistamines (Cetirizine / Loratadine)", "Nasal corticosteroid spray"],
    "migraine":    ["NSAIDs (Ibuprofen / Naproxen)", "Paracetamol"],
    "gastritis":   ["Antacids", "Proton pump inhibitors (Omeprazole)", "H2-blockers"],
    "acne":        ["Topical benzoyl peroxide", "Salicylic acid wash"],
    "eczema":      ["Topical hydrocortisone cream", "Emollients", "Antihistamines"],
    "anxiety":     ["Chamomile supplements", "Magnesium supplements"],
    "insomnia":    ["Melatonin supplements", "Valerian root"],
    "diarrhoea":   ["ORS", "Loperamide", "Probiotics"],
    "nausea":      ["Domperidone", "Ginger supplements"],
    "headache":    ["Paracetamol", "Ibuprofen"],
    "uti":         ["Urinary alkalinisers", "Cranberry supplements"],
    "infection":   ["Antiseptic wound care", "Consult pharmacist"],
}


def _local_medicine_fallback(disease: str, symptoms: list[str]) -> list[str]:
    """
    Build a list of OTC medication categories from symptoms and disease name
    WITHOUT any network call or API key.

    Strategy:
      1. Match each symptom against _SYMPTOM_OTC → collect unique categories.
      2. If nothing matched, match disease name keywords against _DISEASE_OTC.
      3. If still nothing, return a safe generic default.
    """
    collected: list[str] = []
    seen: set[str] = set()

    for sym in symptoms:
        key = sym.strip().lower()
        for cat in _SYMPTOM_OTC.get(key, []):
            if cat not in seen:
                collected.append(cat)
                seen.add(cat)
        if len(collected) >= 5:
            break

    if not collected:
        disease_lower = disease.lower()
        for keyword, cats in _DISEASE_OTC.items():
            if keyword in disease_lower:
                for cat in cats:
                    if cat not in seen:
                        collected.append(cat)
                        seen.add(cat)
                break

    if not collected:
        collected = [
            "Paracetamol (for fever and mild pain)",
            "ORS — Oral Rehydration Salts (if dehydrated)",
            "Consult a pharmacist for disease-specific OTC options",
        ]

    return collected[:5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_fully_missing(
    precautions: Any,
    diet: Any,
    medicines: Any,
) -> bool:
    """Return True if ALL three dataset lookups returned nothing."""
    return (
        precautions         is None
        and diet            is None
        and medicines       is None
    )


def generate_recommendation(
    disease: str,
    severity: str,
    symptoms: list[str],
    gemini_api_key: str | None = None,
) -> dict[str, Any]:
    """
    Build a full recommendation for a predicted disease.

    For diseases IN the local datasets → uses CSV data (instant, no API call).
    For diseases NOT in any dataset    → calls Gemini AI automatically.
    If Gemini is unavailable and severity is mild → uses local OTC fallback
    so the user always receives meaningful medication suggestions.
    """
    disease_key = disease.strip().title()

    precautions_db = dataset_manager.get_precautions()
    medicines_db   = dataset_manager.get_medicine_data()
    diet_db        = dataset_manager.get_diet_data()

    # ── 1. Precautions (AI & Clinical Knowledge Base) ─────────────────────────
    import clinical_precautions
    precautions = clinical_precautions.get_clinical_precautions(
        disease=disease,
        symptoms=symptoms,
        severity=severity,
        api_key=gemini_api_key,
    )

    # ── 2. Diet — prefer the clinical knowledge base (eat + avoid), fall back to CSV ───
    diet_kb = diet_knowledge.get_disease_diet(disease)
    if diet_kb:
        diet_recommendation: dict | list | None = diet_kb
        logger.info("Diet knowledge base matched for disease '%s'.", disease_key)
    elif diet_db and disease_key in diet_db:
        # CSV flat list fallback (eat-only, no avoid column)
        diet_recommendation = {"eat": diet_db[disease_key], "avoid": []}
        logger.info("Diet from CSV dataset for disease '%s'.", disease_key)
    else:
        diet_recommendation = None
        logger.info("No diet data found for disease '%s'.", disease_key)

    # ── 3. Medications vs. Doctor referral ───────────────────────────────────
    medications:    list[str] | None     = None
    doctor_referral: dict[str, str] | None = None

    if severity == "mild":
        if medicines_db and disease_key in medicines_db:
            medications = medicines_db[disease_key]["medications"]
    elif severity in ("moderate", "severe"):
        if medicines_db and disease_key in medicines_db:
            specialist = medicines_db[disease_key]["specialist_type"]
        else:
            specialist = dataset_manager.SPECIALIST_MAP.get(
                disease_key, "General Practitioner"
            )
        urgency = "same-day" if severity == "moderate" else "emergency"
        doctor_referral = {"specialist": specialist, "urgency": urgency}

    # ── 4. AI / local fallback — fires when disease is entirely unknown ───────
    medicines_found = (medicines_db and disease_key in medicines_db) if medicines_db else False
    is_unknown = _is_fully_missing(
        precautions,
        diet_recommendation,
        medications if severity == "mild" else (
            doctor_referral if severity in ("moderate", "severe") else None
        ),
    )

    if is_unknown and not medicines_found:
        logger.info(
            "Disease '%s' not found in any dataset — trying Gemini AI fallback.",
            disease_key,
        )
        try:
            ai_result = ai_fallback.get_ai_recommendation(
                disease=disease,
                severity=severity,
                api_key=gemini_api_key,
            )
            logger.info("Gemini fallback succeeded for disease='%s'.", disease_key)
            return ai_result
        except EnvironmentError as exc:
            logger.warning("Gemini API key not set — using local OTC fallback. %s", exc)
            if severity == "mild":
                # Always give mild cases useful medication suggestions
                medications = _local_medicine_fallback(disease, symptoms)
                logger.info(
                    "Local OTC fallback for mild '%s': %s", disease_key, medications
                )
            else:
                specialist = dataset_manager.SPECIALIST_MAP.get(
                    disease_key, "General Practitioner"
                )
                urgency = "same-day" if severity == "moderate" else "emergency"
                doctor_referral = {"specialist": specialist, "urgency": urgency}
        except RuntimeError as exc:
            logger.error("Gemini fallback failed for '%s': %s", disease_key, exc)
            if severity == "mild" and not medications:
                medications = _local_medicine_fallback(disease, symptoms)

    # ── 5. Assemble final response ────────────────────────────────────────────
    recommendation: dict[str, Any] = {
        "disease":             disease,
        "severity":            severity,
        "precautions":         precautions,
        "medications":         medications,
        "doctor_referral":     doctor_referral,
        "diet_recommendation": diet_recommendation,
        "disclaimer":          DISCLAIMER,
        "source":              "local_dataset",
    }

    logger.info(
        "Recommendation built (local) — disease='%s', severity='%s', "
        "meds=%s, referral=%s, diet=%s, precautions=%s.",
        disease_key, severity,
        bool(medications), bool(doctor_referral),
        bool(diet_recommendation), bool(precautions),
    )
    return recommendation
