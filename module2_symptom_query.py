"""
symptom_query.py
================
Full pipeline: natural-language symptoms -> disease matching from Training.csv
-> severity score -> complete recommendation (medicines, precautions, diet,
doctor referral) for every matched disease.

Usage:
  python symptom_query.py
"""

import sys
import re
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import dataset_manager as dm
from severity_calculator import calculate_severity, MILD_MAX, MODERATE_MAX
import recommend

# ── Load all datasets ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
dm.load_precautions(DATA_DIR / "precautions_df.csv")
dm.load_severity_weights(DATA_DIR / "Symptom-severity.csv")
dm.load_medicine_data(DATA_DIR / "medications.csv")
dm.load_diet_data(DATA_DIR / "diets.csv")
weights = dm.get_severity_weights()

# ── Load Training.csv (for disease matching) ───────────────────────────────
training_df = pd.read_csv(DATA_DIR / "Training.csv")
training_df.columns = training_df.columns.str.strip()
DISEASE_COL = "prognosis"
SYMPTOM_COLS = [c for c in training_df.columns if c != DISEASE_COL]

# ── Map user natural-language symptoms → dataset column names ──────────────
# This mapping bridges everyday English to the exact column names in
# Training.csv / Symptom-severity.csv
SYMPTOM_MAP = {
    "high fever":       ["high_fever"],
    "shaking chills":   ["shivering", "chills"],
    "heavy sweating":   ["sweating"],
    "headaches":        ["headache"],
    "headache":         ["headache"],
    "body aches":       ["muscle_pain", "joint_pain"],
    "body ache":        ["muscle_pain", "joint_pain"],
    "tiredness":        ["fatigue", "malaise"],
    "fatigue":          ["fatigue"],
    "fever":            ["high_fever"],
    "chills":           ["chills", "shivering"],
    "sweating":         ["sweating"],
    "nausea":           ["nausea"],
    "vomiting":         ["vomiting"],
    "loss of appetite": ["loss_of_appetite"],
    "dizziness":        ["dizziness"],
    "weakness":         ["fatigue", "malaise"],
}


def map_user_symptoms(user_text: str) -> tuple[list[str], list[str]]:
    """
    Convert a comma/and-separated free-text symptom string into a list of
    symptom column names recognised by the dataset.

    Returns (matched_cols, unmatched_phrases)
    """
    # Split on commas and 'and'
    phrases = re.split(r",\s*|\band\b", user_text.lower())
    phrases = [p.strip() for p in phrases if p.strip()]

    matched:   list[str] = []
    unmatched: list[str] = []

    for phrase in phrases:
        found = SYMPTOM_MAP.get(phrase, [])
        if found:
            for sym in found:
                if sym not in matched:
                    matched.append(sym)
        else:
            # Try direct column match
            col_name = phrase.replace(" ", "_")
            if col_name in SYMPTOM_COLS:
                if col_name not in matched:
                    matched.append(col_name)
            else:
                unmatched.append(phrase)

    return matched, unmatched


def find_matching_diseases(symptom_cols: list[str], top_n: int = 5) -> list[tuple[str, int]]:
    """
    Score every disease in Training.csv by counting how many of the
    provided symptom columns are marked 1 in that disease's rows.

    Returns a sorted list of (disease_name, match_count), highest first.
    """
    valid_cols = [c for c in symptom_cols if c in SYMPTOM_COLS]

    # For each disease, count avg presence of given symptoms
    scores: dict[str, float] = {}
    for disease, group in training_df.groupby(DISEASE_COL):
        disease_clean = str(disease).strip()
        hit_count = group[valid_cols].astype(float).values.sum()
        total_rows = len(group)
        scores[disease_clean] = hit_count / total_rows  # normalise by row count

    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Return top N with at least 1 hit
    return [(d, round(s, 2)) for d, s in sorted_diseases if s > 0][:top_n]


def print_full_report(
    user_input:     str,
    matched_syms:   list[str],
    unmatched:      list[str],
    sev_result:     dict,
    disease_scores: list[tuple[str, float]],
) -> None:
    """Print a full formatted diagnostic report."""
    WIDE  = "=" * 70
    MID   = "-" * 70

    print("\n" + WIDE)
    print("  VITALSENSE AI — MODULE 2 FULL DIAGNOSTIC REPORT")
    print(WIDE)

    # Input
    print("\n[SYMPTOMS ENTERED]")
    print(f"  {user_input}")

    # Mapped symptoms
    print("\n[MAPPED TO DATASET SYMPTOM NAMES]")
    for s in matched_syms:
        w = weights.get(s, "?")
        print(f"  {s:<35}  weight = {w}")

    if unmatched:
        print("\n[SYMPTOMS NOT RECOGNISED IN DATASET]")
        for u in unmatched:
            print(f"  '{u}'  (skipped from severity score)")

    # Severity summary
    score = sev_result["score"]
    label = sev_result["severity"].upper()
    if sev_result["severity"] == "mild":
        bracket = f"score {score:.0f} <= {MILD_MAX:.0f}"
    elif sev_result["severity"] == "moderate":
        bracket = f"{MILD_MAX:.0f} < score {score:.0f} <= {MODERATE_MAX:.0f}"
    else:
        bracket = f"score {score:.0f} > {MODERATE_MAX:.0f}"

    print(f"\n[SEVERITY SCORE]")
    print(f"  Total weight : {score}")
    print(f"  Level        : {label}  ({bracket})")

    print(f"\n[TOP MATCHING DISEASES FROM TRAINING DATA]")
    for rank, (disease, match_score) in enumerate(disease_scores, 1):
        print(f"  {rank}. {disease:<40}  (symptom match score: {match_score})")

    # Full recommendation per disease
    for rank, (disease, _) in enumerate(disease_scores, 1):
        print(f"\n{WIDE}")
        print(f"  PREDICTION #{rank}  |  {disease}")
        print(WIDE)

        rec = recommend.generate_recommendation(
            disease.strip(), sev_result["severity"], matched_syms
        )

        # Precautions
        print("\n  [PRECAUTIONS]")
        if rec["precautions"]:
            for i, p in enumerate(rec["precautions"], 1):
                print(f"    {i}. {p}")
        else:
            print("    (not available for this disease)")

        print(MID)

        # Medicine or Referral
        if rec["medications"]:
            print("  [MEDICINE RECOMMENDATION]  (mild — OTC)")
            for i, m in enumerate(rec["medications"], 1):
                print(f"    {i}. {m}")
            print("\n  [DOCTOR REFERRAL]  : Not required for mild severity")
        else:
            print("  [MEDICINE RECOMMENDATION]  : Suppressed — consult doctor")
            print("\n  [DOCTOR REFERRAL]")
            dr = rec["doctor_referral"]
            if dr:
                print(f"    Specialist : {dr['specialist']}")
                print(f"    Urgency    : {dr['urgency'].upper()}")
            else:
                print("    (not available)")

        print(MID)

        # Diet
        print("  [DIET RECOMMENDATIONS]")
        if rec["diet_recommendation"]:
            for i, d in enumerate(rec["diet_recommendation"], 1):
                print(f"    {i}. {d}")
        else:
            print("    (not available for this disease)")

        print(MID)
        print(f"  [DISCLAIMER]")
        print(f"    {rec['disclaimer']}")

    print("\n" + WIDE)
    print("  END OF REPORT")
    print(WIDE + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# USER QUERY
# ═══════════════════════════════════════════════════════════════════════════
USER_SYMPTOMS = "high fever, shaking chills, heavy sweating, headaches, body aches, and tiredness"

matched_syms, unmatched = map_user_symptoms(USER_SYMPTOMS)
sev_result = calculate_severity(matched_syms, weights)
disease_scores = find_matching_diseases(matched_syms, top_n=5)

print_full_report(
    user_input=USER_SYMPTOMS,
    matched_syms=matched_syms,
    unmatched=unmatched,
    sev_result=sev_result,
    disease_scores=disease_scores,
)
