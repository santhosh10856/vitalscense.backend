"""
quick_predict.py – VitalSense AI | Dataset-Aware Disease Predictor
===================================================================
Predicts disease by scoring symptom overlap against the training
dataset directly — guaranteeing the correct disease is returned as
long as the user's symptoms are in the dataset for that disease.

Two-stage prediction:
  Stage 1 – Dataset Overlap Score  (primary, always used)
             Scores every disease by: matched / disease_total_symptoms
             The disease whose profile best fits the input wins.
  Stage 2 – ML Model Probability   (secondary / tiebreaker)
             Used only when two diseases share the same overlap score.

This eliminates the "wrong disease" problem caused by symptoms that
overlap across multiple diseases in the ML model's training space.

Usage:
    python quick_predict.py
"""

from difflib import get_close_matches
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parent / "model"
DATASET    = Path(__file__).parent / "dataset.csv"

# ── Common plain-English aliases → model symptom names ────────────
ALIASES = {
    # Cold / respiratory
    "cold": "continuous_sneezing",
    "sneezing": "continuous_sneezing",
    "runny nose": "runny_nose",
    "breathing difficulty": "breathlessness",
    "difficulty breathing": "breathlessness",
    "shortness of breath": "breathlessness",
    "short of breath": "breathlessness",
    "phlegm": "phlegm",
    "mucus": "phlegm",
    "sinus": "sinus_pressure",
    # Fever / temperature
    "high temperature": "high_fever",
    "high fever": "high_fever",
    "temperature": "mild_fever",
    "fever": "mild_fever",      # generic "fever" → mild_fever (safer default)
    "mild fever": "mild_fever",
    # Common Cold specific
    "loss of smell": "loss_of_smell",
    "can't smell": "loss_of_smell",
    "watery eyes": "watering_from_eyes",
    "watering eyes": "watering_from_eyes",
    "congestion": "congestion",
    "nasal congestion": "congestion",
    "sinus pressure": "sinus_pressure",
    # Thirst / urination / diabetes vs UTI
    "thirsty": "excessive_hunger",
    "very thirsty": "excessive_hunger",
    "urinating often": "polyuria",          # Diabetes: excessive urine volume
    "frequent urination": "continuous_feel_of_urine",  # UTI: constant urge
    "increased urination": "polyuria",      # Diabetes
    "excessive urination": "polyuria",      # Diabetes
    "polyuria": "polyuria",                 # Diabetes
    # Eyes / vision
    "blurry vision": "blurred_and_distorted_vision",
    "blurred vision": "blurred_and_distorted_vision",
    "yellow eyes": "yellowing_of_eyes",
    "red eyes": "redness_of_eyes",
    # Weight
    "losing weight": "weight_loss",
    "unexplained weight loss": "weight_loss",
    "weight gain": "weight_gain",
    # Energy / fatigue
    "tired": "fatigue",
    "tiredness": "fatigue",
    "weakness": "fatigue",
    "fatigue": "fatigue",
    "low energy": "fatigue",
    "lethargy": "lethargy",
    # Stomach / GI
    "stomach ache": "stomach_pain",
    "tummy pain": "stomach_pain",
    "belly pain": "belly_pain",
    "abdomen pain": "abdominal_pain",
    "abdominal pain": "abdominal_pain",
    "stomach pain": "stomach_pain",
    "burning pain": "acidity",
    "burning stomach": "acidity",
    "burning upper abdomen": "acidity",
    "acid reflux": "acidity",
    "heartburn": "acidity",
    "bloating": "distention_of_abdomen",
    "bloated": "distention_of_abdomen",
    "burping": "passage_of_gases",
    "frequent burping": "passage_of_gases",
    "gas": "passage_of_gases",
    "indigestion": "indigestion",
    "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    "loss of appetite": "loss_of_appetite",
    "feeling full": "loss_of_appetite",
    "full quickly": "loss_of_appetite",
    "feeling full quickly": "loss_of_appetite",
    "nausea": "nausea",
    "nauseous": "nausea",
    "feeling sick": "nausea",
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "vomiting blood": "stomach_bleeding",
    "blood in vomit": "stomach_bleeding",
    "black stools": "bloody_stool",
    "tarry stools": "bloody_stool",
    "bloody stool": "bloody_stool",
    "black tarry stools": "bloody_stool",
    "internal itching": "internal_itching",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",
    "loose motion": "diarrhoea",
    "constipation": "constipation",
    # Skin
    "rash": "skin_rash",
    "itchy": "itching",
    "itch": "itching",
    "skin rash": "skin_rash",
    "yellow skin": "yellowish_skin",
    "jaundice": "yellowish_skin",
    # Pain
    "chest tightness": "chest_pain",
    "heart pain": "chest_pain",
    "chest pain": "chest_pain",
    "joint ache": "joint_pain",
    "joint pain": "joint_pain",
    "body ache": "muscle_pain",
    "body aches": "muscle_pain",
    "muscle ache": "muscle_pain",
    "muscle pain": "muscle_pain",
    "back pain": "back_pain",
    "neck pain": "neck_pain",
    "headache": "headache",
    "severe headache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    # Other
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "sore throat": "throat_irritation",
    "throat pain": "throat_irritation",
    "cough": "cough",
    "sweating": "sweating",
    "chills": "chills",
    "shivering": "shivering",
    # Dengue-specific
    "pain behind eyes": "pain_behind_the_eyes",
    "eye pain": "pain_behind_the_eyes",
    "pain behind the eyes": "pain_behind_the_eyes",
    "red spots": "red_spots_over_body",
    "red spots on body": "red_spots_over_body",
    "spots on skin": "red_spots_over_body",
    "anxiety": "anxiety",
    "stress": "anxiety",
    "swelling": "swollen_legs",
    "dark urine": "dark_urine",
    "pale urine": "yellow_urine",
    "burning urination": "burning_micturition",
    "pain urinating": "burning_micturition",
    "palpitations": "palpitations",
    "fast heartbeat": "fast_heart_rate",
    "rapid heartbeat": "fast_heart_rate",
    # Tuberculosis specific
    "blood in sputum": "blood_in_sputum",
    "coughing blood": "blood_in_sputum",
    "bloody cough": "blood_in_sputum",
    "rusty sputum": "rusty_sputum",
    "phlegm": "phlegm",
    "mucus cough": "phlegm",
    "malaise": "malaise",
    "swelled lymph nodes": "swelled_lymph_nodes",
    "swollen lymph nodes": "swelled_lymph_nodes",
    "lymph nodes": "swelled_lymph_nodes",
    "night sweats": "sweating",
    "mild fever": "mild_fever",
    # UTI specific
    "bladder discomfort": "bladder_discomfort",
    "bladder pain": "bladder_discomfort",
    "foul smell urine": "foul_smell_of_urine",
    "smelly urine": "foul_smell_of_urine",
    "urine smell": "foul_smell_of_urine",
    "feel of urine": "continuous_feel_of_urine",
    "continuous urge": "continuous_feel_of_urine",
    "urge to urinate": "continuous_feel_of_urine",
    "spotting urination": "spotting__urination",
}


# ════════════════════════════════════════════════════════════════════
# Dataset loader — builds disease → symptom-set map at startup
# ════════════════════════════════════════════════════════════════════

def load_disease_map(dataset_path: Path) -> dict:
    """
    Parse dataset.csv and return:
        { "Disease Name": frozenset({"symptom_a", "symptom_b", ...}), ... }
    """
    df = pd.read_csv(dataset_path)
    disease_col = df.columns[0]          # 'Disease'
    sym_cols    = [c for c in df.columns if c != disease_col]

    disease_map = {}
    for disease, group in df.groupby(disease_col):
        syms = set()
        for col in sym_cols:
            vals = (
                group[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )
            syms.update(v for v in vals if v and v != "nan")
        disease_map[disease.strip()] = frozenset(syms)
    return disease_map


# ════════════════════════════════════════════════════════════════════
# Symptom matching helpers
# ════════════════════════════════════════════════════════════════════

def normalise(text: str) -> str:
    """Lowercase, strip, replace spaces/hyphens with underscores."""
    s = str(text).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def match_symptom(user_input: str, known: list) -> tuple:
    """
    Try to resolve user input to a known model symptom name.
    Returns (matched_name_or_None, method_string).

    Priority:
      1. Exact match after normalisation
      2. Alias map   (plain-English → model name)
      3. Fuzzy match (difflib, cutoff 0.72 — tighter than before to
                      avoid wrong fuzzy mappings)
    """
    raw  = user_input.strip().lower()
    norm = normalise(raw)

    # 1. Exact
    if norm in known:
        return norm, "exact"

    # 2. Alias (check both raw and normalised forms)
    for key in (raw, norm.replace("_", " ")):
        if key in ALIASES and ALIASES[key] in known:
            return ALIASES[key], "alias"

    # 3. Fuzzy — use a tighter cutoff (0.72) to reduce wrong mappings
    close = get_close_matches(norm, known, n=1, cutoff=0.72)
    if close:
        return close[0], "fuzzy"

    return None, "none"


# ════════════════════════════════════════════════════════════════════
# Dataset-aware overlap scorer  (Stage 1 — primary predictor)
# ════════════════════════════════════════════════════════════════════

def predict_by_overlap(recognised: list, disease_map: dict) -> list:
    """
    Score every disease using true Jaccard similarity.

    Jaccard = |intersection| / |union|
             = matched / (disease_symptoms + user_symptoms - matched)

    This is fairer than simple coverage because:
    - A disease with 6 symptoms that matches 4 scores 4/(6+10-4) = 33%
    - A disease with 16 symptoms that matches 10 scores 10/(16+10-10) = 63%
    - Prevents small-symptom diseases from stealing the top spot.

    Returns a list of (disease, score, matched_count, total_count)
    sorted by score descending, then by matched_count descending.
    """
    user_set  = frozenset(recognised)
    n_user    = len(user_set)
    scores    = []

    for disease, sym_set in disease_map.items():
        matched = len(user_set & sym_set)
        if matched == 0:
            continue
        union   = len(sym_set) + n_user - matched   # |A ∪ B|
        jaccard = matched / union
        scores.append((disease, jaccard, matched, len(sym_set)))

    # Primary sort: Jaccard score DESC
    # Secondary sort: matched count DESC (break ties in favour of more matched)
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scores



# ════════════════════════════════════════════════════════════════════
# ML model loader + predictor  (Stage 2 — tiebreaker)
# ════════════════════════════════════════════════════════════════════

def load_model():
    """Load model and symptom list from disk."""
    model_path   = MODEL_DIR / "model.pkl"
    symptom_path = MODEL_DIR / "symptom_list.pkl"
    missing = [p for p in (model_path, symptom_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model file(s): {[str(p) for p in missing]}\n"
            "Please train the model first:\n"
            "  1. Run:  python app.py\n"
            "  2. Then: python live_test.py\n"
            "  3. Now you can run quick_predict.py standalone."
        )
    model        = joblib.load(model_path)
    symptom_list = joblib.load(symptom_path)
    return model, symptom_list


def predict_ml(recognised: list, symptom_list: list, model) -> tuple:
    """Build feature vector and run the ML model. Returns (disease, confidence)."""
    recognised_set = set(recognised)
    input_vector = pd.DataFrame(
        [[1 if sym in recognised_set else 0 for sym in symptom_list]],
        columns=symptom_list,
    )
    proba    = model.predict_proba(input_vector)[0]
    best_idx = int(np.argmax(proba))
    return model.classes_[best_idx], round(float(proba[best_idx]), 4)


# ════════════════════════════════════════════════════════════════════
# Main interactive loop
# ════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 56)
    print("   VitalSense AI  -  Disease Predictor")
    print("=" * 56)

    # ── Load artefacts ──────────────────────────────────────────
    try:
        model, symptom_list = load_model()
        known = symptom_list          # ordered list
        known_set = set(known)
    except FileNotFoundError as e:
        print(f"\n   ERROR: {e}\n")
        return
    except Exception as e:
        print(f"\n   ERROR loading model: {e}\n")
        return

    try:
        disease_map = load_disease_map(DATASET)
    except Exception as e:
        print(f"\n   ERROR loading dataset: {e}\n")
        disease_map = {}

    print(f"   {len(known)} symptoms  |  {len(disease_map)} diseases loaded.\n")

    # ── Collect symptoms ────────────────────────────────────────
    print("   Enter your symptoms (comma-separated or one per line).")
    print("   Type 'done' or leave blank when finished.\n")

    collected = []   # recognised symptom names
    warnings  = []   # unrecognised inputs

    while True:
        raw = input("  Symptom > ").strip()
        if not raw or raw.lower() in ("done", "quit", "q"):
            break

        parts = [p.strip() for p in raw.split(",") if p.strip()]

        for part in parts:
            matched, method = match_symptom(part, known)
            if matched:
                if matched in collected:
                    print(f"            '{matched}' already added.")
                else:
                    tag = {"exact": "[exact]",
                           "alias": "[alias]",
                           "fuzzy": "[fuzzy]"}.get(method, "")
                    print(f"            Matched  ->  '{matched}'  {tag}")
                    collected.append(matched)
            else:
                print(f"            Not recognised: '{part}' (skipped)")
                warnings.append(part)

    # ── Guard ───────────────────────────────────────────────────
    if not collected:
        if warnings:
            print("\n  No valid symptoms recognised. Check your spelling.\n")
        else:
            print("\n  No symptoms entered. Exiting.\n")
        return

    # ── Stage 1: Dataset overlap scoring ───────────────────────
    overlap_scores = predict_by_overlap(collected, disease_map)

    if overlap_scores:
        top_disease, top_score, matched_count, total_count = overlap_scores[0]

        # If top two diseases have EXACTLY the same overlap score,
        # use the ML model probability as the tiebreaker.
        if len(overlap_scores) > 1 and overlap_scores[1][1] == top_score:
            ml_disease, ml_conf = predict_ml(collected, known, model)
            # Pick whichever tied disease the ML model also agrees with
            tied_names = [d for d, s, _, _ in overlap_scores if s == top_score]
            if ml_disease in tied_names:
                top_disease = ml_disease
            confidence_pct = round(top_score * 100, 1)
            method_used = "overlap+ML tiebreak"
        else:
            confidence_pct = round(top_score * 100, 1)
            method_used = "dataset overlap"
    else:
        # Fallback to pure ML if no overlap found (very unlikely)
        ml_disease, ml_conf = predict_ml(collected, known, model)
        top_disease    = ml_disease
        confidence_pct = round(ml_conf * 100, 1)
        matched_count  = len(collected)
        total_count    = len(collected)
        method_used    = "ML model (no dataset overlap)"

    # ── Display results ─────────────────────────────────────────
    filled = int(confidence_pct / 5)
    bar    = "=" * filled + "-" * (20 - filled)

    print()
    print("=" * 56)
    print("   RESULT")
    print("=" * 56)
    print(f"   Symptoms matched  :  {', '.join(collected)}")
    if warnings:
        print(f"   Not recognised    :  {', '.join(warnings)}")
    print()
    print(f"   Predicted Disease :  {top_disease}")
    print(f"   Symptom match     :  {matched_count}/{total_count} "
          f"({confidence_pct}% of disease profile)")
    print(f"   [{bar}]  {confidence_pct}%")
    print(f"   Method            :  {method_used}")

    # Show top 3 other possibilities
    if len(overlap_scores) > 1:
        print()
        print("   Other possibilities:")
        for disease, score, mc, tc in overlap_scores[1:4]:
            pct = round(score * 100, 1)
            print(f"     {disease:<40} {mc}/{tc} ({pct}%)")

    print()
    print("   NOTE: This is for educational use only.")
    print("         Please consult a real doctor for medical advice.")
    print("=" * 56)
    print()


if __name__ == "__main__":
    main()
