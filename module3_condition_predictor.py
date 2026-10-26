"""
condition_predictor.py — VitalSense AI · Module 3
==================================================
Dataset-driven condition/disease prediction using scikit-learn classifiers
trained on two uploaded datasets:

  1. diagnostic .datasets.csv   (25 000 rows, 9 features → 5 conditions)
  2. laboratory.datasets.csv    (12 009 rows, 13 features → 9 diseases)

NO AI API is used — everything is done locally with Random Forest classifiers
trained on the user's own CSV data.

Architecture
------------
  • On first use, both models are trained and cached in memory.
  • When a user pastes a diagnostic report, extracted lab values are mapped
    to dataset column names using an alias table.
  • Whichever dataset has ≥3 matching features gets a prediction run.
  • Both datasets can fire, giving a combined condition assessment.

Thread Safety
-------------
  Model training uses a module-level lock so concurrent Flask requests
  don't race during the initial cold-start training.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("module3.condition_predictor")

# ---------------------------------------------------------------------------
# Paths to the uploaded datasets
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent  # d:\sample

_DIAGNOSTIC_CSV = _BASE_DIR / "diagnostic .datasets.csv"
_LABORATORY_CSV = _BASE_DIR / "laboratory.datasets.csv"

# ---------------------------------------------------------------------------
# Cached models and training state
# ---------------------------------------------------------------------------
_models_lock = threading.Lock()
_models: dict[str, Any] = {
    "diagnostic_clf": None,
    "diagnostic_features": None,
    "diagnostic_label_map": None,
    "laboratory_clf": None,
    "laboratory_features": None,
    "laboratory_label_map": None,
    "trained": False,
}

# ---------------------------------------------------------------------------
# Feature alias maps — map extracted report param names to dataset columns
# ---------------------------------------------------------------------------

# Diagnostic dataset columns:
# Blood_glucose, HbA1C, Systolic_BP, Diastolic_BP, LDL, HDL, Triglycerides,
# Haemoglobin, MCV
_DIAG_ALIASES: dict[str, str] = {
    # Blood glucose
    "fasting_glucose":      "Blood_glucose",
    "random_glucose":       "Blood_glucose",
    "blood_glucose":        "Blood_glucose",
    "glucose":              "Blood_glucose",
    "fbs":                  "Blood_glucose",
    "blood_sugar":          "Blood_glucose",
    # HbA1c
    "hba1c":                "HbA1C",
    "a1c":                  "HbA1C",
    "glycated_hemoglobin":  "HbA1C",
    # Blood Pressure
    "systolic_bp":          "Systolic_BP",
    "systolic":             "Systolic_BP",
    "sys_bp":               "Systolic_BP",
    "diastolic_bp":         "Diastolic_BP",
    "diastolic":            "Diastolic_BP",
    "dia_bp":               "Diastolic_BP",
    # Lipid profile
    "ldl_cholesterol":      "LDL",
    "ldl":                  "LDL",
    "ldl-c":                "LDL",
    "hdl_cholesterol":      "HDL",
    "hdl":                  "HDL",
    "hdl-c":                "HDL",
    "triglycerides":        "Triglycerides",
    "tg":                   "Triglycerides",
    # Haemoglobin / MCV
    "hemoglobin":           "Haemoglobin",
    "haemoglobin":          "Haemoglobin",
    "hb":                   "Haemoglobin",
    "hgb":                  "Haemoglobin",
    "mcv":                  "MCV",
}

# Laboratory dataset columns:
# Gender, Age, Hemoglobin, RBC, WBC, AST, ALT, Cholestrol, Spirometry,
# Creatinine, Glucose, Lipase, Troponin
_LAB_ALIASES: dict[str, str] = {
    # Hemoglobin
    "hemoglobin":           "Hemoglobin",
    "haemoglobin":          "Hemoglobin",
    "hb":                   "Hemoglobin",
    "hgb":                  "Hemoglobin",
    # RBC
    "rbc_count":            "RBC",
    "rbc":                  "RBC",
    "red_blood_cells":      "RBC",
    # WBC
    "wbc_count":            "WBC",
    "wbc":                  "WBC",
    "white_blood_cells":    "WBC",
    "tlc":                  "WBC",
    # Liver enzymes
    "ast":                  "AST (aspartate aminotransferase)",
    "sgot":                 "AST (aspartate aminotransferase)",
    "aspartate_aminotransferase": "AST (aspartate aminotransferase)",
    "alt":                  "ALT (alanine aminotransferase)",
    "sgpt":                 "ALT (alanine aminotransferase)",
    "alanine_aminotransferase": "ALT (alanine aminotransferase)",
    # Cholesterol
    "total_cholesterol":    "Cholestrol",
    "cholesterol":          "Cholestrol",
    "cholestrol":           "Cholestrol",
    # Spirometry
    "spirometry":           "Spirometry",
    "fev1":                 "Spirometry",
    "lung_function":        "Spirometry",
    # Kidney
    "creatinine":           "Creatinine",
    "serum_creatinine":     "Creatinine",
    # Glucose
    "fasting_glucose":      "Glucose",
    "random_glucose":       "Glucose",
    "blood_glucose":        "Glucose",
    "glucose":              "Glucose",
    "blood_sugar":          "Glucose",
    # Pancreatic
    "lipase":               "Lipase",
    "serum_lipase":         "Lipase",
    # Cardiac
    "troponin":             "Troponin",
    "troponin_i":           "Troponin",
    "troponin_t":           "Troponin",
}

# ---------------------------------------------------------------------------
# Condition information — for richer findings
# ---------------------------------------------------------------------------
_CONDITION_INFO: dict[str, dict[str, Any]] = {
    # Diagnostic dataset conditions
    "Fit": {
        "severity": "normal",
        "description": "Your lab values appear to be within healthy ranges. No significant conditions detected.",
        "recommendation": "Continue maintaining a healthy lifestyle with regular check-ups.",
    },
    "Diabetes": {
        "severity": "moderate",
        "description": "Your blood sugar and HbA1c levels suggest possible diabetes or pre-diabetes.",
        "recommendation": "Consult an endocrinologist. Monitor blood sugar regularly, follow a low-sugar diet, and exercise regularly.",
    },
    "Hypertension": {
        "severity": "moderate",
        "description": "Your blood pressure readings indicate possible hypertension (high blood pressure).",
        "recommendation": "Consult a cardiologist. Reduce salt intake, manage stress, exercise regularly, and monitor BP daily.",
    },
    "High_Cholesterol": {
        "severity": "moderate",
        "description": "Your lipid profile shows elevated cholesterol levels, increasing cardiovascular risk.",
        "recommendation": "Consult a cardiologist. Follow a low-fat diet, increase fibre intake, exercise, and consider medication if advised.",
    },
    "Anemia": {
        "severity": "moderate",
        "description": "Your hemoglobin and/or MCV levels suggest anemia (low red blood cell count or hemoglobin).",
        "recommendation": "Consult a hematologist. Increase iron-rich foods (spinach, lentils, red meat), and consider iron supplements.",
    },
    # Laboratory dataset diseases
    "Heart attack": {
        "severity": "severe",
        "description": "Your troponin and other cardiac markers suggest possible cardiac involvement.",
        "recommendation": "Seek immediate cardiology consultation. This requires urgent medical attention.",
    },
    "Kidney Disease": {
        "severity": "moderate",
        "description": "Your creatinine and related values suggest possible kidney function impairment.",
        "recommendation": "Consult a nephrologist. Stay hydrated, reduce salt and protein intake, and monitor kidney function regularly.",
    },
    "Diabetics": {
        "severity": "moderate",
        "description": "Your glucose levels suggest possible diabetes.",
        "recommendation": "Consult an endocrinologist. Monitor blood sugar, follow a balanced diet, and exercise regularly.",
    },
    "Infection": {
        "severity": "moderate",
        "description": "Your WBC count and related markers suggest a possible ongoing infection.",
        "recommendation": "Consult your physician. You may need antibiotics or further investigation to identify the infection source.",
    },
    "Liver Disease": {
        "severity": "moderate",
        "description": "Your liver enzyme levels (AST/ALT) suggest possible liver function abnormality.",
        "recommendation": "Consult a hepatologist/gastroenterologist. Avoid alcohol, reduce fatty foods, and get a liver ultrasound.",
    },
    "Asthma": {
        "severity": "mild",
        "description": "Your spirometry values suggest possible respiratory/airway issues.",
        "recommendation": "Consult a pulmonologist. Avoid triggers, use prescribed inhalers, and monitor breathing regularly.",
    },
    "Pancreatitis": {
        "severity": "moderate",
        "description": "Your lipase levels suggest possible pancreatic inflammation.",
        "recommendation": "Consult a gastroenterologist. Avoid alcohol and fatty foods. You may need imaging studies.",
    },
    "Cardiovascular disease": {
        "severity": "moderate",
        "description": "Your cholesterol and cardiac markers suggest elevated cardiovascular risk.",
        "recommendation": "Consult a cardiologist. Follow a heart-healthy diet, exercise, manage stress, and monitor cholesterol.",
    },
}


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_models() -> None:
    """
    Train Random Forest classifiers on both datasets. Called once on first
    prediction request and cached for subsequent calls.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # ── Train diagnostic model ───────────────────────────────────────────
    if _DIAGNOSTIC_CSV.exists():
        try:
            df = pd.read_csv(_DIAGNOSTIC_CSV)
            features = ["Blood_glucose", "HbA1C", "Systolic_BP", "Diastolic_BP",
                        "LDL", "HDL", "Triglycerides", "Haemoglobin", "MCV"]
            df = df.dropna(subset=features + ["Condition"])

            X = df[features].values
            le = LabelEncoder()
            y = le.fit_transform(df["Condition"])

            clf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
            )
            clf.fit(X, y)

            _models["diagnostic_clf"] = clf
            _models["diagnostic_features"] = features
            _models["diagnostic_label_map"] = {i: label for i, label in enumerate(le.classes_)}

            logger.info(
                "Diagnostic classifier trained: %d samples, %d features, %d classes.",
                len(X), len(features), len(le.classes_),
            )
        except Exception as exc:
            logger.error("Failed to train diagnostic classifier: %s", exc)
    else:
        logger.warning("Diagnostic dataset not found at: %s", _DIAGNOSTIC_CSV)

    # ── Train laboratory model ───────────────────────────────────────────
    if _LABORATORY_CSV.exists():
        try:
            df = pd.read_csv(_LABORATORY_CSV)
            # The Disease column has a trailing space
            disease_col = "Disease " if "Disease " in df.columns else "Disease"

            features = ["Hemoglobin", "RBC", "WBC",
                        "AST (aspartate aminotransferase)",
                        "ALT (alanine aminotransferase)",
                        "Cholestrol", "Spirometry", "Creatinine",
                        "Glucose", "Lipase", "Troponin"]
            df = df.dropna(subset=features + [disease_col])

            X = df[features].values
            le = LabelEncoder()
            y = le.fit_transform(df[disease_col].str.strip())

            clf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
            )
            clf.fit(X, y)

            _models["laboratory_clf"] = clf
            _models["laboratory_features"] = features
            _models["laboratory_label_map"] = {i: label for i, label in enumerate(le.classes_)}

            logger.info(
                "Laboratory classifier trained: %d samples, %d features, %d classes.",
                len(X), len(features), len(le.classes_),
            )
        except Exception as exc:
            logger.error("Failed to train laboratory classifier: %s", exc)
    else:
        logger.warning("Laboratory dataset not found at: %s", _LABORATORY_CSV)

    _models["trained"] = True


def _ensure_trained() -> None:
    """Thread-safe lazy training — train once on first call."""
    if _models["trained"]:
        return
    with _models_lock:
        if _models["trained"]:
            return
        _train_models()


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_conditions(
    extracted_values: dict[str, float],
    gender: str = "Any",
) -> list[dict[str, Any]]:
    """
    Predict possible conditions/diseases from extracted lab values using
    the trained classifiers.

    Parameters
    ----------
    extracted_values : dict
        Mapping of normalised parameter key → numeric value.
        Keys should be in the _normalise_param() format (lowercase, underscored).
        Example: {"hemoglobin": 10.2, "wbc_count": 12500, "alt": 65}

    gender : str
        Patient gender ("Male", "Female", or "Any").

    Returns
    -------
    list of dicts, each with:
        source       : "diagnostic_dataset" or "laboratory_dataset"
        condition    : predicted condition/disease name
        confidence   : prediction probability (0.0 – 1.0)
        severity     : "normal" | "mild" | "moderate" | "severe"
        description  : plain-language explanation
        recommendation : what the patient should do
        top_3        : list of top 3 predictions with probabilities
        features_used : number of features matched from the dataset
        features_total: total features in the dataset
    """
    _ensure_trained()

    predictions: list[dict[str, Any]] = []

    # ── Try diagnostic dataset ───────────────────────────────────────────
    if _models["diagnostic_clf"] is not None:
        result = _predict_single(
            extracted_values,
            _models["diagnostic_clf"],
            _models["diagnostic_features"],
            _models["diagnostic_label_map"],
            _DIAG_ALIASES,
            "diagnostic_dataset",
        )
        if result is not None:
            predictions.append(result)

    # ── Try laboratory dataset ───────────────────────────────────────────
    if _models["laboratory_clf"] is not None:
        result = _predict_single(
            extracted_values,
            _models["laboratory_clf"],
            _models["laboratory_features"],
            _models["laboratory_label_map"],
            _LAB_ALIASES,
            "laboratory_dataset",
        )
        if result is not None:
            predictions.append(result)

    return predictions


def _predict_single(
    extracted_values: dict[str, float],
    clf,
    feature_cols: list[str],
    label_map: dict[int, str],
    alias_map: dict[str, str],
    source_name: str,
    min_features: int = 3,
) -> dict[str, Any] | None:
    """
    Run prediction on a single classifier if enough features match.

    Uses dataset median values for any missing features to allow partial
    predictions (common in real reports that don't include all parameters).
    """
    # Map extracted values to dataset column names
    feature_vector: dict[str, float | None] = {col: None for col in feature_cols}
    matched_count = 0

    for param_key, value in extracted_values.items():
        # Check if this param maps to a column in this dataset
        dataset_col = alias_map.get(param_key)
        if dataset_col and dataset_col in feature_vector:
            feature_vector[dataset_col] = value
            matched_count += 1

    # Need at least min_features to make a meaningful prediction
    if matched_count < min_features:
        return None

    # Fill missing features with dataset medians (loaded from training data)
    csv_path = _DIAGNOSTIC_CSV if source_name == "diagnostic_dataset" else _LABORATORY_CSV
    try:
        df = pd.read_csv(csv_path)
        medians = df[feature_cols].median()
    except Exception:
        # If CSV read fails, use 0 as fallback
        medians = pd.Series({col: 0.0 for col in feature_cols})

    X = []
    for col in feature_cols:
        if feature_vector[col] is not None:
            X.append(feature_vector[col])
        else:
            X.append(float(medians.get(col, 0.0)))

    X = np.array([X])

    # Predict
    predicted_class = clf.predict(X)[0]
    probabilities = clf.predict_proba(X)[0]

    # Top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_3 = [
        {"condition": label_map[idx], "probability": round(float(probabilities[idx]), 3)}
        for idx in top_indices
        if probabilities[idx] > 0.01
    ]

    predicted_label = label_map[predicted_class]
    confidence = float(probabilities[predicted_class])

    # Get condition info
    info = _CONDITION_INFO.get(predicted_label, {
        "severity": "moderate",
        "description": f"The analysis suggests a possible condition: {predicted_label}.",
        "recommendation": "Please consult your doctor for further evaluation.",
    })

    return {
        "source": source_name,
        "condition": predicted_label,
        "confidence": round(confidence, 3),
        "severity": info["severity"],
        "description": info["description"],
        "recommendation": info["recommendation"],
        "top_3": top_3,
        "features_used": matched_count,
        "features_total": len(feature_cols),
    }


def is_available() -> bool:
    """Check if at least one dataset file exists."""
    return _DIAGNOSTIC_CSV.exists() or _LABORATORY_CSV.exists()
