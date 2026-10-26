"""
predict.py – VitalSense AI | Module 1: Disease Prediction
=========================================================
Loads a trained model from disk and predicts the most probable disease
for a list of user-supplied symptom names.

Key design decisions:
  * Unrecognised symptoms are collected in a "warnings" list and IGNORED
    rather than raising an exception – the model still runs with the
    symptoms it does recognise.
  * When ZERO symptoms are recognised, the module calls the Gemini AI
    fallback (module1_ai_fallback.py) instead of raising an error.
    This allows prediction for symptoms not in the training dataset.
  * Confidence is derived from predict_proba(), so the model must support
    probability estimates (both RandomForest and GaussianNB do).
  * The result always contains a "source" key: "model" or "ai".

Author : VitalSense AI Team
"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

# AI fallback for when no symptoms match the training dataset
from module1_ai_fallback import ai_predict_disease, is_gemini_available

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Type alias                                                           #
# ------------------------------------------------------------------ #
PredictionResult = Dict[str, "str | float | List[str]"]


def predict_disease(
    user_symptoms: List[str],
    model_dir: "str | Path" = ".",
    gemini_api_key: "str | None" = None,
) -> PredictionResult:
    """
    Predict the most likely disease from a list of symptom names.

    Parameters
    ----------
    user_symptoms : list[str]
        Raw symptom names as entered by the user (e.g. ["itching", "skin_rash"]).
        Whitespace and capitalisation are normalised before look-up.
    model_dir : str | Path
        Directory containing the three .pkl artefacts saved by train_model.py.
    gemini_api_key : str | None
        Optional Gemini API key for the AI fallback.  If None, the fallback
        reads GEMINI_API_KEY from the environment variable.

    Returns
    -------
    dict with keys:
      "disease"    : str   – predicted disease name
      "confidence" : float – probability assigned by the model (0–1)
      "warnings"   : list  – symptom strings not found in the trained list
      "source"     : str   – "model" (ML) or "ai" (Gemini fallback)

    Raises
    ------
    FileNotFoundError
        If any required .pkl file is missing.
    ValueError
        If none of the user-supplied symptoms are recognised AND the Gemini
        AI fallback is not available (no API key configured).
    """
    model_dir = Path(model_dir)

    # ------------------------------------------------------------------ #
    # 1. Load saved artefacts from disk                                    #
    # ------------------------------------------------------------------ #
    model_path = model_dir / "model.pkl"
    symptom_path = model_dir / "symptom_list.pkl"

    for p in [model_path, symptom_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required model file not found: {p}. "
                "Please train the model first via POST /dataset/train."
            )

    # joblib.load deserialises the pickled Python objects
    model = joblib.load(model_path)
    symptom_list: List[str] = joblib.load(symptom_path)
    logger.info("Model loaded from %s | %d known symptoms", model_dir, len(symptom_list))

    # ------------------------------------------------------------------ #
    # 2. Normalise user-supplied symptom names                            #
    #    Apply the same cleaning as data_loader._clean_symptom() so that  #
    #    "Skin Rash", " skin_rash ", "SKIN RASH" all map to "skin_rash". #
    # ------------------------------------------------------------------ #
    def _normalise(s: str) -> str:
        s = str(s).strip().lower().replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s

    normalised_input = [_normalise(sym) for sym in user_symptoms]

    # Build a set for O(1) membership testing
    known_set = set(symptom_list)

    # ------------------------------------------------------------------ #
    # 3. Separate recognised symptoms from unrecognised ones              #
    # ------------------------------------------------------------------ #
    recognised: List[str] = []
    warnings: List[str] = []

    for raw, norm in zip(user_symptoms, normalised_input):
        if norm in known_set:
            recognised.append(norm)
        else:
            # Collect unrecognised symptoms in warnings; do NOT crash
            warnings.append(raw)
            logger.warning("Unrecognised symptom ignored: '%s'", raw)

    if warnings:
        logger.info("%d symptom(s) not recognised: %s", len(warnings), warnings)

    # ------------------------------------------------------------------ #
    # 4. Guard: if zero symptoms recognised, try Gemini AI fallback        #
    # ------------------------------------------------------------------ #
    if not recognised:
        if is_gemini_available() or gemini_api_key:
            logger.info(
                "No recognised symptoms — activating Gemini AI fallback "
                "for %d symptom(s): %s", len(warnings), warnings
            )
            # ai_predict_disease returns the same dict shape + source='ai'
            return ai_predict_disease(symptoms=warnings, api_key=gemini_api_key)
        else:
            # No API key available — inform the client clearly
            raise ValueError(
                "None of the provided symptoms are recognised by the trained model. "
                f"Unrecognised: {warnings}. "
                "To enable AI prediction for unknown symptoms, set the "
                "GEMINI_API_KEY environment variable. "
                "Fetch GET /dataset/symptoms for the list of supported symptoms."
            )

    # ------------------------------------------------------------------ #
    # 5. Build the input feature vector                                   #
    #    The vector must have the SAME number of columns and the SAME     #
    #    column ORDER as the training data – this is what symptom_list    #
    #    encodes.                                                         #
    # ------------------------------------------------------------------ #
    recognised_set = set(recognised)

    # Build a single-row DataFrame with the SAME column names and order as
    # training. Using a DataFrame (not a plain numpy array) is important:
    # the model was fitted on a DataFrame, so sklearn expects named features.
    input_vector = pd.DataFrame(
        [[1 if sym in recognised_set else 0 for sym in symptom_list]],
        columns=symptom_list,
    )
    logger.info(
        "Input vector: %d active features out of %d",
        int(input_vector.values.sum()), len(symptom_list),
    )

    # ------------------------------------------------------------------ #
    # 6. Predict with probability calibration                             #
    #    predict_proba() returns shape (1, n_classes); we take the max.  #
    # ------------------------------------------------------------------ #
    proba = model.predict_proba(input_vector)[0]   # shape: (n_classes,)
    best_idx: int = int(np.argmax(proba))
    confidence: float = round(float(proba[best_idx]), 4)
    disease: str = model.classes_[best_idx]

    logger.info("Prediction: '%s' (confidence=%.4f)", disease, confidence)

    return {
        "disease":    disease,
        "confidence": confidence,
        "warnings":   warnings,
        "source":     "model",   # ML model prediction (not AI)
    }
