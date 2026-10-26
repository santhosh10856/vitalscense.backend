"""
train_model.py – VitalSense AI | Module 1: Disease Prediction
=============================================================
Trains two classifiers (RandomForestClassifier + GaussianNB) on the
processed dataset, compares their performance on a held-out test set,
and persists the winner along with metadata required for inference.

Saved artifacts (written to model_dir):
  model.pkl          – the winning scikit-learn model
  symptom_list.pkl   – ordered list of symptom column names (feature order)
  disease_classes.pkl – sorted list of disease class names

Author : VitalSense AI Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Type aliases for readability                                         #
# ------------------------------------------------------------------ #
MetricsDict = Dict[str, float]          # e.g. {"accuracy": 0.95, ...}
ModelReport = Dict[str, "MetricsDict | str"]   # returned to caller / API


def train_and_save_model(
    X: pd.DataFrame,
    y: pd.Series,
    symptom_cols: List[str],
    model_dir: "str | Path" = ".",
) -> ModelReport:
    """
    Train RandomForestClassifier and GaussianNB, evaluate both, persist
    the better-performing model and required metadata.

    Parameters
    ----------
    X : pd.DataFrame
        Binary feature matrix (rows = samples, cols = symptoms).
    y : pd.Series
        Target disease labels.
    symptom_cols : list[str]
        Ordered symptom column names; saved so prediction uses identical order.
    model_dir : str | Path
        Directory where model artifacts (.pkl files) will be saved.
        The directory is created automatically if it does not exist.

    Returns
    -------
    report : dict
        {
          "random_forest" : {accuracy, precision, recall, f1},
          "naive_bayes"   : {accuracy, precision, recall, f1},
          "chosen_model"  : "random_forest" | "naive_bayes",
        }
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Train / test split (80 : 20, stratified to preserve class ratios)#
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,      # reproducible splits
        stratify=y,           # ensure every class appears in train & test
    )
    logger.info(
        "Split: %d train samples | %d test samples", len(X_train), len(X_test)
    )

    # ------------------------------------------------------------------ #
    # 2. Define candidate models                                           #
    # ------------------------------------------------------------------ #
    # RandomForestClassifier – ensemble of decision trees.
    #   n_estimators : number of trees; more trees -> more stable estimates.
    #   random_state : for reproducible tree building.
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,    # use all CPU cores for speed
    )

    # GaussianNB – probabilistic classifier based on Bayes' theorem.
    #   Assumes each feature is normally distributed within each class.
    #   Very fast to train; good baseline for high-dimensional binary data.
    nb_model = GaussianNB()

    # ------------------------------------------------------------------ #
    # 3. Train & evaluate both models                                      #
    # ------------------------------------------------------------------ #
    rf_metrics = _train_and_evaluate(rf_model, X_train, X_test, y_train, y_test, "RandomForest")
    nb_metrics = _train_and_evaluate(nb_model, X_train, X_test, y_train, y_test, "GaussianNB")

    # ------------------------------------------------------------------ #
    # 4. Pick the better model by F1 score (macro-averaged)               #
    #    F1 is preferred over accuracy because it accounts for class       #
    #    imbalance – a common issue in medical datasets.                   #
    # ------------------------------------------------------------------ #
    if rf_metrics["f1"] >= nb_metrics["f1"]:
        chosen_model = rf_model
        chosen_name = "random_forest"
        logger.info("RandomForest wins (F1=%.4f vs NB F1=%.4f).", rf_metrics["f1"], nb_metrics["f1"])
    else:
        chosen_model = nb_model
        chosen_name = "naive_bayes"
        logger.info("GaussianNB wins (F1=%.4f vs RF F1=%.4f).", nb_metrics["f1"], rf_metrics["f1"])

    # Re-train the winning model on the FULL dataset for maximum coverage
    # before saving (a common production practice after model selection).
    logger.info("Re-training winner on full dataset before saving...")
    chosen_model.fit(X[symptom_cols], y)

    # ------------------------------------------------------------------ #
    # 5. Persist artefacts                                                 #
    # ------------------------------------------------------------------ #
    classes: List[str] = sorted(y.unique().tolist())

    joblib.dump(chosen_model, model_dir / "model.pkl")
    joblib.dump(symptom_cols, model_dir / "symptom_list.pkl")
    joblib.dump(classes, model_dir / "disease_classes.pkl")

    logger.info("Saved model.pkl, symptom_list.pkl, disease_classes.pkl -> %s", model_dir)

    return {
        "random_forest": rf_metrics,
        "naive_bayes": nb_metrics,
        "chosen_model": chosen_name,
    }


# --------------------------------------------------------------------------- #
# Private helpers                                                               #
# --------------------------------------------------------------------------- #

def _train_and_evaluate(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    name: str,
) -> MetricsDict:
    """
    Fit a scikit-learn model and compute classification metrics on the test set.

    Uses macro-averaging for multi-class metrics so that each disease class
    contributes equally regardless of how many samples it has.

    Parameters
    ----------
    model : sklearn estimator
    X_train, X_test : feature matrices
    y_train, y_test : target series
    name : str  – label for logging

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1  (all 0–1 floats)
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics: MetricsDict = {
        # Fraction of correctly classified samples
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        # Precision: of all predicted positives, how many are truly positive
        "precision": round(
            float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        # Recall: of all actual positives, how many did we catch
        "recall": round(
            float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        # F1 = harmonic mean of precision & recall
        "f1": round(
            float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
    }

    logger.info(
        "%s – Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f",
        name, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
    )
    return metrics
