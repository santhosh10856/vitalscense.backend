"""
dataset_manager.py
==================
Loads, validates, and caches all four reference CSV datasets for Module 2.

Updated for the new dataset files uploaded by the user:
  - precautions_df.csv  : has a leading row-index column + Disease + Precaution_1..4
  - Symptom-severity.csv: same schema (Symptom, weight), different filename
  - medications.csv     : Disease + Medication (Python list as string, no specialist col)
  - diets.csv           : Disease + Diet (Python list as string)

Design decisions
----------------
- Every load_* function validates required columns and raises a specific
  ValueError naming the missing column(s).
- Module-level _CACHE holds parsed data so CSVs are read from disk only once.
- reload() refreshes one or all datasets from the registered file paths.
- The medications CSV stores medication names as a Python list literal
  (e.g. "['Insulin', 'Metformin', ...]") — ast.literal_eval is used to
  parse this safely. Same for the diets CSV.
- Since medications.csv has no Specialist_Type column, a built-in
  SPECIALIST_MAP provides sensible specialist defaults per disease. You may
  extend this map or override it by editing the section below.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in specialist fallback map
# (used because the new medications.csv has no Specialist_Type column)
# ---------------------------------------------------------------------------
# Keys must be title-cased disease names (same convention as the rest of the code).
# Add or edit entries here to match your disease set.
SPECIALIST_MAP: dict[str, str] = {
    "Fungal Infection":                        "Dermatologist",
    "Allergy":                                 "Allergist / Immunologist",
    "Gerd":                                    "Gastroenterologist",
    "Chronic Cholestasis":                     "Gastroenterologist",
    "Drug Reaction":                           "General Practitioner",
    "Peptic Ulcer Disease":                    "Gastroenterologist",
    "Aids":                                    "Infectious Disease Specialist",
    "Diabetes":                                "Endocrinologist",
    "Gastroenteritis":                         "Gastroenterologist",
    "Bronchial Asthma":                        "Pulmonologist",
    "Hypertension":                            "Cardiologist",
    "Migraine":                                "Neurologist",
    "Cervical Spondylosis":                    "Orthopedist",
    "Paralysis (Brain Hemorrhage)":            "Neurologist",
    "Jaundice":                                "Gastroenterologist / Hepatologist",
    "Malaria":                                 "Infectious Disease Specialist",
    "Chicken Pox":                             "Infectious Disease Specialist",
    "Dengue":                                  "Infectious Disease Specialist",
    "Typhoid":                                 "Infectious Disease Specialist",
    "Hepatitis A":                             "Hepatologist",
    "Hepatitis B":                             "Hepatologist",
    "Hepatitis C":                             "Hepatologist",
    "Hepatitis D":                             "Hepatologist",
    "Hepatitis E":                             "Hepatologist",
    "Alcoholic Hepatitis":                     "Hepatologist",
    "Tuberculosis":                            "Pulmonologist",
    "Common Cold":                             "General Practitioner",
    "Pneumonia":                               "Pulmonologist",
    "Dimorphic Hemmorhoids(Piles)":            "Gastroenterologist / Colorectal Surgeon",
    "Heart Attack":                            "Cardiologist",
    "Varicose Veins":                          "Vascular Surgeon",
    "Hypothyroidism":                          "Endocrinologist",
    "Hyperthyroidism":                         "Endocrinologist",
    "Hypoglycemia":                            "Endocrinologist",
    "Osteoarthristis":                         "Rheumatologist / Orthopedist",
    "Arthritis":                               "Rheumatologist",
    "(Vertigo) Paroymsal Positional Vertigo":  "Neurologist / ENT Specialist",
    "Acne":                                    "Dermatologist",
    "Urinary Tract Infection":                 "Urologist",
    "Psoriasis":                               "Dermatologist",
    "Impetigo":                                "Dermatologist",
}

# ---------------------------------------------------------------------------
# Internal in-memory cache
# ---------------------------------------------------------------------------
_CACHE: dict[str, Any] = {
    "precautions":      None,
    "severity_weights": None,
    "medicines":        None,
    "diets":            None,
}

_PATHS: dict[str, Path | None] = {
    "precautions":      None,
    "severity_weights": None,
    "medicines":        None,
    "diets":            None,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Raise ValueError if any of *required* columns are missing from *df*."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] Missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV, stripping whitespace from column names and all string cells."""
    path = Path(path)
    if not path.exists():
        # Fallback check relative to module2 directory or module2/data directory
        base = Path(__file__).parent
        if (base / path).exists():
            path = base / path
        elif (base / "data" / path.name).exists():
            path = base / "data" / path.name
        else:
            raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    # pandas >= 2.1 uses .map() instead of the deprecated .applymap()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def _normalise_disease(raw: str) -> str:
    """
    Normalise a disease name for consistent dict key usage.
    Steps: strip -> collapse internal whitespace runs -> title-case.
    This ensures 'Paroymsal  Positional Vertigo' (double space) and
    'Paroymsal Positional Vertigo' both map to the same dict key.
    """
    import re
    return re.sub(r"\s+", " ", raw.strip()).title()


def _parse_list_cell(cell: Any) -> list[str]:
    """
    Parse a cell that contains a Python list literal string.

    The medications.csv and diets.csv store values like:
        "['Insulin', 'Metformin', 'Sulfonylureas']"

    We use ast.literal_eval for safe parsing (no exec / eval of arbitrary code).
    Falls back to a comma-split if the literal_eval fails.
    """
    if pd.isna(cell) or str(cell).strip().lower() in ("nan", "none", ""):
        return []
    raw = str(cell).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        # Fallback: treat as plain comma-separated string
        return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Public load_* functions
# ---------------------------------------------------------------------------

def load_precautions(path: str | Path) -> dict[str, list[str]]:
    """
    Load precautions_df.csv → disease -> list[precaution strings].

    Expected columns
    ----------------
    (optional index col), Disease, Precaution_1, Precaution_2, Precaution_3, Precaution_4

    The file has a leading numeric row-index column (unnamed or ''); it is
    dropped automatically because we only look up by column name.
    """
    path = Path(path)
    df = _read_csv(path)

    # The file starts with an unnamed index column; drop it if present.
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.iloc[:, 1:]
        df.columns = df.columns.str.strip()

    required = ["Disease", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    _validate_columns(df, required, path.name)

    precaution_cols = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    result: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        disease = _normalise_disease(str(row["Disease"]))
        precautions = [
            str(row[col]).strip()
            for col in precaution_cols
            if pd.notna(row[col]) and str(row[col]).strip().lower() not in ("nan", "none", "")
        ]
        result[disease] = precautions

    logger.info("Loaded precautions for %d disease(s) from '%s'.", len(result), path.name)
    _PATHS["precautions"] = path
    _CACHE["precautions"] = result
    return result


def load_severity_weights(path: str | Path) -> dict[str, float]:
    """
    Load Symptom-severity.csv → symptom (lowercase) -> numeric weight.

    Expected columns: Symptom, weight
    """
    path = Path(path)
    df = _read_csv(path)
    _validate_columns(df, ["Symptom", "weight"], path.name)

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        symptom = str(row["Symptom"]).strip().lower()
        if not symptom or symptom == "nan":
            continue
        try:
            weight = float(row["weight"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"[{path.name}] Non-numeric weight '{row['weight']}' "
                f"for symptom '{symptom}'."
            ) from exc
        result[symptom] = weight

    logger.info(
        "Loaded severity weights for %d symptom(s) from '%s'.", len(result), path.name
    )
    _PATHS["severity_weights"] = path
    _CACHE["severity_weights"] = result
    return result


def load_medicine_data(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load medications.csv → disease -> {"medications": list[str], "specialist_type": str}.

    Expected columns
    ----------------
    Disease, Medication

    The Medication column holds a Python list literal string, e.g.:
        "['Antifungal Cream', 'Fluconazole', ...]"

    Because this file has no Specialist_Type column, the specialist is
    resolved from the built-in SPECIALIST_MAP. If a disease is not in the
    map, a safe default of "General Practitioner" is used.
    """
    path = Path(path)
    df = _read_csv(path)
    _validate_columns(df, ["Disease", "Medication"], path.name)

    result: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        disease_raw = str(row["Disease"]).strip()
        if not disease_raw or disease_raw.lower() == "nan":
            continue
        disease = _normalise_disease(disease_raw)
        medications = _parse_list_cell(row["Medication"])

        # Specialist lookup: SPECIALIST_MAP -> fallback "General Practitioner"
        specialist = SPECIALIST_MAP.get(disease, "General Practitioner")

        result[disease] = {
            "medications":     medications,
            "specialist_type": specialist,
        }

    logger.info("Loaded medicine data for %d disease(s) from '%s'.", len(result), path.name)
    _PATHS["medicines"] = path
    _CACHE["medicines"] = result
    return result


def load_diet_data(path: str | Path) -> dict[str, list[str]]:
    """
    Load diets.csv → disease -> list[diet recommendation strings].

    Expected columns
    ----------------
    Disease, Diet

    The Diet column holds a Python list literal string, e.g.:
        "['Low-Glycemic Diet', 'Fiber-rich foods', ...]"

    Note: this new format has a single Diet list (no separate eat/avoid).
    The API response returns {"recommendations": [...]} instead of
    {"eat": [...], "avoid": [...]}.
    """
    path = Path(path)
    df = _read_csv(path)
    _validate_columns(df, ["Disease", "Diet"], path.name)

    result: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        disease_raw = str(row["Disease"]).strip()
        if not disease_raw or disease_raw.lower() == "nan":
            continue
        disease = _normalise_disease(disease_raw)
        diet_items = _parse_list_cell(row["Diet"])
        result[disease] = diet_items

    logger.info("Loaded diet data for %d disease(s) from '%s'.", len(result), path.name)
    _PATHS["diets"] = path
    _CACHE["diets"] = result
    return result


# ---------------------------------------------------------------------------
# Cache accessors
# ---------------------------------------------------------------------------

def get_precautions() -> dict[str, list[str]] | None:
    """Return cached precautions dict, or None if not yet loaded."""
    return _CACHE["precautions"]


def get_severity_weights() -> dict[str, float] | None:
    """Return cached severity weights dict, or None if not yet loaded."""
    return _CACHE["severity_weights"]


def get_medicine_data() -> dict[str, dict[str, Any]] | None:
    """Return cached medicine data dict, or None if not yet loaded."""
    return _CACHE["medicines"]


def get_diet_data() -> dict[str, list[str]] | None:
    """Return cached diet data dict, or None if not yet loaded."""
    return _CACHE["diets"]


def get_all_diseases() -> list[str]:
    """
    Return the union of all disease names across all loaded datasets.
    Useful for the GET /module2/diseases endpoint.
    """
    disease_sets: list[set[str]] = []
    for key in ("precautions", "medicines", "diets"):
        data = _CACHE[key]
        if data:
            disease_sets.append(set(data.keys()))
    if not disease_sets:
        return []
    return sorted(set.union(*disease_sets))


def is_any_dataset_loaded() -> bool:
    """Return True if at least one dataset has been successfully loaded."""
    return any(v is not None for v in _CACHE.values())


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

def reload(keys: list[str] | None = None) -> dict[str, str]:
    """
    Re-read datasets from disk using the last-registered file paths.
    Call this after uploading new files to refresh the in-memory cache.

    Parameters
    ----------
    keys : Cache keys to refresh ("precautions", "severity_weights",
           "medicines", "diets"). None = refresh all known paths.

    Returns
    -------
    dict mapping each key -> "ok" | "skipped (no path)" | error string.
    """
    _loaders = {
        "precautions":      load_precautions,
        "severity_weights": load_severity_weights,
        "medicines":        load_medicine_data,
        "diets":            load_diet_data,
    }
    target_keys = keys if keys else list(_loaders.keys())
    summary: dict[str, str] = {}

    for key in target_keys:
        path = _PATHS.get(key)
        if path is None:
            summary[key] = "skipped (no path registered)"
            continue
        try:
            _loaders[key](path)
            summary[key] = "ok"
        except Exception as exc:  # noqa: BLE001
            summary[key] = f"error: {exc}"
            logger.error("Reload failed for '%s': %s", key, exc)

    return summary
