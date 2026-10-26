"""
data_loader.py – VitalSense AI | Module 1: Disease Prediction
=============================================================
Handles TWO dataset formats automatically:

  Format A – Symptom-Name (your actual dataset):
    Columns: Disease, Symptom_1, Symptom_2, ..., Symptom_N
    Values : String symptom names (or empty / NaN)
    Action : Pivot into a binary symptom×disease matrix.

  Format B – Binary Matrix (generic):
    Columns: symptom_a, symptom_b, ..., disease
    Values : 0 / 1 integers + label column
    Action : Use as-is after validation.

Author : VitalSense AI Team
"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Module-level logger – callers can configure the root logger as needed.      #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# Column names that are recognised as the label column (case-insensitive)
_LABEL_ALIASES: Tuple[str, ...] = ("disease", "prognosis", "label", "target")

# Minimum dataset quality thresholds
_MIN_ROWS: int = 20
_MIN_CLASSES: int = 2


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def load_and_validate_dataset(
    file_path: "str | Path",
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load a CSV dataset, detect its format, and return a clean binary feature
    matrix ready for scikit-learn.

    Parameters
    ----------
    file_path : str | Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    X : pd.DataFrame
        Binary feature matrix  (rows = samples, cols = symptom names).
    y : pd.Series
        Target labels (disease names).
    symptom_cols : list[str]
        Ordered list of symptom column names used as features.
    classes : list[str]
        Sorted unique disease class names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, has wrong format, or fails quality checks.
    """
    file_path = Path(file_path)
    logger.info("Loading dataset from: %s", file_path)

    # ------------------------------------------------------------------ #
    # 1. Read the file                                                     #
    # ------------------------------------------------------------------ #
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    logger.info("Raw shape: %s", df.shape)

    # ------------------------------------------------------------------ #
    # 2. Detect format                                                     #
    # ------------------------------------------------------------------ #
    if _is_symptom_name_format(df):
        logger.info("Detected Format A – Symptom-Name columns.")
        X, y = _parse_symptom_name_format(df)
    else:
        logger.info("Detected Format B – Binary matrix.")
        X, y = _parse_binary_format(df)

    # ------------------------------------------------------------------ #
    # 3. Drop rows where ALL symptom values are zero (truly empty rows)   #
    # ------------------------------------------------------------------ #
    before = len(X)
    mask = X.sum(axis=1) > 0          # keep rows that have at least 1 symptom
    X = X[mask]
    y = y[mask]
    dropped = before - len(X)
    if dropped:
        logger.warning("Dropped %d rows with no symptoms.", dropped)

    # ------------------------------------------------------------------ #
    # 4. Quality validation                                                #
    # ------------------------------------------------------------------ #
    if len(X) < _MIN_ROWS:
        raise ValueError(
            f"Dataset too small: only {len(X)} usable rows (minimum {_MIN_ROWS})."
        )

    classes: List[str] = sorted(y.unique().tolist())
    if len(classes) < _MIN_CLASSES:
        raise ValueError(
            f"Only {len(classes)} disease class found; need at least {_MIN_CLASSES}."
        )

    symptom_cols: List[str] = X.columns.tolist()

    logger.info(
        "Dataset ready – %d rows | %d symptoms | %d disease classes",
        len(X), len(symptom_cols), len(classes),
    )
    return X, y, symptom_cols, classes


# --------------------------------------------------------------------------- #
# Format detection                                                              #
# --------------------------------------------------------------------------- #

def _is_symptom_name_format(df: pd.DataFrame) -> bool:
    """
    Return True when the DataFrame looks like Format A (symptom-name columns).

    Heuristic:
      * The first column (or a column named 'disease'/'prognosis') is the label.
      * Remaining column *headers* are generic placeholders like
        'Symptom_1', 'Symptom_2', ...
      * The cell *values* contain strings (symptom names), not 0/1 integers.
    """
    cols = [c.strip().lower() for c in df.columns]

    # Check if any column header is a generic placeholder
    has_placeholder_headers = any(
        c.startswith("symptom_") or c.startswith("sym_") for c in cols
    )

    # Check if the non-label columns contain mostly string values
    label_col_idx = _find_label_column_index(df)
    feature_cols = [i for i in range(len(df.columns)) if i != label_col_idx]
    sample_vals = (
        df.iloc[:, feature_cols]
        .stack()          # flatten; NaN rows are automatically dropped
        .dropna()
        .astype(str)
        .str.strip()
        .head(50)
        .tolist()
    )

    # If many values are NOT '0' or '1', it is Format A
    non_binary_count = sum(
        1 for v in sample_vals if v not in ("0", "1", "0.0", "1.0", "")
    )
    majority_non_binary = len(sample_vals) > 0 and (
        non_binary_count / len(sample_vals) > 0.5
    )

    return has_placeholder_headers or majority_non_binary


def _find_label_column_index(df: pd.DataFrame) -> int:
    """
    Return the index of the label (disease) column.

    Priority:
      1. A column whose name matches one of _LABEL_ALIASES (case-insensitive).
      2. The first column (index 0) – common in symptom-name CSVs.
      3. The last column – common in binary-matrix CSVs.
    """
    for idx, col in enumerate(df.columns):
        if col.strip().lower() in _LABEL_ALIASES:
            return idx
    # If first column looks like a disease name (strings, not 0/1)
    first_col_vals = df.iloc[:, 0].dropna().astype(str).str.strip().unique()
    if any(not v.replace(".", "").isdigit() for v in first_col_vals[:10]):
        return 0          # first column is the label
    return len(df.columns) - 1   # fall back to last column


# --------------------------------------------------------------------------- #
# Format A parser – Symptom-Name columns                                       #
# --------------------------------------------------------------------------- #

def _parse_symptom_name_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert Format A (wide symptom-name layout) into a binary matrix.

    Steps:
      1. Identify label column; everything else is a symptom slot.
      2. Strip whitespace from all string values.
      3. Collect all unique non-null symptom names -> binary columns.
      4. For each row, set 1 in each column that matches a present symptom.

    Returns
    -------
    X : pd.DataFrame  – binary symptom matrix
    y : pd.Series     – disease labels
    """
    label_idx = _find_label_column_index(df)
    label_col = df.columns[label_idx]

    # Extract target labels – strip whitespace, drop NaN rows
    y: pd.Series = df[label_col].astype(str).str.strip()

    # Symptom slot columns (everything except the label column)
    symptom_slot_cols = [c for c in df.columns if c != label_col]

    def _clean_symptom(val) -> "str | None":
        """Normalise a symptom string: strip, lowercase, underscores."""
        if pd.isna(val):
            return None
        s = str(val).strip().lower().replace(" ", "_")
        # Collapse any double underscores produced by cleaning
        while "__" in s:
            s = s.replace("__", "_")
        return s if s else None

    # Apply cleaning to all symptom slot columns
    # Note: applymap() was renamed to map() in pandas 2.1+
    symptom_df = df[symptom_slot_cols].map(_clean_symptom)

    # Collect all unique symptom names across the entire dataset.
    # We filter with isinstance(val, str) to exclude any residual NaN floats
    # that may survive after pandas map() on missing cells.
    all_symptoms: List[str] = sorted(
        {
            val
            for val in symptom_df.values.flatten()
            if isinstance(val, str) and val  # skip None and float NaN
        }
    )
    logger.info("Unique symptoms found: %d", len(all_symptoms))

    # Build a set of present symptoms per row for fast membership testing
    symptom_set_per_row = symptom_df.apply(
        lambda row: set(filter(None, row.tolist())), axis=1
    )

    # Construct the binary feature matrix column by column
    binary_data = {
        sym: symptom_set_per_row.map(lambda s, sym=sym: int(sym in s))
        for sym in all_symptoms
    }

    X = pd.DataFrame(binary_data, index=df.index)
    return X, y


# --------------------------------------------------------------------------- #
# Format B parser – Binary matrix                                               #
# --------------------------------------------------------------------------- #

def _parse_binary_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Parse a dataset where every feature column already contains 0/1 values.

    Steps:
      1. Locate the label column.
      2. Coerce feature columns to integer (0/1); non-numeric -> NaN -> 0.
      3. Validate that all values are indeed 0 or 1.

    Returns
    -------
    X : pd.DataFrame  – binary symptom matrix
    y : pd.Series     – disease labels
    """
    label_idx = _find_label_column_index(df)
    label_col = df.columns[label_idx]

    y: pd.Series = df[label_col].astype(str).str.strip()
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].copy()

    # Coerce to numeric; anything unparseable becomes NaN, then fill with 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Validate binary constraint – all values must be 0 or 1
    unique_vals = set(X.values.flatten().tolist())
    invalid = unique_vals - {0, 1}
    if invalid:
        raise ValueError(
            f"Binary format detected but non-binary values found: {invalid}. "
            "Expected only 0 and 1 in feature columns."
        )

    # Normalise column names: strip and underscores
    X.columns = [c.strip().lower().replace(" ", "_") for c in X.columns]

    return X, y
