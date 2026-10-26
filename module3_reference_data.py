"""
reference_data.py — VitalSense AI · Module 3
=============================================
Responsible for loading, validating, and caching the reference-range CSV
that drives all rule-based classification in Module 3.

Key design decisions (for viva):
  * Two-level nested dict:  data[param_key][gender]  gives O(1) lookup.
  * "param_key" is the normalised form of the original Parameter column
    (lowercase, stripped, internal spaces → underscores).  This lets the
    caller use slightly different capitalisations and still match.
  * Thread-safe reload via a module-level lock so a running Flask server
    can hot-swap the CSV without corrupting in-flight requests.
  * No AI / ML anywhere — pure Pandas + Python.
"""

import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns — missing any of these raises a clear ValueError
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: list[str] = [
    "Parameter",
    "Unit",
    "Gender",
    "Min_Normal",
    "Max_Normal",
    "Critical_Low",
    "Critical_High",
]

# ---------------------------------------------------------------------------
# In-memory cache and thread lock
# ---------------------------------------------------------------------------
_cache: dict[str, Any] = {
    "data": None,          # nested dict: {param_key: {gender: range_dict}}
    "param_meta": None,    # list of {name, unit} for /parameters endpoint
    "source_path": None,   # Path of the file currently in cache
}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_param(name: str) -> str:
    """
    Convert a parameter name to the canonical lookup key.

    Rules:
        - Strip leading/trailing whitespace
        - Lowercase
        - Replace one-or-more internal whitespace characters with '_'

    Example: "Blood Urea Nitrogen" → "blood_urea_nitrogen"
    """
    import re
    return re.sub(r"\s+", "_", name.strip().lower())


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Raise ValueError if any REQUIRED_COLUMNS are absent from *df*.
    Names the specific missing columns so the caller can fix the CSV.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Reference CSV is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}. "
            f"Found columns: {list(df.columns)}"
        )


def _build_nested_dict(df: pd.DataFrame) -> tuple[dict, list]:
    """
    Transform the validated DataFrame into two structures:

    1. ``ref_dict``  {param_key: {gender: {unit, min, max, crit_low, crit_high}}}
       Multiple rows with the same Parameter but different Gender values
       are stored under their respective gender keys (Male / Female / Any).

    2. ``param_meta``  [{"name": original_name, "unit": unit}, ...]
       Used by the GET /parameters endpoint; deduplicated by param_key.

    Numeric coercion uses pandas.to_numeric with errors='coerce' so that
    any non-numeric entry becomes NaN rather than crashing the loader.
    """
    ref_dict: dict = {}
    seen_keys: dict[str, str] = {}  # param_key → original display name

    numeric_cols = ["Min_Normal", "Max_Normal", "Critical_Low", "Critical_High"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for _, row in df.iterrows():
        raw_name: str = str(row["Parameter"]).strip()
        key: str = _normalise_param(raw_name)
        gender: str = str(row["Gender"]).strip()
        unit: str = str(row["Unit"]).strip()

        if key not in ref_dict:
            ref_dict[key] = {}
            seen_keys[key] = raw_name  # keep first occurrence as display name

        ref_dict[key][gender] = {
            "unit": unit,
            "min_normal": float(row["Min_Normal"]) if not pd.isna(row["Min_Normal"]) else None,
            "max_normal": float(row["Max_Normal"]) if not pd.isna(row["Max_Normal"]) else None,
            "critical_low": float(row["Critical_Low"]) if not pd.isna(row["Critical_Low"]) else None,
            "critical_high": float(row["Critical_High"]) if not pd.isna(row["Critical_High"]) else None,
        }

    # Build deduplicated param_meta list (one entry per unique parameter)
    param_meta = [
        {"name": seen_keys[key], "unit": list(ref_dict[key].values())[0]["unit"]}
        for key in ref_dict
    ]

    return ref_dict, param_meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_reference_ranges(file_path: str | Path) -> dict:
    """
    Load and validate the reference-range CSV, cache it in memory, and
    return the nested reference dict.

    Parameters
    ----------
    file_path : str | Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    dict
        Nested dict  {param_key: {gender: range_dict}}.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *file_path*.
    ValueError
        If required columns are missing or the file cannot be parsed as CSV.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {path}")

    logger.info("Loading reference ranges from: %s", path)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file '{path}': {exc}") from exc

    # --- Column validation (raises with clear message on failure) ---
    _validate_columns(df)

    # Drop rows where Parameter is completely blank
    df = df[df["Parameter"].notna() & (df["Parameter"].astype(str).str.strip() != "")]
    df.reset_index(drop=True, inplace=True)

    # --- Build the lookup structures ---
    ref_dict, param_meta = _build_nested_dict(df)

    # --- Thread-safe cache update ---
    with _cache_lock:
        _cache["data"] = ref_dict
        _cache["param_meta"] = param_meta
        _cache["source_path"] = path

    logger.info(
        "Reference ranges loaded: %d unique parameters from %s",
        len(ref_dict),
        path,
    )
    return ref_dict


def reload() -> dict:
    """
    Reload the reference CSV from the same path that was used in the
    most-recent successful ``load_reference_ranges()`` call.

    Returns
    -------
    dict
        Freshly-loaded nested reference dict.

    Raises
    ------
    RuntimeError
        If no dataset has been loaded yet (no cached path to reload from).
    """
    with _cache_lock:
        source = _cache.get("source_path")

    if source is None:
        raise RuntimeError(
            "reload() called before any dataset was loaded. "
            "Call load_reference_ranges(file_path) first."
        )

    logger.info("Reloading reference ranges from cached path: %s", source)
    return load_reference_ranges(source)


def get_cached_data() -> dict | None:
    """
    Return the currently-cached reference dict (or None if not yet loaded).
    Thread-safe read.
    """
    with _cache_lock:
        return _cache["data"]


def get_cached_meta() -> list | None:
    """
    Return the currently-cached parameter-metadata list (or None).
    Thread-safe read.
    """
    with _cache_lock:
        return _cache["param_meta"]


def is_loaded() -> bool:
    """Return True if a dataset is currently cached in memory."""
    with _cache_lock:
        return _cache["data"] is not None
