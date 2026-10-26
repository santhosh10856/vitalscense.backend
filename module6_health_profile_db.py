"""
health_profile_db.py — VitalSense AI | Module 6: Personal Health Profile
=========================================================================
SQLite persistence layer for user health profiles.

Tables
------
  user_health_profiles — one row per user_id, all profile fields

Design: Follows the same pattern as module5_medicine/medicine_db.py
        (connection helper, DDL on first connect, WAL journal mode).

Author: VitalSense AI Team
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH: Path = Path(__file__).parent / "health_profile.db"
DB_PATH: str = os.environ.get("HEALTH_PROFILE_DB_PATH", str(_DEFAULT_DB_PATH))

# ---------------------------------------------------------------------------
# All profile fields with metadata for completion tracking
# ---------------------------------------------------------------------------
# (field_name, is_required_for_completion, field_type)
PROFILE_FIELDS: list[tuple[str, bool, str]] = [
    ("full_name",                    True,  "text"),
    ("date_of_birth",                True,  "text"),
    ("gender",                       False, "text"),
    ("height_cm",                    True,  "number"),
    ("weight_kg",                    True,  "number"),
    ("blood_group",                  True,  "text"),
    ("allergies",                    False, "json"),
    ("medical_conditions",           False, "json"),
    ("emergency_contact_name",       True,  "text"),
    ("emergency_contact_phone",      True,  "text"),
    ("emergency_contact_relationship", True, "text"),
    ("current_medications",          False, "json"),
    ("previous_surgeries",           False, "json"),
    ("family_history",               False, "json"),
    ("vaccination_status",           False, "json"),
    ("smoking",                      False, "text"),
    ("alcohol",                      False, "text"),
    ("exercise",                     False, "text"),
    ("diet_preference",              False, "text"),
    ("sleep_hours",                  False, "number"),
    ("health_notes",                 False, "text"),
]

TOTAL_FIELDS = len(PROFILE_FIELDS)

# ---------------------------------------------------------------------------
# DDL — Table creation
# ---------------------------------------------------------------------------
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS user_health_profiles (
    id                             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                        TEXT    UNIQUE NOT NULL,
    full_name                      TEXT    DEFAULT '',
    date_of_birth                  TEXT    DEFAULT '',
    gender                         TEXT    DEFAULT '',
    height_cm                      REAL    DEFAULT 0,
    weight_kg                      REAL    DEFAULT 0,
    bmi                            REAL    DEFAULT 0,
    blood_group                    TEXT    DEFAULT '',
    allergies                      TEXT    DEFAULT '[]',
    medical_conditions             TEXT    DEFAULT '[]',
    emergency_contact_name         TEXT    DEFAULT '',
    emergency_contact_phone        TEXT    DEFAULT '',
    emergency_contact_relationship TEXT    DEFAULT '',
    current_medications            TEXT    DEFAULT '[]',
    previous_surgeries             TEXT    DEFAULT '[]',
    family_history                 TEXT    DEFAULT '[]',
    vaccination_status             TEXT    DEFAULT '[]',
    smoking                        TEXT    DEFAULT '',
    alcohol                        TEXT    DEFAULT '',
    exercise                       TEXT    DEFAULT '',
    diet_preference                TEXT    DEFAULT '',
    sleep_hours                    REAL    DEFAULT 0,
    health_notes                   TEXT    DEFAULT '',
    profile_completion             INTEGER DEFAULT 0,
    created_at                     TEXT    DEFAULT (datetime('now')),
    updated_at                     TEXT    DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """
    Open the SQLite database, run DDL to ensure tables exist,
    and return a connection with row_factory = sqlite3.Row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(_CREATE_TABLES_SQL)
    conn.commit()
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    """Convert a sqlite3.Row to a plain dict, or None."""
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# BMI Calculation
# ---------------------------------------------------------------------------

def _calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate BMI from height (cm) and weight (kg). Returns 0 if invalid."""
    if not height_cm or height_cm <= 0 or not weight_kg or weight_kg <= 0:
        return 0.0
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m * height_m), 1)


def _bmi_category(bmi: float) -> str:
    """Return BMI category string. No diagnosis — just standard WHO classification."""
    if bmi <= 0:
        return "Not calculated"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25.0:
        return "Normal weight"
    if bmi < 30.0:
        return "Overweight"
    return "Obese"


# ---------------------------------------------------------------------------
# Profile Completion
# ---------------------------------------------------------------------------

def _calculate_completion(data: dict) -> int:
    """
    Calculate profile completion percentage based on filled fields.
    Returns integer 0–100.
    """
    filled = 0
    for field_name, _required, field_type in PROFILE_FIELDS:
        value = data.get(field_name)
        if field_type == "json":
            # JSON fields: non-empty array counts as filled
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if parsed and isinstance(parsed, list) and len(parsed) > 0:
                        filled += 1
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(value, list) and len(value) > 0:
                filled += 1
        elif field_type == "number":
            if value and float(value) > 0:
                filled += 1
        else:  # text
            if value and str(value).strip():
                filled += 1

    return round((filled / TOTAL_FIELDS) * 100)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Ensure the database and all tables exist."""
    with _get_connection() as conn:
        logger.info("[Module 6] Health Profile database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# CRUD — Get Profile
# ---------------------------------------------------------------------------

def get_profile(user_id: str) -> Optional[dict]:
    """
    Return the full health profile for a user, or None if not found.
    JSON array fields are parsed into Python lists.
    """
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM user_health_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()

    if row is None:
        return None

    profile = _row_to_dict(row)

    # Parse JSON fields
    json_fields = [
        "allergies", "medical_conditions", "current_medications",
        "previous_surgeries", "family_history", "vaccination_status",
    ]
    for field in json_fields:
        raw = profile.get(field, "[]")
        try:
            profile[field] = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            profile[field] = []

    return profile


# ---------------------------------------------------------------------------
# CRUD — Save / Update Profile
# ---------------------------------------------------------------------------

def save_profile(user_id: str, data: dict) -> dict:
    """
    Insert or update a health profile for the given user_id.
    Auto-calculates BMI and profile_completion.
    Returns the saved profile dict.
    """
    # Convert list fields to JSON strings
    json_fields = [
        "allergies", "medical_conditions", "current_medications",
        "previous_surgeries", "family_history", "vaccination_status",
    ]

    clean_data = {}
    for key, value in data.items():
        if key in json_fields:
            if isinstance(value, list):
                clean_data[key] = json.dumps(value)
            elif isinstance(value, str):
                # Validate it's valid JSON
                try:
                    json.loads(value)
                    clean_data[key] = value
                except (json.JSONDecodeError, TypeError):
                    clean_data[key] = json.dumps([])
            else:
                clean_data[key] = json.dumps([])
        else:
            clean_data[key] = value

    # Auto-calculate BMI
    height = float(clean_data.get("height_cm", 0) or 0)
    weight = float(clean_data.get("weight_kg", 0) or 0)
    clean_data["bmi"] = _calculate_bmi(height, weight)

    # Calculate completion
    clean_data["profile_completion"] = _calculate_completion(clean_data)

    # Timestamp
    clean_data["updated_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    with _get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM user_health_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if existing:
            # Update
            set_clauses = []
            values = []
            for key, value in clean_data.items():
                if key not in ("id", "user_id", "created_at"):
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            values.append(user_id)

            conn.execute(
                f"UPDATE user_health_profiles SET {', '.join(set_clauses)} WHERE user_id = ?",
                values
            )
        else:
            # Insert
            clean_data["user_id"] = user_id
            clean_data["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            columns = list(clean_data.keys())
            placeholders = ", ".join(["?"] * len(columns))
            col_names = ", ".join(columns)

            conn.execute(
                f"INSERT INTO user_health_profiles ({col_names}) VALUES ({placeholders})",
                [clean_data[c] for c in columns]
            )

        conn.commit()

    return get_profile(user_id)


# ---------------------------------------------------------------------------
# Emergency Information
# ---------------------------------------------------------------------------

def get_emergency_info(user_id: str) -> Optional[dict]:
    """
    Return only emergency-critical fields for quick access.
    Returns None if no profile exists.
    """
    profile = get_profile(user_id)
    if profile is None:
        return None

    return {
        "full_name":                      profile.get("full_name", ""),
        "blood_group":                    profile.get("blood_group", ""),
        "allergies":                      profile.get("allergies", []),
        "medical_conditions":             profile.get("medical_conditions", []),
        "current_medications":            profile.get("current_medications", []),
        "emergency_contact_name":         profile.get("emergency_contact_name", ""),
        "emergency_contact_phone":        profile.get("emergency_contact_phone", ""),
        "emergency_contact_relationship": profile.get("emergency_contact_relationship", ""),
    }


# ---------------------------------------------------------------------------
# Smart Health Summary
# ---------------------------------------------------------------------------

def get_profile_summary(user_id: str) -> Optional[dict]:
    """
    Generate a smart health summary from saved profile data.
    This is purely informational — NO medical diagnosis or conclusions.
    """
    profile = get_profile(user_id)
    if profile is None:
        return None

    bmi = profile.get("bmi", 0)
    bmi_cat = _bmi_category(bmi)

    summary = {
        "bmi": bmi,
        "bmi_category": bmi_cat,
        "blood_group": profile.get("blood_group", "Not specified"),
        "allergies": profile.get("allergies", []),
        "medical_conditions": profile.get("medical_conditions", []),
        "current_medications": profile.get("current_medications", []),
        "emergency_contact": {
            "name":         profile.get("emergency_contact_name", ""),
            "phone":        profile.get("emergency_contact_phone", ""),
            "relationship": profile.get("emergency_contact_relationship", ""),
        },
        "lifestyle": {
            "smoking":    profile.get("smoking", "Not specified"),
            "alcohol":    profile.get("alcohol", "Not specified"),
            "exercise":   profile.get("exercise", "Not specified"),
            "diet":       profile.get("diet_preference", "Not specified"),
            "sleep":      profile.get("sleep_hours", 0),
        },
        "profile_completion": profile.get("profile_completion", 0),
        "disclaimer": (
            "This summary is based on your self-reported health information. "
            "It is NOT a medical diagnosis or assessment. Please consult a "
            "qualified healthcare professional for medical advice."
        ),
    }

    return summary


# ---------------------------------------------------------------------------
# AI Context (plain-text for chatbot integration)
# ---------------------------------------------------------------------------

def get_profile_context(user_id: str) -> str:
    """
    Return a plain-text summary of the user's health profile for the AI
    chatbot context. The AI must NEVER diagnose, prescribe, or modify treatment.
    Returns empty string if no profile exists.
    """
    profile = get_profile(user_id)
    if profile is None:
        return ""

    lines = []

    name = profile.get("full_name", "").strip()
    if name:
        lines.append(f"Name: {name}")

    dob = profile.get("date_of_birth", "").strip()
    if dob:
        lines.append(f"Date of Birth: {dob}")

    gender = profile.get("gender", "").strip()
    if gender:
        lines.append(f"Gender: {gender}")

    height = profile.get("height_cm", 0)
    weight = profile.get("weight_kg", 0)
    bmi = profile.get("bmi", 0)
    if height and float(height) > 0:
        lines.append(f"Height: {height} cm")
    if weight and float(weight) > 0:
        lines.append(f"Weight: {weight} kg")
    if bmi and float(bmi) > 0:
        lines.append(f"BMI: {bmi} ({_bmi_category(float(bmi))})")

    bg = profile.get("blood_group", "").strip()
    if bg:
        lines.append(f"Blood Group: {bg}")

    allergies = profile.get("allergies", [])
    if allergies:
        lines.append(f"Allergies: {', '.join(allergies)}")

    conditions = profile.get("medical_conditions", [])
    if conditions:
        lines.append(f"Medical Conditions: {', '.join(conditions)}")

    meds = profile.get("current_medications", [])
    if meds:
        lines.append(f"Current Medications: {', '.join(meds)}")

    surgeries = profile.get("previous_surgeries", [])
    if surgeries:
        lines.append(f"Previous Surgeries: {', '.join(surgeries)}")

    family = profile.get("family_history", [])
    if family:
        lines.append(f"Family Medical History: {', '.join(family)}")

    vaccines = profile.get("vaccination_status", [])
    if vaccines:
        lines.append(f"Vaccinations: {', '.join(vaccines)}")

    smoking = profile.get("smoking", "").strip()
    if smoking:
        lines.append(f"Smoking: {smoking}")

    alcohol = profile.get("alcohol", "").strip()
    if alcohol:
        lines.append(f"Alcohol: {alcohol}")

    exercise = profile.get("exercise", "").strip()
    if exercise:
        lines.append(f"Exercise: {exercise}")

    diet = profile.get("diet_preference", "").strip()
    if diet:
        lines.append(f"Diet Preference: {diet}")

    sleep = profile.get("sleep_hours", 0)
    if sleep and float(sleep) > 0:
        lines.append(f"Sleep: {sleep} hours/night")

    notes = profile.get("health_notes", "").strip()
    if notes:
        lines.append(f"Health Notes: {notes}")

    ec_name = profile.get("emergency_contact_name", "").strip()
    ec_phone = profile.get("emergency_contact_phone", "").strip()
    ec_rel = profile.get("emergency_contact_relationship", "").strip()
    if ec_name:
        ec_line = f"Emergency Contact: {ec_name}"
        if ec_rel:
            ec_line += f" ({ec_rel})"
        if ec_phone:
            ec_line += f" — {ec_phone}"
        lines.append(ec_line)

    if not lines:
        return ""

    return "User Health Profile:\n" + "\n".join(lines)
