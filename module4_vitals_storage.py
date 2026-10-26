"""
vitals_storage.py
=================
VitalSense AI - Heart Rate & SpO2 Monitoring Module
SQLite persistence layer for vitals readings.

Table schema (CREATE TABLE statement)
--------------------------------------
CREATE TABLE IF NOT EXISTS vitals_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    device_id       TEXT    NOT NULL,
    heart_rate_bpm  INTEGER,
    spo2_percent    REAL,
    recorded_at     TEXT    NOT NULL
);

SQLite is part of the Python standard library (sqlite3) so no external
dependency is needed.  The database file path is resolved from an
environment variable VITALSENSE_DB_PATH (default: vitalsense.db in the
same directory as this file) so there are no hardcoded absolute paths.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path resolution (no hardcoded paths)
# ---------------------------------------------------------------------------

# Allow operators to override the DB location via an environment variable.
# Default: a file called vitalsense.db sitting next to this source file.
_DEFAULT_DB_PATH: Path = Path(__file__).parent / "vitalsense.db"
DB_PATH: str = os.environ.get("VITALSENSE_DB_PATH", str(_DEFAULT_DB_PATH))

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS vitals_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    device_id       TEXT    NOT NULL,
    heart_rate_bpm  INTEGER,          -- nullable: device may only send SpO2
    spo2_percent    REAL,             -- nullable: device may only send HR
    recorded_at     TEXT    NOT NULL  -- ISO-8601 string from the wearable
);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """
    Open (and if necessary create) the SQLite database, run DDL to
    ensure the table exists, and return the connection.

    Returns
    -------
    sqlite3.Connection
        A connection with row_factory set to sqlite3.Row so callers
        can access columns by name.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Write-ahead logging for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    """Convert a sqlite3.Row to a plain dict (or None if row is None)."""
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_reading(reading: dict) -> int:
    """
    Persist a single vitals reading to the database.

    Parameters
    ----------
    reading : dict
        Dictionary with keys: user_id, device_id, heart_rate_bpm
        (optional), spo2_percent (optional), recorded_at.

    Returns
    -------
    int
        The auto-incremented primary key (row id) of the inserted row.
    """
    sql = """
        INSERT INTO vitals_readings
            (user_id, device_id, heart_rate_bpm, spo2_percent, recorded_at)
        VALUES
            (:user_id, :device_id, :heart_rate_bpm, :spo2_percent, :recorded_at)
    """
    with _get_connection() as conn:
        cursor = conn.execute(sql, {
            "user_id":        reading["user_id"],
            "device_id":      reading["device_id"],
            "heart_rate_bpm": reading.get("heart_rate_bpm"),   # may be None
            "spo2_percent":   reading.get("spo2_percent"),      # may be None
            "recorded_at":    reading["recorded_at"],
        })
        conn.commit()
        row_id: int = cursor.lastrowid
        logger.debug("Saved reading id=%d for user=%s", row_id, reading["user_id"])
        return row_id


def get_latest(user_id: str) -> Optional[dict]:
    """
    Fetch the most recent reading for a given user.

    Parameters
    ----------
    user_id : str
        The user whose latest reading is requested.

    Returns
    -------
    dict | None
        The most recent row as a dict, or None if no readings exist.
    """
    sql = """
        SELECT *
        FROM   vitals_readings
        WHERE  user_id = ?
        ORDER  BY recorded_at DESC
        LIMIT  1
    """
    with _get_connection() as conn:
        row = conn.execute(sql, (user_id,)).fetchone()
        return _row_to_dict(row)


def get_history(user_id: str, start: str, end: str) -> list[dict]:
    """
    Fetch all readings for a user within an ISO-8601 timestamp range
    (inclusive on both ends).

    Parameters
    ----------
    user_id : str
        Target user.
    start : str
        ISO-8601 start timestamp, e.g. "2026-09-01T00:00:00".
    end : str
        ISO-8601 end timestamp, e.g. "2026-09-01T23:59:59".

    Returns
    -------
    list[dict]
        Ordered list (oldest first) of matching reading dicts.
    """
    sql = """
        SELECT *
        FROM   vitals_readings
        WHERE  user_id     = ?
          AND  recorded_at >= ?
          AND  recorded_at <= ?
        ORDER  BY recorded_at ASC
    """
    with _get_connection() as conn:
        rows = conn.execute(sql, (user_id, start, end)).fetchall()
        return [dict(r) for r in rows]


def get_daily_average(user_id: str, date: str) -> dict:
    """
    Compute average heart_rate_bpm and spo2_percent for a calendar day.

    NULL values in either column are excluded from their respective
    averages (AVG in SQLite already ignores NULLs).

    Parameters
    ----------
    user_id : str
        Target user.
    date : str
        Calendar date in "YYYY-MM-DD" format.

    Returns
    -------
    dict
        Keys: avg_heart_rate_bpm (float | None), avg_spo2_percent (float | None),
              reading_count (int), date (str), user_id (str).
    """
    sql = """
        SELECT
            AVG(heart_rate_bpm) AS avg_heart_rate_bpm,
            AVG(spo2_percent)   AS avg_spo2_percent,
            COUNT(*)            AS reading_count
        FROM   vitals_readings
        WHERE  user_id     = ?
          AND  DATE(recorded_at) = ?   -- SQLite DATE() extracts the date part
    """
    with _get_connection() as conn:
        row = conn.execute(sql, (user_id, date)).fetchone()
        result = dict(row) if row else {}
        result["date"] = date
        result["user_id"] = user_id
        return result
