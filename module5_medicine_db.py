"""
medicine_db.py — VitalSense AI | Module 5: Medicine Reminder
=============================================================
SQLite persistence layer for medicine prescriptions, schedules,
and adherence history.

Tables
------
  users              — linked to existing mock auth by user_id
  medicines          — prescribed medicines with dosage & timing
  medicine_schedules — daily schedule entries (auto-generated)
  medicine_history   — immutable audit trail of actions

Design: Follows the same pattern as module4_vitalsense/vitals_storage.py
        (connection helper, DDL on first connect, WAL journal mode).

Author: VitalSense AI Team
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH: Path = Path(__file__).parent / "medicine_reminder.db"
DB_PATH: str = os.environ.get("MEDICINE_DB_PATH", str(_DEFAULT_DB_PATH))

# ---------------------------------------------------------------------------
# DDL — Table creation statements
# ---------------------------------------------------------------------------
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    UNIQUE NOT NULL,
    name        TEXT    NOT NULL,
    email       TEXT    DEFAULT '',
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS medicines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    dosage          TEXT    NOT NULL,
    frequency       TEXT    NOT NULL,
    reminder_times  TEXT    NOT NULL,
    start_date      TEXT    NOT NULL,
    end_date        TEXT    DEFAULT '',
    food_instruction TEXT   DEFAULT 'before',
    notes           TEXT    DEFAULT '',
    is_enabled      INTEGER DEFAULT 1,
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS medicine_schedules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    medicine_id     INTEGER NOT NULL,
    scheduled_date  TEXT    NOT NULL,
    scheduled_time  TEXT    NOT NULL,
    status          TEXT    DEFAULT 'pending',
    FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS medicine_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    medicine_id     INTEGER NOT NULL,
    schedule_id     INTEGER,
    medicine_name   TEXT    NOT NULL,
    dosage          TEXT    NOT NULL,
    scheduled_date  TEXT    NOT NULL,
    scheduled_time  TEXT    NOT NULL,
    action          TEXT    NOT NULL,
    action_time     TEXT    DEFAULT (datetime('now')),
    notes           TEXT    DEFAULT '',
    FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE SET NULL
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


def _rows_to_list(rows: list) -> list[dict]:
    """Convert a list of sqlite3.Row objects to a list of dicts."""
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Ensure the database and all tables exist."""
    with _get_connection() as conn:
        logger.info("[Module 5] Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

def upsert_user(user_id: str, name: str, email: str = "") -> dict:
    """
    Insert a user if they don't exist, or update name/email if they do.
    Returns the user dict.
    """
    sql_upsert = """
        INSERT INTO users (user_id, name, email)
        VALUES (:user_id, :name, :email)
        ON CONFLICT(user_id) DO UPDATE SET
            name  = excluded.name,
            email = excluded.email
    """
    with _get_connection() as conn:
        conn.execute(sql_upsert, {
            "user_id": user_id,
            "name":    name,
            "email":   email,
        })
        conn.commit()

    return {"user_id": user_id, "name": name, "email": email}


# ---------------------------------------------------------------------------
# Medicine CRUD
# ---------------------------------------------------------------------------

def add_medicine(user_id: str, name: str, dosage: str, frequency: str,
                 reminder_times: list[str], start_date: str,
                 end_date: str = "", food_instruction: str = "before",
                 notes: str = "") -> dict:
    """
    Add a new medicine prescription.

    Parameters
    ----------
    reminder_times : list[str]
        List of HH:MM time strings, e.g. ["08:00", "20:00"]
    """
    sql = """
        INSERT INTO medicines
            (user_id, name, dosage, frequency, reminder_times,
             start_date, end_date, food_instruction, notes)
        VALUES
            (:user_id, :name, :dosage, :frequency, :reminder_times,
             :start_date, :end_date, :food_instruction, :notes)
    """
    with _get_connection() as conn:
        cursor = conn.execute(sql, {
            "user_id":          user_id,
            "name":             name,
            "dosage":           dosage,
            "frequency":        frequency,
            "reminder_times":   json.dumps(reminder_times),
            "start_date":       start_date,
            "end_date":         end_date,
            "food_instruction": food_instruction,
            "notes":            notes,
        })
        conn.commit()
        med_id = cursor.lastrowid
        logger.info("[Module 5] Added medicine id=%d '%s' for user=%s",
                    med_id, name, user_id)

    return get_medicine(med_id)


def get_medicines(user_id: str) -> list[dict]:
    """Return all medicines for a user."""
    sql = """
        SELECT * FROM medicines
        WHERE user_id = ?
        ORDER BY created_at DESC
    """
    with _get_connection() as conn:
        rows = conn.execute(sql, (user_id,)).fetchall()
        result = _rows_to_list(rows)

    # Parse reminder_times from JSON string
    for med in result:
        try:
            med["reminder_times"] = json.loads(med["reminder_times"])
        except (json.JSONDecodeError, TypeError):
            med["reminder_times"] = []

    return result


def get_medicine(med_id: int) -> Optional[dict]:
    """Return a single medicine by ID."""
    sql = "SELECT * FROM medicines WHERE id = ?"
    with _get_connection() as conn:
        row = conn.execute(sql, (med_id,)).fetchone()
        result = _row_to_dict(row)

    if result:
        try:
            result["reminder_times"] = json.loads(result["reminder_times"])
        except (json.JSONDecodeError, TypeError):
            result["reminder_times"] = []

    return result


def update_medicine(med_id: int, **fields) -> Optional[dict]:
    """
    Update one or more fields of a medicine.
    Allowed fields: name, dosage, frequency, reminder_times, start_date,
                    end_date, food_instruction, notes, is_enabled
    """
    allowed = {
        "name", "dosage", "frequency", "reminder_times",
        "start_date", "end_date", "food_instruction", "notes", "is_enabled",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_medicine(med_id)

    # Serialise reminder_times if present
    if "reminder_times" in updates and isinstance(updates["reminder_times"], list):
        updates["reminder_times"] = json.dumps(updates["reminder_times"])

    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")

    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    sql = f"UPDATE medicines SET {set_clause} WHERE id = :id"
    updates["id"] = med_id

    with _get_connection() as conn:
        conn.execute(sql, updates)
        conn.commit()

    logger.info("[Module 5] Updated medicine id=%d fields=%s",
                med_id, list(updates.keys()))
    return get_medicine(med_id)


def delete_medicine(med_id: int) -> bool:
    """Delete a medicine and its related schedules (CASCADE)."""
    with _get_connection() as conn:
        cursor = conn.execute("DELETE FROM medicines WHERE id = ?", (med_id,))
        conn.commit()
        deleted = cursor.rowcount > 0

    if deleted:
        logger.info("[Module 5] Deleted medicine id=%d", med_id)
    return deleted


def toggle_medicine(med_id: int) -> Optional[dict]:
    """Toggle the is_enabled flag of a medicine."""
    med = get_medicine(med_id)
    if not med:
        return None
    new_state = 0 if med["is_enabled"] else 1
    return update_medicine(med_id, is_enabled=new_state)


# ---------------------------------------------------------------------------
# Schedule generation & management
# ---------------------------------------------------------------------------

def generate_today_schedule(user_id: str) -> list[dict]:
    """
    Generate schedule entries for today based on active medicines.
    If entries already exist for today, they are returned as-is.
    """
    today = date.today().isoformat()

    with _get_connection() as conn:
        # Check if schedule already generated for today
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM medicine_schedules "
            "WHERE user_id = ? AND scheduled_date = ?",
            (user_id, today),
        ).fetchone()

        if existing["cnt"] == 0:
            # Generate from active medicines
            medicines = conn.execute(
                "SELECT * FROM medicines WHERE user_id = ? AND is_enabled = 1",
                (user_id,),
            ).fetchall()

            for med in medicines:
                # Check date range
                start = med["start_date"]
                end = med["end_date"]
                if start and today < start:
                    continue
                if end and today > end:
                    continue

                try:
                    times = json.loads(med["reminder_times"])
                except (json.JSONDecodeError, TypeError):
                    times = []

                for t in times:
                    conn.execute(
                        "INSERT INTO medicine_schedules "
                        "(user_id, medicine_id, scheduled_date, scheduled_time, status) "
                        "VALUES (?, ?, ?, ?, 'pending')",
                        (user_id, med["id"], today, t),
                    )

            conn.commit()
            logger.info("[Module 5] Generated today's schedule for user=%s", user_id)

        # Fetch today's schedule with medicine details
        rows = conn.execute("""
            SELECT
                s.id,
                s.user_id,
                s.medicine_id,
                s.scheduled_date,
                s.scheduled_time,
                s.status,
                m.name AS medicine_name,
                m.dosage,
                m.food_instruction,
                m.frequency,
                m.notes AS medicine_notes
            FROM medicine_schedules s
            JOIN medicines m ON s.medicine_id = m.id
            WHERE s.user_id = ? AND s.scheduled_date = ?
            ORDER BY s.scheduled_time ASC
        """, (user_id, today)).fetchall()

    return _rows_to_list(rows)


def mark_schedule(schedule_id: int, action: str, notes: str = "") -> Optional[dict]:
    """
    Mark a schedule entry as taken / missed / snoozed.
    Creates an immutable history record.

    For 'snoozed': creates a new schedule entry +15 minutes later.
    """
    if action not in ("taken", "missed", "snoozed"):
        return None

    with _get_connection() as conn:
        # Fetch the schedule entry
        row = conn.execute(
            "SELECT s.*, m.name AS medicine_name, m.dosage "
            "FROM medicine_schedules s "
            "JOIN medicines m ON s.medicine_id = m.id "
            "WHERE s.id = ?",
            (schedule_id,),
        ).fetchone()

        if not row:
            return None

        schedule = dict(row)
        now_str = datetime.now().isoformat(timespec="seconds")

        # Update schedule status
        new_status = action if action != "snoozed" else "snoozed"
        conn.execute(
            "UPDATE medicine_schedules SET status = ? WHERE id = ?",
            (new_status, schedule_id),
        )

        # Write to history
        conn.execute("""
            INSERT INTO medicine_history
                (user_id, medicine_id, schedule_id, medicine_name,
                 dosage, scheduled_date, scheduled_time, action,
                 action_time, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            schedule["user_id"],
            schedule["medicine_id"],
            schedule_id,
            schedule["medicine_name"],
            schedule["dosage"],
            schedule["scheduled_date"],
            schedule["scheduled_time"],
            action,
            now_str,
            notes,
        ))

        # If snoozed, create a new schedule entry +15 min
        if action == "snoozed":
            try:
                orig_time = datetime.strptime(schedule["scheduled_time"], "%H:%M")
                snooze_time = (orig_time + timedelta(minutes=15)).strftime("%H:%M")
            except ValueError:
                snooze_time = schedule["scheduled_time"]

            conn.execute(
                "INSERT INTO medicine_schedules "
                "(user_id, medicine_id, scheduled_date, scheduled_time, status) "
                "VALUES (?, ?, ?, ?, 'pending')",
                (schedule["user_id"], schedule["medicine_id"],
                 schedule["scheduled_date"], snooze_time),
            )

        conn.commit()

    logger.info("[Module 5] Schedule id=%d marked as '%s'", schedule_id, action)
    return {"schedule_id": schedule_id, "action": action, "action_time": now_str}


# ---------------------------------------------------------------------------
# Dashboard & Analytics
# ---------------------------------------------------------------------------

def get_today_summary(user_id: str) -> dict:
    """
    Return today's dashboard summary:
    total, taken, missed, pending, snoozed, adherence_percent.
    """
    today = date.today().isoformat()

    # Ensure schedule is generated first
    generate_today_schedule(user_id)

    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM medicine_schedules "
            "WHERE user_id = ? AND scheduled_date = ? "
            "GROUP BY status",
            (user_id, today),
        ).fetchall()

    counts = {"pending": 0, "taken": 0, "missed": 0, "snoozed": 0}
    total = 0
    for r in rows:
        status = r["status"]
        count = r["cnt"]
        if status in counts:
            counts[status] = count
        total += count

    # Adherence = taken / (taken + missed) * 100
    decided = counts["taken"] + counts["missed"]
    adherence = round((counts["taken"] / decided * 100), 1) if decided > 0 else 100.0

    return {
        "date":       today,
        "total":      total,
        "taken":      counts["taken"],
        "missed":     counts["missed"],
        "pending":    counts["pending"],
        "snoozed":    counts["snoozed"],
        "adherence_percent": adherence,
    }


def get_history(user_id: str, start_date: str = "", end_date: str = "",
                limit: int = 100) -> list[dict]:
    """
    Return medicine history records, optionally filtered by date range.
    """
    params: list = [user_id]

    sql = """
        SELECT * FROM medicine_history
        WHERE user_id = ?
    """
    if start_date:
        sql += " AND scheduled_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND scheduled_date <= ?"
        params.append(end_date)

    sql += " ORDER BY action_time DESC LIMIT ?"
    params.append(limit)

    with _get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return _rows_to_list(rows)


# ---------------------------------------------------------------------------
# Reminder polling
# ---------------------------------------------------------------------------

def get_due_reminders(user_id: str) -> list[dict]:
    """
    Return schedule entries that are due now (within the current time window)
    and still pending. Used by the browser notification polling endpoint.
    """
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M")

    # Reminders due = scheduled_time <= now AND status is 'pending'
    with _get_connection() as conn:
        rows = conn.execute("""
            SELECT
                s.id,
                s.scheduled_time,
                s.status,
                m.name AS medicine_name,
                m.dosage,
                m.food_instruction
            FROM medicine_schedules s
            JOIN medicines m ON s.medicine_id = m.id
            WHERE s.user_id = ?
              AND s.scheduled_date = ?
              AND s.scheduled_time <= ?
              AND s.status = 'pending'
            ORDER BY s.scheduled_time ASC
        """, (user_id, today, now)).fetchall()

    return _rows_to_list(rows)


# ---------------------------------------------------------------------------
# AI Chatbot context (READ-ONLY)
# ---------------------------------------------------------------------------

def get_medicine_context(user_id: str) -> str:
    """
    Return a plain-text summary of today's medicine schedule for the AI
    chatbot. This is READ-ONLY — it never prescribes or modifies data.

    Example output:
      "Today's medicines for user:
       - 08:00 — Metformin 500mg (1 tablet, after food) — Status: taken ✓
       - 20:00 — Amlodipine 5mg (1 tablet, before food) — Status: pending
       Today's adherence: 50.0%
       IMPORTANT: Do not prescribe new medicines or change dosages."
    """
    schedule = generate_today_schedule(user_id)
    summary = get_today_summary(user_id)

    if not schedule:
        return (
            "No medicines are scheduled for today. "
            "The user has not added any prescriptions yet. "
            "IMPORTANT: Do NOT prescribe any medicines or suggest dosages. "
            "Only a qualified doctor can prescribe medicines."
        )

    lines = ["Today's medicine schedule:"]
    for item in schedule:
        status_icon = {
            "taken": "✅ taken",
            "missed": "❌ missed",
            "pending": "⏳ pending",
            "snoozed": "🔁 snoozed",
        }.get(item["status"], item["status"])

        lines.append(
            f"  - {item['scheduled_time']} — {item['medicine_name']} "
            f"({item['dosage']}, {item['food_instruction']} food) — {status_icon}"
        )

    lines.append(f"\nToday's adherence: {summary['adherence_percent']}%")
    lines.append(f"Taken: {summary['taken']}, Missed: {summary['missed']}, "
                 f"Pending: {summary['pending']}")
    lines.append(
        "\nIMPORTANT: You must NEVER prescribe new medicines, change dosages, "
        "or recommend stopping any medicine. Only relay the information above "
        "from the user's stored prescription schedule. Always advise consulting "
        "their doctor for any medication changes."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-miss expired schedules
# ---------------------------------------------------------------------------

def auto_miss_expired(user_id: str, grace_minutes: int = 30) -> int:
    """
    Mark overdue pending schedules as 'missed' if they are past the
    grace period. Returns the number of entries marked.
    """
    today = date.today().isoformat()
    now = datetime.now()
    cutoff = (now - timedelta(minutes=grace_minutes)).strftime("%H:%M")

    with _get_connection() as conn:
        # Find overdue pending entries
        rows = conn.execute("""
            SELECT s.id, s.medicine_id, s.scheduled_time, s.user_id,
                   m.name AS medicine_name, m.dosage
            FROM medicine_schedules s
            JOIN medicines m ON s.medicine_id = m.id
            WHERE s.user_id = ?
              AND s.scheduled_date = ?
              AND s.scheduled_time < ?
              AND s.status = 'pending'
        """, (user_id, today, cutoff)).fetchall()

        count = 0
        now_str = now.isoformat(timespec="seconds")

        for row in rows:
            r = dict(row)
            conn.execute(
                "UPDATE medicine_schedules SET status = 'missed' WHERE id = ?",
                (r["id"],),
            )
            conn.execute("""
                INSERT INTO medicine_history
                    (user_id, medicine_id, schedule_id, medicine_name,
                     dosage, scheduled_date, scheduled_time, action,
                     action_time, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'missed', ?, 'Auto-marked as missed')
            """, (
                r["user_id"], r["medicine_id"], r["id"],
                r["medicine_name"], r["dosage"],
                today, r["scheduled_time"], now_str,
            ))
            count += 1

        if count:
            conn.commit()
            logger.info("[Module 5] Auto-missed %d overdue schedules for user=%s",
                        count, user_id)

    return count
