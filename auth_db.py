"""
auth_db.py — VitalSense AI | Real Multi-User Authentication (SQLite)
=====================================================================
Provides persistent user registration, login, and session management
using SQLite + Werkzeug password hashing + Flask-Login.

Tables
------
  users:
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    name          TEXT NOT NULL
    email         TEXT UNIQUE NOT NULL
    password_hash TEXT NOT NULL
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP

Usage
-----
  from auth_db import init_auth_db, register_user, authenticate_user, get_user_by_id

Author : VitalSense AI Team
"""

from __future__ import annotations

import os
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger("vitalsense.auth_db")

# --------------------------------------------------------------------------- #
# Database path — stored alongside the app                                     #
# --------------------------------------------------------------------------- #
DB_DIR = Path(__file__).parent / "instance"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "vitalsense_users.db"


# --------------------------------------------------------------------------- #
# User model (lightweight dataclass, no ORM dependency)                        #
# --------------------------------------------------------------------------- #
@dataclass
class User:
    id: int
    name: str
    email: str
    password_hash: str = ""
    created_at: str = ""

    # Flask-Login interface methods
    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    def to_dict(self):
        """Return a safe dict (no password_hash) for API responses."""
        return {
            "id": f"u_{self.id:03d}",
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------- #
# Database helpers                                                             #
# --------------------------------------------------------------------------- #
def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")     # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_auth_db() -> None:
    """Create the users table if it doesn't exist."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT    NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info("Auth DB initialised at %s", DB_PATH)
    finally:
        conn.close()


def register_user(name: str, email: str, password: str) -> tuple[Optional[User], Optional[str]]:
    """
    Register a new user.

    Returns
    -------
    (User, None) on success
    (None, error_message) on failure
    """
    email = email.strip().lower()
    name = name.strip()

    if not email or not name or not password:
        return None, "All fields (name, email, password) are required."

    if len(password) < 6:
        return None, "Password must be at least 6 characters."

    pw_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

    conn = _get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, pw_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid

        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        user = User(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=str(row["created_at"]),
        )
        logger.info("New user registered: %s (%s)", name, email)
        return user, None

    except sqlite3.IntegrityError:
        return None, "That email address is already registered."
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> tuple[Optional[User], Optional[str]]:
    """
    Authenticate a user by email + password.

    Returns
    -------
    (User, None) on success
    (None, error_message) on failure
    """
    email = email.strip().lower()
    if not email or not password:
        return None, "Email and password are required."

    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            return None, "Invalid email or password."

        if not check_password_hash(row["password_hash"], password):
            return None, "Invalid email or password."

        user = User(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=str(row["created_at"]),
        )
        return user, None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[User]:
    """Load a user by ID (used by Flask-Login's user_loader)."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return User(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=str(row["created_at"]),
        )
    finally:
        conn.close()


def get_all_users_count() -> int:
    """Return the total number of registered users."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
        return row["cnt"]
    finally:
        conn.close()
