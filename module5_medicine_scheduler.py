"""
medicine_scheduler.py — VitalSense AI | Module 5: Medicine Reminder
====================================================================
Background thread that periodically checks for overdue medicine
schedules and auto-marks them as missed after a grace period.

Also provides a helper to get currently-due reminders for the
browser notification polling endpoint.

Usage:
    from medicine_scheduler import start_scheduler
    start_scheduler()  # call once at server startup

Author: VitalSense AI Team
"""

from __future__ import annotations

import logging
import threading
import time

from . import medicine_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECK_INTERVAL_SECONDS = 60    # How often to check for overdue schedules
GRACE_PERIOD_MINUTES   = 30    # Minutes after scheduled time before auto-miss

# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _scheduler_loop() -> None:
    """
    Main loop for the background scheduler thread.
    Runs every CHECK_INTERVAL_SECONDS, finds all known users,
    and auto-misses their overdue pending schedules.
    """
    logger.info("[Module 5 Scheduler] Background thread started "
                "(interval=%ds, grace=%dm).",
                CHECK_INTERVAL_SECONDS, GRACE_PERIOD_MINUTES)

    while not _stop_event.is_set():
        try:
            # Get all user_ids that have schedules today
            import sqlite3
            conn = sqlite3.connect(medicine_db.DB_PATH)
            conn.row_factory = sqlite3.Row
            from datetime import date
            today = date.today().isoformat()
            rows = conn.execute(
                "SELECT DISTINCT user_id FROM medicine_schedules "
                "WHERE scheduled_date = ?", (today,)
            ).fetchall()
            conn.close()

            for row in rows:
                user_id = row["user_id"]
                count = medicine_db.auto_miss_expired(
                    user_id, grace_minutes=GRACE_PERIOD_MINUTES
                )
                if count > 0:
                    logger.info("[Module 5 Scheduler] Auto-missed %d for user=%s",
                                count, user_id)

        except Exception as exc:
            # Never let the background thread crash the server
            logger.error("[Module 5 Scheduler] Error in scheduler loop: %s", exc)

        # Sleep in small increments so we can respond to stop_event quickly
        for _ in range(CHECK_INTERVAL_SECONDS):
            if _stop_event.is_set():
                break
            time.sleep(1)

    logger.info("[Module 5 Scheduler] Background thread stopped.")


def start_scheduler() -> None:
    """Start the background scheduler thread (idempotent)."""
    global _scheduler_thread

    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        logger.info("[Module 5 Scheduler] Already running, skipping start.")
        return

    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        name="MedicineScheduler",
        daemon=True,  # thread dies when main process exits
    )
    _scheduler_thread.start()
    logger.info("[Module 5 Scheduler] Started background scheduler thread.")


def stop_scheduler() -> None:
    """Signal the background scheduler thread to stop."""
    _stop_event.set()
    if _scheduler_thread is not None:
        _scheduler_thread.join(timeout=5)
    logger.info("[Module 5 Scheduler] Stop signal sent.")
