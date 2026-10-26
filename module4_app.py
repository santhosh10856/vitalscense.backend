"""
app.py
======
VitalSense AI - Heart Rate & SpO2 Monitoring Module
Flask REST API for receiving, storing, and querying vitals readings
pushed from Bluetooth-connected smartwatches / fitness bands.

Endpoints
---------
POST   /vitals/sync
GET    /vitals/latest/<user_id>
GET    /vitals/history/<user_id>?start=...&end=...
GET    /vitals/daily-average/<user_id>/<date>
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, request, Response, render_template, send_from_directory
from flask_cors import CORS

from vitals_alerts import check_alerts
from vitals_model import VitalsReading, validate_reading
from vitals_storage import get_daily_average, get_history, get_latest, save_reading

# ---------------------------------------------------------------------------
# Application factory & configuration
# ---------------------------------------------------------------------------

# Point Flask at the shared templates/ and static/ folders in the parent
# directory (d:\sample) so we can serve vitals_monitor.html and
# vitals_bluetooth.js from this server, which also owns /vitals/sync.
_BASE = Path(__file__).resolve().parent.parent   # → d:\sample

app: Flask = Flask(
    __name__,
    template_folder=str(_BASE / "templates"),
    static_folder=str(_BASE / "static"),
)

# Enable Cross-Origin Resource Sharing so browser-based dashboards
# and mobile WebView clients can call the API from different origins.
CORS(app)

# Logging: emit to stdout with a timestamp; level is configurable via
# the LOG_LEVEL environment variable (default: INFO).
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _error_response(message: str, errors: list[str] | None = None,
                    status: int = 400) -> tuple[Response, int]:
    """
    Build a JSON error response with a consistent shape.

    Parameters
    ----------
    message : str
        Short description of the error.
    errors : list[str] | None
        Detailed validation error messages (optional).
    status : int
        HTTP status code (default 400).
    """
    body: dict = {"status": "error", "message": message}
    if errors:
        body["errors"] = errors
    return jsonify(body), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/vitals/sync", methods=["POST"])
def sync_vitals() -> tuple[Response, int]:
    """
    POST /vitals/sync
    -----------------
    Accept a JSON body representing a VitalsReading, validate it,
    persist it, and run alert checks.

    Request body (JSON)
    -------------------
    {
        "user_id":        "u-001",
        "device_id":      "BT:AA:BB:CC:DD:EE",
        "heart_rate_bpm": 110,
        "spo2_percent":   97.5,
        "recorded_at":    "2026-09-01T05:30:00Z"
    }

    Responses
    ---------
    200 OK      - Reading saved; includes alert list and attention flag.
    400 Bad     - Missing / invalid fields; returns validation errors.
    500 Error   - Unexpected server-side failure.
    """
    logger.info("POST /vitals/sync from %s", request.remote_addr)

    try:
        data: dict = request.get_json(force=True, silent=True) or {}

        # Build a VitalsReading from the incoming payload
        reading = VitalsReading(
            user_id=data.get("user_id", ""),
            device_id=data.get("device_id", ""),
            heart_rate_bpm=data.get("heart_rate_bpm"),
            spo2_percent=data.get("spo2_percent"),
            recorded_at=data.get("recorded_at", ""),
        )

        # Validate - return 400 with specific error messages if invalid
        errors = validate_reading(reading)
        if errors:
            logger.warning("Validation failed for sync request: %s", errors)
            return _error_response(
                "Validation failed. Please correct the errors and retry.",
                errors=errors,
                status=400,
            )

        # Persist to SQLite
        reading_dict: dict = asdict(reading)
        row_id: int = save_reading(reading_dict)
        logger.info("Reading saved: id=%d user=%s", row_id, reading.user_id)

        # Run alert checks
        alert_result: dict = check_alerts(reading_dict)

        return jsonify({
            "status": "saved",
            "reading_id": row_id,
            "alerts": alert_result["alerts"],
            "requires_attention": alert_result["requires_attention"],
        }), 200

    except Exception as exc:  # noqa: BLE001
        # Log the full exception but never expose a raw traceback to the client
        logger.exception("Unexpected error in /vitals/sync: %s", exc)
        return _error_response("An internal server error occurred.", status=500)


@app.route("/vitals/latest/<user_id>", methods=["GET"])
def latest_vitals(user_id: str) -> tuple[Response, int]:
    """
    GET /vitals/latest/<user_id>
    ----------------------------
    Retrieve the most recent reading for the specified user.

    Path parameter
    --------------
    user_id : str
        The user whose latest reading is requested.

    Responses
    ---------
    200 OK      - Returns the latest reading dict, or null if none found.
    500 Error   - Unexpected server-side failure.
    """
    logger.info("GET /vitals/latest/%s", user_id)

    try:
        result = get_latest(user_id)
        if result is None:
            return jsonify({"status": "not_found", "data": None}), 200
        return jsonify({"status": "ok", "data": result}), 200

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error fetching latest for user %s: %s", user_id, exc)
        return _error_response("An internal server error occurred.", status=500)


@app.route("/vitals/history/<user_id>", methods=["GET"])
def vitals_history(user_id: str) -> tuple[Response, int]:
    """
    GET /vitals/history/<user_id>?start=...&end=...
    ------------------------------------------------
    Return all readings for a user within an ISO-8601 date/time range.

    Query parameters
    ----------------
    start : str  ISO-8601 timestamp (e.g. "2026-09-01T00:00:00")
    end   : str  ISO-8601 timestamp (e.g. "2026-09-01T23:59:59")

    Responses
    ---------
    200 OK      - Ordered list of reading dicts (oldest first).
    400 Bad     - Missing start or end query parameter.
    500 Error   - Unexpected server-side failure.
    """
    logger.info("GET /vitals/history/%s start=%s end=%s",
                user_id, request.args.get("start"), request.args.get("end"))

    try:
        start: str | None = request.args.get("start")
        end: str | None = request.args.get("end")

        if not start or not end:
            return _error_response(
                "Both 'start' and 'end' query parameters are required "
                "(ISO-8601 timestamp format).",
                status=400,
            )

        readings = get_history(user_id, start, end)
        return jsonify({"status": "ok", "count": len(readings), "data": readings}), 200

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error fetching history for user %s: %s", user_id, exc)
        return _error_response("An internal server error occurred.", status=500)


@app.route("/vitals/daily-average/<user_id>/<date>", methods=["GET"])
def daily_average(user_id: str, date: str) -> tuple[Response, int]:
    """
    GET /vitals/daily-average/<user_id>/<date>
    ------------------------------------------
    Return the average heart rate and SpO2 for a given calendar day.

    Path parameters
    ---------------
    user_id : str  Target user.
    date    : str  Calendar date in YYYY-MM-DD format.

    Responses
    ---------
    200 OK  - Dict with avg_heart_rate_bpm, avg_spo2_percent,
              reading_count, date, user_id.
    500     - Unexpected server-side failure.
    """
    logger.info("GET /vitals/daily-average/%s/%s", user_id, date)

    try:
        result = get_daily_average(user_id, date)
        return jsonify({"status": "ok", "data": result}), 200

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error computing daily average for user %s date %s: %s",
                         user_id, date, exc)
        return _error_response("An internal server error occurred.", status=500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vitals Monitor dashboard page
# ---------------------------------------------------------------------------

@app.route("/vitals-monitor", methods=["GET"])
def vitals_monitor() -> Response:
    """
    GET /vitals-monitor
    -------------------
    Serve the live Bluetooth vitals monitoring dashboard.

    Web Bluetooth requires a Secure Context (HTTPS or localhost).
    This route serves the page over http://localhost:5000 so the browser
    allows navigator.bluetooth and loads vitals_bluetooth.js correctly.
    """
    return render_template("vitals_monitor.html")


if __name__ == "__main__":
    # Bind host / port from environment so Docker / cloud deployments
    # do not require code changes.
    host: str = os.environ.get("FLASK_HOST", "0.0.0.0")
    port: int = int(os.environ.get("FLASK_PORT", "5000"))
    debug: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info("Starting VitalSense AI API on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)
