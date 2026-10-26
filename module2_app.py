"""
app.py  —  VitalSense AI  |  Module 2: Medicine Recommendation, Doctor Referral & Diet Plan
============================================================================================
Flask backend exposing FOUR endpoints:

  POST /module2/upload-datasets  — Upload 1-4 CSV reference files.
  POST /module2/recommend        — CSV-first recommendation; Gemini fallback for unknown diseases.
  POST /module2/ai-recommend     — Always calls Gemini AI (no datasets needed, covers any disease).
  GET  /module2/diseases         — List all disease names in the loaded datasets.
  GET  /module2/ai-status        — Check if Gemini API key is configured.

Gemini API key
--------------
Set the GEMINI_API_KEY environment variable, OR send the key in the
Authorization header of each request:  Authorization: Bearer AIza...

Designed to work standalone or to accept the disease + symptoms output from
Module 1's /predict endpoint directly.

Configuration
-------------
- MAX_CONTENT_LENGTH : 10 MB (hard limit on upload body size).
- UPLOAD_FOLDER      : ./data/ relative to this file.
- CORS               : Enabled for all origins (tighten for production).

Logging
-------
Every request logs: endpoint, method, disease queried (if applicable),
severity computed, and wall-clock processing time.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ── Make sure sibling modules are importable when running as `python app.py`
sys.path.insert(0, str(Path(__file__).parent))

import dataset_manager
import recommend as recommender
import ai_fallback
from severity_calculator import calculate_severity

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("module2.app")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow all cross-origin requests

# 10 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Folder where uploaded CSVs are saved (relative to this file)
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "data"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Fixed filenames for each dataset role (prevents arbitrary filename injection).
# These names match the files the user uploaded.
DATASET_FILENAMES: dict[str, str] = {
    "precaution_file": "precautions_df.csv",
    "severity_file":   "Symptom-severity.csv",
    "medicine_file":   "medications.csv",
    "diet_file":       "diets.csv",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _autoload_existing_datasets() -> None:
    """Auto-load datasets from UPLOAD_FOLDER if they exist on startup."""
    for field_name, filename in DATASET_FILENAMES.items():
        csv_path = UPLOAD_FOLDER / filename
        if not csv_path.exists():
            continue
        try:
            if field_name == "precaution_file":
                dataset_manager.load_precautions(csv_path)
            elif field_name == "severity_file":
                dataset_manager.load_severity_weights(csv_path)
            elif field_name == "medicine_file":
                dataset_manager.load_medicine_data(csv_path)
            elif field_name == "diet_file":
                dataset_manager.load_diet_data(csv_path)
            logger.info("Auto-loaded dataset '%s'.", filename)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not auto-load '%s': %s", filename, exc)


# Auto-load existing datasets if available in UPLOAD_FOLDER
_autoload_existing_datasets()


def _json_error(message: str, status: int = 400) -> tuple[Response, int]:
    """Return a JSON error response with a given HTTP status code."""
    return jsonify({"error": message}), status


def _count_rows(csv_path: Path) -> int:
    """Return the number of data rows in a CSV (excluding the header)."""
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:  # noqa: BLE001
        return -1


# ---------------------------------------------------------------------------
# Endpoint 1 — Upload datasets
# ---------------------------------------------------------------------------
@app.route("/module2/upload-datasets", methods=["POST"])
def upload_datasets() -> tuple[Response, int]:
    """
    POST /module2/upload-datasets
    Accept up to 4 CSV files in one multipart/form-data request.

    Form field names   Saved as (in ./data/)
    ─────────────────  ──────────────────────────────
    precaution_file  → symptom_precaution.csv
    severity_file    → symptom_severity.csv
    medicine_file    → medicine_recommendation.csv
    diet_file        → diet_recommendation.csv

    Any subset of the four fields is accepted.
    Returns a per-file summary: rows loaded, success/error status.
    """
    t_start = time.perf_counter()
    logger.info("POST /module2/upload-datasets — request received.")

    results: dict[str, dict] = {}
    upload_count = 0

    for field_name, filename in DATASET_FILENAMES.items():
        if field_name not in request.files:
            continue   # field was not part of this upload — skip silently

        file_obj = request.files[field_name]
        if file_obj.filename == "":
            results[field_name] = {"status": "skipped", "reason": "empty filename"}
            continue

        save_path = UPLOAD_FOLDER / filename
        try:
            file_obj.save(str(save_path))
        except OSError as exc:
            results[field_name] = {
                "status": "error",
                "reason": f"Could not save file: {exc}",
            }
            continue

        # ── Validate and load into cache ─────────────────────────────────────
        row_count = _count_rows(save_path)
        try:
            if field_name == "precaution_file":
                # precautions_df.csv: index col + Disease + Precaution_1..4
                dataset_manager.load_precautions(save_path)
            elif field_name == "severity_file":
                # Symptom-severity.csv: Symptom, weight
                dataset_manager.load_severity_weights(save_path)
            elif field_name == "medicine_file":
                # medications.csv: Disease, Medication (Python list string)
                dataset_manager.load_medicine_data(save_path)
            elif field_name == "diet_file":
                # diets.csv: Disease, Diet (Python list string)
                dataset_manager.load_diet_data(save_path)

            results[field_name] = {
                "status": "ok",
                "saved_as": filename,
                "row_count": row_count,
            }
            upload_count += 1
            logger.info("Uploaded & validated '%s' (%d rows).", filename, row_count)

        except (ValueError, FileNotFoundError) as exc:
            results[field_name] = {
                "status": "validation_error",
                "saved_as": filename,
                "row_count": row_count,
                "reason": str(exc),
            }
            logger.error("Validation failed for '%s': %s", filename, exc)

    elapsed = round((time.perf_counter() - t_start) * 1000, 2)

    if upload_count == 0 and not results:
        return _json_error(
            "No recognised file fields found. "
            f"Expected one or more of: {list(DATASET_FILENAMES.keys())}",
            400,
        )

    response_body = {
        "message": f"{upload_count} dataset(s) uploaded and validated.",
        "processing_time_ms": elapsed,
        "files": results,
    }
    logger.info("Upload complete in %.2f ms — %d file(s) processed.", elapsed, upload_count)
    return jsonify(response_body), 200


# ---------------------------------------------------------------------------
# Endpoint 2 — Generate recommendation
# ---------------------------------------------------------------------------
@app.route("/module2/recommend", methods=["POST"])
def recommend() -> tuple[Response, int]:
    """
    POST /module2/recommend
    Body (JSON)::

        {
          "disease":  "Common Cold",            <- required; output of Module 1 /predict
          "symptoms": ["cough", "runny_nose"]   <- optional; symptom names as in Symptom-severity.csv
        }

    Response fields (new dataset schema):
      - medications         : list[str] | null  (populated only when severity == mild)
      - doctor_referral     : {specialist, urgency} | null  (moderate / severe)
      - diet_recommendation : list[str] | null  (flat list from diets.csv)
      - precautions         : list[str] | null  (from precautions_df.csv)
    """
    t_start = time.perf_counter()
    logger.info("POST /module2/recommend — request received.")

    # ── Guard: at least one dataset must be loaded ────────────────────────────
    if not dataset_manager.is_any_dataset_loaded():
        return _json_error(
            "No datasets have been uploaded yet. "
            "Please POST to /module2/upload-datasets first.",
            400,
        )

    # ── Parse request body ────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body:
        return _json_error("Request body must be valid JSON.", 400)

    disease = body.get("disease", "").strip()
    if not disease:
        return _json_error("'disease' field is required and must not be empty.", 400)

    symptoms: list[str] = body.get("symptoms", [])
    if not isinstance(symptoms, list):
        return _json_error("'symptoms' must be a JSON array of strings.", 400)

    # ── Severity calculation ──────────────────────────────────────────────────
    severity_weights = dataset_manager.get_severity_weights()
    if severity_weights:
        severity_result = calculate_severity(symptoms, severity_weights)
    else:
        # Severity dataset not uploaded — default to "mild" and warn
        logger.warning(
            "Severity weights dataset not loaded; defaulting severity to 'mild'."
        )
        severity_result = {
            "severity": "mild",
            "score": 0.0,
            "skipped_symptoms": symptoms,
        }

    severity_label = severity_result["severity"]
    logger.info(
        "Disease='%s', symptoms=%s → severity='%s' (score=%.2f, skipped=%s)",
        disease,
        symptoms,
        severity_label,
        severity_result["score"],
        severity_result["skipped_symptoms"],
    )

    # ── Gemini API key (from header or environment variable) ──────────────────
    auth_header = request.headers.get("Authorization", "")
    gemini_key  = auth_header.replace("Bearer ", "").strip() or None

    # ── Generate full recommendation (CSV first, Gemini fallback) ─────────────
    try:
        rec = recommender.generate_recommendation(
            disease=disease,
            severity=severity_label,
            symptoms=symptoms,
            gemini_api_key=gemini_key,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in generate_recommendation.")
        return _json_error(f"Internal error while generating recommendation: {exc}", 500)

    # Enrich response with severity metadata from the calculator
    rec["severity_score"]       = severity_result["score"]
    rec["skipped_symptoms"]     = severity_result["skipped_symptoms"]

    elapsed = round((time.perf_counter() - t_start) * 1000, 2)
    rec["processing_time_ms"]   = elapsed

    logger.info(
        "Recommendation complete in %.2f ms — disease='%s', severity='%s'.",
        elapsed, disease, severity_label,
    )
    return jsonify(rec), 200


# ---------------------------------------------------------------------------
# Endpoint 3 — List covered diseases
# ---------------------------------------------------------------------------
@app.route("/module2/diseases", methods=["GET"])
def list_diseases() -> tuple[Response, int]:
    """
    GET /module2/diseases
    Returns the union of all disease names across every loaded dataset.
    Useful for frontend auto-complete and testing.
    """
    t_start = time.perf_counter()
    logger.info("GET /module2/diseases — request received.")

    if not dataset_manager.is_any_dataset_loaded():
        return _json_error(
            "No datasets loaded yet. POST to /module2/upload-datasets first.",
            400,
        )

    diseases = dataset_manager.get_all_diseases()
    elapsed  = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info("Returning %d disease names in %.2f ms.", len(diseases), elapsed)
    return jsonify({
        "count": len(diseases),
        "diseases": diseases,
        "processing_time_ms": elapsed,
    }), 200


# ---------------------------------------------------------------------------
# Endpoint 4 — Pure AI recommendation (Gemini, no dataset required)
# ---------------------------------------------------------------------------
@app.route("/module2/ai-recommend", methods=["POST"])
def ai_recommend() -> tuple[Response, int]:
    """
    POST /module2/ai-recommend
    Always calls Gemini AI regardless of whether the disease is in the
    local datasets.  Works for ANY disease in the world.

    Body (JSON)::

        {
          "disease":  "Lyme Disease",
          "symptoms": ["fatigue", "joint_pain"],
          "severity": "moderate"         <- optional; if omitted, auto-calculated
        }

    Pass your Gemini API key via:
      - Authorization header:  Authorization: Bearer AIza...
      - OR GEMINI_API_KEY environment variable
    """
    t_start = time.perf_counter()
    logger.info("POST /module2/ai-recommend — request received.")

    body = request.get_json(silent=True)
    if not body:
        return _json_error("Request body must be valid JSON.", 400)

    disease = body.get("disease", "").strip()
    if not disease:
        return _json_error("'disease' field is required.", 400)

    symptoms: list[str] = body.get("symptoms", [])
    if not isinstance(symptoms, list):
        return _json_error("'symptoms' must be a JSON array of strings.", 400)

    # Severity: use provided value or auto-calculate from symptoms
    forced_severity = body.get("severity", "").strip().lower()
    if forced_severity in ("mild", "moderate", "severe"):
        severity_label  = forced_severity
        severity_score  = 0.0
        skipped_syms: list[str] = []
    else:
        w = dataset_manager.get_severity_weights()
        if w:
            sev = calculate_severity(symptoms, w)
            severity_label  = sev["severity"]
            severity_score  = sev["score"]
            skipped_syms    = sev["skipped_symptoms"]
        else:
            severity_label, severity_score, skipped_syms = "moderate", 0.0, []

    # Gemini API key
    auth_header = request.headers.get("Authorization", "")
    gemini_key  = auth_header.replace("Bearer ", "").strip() or None

    try:
        rec = ai_fallback.get_ai_recommendation(
            disease=disease,
            severity=severity_label,
            api_key=gemini_key,
        )
    except EnvironmentError:
        return _json_error(
            "Gemini API key not configured. Set GEMINI_API_KEY environment variable "
            "or send it in the Authorization: Bearer <key> header.",
            400,
        )
    except RuntimeError as exc:
        logger.exception("Gemini AI call failed.")
        return _json_error(f"Gemini AI error: {exc}", 500)

    rec["severity_score"]     = severity_score
    rec["skipped_symptoms"]   = skipped_syms
    elapsed = round((time.perf_counter() - t_start) * 1000, 2)
    rec["processing_time_ms"] = elapsed

    logger.info(
        "AI recommendation complete in %.2f ms — disease='%s', severity='%s'.",
        elapsed, disease, severity_label,
    )
    return jsonify(rec), 200


# ---------------------------------------------------------------------------
# Endpoint 5 — AI status check
# ---------------------------------------------------------------------------
@app.route("/module2/ai-status", methods=["GET"])
def ai_status() -> tuple[Response, int]:
    """
    GET /module2/ai-status
    Returns whether Gemini AI fallback is available (API key set).
    """
    available = ai_fallback.is_gemini_available()
    return jsonify({
        "gemini_available": available,
        "model":            ai_fallback.GEMINI_MODEL,
        "note": (
            "Gemini AI fallback is active for any disease not in local datasets."
            if available else
            "Set GEMINI_API_KEY env variable to enable AI fallback for unknown diseases."
        ),
    }), 200


# ---------------------------------------------------------------------------
# Global error handlers (prevent raw stack traces reaching the client)
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(exc) -> tuple[Response, int]:
    return _json_error(str(exc), 400)


@app.errorhandler(404)
def not_found(exc) -> tuple[Response, int]:
    return _json_error("Endpoint not found.", 404)


@app.errorhandler(413)
def too_large(exc) -> tuple[Response, int]:
    return _json_error("Upload too large. Maximum allowed size is 10 MB.", 413)


@app.errorhandler(500)
def server_error(exc) -> tuple[Response, int]:
    logger.exception("Unhandled 500 error.")
    return _json_error("An unexpected internal error occurred.", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("MODULE2_PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting VitalSense AI — Module 2 on port %d (debug=%s).", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
