"""
app.py – VitalSense AI | Module 1: Disease Prediction
======================================================
Flask REST API exposing four endpoints:

  POST /dataset/upload  – validate & persist a CSV dataset
  POST /dataset/train   – train models on the uploaded dataset
  POST /predict         – predict disease from user symptoms
  GET  /dataset/symptoms – list all valid symptom names

Design decisions:
  * CORS enabled so the React/HTML frontend can call this from any origin.
  * 10 MB upload limit enforced via MAX_CONTENT_LENGTH.
  * All endpoints return JSON with a consistent structure.
  * Full try/except on every route; raw tracebacks are NEVER sent to the client.
  * Module-level logging captures every request + outcome.

Author : VitalSense AI Team
"""

import logging
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from data_loader import load_and_validate_dataset
from predict import predict_disease
from train_model import train_and_save_model

# --------------------------------------------------------------------------- #
# Logging – write to stdout so it shows up in the terminal / gunicorn logs     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Flask application                                                             #
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app)  # allow cross-origin requests from any frontend origin

# ---------- Configuration -------------------------------------------------- #
BASE_DIR = Path(__file__).parent          # directory containing app.py
UPLOAD_DIR = BASE_DIR / "uploads"         # where uploaded CSVs are stored
MODEL_DIR = BASE_DIR / "model"            # where trained artefacts are saved
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILENAME = "dataset.csv"          # fixed name for the uploaded file
DATASET_PATH = UPLOAD_DIR / DATASET_FILENAME

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload cap


# =========================================================================== #
# Endpoint helpers                                                              #
# =========================================================================== #

def _success(data: dict, status: int = 200):
    """Wrap a success payload in a consistent JSON envelope."""
    return jsonify({"status": "success", **data}), status


def _error(message: str, status: int = 400):
    """Wrap an error message in a consistent JSON envelope."""
    logger.error("Returning %d – %s", status, message)
    return jsonify({"status": "error", "message": message}), status


# =========================================================================== #
# Routes                                                                        #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# POST /dataset/upload                                                          #
# --------------------------------------------------------------------------- #
@app.route("/dataset/upload", methods=["POST"])
def upload_dataset():
    """
    Accept a multipart CSV upload, validate it, save it, and return metadata.

    Request  : multipart/form-data  with field name "file"
    Response : {status, rows, symptoms, diseases, message}
    """
    logger.info("POST /dataset/upload – incoming request")

    # ---- Validate that a file was actually included in the request ----
    if "file" not in request.files:
        return _error("No file part in the request. Use field name 'file'.")

    file = request.files["file"]
    if file.filename == "":
        return _error("No file selected.")
    if not file.filename.lower().endswith(".csv"):
        return _error("Only CSV files are accepted.")

    try:
        # Save the raw file to disk first so data_loader can parse it
        file.save(str(DATASET_PATH))
        logger.info("File saved to %s", DATASET_PATH)

        # Run validation through data_loader
        X, y, symptom_cols, classes = load_and_validate_dataset(DATASET_PATH)

        logger.info(
            "Upload validated – %d rows | %d symptoms | %d diseases",
            len(X), len(symptom_cols), len(classes),
        )
        return _success({
            "message": "Dataset uploaded and validated successfully.",
            "rows": len(X),
            "symptoms": len(symptom_cols),
            "diseases": len(classes),
        })

    except FileNotFoundError as exc:
        return _error(str(exc), status=404)
    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:
        # Catch-all: log the full traceback server-side, return generic message
        logger.exception("Unexpected error during upload")
        return _error("An unexpected server error occurred during upload.", status=500)


# --------------------------------------------------------------------------- #
# POST /dataset/train                                                           #
# --------------------------------------------------------------------------- #
@app.route("/dataset/train", methods=["POST"])
def train_model():
    """
    Train RandomForest + GaussianNB on the previously uploaded dataset.
    Returns metrics for both models and identifies which one was saved.

    Request  : no body required
    Response : {status, random_forest: {metrics}, naive_bayes: {metrics}, chosen_model}
    """
    logger.info("POST /dataset/train – incoming request")

    if not DATASET_PATH.exists():
        return _error(
            "No dataset found. Please upload a CSV first via POST /dataset/upload.",
            status=400,
        )

    try:
        # Load and process the saved dataset
        X, y, symptom_cols, classes = load_and_validate_dataset(DATASET_PATH)

        # Train both models; winner is saved to MODEL_DIR
        report = train_and_save_model(X, y, symptom_cols, model_dir=MODEL_DIR)

        logger.info("Training complete. Chosen model: %s", report["chosen_model"])
        return _success({
            "message": "Training complete.",
            "random_forest": report["random_forest"],
            "naive_bayes": report["naive_bayes"],
            "chosen_model": report["chosen_model"],
        })

    except ValueError as exc:
        return _error(str(exc), status=400)
    except Exception as exc:
        logger.exception("Unexpected error during training")
        return _error("An unexpected server error occurred during training.", status=500)


# --------------------------------------------------------------------------- #
# POST /predict                                                                 #
# --------------------------------------------------------------------------- #
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict a disease from a list of symptom names.

    Request  : application/json  {"symptoms": ["itching", "skin_rash", ...]}
    Response : {status, disease, confidence, warnings}
    """
    logger.info("POST /predict – incoming request")

    # ---- Parse request body ----
    body = request.get_json(silent=True)
    if body is None:
        return _error("Request body must be JSON with Content-Type: application/json.")

    user_symptoms = body.get("symptoms")
    if not isinstance(user_symptoms, list) or not user_symptoms:
        return _error(
            "Field 'symptoms' must be a non-empty JSON array of strings. "
            "Example: {\"symptoms\": [\"itching\", \"skin_rash\"]}"
        )

    # Check that the model has been trained
    if not (MODEL_DIR / "model.pkl").exists():
        return _error(
            "No trained model found. Please train the model first via POST /dataset/train.",
            status=400,
        )

    try:
        result = predict_disease(user_symptoms, model_dir=MODEL_DIR)

        logger.info(
            "Prediction – disease='%s' confidence=%.4f warnings=%s",
            result["disease"], result["confidence"], result["warnings"],
        )
        return _success({
            "disease": result["disease"],
            "confidence": result["confidence"],
            "warnings": result["warnings"],
        })

    except ValueError as exc:
        # Zero recognised symptoms – client error
        return _error(str(exc), status=400)
    except FileNotFoundError as exc:
        return _error(str(exc), status=404)
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        return _error("An unexpected server error occurred during prediction.", status=500)


# --------------------------------------------------------------------------- #
# GET /vitals-monitor  (VitalSense AI Bluetooth dashboard)                      #
# --------------------------------------------------------------------------- #
@app.route("/vitals-monitor", methods=["GET"])
def vitals_monitor():
    """
    Serve the live Bluetooth vitals monitoring dashboard.

    IMPORTANT: Web Bluetooth API requires a Secure Context.
    Opening the HTML file directly via file:// blocks navigator.bluetooth and
    prevents the /static/vitals_bluetooth.js script from loading correctly.
    Serving through Flask on http://localhost satisfies the browser's
    Secure Context requirement and resolves the static file path correctly.

    Navigate to:  http://localhost:5000/vitals-monitor
    """
    return render_template("vitals_monitor.html")


# --------------------------------------------------------------------------- #
# GET /dataset/symptoms                                                         #
# --------------------------------------------------------------------------- #
@app.route("/dataset/symptoms", methods=["GET"])
def list_symptoms():
    """
    Return the complete list of valid symptom names the model was trained on.
    The frontend can use this to build a symptom checklist UI.

    Response : {status, count, symptoms: [...]}
    """
    logger.info("GET /dataset/symptoms – incoming request")

    symptom_path = MODEL_DIR / "symptom_list.pkl"
    if not symptom_path.exists():
        return _error(
            "Symptom list not available. Please train the model first via POST /dataset/train.",
            status=404,
        )

    try:
        symptom_list = joblib.load(symptom_path)
        return _success({
            "count": len(symptom_list),
            "symptoms": sorted(symptom_list),   # alphabetical for the UI
        })

    except Exception as exc:
        logger.exception("Failed to load symptom list")
        return _error("Could not load the symptom list.", status=500)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info("Starting VitalSense AI – Disease Prediction API on port 5000")
    # debug=False in production; set to True only during development
    app.run(host="0.0.0.0", port=5000, debug=True)
