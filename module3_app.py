"""
app.py — VitalSense AI · Module 3  (Medical Report Analysis + Image Analysis)
==============================================================================
Flask REST API exposing seven endpoints:

  Lab Report Endpoints:
    POST /module3/upload-reference      — Upload & validate reference CSV
    POST /module3/analyze               — Analyse a set of lab values
    GET  /module3/parameters            — List all known parameters + units

  Image Analysis Endpoints:
    POST /module3/image/upload-model-dataset — Upload ZIP of image dataset
    POST /module3/image/train                — Train DenseNet121 on dataset
    POST /module3/image/predict              — Run image screening inference

  Radiology Report Text Analysis:
    POST /module3/report-text/summarize      — LLM plain-language summary

  Combined Full Analysis (NEW):
    POST /module3/analyze-full               — Lab + Image + Report Text in one call

Design choices (for viva):
  * All image routes share the /module3/image/ prefix for clean separation.
  * Training is synchronous in this implementation; for production, move to
    a Celery/RQ background job and poll a /status endpoint.
  * Image size limit: 15 MB (enforced via Flask MAX_CONTENT_LENGTH).
  * Model artifacts saved to ./model_store/ (no hardcoded absolute paths).
  * All error responses are JSON {"error": "..."} — no stack traces leaked.
  * /analyze-full runs each sub-analysis independently; one failure does NOT
    block the others.  Each failed sub-section carries an "error" key instead
    of full results, and combined_overall_status reflects only the analyses
    that succeeded.
"""

import io
import json
import logging
import os
import sys
import traceback
import zipfile
from pathlib import Path

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image as PilImage

# Ensure module3's own directory is on sys.path so sibling imports work
# even when app.py is launched from a different working directory.
_MODULE3_DIR = Path(__file__).resolve().parent
if str(_MODULE3_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE3_DIR))

import reference_data as rd
from report_analyzer import analyze_report

# ---------------------------------------------------------------------------
# Logging — structured, with timestamps
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("module3.app")

# ---------------------------------------------------------------------------
# Flask app & CORS
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/module3/*": {"origins": "*"}})

# ── File size limit: 15 MB for image uploads ──────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB

# ---------------------------------------------------------------------------
# Directory constants — created at startup, no hardcoded absolute paths
# ---------------------------------------------------------------------------
UPLOAD_DIR: Path = _MODULE3_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DATASET_DIR: Path = _MODULE3_DIR / "data" / "image_dataset"
IMAGE_DATASET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_STORE_DIR: Path = _MODULE3_DIR / "model_store"
MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CSV_EXT = ".csv"
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
ALLOWED_DICOM_EXT = ".dcm"
MAX_IMAGE_DIMENSION = 10_000   # pixels; reject pathologically large images

# ---------------------------------------------------------------------------
# Minimum image dimension guard
# ---------------------------------------------------------------------------
MIN_IMAGE_DIMENSION = 32  # pixels


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _error(message: str, status_code: int = 400) -> tuple[Response, int]:
    """Return a JSON error response without leaking any stack trace."""
    logger.error("HTTP %d — %s", status_code, message)
    return jsonify({"error": message}), status_code


def _allowed_file(filename: str) -> bool:
    """Return True if filename ends with .csv (case-insensitive)."""
    return Path(filename).suffix.lower() == ALLOWED_CSV_EXT


def _allowed_image(filename: str) -> bool:
    """Return True if filename is a supported image extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTS or ext == ALLOWED_DICOM_EXT


# ===========================================================================
# LAB REPORT ENDPOINTS (unchanged from Module 3a)
# ===========================================================================

# ---------------------------------------------------------------------------
# Endpoint 1 — Upload reference CSV
# ---------------------------------------------------------------------------

@app.route("/module3/upload-reference", methods=["POST"])
def upload_reference() -> tuple[Response, int]:
    """
    Upload and validate the reference-range CSV.

    Expected request
    ----------------
    Content-Type: multipart/form-data
    Field name  : "file"  (the CSV file)

    Returns 200
    -----------
    {
        "message"        : "Reference dataset loaded successfully.",
        "parameter_count": <int>,
        "parameters"     : [<str>, ...]
    }

    Returns 400 if no file supplied, wrong extension, or CSV fails validation.
    Returns 500 on unexpected server errors.
    """
    if "file" not in request.files:
        return _error("No file part in the request. Send a multipart/form-data request with field 'file'.")

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return _error("No file selected. Please attach a CSV file.")

    if not _allowed_file(file.filename):
        return _error(
            f"Invalid file type '{Path(file.filename).suffix}'. "
            "Only .csv files are accepted."
        )

    save_path: Path = UPLOAD_DIR / "reference_ranges.csv"
    try:
        file.save(str(save_path))
        logger.info("Uploaded CSV saved to: %s", save_path)
    except OSError as exc:
        return _error(f"Could not save uploaded file: {exc}", 500)

    try:
        ref_dict = rd.load_reference_ranges(save_path)
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc), 400)
    except Exception:
        logger.error("Unexpected error loading reference CSV:\n%s", traceback.format_exc())
        return _error("An unexpected error occurred while processing the CSV.", 500)

    meta = rd.get_cached_meta() or []
    return jsonify({
        "message": "Reference dataset loaded successfully.",
        "parameter_count": len(ref_dict),
        "parameters": [p["name"] for p in meta],
    }), 200


# ---------------------------------------------------------------------------
# Endpoint 2 — Analyse lab values
# ---------------------------------------------------------------------------

@app.route("/module3/analyze", methods=["POST"])
def analyze() -> tuple[Response, int]:
    """Analyse a patient's lab values against the loaded reference dataset."""
    if not rd.is_loaded():
        return _error(
            "No reference dataset is loaded. "
            "Please upload a CSV via POST /module3/upload-reference first.", 400
        )
    if not request.is_json:
        return _error("Request Content-Type must be 'application/json'.", 400)

    body = request.get_json(silent=True)
    if body is None:
        return _error("Request body is not valid JSON.", 400)

    lab_values = body.get("lab_values")
    if lab_values is None:
        return _error("Missing required field 'lab_values' in request body.", 400)
    if not isinstance(lab_values, dict):
        return _error(f"'lab_values' must be a JSON object, got {type(lab_values).__name__}.", 400)
    if not lab_values:
        return _error("'lab_values' is empty — provide at least one parameter.", 400)

    gender: str = str(body.get("gender", "Any")).strip()
    logger.info("Analyze request: %d parameters, gender='%s'", len(lab_values), gender)

    try:
        result = analyze_report(lab_values=lab_values, gender=gender)
    except (RuntimeError, ValueError) as exc:
        return _error(str(exc), 400)
    except Exception:
        logger.error("Unexpected error during analysis:\n%s", traceback.format_exc())
        return _error("An unexpected error occurred during analysis.", 500)

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# Endpoint 3 — List all parameters
# ---------------------------------------------------------------------------

@app.route("/module3/parameters", methods=["GET"])
def list_parameters() -> tuple[Response, int]:
    """Return all parameters + units from the cached reference dataset."""
    if not rd.is_loaded():
        return _error(
            "No reference dataset is loaded. "
            "Please upload a CSV via POST /module3/upload-reference first.", 400
        )
    meta = rd.get_cached_meta() or []
    return jsonify({"parameter_count": len(meta), "parameters": meta}), 200


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/module3/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """Health-check endpoint. Returns dataset and model status."""
    model_ready = (MODEL_STORE_DIR / "image_model.keras").exists() or \
                  (MODEL_STORE_DIR / "image_model.h5").exists()
    anthropic_key_set = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return jsonify({
        "status": "ok",
        "module": "module3 — Medical Report Analysis + Image Analysis + Report Text",
        "lab_dataset_loaded"       : rd.is_loaded(),
        "image_model_ready"        : model_ready,
        "report_text_analyzer_ready": anthropic_key_set,
    }), 200


# ===========================================================================
# IMAGE ANALYSIS ENDPOINTS (NEW — Module 3 extension)
# ===========================================================================

# ---------------------------------------------------------------------------
# Image Endpoint 1 — Upload image dataset ZIP
# ---------------------------------------------------------------------------

@app.route("/module3/image/upload-model-dataset", methods=["POST"])
def upload_image_dataset() -> tuple[Response, int]:
    """
    Accept a ZIP file containing the class-folder image dataset and extract it.

    Expected ZIP internal structure:
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg   ← optional

    The ZIP is extracted to: ./data/image_dataset/

    Returns 200
    -----------
    {
        "message"      : "Dataset extracted successfully.",
        "dataset_path" : str,
        "splits_found" : ["train", "val", "test"],
        "classes"      : {
            "train": ["NORMAL", "PNEUMONIA"],
            "val"  : ["NORMAL", "PNEUMONIA"]
        },
        "image_counts" : {
            "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
            "val"  : {"NORMAL":   8,  "PNEUMONIA":   8}
        }
    }

    Returns 400 if no file, wrong type, or ZIP has invalid structure.
    Returns 500 on extraction errors.
    """
    if "file" not in request.files:
        return _error("No file in request. Send field 'file' with a .zip dataset.")

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return _error("No file selected.")

    if Path(file.filename).suffix.lower() != ".zip":
        return _error(
            f"Expected a .zip file, got '{Path(file.filename).suffix}'. "
            "Compress your dataset folder as a .zip and upload again."
        )

    # ── Read ZIP into memory and validate ─────────────────────────────────
    try:
        zip_bytes = io.BytesIO(file.read())
        with zipfile.ZipFile(zip_bytes) as zf:
            # Check for path traversal attacks (zip-slip)
            for member in zf.namelist():
                if ".." in member or member.startswith("/"):
                    return _error(
                        f"Invalid path in ZIP: '{member}'. "
                        "ZIP may contain path traversal entries."
                    )

            # Extract to image dataset directory (overwrites existing)
            zf.extractall(str(IMAGE_DATASET_DIR))
            logger.info("ZIP extracted to: %s", IMAGE_DATASET_DIR)
    except zipfile.BadZipFile:
        return _error("Uploaded file is not a valid ZIP archive.")
    except Exception as exc:
        logger.error("ZIP extraction failed:\n%s", traceback.format_exc())
        return _error(f"Failed to extract ZIP: {exc}", 500)

    # ── Scan extracted structure ───────────────────────────────────────────
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    splits_found = []
    classes_found = {}
    image_counts = {}

    for split_name in ["train", "val", "test"]:
        split_path = IMAGE_DATASET_DIR / split_name
        if not split_path.exists():
            # Also check one level deeper (ZIP may contain a root folder)
            for child in IMAGE_DATASET_DIR.iterdir():
                if child.is_dir():
                    candidate = child / split_name
                    if candidate.exists():
                        split_path = candidate
                        break

        if split_path.exists():
            splits_found.append(split_name)
            class_dirs = sorted(d.name for d in split_path.iterdir() if d.is_dir())
            classes_found[split_name] = class_dirs

            counts: dict[str, int] = {}
            for cls in class_dirs:
                cls_dir = split_path / cls
                n = sum(
                    1 for f in cls_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                )
                counts[cls] = n
            image_counts[split_name] = counts

    if not splits_found:
        return _error(
            "No valid train/val/test folders found in the uploaded ZIP. "
            "Ensure the ZIP contains folders named 'train/' and 'val/' "
            "with class sub-folders."
        )

    if "train" not in splits_found:
        return _error("Missing 'train/' split in the uploaded ZIP.")

    logger.info("Dataset ready: splits=%s, classes=%s", splits_found, classes_found)
    return jsonify({
        "message": "Dataset extracted successfully.",
        "dataset_path": str(IMAGE_DATASET_DIR),
        "splits_found": splits_found,
        "classes": classes_found,
        "image_counts": image_counts,
    }), 200


# ---------------------------------------------------------------------------
# Image Endpoint 2 — Train the model
# ---------------------------------------------------------------------------

@app.route("/module3/image/train", methods=["POST"])
def train_image_model_endpoint() -> tuple[Response, int]:
    """
    Trigger the full training pipeline on the uploaded image dataset.

    Optional JSON body:
    {
        "epochs"        : 10,      // Phase 1 max epochs (default: 10)
        "fine_tune"     : true,    // Run Phase 2 fine-tuning (default: true)
        "fine_tune_epochs": 10,    // Phase 2 max epochs (default: 10)
        "unfreeze_n"    : 30,      // Layers to unfreeze in Phase 2 (default: 30)
        "batch_size"    : 32,      // Batch size (default: 32)
        "img_size"      : [224, 224]  // Input size (default: [224, 224])
    }

    Returns 200
    -----------
    {
        "message"         : "Training complete.",
        "class_names"     : ["NORMAL", "PNEUMONIA"],
        "phase1_history"  : { "accuracy": [...], "val_accuracy": [...], ... },
        "phase2_history"  : { ... } or null if fine_tune=false,
        "test_metrics"    : { "accuracy": 0.92, "f1": 0.91, ... } or null,
        "model_path"      : "./model_store/image_model.keras",
        "roc_curve_path"  : "./model_store/roc_curve.png" or null
    }

    Returns 400 if dataset not found or configuration is invalid.
    Returns 500 on training errors.
    """
    # ── Lazy-import heavy ML modules so server starts fast ─────────────────
    # TensorFlow import is deferred to avoid slow startup when only
    # lab-report endpoints are being used.
    try:
        from image_preprocessing import build_data_generators
        from train_image_model import (
            build_model,
            train_model,
            fine_tune_model,
            evaluate_model,
            save_model_artifacts,
        )
    except ImportError as exc:
        return _error(
            f"ML dependencies not available: {exc}. "
            "Run: pip install tensorflow scikit-learn matplotlib", 500
        )

    # ── Check dataset exists ──────────────────────────────────────────────
    train_dir = IMAGE_DATASET_DIR / "train"
    if not train_dir.exists():
        # Check one level deeper (ZIP may have contained a root folder)
        data_root = None
        try:
            for child in IMAGE_DATASET_DIR.iterdir():
                if child.is_dir() and (child / "train").exists():
                    data_root = child
                    break
        except OSError:
            pass
        if data_root is None:
            return _error(
                "No dataset found. Upload an image dataset ZIP first via "
                "POST /module3/image/upload-model-dataset."
            )
    else:
        data_root = IMAGE_DATASET_DIR

    # ── Parse optional training parameters ────────────────────────────────
    body = request.get_json(silent=True) or {}
    epochs = int(body.get("epochs", 10))
    fine_tune = bool(body.get("fine_tune", True))
    fine_tune_epochs = int(body.get("fine_tune_epochs", 10))
    unfreeze_n = int(body.get("unfreeze_n", 30))
    batch_size = int(body.get("batch_size", 32))
    raw_size = body.get("img_size", [224, 224])
    img_size = tuple(raw_size) if isinstance(raw_size, list) else (224, 224)

    logger.info(
        "Training request: epochs=%d, fine_tune=%s, batch_size=%d, img_size=%s",
        epochs, fine_tune, batch_size, img_size,
    )

    # ── Build data generators ──────────────────────────────────────────────
    try:
        train_gen, val_gen, test_gen, class_names = build_data_generators(
            data_dir=str(data_root),
            img_size=img_size,
            batch_size=batch_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        return _error(str(exc), 400)
    except Exception:
        logger.error("Data generator error:\n%s", traceback.format_exc())
        return _error("Failed to build data generators.", 500)

    num_classes = len(class_names)

    # ── Phase 1: Build and train model (frozen base) ───────────────────────
    try:
        model = build_model(num_classes=num_classes, img_size=img_size)
        phase1_history = train_model(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            epochs=epochs,
            model_dir=str(MODEL_STORE_DIR),
        )
    except Exception:
        logger.error("Phase 1 training error:\n%s", traceback.format_exc())
        return _error("Model training failed. Check server logs.", 500)

    # ── Phase 2: Fine-tuning (optional) ────────────────────────────────────
    phase2_history = None
    if fine_tune:
        try:
            phase2_history = fine_tune_model(
                model=model,
                train_gen=train_gen,
                val_gen=val_gen,
                unfreeze_n=unfreeze_n,
                epochs=fine_tune_epochs,
                model_dir=str(MODEL_STORE_DIR),
            )
        except Exception:
            logger.error("Phase 2 fine-tune error:\n%s", traceback.format_exc())
            logger.warning("Fine-tuning failed — using Phase 1 model only.")

    # ── Evaluate on test set (if available) ────────────────────────────────
    test_metrics = None
    if test_gen is not None:
        try:
            test_metrics = evaluate_model(
                model=model,
                test_gen=test_gen,
                model_dir=str(MODEL_STORE_DIR),
                class_names=class_names,
            )
        except Exception:
            logger.error("Evaluation error:\n%s", traceback.format_exc())
            logger.warning("Evaluation failed — model still saved.")

    # ── Save final model artifacts ─────────────────────────────────────────
    try:
        model_path, labels_path = save_model_artifacts(
            model=model,
            class_names=class_names,
            model_dir=str(MODEL_STORE_DIR),
        )
    except Exception:
        logger.error("Model save error:\n%s", traceback.format_exc())
        return _error("Training succeeded but failed to save model artifacts.", 500)

    logger.info("Training pipeline complete. Model: %s", model_path)
    return jsonify({
        "message": "Training complete.",
        "class_names": class_names,
        "num_classes": num_classes,
        "phase1_history": phase1_history,
        "phase2_history": phase2_history,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "labels_path": labels_path,
    }), 200


# ---------------------------------------------------------------------------
# Image Endpoint 3 — Predict / screen an uploaded image
# ---------------------------------------------------------------------------

@app.route("/module3/image/predict", methods=["POST"])
def predict_image_endpoint() -> tuple[Response, int]:
    """
    Accept a medical image file upload, run the trained model, and return
    a screening result with Grad-CAM heatmap path and assistive disclaimer.

    Expected request: multipart/form-data
        Field "file"            : image file (.jpg / .png / .bmp / .dcm)
        Field "conf_threshold"  : float 0-1, optional (default 0.60)

    Returns 200
    -----------
    {
        "predicted_class"       : "PNEUMONIA",
        "confidence"            : 0.9231,
        "all_probabilities"     : {"NORMAL": 0.0769, "PNEUMONIA": 0.9231},
        "low_confidence_warning": false,
        "flagged_region"        : "./model_store/heatmaps/gradcam_abc123.png",
        "assistive_disclaimer"  : "This is an AI-generated preliminary screening..."
    }

    Returns 400 if:
        - No file attached
        - Unsupported file type (not .jpg/.png/.bmp/.dcm)
        - File exceeds 15 MB (caught by Flask MAX_CONTENT_LENGTH)
        - Image dimensions are unreasonably large (>10000px on any side)
        - Image dimensions are unreasonably small (<32px)
        - No trained model found (must run /train first)
    Returns 500 on unexpected inference errors.
    """
    # ── Validate file presence & extension FIRST (no ML deps needed) ────────
    # Cheap input validation runs before the lazy TF import so that bad file
    # types always get a 400 even when TensorFlow is not installed.
    if "file" not in request.files:
        return _error(
            "No image file in request. "
            "Send a multipart/form-data POST with field 'file'."
        )

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return _error("No file selected.")

    filename = file.filename
    ext = Path(filename).suffix.lower()

    if not _allowed_image(filename):
        return _error(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {sorted(ALLOWED_IMAGE_EXTS | {ALLOWED_DICOM_EXT})}"
        )

    # ── Lazy-import ML dependencies (after input is validated) ───────────────
    # Returns 503 Service Unavailable if TensorFlow is not installed.
    try:
        from predict_image import predict_from_image
    except ImportError as exc:
        return _error(
            f"ML dependencies not installed: {exc}. "
            "Run: pip install tensorflow pillow", 503
        )

    # ── Read file bytes & validate image dimensions ────────────────────────
    file_bytes = file.read()

    if ext != ALLOWED_DICOM_EXT:
        try:
            pil_img = PilImage.open(io.BytesIO(file_bytes))
            w, h = pil_img.size
            if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                return _error(
                    f"Image dimensions {w}×{h} exceed the maximum allowed "
                    f"({MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION} px). "
                    "Please upload a standard-resolution image."
                )
            if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                return _error(
                    f"Image dimensions {w}×{h} are too small (minimum "
                    f"{MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px)."
                )
        except Exception as exc:
            return _error(f"Could not open image file: {exc}")

    # ── Save to temp file for inference ──────────────────────────────────
    # We need a file path for Pillow/pydicom and Grad-CAM
    temp_path: Path = UPLOAD_DIR / f"temp_predict_{os.getpid()}_{filename}"
    try:
        with open(temp_path, "wb") as f_out:
            f_out.write(file_bytes)
    except OSError as exc:
        return _error(f"Could not save uploaded image temporarily: {exc}", 500)

    # ── Confidence threshold ───────────────────────────────────────────────
    try:
        conf_threshold = float(request.form.get("conf_threshold", "0.60"))
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError()
    except ValueError:
        conf_threshold = 0.60

    # ── Select model directory by scan type ───────────────────────────────
    # Each modality has its own trained model saved in a subdirectory.
    # Without this routing, CT images were classified by the X-Ray
    # (NORMAL/PNEUMONIA) binary model, returning completely wrong labels.
    #
    # Usage: include scan_type=ct | mri | xray in the multipart form.
    # Omitting it falls back to the generic model store (legacy behaviour).
    _SCAN_MODEL_MAP: dict[str, Path] = {
        "ct"  : MODEL_STORE_DIR / "ct_model",
        "mri" : MODEL_STORE_DIR / "mri_model",
        "xray": MODEL_STORE_DIR / "xray_model",
    }
    # Read scan_type from form fields first, fall back to query-string param.
    # The JS frontend sends it in BOTH places so one is always available
    # regardless of multipart body-stream consumption order.
    scan_type = (
        request.form.get("scan_type", "") or
        request.args.get("scan_type", "")
    ).lower().strip()
    selected_model_dir = _SCAN_MODEL_MAP.get(scan_type, MODEL_STORE_DIR)
    logger.info(
        "Predict: scan_type='%s' -> model_dir='%s'",
        scan_type or "generic (fallback)", selected_model_dir,
    )

    # ── Run inference ──────────────────────────────────────────────────────
    try:
        result = predict_from_image(
            image_path=str(temp_path),
            model_dir=str(selected_model_dir),
            conf_threshold=conf_threshold,
        )
    except FileNotFoundError as exc:
        return _error(str(exc), 400)
    except (RuntimeError, ValueError) as exc:
        return _error(str(exc), 400)
    except Exception:
        logger.error("Inference error:\n%s", traceback.format_exc())
        return _error("An unexpected error occurred during image inference.", 500)
    finally:
        # Always clean up the temp file
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    logger.info(
        "Prediction complete: class='%s', confidence=%.4f",
        result.get("predicted_class"),
        result.get("confidence"),
    )
    return jsonify(result), 200


# ===========================================================================
# Error handlers
# ===========================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle Flask's built-in 413 when MAX_CONTENT_LENGTH is exceeded."""
    return jsonify({
        "error": "File too large. Maximum allowed upload size is 15 MB."
    }), 413


# ===========================================================================
# RADIOLOGY REPORT TEXT ANALYSIS ENDPOINT
# ===========================================================================


@app.route("/module3/report-text/summarize", methods=["POST"])
def summarize_report_text_endpoint() -> tuple[Response, int]:
    """
    Accept raw radiology report text and return a plain-language LLM summary.

    Expected request: application/json
        { "report_text": "<raw text of the radiology report>" }

    Returns 200
    -----------
    {
        "findings": [
            {
                "original_text" : "<phrase from report>",
                "plain_language": "<patient-friendly rewrite>",
                "severity"      : "normal | mild | moderate | severe"
            }, ...
        ],
        "overall_summary"  : "<one-sentence plain-English summary>",
        "overall_severity" : "normal | attention_needed | urgent",
        "disclaimer"       : "<fixed disclaimer text>"
    }

    Returns 400 if report_text is missing or too short (< 20 chars).
    Returns 502 if the LLM call fails after one retry.
    Returns 500 on unexpected server errors.

    Privacy note: only the text LENGTH and overall_severity are logged,
    never the report content itself.
    """
    if not request.is_json:
        return _error("Request Content-Type must be 'application/json'.", 400)

    body = request.get_json(silent=True)
    if body is None:
        return _error("Request body is not valid JSON.", 400)

    report_text: str = body.get("report_text", "")
    if not isinstance(report_text, str):
        return _error("'report_text' must be a string.", 400)

    # ── Lazy import: allows server to start even if anthropic is not installed
    try:
        from report_text_analyzer import summarize_report_text
    except ImportError as exc:
        return _error(
            f"Report text analyzer dependencies not installed: {exc}. "
            "Run: pip install anthropic python-dotenv", 503
        )

    # ── Delegate to analyzer (handles input validation internally) ────────────
    try:
        result = summarize_report_text(report_text)
    except ValueError as exc:
        # Input too short or empty — caller error
        return _error(str(exc), 400)
    except RuntimeError as exc:
        # LLM failure after retry, or API key missing → 502 Bad Gateway
        logger.error("LLM call failed: %s", exc)
        return _error(str(exc), 502)
    except Exception:
        logger.error("Unexpected error in report-text summarize:\n%s",
                     traceback.format_exc())
        return _error("An unexpected error occurred.", 500)

    # Privacy-safe logging: length + severity only, never the report text
    logger.info(
        "Report text summarised: text_length=%d chars, overall_severity='%s', "
        "findings_count=%d.",
        len(report_text.strip()),
        result.get("overall_severity", "unknown"),
        len(result.get("findings", [])),
    )
    return jsonify(result), 200



# ===========================================================================
# COMBINED FULL-ANALYSIS ENDPOINT
# ===========================================================================

# ---------------------------------------------------------------------------
# Severity ranking used to compute combined_overall_status
# ---------------------------------------------------------------------------
# Maps each sub-analysis status string to a comparable integer so we can
# pick the "worst" across up to three independent sub-analyses.
_STATUS_RANK: dict[str, int] = {
    "normal"          : 0,
    "attention_needed": 1,
    "urgent"          : 2,
}
_RANK_TO_STATUS: dict[int, str] = {v: k for k, v in _STATUS_RANK.items()}


def _extract_sub_status(sub_result: dict | None, key: str) -> str | None:
    """
    Safely read a severity/status string from a sub-analysis result dict.

    Parameters
    ----------
    sub_result : dict | None
        The result dict for one sub-analysis, or None if that analysis
        was not requested / was not attempted.
    key : str
        The key to read from the dict (e.g. ``"overall_status"`` for lab
        analysis, ``"overall_severity"`` for report-text analysis).

    Returns
    -------
    str | None
        The status string if found, or None so the caller can skip this
        sub-analysis when computing the combined worst status.
    """
    if sub_result is None:
        return None
    if "error" in sub_result:
        # Sub-analysis failed — do not count its status toward combined score.
        return None
    return sub_result.get(key)  # may itself be None if key absent


@app.route("/module3/analyze-full", methods=["POST"])
def analyze_full() -> tuple[Response, int]:
    """
    Combined full-analysis endpoint: run lab, image, and/or report-text
    analyses in a single HTTP call.

    Accepts ``multipart/form-data`` with any subset of these fields:

      ``lab_values``   (form field, string)
          JSON-encoded dict of parameter → numeric value.
          Example: ``'{"Hemoglobin": 13.5, "WBC": 6.2}'``

      ``image``        (file field)
          Medical image file (.jpg / .png / .bmp / .dcm).
          Must not exceed the global MAX_CONTENT_LENGTH (15 MB).

      ``report_text``  (form field, string)
          Raw radiology report text (≥ 20 characters).

    At least one of the three fields must be present; otherwise 400.

    Each sub-analysis runs independently:
      * If a sub-analysis fails (bad input, missing model, LLM error, …)
        its result dict contains ``{"error": "<message>"}`` instead of
        the normal result keys.  The other sub-analyses are NOT affected.

    Returns 200
    -----------
    ::

        {
          "lab_analysis": { <analyze_report result> } | { "error": str } | null,
          "image_analysis": { <predict_from_image result> } | { "error": str } | null,
          "report_text_analysis": { <summarize_report_text result> } | { "error": str } | null,
          "combined_overall_status": "normal | attention_needed | urgent",
          "assistive_disclaimer": str
        }

    ``combined_overall_status`` is the WORST status found across all
    sub-analyses that succeeded (failed ones are excluded from the roll-up).
    If every sub-analysis that was attempted failed, the status defaults
    to ``"attention_needed"`` so the frontend never silently shows "normal"
    without real evidence.

    Returns 400 if no recognised input field is present.
    Returns 500 on unexpected server errors unrelated to individual analyses.

    Logging
    -------
    Logs which sub-analyses were requested/skipped and the combined result.
    For images, logs filename + extension only.  Report text length is logged
    but never its content (privacy).
    """

    # ── 0. Reject requests that carry no recognised input at all ─────────────
    has_lab      = "lab_values"  in request.form
    has_image    = "file"        in request.files   # JS sends fd.append('file', imageFile)
    has_text     = "report_text" in request.form

    if not (has_lab or has_image or has_text):
        return _error(
            "No input provided. Send at least one of: "
            "'lab_values' (form field, JSON string), "
            "'image' (file field), "
            "'report_text' (form field, string).",
            400,
        )

    logger.info(
        "analyze-full request: lab=%s, image=%s, report_text=%s",
        has_lab, has_image, has_text,
    )

    # Placeholders — None means "not requested"
    lab_result:  dict | None = None
    img_result:  dict | None = None
    text_result: dict | None = None

    # ── 1. Lab-value analysis ─────────────────────────────────────────────────
    if has_lab:
        raw_lab = request.form["lab_values"].strip()
        try:
            lab_values_dict = json.loads(raw_lab) if raw_lab else {}
            if not isinstance(lab_values_dict, dict):
                raise ValueError("'lab_values' must be a JSON object.")
            if not lab_values_dict:
                raise ValueError("'lab_values' JSON object is empty.")

            gender = request.form.get("gender", "Any").strip()
            lab_result = analyze_report(lab_values=lab_values_dict, gender=gender)

            logger.info(
                "analyze-full / lab: overall_status='%s', params=%d.",
                lab_result.get("overall_status"),
                len(lab_values_dict),
            )

        except (json.JSONDecodeError, ValueError) as exc:
            # Bad JSON or empty dict — record error but continue
            logger.warning("analyze-full / lab failed (input error): %s", exc)
            lab_result = {"error": str(exc)}
        except RuntimeError as exc:
            # No reference dataset loaded, or other domain error
            logger.warning("analyze-full / lab failed (runtime): %s", exc)
            lab_result = {"error": str(exc)}
        except Exception:
            logger.error(
                "analyze-full / lab unexpected error:\n%s", traceback.format_exc()
            )
            lab_result = {"error": "An unexpected error occurred during lab analysis."}

    # ── 2. Image analysis ─────────────────────────────────────────────────────
    if has_image:
        # Lazy-import so server starts even without TensorFlow installed
        try:
            from predict_image import predict_from_image  # type: ignore[import]
        except ImportError as exc:
            logger.warning("analyze-full / image: ML deps missing — %s", exc)
            img_result = {
                "error": (
                    f"Image analysis dependencies not installed: {exc}. "
                    "Run: pip install tensorflow pillow"
                )
            }

        if img_result is None:  # import succeeded — attempt inference
            image_file = request.files["file"]   # JS sends fd.append('file', ...)
            img_filename = image_file.filename or "upload"
            img_ext = Path(img_filename).suffix.lower()

            # ── Validate extension ─────────────────────────────────────────
            if not _allowed_image(img_filename):
                logger.warning(
                    "analyze-full / image: unsupported type '%s'.", img_ext
                )
                img_result = {
                    "error": (
                        f"Unsupported image type '{img_ext}'. "
                        f"Accepted: {sorted(ALLOWED_IMAGE_EXTS | {ALLOWED_DICOM_EXT})}"
                    )
                }
            else:
                # ── Validate image dimensions (non-DICOM only) ─────────────
                file_bytes = image_file.read()
                if img_ext != ALLOWED_DICOM_EXT:
                    try:
                        pil_img = PilImage.open(io.BytesIO(file_bytes))
                        w, h = pil_img.size
                        if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                            img_result = {
                                "error": (
                                    f"Image {w}×{h} px exceeds the "
                                    f"{MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION} px limit."
                                )
                            }
                        elif w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                            img_result = {
                                "error": (
                                    f"Image {w}×{h} px is below the minimum "
                                    f"{MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px."
                                )
                            }
                    except Exception as exc:
                        img_result = {"error": f"Could not open image file: {exc}"}

                if img_result is None:  # dimensions OK — save and infer
                    # Route to the correct model subdirectory for this scan type.
                    # The 'scan_type' form field (ct / mri / xray) selects the
                    # appropriate specialist model; omitting it falls back to the
                    # generic model store (legacy / single-model deployments).
                    _FULL_SCAN_MAP: dict[str, Path] = {
                        "ct"  : MODEL_STORE_DIR / "ct_model",
                        "mri" : MODEL_STORE_DIR / "mri_model",
                        "xray": MODEL_STORE_DIR / "xray_model",
                    }
                    full_scan_type = request.form.get("scan_type", "").lower().strip()
                    full_model_dir = _FULL_SCAN_MAP.get(full_scan_type, MODEL_STORE_DIR)

                    temp_path: Path = UPLOAD_DIR / (
                        f"full_predict_{os.getpid()}_{img_filename}"
                    )
                    try:
                        temp_path.write_bytes(file_bytes)
                        img_result = predict_from_image(
                            image_path=str(temp_path),
                            model_dir=str(full_model_dir),
                        )
                        logger.info(
                            "analyze-full / image: class='%s', conf=%.4f, "
                            "low_conf_warn=%s, ext='%s'.",
                            img_result.get("predicted_class"),
                            img_result.get("confidence", 0.0),
                            img_result.get("low_confidence_warning"),
                            img_ext,
                        )
                    except (FileNotFoundError, ValueError, RuntimeError) as exc:
                        logger.warning(
                            "analyze-full / image inference failed: %s", exc
                        )
                        img_result = {"error": str(exc)}
                    except Exception:
                        logger.error(
                            "analyze-full / image unexpected error:\n%s",
                            traceback.format_exc(),
                        )
                        img_result = {
                            "error": "An unexpected error occurred during image inference."
                        }
                    finally:
                        # Always clean up the temp file regardless of outcome
                        try:
                            temp_path.unlink(missing_ok=True)
                        except OSError:
                            pass

    # ── 3. Radiology report text analysis ────────────────────────────────────
    if has_text:
        # Lazy-import so server starts even without the anthropic package
        try:
            from report_text_analyzer import summarize_report_text  # type: ignore[import]
        except ImportError as exc:
            logger.warning("analyze-full / report-text: deps missing — %s", exc)
            text_result = {
                "error": (
                    f"Report text analyzer dependencies not installed: {exc}. "
                    "Run: pip install anthropic python-dotenv"
                )
            }

        if text_result is None:  # import succeeded
            raw_text = request.form["report_text"]
            try:
                text_result = summarize_report_text(raw_text)
                # Privacy-safe: log length + severity, never the content
                logger.info(
                    "analyze-full / report-text: text_length=%d chars, "
                    "overall_severity='%s', findings=%d.",
                    len(raw_text.strip()),
                    text_result.get("overall_severity"),
                    len(text_result.get("findings", [])),
                )
            except ValueError as exc:
                # Text too short / empty
                logger.warning(
                    "analyze-full / report-text input error: %s", exc
                )
                text_result = {"error": str(exc)}
            except RuntimeError as exc:
                # LLM failure after retry, or API key not set
                logger.warning(
                    "analyze-full / report-text LLM error: %s", exc
                )
                text_result = {"error": str(exc)}
            except Exception:
                logger.error(
                    "analyze-full / report-text unexpected error:\n%s",
                    traceback.format_exc(),
                )
                text_result = {
                    "error": "An unexpected error occurred during report text analysis."
                }

    # ── 4. Compute combined_overall_status ────────────────────────────────────
    # Strategy: take the WORST rank across all sub-analyses that SUCCEEDED.
    # A sub-analysis is excluded from this roll-up if it was not requested
    # (None) or if it contains an "error" key.
    #
    # Lab uses "overall_status"; image uses predicted_class / low_confidence_warning
    # (no single severity field); report-text uses "overall_severity".
    # Image analysis does not emit a scalar severity, so we derive one:
    #   low_confidence_warning=True → "attention_needed" (inconclusive)
    #   predicted_class contains "normal" or "NORMAL" → "normal"
    #   anything else → "attention_needed" (flag for human review)
    #
    # Rationale for image mapping: the model only says WHAT class it found;
    # whether that class is "urgent" is domain-specific and not encoded in
    # the general-purpose model output.  We conservatively flag non-normal
    # findings as attention_needed rather than urgent.

    sub_statuses: list[str] = []

    # -- Lab severity
    lab_status = _extract_sub_status(lab_result, "overall_status")
    if lab_status and lab_status in _STATUS_RANK:
        sub_statuses.append(lab_status)

    # -- Image severity (derived — see rationale in endpoint docstring)
    # Mapping:
    #   low_confidence_warning=True  → "attention_needed" (model is uncertain)
    #   predicted class = normal     → "normal"
    #   pathological keyword found   → "urgent"  (PNEUMONIA, TUMOR, CYST, STONE …)
    #   any other non-normal class   → "attention_needed" (flag for review)
    _IMG_URGENT_KEYWORDS = {
        "pneumonia", "tumor", "tumour", "cancer", "malignant",
        "cyst", "stone", "calcul", "nodule", "mass", "lesion",
        "fracture", "bleed", "haemorrhage", "hemorrhage",
        "effusion", "abscess", "infarct",
    }
    if img_result is not None and "error" not in img_result:
        if img_result.get("low_confidence_warning"):
            sub_statuses.append("attention_needed")
        else:
            predicted_class: str = str(
                img_result.get("predicted_class", "")
            ).lower()
            if "normal" in predicted_class or "no finding" in predicted_class:
                sub_statuses.append("normal")
            elif any(kw in predicted_class for kw in _IMG_URGENT_KEYWORDS):
                sub_statuses.append("urgent")
            else:
                # Unknown non-normal class — flag for review conservatively
                sub_statuses.append("attention_needed")

    # -- Report-text severity
    text_status = _extract_sub_status(text_result, "overall_severity")
    if text_status and text_status in _STATUS_RANK:
        sub_statuses.append(text_status)

    if sub_statuses:
        # Pick the worst (highest rank) among all successful sub-analyses
        worst_rank = max(_STATUS_RANK.get(s, 0) for s in sub_statuses)
        combined_status = _RANK_TO_STATUS[worst_rank]
    else:
        # Every attempted sub-analysis failed — flag conservatively
        combined_status = "attention_needed"

    # ── 5. Assemble response ──────────────────────────────────────────────────
    COMBINED_DISCLAIMER = (
        "This report combines multiple AI-generated preliminary assessments. "
        "None of these results are a diagnosis. "
        "Please consult a licensed healthcare professional to review all "
        "findings together."
    )

    logger.info(
        "analyze-full complete: lab=%s, image=%s, text=%s, combined='%s'.",
        "ok" if (lab_result  and "error" not in lab_result)  else ("err" if lab_result  else "skip"),
        "ok" if (img_result  and "error" not in img_result)  else ("err" if img_result  else "skip"),
        "ok" if (text_result and "error" not in text_result) else ("err" if text_result else "skip"),
        combined_status,
    )

    return jsonify({
        "lab_analysis"         : lab_result,
        "image_analysis"       : img_result,
        "report_text_analysis" : text_result,
        "combined_overall_status": combined_status,
        "assistive_disclaimer" : COMBINED_DISCLAIMER,
    }), 200


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    port = int(os.environ.get("MODULE3_PORT", 5003))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info("Starting VitalSense AI — Module 3 on port %d (debug=%s)", port, debug)

    # Optionally auto-load a reference CSV at startup if one already exists
    default_csv: Path = UPLOAD_DIR / "reference_ranges.csv"
    if default_csv.exists():
        try:
            rd.load_reference_ranges(default_csv)
            logger.info("Auto-loaded reference dataset from: %s", default_csv)
        except Exception as exc:
            logger.warning("Could not auto-load default CSV: %s", exc)

    app.run(host="0.0.0.0", port=port, debug=debug)
