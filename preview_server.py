"""
preview_server.py — VitalSense AI | Unified Server
====================================================
Single Flask server for ALL VitalSense AI functionality.
Module 2 (Medicine/Diet Recommendation) and Module 3 (Report/Image Analysis)
are embedded directly — no separate backend processes needed.

Routes served
-------------
  Pages:
    GET  /                 -> redirect to /signup
    GET  /signup           -> signup.html
    GET  /login            -> login.html
    GET  /predict          -> predict.html  (Module 1)
    GET  /home             -> home.html
    GET  /module2          -> recommend.html
    GET  /xray             -> upload-report.html

  Module 1 — Disease Prediction:
    POST /predict              -> disease prediction (ML or Gemini AI fallback)
    GET  /dataset/symptoms     -> symptom list (JSON)

  Module 2 — Medicine & Diet (EMBEDDED):
    POST /module2/recommend        -> recommendation (CSV + AI fallback)
    POST /module2/ai-recommend     -> pure AI recommendation (Gemini)
    POST /module2/upload-datasets  -> upload CSV datasets
    GET  /module2/diseases         -> list diseases in loaded datasets
    GET  /module2/ai-status        -> Gemini AI availability check
    POST /module2/ai-precautions   -> AI clinical precautions

  Module 3 — Report & Image Analysis (EMBEDDED):
    POST /module3/upload-reference       -> upload reference CSV
    POST /module3/analyze                -> analyse lab values
    GET  /module3/parameters             -> list parameters + units
    GET  /module3/health                 -> health / status check
    POST /module3/image/predict          -> image screening (subprocess + mock)
    POST /module3/report-text/summarize  -> radiology report text summary
    POST /module3/analyze-full           -> combined lab + image + text analysis

  Auth (Mock):
    POST /auth/signup, /auth/login, /auth/logout, etc.

Run with:
  python preview_server.py
Then open:
  http://localhost:5050/home
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Load .env file automatically if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, jsonify, redirect, render_template, request, url_for, session
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
import joblib
from predict import predict_disease

# ── Module 2 — direct imports (no separate server on port 5001 needed) ───
_module2_dir = str(Path(__file__).parent / "module2")
if _module2_dir not in sys.path:
    sys.path.insert(0, _module2_dir)

import clinical_precautions
import dataset_manager as m2_dataset_manager
import recommend as m2_recommender
import ai_fallback as m2_ai_fallback
from severity_calculator import calculate_severity as m2_calculate_severity

# ── Module 3 — direct imports (no separate server on port 5003 needed) ───
_module3_dir_path = Path(__file__).parent / "module3"
_module3_dir = str(_module3_dir_path)
if _module3_dir not in sys.path:
    sys.path.insert(0, _module3_dir)

import reference_data as m3_rd
from report_analyzer import analyze_report as m3_analyze_report

# ── Module 5 — Medicine Reminder ─────────────────────────────────────────
_module5_dir = str(Path(__file__).parent / "module5_medicine")
if _module5_dir not in sys.path:
    sys.path.insert(0, _module5_dir)

from module5_medicine import medicine_db as m5_db
from module5_medicine import medicine_scheduler as m5_scheduler

# ── Module 4 — VitalSense Vitals Monitor ─────────────────────────────────
_module4_dir = str(Path(__file__).parent / "module4_vitalsense")
if _module4_dir not in sys.path:
    sys.path.insert(0, _module4_dir)

from dataclasses import asdict as _m4_asdict
from vitals_model import VitalsReading as M4_VitalsReading, validate_reading as m4_validate
from vitals_storage import save_reading as m4_save, get_latest as m4_latest, get_history as m4_history, get_daily_average as m4_daily_avg
from vitals_alerts import check_alerts as m4_check_alerts

# ── Module 6 — Personal Health Profile ────────────────────────────────────
from module6_health_profile import health_profile_db as m6_db

# ── Module 7 — Natural & Herbal Remedies ──────────────────────────────────
from module7_herbal import herbal_kb as m7_kb
from module7_herbal import herbal_ai as m7_ai

# ── Auth Database (SQLite + Flask-Login) ─────────────────────────────────
from auth_db import init_auth_db, register_user, authenticate_user, get_user_by_id, get_all_users_count


# =========================================================================
# Logging
# =========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("vitalsense.unified")


# =========================================================================
# Flask App
# =========================================================================
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "vitalsense-dev-secret-key-change-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

# ── CORS — allow APK and any frontend origin to call APIs ────────────────
CORS(app, supports_credentials=True)

# ── Flask-Login setup ────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))

# ── Initialise auth database ─────────────────────────────────────────────
init_auth_db()

# ── Real model artefacts (Module 1) ──────────────────────────────────────
MODEL_DIR    = Path(__file__).parent / "model"
SYMPTOM_LIST = joblib.load(MODEL_DIR / "symptom_list.pkl")   # 131 real symptoms


# =========================================================================
# Module 2 — Initialisation (auto-load existing datasets on startup)
# =========================================================================
_M2_BASE_DIR = Path(__file__).parent / "module2"
_M2_UPLOAD_FOLDER = _M2_BASE_DIR / "data"
_M2_UPLOAD_FOLDER.mkdir(exist_ok=True)

_M2_DATASET_FILENAMES: dict[str, str] = {
    "precaution_file": "precautions_df.csv",
    "severity_file":   "Symptom-severity.csv",
    "medicine_file":   "medications.csv",
    "diet_file":       "diets.csv",
}


def _m2_autoload_datasets() -> None:
    """Auto-load Module 2 datasets from disk if CSVs already exist."""
    for field_name, filename in _M2_DATASET_FILENAMES.items():
        csv_path = _M2_UPLOAD_FOLDER / filename
        if not csv_path.exists():
            continue
        try:
            if field_name == "precaution_file":
                m2_dataset_manager.load_precautions(csv_path)
            elif field_name == "severity_file":
                m2_dataset_manager.load_severity_weights(csv_path)
            elif field_name == "medicine_file":
                m2_dataset_manager.load_medicine_data(csv_path)
            elif field_name == "diet_file":
                m2_dataset_manager.load_diet_data(csv_path)
            logger.info("[Module 2] Auto-loaded dataset '%s'.", filename)
        except Exception as exc:
            logger.warning("[Module 2] Could not auto-load '%s': %s", filename, exc)


_m2_autoload_datasets()


# =========================================================================
# Module 3 — Initialisation
# =========================================================================
_M3_DIR = _module3_dir_path
_M3_UPLOAD_DIR = _M3_DIR / "uploads"
_M3_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_M3_MODEL_STORE_DIR = _M3_DIR / "model_store"
_M3_MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

_M3_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
_M3_ALLOWED_DICOM_EXT  = ".dcm"

# Auto-load reference CSV if it exists on disk
_m3_default_csv = _M3_UPLOAD_DIR / "reference_ranges.csv"
if _m3_default_csv.exists():
    try:
        m3_rd.load_reference_ranges(_m3_default_csv)
        logger.info("[Module 3] Auto-loaded reference dataset.")
    except Exception as exc:
        logger.warning("[Module 3] Could not auto-load default CSV: %s", exc)

# ── Subprocess inference setup (image analysis via venv312 Python + TF) ──
_VENV_PYTHON  = _M3_DIR / "venv312" / "Scripts" / "python.exe"
_INFER_SCRIPT = _M3_DIR / "infer.py"

DISCLAIMER = (
    "This is an AI-generated preliminary screening result based on image patterns. "
    "It is NOT a medical diagnosis. A qualified radiologist or physician must review "
    "the original image and confirm any finding before any medical decision is made."
)

# Fallback mock responses (used ONLY when real inference is unavailable)
_IMAGE_MOCKS = {
    "xray": {"predicted_class": "PNEUMONIA", "confidence": 0.8731, "low_confidence_warning": False, "assistive_disclaimer": DISCLAIMER},
    "mri":  {"predicted_class": "Glioma Tumour", "confidence": 0.7642, "low_confidence_warning": False, "assistive_disclaimer": DISCLAIMER},
    "ct":   {"predicted_class": "Normal", "confidence": 0.9215, "low_confidence_warning": False, "assistive_disclaimer": DISCLAIMER},
}

_TEXT_MOCK = {
    "findings": [
        {"original_text": "Haemoglobin: 9.8 g/dL (Ref: 12-16 g/dL)", "plain_language": "Your haemoglobin is lower than the normal range, which may indicate mild anaemia.", "severity": "moderate"},
        {"original_text": "Fasting Blood Glucose: 118 mg/dL (Ref: 70-100 mg/dL)", "plain_language": "Your fasting blood sugar is slightly above normal. This could suggest pre-diabetes.", "severity": "moderate"},
        {"original_text": "Serum Creatinine: 0.9 mg/dL (Ref: 0.6-1.2 mg/dL)", "plain_language": "Your kidney function marker is within the normal range.", "severity": "normal"},
    ],
    "overall_summary": "Your report shows mild anaemia and slightly elevated fasting blood sugar. Kidney function appears normal.",
    "overall_severity": "moderate",
    "disclaimer": "This AI-generated summary is for educational purposes only and does not constitute medical advice.",
}


# =========================================================================
# Auth tokens (JWT-style session tokens for API compatibility)
# =========================================================================
import secrets as _secrets
def _generate_token() -> str:
    return _secrets.token_urlsafe(32)

# ── Disease-specific foods to AVOID lookup ────────────────────────────────
AVOID_FOODS: dict[str, list[str]] = {
    'Fungal Infection':      ['Sugary foods and refined carbs', 'Processed foods', 'Alcohol', 'Yeast-containing foods (bread, beer)', 'High-lactose dairy'],
    'Allergy':               ['Known allergens (dust, pollen triggers)', 'Processed foods with artificial additives', 'Alcohol', 'Spicy foods', 'Fermented foods'],
    'Gerd':                  ['Spicy and acidic foods', 'Tomatoes and citrus fruits', 'Alcohol and caffeine', 'Fried and fatty foods', 'Chocolate and mint'],
    'Chronic Cholestasis':   ['Alcohol (absolutely)', 'Oily and fried foods', 'Red meat', 'High-fat dairy', 'Processed and salty snacks'],
    'Drug Reaction':         ['Alcohol (interacts with medications)', 'Grapefruit (CYP3A4 inhibitor)', 'High-sodium foods', 'Heavily processed snacks', 'Caffeine'],
    'Peptic Ulcer Disease':  ['Spicy and acidic foods', 'Alcohol', 'Caffeine and coffee', 'Fried fatty foods', 'Carbonated drinks'],
    'Aids':                  ['Raw or undercooked meat/eggs/fish', 'Unpasteurised dairy', 'Raw sprouts', 'Alcohol', 'High-sugar processed foods'],
    'Diabetes':              ['White rice, maida and refined carbs', 'Sugary drinks and juices', 'Deep-fried snacks', 'Alcohol', 'Full-fat dairy and processed meats'],
    'Gastroenteritis':       ['Dairy products (lactose intolerance risk post-illness)', 'Fatty and fried foods', 'Alcohol', 'Caffeinated drinks', 'Spicy foods'],
    'Bronchial Asthma':      ['Sulphite-preserved foods (wine, dried fruits)', 'Processed and junk food', 'Very cold drinks', 'Shellfish if allergic', 'Excessive dairy'],
    'Hypertension':          ['High-sodium processed foods', 'Red meat and saturated fats', 'Alcohol', 'Caffeine in excess', 'Pickles and ready-made sauces'],
    'Migraine':              ['Red wine and alcohol', 'Aged cheeses (tyramine)', 'Processed meats with nitrates', 'Excess caffeine', 'Artificial sweeteners'],
    'Cervical Spondylosis':  ['Inflammatory foods (trans fats, refined sugars)', 'Alcohol', 'Excess salt', 'Processed and packaged snacks', 'Sugary carbonated drinks'],
    'Paralysis (Brain Hemorrhage)': ['High-sodium foods', 'Trans fats and saturated fats', 'Alcohol', 'Excess sugar', 'Red meat in large amounts'],
    'Jaundice':              ['Alcohol (absolutely)', 'Oily and fried foods', 'High-salt processed foods', 'Red meat', 'Raw or undercooked shellfish'],
    'Malaria':               ['Heavy oily meals', 'Raw salads during acute phase', 'Alcohol', 'Very spicy foods', 'Caffeinated drinks'],
    'Chicken Pox':           ['Salty and spicy foods', 'Acidic foods (citrus, tomatoes)', 'Hard crunchy foods', 'Foods high in arginine (nuts, chocolate)', 'Alcohol'],
    'Dengue':                ['Oily and spicy foods', 'Alcohol', 'Aspirin and NSAIDs (bleeding risk)', 'High-fibre foods during acute phase', 'Caffeinated drinks'],
    'Typhoid':               ['Raw salads and uncooked vegetables', 'Spicy or acidic foods', 'High-fibre foods during acute phase', 'Alcohol', 'Fried and oily snacks'],
    'Hepatitis A':           ['Alcohol (absolutely)', 'Fatty and fried foods', 'High-sugar processed foods', 'Raw shellfish', 'Excess salt'],
    'Hepatitis B':           ['Alcohol (absolutely)', 'Fatty and fried foods', 'High-salt foods', 'Raw or undercooked meats', 'High-iron supplements without advice'],
    'Hepatitis C':           ['Alcohol (absolutely)', 'High-fat and fried foods', 'Red meat', 'High-iron foods in excess', 'Refined sugar'],
    'Hepatitis D':           ['Alcohol (absolutely)', 'Fatty foods', 'Processed snacks', 'Raw meats', 'High-sugar foods'],
    'Hepatitis E':           ['Alcohol (absolutely)', 'Oily and fried foods', 'Raw or undercooked shellfish', 'Excess salt', 'Processed snacks'],
    'Alcoholic Hepatitis':   ['Alcohol (absolutely — must stop completely)', 'Fatty foods', 'Excess salt', 'Processed meats', 'High-sugar foods'],
    'Tuberculosis':          ['Alcohol (interferes with TB medication)', 'Tobacco', 'Raw or undercooked foods', 'Very high-fat meals', 'Refined sugar'],
    'Common Cold':           ['Dairy products (can thicken mucus)', 'Alcohol', 'Fried and processed foods', 'Cold sugary drinks', 'Very spicy foods'],
    'Pneumonia':             ['Cold drinks and ice cream', 'Dairy in large quantities', 'Fried and greasy foods', 'Alcohol', 'Heavily spiced dishes'],
    'Dimorphic Hemmorhoids(Piles)': ['Spicy foods', 'Alcohol', 'Low-fibre processed foods', 'Caffeinated drinks', 'Red meat in excess'],
    'Heart Attack':          ['Trans fats and saturated fats', 'High-sodium processed foods', 'Alcohol', 'Sugary drinks', 'Red meat and processed meats'],
    'Varicose Veins':        ['High-sodium foods (cause water retention)', 'Alcohol', 'Refined carbohydrates', 'Trans fats', 'Excess caffeine'],
    'Hypothyroidism':        ['Goitrogenic raw vegetables (broccoli, cabbage) in excess', 'Soy products in excess', 'Gluten (if sensitive)', 'Alcohol', 'Excess sugar'],
    'Hyperthyroidism':       ['Iodine-rich foods (seaweed, iodised salt) in excess', 'Caffeine', 'Alcohol', 'Refined sugar', 'Very spicy stimulating foods'],
    'Hypoglycemia':          ['Long gaps between meals', 'Alcohol on an empty stomach', 'High-sugar foods that cause rapid spikes then crashes', 'Very low-carb diets', 'Excess caffeine'],
    'Osteoarthristis':       ['Inflammatory foods (processed meats, trans fats)', 'Refined sugar', 'Alcohol', 'High-sodium foods', 'Vegetable oils high in omega-6'],
    'Arthritis':             ['Processed and ultra-processed foods', 'Refined sugar and sugary drinks', 'Red meat', 'Alcohol', 'High-salt foods'],
    '(Vertigo) Paroymsal Positional Vertigo': ['Alcohol', 'Excess salt (worsens fluid in inner ear)', 'Caffeine', 'High-sugar foods', 'Processed snacks'],
    'Acne':                  ['High-glycaemic foods (white bread, white rice)', 'Dairy products', 'Fried and greasy foods', 'Excess sugar', 'Chocolate in large amounts'],
    'Urinary Tract Infection': ['Alcohol and caffeine (bladder irritants)', 'Spicy foods', 'Sugary foods and drinks', 'Artificial sweeteners', 'Carbonated soft drinks'],
    'Psoriasis':             ['Alcohol', 'Red meat and processed meats', 'Dairy', 'Gluten (for sensitive individuals)', 'Nightshade vegetables if trigger (tomatoes, peppers)'],
    'Impetigo':              ['Sugary foods (can promote bacterial growth)', 'Processed and salty snacks', 'Alcohol', 'Foods that weaken immunity', 'Excess refined carbohydrates'],
}


# =========================================================================
# Page routes
# =========================================================================
@app.route("/")
def index():
    return redirect(url_for("signup"))


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/predict")
def predict_page():
    """Disease Prediction screen — Module 1."""
    return render_template("predict.html")


@app.route('/home')
def home():
    """Post-login home hub."""
    return render_template('home.html')


@app.route('/module2')
def module2():
    """Recommendation screen — Module 2."""
    return render_template('recommend.html')


@app.route('/xray')
def xray():
    """Medical Reports Analyser screen — upload X-Ray / MRI / CT / Diagnostic reports."""
    return render_template('upload-report.html')


@app.route('/assistant')
def assistant_page():
    """AI Assistant chat page."""
    return render_template('ai_assistant.html')


@app.route('/health-profile')
def health_profile_page():
    """Personal Health Profile page — Module 6."""
    return render_template('health_profile.html')


@app.route('/herbal-remedies')
def herbal_remedies_page():
    """Natural & Herbal Remedies page — Module 7."""
    return render_template('herbal_remedies.html')


@app.route('/settings')
def settings_page():
    """User settings page — profile, notifications, appearance, etc."""
    return render_template('settings.html')


# =========================================================================
# Mock APIs
# =========================================================================

@app.get('/auth/me')
def auth_me():
    """Return the currently authenticated user from the session."""
    if current_user.is_authenticated:
        return jsonify(current_user.to_dict()), 200
    return jsonify({"id": "u_anon", "name": "there", "email": ""}), 200


@app.get('/vitals/latest/<user_id>')
def vitals_latest(user_id):
    """Return mock latest vitals for home.js preview."""
    import datetime
    recorded = (datetime.datetime.utcnow() - datetime.timedelta(minutes=3)).isoformat() + 'Z'
    return json.dumps({
        'heart_rate_bpm': 72,
        'spo2_percent': 98.4,
        'recorded_at': recorded
    }), 200, {'Content-Type': 'application/json'}


@app.post('/auth/logout')
def auth_logout():
    """Clear session and confirm logout."""
    logout_user()
    return jsonify({'ok': True}), 200


# ── Weather cache & helper ────────────────────────────────────────────────
_WEATHER_CACHE: dict[str, tuple[float, dict]] = {}  # key -> (timestamp, data)
_WEATHER_CACHE_TTL = 600  # 10 minutes cache

@app.get('/api/weather')
def get_live_weather():
    """
    GET /api/weather?lat=...&lon=...&city=...
    Fetch live real-time weather using Open-Meteo & reverse geocoding.
    If no coordinates are provided, uses IP geolocation fallback (or Chennai default).
    Cached for 10 minutes per location.
    """
    import urllib.request
    import urllib.parse
    import time

    lat_arg = request.args.get('lat')
    lon_arg = request.args.get('lon')
    city_arg = request.args.get('city')

    lat: float | None = None
    lon: float | None = None
    city = city_arg
    country = "India"

    if lat_arg and lon_arg:
        try:
            lat = round(float(lat_arg), 4)
            lon = round(float(lon_arg), 4)
        except (ValueError, TypeError):
            lat, lon = None, None

    if lat is None or lon is None:
        cache_key_ip = "auto_ip"
        now = time.time()
        if cache_key_ip in _WEATHER_CACHE and (now - _WEATHER_CACHE[cache_key_ip][0]) < _WEATHER_CACHE_TTL:
            return jsonify(_WEATHER_CACHE[cache_key_ip][1])

        try:
            req = urllib.request.Request(
                'https://ipwho.is/',
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                ip_data = json.loads(resp.read().decode('utf-8'))
                if ip_data.get('success'):
                    lat = float(ip_data.get('latitude', 13.0827))
                    lon = float(ip_data.get('longitude', 80.2707))
                    city = ip_data.get('city') or "Chennai"
                    country = ip_data.get('country') or "India"
        except Exception:
            pass

    if lat is None or lon is None:
        lat, lon = 13.0827, 80.2707
        city = city or "Chennai"
        country = country or "India"

    coord_key = f"{round(lat, 2)}_{round(lon, 2)}"
    now = time.time()
    if coord_key in _WEATHER_CACHE and (now - _WEATHER_CACHE[coord_key][0]) < _WEATHER_CACHE_TTL:
        cached_res = dict(_WEATHER_CACHE[coord_key][1])
        if city and not cached_res.get('city'):
            cached_res['city'] = city
            cached_res['location'] = f"{city}, {country}" if country else city
        return jsonify(cached_res)

    if not city:
        try:
            geo_url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
            req = urllib.request.Request(geo_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                geo_data = json.loads(resp.read().decode('utf-8'))
                city = geo_data.get('city') or geo_data.get('locality') or geo_data.get('principalSubdivision')
                if geo_data.get('countryName'):
                    country = geo_data['countryName']
        except Exception:
            pass

    city = city or "Local Area"

    try:
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation"
            "&hourly=temperature_2m,weather_code"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"
            "&forecast_days=7&timezone=auto"
        )
        req = urllib.request.Request(om_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            raw_w = json.loads(resp.read().decode('utf-8'))

        raw_w['success'] = True
        raw_w['city'] = city
        raw_w['country'] = country
        raw_w['location'] = f"{city}, {country}" if country and country != city else city
        raw_w['timestamp'] = now

        _WEATHER_CACHE[coord_key] = (now, raw_w)
        _WEATHER_CACHE["auto_ip"] = (now, raw_w)

        return jsonify(raw_w)

    except Exception as exc:
        logger.error("[Weather] Open-Meteo fetch failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


# =========================================================================
# DOCTOR SUPPORT MODULE — Routes
# =========================================================================

@app.get('/doctors')
def doctors_page():
    """Render the Doctor Support page."""
    return render_template('doctors.html')


# ── Nearby Healthcare Search (Overpass / OpenStreetMap) ───────────────────
_HEALTHCARE_CACHE: dict[str, tuple[float, list]] = {}
_HEALTHCARE_CACHE_TTL = 300  # 5 min cache

@app.get('/api/nearby-healthcare')
def api_nearby_healthcare():
    """
    GET /api/nearby-healthcare?lat=...&lon=...&radius=5000&type=all
    Search nearby hospitals, clinics, doctors, pharmacies via Overpass API.
    """
    import urllib.request
    import math

    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    radius = request.args.get('radius', 5000, type=int)
    hc_type = request.args.get('type', 'all')

    if lat is None or lon is None:
        return jsonify({"success": False, "error": "lat and lon required"}), 400

    radius = min(radius, 25000)  # cap at 25km
    cache_key = f"{round(lat,3)}_{round(lon,3)}_{radius}"
    now = time.time()

    if cache_key in _HEALTHCARE_CACHE and (now - _HEALTHCARE_CACHE[cache_key][0]) < _HEALTHCARE_CACHE_TTL:
        all_results = _HEALTHCARE_CACHE[cache_key][1]
    else:
        # Build Overpass QL query for healthcare POIs
        overpass_query = f"""
        [out:json][timeout:15];
        (
          node["amenity"="hospital"](around:{radius},{lat},{lon});
          way["amenity"="hospital"](around:{radius},{lat},{lon});
          node["amenity"="clinic"](around:{radius},{lat},{lon});
          way["amenity"="clinic"](around:{radius},{lat},{lon});
          node["amenity"="doctors"](around:{radius},{lat},{lon});
          way["amenity"="doctors"](around:{radius},{lat},{lon});
          node["amenity"="dentist"](around:{radius},{lat},{lon});
          way["amenity"="dentist"](around:{radius},{lat},{lon});
          node["amenity"="pharmacy"](around:{radius},{lat},{lon});
          way["amenity"="pharmacy"](around:{radius},{lat},{lon});
          node["healthcare"](around:{radius},{lat},{lon});
          way["healthcare"](around:{radius},{lat},{lon});
        );
        out center body;
        """
        try:
            encoded = urllib.parse.urlencode({"data": overpass_query})
            req = urllib.request.Request(
                f"https://overpass-api.de/api/interpreter?{encoded}",
                headers={"User-Agent": "VitalSenseAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.error("[Healthcare] Overpass API error: %s", exc)
            return jsonify({"success": False, "error": "Search service temporarily unavailable"}), 503

        all_results = []
        seen_names = set()

        for elem in raw.get("elements", []):
            tags = elem.get("tags", {})
            name = tags.get("name", "").strip()
            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            # Determine coordinates (nodes vs ways with center)
            e_lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
            e_lon = elem.get("lon") or (elem.get("center", {}).get("lon"))
            if not e_lat or not e_lon:
                continue

            # Classify type
            amenity = tags.get("amenity", "")
            healthcare = tags.get("healthcare", "")
            if amenity == "hospital" or healthcare == "hospital":
                place_type = "hospital"
            elif amenity == "clinic" or healthcare == "clinic":
                place_type = "clinic"
            elif amenity == "doctors" or healthcare == "doctor":
                place_type = "doctor"
            elif amenity == "dentist" or healthcare == "dentist":
                place_type = "dentist"
            elif amenity == "pharmacy" or healthcare == "pharmacy":
                place_type = "pharmacy"
            else:
                place_type = "clinic"

            emergency = tags.get("emergency", "")
            if emergency in ("yes", "emergency"):
                place_type = "emergency"

            # Haversine distance
            dlat = math.radians(e_lat - lat)
            dlon = math.radians(e_lon - lon)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(e_lat)) * math.sin(dlon/2)**2
            dist_km = round(6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)

            # Build address
            addr_parts = []
            for k in ("addr:street", "addr:housenumber", "addr:city", "addr:postcode"):
                if tags.get(k):
                    addr_parts.append(tags[k])
            address = ", ".join(addr_parts) if addr_parts else tags.get("addr:full", "")

            # Specialty
            specialty = tags.get("healthcare:speciality", "") or tags.get("medical_system:speciality", "") or tags.get("health_specialty:type", "")

            opening_hours = tags.get("opening_hours", "")
            phone = tags.get("phone", "") or tags.get("contact:phone", "")
            website = tags.get("website", "") or tags.get("contact:website", "")

            all_results.append({
                "name": name,
                "type": place_type,
                "lat": e_lat,
                "lon": e_lon,
                "distance_km": dist_km,
                "address": address,
                "phone": phone,
                "website": website,
                "specialty": specialty,
                "opening_hours": opening_hours,
                "emergency": emergency in ("yes", "emergency"),
            })

        all_results.sort(key=lambda x: x["distance_km"])
        _HEALTHCARE_CACHE[cache_key] = (now, all_results)

    # Filter by type if specified
    if hc_type and hc_type != "all":
        filtered = [r for r in all_results if r["type"] == hc_type]
    else:
        filtered = all_results

    return jsonify({
        "success": True,
        "count": len(filtered),
        "results": filtered[:50],
    })


# ── Geocode search (Nominatim) ───────────────────────────────────────────
@app.get('/api/geocode')
def api_geocode():
    """
    GET /api/geocode?q=Chennai
    Forward geocode a city/area/PIN to lat/lon via Nominatim.
    """
    import urllib.request

    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"success": False, "error": "query required"}), 400

    try:
        encoded = urllib.parse.urlencode({"q": query, "format": "json", "limit": "1"})
        req = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{encoded}",
            headers={"User-Agent": "VitalSenseAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = json.loads(resp.read().decode("utf-8"))

        if not results:
            return jsonify({"success": False, "error": "Location not found"}), 404

        r = results[0]
        return jsonify({
            "success": True,
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "display_name": r.get("display_name", query),
        })
    except Exception as exc:
        logger.error("[Geocode] Nominatim error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500


# ── Symptom Triage Engine ────────────────────────────────────────────────
_BODY_AREA_SYMPTOMS = {
    "head": [
        "Headache", "Dizziness", "Blurred vision", "Fainting",
        "Confusion", "Memory problems", "Ear pain", "Hearing loss",
        "Sore throat", "Runny nose", "Nasal congestion", "Nosebleeds",
        "Eye redness", "Eye pain", "Jaw pain", "Facial swelling",
    ],
    "chest": [
        "Chest pain", "Chest tightness", "Shortness of breath",
        "Rapid heartbeat", "Palpitations", "Coughing",
        "Coughing blood", "Wheezing", "Difficulty breathing",
    ],
    "abdomen": [
        "Stomach pain", "Nausea", "Vomiting", "Diarrhea",
        "Constipation", "Bloating", "Loss of appetite",
        "Blood in stool", "Heartburn", "Abdominal cramps",
    ],
    "limbs": [
        "Joint pain", "Muscle pain", "Swollen joints", "Numbness",
        "Tingling", "Weakness in legs", "Weakness in arms",
        "Back pain", "Neck pain", "Stiffness", "Difficulty walking",
    ],
    "skin": [
        "Skin rash", "Itching", "Redness", "Swelling",
        "Blisters", "Bruising easily", "Dry skin",
        "Skin discoloration", "Wound not healing", "Hair loss",
    ],
    "general": [
        "Fever", "Fatigue", "Weight loss", "Weight gain",
        "Night sweats", "Chills", "Swollen lymph nodes",
        "Frequent urination", "Painful urination", "Excessive thirst",
    ],
    "mouth": [
        "Toothache", "Bleeding gums", "Mouth sores", "Bad breath",
        "Difficulty swallowing", "Jaw stiffness", "Tooth sensitivity",
    ],
}

# Emergency symptoms — always flag critical
_EMERGENCY_SYMPTOMS = {
    "Chest pain", "Coughing blood", "Blood in stool", "Fainting",
    "Difficulty breathing", "Confusion", "Severe headache",
}

# Symptom → Specialist mapping
_SYMPTOM_SPECIALIST_MAP = {
    "Headache": "Neurologist", "Dizziness": "Neurologist",
    "Blurred vision": "Ophthalmologist", "Fainting": "Cardiologist",
    "Confusion": "Neurologist", "Memory problems": "Neurologist",
    "Ear pain": "ENT Specialist", "Hearing loss": "ENT Specialist",
    "Sore throat": "ENT Specialist", "Runny nose": "ENT Specialist",
    "Nasal congestion": "ENT Specialist", "Nosebleeds": "ENT Specialist",
    "Eye redness": "Ophthalmologist", "Eye pain": "Ophthalmologist",
    "Jaw pain": "Dentist", "Facial swelling": "ENT Specialist",
    "Chest pain": "Cardiologist", "Chest tightness": "Cardiologist",
    "Shortness of breath": "Pulmonologist", "Rapid heartbeat": "Cardiologist",
    "Palpitations": "Cardiologist", "Coughing": "Pulmonologist",
    "Coughing blood": "Pulmonologist", "Wheezing": "Pulmonologist",
    "Difficulty breathing": "Pulmonologist",
    "Stomach pain": "Gastroenterologist", "Nausea": "Gastroenterologist",
    "Vomiting": "Gastroenterologist", "Diarrhea": "Gastroenterologist",
    "Constipation": "Gastroenterologist", "Bloating": "Gastroenterologist",
    "Loss of appetite": "Gastroenterologist",
    "Blood in stool": "Gastroenterologist", "Heartburn": "Gastroenterologist",
    "Abdominal cramps": "Gastroenterologist",
    "Joint pain": "Orthopedic", "Muscle pain": "Orthopedic",
    "Swollen joints": "Orthopedic", "Numbness": "Neurologist",
    "Tingling": "Neurologist", "Weakness in legs": "Neurologist",
    "Weakness in arms": "Neurologist", "Back pain": "Orthopedic",
    "Neck pain": "Orthopedic", "Stiffness": "Orthopedic",
    "Difficulty walking": "Orthopedic",
    "Skin rash": "Dermatologist", "Itching": "Dermatologist",
    "Redness": "Dermatologist", "Swelling": "General Physician",
    "Blisters": "Dermatologist", "Bruising easily": "Hematologist",
    "Dry skin": "Dermatologist", "Skin discoloration": "Dermatologist",
    "Wound not healing": "Dermatologist", "Hair loss": "Dermatologist",
    "Fever": "General Physician", "Fatigue": "General Physician",
    "Weight loss": "Endocrinologist", "Weight gain": "Endocrinologist",
    "Night sweats": "General Physician", "Chills": "General Physician",
    "Swollen lymph nodes": "General Physician",
    "Frequent urination": "Urologist", "Painful urination": "Urologist",
    "Excessive thirst": "Endocrinologist",
    "Toothache": "Dentist", "Bleeding gums": "Dentist",
    "Mouth sores": "Dentist", "Bad breath": "Dentist",
    "Difficulty swallowing": "ENT Specialist",
    "Jaw stiffness": "Dentist", "Tooth sensitivity": "Dentist",
}


@app.post('/api/symptom-check')
def api_symptom_check():
    """
    POST /api/symptom-check
    Body JSON: { symptoms: [...], bodyArea: "...", duration: "...", severity: 1-10 }
    Returns care level, specialist recommendation, guidance text.
    Never diagnoses diseases or prescribes medicines.
    """
    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    body_area = data.get("bodyArea", "general")
    duration = data.get("duration", "1-3 days")
    severity = data.get("severity", 5)

    if not symptoms:
        return jsonify({"success": False, "error": "Please select at least one symptom"}), 400

    try:
        severity = int(severity)
    except (ValueError, TypeError):
        severity = 5
    severity = max(1, min(10, severity))

    # Check for emergency symptoms
    has_emergency = any(s in _EMERGENCY_SYMPTOMS for s in symptoms)

    # Determine specialist based on most specific symptom
    specialist_votes: dict[str, int] = {}
    for s in symptoms:
        spec = _SYMPTOM_SPECIALIST_MAP.get(s, "General Physician")
        specialist_votes[spec] = specialist_votes.get(spec, 0) + 1
    specialist = max(specialist_votes, key=specialist_votes.get) if specialist_votes else "General Physician"

    # Duration scoring
    duration_score = 0
    dur = duration.lower()
    if "hour" in dur or "just started" in dur:
        duration_score = 1
    elif "1-3 day" in dur or "1 day" in dur or "2 day" in dur:
        duration_score = 2
    elif "3-7 day" in dur or "week" in dur:
        duration_score = 3
    elif "1-2 week" in dur:
        duration_score = 4
    elif "2+ week" in dur or "month" in dur or "chronic" in dur:
        duration_score = 5

    # Triage logic
    total_score = severity + duration_score + (5 if has_emergency else 0) + (len(symptoms) - 1)

    if has_emergency or total_score >= 16 or severity >= 9:
        care_level = "emergency"
        urgency = "critical"
        guidance = (
            "Based on the symptoms you've described, we recommend seeking "
            "immediate emergency medical care. Please call emergency services "
            "or go to the nearest emergency department right away."
        )
    elif total_score >= 11 or severity >= 7:
        care_level = "specialist"
        urgency = "high"
        guidance = (
            f"Your symptoms suggest you should consult a {specialist} for "
            "specialized evaluation. Please schedule an appointment within "
            "the next 1-2 days."
        )
    elif total_score >= 6 or severity >= 4:
        care_level = "see-doctor"
        urgency = "moderate"
        guidance = (
            "We recommend seeing a doctor for proper evaluation. "
            f"A {specialist} would be appropriate for your symptoms. "
            "Schedule a visit within the next few days."
        )
    else:
        care_level = "self-care"
        urgency = "low"
        guidance = (
            "Based on the information provided, your symptoms appear mild. "
            "Monitor your condition, rest, and stay hydrated. If symptoms "
            "worsen or persist beyond 3 days, please consult a doctor."
        )
        specialist = "General Physician"

    return jsonify({
        "success": True,
        "careLevel": care_level,
        "urgency": urgency,
        "specialist": specialist,
        "guidance": guidance,
        "symptoms": symptoms,
        "bodyArea": body_area,
        "duration": duration,
        "severity": severity,
        "disclaimer": (
            "VitalSense AI provides informational guidance and is not a "
            "substitute for professional medical advice. In an emergency, "
            "seek immediate medical care."
        ),
    })


@app.get('/api/symptom-list')
def api_symptom_list():
    """GET /api/symptom-list?area=head — returns symptom options for a body area."""
    area = request.args.get('area', 'general')
    symptoms = _BODY_AREA_SYMPTOMS.get(area, _BODY_AREA_SYMPTOMS["general"])
    return jsonify({"success": True, "area": area, "symptoms": sorted(symptoms)})


# =========================================================================
# MODULE 2 — DIRECT ROUTES (Embedded — no proxy to port 5001)
# =========================================================================

@app.post('/module2/recommend')
def module2_recommend():
    """
    POST /module2/recommend
    Direct recommendation (embedded Module 2) with diet/precaution enrichment.
    Replaces the old proxy to port 5001.
    """
    t_start = time.perf_counter()
    body = request.get_json(silent=True) or {}

    disease = body.get('disease', '').strip()
    if not disease:
        return jsonify({'error': "'disease' field is required and must not be empty."}), 400

    symptoms: list[str] = body.get('symptoms', [])
    if not isinstance(symptoms, list):
        return jsonify({'error': "'symptoms' must be a JSON array of strings."}), 400

    # ── Severity calculation ──────────────────────────────────────────────
    severity_weights = m2_dataset_manager.get_severity_weights()
    if severity_weights:
        severity_result = m2_calculate_severity(symptoms, severity_weights)
    else:
        severity_result = {
            'severity': 'mild',
            'score': 0.0,
            'skipped_symptoms': symptoms,
        }

    severity_label: str = severity_result['severity']

    # ── Gemini API key (from header or environment variable) ──────────────
    auth_header = request.headers.get('Authorization', '')
    gemini_key  = auth_header.replace('Bearer ', '').strip() or None

    # ── Generate recommendation (CSV first, AI fallback) ──────────────────
    try:
        if m2_dataset_manager.is_any_dataset_loaded():
            rec = m2_recommender.generate_recommendation(
                disease=disease,
                severity=severity_label,
                symptoms=symptoms,
                gemini_api_key=gemini_key,
            )
        else:
            # No datasets loaded — use AI fallback directly
            try:
                rec = m2_ai_fallback.get_ai_recommendation(
                    disease=disease,
                    severity=severity_label,
                    api_key=gemini_key,
                )
            except EnvironmentError:
                return jsonify({
                    'error': 'No datasets uploaded and Gemini API key not configured. '
                             'Upload datasets via /module2/upload-datasets or set GEMINI_API_KEY.'
                }), 400
            except RuntimeError as exc:
                return jsonify({'error': f'AI recommendation failed: {exc}'}), 500
    except Exception as exc:
        logger.exception("Unexpected error in module2 recommend.")
        return jsonify({'error': f'Internal error while generating recommendation: {exc}'}), 500

    rec['severity_score']   = severity_result['score']
    rec['skipped_symptoms'] = severity_result['skipped_symptoms']

    # ── Enrich medications field ──────────────────────────────────────────
    meds = rec.get('medications')
    if isinstance(meds, list) and meds and not rec.get('medicine_category'):
        rec['medicine_category'] = ', '.join(meds)

    # ── Convert flat diet list → {eat, avoid} object ──────────────────────
    diet = rec.get('diet_recommendation')
    if isinstance(diet, list):
        disease_key = disease.strip().title()
        avoid = AVOID_FOODS.get(disease_key, [])
        rec['diet_recommendation'] = {'eat': diet, 'avoid': avoid}

    # ── Enrich precautions with AI / Clinical Precautions ─────────────────
    precautions = rec.get('precautions')
    if not precautions or len(precautions) <= 4 or any(len(str(p).split()) <= 3 for p in precautions):
        rec['precautions'] = clinical_precautions.get_clinical_precautions(
            disease=disease,
            symptoms=symptoms,
            severity=severity_label,
            api_key=gemini_key,
        )

    elapsed = round((time.perf_counter() - t_start) * 1000, 2)
    rec['processing_time_ms'] = elapsed

    logger.info("Module 2 recommend: disease='%s', severity='%s', time=%.2fms",
                disease, severity_label, elapsed)
    return jsonify(rec), 200


@app.post('/module2/ai-precautions')
def module2_ai_precautions():
    """Live AI Precautions generator endpoint (accepts optional API key)."""
    body = request.get_json(silent=True) or {}
    disease = body.get('disease', '').strip()
    symptoms = body.get('symptoms', [])
    severity = body.get('severity', 'mild')
    auth_header = request.headers.get('Authorization', '')
    api_key = body.get('api_key') or auth_header.replace('Bearer ', '').strip() or None

    precautions = clinical_precautions.get_clinical_precautions(
        disease=disease,
        symptoms=symptoms,
        severity=severity,
        api_key=api_key
    )
    return jsonify({
        'disease': disease,
        'precautions': precautions,
        'count': len(precautions),
        'source': 'gemini_ai' if api_key else 'clinical_knowledge_base'
    }), 200


@app.post('/module2/upload-datasets')
def module2_upload_datasets():
    """
    POST /module2/upload-datasets
    Accept up to 4 CSV files in one multipart/form-data request.
    Directly handles dataset loading (embedded Module 2).
    """
    t_start = time.perf_counter()
    results: dict[str, dict] = {}
    upload_count = 0

    for field_name, filename in _M2_DATASET_FILENAMES.items():
        if field_name not in request.files:
            continue

        file_obj = request.files[field_name]
        if file_obj.filename == "":
            results[field_name] = {"status": "skipped", "reason": "empty filename"}
            continue

        save_path = _M2_UPLOAD_FOLDER / filename
        try:
            file_obj.save(str(save_path))
        except OSError as exc:
            results[field_name] = {"status": "error", "reason": f"Could not save file: {exc}"}
            continue

        # Count rows
        try:
            with save_path.open("r", encoding="utf-8") as fh:
                row_count = max(0, sum(1 for _ in fh) - 1)
        except Exception:
            row_count = -1

        # Validate and load into cache
        try:
            if field_name == "precaution_file":
                m2_dataset_manager.load_precautions(save_path)
            elif field_name == "severity_file":
                m2_dataset_manager.load_severity_weights(save_path)
            elif field_name == "medicine_file":
                m2_dataset_manager.load_medicine_data(save_path)
            elif field_name == "diet_file":
                m2_dataset_manager.load_diet_data(save_path)

            results[field_name] = {"status": "ok", "saved_as": filename, "row_count": row_count}
            upload_count += 1
            logger.info("[Module 2] Uploaded & validated '%s' (%d rows).", filename, row_count)
        except (ValueError, FileNotFoundError) as exc:
            results[field_name] = {
                "status": "validation_error",
                "saved_as": filename,
                "row_count": row_count,
                "reason": str(exc),
            }
            logger.error("[Module 2] Validation failed for '%s': %s", filename, exc)

    elapsed = round((time.perf_counter() - t_start) * 1000, 2)

    if upload_count == 0 and not results:
        return jsonify({
            "error": f"No recognised file fields found. Expected one or more of: {list(_M2_DATASET_FILENAMES.keys())}"
        }), 400

    return jsonify({
        "message": f"{upload_count} dataset(s) uploaded and validated.",
        "processing_time_ms": elapsed,
        "files": results,
    }), 200


@app.get('/module2/diseases')
def module2_diseases():
    """Return the union of all disease names across every loaded dataset."""
    if not m2_dataset_manager.is_any_dataset_loaded():
        return jsonify({"error": "No datasets loaded yet. POST to /module2/upload-datasets first."}), 400
    diseases = m2_dataset_manager.get_all_diseases()
    return jsonify({"count": len(diseases), "diseases": diseases}), 200


@app.post('/module2/ai-recommend')
def module2_ai_recommend():
    """
    POST /module2/ai-recommend
    Always calls Gemini AI regardless of whether the disease is in the
    local datasets.  Works for ANY disease in the world.
    """
    t_start = time.perf_counter()
    body = request.get_json(silent=True) or {}

    disease = body.get('disease', '').strip()
    if not disease:
        return jsonify({'error': "'disease' field is required."}), 400

    symptoms: list[str] = body.get('symptoms', [])
    if not isinstance(symptoms, list):
        return jsonify({'error': "'symptoms' must be a JSON array of strings."}), 400

    # Severity: use provided value or auto-calculate
    forced_severity = body.get('severity', '').strip().lower()
    if forced_severity in ('mild', 'moderate', 'severe'):
        severity_label  = forced_severity
        severity_score  = 0.0
        skipped_syms: list[str] = []
    else:
        w = m2_dataset_manager.get_severity_weights()
        if w:
            sev = m2_calculate_severity(symptoms, w)
            severity_label  = sev['severity']
            severity_score  = sev['score']
            skipped_syms    = sev['skipped_symptoms']
        else:
            severity_label, severity_score, skipped_syms = 'moderate', 0.0, []

    # Gemini API key
    auth_header = request.headers.get('Authorization', '')
    gemini_key  = auth_header.replace('Bearer ', '').strip() or None

    try:
        rec = m2_ai_fallback.get_ai_recommendation(
            disease=disease,
            severity=severity_label,
            api_key=gemini_key,
        )
    except EnvironmentError:
        return jsonify({
            'error': 'Gemini API key not configured. Set GEMINI_API_KEY environment variable '
                     'or send it in the Authorization: Bearer <key> header.'
        }), 400
    except RuntimeError as exc:
        logger.exception("Gemini AI call failed.")
        return jsonify({'error': f'Gemini AI error: {exc}'}), 500

    rec['severity_score']   = severity_score
    rec['skipped_symptoms'] = skipped_syms

    elapsed = round((time.perf_counter() - t_start) * 1000, 2)
    rec['processing_time_ms'] = elapsed

    logger.info("Module 2 AI recommend: disease='%s', severity='%s', time=%.2fms",
                disease, severity_label, elapsed)
    return jsonify(rec), 200


@app.get('/module2/ai-status')
def module2_ai_status():
    """Check if Gemini AI fallback is available (API key set)."""
    available = m2_ai_fallback.is_gemini_available()
    return jsonify({
        'gemini_available': available,
        'model':            m2_ai_fallback.GEMINI_MODEL,
        'note': (
            'Gemini AI fallback is active for any disease not in local datasets.'
            if available else
            'Set GEMINI_API_KEY env variable to enable AI fallback for unknown diseases.'
        ),
    }), 200


# =========================================================================
# Auth & Prediction endpoints (Real SQLite Multi-User Auth)
# =========================================================================

@app.post("/auth/signup")
def auth_signup():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    pw    = data.get("password") or ""

    if not email or not name or not pw:
        return jsonify({"error": "All fields are required."}), 400

    user, error = register_user(name, email, pw)
    if error:
        status = 409 if "already registered" in error else 400
        return jsonify({"error": error}), status

    login_user(user, remember=True)
    token = _generate_token()
    return jsonify({"token": token, "user": user.to_dict()}), 201


@app.post("/auth/login")
def auth_login():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    pw    = data.get("password") or ""

    if not email or not pw:
        return jsonify({"error": "Email and password are required."}), 400

    user, error = authenticate_user(email, pw)
    if error:
        return jsonify({"error": error}), 401

    login_user(user, remember=True)
    token = _generate_token()
    return jsonify({"token": token, "user": user.to_dict()}), 200


@app.post("/auth/forgot-password")
def auth_forgot_password():
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required."}), 400

    # Always return the same status — never confirm whether the email exists
    return jsonify({"status": "reset_link_sent"}), 200


# ── Real prediction endpoints ────────────────────────────────────────────
@app.get("/dataset/symptoms")
def dataset_symptoms():
    # Serve the real symptom list from the trained model.
    display_symptoms = sorted(s.replace("_", " ") for s in SYMPTOM_LIST)
    return jsonify({"symptoms": display_symptoms}), 200


@app.post("/auth/set-gemini-key")
def set_gemini_key():
    """
    Accept a Gemini API key from the browser and inject it into the current
    process environment so the AI fallback can use it without a server restart.
    """
    data = request.get_json(silent=True) or {}
    key  = (data.get("key") or "").strip()
    if not key:
        return jsonify({"error": "No key provided."}), 400
    os.environ["GEMINI_API_KEY"] = key
    return jsonify({"ok": True, "message": "Gemini API key saved for this session."}), 200


@app.get("/auth/ai-status")
def ai_status():
    """Return whether the Gemini AI fallback is currently available."""
    from module1_ai_fallback import is_gemini_available
    return jsonify({"ai_available": is_gemini_available()}), 200


@app.post("/predict")
def predict_api():
    data          = request.get_json(silent=True) or {}
    raw_symptoms  = [str(s).strip().lower() for s in (data.get("symptoms") or [])]

    if not raw_symptoms:
        return jsonify({"error": "No symptoms provided."}), 400

    # Extract Gemini API key from Authorization header (Bearer token)
    auth_header   = request.headers.get("Authorization", "")
    gemini_key    = auth_header.replace("Bearer ", "").strip() or None

    try:
        result = predict_disease(raw_symptoms, model_dir=MODEL_DIR,
                                 gemini_api_key=gemini_key)
        return jsonify({
            "predicted_disease": result["disease"],
            "confidence":        result["confidence"],
            "warnings":          result["warnings"],
            "source":            result.get("source", "model"),
        }), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        err_str = str(exc)
        # Detect invalid API key errors from Gemini
        if "api key not valid" in err_str.lower() or \
           "invalid api key" in err_str.lower() or \
           "api_key_invalid" in err_str.lower() or \
           "400" in err_str:
            return jsonify({
                "error":      "Your Gemini API key is invalid. Please check it in AI Settings.",
                "error_code": "invalid_api_key",
            }), 401
        return jsonify({"error": "Prediction failed: " + err_str}), 500


# =========================================================================
# MODULE 3 — SUBPROCESS INFERENCE HELPER
# =========================================================================

def _run_subprocess_inference(image_bytes: bytes, report_type: str):
    """
    Run real TensorFlow inference by calling infer.py as a subprocess.
    The venv Python is NOT subject to the App Control DLL block that
    affects the parent Flask process. Returns a dict or None.
    """
    if not _VENV_PYTHON.exists() or not _INFER_SCRIPT.exists():
        return None

    suffix = ".png"
    tmp = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = subprocess.run(
            [str(_VENV_PYTHON), str(_INFER_SCRIPT), tmp_path, report_type],
            capture_output=True, text=True, timeout=120
        )
        stdout = result.stdout.strip()
        if result.returncode == 0 and stdout:
            # Find the last line that is valid JSON (ignore TF log noise)
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    logger.info("[INFER] %s: %s (conf=%.4f)",
                                report_type,
                                data.get('predicted_class'),
                                data.get('confidence', 0))
                    return data
        else:
            logger.warning("[INFER] Error (rc=%d): %s",
                           result.returncode, result.stderr[-300:])
    except subprocess.TimeoutExpired:
        logger.warning("[INFER] Timeout after 120s")
    except Exception as e:
        logger.warning("[INFER] Exception: %s", e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


# =========================================================================
# MODULE 3 — DIRECT ROUTES (Embedded — no proxy to port 5003)
# =========================================================================

@app.route("/module3/image/predict", methods=["POST"])
def module3_image_predict():
    """
    Real AI image analysis — tries in order:
      1. Subprocess inference (venv Python + TF, not DLL-blocked) — REAL predictions
      2. Hardcoded mock (last resort)
    No proxy to port 5003 needed.
    """
    # JS sends scan_type (not report_type) — read from both places
    report_type = (
        request.args.get("scan_type") or
        request.form.get("scan_type") or
        request.args.get("report_type") or
        request.form.get("report_type") or
        "xray"
    ).lower()

    # ── Step 1: Subprocess inference (real TF, child process not blocked) ──
    try:
        file_obj    = request.files.get("file")
        image_bytes = file_obj.read() if file_obj else None
    except Exception:
        image_bytes = None

    if image_bytes:
        result = _run_subprocess_inference(image_bytes, report_type)
        if result:
            return json.dumps(result), 200, {"Content-Type": "application/json"}

    # ── Step 2: Mock fallback ────────────────────────────────────────────
    logger.info("Image predict: using mock fallback for scan_type='%s'", report_type)
    return json.dumps(dict(_IMAGE_MOCKS.get(report_type, _IMAGE_MOCKS["xray"]))), 200, {"Content-Type": "application/json"}


@app.route("/module3/report-text/summarize", methods=["POST"])
def module3_report_text_summarize():
    """Direct radiology report-text summarisation (embedded Module 3)."""
    if not request.is_json:
        return jsonify({"error": "Request Content-Type must be 'application/json'."}), 400

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body is not valid JSON."}), 400

    report_text: str = body.get("report_text", "")
    if not isinstance(report_text, str):
        return jsonify({"error": "'report_text' must be a string."}), 400

    # Import the dataset-based report text analyzer (no AI/API needed)
    try:
        from report_text_analyzer import summarize_report_text
    except ImportError as exc:
        logger.error("Report text analyzer not available: %s", exc)
        return jsonify({
            "error": "Report text analyzer could not be loaded. "
                     "Please check that module3 dependencies are installed."
        }), 503

    try:
        result = summarize_report_text(report_text)
        # Privacy-safe: log length + severity, never the content
        logger.info("Report text analysed: text_length=%d chars, overall_severity='%s'.",
                     len(report_text.strip()), result.get("overall_severity"))
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        logger.error("Report analysis failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    except Exception:
        logger.error("Unexpected error in report-text summarize:\n%s", traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred while analysing your report. Please try again."}), 500


@app.route("/module3/report-text/upload-pdf", methods=["POST"])
def module3_report_text_upload_pdf():
    """
    Accept a PDF diagnostic report upload, extract text server-side using
    PyPDF2, and run the same summarize_report_text() analysis pipeline.

    This solves the browser limitation where PDF text extraction is not
    possible client-side.

    Request : multipart/form-data with field name "file" (PDF)
    Response: Same JSON shape as /module3/report-text/summarize
    """
    # ── Validate that a file was included ──────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Use field name 'file'."}), 400

    file_obj = request.files["file"]
    if file_obj.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename_lower = (file_obj.filename or "").lower()
    if not filename_lower.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted on this endpoint. "
                        "For images, use the image analysis endpoint instead."}), 400

    # ── Extract text from PDF using PyPDF2 ─────────────────────────────────
    try:
        import PyPDF2
    except ImportError:
        logger.error("PyPDF2 is not installed — cannot extract PDF text.")
        return jsonify({
            "error": "PDF text extraction library is not installed on the server. "
                     "Please install PyPDF2 (pip install PyPDF2)."
        }), 503

    try:
        pdf_bytes = io.BytesIO(file_obj.read())
        reader = PyPDF2.PdfReader(pdf_bytes)

        extracted_pages = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text.strip())

        report_text = "\n\n".join(extracted_pages).strip()
        logger.info(
            "PDF text extracted: filename='%s', pages=%d, text_length=%d chars.",
            file_obj.filename, len(reader.pages), len(report_text),
        )

    except Exception as exc:
        logger.error("Failed to read PDF file: %s", exc)
        return jsonify({
            "error": "Could not read the uploaded PDF file. It may be corrupted, "
                     "password-protected, or scanned (image-only). "
                     "Please paste your report text manually instead."
        }), 400

    # ── Validate extracted text ────────────────────────────────────────────
    if not report_text or len(report_text) < 20:
        return jsonify({
            "error": "Could not extract enough text from this PDF. "
                     "The file may be a scanned/image-only PDF. "
                     "Please paste your report text manually in the text area."
        }), 400

    # ── Run the same analysis pipeline ─────────────────────────────────────
    try:
        from report_text_analyzer import summarize_report_text
    except ImportError as exc:
        logger.error("Report text analyzer not available: %s", exc)
        return jsonify({
            "error": "Report text analyzer could not be loaded. "
                     "Please check that module3 dependencies are installed."
        }), 503

    try:
        result = summarize_report_text(report_text)
        logger.info(
            "PDF report analysed: filename='%s', overall_severity='%s'.",
            file_obj.filename, result.get("overall_severity"),
        )
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        logger.error("PDF report analysis failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    except Exception:
        logger.error("Unexpected error in PDF report analysis:\n%s", traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred while analysing your PDF report. "
                     "Please try again."
        }), 500


@app.route("/module3/analyze", methods=["POST"])
def module3_analyze():
    """Direct lab value analysis (embedded Module 3)."""
    if not m3_rd.is_loaded():
        return jsonify({
            "error": "No reference dataset is loaded. "
                     "Please upload a CSV via POST /module3/upload-reference first."
        }), 400

    if not request.is_json:
        return jsonify({"error": "Request Content-Type must be 'application/json'."}), 400

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body is not valid JSON."}), 400

    lab_values = body.get("lab_values")
    if lab_values is None:
        return jsonify({"error": "Missing required field 'lab_values' in request body."}), 400
    if not isinstance(lab_values, dict):
        return jsonify({"error": f"'lab_values' must be a JSON object, got {type(lab_values).__name__}."}), 400
    if not lab_values:
        return jsonify({"error": "'lab_values' is empty — provide at least one parameter."}), 400

    gender: str = str(body.get("gender", "Any")).strip()
    logger.info("Analyze request: %d parameters, gender='%s'", len(lab_values), gender)

    try:
        result = m3_analyze_report(lab_values=lab_values, gender=gender)
    except (RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.error("Unexpected error during analysis:\n%s", traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred during analysis."}), 500

    return jsonify(result), 200


@app.route("/module3/upload-reference", methods=["POST"])
def module3_upload_reference():
    """Upload and validate the reference-range CSV (embedded Module 3)."""
    if "file" not in request.files:
        return jsonify({
            "error": "No file part in the request. Send a multipart/form-data request with field 'file'."
        }), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "No file selected. Please attach a CSV file."}), 400

    if Path(file.filename).suffix.lower() != ".csv":
        return jsonify({
            "error": f"Invalid file type '{Path(file.filename).suffix}'. Only .csv files are accepted."
        }), 400

    save_path = _M3_UPLOAD_DIR / "reference_ranges.csv"
    try:
        file.save(str(save_path))
        logger.info("Uploaded CSV saved to: %s", save_path)
    except OSError as exc:
        return jsonify({"error": f"Could not save uploaded file: {exc}"}), 500

    try:
        ref_dict = m3_rd.load_reference_ranges(save_path)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.error("Unexpected error loading reference CSV:\n%s", traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred while processing the CSV."}), 500

    meta = m3_rd.get_cached_meta() or []
    return jsonify({
        "message": "Reference dataset loaded successfully.",
        "parameter_count": len(ref_dict),
        "parameters": [p["name"] for p in meta],
    }), 200


@app.route("/module3/parameters", methods=["GET"])
def module3_parameters():
    """Return all parameters + units from the cached reference dataset."""
    if not m3_rd.is_loaded():
        return jsonify({
            "error": "No reference dataset is loaded. "
                     "Please upload a CSV via POST /module3/upload-reference first."
        }), 400
    meta = m3_rd.get_cached_meta() or []
    return jsonify({"parameter_count": len(meta), "parameters": meta}), 200


@app.route("/module3/health", methods=["GET"])
def module3_health():
    """Health-check endpoint. Returns dataset and model status."""
    model_ready = (
        (_M3_MODEL_STORE_DIR / "image_model.keras").exists() or
        (_M3_MODEL_STORE_DIR / "image_model.h5").exists()
    )
    return jsonify({
        "status": "ok",
        "module": "module3 — Medical Report + Image Analysis (embedded in unified server)",
        "lab_dataset_loaded":        m3_rd.is_loaded(),
        "image_model_ready":         model_ready,
        "report_text_analyzer_ready": m3_rd.is_loaded(),  # dataset-based, no API key needed
    }), 200


# =========================================================================
# MODULE 3 — COMBINED FULL ANALYSIS
# =========================================================================

# Severity ranking for combined_overall_status
_STATUS_RANK: dict[str, int] = {
    "normal":           0,
    "attention_needed": 1,
    "urgent":           2,
}
_RANK_TO_STATUS: dict[int, str] = {v: k for k, v in _STATUS_RANK.items()}

# Pathological keywords for image → severity mapping
_IMG_URGENT_KEYWORDS = {
    "pneumonia", "tumor", "tumour", "cancer", "malignant",
    "cyst", "stone", "calcul", "nodule", "mass", "lesion",
    "fracture", "bleed", "haemorrhage", "hemorrhage",
    "effusion", "abscess", "infarct",
}

COMBINED_DISCLAIMER = (
    "This report combines multiple AI-generated preliminary assessments. "
    "None of these results are a diagnosis. "
    "Please consult a licensed healthcare professional to review all "
    "findings together."
)


@app.route("/module3/analyze-full", methods=["POST"])
def module3_analyze_full():
    """
    Combined full-analysis endpoint: run lab, image, and/or report-text
    analyses in a single HTTP call (embedded Module 3).
    Each sub-analysis runs independently; one failure does NOT block the others.

    Accepts multipart/form-data with any subset of:
      lab_values  (form field, JSON string)
      file        (file field, medical image)
      report_text (form field, raw radiology text)

    At least one must be present; otherwise 400.
    """
    has_lab   = "lab_values"  in request.form
    has_image = "file"        in request.files
    has_text  = "report_text" in request.form

    if not (has_lab or has_image or has_text):
        return jsonify({
            "error": "No input provided. Send at least one of: "
                     "'lab_values' (form field, JSON string), "
                     "'file' (file field), "
                     "'report_text' (form field, string).",
        }), 400

    logger.info("analyze-full request: lab=%s, image=%s, report_text=%s",
                has_lab, has_image, has_text)

    # Placeholders — None means "not requested"
    lab_result:  dict | None = None
    img_result:  dict | None = None
    text_result: dict | None = None

    # ── 1. Lab-value analysis ─────────────────────────────────────────────
    if has_lab:
        raw_lab = request.form["lab_values"].strip()
        try:
            lab_values_dict = json.loads(raw_lab) if raw_lab else {}
            if not isinstance(lab_values_dict, dict):
                raise ValueError("'lab_values' must be a JSON object.")
            if not lab_values_dict:
                raise ValueError("'lab_values' JSON object is empty.")

            gender = request.form.get("gender", "Any").strip()
            lab_result = m3_analyze_report(lab_values=lab_values_dict, gender=gender)

            logger.info("analyze-full / lab: overall_status='%s', params=%d.",
                        lab_result.get("overall_status"), len(lab_values_dict))

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("analyze-full / lab failed (input error): %s", exc)
            lab_result = {"error": str(exc)}
        except RuntimeError as exc:
            logger.warning("analyze-full / lab failed (runtime): %s", exc)
            lab_result = {"error": str(exc)}
        except Exception:
            logger.error("analyze-full / lab unexpected error:\n%s", traceback.format_exc())
            lab_result = {"error": "An unexpected error occurred during lab analysis."}

    # ── 2. Image analysis (subprocess inference) ──────────────────────────
    if has_image:
        image_file = request.files["file"]
        img_filename = image_file.filename or "upload"
        img_ext = Path(img_filename).suffix.lower()

        # Validate extension
        if img_ext not in _M3_ALLOWED_IMAGE_EXTS and img_ext != _M3_ALLOWED_DICOM_EXT:
            img_result = {
                "error": f"Unsupported image type '{img_ext}'. "
                         f"Accepted: {sorted(_M3_ALLOWED_IMAGE_EXTS | {_M3_ALLOWED_DICOM_EXT})}"
            }
        else:
            file_bytes = image_file.read()
            scan_type = request.form.get("scan_type", "xray").lower().strip()

            # Try real inference via subprocess
            result = _run_subprocess_inference(file_bytes, scan_type)
            if result:
                img_result = result
                logger.info("analyze-full / image: class='%s', conf=%.4f.",
                            img_result.get("predicted_class"),
                            img_result.get("confidence", 0.0))
            else:
                # Mock fallback
                img_result = dict(_IMAGE_MOCKS.get(scan_type, _IMAGE_MOCKS["xray"]))
                logger.info("analyze-full / image: using mock for scan_type='%s'.", scan_type)

    # ── 3. Diagnostic report text analysis (dataset-based, no AI) ───────────
    if has_text:
        try:
            from report_text_analyzer import summarize_report_text
        except ImportError as exc:
            logger.warning("analyze-full / report-text: deps missing — %s", exc)
            text_result = {
                "error": f"Report text analyzer could not be loaded: {exc}."
            }

        if text_result is None:  # import succeeded
            raw_text = request.form["report_text"]
            try:
                text_result = summarize_report_text(raw_text)
                logger.info("analyze-full / report-text: text_length=%d, severity='%s'.",
                            len(raw_text.strip()), text_result.get("overall_severity"))
            except ValueError as exc:
                logger.warning("analyze-full / report-text input error: %s", exc)
                text_result = {"error": str(exc)}
            except RuntimeError as exc:
                logger.warning("analyze-full / report-text analysis error: %s", exc)
                text_result = {"error": str(exc)}
            except Exception:
                logger.error("analyze-full / report-text unexpected error:\n%s",
                             traceback.format_exc())
                text_result = {"error": "An unexpected error occurred during report text analysis."}

    # ── 4. Compute combined_overall_status ────────────────────────────────
    sub_statuses: list[str] = []

    # -- Lab severity
    if lab_result and "error" not in lab_result:
        lab_status = lab_result.get("overall_status")
        if lab_status and lab_status in _STATUS_RANK:
            sub_statuses.append(lab_status)

    # -- Image severity (derived)
    if img_result is not None and "error" not in img_result:
        if img_result.get("low_confidence_warning"):
            sub_statuses.append("attention_needed")
        else:
            predicted_class = str(img_result.get("predicted_class", "")).lower()
            if "normal" in predicted_class or "no finding" in predicted_class:
                sub_statuses.append("normal")
            elif any(kw in predicted_class for kw in _IMG_URGENT_KEYWORDS):
                sub_statuses.append("urgent")
            else:
                sub_statuses.append("attention_needed")

    # -- Report-text severity
    if text_result and "error" not in text_result:
        text_status = text_result.get("overall_severity")
        if text_status and text_status in _STATUS_RANK:
            sub_statuses.append(text_status)

    if sub_statuses:
        worst_rank = max(_STATUS_RANK.get(s, 0) for s in sub_statuses)
        combined_status = _RANK_TO_STATUS[worst_rank]
    else:
        # Every attempted sub-analysis failed — flag conservatively
        combined_status = "attention_needed"

    # ── 5. Assemble response ──────────────────────────────────────────────
    logger.info(
        "analyze-full complete: lab=%s, image=%s, text=%s, combined='%s'.",
        "ok" if (lab_result  and "error" not in lab_result)  else ("err" if lab_result  else "skip"),
        "ok" if (img_result  and "error" not in img_result)  else ("err" if img_result  else "skip"),
        "ok" if (text_result and "error" not in text_result) else ("err" if text_result else "skip"),
        combined_status,
    )

    return jsonify({
        "lab_analysis":          lab_result,
        "image_analysis":        img_result,
        "report_text_analysis":  text_result,
        "combined_overall_status": combined_status,
        "assistive_disclaimer":  COMBINED_DISCLAIMER,
    }), 200


# =========================================================================
# MODULE 5 — MEDICINE REMINDER (Embedded)
# =========================================================================

# Initialise database on import
m5_db.init_db()
logger.info("[Module 5] Medicine Reminder DB ready.")

# ── Page route ────────────────────────────────────────────────────────────
@app.route('/medicine-reminder')
def medicine_reminder_page():
    """Medicine Reminder dashboard — Module 5."""
    return render_template('medicine_reminder.html')


# ── Helper: get user_id from request ──────────────────────────────────────
def _m5_user_id() -> str:
    """Extract user_id from query string, JSON body, or default."""
    if request.is_json:
        body = request.get_json(silent=True) or {}
        uid = body.get('user_id', '')
        if uid:
            return uid
    uid = request.args.get('user_id', '')
    if uid:
        return uid
    # Fall back to current mock user
    return _current_user.get('id', 'u_001')


# ── GET /api/medicine/list ────────────────────────────────────────────────
@app.get('/api/medicine/list')
def api_medicine_list():
    """Return all medicines for the current user."""
    user_id = _m5_user_id()
    medicines = m5_db.get_medicines(user_id)
    return jsonify({'medicines': medicines, 'count': len(medicines)}), 200


# ── POST /api/medicine/add ────────────────────────────────────────────────
@app.post('/api/medicine/add')
def api_medicine_add():
    """Add a new prescribed medicine."""
    body = request.get_json(silent=True) or {}
    user_id = body.get('user_id') or _m5_user_id()

    required = ['name', 'dosage', 'frequency', 'reminder_times', 'start_date']
    missing = [f for f in required if not body.get(f)]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    med = m5_db.add_medicine(
        user_id=user_id,
        name=body['name'],
        dosage=body['dosage'],
        frequency=body['frequency'],
        reminder_times=body['reminder_times'],
        start_date=body['start_date'],
        end_date=body.get('end_date', ''),
        food_instruction=body.get('food_instruction', 'before'),
        notes=body.get('notes', ''),
    )
    return jsonify({'message': 'Medicine added.', 'medicine': med}), 201


# ── PUT /api/medicine/update/<id> ─────────────────────────────────────────
@app.put('/api/medicine/update/<int:med_id>')
def api_medicine_update(med_id: int):
    """Update an existing medicine."""
    body = request.get_json(silent=True) or {}
    med = m5_db.update_medicine(med_id, **body)
    if not med:
        return jsonify({'error': 'Medicine not found.'}), 404
    return jsonify({'message': 'Medicine updated.', 'medicine': med}), 200


# ── DELETE /api/medicine/delete/<id> ──────────────────────────────────────
@app.delete('/api/medicine/delete/<int:med_id>')
def api_medicine_delete(med_id: int):
    """Delete a medicine."""
    deleted = m5_db.delete_medicine(med_id)
    if not deleted:
        return jsonify({'error': 'Medicine not found.'}), 404
    return jsonify({'message': 'Medicine deleted.'}), 200


# ── POST /api/medicine/toggle/<id> ────────────────────────────────────────
@app.post('/api/medicine/toggle/<int:med_id>')
def api_medicine_toggle(med_id: int):
    """Enable or disable a medicine."""
    med = m5_db.toggle_medicine(med_id)
    if not med:
        return jsonify({'error': 'Medicine not found.'}), 404
    state = 'enabled' if med['is_enabled'] else 'disabled'
    return jsonify({'message': f'Medicine {state}.', 'medicine': med}), 200


# ── GET /api/medicine/schedule/today ──────────────────────────────────────
@app.get('/api/medicine/schedule/today')
def api_medicine_schedule_today():
    """Get today's schedule (auto-generates if needed)."""
    user_id = _m5_user_id()
    schedule = m5_db.generate_today_schedule(user_id)
    return jsonify({'schedule': schedule, 'count': len(schedule)}), 200


# ── POST /api/medicine/schedule/action ────────────────────────────────────
@app.post('/api/medicine/schedule/action')
def api_medicine_schedule_action():
    """Mark a schedule entry as taken / missed / snoozed."""
    body = request.get_json(silent=True) or {}
    schedule_id = body.get('schedule_id')
    action = body.get('action', '').lower()

    if not schedule_id:
        return jsonify({'error': 'schedule_id is required.'}), 400
    if action not in ('taken', 'missed', 'snoozed'):
        return jsonify({'error': 'action must be: taken, missed, or snoozed.'}), 400

    result = m5_db.mark_schedule(int(schedule_id), action, body.get('notes', ''))
    if not result:
        return jsonify({'error': 'Schedule entry not found.'}), 404
    return jsonify({'message': f'Marked as {action}.', 'result': result}), 200


# ── GET /api/medicine/dashboard ───────────────────────────────────────────
@app.get('/api/medicine/dashboard')
def api_medicine_dashboard():
    """Dashboard stats: total, taken, missed, pending, adherence %."""
    user_id = _m5_user_id()
    summary = m5_db.get_today_summary(user_id)
    return jsonify(summary), 200


# ── GET /api/medicine/history ─────────────────────────────────────────────
@app.get('/api/medicine/history')
def api_medicine_history():
    """Medicine history with optional date range filter."""
    user_id = _m5_user_id()
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    history = m5_db.get_history(user_id, start_date, end_date)
    return jsonify({'history': history, 'count': len(history)}), 200


# ── GET /api/medicine/reminders/poll ──────────────────────────────────────
@app.get('/api/medicine/reminders/poll')
def api_medicine_poll():
    """Poll for currently-due medicine reminders (for browser notifications)."""
    user_id = _m5_user_id()
    # Also auto-miss expired schedules
    m5_db.auto_miss_expired(user_id)
    reminders = m5_db.get_due_reminders(user_id)
    return jsonify({'reminders': reminders, 'count': len(reminders)}), 200


# ── GET /api/medicine/context ─────────────────────────────────────────────
@app.get('/api/medicine/context')
def api_medicine_context():
    """
    Read-only medicine context for the AI chatbot.
    Returns a plain-text summary of today's medicines.
    The AI must NEVER prescribe medicines or change dosages.
    """
    user_id = _m5_user_id()
    context = m5_db.get_medicine_context(user_id)
    return jsonify({'context': context}), 200


# =========================================================================
# Module 4 — VitalSense Vitals Monitor (EMBEDDED)
# =========================================================================

@app.post('/vitals/sync')
def m4_sync_vitals():
    """
    POST /vitals/sync
    Accept a vitals reading from the Bluetooth dashboard, validate, save, and check alerts.
    """
    logger.info('POST /vitals/sync from %s', request.remote_addr)
    try:
        data = request.get_json(force=True, silent=True) or {}
        reading = M4_VitalsReading(
            user_id=data.get('user_id', ''),
            device_id=data.get('device_id', ''),
            heart_rate_bpm=data.get('heart_rate_bpm'),
            spo2_percent=data.get('spo2_percent'),
            recorded_at=data.get('recorded_at', ''),
        )
        errors = m4_validate(reading)
        if errors:
            return jsonify({'status': 'error', 'message': 'Validation failed.', 'errors': errors}), 400

        reading_dict = _m4_asdict(reading)
        row_id = m4_save(reading_dict)
        alert_result = m4_check_alerts(reading_dict)

        return jsonify({
            'status': 'saved',
            'reading_id': row_id,
            'alerts': alert_result['alerts'],
            'requires_attention': alert_result['requires_attention'],
        }), 200
    except Exception as exc:
        logger.exception('Unexpected error in /vitals/sync: %s', exc)
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500


@app.get('/vitals/latest/<user_id>')
def m4_latest_vitals(user_id: str):
    """GET /vitals/latest/<user_id> — most recent reading for a user."""
    try:
        result = m4_latest(user_id)
        if result is None:
            return jsonify({'status': 'not_found', 'data': None}), 200
        return jsonify({'status': 'ok', 'data': result}), 200
    except Exception as exc:
        logger.exception('Error fetching latest vitals for %s', user_id)
        return jsonify({'status': 'error', 'message': 'Internal error.'}), 500


@app.get('/vitals/history/<user_id>')
def m4_vitals_history(user_id: str):
    """GET /vitals/history/<user_id>?start=...&end=... — readings in a time range."""
    try:
        start = request.args.get('start')
        end = request.args.get('end')
        if not start or not end:
            return jsonify({'status': 'error', 'message': "Both 'start' and 'end' are required."}), 400
        readings = m4_history(user_id, start, end)
        return jsonify({'status': 'ok', 'count': len(readings), 'data': readings}), 200
    except Exception as exc:
        logger.exception('Error fetching vitals history for %s', user_id)
        return jsonify({'status': 'error', 'message': 'Internal error.'}), 500


@app.get('/vitals/daily-average/<user_id>/<date>')
def m4_daily_average(user_id: str, date: str):
    """GET /vitals/daily-average/<user_id>/<date> — average HR/SpO2 for a day."""
    try:
        result = m4_daily_avg(user_id, date)
        return jsonify({'status': 'ok', 'data': result}), 200
    except Exception as exc:
        logger.exception('Error computing daily average for %s/%s', user_id, date)
        return jsonify({'status': 'error', 'message': 'Internal error.'}), 500


@app.get('/vitals-monitor')
def vitals_monitor_page():
    """Serve the live Bluetooth vitals monitoring dashboard."""
    return render_template('vitals_monitor.html')


# =========================================================================
# AI Assistant — Chat API (Local Medical Knowledge Base)
# =========================================================================

from medical_kb import find_best_response


@app.get('/api/ai-chat/status')
def api_ai_chat_status():
    """AI chat is always available — uses local knowledge base, no API key needed."""
    return jsonify({'available': True}), 200


@app.post('/api/ai-chat')
def api_ai_chat():
    """
    POST /api/ai-chat
    Accept a user message, match against the local medical knowledge base, return reply.

    Request  : {"message": "...", "history": [...]}
    Response : {"reply": "...", "suggestions": [...]}
    """
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()

    if not message:
        return jsonify({'error': 'No message provided.'}), 400

    # ── Fetch medicine context (for personalized answers) ────────────────
    medicine_context = ''
    try:
        user_id = _m5_user_id()
        med_ctx = m5_db.get_medicine_context(user_id)
        if med_ctx and med_ctx.strip():
            medicine_context = med_ctx
    except Exception:
        pass  # Non-critical — continue without medicine context

    # ── Fetch health profile context ─────────────────────────────────────
    profile_context = ''
    try:
        user_id = _m5_user_id()
        profile_ctx = m6_db.get_profile_context(user_id)
        if profile_ctx and profile_ctx.strip():
            profile_context = profile_ctx
    except Exception:
        pass  # Non-critical — continue without profile context

    # ── Match against local knowledge base ───────────────────────────────
    try:
        result = find_best_response(message, medicine_context=medicine_context, profile_context=profile_context)
        reply_text = result['reply']
        suggestions = result['suggestions']

        logger.info(
            '[AI Chat] User: %.80s… → KB matched=%s',
            message, result.get('matched', False),
        )

    except Exception as exc:
        logger.exception('[AI Chat] Knowledge base lookup failed')
        return jsonify({
            'error': f'An error occurred while processing your question: {str(exc)}'
        }), 500

    # ── Generate follow-up suggestions (merge KB + contextual) ───────────
    if not suggestions:
        suggestions = _generate_suggestions(message, reply_text)

    return jsonify({
        'reply': reply_text,
        'suggestions': suggestions[:3]
    }), 200


def _generate_suggestions(user_msg: str, ai_reply: str) -> list[str]:
    """Generate contextual follow-up question suggestions based on the conversation."""
    suggestions = []
    lower_reply = ai_reply.lower()
    lower_msg = user_msg.lower()

    # Symptom-related suggestions
    if any(w in lower_msg for w in ['symptom', 'pain', 'ache', 'feel', 'hurt']):
        suggestions.append('What home remedies can help?')
        suggestions.append('When should I see a doctor?')

    # Medicine-related suggestions
    if any(w in lower_msg for w in ['medicine', 'medication', 'drug', 'tablet', 'pill']):
        suggestions.append('What are the side effects?')
        suggestions.append('Show my medicine schedule')

    # Report/test-related suggestions
    if any(w in lower_msg for w in ['report', 'test', 'result', 'blood', 'scan']):
        suggestions.append('Is this value normal?')
        suggestions.append('What should I do next?')

    # Condition-related suggestions
    if any(w in lower_reply for w in ['diabetes', 'hypertension', 'cholesterol', 'thyroid']):
        suggestions.append('What diet changes help?')
        suggestions.append('What precautions should I take?')

    # General fallback
    if not suggestions:
        suggestions = [
            'Tell me more about this',
            'What precautions should I take?',
            'Should I consult a specialist?'
        ]

    return suggestions[:3]  # Max 3 suggestions


# =========================================================================
# MODULE 6 — PERSONAL HEALTH PROFILE (Embedded)
# =========================================================================

# Initialise database on import
m6_db.init_db()
logger.info("[Module 6] Health Profile DB ready.")


# ── GET /api/health-profile ───────────────────────────────────────────────
@app.get('/api/health-profile')
def api_health_profile_get():
    """Return the saved health profile for the current user."""
    user_id = _m5_user_id()
    profile = m6_db.get_profile(user_id)
    if profile is None:
        return jsonify({'profile': None}), 200
    return jsonify({'profile': profile}), 200


# ── POST /api/health-profile ──────────────────────────────────────────────
@app.post('/api/health-profile')
def api_health_profile_save():
    """Save or update the health profile for the current user."""
    body = request.get_json(silent=True) or {}
    user_id = _m5_user_id()

    try:
        profile = m6_db.save_profile(user_id, body)
        logger.info('[Module 6] Profile saved for user %s (completion=%d%%)',
                    user_id, profile.get('profile_completion', 0))
        return jsonify({'profile': profile, 'message': 'Profile saved successfully.'}), 200
    except Exception as exc:
        logger.exception('[Module 6] Error saving profile')
        return jsonify({'error': f'Could not save profile: {exc}'}), 500


# ── GET /api/health-profile/emergency ─────────────────────────────────────
@app.get('/api/health-profile/emergency')
def api_health_profile_emergency():
    """Return emergency-critical fields only for quick access."""
    user_id = _m5_user_id()
    info = m6_db.get_emergency_info(user_id)
    if info is None:
        return jsonify({'emergency': None}), 200
    return jsonify({'emergency': info}), 200


# ── GET /api/health-profile/summary ───────────────────────────────────────
@app.get('/api/health-profile/summary')
def api_health_profile_summary():
    """Return smart health summary (BMI category, key info, no diagnosis)."""
    user_id = _m5_user_id()
    summary = m6_db.get_profile_summary(user_id)
    if summary is None:
        return jsonify({'summary': None}), 200
    return jsonify({'summary': summary}), 200


# ── GET /api/health-profile/context ───────────────────────────────────────
@app.get('/api/health-profile/context')
def api_health_profile_context():
    """Plain-text health profile context for AI assistant integration."""
    user_id = _m5_user_id()
    context = m6_db.get_profile_context(user_id)
    return jsonify({'context': context}), 200


# =========================================================================
# Module 7 — Natural & Herbal Remedies API Routes
# =========================================================================

@app.get('/api/herbal/remedies')
def api_herbal_remedies():
    """GET /api/herbal/remedies?disease=...&severity=mild
    Get herbal remedies for a disease."""
    disease = request.args.get('disease', '').strip()
    severity = request.args.get('severity', 'mild').strip()
    if not disease:
        return jsonify({'error': 'disease parameter is required'}), 400

    result = m7_kb.get_herbal_remedies(disease, severity)
    logger.info('[Module 7] Herbal lookup for "%s" (severity=%s) → %d remedies',
                disease, severity, len(result.get('remedies', [])))
    return jsonify(result), 200


@app.get('/api/herbal/herbs')
def api_herbal_herbs():
    """GET /api/herbal/herbs — List all herbs in the database."""
    herbs = m7_kb.get_all_herbs()
    return jsonify({'herbs': herbs, 'count': len(herbs)}), 200


@app.get('/api/herbal/herb/<name>')
def api_herbal_herb_detail(name):
    """GET /api/herbal/herb/<name> — Get detailed herb profile."""
    profile = m7_kb.get_herb_profile(name)
    if profile is None:
        return jsonify({'error': f'Herb "{name}" not found'}), 404
    return jsonify({'herb': profile}), 200


@app.get('/api/herbal/diseases')
def api_herbal_diseases():
    """GET /api/herbal/diseases — List all diseases with herbal remedies."""
    diseases = m7_kb.get_all_diseases()
    return jsonify({'diseases': diseases, 'count': len(diseases)}), 200


@app.post('/api/herbal/ai-suggest')
def api_herbal_ai_suggest():
    """POST /api/herbal/ai-suggest — AI-powered herbal suggestion (Gemini)."""
    body = request.get_json(force=True, silent=True) or {}
    disease = body.get('disease', '').strip()
    if not disease:
        return jsonify({'error': 'disease field is required'}), 400

    if not m7_ai.is_available():
        return jsonify({'error': 'AI service is not available. Set GEMINI_API_KEY in .env to enable.'}), 503

    result = m7_ai.suggest_herbal_remedies(disease, body.get('symptoms'))
    if result is None:
        return jsonify({'error': 'AI suggestion failed. Please try again.'}), 500

    if result.get('serious_condition'):
        return jsonify({
            'disease': disease,
            'remedies': [],
            'severity_warning': f'"{disease}" is a serious condition. Herbal remedies are not appropriate. Consult a doctor immediately.',
            'doctor_referral': True,
            'disclaimer': m7_kb.DISCLAIMER,
        }), 200

    logger.info('[Module 7] AI herbal suggestion for "%s" → %d remedies',
                disease, len(result.get('remedies', [])))
    return jsonify(result), 200


# =========================================================================
# Entry point
# =========================================================================
if __name__ == "__main__":
    # Start Module 5 background scheduler
    m5_scheduler.start_scheduler()

    print("\n  VitalSense AI - Unified Server (All Modules Embedded)")
    print("  " + "=" * 55)
    print("  Home      -> http://localhost:5050/home")
    print("  Sign Up   -> http://localhost:5050/signup")
    print("  Log In    -> http://localhost:5050/login")
    print("  Predict   -> http://localhost:5050/predict")
    print("  Module 2  -> http://localhost:5050/module2")
    print("  X-Ray     -> http://localhost:5050/xray")
    print("  Medicine  -> http://localhost:5050/medicine-reminder")
    print("  Vitals    -> http://localhost:5050/vitals-monitor")
    print("  Assistant -> http://localhost:5050/assistant")
    print("  Profile   -> http://localhost:5050/health-profile")
    print("  Herbal    -> http://localhost:5050/herbal-remedies")
    print("  " + "-" * 55)
    print("  Module 2 API -> EMBEDDED (no separate server needed)")
    print("  Module 3 API -> EMBEDDED (no separate server needed)")
    print("  Module 4 API -> EMBEDDED (Vitals Monitor)")
    print("  Module 5 API -> EMBEDDED (Medicine Reminder)")
    print("  Module 6 API -> EMBEDDED (Health Profile)")
    print("  Module 7 API -> EMBEDDED (Natural & Herbal Remedies)")
    print("  AI Chat API  -> EMBEDDED (Local Medical KB — No API Key Needed)")
    print()
    app.run(host="0.0.0.0", port=5050, debug=True)
