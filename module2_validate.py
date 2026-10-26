"""
validate.py
===========
Standalone end-to-end validation for Module 2 using the new uploaded datasets.
Run from the module2/ directory: python validate.py
No Flask server required.

New dataset files (in data/):
  precautions_df.csv    -> Disease + Precaution_1..4 (with leading index col)
  Symptom-severity.csv  -> Symptom, weight (134 symptoms)
  medications.csv       -> Disease, Medication  (Python list string)
  diets.csv             -> Disease, Diet        (Python list string)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 65)
print("VitalSense AI -- Module 2 Validation (new datasets)")
print("=" * 65)

# ============================================================
# 1. dataset_manager
# ============================================================
print("\n[1] Loading new datasets...")
import dataset_manager as dm

DATA_DIR = Path(__file__).parent / "data"
p = dm.load_precautions(DATA_DIR / "precautions_df.csv")
print(f"  Precautions loaded    : {len(p)} diseases")
assert "Drug Reaction" in p, "Drug Reaction not found"
assert len(p["Drug Reaction"]) >= 3, "Expected at least 3 precautions"
print(f"    Sample [{list(p.keys())[0]}]: {p[list(p.keys())[0]]}")

s = dm.load_severity_weights(DATA_DIR / "Symptom-severity.csv")
print(f"  Severity weights      : {len(s)} symptoms")
assert "itching" in s and s["itching"] == 1.0
assert "high_fever" in s and s["high_fever"] == 7.0

m = dm.load_medicine_data(DATA_DIR / "medications.csv")
print(f"  Medicine data         : {len(m)} diseases")
assert "Diabetes" in m
assert isinstance(m["Diabetes"]["medications"], list) and len(m["Diabetes"]["medications"]) > 0
assert "specialist_type" in m["Diabetes"]
print(f"    Diabetes specialist : {m['Diabetes']['specialist_type']}")
print(f"    Diabetes meds (first 2): {m['Diabetes']['medications'][:2]}")

d = dm.load_diet_data(DATA_DIR / "diets.csv")
print(f"  Diet data             : {len(d)} diseases")
assert "Diabetes" in d
assert isinstance(d["Diabetes"], list) and len(d["Diabetes"]) > 0
print(f"    Diabetes diet (first 2): {d['Diabetes'][:2]}")

diseases = dm.get_all_diseases()
print(f"  Total unique diseases : {len(diseases)}")
print(f"  Sample                : {diseases[:5]}")

# ============================================================
# 2. severity_calculator — new symptom names from Symptom-severity.csv
# ============================================================
print("\n[2] Severity calculator (using new symptom names)...")
from severity_calculator import calculate_severity

# Mild (itching=1, skin_rash=3, nodal_skin_eruptions=4 => 8 <= 12)
r_mild = calculate_severity(["itching", "skin_rash", "nodal_skin_eruptions"], s)
print(f"  Mild case    : score={r_mild['score']}, severity={r_mild['severity']!r}")
assert r_mild["severity"] == "mild", f"Expected mild, got {r_mild['severity']}"

# Moderate (fatigue=4, vomiting=5, nausea=5, headache=3 => 17, 12 < 17 <= 25)
r_mod = calculate_severity(["fatigue", "vomiting", "nausea", "headache"], s)
print(f"  Moderate case: score={r_mod['score']}, severity={r_mod['severity']!r}")
assert r_mod["severity"] == "moderate", f"Expected moderate, got {r_mod['severity']}"

# Severe (chest_pain=7, high_fever=7, breathlessness=4, diarrhoea=6 => 24+? let's check)
r_sev = calculate_severity(["chest_pain", "high_fever", "swelling_of_stomach", "malaise"], s)
print(f"  Severe case  : score={r_sev['score']}, severity={r_sev['severity']!r}")
assert r_sev["severity"] == "severe", f"Expected severe, got {r_sev['severity']}"

# Skipped symptom
r_skip = calculate_severity(["itching", "unknown_xyz_symptom"], s)
print(f"  Skipped      : {r_skip['skipped_symptoms']}")
assert "unknown_xyz_symptom" in r_skip["skipped_symptoms"]

# ============================================================
# 3. recommend.py
# ============================================================
print("\n[3] Recommendation engine...")
import recommend

# Mild -> medications returned, no referral
rec_mild = recommend.generate_recommendation(
    "Common Cold", "mild", ["runny_nose", "cough"]
)
assert rec_mild["medications"] is not None, "Mild: medications should not be None"
assert rec_mild["doctor_referral"] is None, "Mild: doctor_referral should be None"
assert rec_mild["diet_recommendation"] is not None
assert isinstance(rec_mild["diet_recommendation"], list)
assert rec_mild["precautions"] is not None
assert rec_mild["disclaimer"]
print(f"  Mild   [OK]  meds (first): {rec_mild['medications'][0]}")
print(f"  Mild   [OK]  diet (first): {rec_mild['diet_recommendation'][0]}")

# Moderate -> referral same-day, no medications
rec_mod = recommend.generate_recommendation(
    "Tuberculosis", "moderate", ["fatigue", "cough"]
)
assert rec_mod["medications"] is None, "Moderate: medications should be None"
assert rec_mod["doctor_referral"] is not None
assert rec_mod["doctor_referral"]["urgency"] == "same-day"
print(f"  Moderate [OK]  specialist='{rec_mod['doctor_referral']['specialist']}', urgency=same-day")

# Severe -> emergency referral
rec_sev = recommend.generate_recommendation(
    "Heart Attack", "severe", ["chest_pain", "breathlessness"]
)
assert rec_sev["doctor_referral"]["urgency"] == "emergency"
print(f"  Severe [OK]  specialist='{rec_sev['doctor_referral']['specialist']}', urgency=emergency")

# Unknown disease -> graceful nulls, no crash
rec_unk = recommend.generate_recommendation("Alien Syndrome", "mild", [])
assert rec_unk["medications"] is None
assert rec_unk["precautions"] is None
assert rec_unk["diet_recommendation"] is None
print("  Unknown [OK]  graceful nulls, no crash")

# ============================================================
# 4. Reload
# ============================================================
reload_result = dm.reload()
print(f"\n[4] Reload: {reload_result}")
assert all(v == "ok" for v in reload_result.values()), f"Reload issue: {reload_result}"

print("\n" + "=" * 65)
print("ALL CHECKS PASSED")
print("=" * 65)
