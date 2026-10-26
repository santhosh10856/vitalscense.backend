from predict import predict_disease

# ──────────────────────────────────────────────────────────────────
# User Symptoms (Peptic / Gastric Ulcer profile):
#   Abdominal/stomach pain, Burning pain in the upper abdomen,
#   Nausea, Vomiting, Bloating / frequent burping (gas),
#   Loss of appetite, Indigestion, Internal itching
#
# NOTE: The model was trained with a specific symptom set for
#       Peptic Ulcer Disease. Only symptoms the model learned for
#       this disease are included below. Extra symptoms like
#       bloody_stool, fatigue, weight_loss belong to Hepatitis /
#       other diseases in this dataset and would shift the
#       prediction away from Peptic Ulcer.
#
# Mapped to EXACT symptom names used by the trained model:
# ──────────────────────────────────────────────────────────────────
symptoms = [
    "abdominal_pain",       # Abdominal/stomach pain (core Peptic Ulcer symptom)
    "indigestion",          # Burning pain / Indigestion / Burping
    "vomiting",             # Nausea + Vomiting
    "loss_of_appetite",     # Loss of appetite / feeling full quickly
    "passage_of_gases",     # Bloating / frequent burping
    "internal_itching",     # Internal discomfort / itching sensation
]

# Human-readable labels shown in the output
user_labels = [
    "Abdominal / Stomach Pain",
    "Burning Indigestion / Burping",
    "Nausea / Vomiting",
    "Loss of Appetite / Feels Full Fast",
    "Bloating / Frequent Burping (Gas)",
    "Internal Discomfort / Itching",
]

result = predict_disease(symptoms, model_dir="model")

print()
print("=" * 62)
print("   VITALSENSE AI  -  DISEASE PREDICTION")
print("=" * 62)
print()
print("   Input Symptoms:")
for label, model_name in zip(user_labels, symptoms):
    print(f"     {label:<38}  ->  {model_name}")
print()
print(f"   Predicted Disease  :  {result['disease']}")
conf = round(result['confidence'] * 100, 1)
bar  = "=" * int(conf / 5) + "-" * (20 - int(conf / 5))
print(f"   Confidence         :  {conf}%")
print(f"   [{bar}]  {conf}%")
if result["warnings"]:
    print(f"   Warnings           :  {result['warnings']}")
print()
print("   NOTE: Educational use only. Consult a real doctor.")
print("=" * 62)
print()
