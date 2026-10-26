"""
medical_kb.py — VitalSense AI: Local Medical Knowledge Base
=============================================================
A comprehensive offline medical Q&A engine that matches user questions
to curated medical knowledge entries using keyword scoring.

No API key or internet connection required.

Author : VitalSense AI Team
"""

from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


# =========================================================================
# Medical Knowledge Base — Curated Q&A Entries
# =========================================================================
# Each entry has:
#   - "keywords"   : list of trigger words/phrases (matched against user input)
#   - "response"   : the answer to return
#   - "category"   : topic category for follow-up suggestions
#   - "follow_ups" : contextual follow-up suggestions

MEDICAL_KB: list[dict[str, Any]] = [

    # ── COMMON COLD & FLU ────────────────────────────────────────────────
    {
        "keywords": ["cold", "common cold", "runny nose", "sneezing", "stuffy nose", "nasal congestion"],
        "response": (
            "### Common Cold\n\n"
            "The **common cold** is a viral infection of the upper respiratory tract. "
            "It's usually harmless and resolves within 7–10 days.\n\n"
            "**Common Symptoms:**\n"
            "- Runny or stuffy nose\n"
            "- Sneezing and sore throat\n"
            "- Mild body aches and low-grade fever\n"
            "- Cough and mild fatigue\n\n"
            "**Home Remedies:**\n"
            "- Stay hydrated — drink warm fluids like soup, tea, or warm water with honey\n"
            "- Rest as much as possible\n"
            "- Use saline nasal drops for congestion\n"
            "- Gargle with warm salt water for sore throat\n"
            "- Steam inhalation can ease nasal blockage\n\n"
            "**When to See a Doctor:**\n"
            "- Fever above 103°F (39.4°C)\n"
            "- Symptoms lasting more than 10 days\n"
            "- Difficulty breathing or chest pain\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "respiratory",
        "follow_ups": ["What's the difference between a cold and the flu?", "What foods help recover from a cold?", "When should I see a doctor?"]
    },

    {
        "keywords": ["flu", "influenza", "high fever", "body ache", "chills", "muscle pain"],
        "response": (
            "### Influenza (Flu)\n\n"
            "**Influenza** is a contagious respiratory illness caused by influenza viruses. "
            "It can range from mild to severe and can sometimes lead to complications.\n\n"
            "**Key Symptoms:**\n"
            "- High fever (100–104°F / 38–40°C)\n"
            "- Severe body aches and muscle pain\n"
            "- Chills and sweats\n"
            "- Headache and extreme fatigue\n"
            "- Dry, persistent cough\n"
            "- Sore throat and nasal congestion\n\n"
            "**How It Differs from a Cold:**\n"
            "- Flu symptoms come on **suddenly** and are more severe\n"
            "- Body aches are more intense with flu\n"
            "- Fever is higher and lasts longer\n\n"
            "**What You Can Do:**\n"
            "- Rest and drink plenty of fluids\n"
            "- Take paracetamol/acetaminophen for fever and pain\n"
            "- Stay home to avoid spreading the virus\n"
            "- Annual flu vaccination is the best prevention\n\n"
            "**Seek Immediate Care If:**\n"
            "- Difficulty breathing or shortness of breath\n"
            "- Persistent chest pain or pressure\n"
            "- Confusion or severe weakness\n"
            "- Symptoms improve then return with worse fever/cough\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "respiratory",
        "follow_ups": ["How long does the flu last?", "Should I get a flu vaccine?", "What complications can flu cause?"]
    },

    # ── DIABETES ──────────────────────────────────────────────────────────
    {
        "keywords": ["diabetes", "blood sugar", "glucose", "diabetic", "sugar level", "hyperglycemia", "type 2 diabetes", "type 1 diabetes", "insulin"],
        "response": (
            "### Diabetes\n\n"
            "**Diabetes mellitus** is a chronic condition where the body cannot properly process blood glucose (sugar).\n\n"
            "**Types:**\n"
            "- **Type 1:** The immune system attacks insulin-producing cells. Usually diagnosed in childhood.\n"
            "- **Type 2:** The body becomes resistant to insulin or doesn't produce enough. Most common type (90–95% of cases).\n"
            "- **Gestational:** Develops during pregnancy and usually resolves after delivery.\n\n"
            "**Common Symptoms:**\n"
            "- Frequent urination and increased thirst\n"
            "- Unexplained weight loss\n"
            "- Extreme fatigue and blurred vision\n"
            "- Slow-healing wounds\n"
            "- Tingling or numbness in hands/feet\n\n"
            "**Normal Blood Sugar Levels:**\n"
            "- Fasting: 70–100 mg/dL\n"
            "- After meals (2 hours): Less than 140 mg/dL\n"
            "- HbA1c: Below 5.7% (normal), 5.7–6.4% (prediabetes), 6.5%+ (diabetes)\n\n"
            "**Management Tips:**\n"
            "- Monitor blood sugar regularly\n"
            "- Follow a balanced, low-glycemic diet\n"
            "- Exercise at least 30 minutes daily\n"
            "- Take prescribed medications consistently\n"
            "- Regular check-ups with your doctor\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "chronic",
        "follow_ups": ["What diet is best for diabetes?", "What is HbA1c?", "What are diabetes complications?"]
    },

    # ── BLOOD PRESSURE ────────────────────────────────────────────────────
    {
        "keywords": ["blood pressure", "hypertension", "high bp", "low bp", "bp", "systolic", "diastolic", "high blood pressure", "low blood pressure", "hypotension"],
        "response": (
            "### Blood Pressure\n\n"
            "**Blood pressure** measures the force of blood against your artery walls. It's recorded as two numbers: systolic (top) over diastolic (bottom).\n\n"
            "**Blood Pressure Categories:**\n"
            "- **Normal:** Less than 120/80 mmHg\n"
            "- **Elevated:** 120–129 / less than 80 mmHg\n"
            "- **High (Stage 1):** 130–139 / 80–89 mmHg\n"
            "- **High (Stage 2):** 140+ / 90+ mmHg\n"
            "- **Crisis:** Above 180/120 mmHg — **seek emergency care immediately**\n\n"
            "**Risks of High Blood Pressure:**\n"
            "- Heart attack and stroke\n"
            "- Kidney damage\n"
            "- Vision problems\n"
            "- Heart failure\n\n"
            "**How to Manage:**\n"
            "- Reduce salt intake (less than 2,300 mg/day)\n"
            "- Exercise regularly (150 minutes/week)\n"
            "- Maintain a healthy weight\n"
            "- Limit alcohol and quit smoking\n"
            "- Manage stress through relaxation techniques\n"
            "- Take prescribed medications as directed\n\n"
            "**Low Blood Pressure** (below 90/60 mmHg) can cause dizziness, fainting, and fatigue. "
            "Stay hydrated, eat small frequent meals, and stand up slowly.\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "chronic",
        "follow_ups": ["What foods lower blood pressure?", "What causes high blood pressure?", "How often should I check my BP?"]
    },

    # ── HEADACHE & MIGRAINE ───────────────────────────────────────────────
    {
        "keywords": ["headache", "migraine", "head pain", "tension headache", "cluster headache", "head ache"],
        "response": (
            "### Headache & Migraine\n\n"
            "Headaches are one of the most common health complaints. Most are not serious, but some patterns need medical attention.\n\n"
            "**Types of Headaches:**\n\n"
            "**1. Tension Headache** (most common)\n"
            "- Dull, aching pain on both sides\n"
            "- Feels like a tight band around the head\n"
            "- Triggered by stress, poor posture, or lack of sleep\n\n"
            "**2. Migraine**\n"
            "- Intense, throbbing pain (usually one side)\n"
            "- Can last 4–72 hours\n"
            "- Often accompanied by nausea, vomiting, light/sound sensitivity\n"
            "- May have \"aura\" (visual disturbances before pain)\n\n"
            "**3. Cluster Headache**\n"
            "- Severe burning or piercing pain behind one eye\n"
            "- Occurs in \"clusters\" — multiple times daily for weeks\n"
            "- Less common but very intense\n\n"
            "**Relief Tips:**\n"
            "- Rest in a quiet, dark room\n"
            "- Apply a cold compress to your forehead\n"
            "- Stay hydrated and eat regular meals\n"
            "- Over-the-counter pain relief (paracetamol/ibuprofen)\n"
            "- Practice relaxation techniques\n\n"
            "**⚠️ See a Doctor Immediately If:**\n"
            "- Sudden, severe headache (\"worst headache of your life\")\n"
            "- Headache with fever, stiff neck, confusion, or seizures\n"
            "- Headache after a head injury\n"
            "- Progressive headaches getting worse over days\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "neurological",
        "follow_ups": ["What triggers migraines?", "When is a headache serious?", "What natural remedies help headaches?"]
    },

    # ── FEVER ─────────────────────────────────────────────────────────────
    {
        "keywords": ["fever", "temperature", "high temperature", "pyrexia", "fever treatment"],
        "response": (
            "### Fever\n\n"
            "A **fever** is a temporary rise in body temperature, often due to an illness. "
            "It's actually your body's natural defense mechanism fighting infection.\n\n"
            "**Temperature Guide:**\n"
            "- **Normal:** 97.8–99.1°F (36.5–37.3°C)\n"
            "- **Low-grade fever:** 99.1–100.4°F (37.3–38°C)\n"
            "- **Moderate fever:** 100.4–103°F (38–39.4°C)\n"
            "- **High fever:** Above 103°F (39.4°C)\n"
            "- **⚠️ Dangerous:** Above 105°F (40.5°C) — seek emergency care\n\n"
            "**What to Do:**\n"
            "- Rest and drink plenty of fluids\n"
            "- Take paracetamol or ibuprofen as directed\n"
            "- Wear light, comfortable clothing\n"
            "- Use a lukewarm (not cold) sponge bath\n"
            "- Keep the room at a comfortable temperature\n\n"
            "**When to See a Doctor:**\n"
            "- Fever above 103°F in adults\n"
            "- Fever lasting more than 3 days\n"
            "- Fever with severe headache, stiff neck, or rash\n"
            "- Fever in infants under 3 months (any fever)\n"
            "- Difficulty breathing with fever\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "general",
        "follow_ups": ["What causes fever?", "How to reduce fever naturally?", "When is fever dangerous?"]
    },

    # ── CHEST PAIN (EMERGENCY) ────────────────────────────────────────────
    {
        "keywords": ["chest pain", "chest tightness", "heart attack", "cardiac", "heart pain", "angina", "chest pressure"],
        "response": (
            "### ⚠️ Chest Pain — Important Information\n\n"
            "Chest pain can have many causes, ranging from minor to life-threatening. "
            "**If you are currently experiencing severe chest pain, call emergency services (112/911) immediately.**\n\n"
            "**Possible Causes:**\n\n"
            "**Heart-Related (seek immediate help):**\n"
            "- Heart attack — crushing pressure, may radiate to arm/jaw\n"
            "- Angina — chest pain during exertion, relieved by rest\n"
            "- Pericarditis — sharp pain that worsens with breathing\n\n"
            "**Non-Heart-Related:**\n"
            "- Acid reflux/GERD — burning sensation after eating\n"
            "- Muscle strain — pain worsens with movement\n"
            "- Anxiety/Panic attack — with rapid heartbeat and shortness of breath\n"
            "- Costochondritis — inflammation of rib cartilage\n\n"
            "**⚠️ Call Emergency Services (112/911) If:**\n"
            "- Crushing or squeezing chest pain\n"
            "- Pain spreading to arm, jaw, neck, or back\n"
            "- Shortness of breath with chest pain\n"
            "- Chest pain with sweating, nausea, or dizziness\n"
            "- Sudden onset of severe chest pain\n\n"
            "**Do NOT ignore chest pain** — even if it turns out to be non-cardiac, "
            "it's always better to get it checked by a medical professional.\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "emergency",
        "follow_ups": ["What are heart attack warning signs?", "How to prevent heart disease?", "What is angina?"]
    },

    # ── COUGH ─────────────────────────────────────────────────────────────
    {
        "keywords": ["cough", "dry cough", "wet cough", "persistent cough", "coughing", "chronic cough", "phlegm", "mucus"],
        "response": (
            "### Cough\n\n"
            "A **cough** is a natural reflex that helps clear your airways. "
            "It can be acute (less than 3 weeks) or chronic (more than 8 weeks).\n\n"
            "**Types of Cough:**\n"
            "- **Dry cough:** No mucus, often from viral infections, allergies, or irritants\n"
            "- **Wet/productive cough:** Produces mucus/phlegm, common in infections\n"
            "- **Barking cough:** Harsh, croup-like sound, common in children\n\n"
            "**Common Causes:**\n"
            "- Cold, flu, or other respiratory infections\n"
            "- Allergies or asthma\n"
            "- Acid reflux (GERD)\n"
            "- Smoking or air pollution\n"
            "- Post-nasal drip\n\n"
            "**Home Remedies:**\n"
            "- Honey with warm water or tea (not for children under 1)\n"
            "- Stay hydrated to thin mucus\n"
            "- Use a humidifier\n"
            "- Gargle with warm salt water\n"
            "- Elevate your head while sleeping\n\n"
            "**See a Doctor If:**\n"
            "- Cough lasts more than 3 weeks\n"
            "- Coughing up blood\n"
            "- High fever with cough\n"
            "- Difficulty breathing or wheezing\n"
            "- Unexplained weight loss with chronic cough\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "respiratory",
        "follow_ups": ["What causes chronic cough?", "Best home remedies for cough?", "When is cough serious?"]
    },

    # ── STOMACH / DIGESTION ───────────────────────────────────────────────
    {
        "keywords": ["stomach pain", "stomach ache", "abdominal pain", "digestion", "indigestion", "bloating", "gas", "acidity", "acid reflux", "gerd", "gastric", "nausea", "vomiting", "diarrhea", "constipation"],
        "response": (
            "### Digestive Issues\n\n"
            "Digestive problems are extremely common and usually not serious. Here's an overview:\n\n"
            "**Acidity / Acid Reflux (GERD):**\n"
            "- Burning sensation in chest (heartburn)\n"
            "- Sour taste in mouth, belching\n"
            "- **Help:** Avoid spicy/oily food, don't lie down after eating, eat smaller meals\n\n"
            "**Bloating & Gas:**\n"
            "- Feeling of fullness, abdominal swelling\n"
            "- **Help:** Eat slowly, avoid carbonated drinks, reduce beans/lentils if sensitive\n\n"
            "**Constipation:**\n"
            "- Fewer than 3 bowel movements per week\n"
            "- **Help:** Increase fiber (fruits, vegetables, whole grains), drink more water, exercise\n\n"
            "**Diarrhea:**\n"
            "- Loose, watery stools 3+ times daily\n"
            "- **Help:** Stay hydrated with ORS, eat bland foods (BRAT diet: bananas, rice, applesauce, toast)\n\n"
            "**Nausea & Vomiting:**\n"
            "- Can be caused by infections, food poisoning, motion sickness, or medications\n"
            "- **Help:** Sip clear fluids, eat light bland foods, ginger tea can help\n\n"
            "**⚠️ See a Doctor If:**\n"
            "- Severe abdominal pain\n"
            "- Blood in stool or vomit\n"
            "- Persistent vomiting (can't keep fluids down)\n"
            "- Symptoms lasting more than a week\n"
            "- Unexplained weight loss\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "digestive",
        "follow_ups": ["What foods help digestion?", "What causes acidity?", "When is stomach pain serious?"]
    },

    # ── SKIN PROBLEMS ─────────────────────────────────────────────────────
    {
        "keywords": ["skin rash", "rash", "itching", "eczema", "acne", "pimple", "skin infection", "fungal", "dermatitis", "psoriasis", "hives", "skin allergy", "skin problem"],
        "response": (
            "### Common Skin Problems\n\n"
            "Skin conditions are varied and can range from mild irritation to conditions needing specialist care.\n\n"
            "**Acne:**\n"
            "- Pimples, blackheads, whiteheads on face, chest, or back\n"
            "- **Care:** Wash face twice daily, avoid touching face, use non-comedogenic products\n\n"
            "**Eczema (Atopic Dermatitis):**\n"
            "- Red, itchy, dry, cracked skin\n"
            "- **Care:** Moisturize frequently, avoid harsh soaps, use prescribed creams\n\n"
            "**Fungal Infections (Ringworm, Athlete's Foot):**\n"
            "- Red, circular, itchy patches\n"
            "- **Care:** Keep area dry, use antifungal cream, don't share towels\n\n"
            "**Hives (Urticaria):**\n"
            "- Raised, itchy welts that appear suddenly\n"
            "- Often caused by allergies\n"
            "- **Care:** Antihistamines, avoid known triggers\n\n"
            "**General Skin Care Tips:**\n"
            "- Keep skin clean and moisturized\n"
            "- Use sunscreen (SPF 30+) when outdoors\n"
            "- Stay hydrated and eat a balanced diet\n"
            "- Don't scratch or pick at skin conditions\n\n"
            "**See a Dermatologist If:**\n"
            "- Rash spreads rapidly or is painful\n"
            "- Skin changes (new moles, color changes)\n"
            "- Condition doesn't improve with home care\n"
            "- Signs of infection (pus, increasing redness, warmth)\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "dermatology",
        "follow_ups": ["How to prevent acne?", "What causes skin rashes?", "When should I see a dermatologist?"]
    },

    # ── ALLERGIES ─────────────────────────────────────────────────────────
    {
        "keywords": ["allergy", "allergic", "allergies", "allergic reaction", "food allergy", "pollen", "dust allergy", "hay fever", "anaphylaxis"],
        "response": (
            "### Allergies\n\n"
            "An **allergy** occurs when your immune system reacts to a normally harmless substance.\n\n"
            "**Common Allergens:**\n"
            "- Pollen, dust mites, and mold\n"
            "- Pet dander (cats, dogs)\n"
            "- Foods (nuts, shellfish, eggs, milk)\n"
            "- Insect stings (bees, wasps)\n"
            "- Medications (penicillin, aspirin)\n\n"
            "**Symptoms:**\n"
            "- Sneezing, runny/stuffy nose\n"
            "- Itchy, watery eyes\n"
            "- Skin rash or hives\n"
            "- Swelling of face, lips, or throat\n"
            "- In severe cases: difficulty breathing (anaphylaxis)\n\n"
            "**Management:**\n"
            "- Identify and avoid your triggers\n"
            "- Take antihistamines as needed\n"
            "- Keep windows closed during high pollen days\n"
            "- Use dust-proof covers on pillows and mattresses\n"
            "- Carry an epinephrine auto-injector if prescribed\n\n"
            "**⚠️ EMERGENCY — Anaphylaxis Signs:**\n"
            "- Difficulty breathing or swallowing\n"
            "- Swelling of tongue or throat\n"
            "- Rapid heartbeat, dizziness, or fainting\n"
            "- **Call 112/911 immediately** and use epinephrine if available\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "immune",
        "follow_ups": ["How to identify food allergies?", "What is anaphylaxis?", "Can allergies be cured?"]
    },

    # ── ASTHMA ────────────────────────────────────────────────────────────
    {
        "keywords": ["asthma", "wheezing", "breathless", "shortness of breath", "difficulty breathing", "inhaler", "breathing problem", "breathing difficulty"],
        "response": (
            "### Asthma\n\n"
            "**Asthma** is a chronic condition where the airways become inflamed and narrow, "
            "making it difficult to breathe.\n\n"
            "**Symptoms:**\n"
            "- Wheezing (whistling sound when breathing)\n"
            "- Shortness of breath\n"
            "- Chest tightness\n"
            "- Coughing, especially at night or early morning\n\n"
            "**Common Triggers:**\n"
            "- Allergens (dust, pollen, pet dander)\n"
            "- Cold air or weather changes\n"
            "- Exercise\n"
            "- Smoke, pollution, or strong odors\n"
            "- Respiratory infections\n"
            "- Stress\n\n"
            "**Management:**\n"
            "- Use prescribed inhalers correctly and regularly\n"
            "- Keep a rescue inhaler accessible at all times\n"
            "- Identify and avoid your triggers\n"
            "- Follow your asthma action plan\n"
            "- Get regular check-ups\n\n"
            "**⚠️ Seek Emergency Help If:**\n"
            "- Severe breathlessness — can't speak in full sentences\n"
            "- Rescue inhaler isn't helping\n"
            "- Lips or fingernails turning blue\n"
            "- Rapid worsening of symptoms\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "respiratory",
        "follow_ups": ["What triggers asthma attacks?", "How to use an inhaler correctly?", "Can asthma be cured?"]
    },

    # ── THYROID ───────────────────────────────────────────────────────────
    {
        "keywords": ["thyroid", "hypothyroidism", "hyperthyroidism", "tsh", "thyroid test", "goiter", "thyroid problem"],
        "response": (
            "### Thyroid Disorders\n\n"
            "The **thyroid gland** (in your neck) produces hormones that regulate metabolism, energy, and body temperature.\n\n"
            "**Hypothyroidism (Underactive Thyroid):**\n"
            "- Fatigue, weight gain, feeling cold\n"
            "- Dry skin, hair loss, constipation\n"
            "- Depression, brain fog, slow heart rate\n"
            "- Treatment: Daily thyroid hormone medication (levothyroxine)\n\n"
            "**Hyperthyroidism (Overactive Thyroid):**\n"
            "- Weight loss, rapid heartbeat, anxiety\n"
            "- Trembling hands, increased sweating\n"
            "- Difficulty sleeping, irritability\n"
            "- Treatment: Medication, radioactive iodine, or surgery\n\n"
            "**Key Tests:**\n"
            "- **TSH:** 0.4–4.0 mIU/L (normal)\n"
            "- **T3 & T4:** Measure actual hormone levels\n"
            "- High TSH + low T4 = hypothyroidism\n"
            "- Low TSH + high T4 = hyperthyroidism\n\n"
            "**Tips:**\n"
            "- Take thyroid medication on an empty stomach, 30 min before food\n"
            "- Get regular blood tests to monitor levels\n"
            "- Don't skip doses or stop medication without doctor's advice\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "endocrine",
        "follow_ups": ["What is TSH?", "What diet helps thyroid problems?", "Can thyroid be cured?"]
    },

    # ── CHOLESTEROL ───────────────────────────────────────────────────────
    {
        "keywords": ["cholesterol", "ldl", "hdl", "triglycerides", "lipid", "lipid profile", "high cholesterol"],
        "response": (
            "### Cholesterol\n\n"
            "**Cholesterol** is a waxy substance in your blood. Your body needs it, but too much can build up in arteries and increase heart disease risk.\n\n"
            "**Types of Cholesterol:**\n"
            "- **LDL (\"Bad\" cholesterol):** Builds up in artery walls. Target: below 100 mg/dL\n"
            "- **HDL (\"Good\" cholesterol):** Helps remove LDL. Target: above 40 mg/dL (men), 50 mg/dL (women)\n"
            "- **Triglycerides:** A type of fat. Target: below 150 mg/dL\n"
            "- **Total Cholesterol:** Target: below 200 mg/dL\n\n"
            "**Risk Factors for High Cholesterol:**\n"
            "- Poor diet (high in saturated and trans fats)\n"
            "- Lack of exercise\n"
            "- Obesity, smoking, and genetics\n\n"
            "**How to Improve Cholesterol:**\n"
            "- Eat heart-healthy foods (oats, nuts, fish, olive oil)\n"
            "- Reduce saturated fats (fried food, red meat, full-fat dairy)\n"
            "- Exercise at least 30 minutes most days\n"
            "- Quit smoking\n"
            "- Take prescribed statins or medications as directed\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "cardiovascular",
        "follow_ups": ["What foods lower cholesterol?", "What is LDL vs HDL?", "How often should I check cholesterol?"]
    },

    # ── HbA1c ─────────────────────────────────────────────────────────────
    {
        "keywords": ["hba1c", "a1c", "glycated hemoglobin", "hemoglobin a1c", "average blood sugar"],
        "response": (
            "### HbA1c (Glycated Hemoglobin)\n\n"
            "**HbA1c** is a blood test that measures your **average blood sugar level over the past 2–3 months**. "
            "Think of it as a \"report card\" for blood sugar control.\n\n"
            "**How It Works:**\n"
            "When blood sugar is high, sugar molecules stick to hemoglobin (a protein in red blood cells). "
            "The HbA1c test measures the percentage of hemoglobin that has sugar attached.\n\n"
            "**What the Numbers Mean:**\n"
            "- **Below 5.7%:** Normal — no diabetes\n"
            "- **5.7% – 6.4%:** Prediabetes — increased risk, lifestyle changes recommended\n"
            "- **6.5% or higher:** Diabetes\n\n"
            "**For Diabetic Patients:**\n"
            "- Target is usually **below 7%** (your doctor may set a different target)\n"
            "- Each 1% drop in HbA1c reduces complication risk significantly\n\n"
            "**Approximate Blood Sugar Equivalents:**\n"
            "- HbA1c 6% ≈ average blood sugar 126 mg/dL\n"
            "- HbA1c 7% ≈ average blood sugar 154 mg/dL\n"
            "- HbA1c 8% ≈ average blood sugar 183 mg/dL\n\n"
            "**Tips to Lower HbA1c:**\n"
            "- Follow a consistent, balanced diet\n"
            "- Exercise regularly\n"
            "- Take medications as prescribed\n"
            "- Monitor blood sugar at home\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "lab_test",
        "follow_ups": ["What is a normal HbA1c?", "How to lower HbA1c naturally?", "How often should HbA1c be tested?"]
    },

    # ── BACK PAIN ─────────────────────────────────────────────────────────
    {
        "keywords": ["back pain", "lower back pain", "spine pain", "sciatica", "back ache", "backache", "slipped disc", "lumbar"],
        "response": (
            "### Back Pain\n\n"
            "**Back pain** is one of the most common reasons for doctor visits. Most cases improve within a few weeks with self-care.\n\n"
            "**Common Causes:**\n"
            "- Muscle strain from heavy lifting or sudden movements\n"
            "- Poor posture (especially sitting for long hours)\n"
            "- Herniated (slipped) disc\n"
            "- Sciatica (pain radiating down the leg)\n"
            "- Arthritis or age-related wear and tear\n\n"
            "**Self-Care Tips:**\n"
            "- Stay active — complete bed rest is NOT recommended\n"
            "- Apply ice (first 48 hours) then heat for relief\n"
            "- Over-the-counter pain relievers (ibuprofen, paracetamol)\n"
            "- Gentle stretching and walking\n"
            "- Maintain good posture, use ergonomic chairs\n"
            "- Sleep on your side with a pillow between knees\n\n"
            "**Prevention:**\n"
            "- Strengthen your core muscles\n"
            "- Lift properly (bend at knees, not waist)\n"
            "- Take breaks from sitting every 30–60 minutes\n"
            "- Maintain a healthy weight\n\n"
            "**See a Doctor If:**\n"
            "- Pain lasts more than 6 weeks\n"
            "- Pain radiates down a leg below the knee\n"
            "- Numbness, tingling, or weakness in legs\n"
            "- Loss of bowel or bladder control (emergency)\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "musculoskeletal",
        "follow_ups": ["What exercises help back pain?", "What is sciatica?", "How to improve posture?"]
    },

    # ── MENTAL HEALTH ─────────────────────────────────────────────────────
    {
        "keywords": ["anxiety", "depression", "stress", "mental health", "panic attack", "worried", "sad", "insomnia", "sleep problem", "can't sleep", "feeling low", "mood"],
        "response": (
            "### Mental Health & Wellness\n\n"
            "Mental health is just as important as physical health. It's okay to seek help — it's a sign of strength, not weakness.\n\n"
            "**Stress & Anxiety:**\n"
            "- Feeling worried, restless, or on edge\n"
            "- Difficulty concentrating, muscle tension\n"
            "- **Help:** Deep breathing exercises, regular exercise, limit caffeine, talk to someone\n\n"
            "**Depression:**\n"
            "- Persistent sadness or loss of interest\n"
            "- Changes in sleep, appetite, or energy\n"
            "- Feelings of worthlessness or guilt\n"
            "- **Help:** Professional counseling, maintain routines, stay connected with people\n\n"
            "**Sleep Problems (Insomnia):**\n"
            "- Difficulty falling or staying asleep\n"
            "- **Help:** Keep a regular sleep schedule, avoid screens before bed, limit caffeine, keep bedroom dark and cool\n\n"
            "**Panic Attacks:**\n"
            "- Sudden intense fear with rapid heartbeat, sweating, trembling\n"
            "- **During an attack:** Focus on slow breathing (in for 4 counts, hold 4, out for 4)\n\n"
            "**General Tips:**\n"
            "- Exercise regularly (it's a natural mood booster)\n"
            "- Practice mindfulness or meditation\n"
            "- Maintain social connections\n"
            "- Limit alcohol and substance use\n"
            "- Get adequate sleep (7–9 hours)\n\n"
            "**⚠️ If you or someone is having thoughts of self-harm or suicide, "
            "please reach out to a crisis helpline immediately:**\n"
            "- **India:** iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345\n"
            "- **US:** National Suicide Prevention: 988\n\n"
            "*You are not alone. Help is available. Please talk to a professional.*"
        ),
        "category": "mental_health",
        "follow_ups": ["How to manage stress?", "What are signs of depression?", "How to improve sleep?"]
    },

    # ── VITAMIN DEFICIENCY ────────────────────────────────────────────────
    {
        "keywords": ["vitamin", "vitamin d", "vitamin b12", "iron deficiency", "anemia", "vitamin c", "calcium", "deficiency", "supplements"],
        "response": (
            "### Common Vitamin & Mineral Deficiencies\n\n"
            "Nutrient deficiencies can cause a range of symptoms. Here are the most common ones:\n\n"
            "**Vitamin D:**\n"
            "- Symptoms: Fatigue, bone pain, muscle weakness, mood changes\n"
            "- Normal: 30–100 ng/mL | Deficient: below 20 ng/mL\n"
            "- Sources: Sunlight, fatty fish, fortified milk, eggs\n\n"
            "**Vitamin B12:**\n"
            "- Symptoms: Fatigue, weakness, numbness/tingling, balance problems\n"
            "- Normal: 200–900 pg/mL | Deficient: below 200 pg/mL\n"
            "- Sources: Meat, fish, eggs, dairy (vegetarians are at higher risk)\n\n"
            "**Iron (Anemia):**\n"
            "- Symptoms: Fatigue, pale skin, shortness of breath, cold hands/feet\n"
            "- Normal hemoglobin: Men 13.5–17.5 g/dL | Women 12–15.5 g/dL\n"
            "- Sources: Red meat, spinach, beans, fortified cereals\n\n"
            "**Calcium:**\n"
            "- Symptoms: Muscle cramps, brittle nails, numbness\n"
            "- Normal: 8.5–10.5 mg/dL\n"
            "- Sources: Dairy, leafy greens, tofu, fortified foods\n\n"
            "**Vitamin C:**\n"
            "- Symptoms: Fatigue, easy bruising, slow wound healing\n"
            "- Sources: Citrus fruits, berries, bell peppers, tomatoes\n\n"
            "**Note:** Always get tested before starting supplements. Excess vitamins can also be harmful.\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "nutrition",
        "follow_ups": ["What foods are rich in vitamin D?", "How to treat iron deficiency?", "Should I take supplements?"]
    },

    # ── COVID-19 ──────────────────────────────────────────────────────────
    {
        "keywords": ["covid", "covid-19", "coronavirus", "corona", "sars-cov", "covid symptoms", "covid vaccine"],
        "response": (
            "### COVID-19\n\n"
            "**COVID-19** is a respiratory illness caused by the SARS-CoV-2 virus.\n\n"
            "**Common Symptoms:**\n"
            "- Fever, cough, and fatigue\n"
            "- Loss of taste or smell\n"
            "- Sore throat and body aches\n"
            "- Shortness of breath\n"
            "- Headache and congestion\n\n"
            "**What to Do If Infected:**\n"
            "- Isolate from others to prevent spread\n"
            "- Rest and drink plenty of fluids\n"
            "- Monitor your oxygen levels (below 94% → seek help)\n"
            "- Take paracetamol for fever and body aches\n"
            "- Contact your doctor for treatment guidance\n\n"
            "**Prevention:**\n"
            "- Get vaccinated and boosted as recommended\n"
            "- Wash hands frequently\n"
            "- Wear masks in crowded/indoor settings\n"
            "- Maintain physical distancing when possible\n\n"
            "**⚠️ Seek Emergency Care If:**\n"
            "- Difficulty breathing or persistent chest pain\n"
            "- Confusion or inability to stay awake\n"
            "- Bluish lips or face\n"
            "- Oxygen saturation below 94%\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "infectious",
        "follow_ups": ["What are long COVID symptoms?", "When should I get tested?", "Is the vaccine safe?"]
    },

    # ── KIDNEY HEALTH ─────────────────────────────────────────────────────
    {
        "keywords": ["kidney", "kidney stone", "kidney disease", "creatinine", "urea", "renal", "kidney infection", "uti", "urinary", "urine"],
        "response": (
            "### Kidney Health\n\n"
            "Your **kidneys** filter waste from your blood, balance fluids, and regulate blood pressure.\n\n"
            "**Common Kidney Issues:**\n\n"
            "**Kidney Stones:**\n"
            "- Severe pain in side/back, pain during urination\n"
            "- Blood in urine, nausea\n"
            "- Prevention: Drink 2–3 liters of water daily, reduce salt, limit oxalate-rich foods\n\n"
            "**Urinary Tract Infection (UTI):**\n"
            "- Burning during urination, frequent urge to urinate\n"
            "- Cloudy or strong-smelling urine\n"
            "- Help: Drink plenty of water, see a doctor for antibiotics\n\n"
            "**Chronic Kidney Disease (CKD):**\n"
            "- Often silent in early stages\n"
            "- Later: swelling (feet/ankles), fatigue, poor appetite\n"
            "- Risk factors: diabetes, high BP, family history\n\n"
            "**Key Tests:**\n"
            "- **Creatinine:** Men: 0.7–1.3 mg/dL | Women: 0.6–1.1 mg/dL\n"
            "- **BUN (Blood Urea Nitrogen):** 7–20 mg/dL\n"
            "- **GFR:** Above 90 = normal, below 60 = possible CKD\n\n"
            "**Kidney Care Tips:**\n"
            "- Drink adequate water daily\n"
            "- Control blood sugar and blood pressure\n"
            "- Reduce salt intake\n"
            "- Don't overuse painkillers (NSAIDs)\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "renal",
        "follow_ups": ["How to prevent kidney stones?", "What is creatinine?", "What foods are good for kidneys?"]
    },

    # ── LIVER HEALTH ──────────────────────────────────────────────────────
    {
        "keywords": ["liver", "jaundice", "hepatitis", "fatty liver", "liver function", "sgpt", "sgot", "alt", "ast", "bilirubin", "liver test"],
        "response": (
            "### Liver Health\n\n"
            "Your **liver** is a vital organ that processes nutrients, filters toxins, and produces bile for digestion.\n\n"
            "**Common Liver Conditions:**\n\n"
            "**Fatty Liver:**\n"
            "- Fat accumulates in liver cells\n"
            "- Often caused by obesity, diabetes, or alcohol\n"
            "- Usually no symptoms in early stages\n"
            "- Management: Weight loss, healthy diet, exercise\n\n"
            "**Hepatitis:**\n"
            "- Inflammation of the liver (A, B, C, D, E)\n"
            "- Symptoms: Jaundice (yellow skin/eyes), fatigue, abdominal pain\n"
            "- Prevention: Vaccination (Hep A & B), safe hygiene practices\n\n"
            "**Key Liver Tests (LFT):**\n"
            "- **SGPT/ALT:** 7–56 U/L (normal)\n"
            "- **SGOT/AST:** 10–40 U/L (normal)\n"
            "- **Bilirubin:** 0.1–1.2 mg/dL (normal)\n"
            "- Elevated levels may indicate liver damage or disease\n\n"
            "**Liver Care Tips:**\n"
            "- Limit alcohol consumption\n"
            "- Maintain a healthy weight\n"
            "- Avoid unnecessary medications and supplements\n"
            "- Eat a balanced diet rich in fruits and vegetables\n"
            "- Get vaccinated for Hepatitis A and B\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "hepatic",
        "follow_ups": ["What causes fatty liver?", "What is SGPT/ALT?", "How to improve liver health?"]
    },

    # ── PREGNANCY ─────────────────────────────────────────────────────────
    {
        "keywords": ["pregnancy", "pregnant", "prenatal", "morning sickness", "pregnancy symptoms", "trimester", "baby"],
        "response": (
            "### Pregnancy Information\n\n"
            "**Pregnancy** is a beautiful journey that lasts approximately 40 weeks, divided into three trimesters.\n\n"
            "**Early Signs of Pregnancy:**\n"
            "- Missed period\n"
            "- Nausea/morning sickness\n"
            "- Breast tenderness\n"
            "- Fatigue and frequent urination\n\n"
            "**Trimester Overview:**\n"
            "- **1st (Weeks 1–12):** Baby's organs form. Morning sickness common.\n"
            "- **2nd (Weeks 13–26):** Baby grows rapidly. Energy often returns.\n"
            "- **3rd (Weeks 27–40):** Baby gains weight. Prepare for delivery.\n\n"
            "**Important Tips:**\n"
            "- Take prenatal vitamins (especially folic acid)\n"
            "- Attend all prenatal check-ups\n"
            "- Eat a balanced diet rich in iron, calcium, and protein\n"
            "- Stay active with gentle exercise (walking, yoga)\n"
            "- Avoid alcohol, smoking, and raw/undercooked foods\n"
            "- Get adequate rest and sleep\n\n"
            "**⚠️ Seek Immediate Care If:**\n"
            "- Heavy bleeding or severe abdominal pain\n"
            "- Severe headache with vision changes\n"
            "- Sudden swelling of face or hands\n"
            "- Baby stops moving or moves less\n"
            "- Water breaks before due date\n\n"
            "**Note:** I can provide general information, but every pregnancy is unique. "
            "Always consult your OB-GYN for personalized care.\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "obstetrics",
        "follow_ups": ["What foods are best during pregnancy?", "What tests are needed during pregnancy?", "What are pregnancy warning signs?"]
    },

    # ── EYE HEALTH ────────────────────────────────────────────────────────
    {
        "keywords": ["eye", "vision", "eye pain", "blurry vision", "eye infection", "conjunctivitis", "red eye", "dry eyes", "eye strain"],
        "response": (
            "### Eye Health\n\n"
            "Your eyes are precious. Here's information about common eye concerns:\n\n"
            "**Eye Strain (Digital Eye Strain):**\n"
            "- Caused by prolonged screen use\n"
            "- Symptoms: Tired eyes, headache, blurred vision\n"
            "- **20-20-20 Rule:** Every 20 minutes, look at something 20 feet away for 20 seconds\n\n"
            "**Conjunctivitis (Pink Eye):**\n"
            "- Red, itchy, watery eyes with discharge\n"
            "- Can be viral, bacterial, or allergic\n"
            "- Don't touch or rub eyes, wash hands frequently\n\n"
            "**Dry Eyes:**\n"
            "- Stinging, burning, or scratchy sensation\n"
            "- Help: Use artificial tears, blink more often, use a humidifier\n\n"
            "**General Eye Care:**\n"
            "- Get regular eye exams (every 1–2 years)\n"
            "- Wear sunglasses with UV protection\n"
            "- Eat foods rich in vitamins A, C, E (carrots, leafy greens, fish)\n"
            "- Don't rub your eyes\n"
            "- Take breaks from screens\n\n"
            "**See an Eye Doctor Immediately If:**\n"
            "- Sudden vision loss or changes\n"
            "- Flashes of light or floating spots\n"
            "- Eye injury or foreign object in eye\n"
            "- Severe eye pain with nausea\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "ophthalmology",
        "follow_ups": ["How to reduce eye strain?", "What foods are good for eyes?", "When should I see an eye doctor?"]
    },

    # ── DENTAL HEALTH ─────────────────────────────────────────────────────
    {
        "keywords": ["tooth", "teeth", "dental", "toothache", "cavity", "gum", "gum disease", "mouth ulcer", "bad breath", "dental pain"],
        "response": (
            "### Dental Health\n\n"
            "Good oral health is essential for overall well-being.\n\n"
            "**Common Dental Problems:**\n\n"
            "**Toothache:**\n"
            "- Usually caused by cavities, infections, or gum disease\n"
            "- Temporary relief: Rinse with warm salt water, use clove oil, take ibuprofen\n"
            "- See a dentist as soon as possible\n\n"
            "**Cavities (Tooth Decay):**\n"
            "- Caused by bacteria that produce acid, eating away tooth enamel\n"
            "- Prevention: Brush twice daily, floss, limit sugary foods\n\n"
            "**Gum Disease:**\n"
            "- Gingivitis (early): Red, swollen, bleeding gums\n"
            "- Periodontitis (advanced): Gums pulling away from teeth\n"
            "- Prevention: Regular brushing, flossing, and dental check-ups\n\n"
            "**Mouth Ulcers:**\n"
            "- Small painful sores inside the mouth\n"
            "- Usually heal in 1–2 weeks\n"
            "- Rinse with warm salt water, avoid spicy/acidic foods\n\n"
            "**Dental Care Tips:**\n"
            "- Brush twice daily for 2 minutes with fluoride toothpaste\n"
            "- Floss daily\n"
            "- Visit your dentist every 6 months\n"
            "- Limit sugary snacks and drinks\n"
            "- Replace your toothbrush every 3–4 months\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "dental",
        "follow_ups": ["How to prevent cavities?", "What causes bad breath?", "How often should I visit the dentist?"]
    },

    # ── FIRST AID ─────────────────────────────────────────────────────────
    {
        "keywords": ["first aid", "burn", "wound", "cut", "bleeding", "fracture", "sprain", "choking", "cpr", "emergency", "injury"],
        "response": (
            "### Basic First Aid Guide\n\n"
            "Knowing basic first aid can save lives. Here are key situations:\n\n"
            "**Cuts & Wounds:**\n"
            "- Clean with running water, apply antiseptic\n"
            "- Apply pressure with clean cloth to stop bleeding\n"
            "- Cover with sterile bandage\n"
            "- Seek help for deep cuts or if bleeding won't stop\n\n"
            "**Burns:**\n"
            "- Cool the burn under running water for 10–20 minutes\n"
            "- Do NOT apply ice, butter, or toothpaste\n"
            "- Cover loosely with a sterile dressing\n"
            "- Seek medical help for large or severe burns\n\n"
            "**Sprains:**\n"
            "- Follow **RICE:** Rest, Ice, Compression, Elevation\n"
            "- Apply ice for 15–20 minutes every 2–3 hours\n"
            "- See a doctor if you can't bear weight\n\n"
            "**Choking:**\n"
            "- Encourage coughing if the person can still breathe\n"
            "- Perform back blows (5) followed by abdominal thrusts (Heimlich maneuver)\n"
            "- Call emergency services if the person can't breathe\n\n"
            "**Fractures:**\n"
            "- Do NOT try to straighten the limb\n"
            "- Immobilize the area, apply ice\n"
            "- Seek immediate medical attention\n\n"
            "**⚠️ Always call emergency services (112/911) for serious injuries, unconsciousness, or life-threatening situations.**\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "emergency",
        "follow_ups": ["How to perform CPR?", "What should be in a first aid kit?", "How to treat a burn?"]
    },

    # ── DIET & NUTRITION ──────────────────────────────────────────────────
    {
        "keywords": ["diet", "nutrition", "healthy eating", "weight loss", "weight gain", "lose weight", "gain weight", "overweight", "obesity", "obese", "calories", "protein", "carbs", "food", "balanced diet", "bmi", "weight"],
        "response": (
            "### Diet & Nutrition\n\n"
            "A **balanced diet** is the foundation of good health.\n\n"
            "**Key Components of a Healthy Diet:**\n"
            "- **Fruits & Vegetables:** At least 5 servings daily\n"
            "- **Whole Grains:** Brown rice, oats, whole wheat bread\n"
            "- **Protein:** Lean meat, fish, eggs, beans, lentils, tofu\n"
            "- **Healthy Fats:** Nuts, seeds, olive oil, avocado\n"
            "- **Dairy:** Milk, yogurt, cheese (or fortified alternatives)\n"
            "- **Water:** At least 8 glasses (2 liters) daily\n\n"
            "**Foods to Limit:**\n"
            "- Processed and ultra-processed foods\n"
            "- Added sugars and sugary drinks\n"
            "- Excessive salt (less than 5g/day)\n"
            "- Saturated and trans fats\n"
            "- Alcohol\n\n"
            "**Weight Management:**\n"
            "- **BMI Calculator:** Weight(kg) ÷ Height(m)²\n"
            "- Underweight: below 18.5 | Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: 30+\n"
            "- For healthy weight loss: aim for 0.5–1 kg per week\n"
            "- Combine diet with regular exercise\n\n"
            "**Tips:**\n"
            "- Eat regular meals, don't skip breakfast\n"
            "- Practice portion control\n"
            "- Cook at home more often\n"
            "- Read food labels\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "nutrition",
        "follow_ups": ["What is a balanced diet plan?", "How to calculate BMI?", "What foods boost immunity?"]
    },

    # ── EXERCISE & FITNESS ────────────────────────────────────────────────
    {
        "keywords": ["exercise", "fitness", "workout", "walking", "yoga", "physical activity", "gym", "stretching"],
        "response": (
            "### Exercise & Fitness\n\n"
            "Regular physical activity is one of the best things you can do for your health.\n\n"
            "**Recommended Activity Levels (Adults):**\n"
            "- At least **150 minutes** of moderate exercise per week, OR\n"
            "- **75 minutes** of vigorous exercise per week\n"
            "- Plus muscle-strengthening activities 2 or more days/week\n\n"
            "**Types of Exercise:**\n"
            "- **Cardio:** Walking, jogging, swimming, cycling (heart health)\n"
            "- **Strength:** Weight lifting, resistance bands, bodyweight exercises\n"
            "- **Flexibility:** Yoga, stretching, Pilates\n"
            "- **Balance:** Tai chi, single-leg stands (especially for older adults)\n\n"
            "**Benefits:**\n"
            "- Reduces risk of heart disease, diabetes, and cancer\n"
            "- Improves mood and reduces anxiety/depression\n"
            "- Strengthens bones and muscles\n"
            "- Improves sleep quality\n"
            "- Helps maintain a healthy weight\n\n"
            "**Getting Started:**\n"
            "- Start slowly and increase gradually\n"
            "- Walking 30 minutes daily is a great start\n"
            "- Find activities you enjoy\n"
            "- Warm up before and cool down after exercise\n"
            "- Stay hydrated during workouts\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "fitness",
        "follow_ups": ["What exercise is best for beginners?", "How much should I walk daily?", "What are the benefits of yoga?"]
    },

    # ── BLOOD TEST ────────────────────────────────────────────────────────
    {
        "keywords": ["blood test", "cbc", "blood count", "wbc", "rbc", "platelet", "hemoglobin", "blood report", "lab test", "lab report"],
        "response": (
            "### Understanding Blood Tests\n\n"
            "A **Complete Blood Count (CBC)** is one of the most common blood tests. Here's what it measures:\n\n"
            "**Red Blood Cells (RBC):**\n"
            "- Carry oxygen throughout your body\n"
            "- Normal: Men 4.5–5.5 million/µL | Women 4.0–5.0 million/µL\n"
            "- Low = possible anemia | High = possible dehydration\n\n"
            "**Hemoglobin (Hb):**\n"
            "- Oxygen-carrying protein in RBCs\n"
            "- Normal: Men 13.5–17.5 g/dL | Women 12.0–15.5 g/dL\n"
            "- Low = anemia\n\n"
            "**White Blood Cells (WBC):**\n"
            "- Fight infections\n"
            "- Normal: 4,000–11,000/µL\n"
            "- High = possible infection | Low = weakened immune system\n\n"
            "**Platelets:**\n"
            "- Help blood clot\n"
            "- Normal: 150,000–400,000/µL\n"
            "- Low = risk of bleeding | High = risk of clotting\n\n"
            "**Other Common Tests:**\n"
            "- **ESR:** Inflammation marker (normal: 0–20 mm/hr)\n"
            "- **Blood Sugar (Fasting):** 70–100 mg/dL\n"
            "- **Creatinine:** Kidney function\n"
            "- **Lipid Profile:** Cholesterol and triglycerides\n\n"
            "**Tip:** You can upload your blood report in VitalSense AI's **Report Analysis** feature for a detailed breakdown!\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "lab_test",
        "follow_ups": ["What does low hemoglobin mean?", "What is ESR?", "Upload my report for analysis"]
    },

    # ── HEART HEALTH ──────────────────────────────────────────────────────
    {
        "keywords": ["heart", "heart disease", "heart health", "heart rate", "pulse", "cardiac", "palpitations", "irregular heartbeat", "arrhythmia", "ecg", "ekg"],
        "response": (
            "### Heart Health\n\n"
            "Your **heart** beats about 100,000 times a day, pumping blood throughout your body.\n\n"
            "**Normal Heart Rate:**\n"
            "- Adults: 60–100 beats per minute (BPM) at rest\n"
            "- Athletes may have resting heart rates of 40–60 BPM\n"
            "- Above 100 BPM at rest = tachycardia\n"
            "- Below 60 BPM = bradycardia (normal for fit people)\n\n"
            "**Heart Disease Risk Factors:**\n"
            "- High blood pressure and cholesterol\n"
            "- Diabetes, smoking, obesity\n"
            "- Family history, sedentary lifestyle\n"
            "- Excessive stress and poor diet\n\n"
            "**Palpitations:**\n"
            "- Feeling like your heart is racing, fluttering, or skipping beats\n"
            "- Often harmless, triggered by caffeine, stress, or exercise\n"
            "- See a doctor if frequent or accompanied by dizziness/fainting\n\n"
            "**Heart-Healthy Habits:**\n"
            "- Exercise regularly (aim for 150 min/week)\n"
            "- Eat a heart-healthy diet (fruits, vegetables, whole grains, fish)\n"
            "- Maintain healthy weight and blood pressure\n"
            "- Don't smoke and limit alcohol\n"
            "- Manage stress\n"
            "- Get regular health check-ups\n\n"
            "**⚠️ Heart Attack Warning Signs:**\n"
            "- Chest pain/pressure, pain in arm/jaw/back\n"
            "- Shortness of breath, nausea, cold sweat\n"
            "- **Call 112/911 immediately**\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "cardiovascular",
        "follow_ups": ["What is a normal heart rate?", "How to prevent heart disease?", "What are palpitations?"]
    },

    # ── MEDICINE QUESTIONS ────────────────────────────────────────────────
    {
        "keywords": ["medicine", "medication", "drug", "tablet", "pill", "side effect", "dose", "dosage", "prescription", "paracetamol", "ibuprofen", "antibiotic"],
        "response": (
            "### Understanding Medicines\n\n"
            "Here are some general guidelines about commonly used medications:\n\n"
            "**Paracetamol (Acetaminophen):**\n"
            "- Used for: Fever, headache, body pain\n"
            "- Usual dose: 500mg–1000mg every 4–6 hours (max 4g/day)\n"
            "- ⚠️ Avoid exceeding maximum dose — can cause liver damage\n\n"
            "**Ibuprofen:**\n"
            "- Used for: Pain, inflammation, fever\n"
            "- Take with food to avoid stomach irritation\n"
            "- ⚠️ Not recommended for those with kidney problems or stomach ulcers\n\n"
            "**Antibiotics:**\n"
            "- Used for bacterial infections ONLY (not for colds or flu)\n"
            "- Always complete the full course, even if you feel better\n"
            "- Don't share or use leftover antibiotics\n\n"
            "**Important Medicine Rules:**\n"
            "- Take medicines as prescribed (right dose, right time)\n"
            "- Don't stop prescribed medication without consulting your doctor\n"
            "- Tell your doctor about ALL medicines you take (including supplements)\n"
            "- Check expiry dates\n"
            "- Store medicines as directed (room temp, refrigerated, etc.)\n\n"
            "**You can check your personalized medicine schedule in VitalSense AI's Medicine Reminder feature!**\n\n"
            "**Note:** I cannot prescribe or recommend specific medications. "
            "Always consult your doctor for medication advice.\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "pharmacy",
        "follow_ups": ["Show my medicine schedule", "What are common side effects?", "Can I take two medicines together?"]
    },

    # ── X-RAY / SCAN ──────────────────────────────────────────────────────
    {
        "keywords": ["x-ray", "xray", "ct scan", "mri", "scan", "ultrasound", "imaging", "radiology", "brain scan", "brain mri"],
        "response": (
            "### Medical Imaging & Scans\n\n"
            "Medical imaging helps doctors see inside your body to diagnose conditions.\n\n"
            "**X-Ray:**\n"
            "- Uses radiation to create images of bones and some organs\n"
            "- Common for: Fractures, chest infections, dental problems\n"
            "- Quick, painless, low radiation dose\n\n"
            "**CT Scan (Computed Tomography):**\n"
            "- Detailed cross-sectional images using X-rays\n"
            "- Common for: Head injuries, abdominal problems, cancer screening\n"
            "- More detailed than regular X-rays\n\n"
            "**MRI (Magnetic Resonance Imaging):**\n"
            "- Uses magnetic fields (no radiation)\n"
            "- Excellent for: Brain, spine, joints, soft tissues\n"
            "- Takes 15–90 minutes, need to lie still\n"
            "- Cannot have MRI with certain metal implants\n\n"
            "**Ultrasound:**\n"
            "- Uses sound waves (no radiation)\n"
            "- Common for: Pregnancy, abdominal organs, heart\n"
            "- Safe, painless, no side effects\n\n"
            "**You can upload your scan reports in VitalSense AI's Report Analysis feature for AI-powered analysis!**\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "radiology",
        "follow_ups": ["What is the difference between CT and MRI?", "Is radiation from X-rays harmful?", "Upload my scan for analysis"]
    },

    # ── GREETINGS & GENERAL ───────────────────────────────────────────────
    {
        "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "help"],
        "response": (
            "### Hello! 👋\n\n"
            "I'm your **VitalSense AI Health Assistant**. I'm here to help you understand health topics and provide general wellness guidance.\n\n"
            "**Here's what I can help you with:**\n\n"
            "- 🏥 **Symptoms & Conditions** — Ask about any symptoms or medical conditions\n"
            "- 💊 **Medications** — Understand your medicines and how they work\n"
            "- 📋 **Lab Reports** — Explain blood test results and normal ranges\n"
            "- 🥗 **Diet & Nutrition** — Healthy eating tips and dietary guidance\n"
            "- 🏃 **Exercise & Fitness** — Activity recommendations\n"
            "- 🧠 **Mental Health** — Stress, anxiety, and wellness tips\n"
            "- 🩺 **Medical Terms** — Plain-English explanations\n"
            "- 🚨 **First Aid** — Basic emergency guidance\n\n"
            "**Try asking me something like:**\n"
            "- \"What are symptoms of diabetes?\"\n"
            "- \"What does high blood pressure mean?\"\n"
            "- \"Explain my HbA1c result\"\n"
            "- \"I have a headache and fever\"\n\n"
            "**Remember:** I provide general health information only. "
            "Always consult a healthcare professional for diagnosis and treatment.\n\n"
            "How can I help you today? 😊"
        ),
        "category": "greeting",
        "follow_ups": ["What are common symptoms of diabetes?", "What does high blood pressure mean?", "Explain what HbA1c means"]
    },

    # ── THANK YOU / BYE ───────────────────────────────────────────────────
    {
        "keywords": ["thank you", "thanks", "thank", "bye", "goodbye", "see you", "good bye"],
        "response": (
            "You're welcome! 😊 I'm glad I could help.\n\n"
            "**Remember these VitalSense AI features:**\n"
            "- 🩺 **AI Health Check** — Analyze your symptoms\n"
            "- 📊 **Report Analysis** — Upload and understand your medical reports\n"
            "- 💊 **Medicine Reminder** — Track your medication schedule\n"
            "- 🏥 **Doctor Support** — Find specialists near you\n\n"
            "Take care of your health, and don't hesitate to come back if you have more questions! 💙\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "category": "greeting",
        "follow_ups": ["What are symptoms of diabetes?", "Tell me about blood pressure", "How to stay healthy?"]
    },
]


# =========================================================================
# Matching Engine
# =========================================================================

def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric chars for matching."""
    return re.sub(r'[^a-z0-9\s]', '', text.lower().strip())


def _score_entry(user_input: str, entry: dict) -> float:
    """
    Score how well a KB entry matches the user's input.
    Higher score = better match.
    Uses keyword presence + bonus for multi-word phrase matches.
    """
    normalized = _normalize(user_input)
    words = set(normalized.split())
    score = 0.0

    for keyword in entry["keywords"]:
        kw_lower = keyword.lower()
        # Exact phrase match in original input (strong signal)
        if kw_lower in user_input.lower():
            # Multi-word phrases get higher score
            word_count = len(kw_lower.split())
            score += 3.0 * word_count
        # Individual word match (weaker signal)
        elif kw_lower in words:
            score += 1.5

    return score


def find_best_response(user_input: str, medicine_context: str = "", profile_context: str = "") -> dict:
    """
    Find the best matching KB entry for the user's input.

    Returns dict with 'reply', 'suggestions', and 'matched'.
    """
    if not user_input.strip():
        return {
            "reply": "Could you please describe your health question? I'm here to help! 😊",
            "suggestions": ["What are symptoms of diabetes?", "Tell me about blood pressure", "What is HbA1c?"],
            "matched": False,
        }

    lower_input = user_input.lower()

    # ── Health Profile queries (checked FIRST) ───────────────────────────
    _PROFILE_KEYWORDS = [
        "my health profile", "my profile", "health profile",
        "my allergies", "what allergies", "saved allergies",
        "my medical conditions", "my conditions", "what conditions",
        "my blood group", "blood type",
        "my medications", "my medicines", "what medications",
        "my emergency contact", "emergency contact",
        "my bmi", "my weight", "my height",
        "my medical history", "my surgeries",
        "my family history", "family medical history",
        "my vaccinations", "my vaccines",
        "my diet", "my lifestyle", "my sleep",
    ]

    is_profile_query = any(kw in lower_input for kw in _PROFILE_KEYWORDS)

    if is_profile_query and profile_context:
        # Build a formatted response from the profile context
        reply = (
            "### 🩺 Your Health Profile\n\n"
            "Here's what I found in your saved health profile:\n\n"
        )

        # Parse profile context lines into formatted output
        lines = profile_context.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("User Health Profile:"):
                continue  # skip header
            if not line:
                continue
            if ':' in line:
                key, _, value = line.partition(':')
                reply += f"- **{key.strip()}**: {value.strip()}\n"
            else:
                reply += f"- {line}\n"

        reply += (
            "\n---\n\n"
            "💡 You can view and edit your complete profile anytime at "
            "[Health Profile](/health-profile).\n\n"
            "*This information is based on what you've saved. "
            "It does not constitute medical advice or diagnosis. "
            "Always consult a qualified healthcare professional for medical guidance.*"
        )

        return {
            "reply": reply,
            "suggestions": [
                "What medical conditions are in my profile?",
                "Show my emergency contact",
                "Tell me about my BMI",
            ],
            "matched": True,
        }

    if is_profile_query and not profile_context:
        return {
            "reply": (
                "### 🩺 Health Profile Not Found\n\n"
                "It looks like you haven't set up your health profile yet.\n\n"
                "You can create your profile by going to "
                "[**Health Profile**](/health-profile) and filling in your details. "
                "Once saved, I'll be able to tell you about your allergies, "
                "conditions, medications, emergency contact, and more!\n\n"
                "*This AI assistant only reports saved information — "
                "it does not diagnose or prescribe.*"
            ),
            "suggestions": [
                "What are symptoms of diabetes?",
                "Tell me about blood pressure",
                "How to stay healthy?",
            ],
            "matched": True,
        }

    # Score all entries
    scored = []
    for entry in MEDICAL_KB:
        s = _score_entry(user_input, entry)
        if s > 0:
            scored.append((s, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] >= 1.5:
        best = scored[0][1]
        reply = best["response"]

        # Append medicine context if user asks about their medicines
        if medicine_context and any(
            w in lower_input
            for w in ["my medicine", "my medication", "my tablet", "my pill",
                       "my schedule", "my medicines", "my medications",
                       "current medicine", "current medication"]
        ):
            reply += f"\n\n---\n\n### Your Current Medicines\n{medicine_context}\n"

        return {
            "reply": reply,
            "suggestions": best.get("follow_ups", [])[:3],
            "matched": True,
        }

    # ── Fallback: no good match ──────────────────────────────────────────
    return {
        "reply": (
            "I appreciate your question! While I don't have specific information on that exact topic "
            "in my knowledge base, here are some things I **can** help you with:\n\n"
            "- 🏥 **Symptoms & Conditions** — diabetes, blood pressure, thyroid, headache, fever, allergies, asthma, etc.\n"
            "- 💊 **Medications** — common medicines, side effects, usage guidelines\n"
            "- 📋 **Lab Reports** — blood tests, cholesterol, HbA1c, liver & kidney tests\n"
            "- 🥗 **Diet & Nutrition** — healthy eating, weight management, vitamins\n"
            "- 🧠 **Mental Health** — stress, anxiety, depression, sleep issues\n"
            "- 🚨 **First Aid** — burns, cuts, sprains, choking\n"
            "- 👁️ **Specialized** — eye health, dental care, skin problems, pregnancy\n"
            "- 🩺 **Health Profile** — your saved allergies, conditions, medications, emergency contact\n\n"
            "Try rephrasing your question, or ask about one of these topics!\n\n"
            "You can also use VitalSense AI's **AI Health Check** feature for symptom analysis, "
            "or **Report Analysis** to upload and understand your medical reports.\n\n"
            "*This information is for educational purposes only and is not a substitute for professional medical advice.*"
        ),
        "suggestions": ["What are symptoms of diabetes?", "Tell me about blood tests", "What's in my health profile?"],
        "matched": False,
    }

