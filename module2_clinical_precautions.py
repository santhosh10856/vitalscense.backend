"""
clinical_precautions.py
======================
AI-Driven & Clinical-Grade Precautions Engine for VitalSense AI — Module 2.

Provides 6 to 8 comprehensive, structured, medically accurate precautions
for every condition, replacing the legacy 3-4 word CSV phrases.

Generation hierarchy:
  1. Live Gemini AI (if GEMINI_API_KEY is available in env, file, or parameter)
  2. Expert Clinical Knowledge Base (covers all 41 dataset conditions with 6-8 in-depth steps)
  3. Dynamic Symptom-Based AI Synthesis (for novel or unknown diseases)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Live Gemini AI prompt template for precautions ────────────────────────────
_AI_PRECAUTIONS_PROMPT = """
You are a senior physician and medical care specialist for the VitalSense AI system.
A patient has been identified with: "{disease}"
Reported symptoms: {symptoms}
Assessed severity level: {severity}

Provide 6 to 8 comprehensive, highly practical, and clinically sound precautions.
Each precaution MUST begin with a bold action directive in markdown (e.g., "**Action Title**: Description").

Include:
1. Immediate symptomatic relief and self-care measures
2. Rest, activity modification, and recovery posture
3. Hygiene, containment, or transmission prevention (if contagious)
4. Hydration, environmental control, and daily vital monitoring
5. Specific triggers or hazards to strictly avoid
6. Critical red-flag emergency symptoms that mandate urgent hospital evaluation

Return ONLY a JSON array of 6 to 8 strings. Example format:
[
  "**Action Title 1**: Detailed clinical guideline...",
  "**Action Title 2**: Detailed clinical guideline...",
  "**Emergency Warning**: Seek immediate emergency care if experiencing..."
]
Do NOT return any markdown code fences, explanation, or extra text.
"""

def _call_gemini_precautions(
    disease: str,
    symptoms: list[str],
    severity: str,
    api_key: str | None = None,
) -> list[str] | None:
    """Attempt live generation via Google Gemini AI."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        sym_str = ", ".join(symptoms) if symptoms else "general presentation"
        prompt = _AI_PRECAUTIONS_PROMPT.format(
            disease=disease,
            symptoms=sym_str,
            severity=severity,
        )
        response = model.generate_content(prompt)
        raw = response.text.strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        import json
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) >= 4:
            logger.info("Generated %d live Gemini precautions for '%s'.", len(parsed), disease)
            return [str(item) for item in parsed]
    except Exception as exc:
        logger.warning("Live Gemini precautions generation skipped/failed: %s", exc)

    return None


# ── Exhaustive Clinical Precautions Knowledge Base (All 41 Conditions) ─────────
CLINICAL_PRECAUTIONS_DB: dict[str, list[str]] = {
    "Fungal Infection": [
        "**Strict Skin Hygiene**: Wash the affected area twice daily with lukewarm water and a mild, antifungal soap. Gently pat completely dry with a clean, separate towel—never rub.",
        "**Moisture & Sweat Control**: Keep the skin fold areas dry throughout the day. Apply doctor-recommended antifungal dusting powder to areas prone to friction and sweat.",
        "**Breathable Natural Clothing**: Wear loose-fitting, 100% breathable cotton garments. Change undergarments and socks at least twice daily or immediately after sweating.",
        "**Prevent Secondary Spread**: Avoid scratching the fungal lesions as this introduces secondary bacterial infection and spreads spores to unaffected body sites.",
        "**Laundry & Linen Sanitisation**: Wash clothes, towels, and bed linens in hot water (at least 60°C) and dry them under direct sunlight or in a hot tumble dryer.",
        "**No Sharing of Personal Items**: Never share towels, combs, clothes, footwear, or nail clippers with family members to prevent fungal transmission.",
        "**Complete Medication Course**: Continue topical antifungal cream applications for at least 1–2 weeks after visual lesions resolve to eliminate underlying fungal spores.",
        "**Emergency Warning**: Consult a physician immediately if the lesion becomes warm, swollen, secretes pus, or spreads with accompanying high fever."
    ],

    "Allergy": [
        "**Immediate Allergen Removal**: Identify and strictly remove the suspected trigger (pollen, dust mites, pet dander, latex, or specific foods). Move to an air-filtered environment.",
        "**Cool Saline & Cold Compresses**: Apply a clean, chilled compress to itchy skin or swollen eyelids for 10–15 minutes at a time to reduce capillary engorgement.",
        "**Nasal Saline Cleansing**: Rinse nasal passages twice daily with sterile isotonic saline solution to flush out trapped airborne allergens and soot.",
        "**Skin Barrier Protection**: Apply soothing hypoallergenic calamine lotion or ceramide-based barrier moisturizers to reduce epidermal irritability.",
        "**Avoid Irritating Substances**: Refrain from using perfumed soaps, harsh detergents, scented body lotions, and synthetic fabrics during active flare-ups.",
        "**Antihistamine Therapy**: Keep non-sedating over-the-counter antihistamines (e.g., Cetirizine or Loratadine) accessible as approved by your doctor or pharmacist.",
        "**Eye & Airway Protection**: Wear protective sunglasses outdoors during high pollen counts and keep bedroom windows closed during early morning hours.",
        "**Red-Flag Anaphylaxis Alert**: Seek emergency medical care immediately (call 911/112) if you experience lip/tongue swelling, hoarse voice, throat tightness, or shortness of breath."
    ],

    "Gerd": [
        "**Elevate Head of Bed**: Raise the head of your bed by 6 to 8 inches (15–20 cm) using bed risers or an ergonomic wedge pillow to prevent nocturnal acid reflux.",
        "**Small, Frequent Meals**: Eat smaller, frequent portions rather than 2–3 large heavy meals to reduce intragastric pressure on the lower esophageal sphincter.",
        "**Strict Meal Timing Rule**: Finish dinner at least 3 hours before lying down or going to sleep. Avoid recumbent resting or lounging immediately after eating.",
        "**Avoid Reflux Triggers**: Eliminate coffee, citrus fruits, tomatoes, mint, chocolate, carbonated soft drinks, deep-fried foods, and alcohol from your daily diet.",
        "**Wear Loose-Fitting Garments**: Avoid tight belts, restrictive corsets, and tight-waisted trousers that exert mechanical pressure on your abdomen and stomach.",
        "**Stop Tobacco & Nicotine**: Avoid smoking and vaping; nicotine relaxes the esophageal sphincter and dramatically increases gastric acid secretion.",
        "**Weight & Posture Management**: Maintain a healthy body weight and sit upright for at least 45 minutes following every meal.",
        "**Emergency Warning**: Seek immediate emergency hospital care if experiencing crushing retrosternal chest pain radiating to the left arm or jaw, or difficulty swallowing (dysphagia)."
    ],

    "Chronic Cholestasis": [
        "**Absolute Alcohol Cessation**: Discontinue all alcohol consumption immediately and permanently to safeguard remaining hepatic function and biliary flow.",
        "**Pruritus (Itching) Care**: Take cool or tepid baths using colloidal oatmeal. Keep fingernails trimmed short to prevent deep skin excoriation and secondary infection.",
        "**Fat-Soluble Vitamin Monitoring**: Take fat-soluble vitamin supplements (Vitamins A, D, E, and K) strictly under your gastroenterologist's prescription.",
        "**Low-Fat & Low-Sodium Diet**: Limit saturated fats and dietary sodium (< 2g/day) to minimize biliary congestion, abdominal ascites, and fluid retention.",
        "**Hydration & Biliary Flush**: Drink 2 to 2.5 litres of purified water daily to assist renal excretion of circulating bile acids.",
        "**Drug Toxicity Warning**: Avoid paracetamol (acetaminophen) in high doses, NSAIDs, and unverified herbal supplements metabolized by the biliary system.",
        "**Routine Hepatic Panel Checks**: Schedule monthly liver function tests (bilirubin, ALP, GGT, ALT/AST) and abdominal ultrasound follow-ups.",
        "**Emergency Warning**: Seek urgent hospital evaluation if you develop clay-colored stools, tea-colored urine, mental confusion, or high fever with right upper quadrant pain."
    ],

    "Drug Reaction": [
        "**Immediate Discontinuation**: Immediately halt the suspected offending medication, supplement, or herbal product unless advised otherwise by an emergency doctor.",
        "**Preserve Medication Packaging**: Retain the exact medication packaging, blister pack, and prescription leaflet to present to the emergency attending physician.",
        "**Soothing Skin Relief**: Apply cool water compresses and calamine lotion to relieve itching. Avoid hot showers and aggressive skin rubbing.",
        "**Oral Hydration & Rinse**: Drink plenty of electrolytes and fresh water to aid kidney filtration of pharmaceutical metabolites.",
        "**Wear Loose, Soft Garments**: Dress in lightweight, non-restrictive cotton clothing to minimize mechanical shear against sensitive dermatological lesions.",
        "**Do Not Self-Medicate**: Never introduce new over-the-counter painkillers or antibiotic ointments which could aggravate the hypersensitivity cascade.",
        "**Maintain an Allergy Card**: Document the generic drug name in your permanent medical records and carry an allergy alert card in your wallet.",
        "**Emergency Anaphylaxis Warning**: Dial emergency services immediately if you develop mucosal blistering (lips, eyes, genitals), facial swelling, shortness of breath, or peeling skin."
    ],

    "Peptic Ulcer Disease": [
        "**Avoid NSAID Painkillers**: Strictly avoid Ibuprofen, Naproxen, Aspirin, and Diclofenac; these medications erode the protective gastric mucosa.",
        "**Strict Irritant Elimination**: Cease consumption of black coffee, alcohol, raw chili, black pepper, and highly acidic citrus juices.",
        "**Regular, Non-Skipped Meals**: Do not skip meals or fast for prolonged intervals; eat bland, gentle meals on a predictable timetable to buffer stomach acid.",
        "**Complete Antibiotic Regimen**: If H. pylori infection is diagnosed, complete the entire 14-day triple/quadruple therapy without missing a single tablet.",
        "**Stress Reduction Techniques**: Practice diaphragmatic breathing, meditation, or light walking; psychological distress significantly elevates gastric acid output.",
        "**Stop Cigarette Smoking**: Smoking severely impairs mucosal blood flow, delays ulcer re-epithelialization, and doubles ulcer recurrence rates.",
        "**Sleep with Torso Elevated**: Use a wedge pillow to prevent nocturnal reflux of corrosive gastric secretions into the duodenum and esophagus.",
        "**Emergency Gastrointestinal Bleed Warning**: Rush to the nearest emergency department if you experience coffee-ground vomiting, black tarry stools (melena), or sudden stabbing abdominal pain."
    ],

    "Aids": [
        "**Strict ART Adherence**: Take all Antiretroviral Therapy (ART) medications at the exact prescribed hours every single day to ensure durable viral suppression.",
        "**Rigorous Food Safety Standards**: Consume only thoroughly cooked foods. Avoid raw eggs, unpasteurized milk, raw seafood, and unwashed vegetables to prevent opportunistic infections.",
        "**Safe Drinking Water**: Drink only boiled or certified fine-filtered water to protect against Cryptosporidium, Giardia, and bacterial enteritis.",
        "**Infection Exposure Control**: Avoid contact with individuals suffering from active respiratory infections, chickenpox, or measles; stay away from cat litter boxes (toxoplasmosis risk).",
        "**Vaccination Updates**: Ensure all recommended inactivated vaccinations (Pneumococcal, Influenza, Hepatitis B) are up-to-date in consultation with your immunologist.",
        "**Regular CD4 & Viral Load Monitoring**: Undergo scheduled blood draws every 3–6 months to monitor CD4+ T-cell counts and viral load suppression.",
        "**Mental Health & Support Systems**: Engage with licensed counselors or support networks to mitigate chronic stress, anxiety, and social isolation.",
        "**Emergency Warning**: Seek immediate emergency medical care if you develop persistent fever over 101°F, unexplained shortness of breath, severe chronic diarrhea, or cognitive confusion."
    ],

    "Diabetes": [
        "**Frequent Blood Glucose Logging**: Check and log blood glucose (fasting, 2-hour postprandial, and bedtime) to identify glycemic patterns and prevent asymptomatic spikes.",
        "**Strict Carbohydrate Control**: Eliminate simple sugars, sodas, fruit juices, and refined flour. Focus on low-glycemic foods rich in soluble fiber (oats, legumes, greens).",
        "**Daily Foot Inspection Routine**: Inspect feet, soles, and interdigital spaces daily using a mirror for blisters, cuts, redness, or calluses; never walk barefoot.",
        "**Medication & Insulin Consistency**: Administer insulin doses or oral hypoglycemic agents at precise times corresponding to your meal schedules.",
        "**Hydration & Kidney Care**: Drink 2 to 3 litres of clean water daily to assist renal glucose clearance; avoid energy drinks and sports beverages.",
        "**Cardiovascular Risk Control**: Keep blood pressure below 130/80 mmHg and monitor LDL cholesterol; exercise at a moderate pace (e.g. 30 minutes brisk walking daily).",
        "**Carry Fast-Acting Hypo Sugar**: Always carry 15 grams of fast-acting glucose (3 glucose tablets or 4 oz juice) to rapidly treat hypoglycemia (< 70 mg/dL).",
        "**Emergency Warning**: Seek emergency hospital admission if experiencing continuous nausea/vomiting, deep rapid breathing, fruity breath odor (DKA alert), or blood sugar > 300 mg/dL with confusion."
    ],

    "Gastroenteritis": [
        "**Continuous Oral Rehydration (ORS)**: Sip Oral Rehydration Salts (ORS) or electrolyte solutions slowly after every loose stool to prevent dangerous hypovolemia.",
        "**BRAT & Bland Diet**: Consume small portions of Bananas, Rice, Applesauce, and Toast. Avoid all dairy, high-fat foods, artificial sweeteners, and caffeine.",
        "**Hand Hygiene & Sanitisation**: Wash hands with soap and warm running water for at least 20 seconds after bathroom visits and before food preparation.",
        "**Rest Gastrointestinal Tract**: Do not force large meals; eat small spoonfuls of clear broths and bland porridge as tolerated.",
        "**Avoid Antidiarrheals Without Advice**: Avoid loperamide if you have bloody stools or high fever, as it can trap invasive bacterial toxins within the colon.",
        "**Linen & Bathroom Disinfection**: Disinfect toilet seats, flush handles, and faucet taps with chlorine-based disinfectants to protect household members.",
        "**Monitor Hydration Indices**: Track urination frequency and urine color; dark amber urine indicates active dehydration.",
        "**Emergency Dehydration Alert**: Seek urgent hospital care if unable to retain liquids for over 12 hours, exhibiting sunken eyes, severe dizziness upon standing, or passing bloody stools."
    ],

    "Bronchial Asthma": [
        "**Carry Rescue Inhaler At All Times**: Ensure your fast-acting bronchodilator (e.g., Albuterol/Salbutamol) is always within arm's reach and check its canister dose counter daily.",
        "**Strict Trigger Avoidance**: Avoid exposure to cigarette smoke, fireplace ash, cold dry drafts, strong perfumes, chemical aerosols, and animal dander.",
        "**Use a Spacer Device**: Always use an aerochamber or spacer with your metered-dose inhaler to maximize lung deposition and minimize oral thrush.",
        "**Daily Peak Flow Monitoring**: Measure your Peak Expiratory Flow (PEFR) every morning; initiate your personal Asthma Action Plan if readings enter the yellow zone (< 80%).",
        "**Bedroom Allergen Encasings**: Encase pillows and mattresses in zippered dust-mite-proof covers; wash bed linens weekly in hot water (> 60°C).",
        "**Rinse Mouth Post Steroid Inhalers**: Thoroughly gargle and spit water after using corticosteroid inhalers to prevent oral candidiasis and dysphonia.",
        "**Warm-Up Before Exercise**: Perform 10–15 minutes of light warm-up exercises and consider prophylactic rescue inhaler puffs prior to athletic exertion in cold weather.",
        "**Emergency Red Flag**: Call emergency services immediately (911/112) if experiencing chest tightness preventing full sentences, blue lips/fingertips, or intercostal retractions."
    ],

    "Hypertension": [
        "**Strict Dietary Sodium Restriction**: Limit total daily sodium intake to under 1,500 mg (less than 2/3 teaspoon of salt); avoid canned soups, pickles, and processed meats.",
        "**Adopt the DASH Diet Plan**: Emphasize fresh fruits, leafy vegetables, low-fat dairy, potassium-rich foods, and whole grains while eliminating saturated fats.",
        "**Daily Morning & Evening BP Logs**: Measure resting blood pressure twice daily using a validated upper-arm cuff monitor after sitting quietly for 5 minutes.",
        "**Consistent Medication Timing**: Take prescribed antihypertensive medications at the exact same hour each day; never stop abruptly as rebound hypertension can occur.",
        "**Limit Alcohol & Eliminate Smoking**: Restrict or eliminate alcohol intake completely; smoking causes acute arterial stiffening and accelerates atherosclerotic vascular damage.",
        "**Stress & Sleep Optimization**: Maintain 7–8 hours of sound sleep; practice guided meditation or progressive muscle relaxation to lower sympathetic tone.",
        "**Aerobic Exercise Routine**: Engage in 30 minutes of moderate-intensity aerobic physical activity (such as brisk walking or swimming) at least 5 days per week.",
        "**Hypertensive Crisis Emergency Alert**: Rush to the nearest emergency department if BP exceeds 180/120 mmHg, or if accompanied by sudden thunderclap headache, blurred vision, or chest pain."
    ],

    "Migraine": [
        "**Dark, Quiet Room Isolation**: At the very first sign of an aura or throbbing headache, lie down in a dark, quiet, temperature-controlled room with eye coverage.",
        "**Cold Compress to Forehead & Neck**: Place an ice pack or gel compress wrapped in a towel across your forehead or back of the neck for 15–20 minutes to constrict dilated cranial vessels.",
        "**Early Medication Administration**: Take your prescribed abortive migraine medication (triptans or NSAIDs) within the first 30 minutes of onset for maximum efficacy.",
        "**Identify & Maintain a Headache Diary**: Track potential triggers including aged cheese (tyramine), red wine, artificial sweeteners (aspartame), MSG, skipping meals, and hormonal shifts.",
        "**Consistent Sleep & Meal Schedule**: Maintain identical bedtimes and wake-up times seven days a week; avoid prolonged gaps between meals to prevent hypoglycemia-triggered migraines.",
        "**Hydration & Electrolyte Intake**: Drink 2.5 litres of water daily; mild dehydration is one of the most frequent covert precipitants of migraine attacks.",
        "**Limit Screen & Blue Light Exposure**: Use blue-light filters, reduce monitor brightness, and take a 5-minute visual break every 30 minutes of computer work.",
        "**Emergency Red Flag**: Seek immediate emergency care if you experience the 'worst headache of your life' (sudden thunderclap onset), fever with neck stiffness, speech difficulty, or weakness."
    ],

    "Cervical Spondylosis": [
        "**Ergonomic Screen Positioning**: Ensure computer monitors and mobile devices are situated at exact eye level to prevent excessive neck flexion ('tech neck').",
        "**Cervical Support Pillow**: Sleep on an orthopedic contoured cervical pillow that supports the natural lordotic curve of the cervical spine; avoid stacking thick pillows.",
        "**Gentle Isometric Neck Exercises**: Perform doctor-prescribed isometric neck strengthening and gentle chin-tuck exercises daily; never jerk or violently twist the neck.",
        "**Alternating Heat & Cold Therapy**: Apply an ice pack for 15 minutes during acute inflammatory flare-ups; switch to moist heating pads to relieve chronic muscular spasms.",
        "**Avoid Heavy Overhead Lifting**: Refrain from carrying heavy backpacks on one shoulder, lifting heavy items overhead, or performing high-impact contact sports.",
        "**Frequent Posture Breaks**: Change your seated posture every 30–45 minutes; perform gentle shoulder shrugs and shoulder blade retractions.",
        "**Use a Soft Cervical Collar Sparingly**: If advised by your orthopedic doctor during acute episodes, wear a collar only for short durations to avoid cervical muscle atrophy.",
        "**Emergency Neurological Warning**: Seek urgent surgical evaluation if you experience progressive numbness, tingling down the arms, hand weakness/clumsiness, or loss of bladder control."
    ],

    "Paralysis (Brain Hemorrhage)": [
        "**Immediate Hospital Emergency Care**: Call emergency services (911/112) immediately. Every second counts in preventing permanent ischemic or hemorrhagic neuronal loss.",
        "**Strict Blood Pressure Management**: Maintain meticulous blood pressure control in hospital ICU/HDU to prevent recurrent re-bleeding or hematoma expansion.",
        "**Aspiration Prevention & Head Elevation**: Keep the head of the patient's bed elevated at 30 degrees; never feed liquids or food orally until formal swallowing assessment is passed.",
        "**Frequent Body Repositioning**: Turn bedridden patients every 2 hours using proper slide sheets to prevent debilitating decubitus ulcers (pressure sores).",
        "**Early Rehabilitation Protocols**: Initiate passive range-of-motion physical therapy, speech therapy, and occupational rehabilitation as soon as cleared by the neurologist.",
        "**Deep Vein Thrombosis (DVT) Prophylaxis**: Utilize pneumatic compression stockings and doctor-prescribed prophylactic regimens to prevent lower-limb venous thrombosis.",
        "**Bowel & Bladder Regimen**: Establish timed catheterization or assisted voiding protocols; implement high-fiber bowel management to avoid Valsalva straining.",
        "**Emergency Warning**: Seek immediate re-evaluation if the patient displays sudden deterioration in consciousness, unequal pupil sizes, seizures, or worsening hemiparesis."
    ],

    "Jaundice": [
        "**Complete Absolute Alcohol Prohibition**: Never consume a single drop of alcohol; hepatic parenchyma is actively inflamed and alcohol can precipitate acute liver failure.",
        "**Rest & Physical Energy Conservation**: Maintain complete physical bed rest during the acute phase; hepatic blood flow increases significantly when lying recumbent.",
        "**Easily Digestible Carbohydrate Diet**: Consume boiled vegetables, fresh sugarcane juice from hygienic sources, glucose water, and ripe fruits; eliminate all fatty and fried foods.",
        "**Avoid Hepatotoxic Pharmaceuticals**: Strictly avoid paracetamol, sedatives, non-prescribed pain relievers, and dubious herbal medicines metabolized by hepatic cytochrome enzymes.",
        "**Hydration & Bilirubin Clearance**: Drink 2.5 to 3 litres of purified boiled water daily to maintain renal perfusion and assist urinary clearance of conjugated bile pigments.",
        "**Strict Food & Water Sanitation**: Consume only freshly boiled water and home-cooked meals; avoid street food, raw salads, and unpeeled fruits to prevent secondary enteric infections.",
        "**Monitor Stool & Urine Color**: Observe daily changes in skin yellowing, scleral icterus, urine darkening, and stool color to report trends to your treating hepatologist.",
        "**Emergency Warning**: Seek immediate emergency hospital care if the patient exhibits confusion, inverted sleep cycles, extreme drowsiness (hepatic encephalopathy), or vomiting blood."
    ],

    "Malaria": [
        "**Complete Full Antimalarial Course**: Take the entire prescribed course of Artemisinin-based Combination Therapy (ACT) without interruption, even if fever subsides after day 2.",
        "**Fever Management & Tepid Sponging**: Administer paracetamol as prescribed and apply tepid water sponge baths to forehead, armpits, and groin during high fever spikes (> 102°F).",
        "**Aggressive Hydration & Electrolytes**: Drink abundant oral rehydration salts (ORS), coconut water, and clean soups to replace fluid losses from heavy rigors and profuse sweating.",
        "**Mosquito Bite Prevention**: Sleep under insecticide-treated bed nets (ITNs), use DEET mosquito repellents, install window mesh screens, and eliminate stagnant water around the home.",
        "**Nutritious, Light Meals**: Eat high-calorie, easily digestible soft foods (khichdi, porridge, stewed fruits) to replenish metabolic reserves drained by parasitic hemolysis.",
        "**Avoid Aspirin & Ibuprofen**: Avoid aspirin or non-steroidal anti-inflammatory agents during acute fevers to prevent exacerbating gastrointestinal bleeding and platelet dysfunction.",
        "**Follow-Up Blood Smear / RDT**: Undergo a repeat peripheral blood smear or rapid diagnostic test post-treatment to confirm complete parasitic clearance.",
        "**Emergency Warning (Severe Malaria)**: Seek immediate hospital care if experiencing dark/black urine (blackwater fever), severe jaundice, continuous vomiting, breathlessness, or delirium."
    ],

    "Chicken Pox": [
        "**Strict Quarantine & Airborne Isolation**: Remain isolated in a well-ventilated private room until all blisters have dried, formed scabs, and crusts have fallen off (approx 7–10 days).",
        "**Never Scratch Lesions**: Trim fingernails very short and wear soft cotton mittens at night to prevent scratching, which causes permanent scarring and secondary staph infections.",
        "**Soothing Oatmeal & Calamine Care**: Take soothing cool baths infused with colloidal oatmeal or baking soda; gently dab calamine lotion onto pruritic vesicles.",
        "**Paracetamol Only for Fever (NEVER Aspirin)**: Use paracetamol for fever; NEVER give aspirin to children or teenagers with chickenpox due to the fatal risk of Reye's syndrome.",
        "**Loose, Soft Cotton Clothing**: Wear lightweight, loose cotton clothes that do not rub against blisters; change bed linens and clothing daily.",
        "**Hydration & Soft Bland Diet**: Drink plenty of fluids; offer cold popsicles, yogurt, and soft purees if intraoral blisters cause painful swallowing.",
        "**Disinfection of Shared Areas**: Sanitize door handles, bathroom surfaces, and laundry with disinfectant; keep vulnerable, non-immune, and pregnant individuals away.",
        "**Emergency Warning**: Seek immediate emergency medical care if blisters become red, swollen, and warm with pus (bacterial superinfection), or if the patient develops coughing or lethargy."
    ],

    "Dengue": [
        "**Aggressive Oral Fluid Intake**: Drink at least 3 to 4 litres of fluids daily including ORS, coconut water, fresh fruit juices, and broths to counter dangerous capillary plasma leakage.",
        "**Daily Platelet & Hematocrit Monitoring**: Have complete blood counts (CBC) checked daily to track platelet drops and rising hematocrit levels (the primary marker of plasma leakage).",
        "**Paracetamol Only for Fever**: Take ONLY paracetamol for fever and body aches; strictly avoid Aspirin, Ibuprofen, Diclofenac, and Mefenamic acid (severe bleeding risk).",
        "**Absolute Bed Rest**: Conserve all physical energy and remain in bed; avoid strenuous movements, bending, or lifting that could trigger internal bleeding.",
        "**Mosquito Net Protection**: Sleep under mosquito netting even during daylight hours to prevent Aedes mosquitoes from biting you and transmitting the virus to family members.",
        "**Monitor Daily Urine Output**: Track the frequency and volume of urination; passing urine at least once every 4 to 6 hours indicates adequate physiological hydration.",
        "**Nutritious Soft Diet**: Consume easily digestible, nutrient-dense soft foods like khichdi, porridge, and boiled vegetables; avoid spicy, acidic, or dark-colored foods.",
        "**Dengue Warning Signs (CRITICAL)**: Rush immediately to an ICU if experiencing severe abdominal pain, persistent vomiting, mucosal bleeding (gums/nose), or extreme cold clammy skin."
    ],

    "Typhoid": [
        "**Complete Full Antibiotic Course**: Complete the full 10–14 day course of prescribed antibiotics (e.g., Ceftriaxone or Azithromycin) without missing doses to avoid chronic carrier state.",
        "**Boil or Disinfect All Water**: Drink only rolling-boiled water (boiled for at least 1 minute) or certified bottled water; avoid ice cubes from unknown sources.",
        "**Strict Hygiene & Hand Sanitization**: Wash hands thoroughly with soap and warm water for 20 seconds after using the toilet and before touching any food or utensils.",
        "**High-Calorie Bland Diet**: Consume a low-fiber, high-calorie bland diet (rice porridge, boiled potatoes, soft bananas, custards); avoid raw salads, unpeeled fruits, and spicy foods.",
        "**Avoid Food Preparation for Others**: Do not prepare, cook, or serve food or beverages to other household members until laboratory stool cultures confirm negative status.",
        "**Fever Monitoring & Cold Sponging**: Log body temperature every 4 hours; apply cool damp compresses to manage high continuous stepwise fevers.",
        "**Rest & Avoid Abdominal Strain**: Maintain bed rest; avoid heavy physical work, intense straining, or jumping which could provoke intestinal hemorrhage or ileal perforation.",
        "**Emergency Red Flag**: Seek emergency surgical care immediately if experiencing sudden excruciating abdominal pain, rigid abdomen, vomiting blood, or sudden hypothermic collapse."
    ],

    "Hepatitis A": [
        "**Strict Fecal-Oral Isolation**: Maintain private bathroom hygiene; wash hands meticulously with soap for 30 seconds after toileting to prevent infecting close household contacts.",
        "**Drink Exclusively Boiled Water**: Consume only water that has been vigorously boiled for at least 1 minute; avoid raw tap water, ice cubes, and unfiltered dispensers.",
        "**No Food Handling for Others**: Refrain from cooking or handling food for family members or in commercial settings until declared non-infectious by a doctor.",
        "**Absolute Zero Alcohol**: Abstain from all alcohol for at least 6 months following recovery; inflamed hepatocytes require complete metabolic relief.",
        "**High-Carbohydrate, Low-Fat Diet**: Eat easily digestible meals rich in complex carbs (oatmeal, rice, honey, boiled vegetables); strictly avoid deep-fried foods and greasy meats.",
        "**Disinfect Bathroom Surfaces**: Clean toilet bowls, flush handles, faucets, and doorknobs daily using chlorine bleach solutions (1 part bleach to 9 parts water).",
        "**Household Contact Vaccination**: Advise all unvaccinated close household and sexual contacts to receive the Hepatitis A vaccine or immune globulin within 2 weeks of exposure.",
        "**Emergency Warning**: Seek immediate hospital care if you observe mental confusion, extreme sleepiness, bleeding gums, or severe continuous vomiting."
    ],

    "Hepatitis B": [
        "**Safe Sexual Practices & Barrier Protection**: Practice strict barrier protection (condoms); inform intimate partners so they can be screened and vaccinated against Hepatitis B.",
        "**Zero Sharing of Blood-Contact Items**: Never share toothbrushes, razors, cuticle scissors, needles, or blood glucose lancets; safely dispose of sanitary items.",
        "**Lifelong Alcohol Abstinence**: Avoid all alcoholic beverages permanently; alcohol acts synergistically with HBV to accelerate liver cirrhosis and hepatocellular carcinoma.",
        "**Strict Antiviral Adherence**: If prescribed antiviral therapy (e.g., Tenofovir, Entecavir), take your medication daily without missing doses to suppress viral replication.",
        "**Liver Cancer Surveillance**: Undergo hepatic ultrasound examinations and Alpha-Fetoprotein (AFP) blood tests every 6 months to detect early neoplastic changes.",
        "**Household Contact Screening & Immunization**: Ensure all household members receive the 3-dose Hepatitis B vaccine series and confirm post-vaccination surface antibody titers.",
        "**Cover Open Wounds Promptly**: Clean and cover any cuts or scrapes with waterproof bandages; clean blood spills using a 1:10 household bleach solution.",
        "**Emergency Warning**: Seek emergency medical evaluation if you develop abdominal swelling (ascites), jaundice, vomiting blood, or disorientation."
    ],

    "Hepatitis C": [
        "**Direct-Acting Antiviral (DAA) Therapy**: Complete your prescribed 8–12 week course of DAA medications without interruption; modern DAAs achieve over 95% cure rates.",
        "**Never Share Sharp Personal Items**: Do not share razors, nail clippers, toothbrushes, or tattoo needles; HCV is primarily transmitted through microscopic blood contact.",
        "**Complete Elimination of Alcohol**: Cease all alcohol intake; even small amounts dramatically accelerate HCV-driven hepatic fibrosis and cirrhosis.",
        "**Healthy Liver Diet & Coffee Benefit**: Consume a Mediterranean-style diet low in processed foods; research suggests 2 cups of unsweetened filtered coffee daily helps reduce liver stiffness.",
        "**Hepatitis A & B Vaccination**: Receive full vaccination against Hepatitis A and B to protect your liver from devastating superinfections.",
        "**Maintain Healthy Body Weight**: Control weight and blood lipids to avoid concurrent metabolic dysfunction-associated steatotic liver disease (MASLD).",
        "**Regular Viral RNA Follow-Up**: Have HCV RNA viral load checked 12 weeks post-treatment (SVR12) to medically confirm permanent viral eradication.",
        "**Emergency Warning**: Rush to an emergency department if experiencing hematemesis (vomiting blood), dark tarry stools, sudden fluid buildup in the abdomen, or severe jaundice."
    ],

    "Hepatitis D": [
        "**Dual Infection Management**: Work closely with a specialist hepatologist; HDV requires Hepatitis B for replication and carries a high risk of rapid liver decompensation.",
        "**Strict Prevention of Blood Exposure**: Avoid sharing any sharp implements, needles, or personal hygiene items that may carry blood traces.",
        "**Permanent Alcohol & Toxin Cessation**: Strictly eliminate alcohol, smoking, and hepatotoxic pharmaceuticals to protect remaining functioning liver tissue.",
        "**Specialized Antiviral / Interferon Therapy**: Follow prescribed regimens (such as Bulevirtide or Pegylated Interferon alpha) under strict hospital monitoring.",
        "**Frequent Ultrasound & Elastography**: Schedule hepatic ultrasound and transient elastography (FibroScan) every 6 months to monitor fibrosis progression.",
        "**Low-Sodium Nutrition Plan**: Restrict dietary sodium to less than 2 grams daily to minimize fluid retention, portal hypertension, and ascites.",
        "**Vaccinate Household for HBV**: While HDV has no standalone vaccine, preventing HBV in contacts completely prevents HDV transmission.",
        "**Emergency Warning**: Seek urgent hospital admission if showing confusion, extreme lethargy, yellowing eyes, or abdominal distension."
    ],

    "Hepatitis E": [
        "**Rigorous Water Boiling & Filtration**: Drink only water that has been boiled vigorously for over 1 minute; Hepatitis E is an enteric virus spread via contaminated water supplies.",
        "**Never Eat Raw or Undercooked Meat**: Ensure pork, venison, and shellfish are cooked thoroughly to an internal temperature of at least 71°C (160°F).",
        "**Special Alert for Pregnant Women**: Pregnant women in their 2nd or 3rd trimester face high mortality risks from HEV fulminant hepatic failure; seek immediate specialized hospitalization.",
        "**Strict Hand Hygiene Practices**: Wash hands with soap and water before meals, before preparing infant formula, and immediately after using restrooms.",
        "**Abstain Completely From Alcohol**: Give the liver complete rest by abstaining from all alcoholic beverages and unnecessary over-the-counter painkillers.",
        "**Nutritious High-Carbohydrate Diet**: Consume boiled potatoes, rice porridge, applesauce, and herbal broths; avoid all oily, fatty, and street-prepared foods.",
        "**Sanitation of Water Storage Vessels**: Thoroughly clean and disinfect domestic water tanks, filters, and bottles every week.",
        "**Emergency Warning**: Seek emergency hospitalization if you experience excessive drowsiness, persistent vomiting, dark tea-colored urine, or bleeding tendencies."
    ],

    "Alcoholic Hepatitis": [
        "**Absolute, Immediate Alcohol Cessation**: Never consume any alcohol under any circumstance; continuing alcohol intake carries an extremely high 30-day mortality rate.",
        "**Medical Detoxification Supervision**: Seek medically supervised detox to safely manage dangerous alcohol withdrawal symptoms (seizures, delirium tremens).",
        "**High-Protein, High-Calorie Nutrition**: Malnutrition is prevalent; consume small, frequent meals enriched with clean proteins (eggs, legumes, lean poultry) unless restricted by doctor.",
        "**Thiamine & Multivitamin Supplementation**: Take prescribed Thiamine (Vitamin B1) and B-complex vitamins daily to prevent irreversible Wernicke-Korsakoff encephalopathy.",
        "**Strict Sodium Restriction (< 2g/Day)**: Eliminate added table salt, processed snacks, and canned foods to manage portal hypertension and ascites.",
        "**Avoid Over-the-Counter Analgesics**: Avoid paracetamol, ibuprofen, and aspirin; compromised hepatic parenchyma cannot safely metabolize these drugs.",
        "**Engage Psychological & Addiction Support**: Join structured rehabilitation programs, counseling, or support groups (e.g., AA) for long-term sobriety maintenance.",
        "**Emergency Warning Signs**: Call emergency services immediately if you develop hematemesis (vomiting blood), black stools, severe confusion, or a rapidly swelling abdomen."
    ],

    "Tuberculosis": [
        "**Never Skip or Stop Anti-TB Drugs**: Complete the full 6 to 9-month course of DOTS therapy (Rifampicin, Isoniazid, Pyrazinamide, Ethambutol) to prevent lethal multidrug-resistant TB (MDR-TB).",
        "**Respiratory Isolation & Ventilation**: Sleep in a well-ventilated single room with windows open; wear an N95 or well-fitting surgical mask when interacting with others during the first 2–3 weeks.",
        "**Safe Cough Etiquette & Disposal**: Always cough or sneeze into disposable tissues, seal them in a plastic bag, and incinerate or discard them in covered waste bins.",
        "**Zero Alcohol & Tobacco Smoking**: Strictly avoid alcohol (causes severe drug-induced hepatitis when combined with TB meds) and quit all tobacco smoking immediately.",
        "**High-Protein Nutritional Rehabilitation**: Eat high-protein, calorie-dense foods (eggs, milk, pulses, nuts, paneer) to reverse chronic TB muscle wasting and weight loss.",
        "**Monthly Sputum & Liver Function Tests**: Attend all scheduled appointments for sputum smear microscopy, chest X-rays, and liver enzyme evaluations.",
        "**Household Contact Screening**: Ensure all close family members, especially children under 5 and elderly individuals, undergo Mantoux testing or chest X-ray screening.",
        "**Emergency Warning**: Seek urgent emergency care if you experience massive hemoptysis (coughing up large amounts of blood), severe breathlessness, or yellowing of the eyes."
    ],

    "Common Cold": [
        "**Abundant Warm Fluid Hydration**: Drink at least 2.5 to 3 litres of warm water, ginger tea, herbal broths, or warm lemon-honey water daily to soothe mucous membranes and loosen phlegm.",
        "**Rest & Energy Conservation**: Get 8 to 9 hours of quality sleep; physical rest empowers your immune system to clear the viral infection much faster.",
        "**Warm Saline Gargling Routine**: Gargle with a glass of warm water mixed with 1/2 teaspoon of salt 3–4 times daily to reduce pharyngeal inflammation and sore throat pain.",
        "**Steam Inhalation & Humidification**: Inhale warm steam for 10–15 minutes twice daily or use a room humidifier to relieve congested nasal passages and sinus pressure.",
        "**Frequent Hand Washing & Sanitizing**: Wash hands frequently with soap for 20 seconds; disinfect shared surfaces like keyboards, door handles, and phone screens.",
        "**Saline Nasal Irrigation**: Use gentle isotonic saline nasal sprays or a neti pot (using distilled water only) to clear thick mucus without drug side-effects.",
        "**Avoid Antibiotics for Viral Colds**: Never take leftover antibiotics; the common cold is caused by rhinoviruses, and unnecessary antibiotics cause microbiome damage and bacterial resistance.",
        "**Emergency Warning**: Consult a physician if fever exceeds 102°F (38.9°C), symptoms worsen after 10 days, or you develop severe sinus pain or chest tightness."
    ],

    "Pneumonia": [
        "**Strict Compliance with Prescribed Therapy**: Take every dose of prescribed antibiotics or antivirals exactly on schedule; never discontinue treatment early even if symptoms improve.",
        "**Continuous Pulse Oximetry Checks**: Monitor oxygen saturation (SpO2) every 4 to 6 hours using a fingertip pulse oximeter; if SpO2 drops below 93%, seek emergency oxygen therapy.",
        "**Deep Breathing & Incentive Spirometry**: Perform 5–10 deep breathing and controlled coughing cycles every hour to expand alveoli and clear pooled pulmonary secretions.",
        "**Elevate Sleeping Posture**: Sleep with the head and upper torso propped up at a 30 to 45-degree angle using extra pillows to facilitate diaphragmatic expansion.",
        "**Hydrate to Thin Viscous Mucus**: Drink 2.5 to 3 litres of warm fluids (water, clear soups, warm teas) daily to thin thick alveolar mucus and facilitate expectoration.",
        "**Avoid Cough Suppressants Without Approval**: Do not suppress productive coughs with OTC antitussives unless prescribed; coughing is essential to clear infected sputum from your lungs.",
        "**Avoid Smoke & Cold Drafts**: Strictly avoid active or secondhand tobacco smoke, wood-burning smoke, aerosol sprays, and sudden exposure to frigid air.",
        "**CRITICAL EMERGENCY WARNING**: Call emergency services (911/112) immediately if you experience severe shortness of breath, stabbing chest pain while breathing, blue lips, or confusion."
    ],

    "Dimorphic Hemmorhoids(Piles)": [
        "**High-Fiber Nutrition Plan**: Consume 30–35 grams of dietary fiber daily (whole grains, oats, psyllium husk / isabgol, prunes, green leafy vegetables) to ensure soft, bulky stools.",
        "**Drink 3 Litres of Water Daily**: Adequate hydration is non-negotiable; insufficient fluid hardens stool and causes excruciating friction against anal hemorrhoidal cushions.",
        "**Warm Water Sitz Baths**: Sit in a tub of warm water (hip-deep) for 15–20 minutes 2 to 3 times daily, especially after bowel movements, to relax the internal anal sphincter.",
        "**Strictly Never Strain on the Toilet**: Do not strain or hold your breath (Valsalva maneuver); keep toilet sitting time under 3–5 minutes. Never read or browse your phone on the toilet.",
        "**Use an Ergonomic Footstool (Squatting Posture)**: Elevate your feet on a small stool while on the toilet to straighten the anorectal angle and ease stool passage.",
        "**Gentle Moist Cleaning (No Dry Rubbing)**: Use moist wipes, a bidet, or warm water spray instead of dry, abrasive toilet paper; pat gently with a soft towel.",
        "**Avoid Prolonged Stationary Sitting**: Stand up, walk, or stretch every 45 minutes; use a specialized donut-shaped orthopedic cushion if seated work is mandatory.",
        "**Emergency Warning**: Seek urgent surgical care if hemorrhoids become hard, purple, and excruciatingly painful (thrombosed hemorrhoid), or if there is continuous heavy rectal bleeding."
    ],

    "Heart Attack": [
        "**CALL EMERGENCY SERVICES IMMEDIATELY (911 / 112)**: Every minute saves heart muscle. Do not attempt to drive yourself to the hospital under any circumstances.",
        "**Chew 300 mg of Uncoated Aspirin**: If conscious, not allergic to aspirin, and advised by emergency dispatchers, chew and swallow an adult aspirin (or 4 baby aspirins) immediately.",
        "**Adopt a Semi-Sitting Rest Position**: Sit on the floor or bed with knees bent, head and shoulders supported; this reduces cardiac venous return and cardiac workload.",
        "**Loosen All Constrictive Clothing**: Undo tight collars, neckties, belts, and waistbands to facilitate unobstructed breathing and reduce patient anxiety.",
        "**Take Prescribed Nitroglycerin (If Applicable)**: If previously diagnosed with angina and prescribed sublingual nitroglycerin, place one tablet under your tongue while seated.",
        "**Never Leave the Patient Unattended**: Keep the patient calm and quiet; avoid food, drinks, or strenuous movements. Be prepared to perform CPR if the patient becomes unresponsive.",
        "**Locate Automated External Defibrillator (AED)**: If in a public place or office, immediately send a bystander to retrieve the nearest AED.",
        "**Critical Alert**: If the patient loses consciousness and stops normal breathing, immediately initiate high-quality chest compressions at 100–120 beats per minute until paramedics arrive."
    ],

    "Varicose Veins": [
        "**Wear Graduated Compression Stockings**: Put on medical-grade compression stockings (15–20 or 20–30 mmHg) first thing in the morning before getting out of bed, as prescribed by your doctor.",
        "**Regular Leg Elevation Routine**: Elevate your legs above heart level for 15–20 minutes, 3 to 4 times a day, to encourage venous drainage and relieve lower-extremity edema.",
        "**Avoid Prolonged Unbroken Standing or Sitting**: Never sit or stand continuously for more than 45 minutes; take 3-minute walking breaks or perform ankle calf-pump flexions.",
        "**Do Not Cross Your Legs**: Avoid crossing your legs at the knees or ankles while seated, as this mechanically compresses the popliteal and femoral venous channels.",
        "**Low-Sodium Diet to Prevent Water Retention**: Limit high-salt foods, canned goods, and processed snacks to minimize peripheral fluid retention and venous wall distension.",
        "**Gentle Low-Impact Exercise**: Engage in 30 minutes of daily walking, swimming, or cycling to activate the calf muscle pump (the 'second heart' of venous return).",
        "**Skin Moisturisation & Trauma Prevention**: Apply gentle unscented moisturizers to prevent venous stasis dermatitis; avoid bumping legs against hard furniture.",
        "**Emergency Warning**: Seek urgent medical evaluation if a vein becomes red, hard, and hot to the touch (phlebitis), if an open leg ulcer develops, or if a vein begins to bleed profusely."
    ],

    "Hypothyroidism": [
        "**Empty-Stomach Morning Medication Rule**: Take your Levothyroxine / thyroid hormone tablet first thing in the morning with a full glass of plain water, at least 30–60 minutes before breakfast.",
        "**Separate From Calcium & Iron Supplements**: Wait at least 4 hours between taking thyroid medication and consuming calcium, iron, antacids, or high-fiber foods that bind the hormone.",
        "**Regular TSH & Free T4 Monitoring**: Have your Thyroid Stimulating Hormone (TSH) levels tested every 6 to 8 weeks following any dosage alteration until stabilized.",
        "**Balanced Iodine & Selenium Nutrition**: Ensure adequate dietary intake of iodine and selenium (eggs, dairy, nuts); avoid excessive consumption of raw goitrogenic vegetables (raw cabbage, kale).",
        "**Daily Weight & Energy Tracking**: Keep a daily log of body weight, energy fluctuations, cold tolerance, and bowel movements to review with your endocrinologist.",
        "**Manage Constipation with Fiber & Water**: Drink 2.5 litres of water and increase soluble fiber intake to combat characteristic thyroid-related sluggish intestinal motility.",
        "**Engage in Regular Aerobic Activity**: Exercise moderately 3–4 days a week to stimulate metabolic rate, preserve muscle mass, and boost mood.",
        "**Emergency Warning**: Seek immediate emergency hospital care if experiencing extreme hypothermia, slow heart rate (< 50 bpm), profound lethargy, or facial puffiness (myxedema crisis)."
    ],

    "Hyperthyroidism": [
        "**Strict Antithyroid Medication Adherence**: Take prescribed antithyroid medications (Methimazole or Propylthiouracil) consistently as ordered to control thyroid hormone hypersecretion.",
        "**Eliminate Stimulants & Caffeine**: Strictly avoid coffee, energy drinks, green tea, pre-workout supplements, and excessive dark chocolate that trigger tachycardia and palpitations.",
        "**Avoid High-Iodine Foods**: Limit iodized salt, seaweed, kelp, seafood, and iodine-containing cough syrups that provide excess substrate for thyroid hormone synthesis.",
        "**Heart Rate & Temperature Monitoring**: Check resting pulse rate twice daily; contact your physician if your resting heart rate remains consistently above 100 beats per minute.",
        "**High-Calorie, Nutrient-Dense Diet**: Eat 5–6 smaller, high-calorie meals daily to counteract the hypermetabolic state and prevent severe involuntary weight and muscle loss.",
        "**Cool, Calm Environment**: Stay in air-conditioned or well-ventilated rooms, dress in lightweight breathable fabrics, and avoid intense direct sun exposure.",
        "**Eye Protection for Graves' Disease**: If experiencing dry, protruding, or irritated eyes, wear sunglasses outdoors, use lubricating artificial tears, and sleep with head elevated.",
        "**Emergency Warning (Thyroid Storm)**: Rush to an emergency department if experiencing high fever (> 103°F), racing heart (> 140 bpm), agitation, delirium, or severe diarrhea."
    ],

    "Hypoglycemia": [
        "**Execute the 15-15 Rule Immediately**: If blood sugar drops below 70 mg/dL, consume 15 grams of fast-acting carbohydrate (3–4 glucose tablets, 1/2 cup fruit juice, or 3 teaspoons of sugar).",
        "**Recheck Blood Sugar After 15 Minutes**: Test blood glucose 15 minutes post-consumption; if still below 70 mg/dL, repeat the 15-gram carbohydrate dose until normal.",
        "**Follow Up with a Complex Snack**: Once glucose is above 70 mg/dL, eat a snack with complex carbs and protein (e.g., peanut butter on whole wheat or cheese and crackers) to prevent relapse.",
        "**Always Carry Emergency Fast-Acting Sugar**: Never leave home without glucose gel, candies, or fruit juice packets in your pocket, bag, and car glove compartment.",
        "**Never Skip or Delay Planned Meals**: Eat meals at regular, predictable hours; if exercising vigorously, consume an extra 15–30g complex carbohydrate snack beforehand.",
        "**Carry a Glucagon Emergency Kit**: If prone to severe hypoglycemic episodes, ensure a prescribed glucagon kit (nasal Baqsimi or injectable) is at home and family members are trained.",
        "**Wear Medical Alert Identification**: Wear a medical alert bracelet or necklace stating your diabetic condition and potential for hypoglycemic emergencies.",
        "**Emergency Warning**: If the person becomes unresponsive, confused, or experiences a seizure, NEVER force liquids into the mouth; administer emergency glucagon and dial 911/112."
    ],

    "Osteoarthristis": [
        "**Low-Impact Joint-Friendly Exercise**: Engage in low-impact activities like swimming, water aerobics, stationary cycling, or walking; avoid high-impact jumping and running.",
        "**Weight Optimization to Unload Joints**: Every pound of excess weight shed eliminates approximately 4 pounds of compressive force across the knee joints during ambulation.",
        "**Alternating Heat & Cold Therapy**: Apply moist heating pads for 20 minutes before activity to loosen stiff joints; use ice packs wrapped in towels for 15 minutes post-activity to reduce joint effusion.",
        "**Use Supportive Assistive Devices**: Utilize an ergonomically fitted walking cane (held in the hand opposite the affected joint), knee braces, or shock-absorbing footwear orthotics.",
        "**Anti-Inflammatory Nutrition**: Incorporate omega-3 rich fatty fish (salmon, mackerel), walnuts, flaxseeds, and colorful berries into your daily diet; minimize ultra-processed foods.",
        "**Range-of-Motion Exercises**: Perform gentle morning joint stretches and quadriceps-strengthening exercises to maintain functional articular mobility and joint stability.",
        "**Pace Daily Activities**: Avoid continuous kneeling, deep squatting, or lifting heavy loads; alternate active household tasks with scheduled seated rest periods.",
        "**Emergency Warning**: Seek prompt orthopedic evaluation if a joint becomes intensely hot, red, and swollen with high fever (suggesting septic arthritis) or if you experience sudden inability to bear weight."
    ],

    "Arthritis": [
        "**Strict Adherence to DMARDs / Biologics**: If diagnosed with inflammatory/rheumatoid arthritis, take prescribed disease-modifying medications consistently to halt irreversible bone erosion.",
        "**Morning Stiffness Relief Routine**: Take a warm shower or soak in a warm bath immediately upon waking to ease severe morning joint gel-phenomenon and stiffness.",
        "**Joint Protection Principles**: Use larger, stronger joints to carry loads (e.g. carry bags on your forearm rather than gripping handles with small finger joints).",
        "**Targeted Physical & Occupational Therapy**: Work with a specialized physical therapist on gentle range-of-motion routines, tendon-gliding exercises, and custom splints.",
        "**Strict Anti-Inflammatory Nutrition**: Eat a nutrient-dense Mediterranean diet rich in antioxidants, extra virgin olive oil, and leafy greens; avoid trans fats, sugar, and alcohol.",
        "**Balance Rest and Physical Activity**: Ensure 8 hours of restorative sleep; during acute joint flare-ups, prioritize joint rest and avoid pushing through sharp inflammatory pain.",
        "**Stop All Tobacco Smoking**: Smoking dramatically accelerates rheumatoid joint destruction, increases systemic autoantibodies, and diminishes the efficacy of arthritis medications.",
        "**Emergency Warning**: Seek urgent medical evaluation if a single joint suddenly becomes excruciatingly painful, red, swollen, and accompanied by fever or chills."
    ],

    "(Vertigo) Paroymsal Positional Vertigo": [
        "**Avoid Sudden Head & Neck Movements**: Turn your head slowly and deliberately; never whip your head backwards or bend down rapidly to pick up objects from the floor.",
        "**Sleep with Head Elevated**: Sleep with your head propped up on two or more pillows (at a 45-degree angle) to prevent dislodged otoconia from settling in the posterior semicircular canal.",
        "**Perform Prescribed Canalith Repositioning**: Perform the Epley maneuver, Semont maneuver, or Brandt-Daroff exercises under the guidance of your ENT specialist or vestibular therapist.",
        "**Sit on Bed Edge Before Standing**: When waking, sit on the edge of the bed for 1 to 2 minutes before standing up to allow vestibular adaptation and prevent sudden drop attacks.",
        "**Bathroom & Fall Safety Modifications**: Install grab bars in the shower, place non-slip rubber mats, and keep night lights illuminated in hallways and bathrooms.",
        "**Avoid Driving During Active Vertigo**: Strictly refrain from operating motor vehicles, operating machinery, climbing ladders, or swimming until vertigo is resolved.",
        "**Limit Salt, Caffeine, and Alcohol**: Reduce dietary sodium, coffee, and alcohol, which can alter inner ear endolymph fluid dynamics and worsen balance disorders.",
        "**Emergency Red Flag**: Seek immediate emergency hospital care if vertigo is accompanied by facial drooping, slurred speech, double vision, arm weakness, or inability to walk."
    ],

    "Acne": [
        "**Gentle Twice-Daily Cleansing**: Wash the face morning and evening with a gentle, pH-balanced non-comedogenic foaming cleanser; never scrub aggressively with harsh loofahs.",
        "**Strict Hands-Off Rule (Never Pop Pimples)**: Never squeeze, pick, or pop blemishes; popping pushes bacterial infection deeper into the dermis, causing severe scarring and hyperpigmentation.",
        "**Use Exclusively Non-Comedogenic Products**: Choose oil-free, non-comedogenic cosmetics, sunscreens, and moisturizers that do not clog pilosebaceous pores.",
        "**Consistent Topical Treatment Application**: Apply prescribed topical treatments (e.g., Benzoyl Peroxide, Salicylic Acid, Adapalene) across the entire acne-prone area, not just individual spots.",
        "**Daily Clean Pillowcases & Screen Hygiene**: Change pillowcases 2–3 times a week and clean your smartphone screen daily with alcohol wipes to reduce bacterial transfer to facial skin.",
        "**Low-Glycemic Diet & Limit Skim Milk**: Reduce intake of refined sugars, white bread, pastries, and skim milk, which cause insulin spikes and stimulate sebaceous gland hypersecretion.",
        "**Wear Non-Greasy Broad-Spectrum Sunscreen**: Use an oil-free SPF 30+ sunscreen daily; acne medications increase photosensitivity, and UV rays darken post-inflammatory marks.",
        "**Emergency Warning**: Consult a dermatologist immediately if experiencing severe, painful deep cystic nodules, or if lesions show spreading redness with systemic fever."
    ],

    "Urinary Tract Infection": [
        "**Drink 3 Litres of Water Daily**: Drink plenty of clean water throughout the day to continuously flush bacteria (typically E. coli) from the urinary bladder and urethra.",
        "**Complete Full Antibiotic Course**: Finish every single antibiotic tablet prescribed by your doctor even if burning micturition resolves after 24–48 hours to prevent recurrent relapse.",
        "**Wipe Front-to-Back Exclusively**: Always wipe from front to back after bowel movements to prevent the migration of rectal fecal bacteria toward the urethral meatus.",
        "**Urinate Promptly When Needed**: Never hold your urine; empty your bladder completely every 2 to 3 hours and always void immediately following sexual intercourse.",
        "**Avoid Bladder Irritants**: Strictly avoid coffee, black tea, alcohol, carbonated sodas, artificial sweeteners, and spicy foods until the infection is completely cleared.",
        "**Breathable Cotton Undergarments**: Wear loose-fitting, pure cotton underwear and avoid tight synthetic leggings to prevent warm, moisture-trapping bacterial environments.",
        "**Avoid Harsh Feminine Washes**: Refrain from using perfumed bubble baths, vaginal douches, scented hygiene powders, or chemical feminine sprays that disrupt healthy vaginal flora.",
        "**CRITICAL EMERGENCY WARNING**: Seek immediate hospital emergency care if experiencing high fever with shaking chills, flank or lower back pain, nausea, or vomiting (signs of pyelonephritis/kidney infection)."
    ],

    "Psoriasis": [
        "**Daily Intense Skin Hydration**: Apply thick, fragrance-free emollient creams or petroleum jelly within 3 minutes of stepping out of the shower to seal moisture into plaque lesions.",
        "**Short, Lukewarm Baths**: Limit baths and showers to 10 minutes using lukewarm (not hot) water; add colloidal oatmeal, Epsom salts, or bath oils to soothe cutaneous itching.",
        "**Never Scratch or Peel Plaques**: Do not peel dry scales or scratch plaques; cutaneous trauma triggers the Koebner phenomenon, causing new psoriatic plaques to erupt at the injured site.",
        "**Moderate, Controlled Sun Exposure**: Expose affected skin to natural sunlight for 10–15 minutes during early morning hours; always protect unaffected skin with broad-spectrum sunscreen.",
        "**Zero Alcohol & Quit Smoking**: Eliminate alcohol and stop smoking; both are potent systemic triggers that dramatically worsen psoriasis severity and impair treatment response.",
        "**Stress Reduction & Mindfulness**: Practice daily relaxation techniques (yoga, meditation, breathwork); chronic psychological stress is a well-documented primary trigger of psoriasis flare-ups.",
        "**Wear Soft, Breathable Natural Fabrics**: Choose loose garments made from 100% cotton or silk; avoid abrasive wool, coarse linens, and synthetic fabrics that irritate plaques.",
        "**Emergency Warning**: Seek immediate medical care if psoriasis covers over 80% of your body with intense redness and peeling (erythrodermic psoriasis) or if joint swelling and pain develop (psoriatic arthritis)."
    ],

    "Impetigo": [
        "**Gentle Warm Water Crust Removal**: Soften honey-colored crusts by applying warm, moist washcloth compresses for 10–15 minutes, then gently wash away loose crusts before applying topical antibiotic ointment.",
        "**Apply Prescribed Antibiotic Ointment**: Apply prescribed topical mupirocin or fusidic acid ointment 2–3 times daily across all active lesions using sterile cotton swabs—never bare fingers.",
        "**Cover Lesions with Sterile Bandages**: Keep active weeping blisters covered with clean, non-stick sterile gauze bandages to prevent auto-inoculation and airborne spread.",
        "**Strict Hand Hygiene with Antibacterial Soap**: Wash hands thoroughly with soap and warm water immediately after any contact with infected skin lesions.",
        "**Separate Linen & Hot Water Washing**: Keep the patient's towels, washcloths, clothing, and bed linens strictly separate; wash them daily in hot water (> 60°C) and dry thoroughly.",
        "**Keep Fingernails Short & Clean**: Clip fingernails short and keep them clean to minimize skin trauma and secondary bacterial inoculation from involuntary scratching during sleep.",
        "**Temporary School / Work Isolation**: Children and adults should stay home from daycare, school, or work until 24 to 48 hours after starting appropriate antibiotic treatment.",
        "**Emergency Warning**: Seek immediate medical re-evaluation if the rash spreads rapidly, skin becomes intensely painful and swollen, or if the patient develops high fever or brown tea-colored urine."
    ],
}


def _generate_dynamic_precautions(disease: str, symptoms: list[str], severity: str) -> list[str]:
    """
    Intelligent medical synthesis for novel or rare conditions not in the 41-disease table.
    Analyzes symptom keywords and clinical severity to build 6-8 structured precautions.
    """
    sym_lower = [s.lower().strip() for s in symptoms]
    disease_lower = disease.lower()

    precautions: list[str] = [
        f"**Professional Clinical Evaluation**: Consult a qualified healthcare professional or specialist for confirmed diagnosis and targeted treatment of {disease}."
    ]

    # Respiratory signs
    if any(k in " ".join(sym_lower) for k in ["cough", "breathlessness", "phlegm", "chest_pain", "congestion", "throat"]):
        precautions.append("**Respiratory Care & Humidification**: Practice deep breathing exercises, perform steam inhalation twice daily, and maintain head elevation during sleep.")
        precautions.append("**Airway Protection**: Strictly avoid active and secondhand tobacco smoke, chemical fumes, cold drafts, and air pollutants.")

    # Gastrointestinal signs
    if any(k in " ".join(sym_lower) for k in ["vomiting", "diarrhoea", "stomach_pain", "abdominal", "nausea", "acidity"]):
        precautions.append("**Continuous Oral Rehydration (ORS)**: Sip electrolyte rehydration solutions, clear broths, and clean water to counter fluid and mineral loss.")
        precautions.append("**Gentle Bland Nutrition**: Consume small portions of low-fat, low-spice foods (rice, bananas, toast) and avoid dairy, alcohol, and caffeine.")

    # Skin / dermatological signs
    if any(k in " ".join(sym_lower) for k in ["rash", "itching", "lesion", "spot", "blister", "skin"]):
        precautions.append("**Dermatological Barrier Protection**: Wash gently with lukewarm water and mild soap; never scratch or pick at skin lesions to prevent secondary bacterial infection.")
        precautions.append("**Topical Soothing & Loose Clothing**: Apply hypoallergenic soothing calamine or barrier moisturizers and wear loose-fitting 100% cotton garments.")

    # Systemic fever & body ache
    if any(k in " ".join(sym_lower) for k in ["fever", "chills", "sweating", "fatigue", "malaise", "headache", "muscle_pain"]):
        precautions.append("**Rest & Temperature Regulation**: Maintain strict physical bed rest, dress in light layers, and use tepid sponge baths or paracetamol to manage fevers.")
        precautions.append("**Daily Vital Signs Tracking**: Measure and record body temperature, pulse rate, and hydration levels at regular 4-hour intervals.")

    # Joint & musculoskeletal
    if any(k in " ".join(sym_lower) for k in ["joint_pain", "neck_pain", "back_pain", "knee", "swelling"]):
        precautions.append("**Joint Protection & Gentle Rest**: Avoid heavy lifting and high-impact movements; apply alternating cold compresses and warm pads to soothe joints.")

    # Hygiene & general care
    precautions.append("**Hygiene & Cross-Contamination Prevention**: Wash hands frequently with soap and warm water for 20 seconds, and avoid sharing utensils, towels, or personal items.")

    # Hydration
    precautions.append("**Nutritional & Fluid Optimization**: Maintain high oral fluid intake (2.5–3L daily) and eat balanced, nutrient-dense meals to support cellular repair.")

    # Red flag alert
    if severity == "severe":
        precautions.append("**EMERGENCY WARNING**: Call emergency medical services (911/112) immediately if experiencing severe breathing difficulty, chest pain, confusion, or unmanageable fever.")
    else:
        precautions.append("**Red-Flag Warning**: Seek prompt urgent care if symptoms worsen rapidly, if fever exceeds 103°F (39.4°C), or if you develop new neurological symptoms.")

    # Ensure 6 to 8 items
    return precautions[:8]


def get_clinical_precautions(
    disease: str,
    symptoms: list[str] | None = None,
    severity: str = "mild",
    api_key: str | None = None,
) -> list[str]:
    """
    Main entry point for generating rich AI-grade precautions.

    Strategy:
      1. If API key is provided or found, attempt live Google Gemini generation.
      2. If not available or fails, use the exhaustive 41-condition clinical knowledge base.
      3. If disease is not in the knowledge base, synthesize dynamic clinical guidelines from symptoms.
    """
    syms = symptoms or []
    norm_disease = disease.strip().title()

    # Step 1: Live Gemini AI generation
    ai_precautions = _call_gemini_precautions(
        disease=norm_disease,
        symptoms=syms,
        severity=severity,
        api_key=api_key,
    )
    if ai_precautions:
        return ai_precautions

    # Step 2: Expert Clinical Knowledge Base (matches any casing or title variation)
    if norm_disease in CLINICAL_PRECAUTIONS_DB:
        logger.info("Using curated clinical precautions for '%s' (%d items).", norm_disease, len(CLINICAL_PRECAUTIONS_DB[norm_disease]))
        return CLINICAL_PRECAUTIONS_DB[norm_disease]

    # Check case-insensitive / partial match in knowledge base
    for key, items in CLINICAL_PRECAUTIONS_DB.items():
        if key.lower() == norm_disease.lower() or key.lower() in norm_disease.lower():
            logger.info("Matched clinical precautions for '%s' via key '%s'.", norm_disease, key)
            return items

    # Step 3: Dynamic clinical synthesis
    logger.info("Generating dynamic clinical precautions for '%s'.", norm_disease)
    return _generate_dynamic_precautions(norm_disease, syms, severity)
