"""
herbal_kb.py — VitalSense AI | Module 7: Natural & Herbal Remedies
====================================================================
Curated knowledge base mapping diseases to herbal/natural remedies
and a comprehensive herb encyclopedia.

Data sources: Traditional Ayurveda, AYUSH guidelines, WHO Monographs
on Medicinal Plants, PubMed-indexed clinical studies.

Author: VitalSense AI Team
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Disclaimer — appended to every response
# ──────────────────────────────────────────────────────────────────────────────
DISCLAIMER: str = (
    "These herbal suggestions are for informational purposes only and are NOT "
    "a substitute for professional medical advice. Consult a qualified "
    "healthcare practitioner before using any herbal remedy, especially if you "
    "are pregnant, nursing, taking medications, or have a chronic condition."
)

# ──────────────────────────────────────────────────────────────────────────────
# Severity guard — diseases that MUST NOT receive herbal-only advice
# ──────────────────────────────────────────────────────────────────────────────
SEVERE_CONDITIONS: set[str] = {
    "heart attack", "paralysis (brain hemorrhage)", "aids", "tuberculosis",
    "pneumonia", "dengue", "malaria", "typhoid", "hepatitis b", "hepatitis c",
    "hepatitis d", "alcoholic hepatitis", "jaundice", "chicken pox",
    "drug reaction", "chronic cholestasis",
}

MODERATE_CONDITIONS: set[str] = {
    "peptic ulcer disease", "hepatitis a", "hepatitis e", "gastroenteritis",
    "cervical spondylosis", "impetigo",
}


def _normalise(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ──────────────────────────────────────────────────────────────────────────────
# HERB DATABASE — 25+ medicinal plants with full profiles
# ──────────────────────────────────────────────────────────────────────────────
HERB_DATABASE: dict[str, dict[str, Any]] = {
    "tulsi": {
        "common_name": "Tulsi (Holy Basil)",
        "scientific_name": "Ocimum tenuiflorum",
        "family": "Lamiaceae",
        "origin": "Indian subcontinent",
        "parts_used": "Leaves, seeds, stems",
        "emoji": "🌿",
        "key_compounds": ["Eugenol", "Ursolic acid", "Rosmarinic acid", "Linalool"],
        "therapeutic_properties": [
            "Adaptogenic", "Anti-inflammatory", "Antimicrobial",
            "Antioxidant", "Immunomodulatory", "Antipyretic"
        ],
        "traditional_uses": "Respiratory infections, fever, stress, digestive issues, skin disorders",
        "contraindications": "May slow blood clotting; avoid 2 weeks before surgery. May lower blood sugar — monitor if diabetic.",
        "evidence_level": "Well-Studied",
        "description": "Sacred herb of India, Tulsi has been used in Ayurveda for over 5,000 years. Known as the 'Queen of Herbs', it supports immunity, respiratory health, and stress management."
    },
    "turmeric": {
        "common_name": "Turmeric (Haldi)",
        "scientific_name": "Curcuma longa",
        "family": "Zingiberaceae",
        "origin": "South Asia",
        "parts_used": "Rhizome (root)",
        "emoji": "🟡",
        "key_compounds": ["Curcumin", "Demethoxycurcumin", "Turmerone"],
        "therapeutic_properties": [
            "Anti-inflammatory", "Antioxidant", "Antimicrobial",
            "Hepatoprotective", "Analgesic", "Wound healing"
        ],
        "traditional_uses": "Joint pain, inflammation, wound healing, digestive issues, skin conditions",
        "contraindications": "High doses may cause GI upset. May interact with blood thinners and diabetes medications.",
        "evidence_level": "Well-Studied",
        "description": "The golden spice of India, turmeric contains curcumin — one of the most studied natural anti-inflammatory compounds. Used in Ayurveda, Siddha, and Unani medicine systems."
    },
    "ginger": {
        "common_name": "Ginger (Adrak)",
        "scientific_name": "Zingiber officinale",
        "family": "Zingiberaceae",
        "origin": "Southeast Asia",
        "parts_used": "Rhizome (root)",
        "emoji": "🫚",
        "key_compounds": ["Gingerol", "Shogaol", "Zingerone", "Paradol"],
        "therapeutic_properties": [
            "Antiemetic", "Anti-inflammatory", "Analgesic",
            "Digestive stimulant", "Antioxidant", "Carminative"
        ],
        "traditional_uses": "Nausea, digestive disorders, cold, cough, pain, inflammation",
        "contraindications": "May interact with blood thinners. Excess may cause heartburn. Use cautiously during pregnancy (small amounts OK).",
        "evidence_level": "Well-Studied",
        "description": "One of the most widely used spices globally, ginger has powerful anti-nausea and anti-inflammatory properties backed by extensive clinical research."
    },
    "neem": {
        "common_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "family": "Meliaceae",
        "origin": "Indian subcontinent",
        "parts_used": "Leaves, bark, oil, seeds",
        "emoji": "🌳",
        "key_compounds": ["Azadirachtin", "Nimbin", "Nimbidin", "Gedunin"],
        "therapeutic_properties": [
            "Antimicrobial", "Antifungal", "Anti-inflammatory",
            "Antiparasitic", "Blood purifier", "Immunostimulant"
        ],
        "traditional_uses": "Skin infections, acne, fungal infections, blood purification, dental health",
        "contraindications": "Not for internal use in large doses. Avoid during pregnancy. May affect fertility. Not for children under 2.",
        "evidence_level": "Well-Studied",
        "description": "Called the 'Village Pharmacy' in India, neem has broad-spectrum antimicrobial activity. Every part of the tree has medicinal value."
    },
    "aloe_vera": {
        "common_name": "Aloe Vera (Ghritkumari)",
        "scientific_name": "Aloe barbadensis miller",
        "family": "Asphodelaceae",
        "origin": "Arabian Peninsula",
        "parts_used": "Leaf gel, latex",
        "emoji": "🪴",
        "key_compounds": ["Acemannan", "Aloin", "Barbaloin", "Anthraquinones"],
        "therapeutic_properties": [
            "Wound healing", "Anti-inflammatory", "Moisturising",
            "Laxative (latex)", "Antimicrobial", "Soothing"
        ],
        "traditional_uses": "Burns, skin conditions, constipation, digestive health, wound healing",
        "contraindications": "Latex component is a strong laxative — avoid in pregnancy and IBD. Topical use is generally safe.",
        "evidence_level": "Well-Studied",
        "description": "Known as the 'plant of immortality' in ancient Egypt, aloe vera gel is one of the most versatile natural remedies for skin health and digestive support."
    },
    "ashwagandha": {
        "common_name": "Ashwagandha (Indian Ginseng)",
        "scientific_name": "Withania somnifera",
        "family": "Solanaceae",
        "origin": "India, North Africa",
        "parts_used": "Root, leaves",
        "emoji": "💪",
        "key_compounds": ["Withanolides", "Withaferin A", "Sitoindosides"],
        "therapeutic_properties": [
            "Adaptogenic", "Anti-stress", "Immunomodulatory",
            "Anti-inflammatory", "Neuroprotective", "Thyroid support"
        ],
        "traditional_uses": "Stress, anxiety, insomnia, fatigue, weakness, thyroid imbalance",
        "contraindications": "Avoid in pregnancy. May interact with thyroid, blood pressure, and diabetes medications. May cause drowsiness.",
        "evidence_level": "Well-Studied",
        "description": "Premier Ayurvedic adaptogen used for 3,000+ years. Modern research confirms its stress-reducing, strength-enhancing, and neuroprotective properties."
    },
    "amla": {
        "common_name": "Amla (Indian Gooseberry)",
        "scientific_name": "Phyllanthus emblica",
        "family": "Phyllanthaceae",
        "origin": "Indian subcontinent",
        "parts_used": "Fruit",
        "emoji": "🟢",
        "key_compounds": ["Vitamin C (highest natural source)", "Gallic acid", "Ellagic acid", "Tannins"],
        "therapeutic_properties": [
            "Antioxidant", "Immunomodulatory", "Anti-inflammatory",
            "Hepatoprotective", "Anti-diabetic", "Hair/skin nourishing"
        ],
        "traditional_uses": "Immunity boosting, hair health, digestion, diabetes management, liver health",
        "contraindications": "May lower blood sugar — monitor if diabetic. High vitamin C may interact with certain medications.",
        "evidence_level": "Well-Studied",
        "description": "Amla contains 20× more vitamin C than oranges. A cornerstone of Ayurveda, it is one of the three fruits in the revered Triphala formulation."
    },
    "giloy": {
        "common_name": "Giloy (Guduchi)",
        "scientific_name": "Tinospora cordifolia",
        "family": "Menispermaceae",
        "origin": "India, Myanmar, Sri Lanka",
        "parts_used": "Stem, leaves, root",
        "emoji": "🍃",
        "key_compounds": ["Berberine", "Tinosporin", "Giloin", "Cordifolioside"],
        "therapeutic_properties": [
            "Immunomodulatory", "Antipyretic", "Anti-inflammatory",
            "Antioxidant", "Hepatoprotective", "Anti-allergic"
        ],
        "traditional_uses": "Fever, immunity, allergies, liver health, blood sugar regulation",
        "contraindications": "May lower blood sugar excessively. Avoid in autoimmune conditions. Not for use during pregnancy.",
        "evidence_level": "Preliminary Studies",
        "description": "Known as 'Amrita' (nectar of immortality) in Ayurveda. Giloy gained mainstream popularity during the COVID-19 pandemic for its immunity-boosting properties."
    },
    "triphala": {
        "common_name": "Triphala",
        "scientific_name": "Blend of Amalaki, Bibhitaki, Haritaki",
        "family": "Multiple families",
        "origin": "India",
        "parts_used": "Dried fruit powder (3 fruits)",
        "emoji": "🫙",
        "key_compounds": ["Gallic acid", "Chebulagic acid", "Chebulinic acid", "Flavonoids"],
        "therapeutic_properties": [
            "Digestive tonic", "Mild laxative", "Antioxidant",
            "Anti-inflammatory", "Detoxifying", "Prebiotic"
        ],
        "traditional_uses": "Constipation, digestion, detoxification, eye health, immunity",
        "contraindications": "Avoid during pregnancy and diarrhea. Start with low doses. May cause loose stools initially.",
        "evidence_level": "Preliminary Studies",
        "description": "The most famous Ayurvedic formulation — a synergistic blend of Amla, Bibhitaki, and Haritaki. Used for digestive health and gentle detoxification for millennia."
    },
    "brahmi": {
        "common_name": "Brahmi",
        "scientific_name": "Bacopa monnieri",
        "family": "Plantaginaceae",
        "origin": "India, wetlands of Asia",
        "parts_used": "Whole plant, leaves",
        "emoji": "🧠",
        "key_compounds": ["Bacosides A & B", "Bacopasaponins", "Alkaloids"],
        "therapeutic_properties": [
            "Nootropic", "Anxiolytic", "Memory enhancing",
            "Neuroprotective", "Antioxidant", "Adaptogenic"
        ],
        "traditional_uses": "Memory enhancement, anxiety, concentration, neurological support",
        "contraindications": "May cause GI upset. May interact with thyroid medications. Can increase secretions in lungs.",
        "evidence_level": "Well-Studied",
        "description": "Named after Lord Brahma (creator of the universe), Brahmi is Ayurveda's premier brain tonic. Multiple clinical trials confirm memory and cognitive benefits."
    },
    "mulethi": {
        "common_name": "Mulethi (Licorice)",
        "scientific_name": "Glycyrrhiza glabra",
        "family": "Fabaceae",
        "origin": "Southern Europe, Western Asia",
        "parts_used": "Root",
        "emoji": "🪵",
        "key_compounds": ["Glycyrrhizin", "Liquiritigenin", "Glabridin"],
        "therapeutic_properties": [
            "Demulcent", "Expectorant", "Anti-inflammatory",
            "Antimicrobial", "Adrenal support", "Soothing"
        ],
        "traditional_uses": "Sore throat, cough, acidity, peptic ulcer, skin lightening",
        "contraindications": "Prolonged use may raise blood pressure and lower potassium. Avoid in hypertension, kidney disease, pregnancy.",
        "evidence_level": "Well-Studied",
        "description": "One of the oldest and most widely used herbs worldwide. Its sweet root is 50× sweeter than sugar and has powerful soothing properties for the throat and stomach."
    },
    "peppermint": {
        "common_name": "Peppermint (Pudina)",
        "scientific_name": "Mentha piperita",
        "family": "Lamiaceae",
        "origin": "Europe, Middle East",
        "parts_used": "Leaves, essential oil",
        "emoji": "🌱",
        "key_compounds": ["Menthol", "Menthone", "Menthyl acetate", "Rosmarinic acid"],
        "therapeutic_properties": [
            "Carminative", "Antispasmodic", "Decongestant",
            "Analgesic (topical)", "Cooling", "Digestive"
        ],
        "traditional_uses": "Indigestion, headache, nasal congestion, nausea, IBS symptoms",
        "contraindications": "May worsen GERD/acid reflux. Essential oil should not be applied near infant/child faces. May interact with certain medications.",
        "evidence_level": "Well-Studied",
        "description": "The menthol in peppermint provides instant cooling relief. Peppermint oil capsules are an evidence-based treatment for IBS in many countries."
    },
    "chamomile": {
        "common_name": "Chamomile (Babune ka Phool)",
        "scientific_name": "Matricaria chamomilla",
        "family": "Asteraceae",
        "origin": "Europe, Western Asia",
        "parts_used": "Dried flowers",
        "emoji": "🌼",
        "key_compounds": ["Apigenin", "Bisabolol", "Chamazulene", "Matricin"],
        "therapeutic_properties": [
            "Anxiolytic", "Sedative (mild)", "Anti-inflammatory",
            "Antispasmodic", "Digestive aid", "Wound healing"
        ],
        "traditional_uses": "Anxiety, insomnia, digestive cramps, skin inflammation, wound healing",
        "contraindications": "Allergy risk for those allergic to ragweed/daisy family. May interact with blood thinners and sedatives.",
        "evidence_level": "Well-Studied",
        "description": "One of the most ancient medicinal herbs, chamomile tea is a globally trusted remedy for relaxation and digestive comfort."
    },
    "eucalyptus": {
        "common_name": "Eucalyptus (Nilgiri)",
        "scientific_name": "Eucalyptus globulus",
        "family": "Myrtaceae",
        "origin": "Australia",
        "parts_used": "Leaves, essential oil",
        "emoji": "🌲",
        "key_compounds": ["Eucalyptol (1,8-cineole)", "α-Pinene", "Limonene"],
        "therapeutic_properties": [
            "Decongestant", "Antimicrobial", "Anti-inflammatory",
            "Expectorant", "Analgesic", "Antiseptic"
        ],
        "traditional_uses": "Respiratory congestion, cold, cough, muscle pain, wound antiseptic",
        "contraindications": "Essential oil should NEVER be ingested. Keep away from young children's faces. May trigger asthma in sensitive individuals.",
        "evidence_level": "Well-Studied",
        "description": "Eucalyptus oil is a powerful decongestant used in steam inhalation worldwide. Its antimicrobial properties make it a staple in natural medicine chests."
    },
    "cinnamon": {
        "common_name": "Cinnamon (Dalchini)",
        "scientific_name": "Cinnamomum verum / C. cassia",
        "family": "Lauraceae",
        "origin": "Sri Lanka, South India",
        "parts_used": "Bark",
        "emoji": "🫕",
        "key_compounds": ["Cinnamaldehyde", "Eugenol", "Linalool", "Coumarin"],
        "therapeutic_properties": [
            "Anti-diabetic", "Antimicrobial", "Anti-inflammatory",
            "Antioxidant", "Warming", "Carminative"
        ],
        "traditional_uses": "Blood sugar regulation, cold, cough, digestion, oral health",
        "contraindications": "Cassia cinnamon contains coumarin — limit intake to avoid liver strain. May lower blood sugar — monitor if diabetic.",
        "evidence_level": "Preliminary Studies",
        "description": "Cinnamon has shown promising results in blood sugar regulation. Ceylon cinnamon (true cinnamon) is preferred over Cassia for medicinal use."
    },
    "fenugreek": {
        "common_name": "Fenugreek (Methi)",
        "scientific_name": "Trigonella foenum-graecum",
        "family": "Fabaceae",
        "origin": "Mediterranean, South Asia",
        "parts_used": "Seeds, leaves",
        "emoji": "🌾",
        "key_compounds": ["Diosgenin", "Galactomannan", "4-Hydroxyisoleucine", "Trigonelline"],
        "therapeutic_properties": [
            "Anti-diabetic", "Galactagogue", "Anti-inflammatory",
            "Cholesterol-lowering", "Digestive", "Appetite stimulant"
        ],
        "traditional_uses": "Diabetes management, lactation support, digestion, joint pain, hair health",
        "contraindications": "May lower blood sugar significantly. Avoid in pregnancy (uterine stimulant). May interact with blood thinners.",
        "evidence_level": "Well-Studied",
        "description": "Methi seeds soaked overnight are a time-tested Indian remedy for blood sugar control. Clinical studies support their anti-diabetic and cholesterol-lowering effects."
    },
    "honey": {
        "common_name": "Honey (Shahad)",
        "scientific_name": "Apis mellifera (product)",
        "family": "Apidae (bee product)",
        "origin": "Worldwide",
        "parts_used": "Raw, unprocessed honey",
        "emoji": "🍯",
        "key_compounds": ["Hydrogen peroxide", "Methylglyoxal (Manuka)", "Flavonoids", "Phenolic acids"],
        "therapeutic_properties": [
            "Antimicrobial", "Wound healing", "Demulcent",
            "Antioxidant", "Cough suppressant", "Prebiotic"
        ],
        "traditional_uses": "Cough suppression, wound healing, sore throat, burns, energy boost",
        "contraindications": "NEVER give to infants under 1 year (botulism risk). Diabetics should use cautiously due to sugar content.",
        "evidence_level": "Well-Studied",
        "description": "Clinical studies show honey is as effective as OTC cough suppressants. Manuka honey has additional antimicrobial properties for wound care."
    },
    "garlic": {
        "common_name": "Garlic (Lahsun)",
        "scientific_name": "Allium sativum",
        "family": "Amaryllidaceae",
        "origin": "Central Asia",
        "parts_used": "Bulb (cloves)",
        "emoji": "🧄",
        "key_compounds": ["Allicin", "Ajoene", "Diallyl disulfide", "S-allyl cysteine"],
        "therapeutic_properties": [
            "Antimicrobial", "Cardiovascular protective", "Anti-inflammatory",
            "Immune boosting", "Antihypertensive", "Cholesterol-lowering"
        ],
        "traditional_uses": "Cold/flu prevention, blood pressure, cholesterol, infections, immune support",
        "contraindications": "May interact with blood thinners and HIV medications. Raw garlic may cause GI irritation. Stop 2 weeks before surgery.",
        "evidence_level": "Well-Studied",
        "description": "Allicin, released when garlic is crushed, is a potent antimicrobial. Regular garlic consumption is linked to reduced cardiovascular risk in numerous studies."
    },
    "black_pepper": {
        "common_name": "Black Pepper (Kali Mirch)",
        "scientific_name": "Piper nigrum",
        "family": "Piperaceae",
        "origin": "South India (Kerala)",
        "parts_used": "Dried fruit (peppercorn)",
        "emoji": "⚫",
        "key_compounds": ["Piperine", "Piperonal", "Chavicine"],
        "therapeutic_properties": [
            "Bioavailability enhancer", "Digestive stimulant", "Anti-inflammatory",
            "Antioxidant", "Thermogenic", "Antimicrobial"
        ],
        "traditional_uses": "Enhancing absorption of other herbs (especially turmeric), digestion, cold, cough",
        "contraindications": "May increase absorption of certain medications. Excessive use may cause GI irritation.",
        "evidence_level": "Well-Studied",
        "description": "The 'King of Spices' — piperine increases curcumin absorption by 2,000%. Always pair black pepper with turmeric for maximum benefit."
    },
    "fennel": {
        "common_name": "Fennel (Saunf)",
        "scientific_name": "Foeniculum vulgare",
        "family": "Apiaceae",
        "origin": "Mediterranean",
        "parts_used": "Seeds, leaves",
        "emoji": "🫛",
        "key_compounds": ["Anethole", "Fenchone", "Estragole", "Flavonoids"],
        "therapeutic_properties": [
            "Carminative", "Antispasmodic", "Digestive",
            "Galactagogue", "Anti-inflammatory", "Breath freshener"
        ],
        "traditional_uses": "Bloating, gas, indigestion, colic in infants, lactation support, fresh breath",
        "contraindications": "Estrogenic effects — avoid in hormone-sensitive conditions. Large doses may interact with certain medications.",
        "evidence_level": "Preliminary Studies",
        "description": "Chewing fennel seeds after meals is one of India's oldest digestive traditions. It relieves bloating, freshens breath, and soothes the stomach."
    },
    "ajwain": {
        "common_name": "Ajwain (Carom Seeds)",
        "scientific_name": "Trachyspermum ammi",
        "family": "Apiaceae",
        "origin": "India, Middle East",
        "parts_used": "Seeds",
        "emoji": "🌰",
        "key_compounds": ["Thymol", "γ-Terpinene", "p-Cymene", "Carvacrol"],
        "therapeutic_properties": [
            "Carminative", "Antispasmodic", "Antimicrobial",
            "Digestive", "Bronchodilator", "Antiparasitic"
        ],
        "traditional_uses": "Stomach ache, gas, bloating, cold, cough, asthma relief",
        "contraindications": "May worsen acid reflux. Use in moderation during pregnancy. May cause mouth ulcers in excess.",
        "evidence_level": "Preliminary Studies",
        "description": "Ajwain water (boiled with seeds) is a trusted Indian grandmother's remedy for stomach aches and gas, especially in children."
    },
    "cumin": {
        "common_name": "Cumin (Jeera)",
        "scientific_name": "Cuminum cyminum",
        "family": "Apiaceae",
        "origin": "Eastern Mediterranean, India",
        "parts_used": "Seeds",
        "emoji": "🟤",
        "key_compounds": ["Cuminaldehyde", "Thymol", "Iron", "Flavonoids"],
        "therapeutic_properties": [
            "Digestive", "Carminative", "Iron-rich",
            "Anti-diabetic", "Antioxidant", "Appetiser"
        ],
        "traditional_uses": "Digestion, bloating, anaemia, weight management, lactation support",
        "contraindications": "Generally very safe. High doses may lower blood sugar. May slow blood clotting in very large quantities.",
        "evidence_level": "Preliminary Studies",
        "description": "Jeera water (cumin water) is one of India's most popular home remedies for digestion and weight management. Rich in iron and antioxidants."
    },
    "clove": {
        "common_name": "Clove (Laung)",
        "scientific_name": "Syzygium aromaticum",
        "family": "Myrtaceae",
        "origin": "Indonesia (Maluku Islands)",
        "parts_used": "Dried flower buds, oil",
        "emoji": "🫘",
        "key_compounds": ["Eugenol", "Acetyl eugenol", "β-Caryophyllene", "Gallic acid"],
        "therapeutic_properties": [
            "Analgesic (dental)", "Antimicrobial", "Anti-inflammatory",
            "Antioxidant", "Antiemetic", "Local anaesthetic"
        ],
        "traditional_uses": "Toothache, oral health, cough, nausea, digestive issues",
        "contraindications": "Clove oil is potent — never apply undiluted to skin. May interact with blood thinners. Avoid in children under 2.",
        "evidence_level": "Well-Studied",
        "description": "Eugenol in clove is a clinically proven dental analgesic. Placing a clove on a painful tooth is one of the world's oldest and most effective home remedies."
    },
    "cardamom": {
        "common_name": "Cardamom (Elaichi)",
        "scientific_name": "Elettaria cardamomum",
        "family": "Zingiberaceae",
        "origin": "South India",
        "parts_used": "Seeds, pods",
        "emoji": "💚",
        "key_compounds": ["1,8-Cineole", "α-Terpinyl acetate", "Linalool", "Limonene"],
        "therapeutic_properties": [
            "Carminative", "Digestive", "Antimicrobial",
            "Antispasmodic", "Breath freshener", "Diuretic"
        ],
        "traditional_uses": "Bad breath, digestion, nausea, detox, respiratory health",
        "contraindications": "Generally very safe. Gallstone patients should consult doctor. May interact with anticoagulants in large amounts.",
        "evidence_level": "Preliminary Studies",
        "description": "The 'Queen of Spices' complements black pepper (the King). Cardamom tea is a traditional remedy for nausea and indigestion across South Asia and the Middle East."
    },
    "shatavari": {
        "common_name": "Shatavari",
        "scientific_name": "Asparagus racemosus",
        "family": "Asparagaceae",
        "origin": "India, Himalayas",
        "parts_used": "Root (tuberous)",
        "emoji": "🌸",
        "key_compounds": ["Shatavarin", "Saponins", "Isoflavones", "Racemosol"],
        "therapeutic_properties": [
            "Adaptogenic", "Galactagogue", "Immunomodulatory",
            "Antioxidant", "Female reproductive tonic", "Demulcent"
        ],
        "traditional_uses": "Female reproductive health, lactation, immunity, acidity, stress",
        "contraindications": "Avoid in hormone-sensitive conditions (estrogen-like effects). Not recommended in kidney disorders or oedema.",
        "evidence_level": "Preliminary Studies",
        "description": "Known as 'She who possesses a hundred husbands', Shatavari is Ayurveda's premier female health tonic, supporting reproductive health and hormonal balance."
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# DISEASE → HERBAL REMEDIES MAPPING (40+ diseases)
# ──────────────────────────────────────────────────────────────────────────────
HERBAL_REMEDIES: dict[str, list[dict[str, Any]]] = {

    # ── Respiratory ──────────────────────────────────────────────────────────
    "common cold": [
        {
            "herb": "tulsi",
            "preparation": "Boil 8-10 fresh Tulsi leaves in 2 cups of water for 5 minutes. Add a pinch of black pepper and half a teaspoon of honey. Drink warm.",
            "dosage": "2-3 cups daily during cold symptoms",
            "benefits": "Relieves nasal congestion, reduces fever, boosts immunity, antimicrobial action",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ginger",
            "preparation": "Crush 1-inch fresh ginger, boil in water for 10 min. Add lemon juice and honey. Drink as 'adrak chai'.",
            "dosage": "3-4 cups daily",
            "benefits": "Clears congestion, anti-inflammatory, warms the body, soothes throat",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Warm one glass of milk. Add ½ tsp turmeric powder and a pinch of black pepper (for absorption). Known as 'Golden Milk' or 'Haldi Doodh'.",
            "dosage": "1 glass at bedtime",
            "benefits": "Anti-inflammatory, immune-boosting, antimicrobial, soothes sore throat",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "honey",
            "preparation": "Take 1 tablespoon raw honey directly or mix in warm water with lemon. Do NOT heat honey above 40°C.",
            "dosage": "2-3 tablespoons daily",
            "benefits": "Soothes throat, suppresses cough, antimicrobial, energy boost",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "eucalyptus",
            "preparation": "Add 3-4 drops eucalyptus oil to a bowl of hot water. Cover head with towel and inhale steam for 5-10 minutes.",
            "dosage": "2-3 times daily during active cold",
            "benefits": "Decongests nasal passages, antimicrobial, soothes airways",
            "evidence_level": "Well-Studied"
        },
    ],

    "cough": [
        {
            "herb": "honey",
            "preparation": "Mix 1 tbsp raw honey with ½ tsp fresh ginger juice. Take directly or dissolve in warm water.",
            "dosage": "3-4 times daily, especially at bedtime",
            "benefits": "Clinically proven cough suppressant — as effective as dextromethorphan in studies",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "mulethi",
            "preparation": "Chew a small piece of mulethi root, or boil 1 tsp mulethi powder in water for 5 min. Strain and drink.",
            "dosage": "2-3 times daily, max 2 weeks",
            "benefits": "Soothes throat, reduces inflammation, expectorant action",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "tulsi",
            "preparation": "Chew 5-6 fresh Tulsi leaves with a pinch of salt, or make Tulsi tea with honey.",
            "dosage": "3-4 times daily",
            "benefits": "Antimicrobial, reduces throat irritation, boosts immunity",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "black_pepper",
            "preparation": "Mix ¼ tsp freshly ground black pepper with 1 tbsp honey. Lick slowly. Known as 'pepper-honey' remedy.",
            "dosage": "2-3 times daily after meals",
            "benefits": "Breaks up chest congestion, enhances absorption, warming effect",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "bronchial asthma": [
        {
            "herb": "tulsi",
            "preparation": "Boil 10-15 Tulsi leaves in water. Add ½ tsp honey. Drink warm twice daily.",
            "dosage": "2 cups daily as maintenance",
            "benefits": "Bronchodilatory effect, anti-inflammatory, reduces respiratory inflammation",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "ginger",
            "preparation": "Mix 1 tsp ginger juice with 1 tsp honey. Take on empty stomach.",
            "dosage": "Once daily in the morning",
            "benefits": "Relaxes airway muscles, anti-inflammatory",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "ajwain",
            "preparation": "Boil 1 tsp ajwain seeds in water. Inhale the steam. Or wrap in cloth and inhale aroma.",
            "dosage": "1-2 times daily",
            "benefits": "Bronchodilatory (thymol content), clears airways, antispasmodic",
            "evidence_level": "Traditional"
        },
    ],

    # ── Digestive ────────────────────────────────────────────────────────────
    "acidity": [
        {
            "herb": "fennel",
            "preparation": "Chew 1 tsp fennel seeds after meals, or soak 1 tsp in water overnight and drink the strained water in the morning.",
            "dosage": "After each meal or 1 glass saunf water in morning",
            "benefits": "Neutralises acid, reduces bloating, soothes stomach lining",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "mulethi",
            "preparation": "Chew a small stick of mulethi root, or take ½ tsp mulethi powder with water before meals.",
            "dosage": "Twice daily before meals, max 2 weeks",
            "benefits": "Coats and protects stomach lining, reduces acid secretion",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "aloe_vera",
            "preparation": "Extract 2 tbsp fresh aloe vera gel (clear inner gel only). Mix with water and drink on empty stomach.",
            "dosage": "Once daily, 30 min before breakfast",
            "benefits": "Soothes inflamed stomach lining, cooling effect, heals irritation",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "cumin",
            "preparation": "Dry roast 1 tsp cumin seeds. Crush and mix with a glass of water. Drink after meals.",
            "dosage": "After meals, 2-3 times daily",
            "benefits": "Stimulates digestive enzymes, reduces gas, neutralises acid",
            "evidence_level": "Traditional"
        },
    ],

    "gerd": [
        {
            "herb": "aloe_vera",
            "preparation": "Drink 30 ml pure aloe vera juice (decolorised, latex-free) 20 min before meals.",
            "dosage": "30 ml twice daily before meals",
            "benefits": "Soothes oesophageal lining, reduces acid reflux symptoms",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "mulethi",
            "preparation": "Take DGL (deglycyrrhizinated licorice) tablets or chew mulethi root 30 min before meals.",
            "dosage": "380-400 mg DGL, 3 times daily before meals",
            "benefits": "Increases mucus production, protects oesophagus and stomach lining",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "chamomile",
            "preparation": "Steep 1 chamomile tea bag or 1 tbsp dried flowers in hot water for 5 min. Drink after meals.",
            "dosage": "2-3 cups daily, after meals",
            "benefits": "Reduces stomach inflammation, antispasmodic, calming",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "constipation": [
        {
            "herb": "triphala",
            "preparation": "Mix 1 tsp Triphala powder in a glass of warm water. Drink at bedtime.",
            "dosage": "1 tsp at bedtime daily",
            "benefits": "Gentle, non-habit-forming laxative. Tones the colon. Prebiotic effect.",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "aloe_vera",
            "preparation": "Take 1-2 tbsp aloe vera juice (with latex) at bedtime. Use sparingly.",
            "dosage": "Once daily at bedtime, short-term only (1-2 weeks)",
            "benefits": "Natural laxative (anthraquinones in latex stimulate bowel movement)",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "fennel",
            "preparation": "Boil 1 tsp fennel seeds + 1 tsp ajwain in water for 5 min. Drink warm.",
            "dosage": "Twice daily — morning and evening",
            "benefits": "Stimulates peristalsis, relieves gas, antispasmodic",
            "evidence_level": "Traditional"
        },
    ],

    "indigestion": [
        {
            "herb": "ginger",
            "preparation": "Grate 1-inch ginger into hot water. Add lemon and a pinch of rock salt. Drink 15 min before meals.",
            "dosage": "Before meals, 2-3 times daily",
            "benefits": "Stimulates digestive enzymes, relieves nausea, carminative",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ajwain",
            "preparation": "Chew ½ tsp ajwain seeds with a pinch of black salt and sip warm water. Or boil in water as ajwain water.",
            "dosage": "After meals or when symptoms appear",
            "benefits": "Rapid relief from gas and bloating (thymol action)",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "peppermint",
            "preparation": "Steep 5-6 fresh peppermint leaves in hot water for 5 min. Drink as tea.",
            "dosage": "1-2 cups after meals",
            "benefits": "Relaxes digestive muscles, relieves cramps, carminative",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "cumin",
            "preparation": "Boil 1 tsp cumin seeds in 2 cups water until reduced to 1 cup. Strain and drink warm.",
            "dosage": "1-2 cups daily",
            "benefits": "Enhances digestive enzyme secretion, reduces bloating",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "gastroenteritis": [
        {
            "herb": "ginger",
            "preparation": "Sip ginger tea (grated fresh ginger in hot water) in small amounts throughout the day.",
            "dosage": "Small sips every 1-2 hours",
            "benefits": "Reduces nausea and vomiting, anti-inflammatory on gut",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "cumin",
            "preparation": "Boil 1 tsp cumin in water with a pinch of salt. Strain and sip slowly when tolerating liquids.",
            "dosage": "As tolerated, 3-4 times daily",
            "benefits": "Restores digestive function, antimicrobial, carminative",
            "evidence_level": "Traditional"
        },
    ],

    "diarrhoea": [
        {
            "herb": "ginger",
            "preparation": "Grate fresh ginger, extract juice. Mix ½ tsp with warm water and a pinch of rock salt.",
            "dosage": "3-4 times daily in small sips",
            "benefits": "Antispasmodic, reduces intestinal cramps, anti-inflammatory",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "fennel",
            "preparation": "Chew fennel seeds or brew fennel tea after each loose motion episode.",
            "dosage": "After each episode, up to 4 times daily",
            "benefits": "Antispasmodic, carminative, reduces intestinal spasms",
            "evidence_level": "Preliminary Studies"
        },
    ],

    # ── Pain & Inflammation ──────────────────────────────────────────────────
    "headache": [
        {
            "herb": "peppermint",
            "preparation": "Apply diluted peppermint oil (2-3 drops in 1 tsp coconut oil) on temples and forehead. Gently massage.",
            "dosage": "Apply every 30-60 min as needed",
            "benefits": "Menthol provides cooling relief, relaxes tense muscles, analgesic",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ginger",
            "preparation": "Drink strong ginger tea (2-inch piece grated in water, boil 10 min). Or take 250 mg ginger supplement.",
            "dosage": "At onset of headache, repeat after 4 hours if needed",
            "benefits": "Anti-inflammatory, comparable to sumatriptan for migraines in some studies",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "clove",
            "preparation": "Crush 2-3 cloves, mix with coconut oil. Apply on forehead and temples.",
            "dosage": "As needed for headache episodes",
            "benefits": "Eugenol provides analgesic and anti-inflammatory effects",
            "evidence_level": "Traditional"
        },
    ],

    "migraine": [
        {
            "herb": "ginger",
            "preparation": "At first sign of migraine, drink ginger tea or take 250 mg ginger powder capsule.",
            "dosage": "250 mg at onset, can repeat after 4 hours",
            "benefits": "Studies show comparable efficacy to sumatriptan for acute migraine",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "peppermint",
            "preparation": "Apply diluted peppermint oil (10% solution) on temples. Also inhale peppermint aroma.",
            "dosage": "Apply at onset, reapply every 30 min",
            "benefits": "Cools, reduces pain intensity, relaxes blood vessels",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "brahmi",
            "preparation": "Take 300-450 mg Bacopa monnieri extract daily as preventive (not acute treatment).",
            "dosage": "300 mg daily for at least 8-12 weeks for preventive benefit",
            "benefits": "Neuroprotective, reduces migraine frequency over time",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "joint pain": [
        {
            "herb": "turmeric",
            "preparation": "Take 500 mg curcumin supplement with black pepper, OR make golden paste (turmeric + black pepper + coconut oil).",
            "dosage": "500-1000 mg curcumin daily with meals",
            "benefits": "Powerful anti-inflammatory — comparable to ibuprofen in some joint pain studies",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ginger",
            "preparation": "Apply warm ginger paste (grated ginger + sesame oil) on affected joints. Also take ginger tea internally.",
            "dosage": "Apply paste 2x daily; drink ginger tea 2-3 cups daily",
            "benefits": "Reduces joint inflammation, improves mobility, analgesic",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ashwagandha",
            "preparation": "Take 300-500 mg ashwagandha root extract with warm milk at bedtime.",
            "dosage": "300-500 mg twice daily for 8-12 weeks",
            "benefits": "Anti-inflammatory, strengthens muscles, reduces pain perception",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "osteoarthristis": [
        {
            "herb": "turmeric",
            "preparation": "Take curcumin extract (500 mg) with piperine, twice daily with food.",
            "dosage": "1000 mg curcumin daily (split into 2 doses)",
            "benefits": "Reduces knee osteoarthritis pain and stiffness in clinical trials",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ginger",
            "preparation": "Take 500 mg ginger extract daily, or apply ginger compress on affected joints.",
            "dosage": "500-1000 mg daily",
            "benefits": "Reduces pain, swelling, and morning stiffness",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ashwagandha",
            "preparation": "Take 500 mg Ashwagandha root extract with warm water after meals.",
            "dosage": "500 mg twice daily for 12 weeks",
            "benefits": "Reduces inflammation markers, improves joint mobility",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "arthritis": [
        {
            "herb": "turmeric",
            "preparation": "Golden paste: mix 1 tsp turmeric + ¼ tsp black pepper + 1 tsp coconut oil. Take with warm water.",
            "dosage": "½ tsp golden paste twice daily, or 1000 mg curcumin supplement",
            "benefits": "Anti-inflammatory rivalling NSAIDs in some trials for rheumatoid arthritis",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "ginger",
            "preparation": "Warm ginger compress: soak cloth in hot ginger water, apply to joints for 20 min.",
            "dosage": "Apply twice daily; also drink ginger tea",
            "benefits": "Reduces swelling, pain, and inflammation",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "giloy",
            "preparation": "Boil 1-inch giloy stem in 2 cups water until reduced by half. Strain and drink.",
            "dosage": "1 cup twice daily for 4-6 weeks",
            "benefits": "Immunomodulatory, reduces autoimmune inflammation",
            "evidence_level": "Traditional"
        },
    ],

    # ── Skin ─────────────────────────────────────────────────────────────────
    "acne": [
        {
            "herb": "neem",
            "preparation": "Make paste of fresh neem leaves (grind with water). Apply on affected areas for 15-20 min. Rinse off.",
            "dosage": "Apply 2-3 times per week; leave for 15-20 min",
            "benefits": "Antibacterial, antifungal — targets acne-causing P. acnes bacteria",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "aloe_vera",
            "preparation": "Apply fresh aloe vera gel directly on acne spots. Leave on overnight.",
            "dosage": "Apply nightly on clean skin",
            "benefits": "Soothes inflammation, accelerates healing, reduces scarring",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Mix ½ tsp turmeric with 1 tbsp honey and 1 tsp aloe gel. Apply as face mask for 15 min.",
            "dosage": "2-3 times per week",
            "benefits": "Antimicrobial, reduces redness, fades acne scars",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "skin rash": [
        {
            "herb": "aloe_vera",
            "preparation": "Apply fresh aloe vera gel generously on the rash area. Let it absorb naturally.",
            "dosage": "3-4 times daily on affected area",
            "benefits": "Cooling, anti-inflammatory, soothes itching, promotes healing",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "neem",
            "preparation": "Boil neem leaves in bath water and use for bathing. Or apply diluted neem oil on rash.",
            "dosage": "Daily neem bath; or apply neem oil 2x daily",
            "benefits": "Antibacterial, antifungal, antipruritic (anti-itch)",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Mix turmeric with coconut oil to make a paste. Apply on rash for 20 min.",
            "dosage": "Twice daily until rash subsides",
            "benefits": "Anti-inflammatory, antimicrobial, promotes skin healing",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "fungal infection": [
        {
            "herb": "neem",
            "preparation": "Apply neem oil (diluted with coconut oil 1:1) directly on the infected area. Also bathe with neem-infused water.",
            "dosage": "Apply 2-3 times daily; neem bath once daily",
            "benefits": "Potent antifungal (azadirachtin disrupts fungal cell membranes)",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "garlic",
            "preparation": "Crush 2-3 garlic cloves, mix with coconut oil. Apply on the affected area for 20 min.",
            "dosage": "Apply twice daily for 2-4 weeks",
            "benefits": "Allicin is a powerful antifungal — effective against Candida and dermatophytes",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Make a paste of turmeric powder and water. Apply on the infected area and leave for 30 min.",
            "dosage": "2-3 times daily",
            "benefits": "Antifungal, anti-inflammatory, prevents spread",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "aloe_vera",
            "preparation": "Apply fresh aloe vera gel on the area after cleaning. Let it dry naturally.",
            "dosage": "3-4 times daily",
            "benefits": "Antifungal, soothes itching, promotes skin repair",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "psoriasis": [
        {
            "herb": "aloe_vera",
            "preparation": "Apply 0.5% aloe vera cream or pure gel on psoriasis plaques. Consistent daily use is key.",
            "dosage": "Apply 3 times daily for at least 4-8 weeks",
            "benefits": "Reduces scaling, redness, and itching. FDA-approved for psoriasis studies.",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Take 500 mg curcumin supplement orally AND apply turmeric paste topically on plaques.",
            "dosage": "500 mg oral curcumin daily + topical application",
            "benefits": "Modulates immune response, reduces skin inflammation",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "neem",
            "preparation": "Bathe with neem-leaf-infused water. Apply neem oil mixed with coconut oil on plaques.",
            "dosage": "Daily neem bath; oil application twice daily",
            "benefits": "Anti-inflammatory, reduces itching, antimicrobial",
            "evidence_level": "Traditional"
        },
    ],

    # ── Fever & Immunity ─────────────────────────────────────────────────────
    "fever": [
        {
            "herb": "tulsi",
            "preparation": "Boil 10-12 Tulsi leaves with 1 tsp grated ginger in 2 cups water. Add honey when warm.",
            "dosage": "Every 4-6 hours during fever",
            "benefits": "Antipyretic, immunostimulant, traditional fever remedy in India",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "giloy",
            "preparation": "Boil 1-inch giloy stem in 2 cups water until reduced to 1 cup. Add honey and drink.",
            "dosage": "Twice daily during fever",
            "benefits": "Potent antipyretic, increases platelet count, immunomodulatory",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "cinnamon",
            "preparation": "Boil 1 cinnamon stick in water for 10 min. Add honey and lemon.",
            "dosage": "2-3 cups daily during fever",
            "benefits": "Warming, antimicrobial, induces mild sweating to break fever",
            "evidence_level": "Traditional"
        },
    ],

    "allergy": [
        {
            "herb": "tulsi",
            "preparation": "Chew 5-6 fresh tulsi leaves daily on empty stomach. Or brew tulsi tea.",
            "dosage": "Daily — morning, on empty stomach",
            "benefits": "Anti-allergic, stabilises mast cells, immunomodulatory",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "turmeric",
            "preparation": "Add 1 tsp turmeric to warm milk with black pepper. Drink daily.",
            "dosage": "1 cup golden milk daily",
            "benefits": "Curcumin inhibits histamine release, reduces allergic inflammation",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "honey",
            "preparation": "Take 1 tbsp local raw honey daily. Theory: gradual pollen desensitisation (for pollen allergies).",
            "dosage": "1 tbsp daily — start weeks before allergy season",
            "benefits": "May reduce seasonal allergy symptoms with regular local honey intake",
            "evidence_level": "Traditional"
        },
        {
            "herb": "giloy",
            "preparation": "Take 500 mg Giloy extract twice daily or drink giloy juice.",
            "dosage": "500 mg twice daily for 6-8 weeks",
            "benefits": "Anti-allergic, reduces sneezing and nasal congestion",
            "evidence_level": "Preliminary Studies"
        },
    ],

    # ── Mental Health & Sleep ────────────────────────────────────────────────
    "anxiety": [
        {
            "herb": "ashwagandha",
            "preparation": "Take 300 mg ashwagandha root extract (KSM-66 or Sensoril) with warm milk at bedtime.",
            "dosage": "300-600 mg daily for at least 8 weeks",
            "benefits": "Clinically proven to reduce cortisol by 30% and anxiety scores in RCTs",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "chamomile",
            "preparation": "Steep chamomile tea bag or 2 tbsp dried flowers in hot water for 5 min. Drink warm.",
            "dosage": "2-3 cups daily, especially before bed",
            "benefits": "Apigenin binds GABA receptors — mild anxiolytic effect",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "brahmi",
            "preparation": "Take 300 mg Bacopa extract daily with food.",
            "dosage": "300 mg daily for 12 weeks for full effect",
            "benefits": "Reduces anxiety, improves cognition, neuroprotective",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "tulsi",
            "preparation": "Drink 2-3 cups tulsi tea throughout the day. Or take tulsi extract capsules.",
            "dosage": "2-3 cups tea daily or 300 mg extract",
            "benefits": "Adaptogenic, reduces stress hormones, promotes mental clarity",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "insomnia": [
        {
            "herb": "ashwagandha",
            "preparation": "Mix 1 tsp ashwagandha powder in warm milk with a pinch of nutmeg. Drink 30 min before bed.",
            "dosage": "300-600 mg daily at bedtime",
            "benefits": "Improves sleep quality, reduces sleep onset latency, triethylene glycol promotes sleep",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "chamomile",
            "preparation": "Strong chamomile tea — steep for 10 min. Drink 30-45 min before bedtime.",
            "dosage": "1 strong cup nightly",
            "benefits": "Mild sedative (apigenin), promotes relaxation, reduces night-time anxiety",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "brahmi",
            "preparation": "Take 300 mg Brahmi extract with warm milk at night.",
            "dosage": "300 mg at bedtime",
            "benefits": "Calms the mind, reduces racing thoughts, improves sleep onset",
            "evidence_level": "Preliminary Studies"
        },
    ],

    # ── Metabolic ────────────────────────────────────────────────────────────
    "diabetes": [
        {
            "herb": "fenugreek",
            "preparation": "Soak 1 tbsp fenugreek (methi) seeds in water overnight. Drink the water and eat the seeds on empty stomach.",
            "dosage": "1 tbsp soaked seeds daily on empty stomach",
            "benefits": "Clinically shown to lower fasting blood sugar by 13-25% in type 2 diabetes studies",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "cinnamon",
            "preparation": "Add ½ tsp Ceylon cinnamon powder to warm water or tea. Drink before meals.",
            "dosage": "1-2 g (½ tsp) daily, use Ceylon cinnamon (not Cassia)",
            "benefits": "Improves insulin sensitivity, lowers fasting blood glucose",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "amla",
            "preparation": "Eat 1-2 fresh amla fruits daily, or take 500 mg amla extract.",
            "dosage": "1-2 fruits daily or 500 mg extract",
            "benefits": "Lowers blood sugar, improves lipid profile, antioxidant protection for pancreas",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "giloy",
            "preparation": "Take giloy juice (20 ml) mixed with water on empty stomach.",
            "dosage": "20 ml juice daily",
            "benefits": "Hypoglycaemic action, enhances insulin secretion",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "hypoglycemia": [
        {
            "herb": "amla",
            "preparation": "Eat 1 fresh amla or take amla juice with honey.",
            "dosage": "Once daily",
            "benefits": "Stabilises blood sugar fluctuations, rich in vitamin C",
            "evidence_level": "Traditional"
        },
        {
            "herb": "ashwagandha",
            "preparation": "Take 300 mg ashwagandha extract with meals to stabilise energy levels.",
            "dosage": "300 mg with meals",
            "benefits": "Adaptogenic — helps stabilise metabolic fluctuations",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "hypertension": [
        {
            "herb": "garlic",
            "preparation": "Eat 1-2 raw garlic cloves on empty stomach in the morning. Or take aged garlic extract.",
            "dosage": "2 cloves daily or 600-1200 mg aged garlic extract",
            "benefits": "Meta-analyses show garlic lowers systolic BP by 8-10 mmHg",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "amla",
            "preparation": "Take 2 tbsp amla juice with water every morning.",
            "dosage": "2 tbsp daily on empty stomach",
            "benefits": "Vasodilatory, reduces cholesterol, antioxidant",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "cardamom",
            "preparation": "Crush 2-3 cardamom pods and boil in water. Drink as tea.",
            "dosage": "2-3 cups daily for 12 weeks",
            "benefits": "Studies show modest BP reduction with regular cardamom use",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "hypothyroidism": [
        {
            "herb": "ashwagandha",
            "preparation": "Take 600 mg ashwagandha root extract daily. Studies show it raises T4 levels.",
            "dosage": "600 mg daily for 8 weeks (monitor thyroid levels)",
            "benefits": "May stimulate thyroid hormone production (T3 and T4)",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "amla",
            "preparation": "Eat 2 fresh amla daily or take amla juice.",
            "dosage": "2 fruits or 30 ml juice daily",
            "benefits": "Supports thyroid function, rich in antioxidants, boosts metabolism",
            "evidence_level": "Traditional"
        },
    ],

    "hyperthyroidism": [
        {
            "herb": "brahmi",
            "preparation": "Take 300-500 mg Brahmi extract daily.",
            "dosage": "300-500 mg daily",
            "benefits": "May help regulate thyroid function, calming effect",
            "evidence_level": "Traditional"
        },
        {
            "herb": "tulsi",
            "preparation": "Drink tulsi tea 2-3 times daily.",
            "dosage": "2-3 cups daily",
            "benefits": "Adaptogenic, may help normalise thyroid function",
            "evidence_level": "Traditional"
        },
    ],

    # ── Urinary ──────────────────────────────────────────────────────────────
    "urinary tract infection": [
        {
            "herb": "amla",
            "preparation": "Drink 30 ml amla juice with water twice daily.",
            "dosage": "30 ml twice daily",
            "benefits": "Vitamin C acidifies urine (inhibits bacterial growth), immune boost",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "tulsi",
            "preparation": "Boil tulsi leaves in water. Drink as tea 3-4 times daily.",
            "dosage": "3-4 cups tulsi tea daily",
            "benefits": "Antimicrobial, reduces bladder inflammation",
            "evidence_level": "Traditional"
        },
        {
            "herb": "fennel",
            "preparation": "Drink fennel water (boiled fennel seeds strained) throughout the day.",
            "dosage": "4-5 glasses daily",
            "benefits": "Mild diuretic, flushes bacteria, anti-inflammatory",
            "evidence_level": "Traditional"
        },
    ],

    # ── Cardiovascular ───────────────────────────────────────────────────────
    "varicose veins": [
        {
            "herb": "turmeric",
            "preparation": "Take 500 mg curcumin daily with food. Also apply turmeric paste on veins.",
            "dosage": "500 mg curcumin daily",
            "benefits": "Anti-inflammatory, improves circulation, reduces swelling",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "ginger",
            "preparation": "Drink ginger tea 2-3 times daily. Ginger improves blood circulation.",
            "dosage": "2-3 cups daily",
            "benefits": "Improves blood flow, anti-inflammatory, reduces blood viscosity",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "garlic",
            "preparation": "Eat 2 raw garlic cloves daily. Garlic improves circulation and reduces inflammation.",
            "dosage": "2 cloves daily on empty stomach",
            "benefits": "Improves venous health, reduces inflammation in blood vessels",
            "evidence_level": "Preliminary Studies"
        },
    ],

    # ── Oral Health ──────────────────────────────────────────────────────────
    "toothache": [
        {
            "herb": "clove",
            "preparation": "Place a whole clove on the painful tooth and bite gently. Or apply clove oil (diluted) with cotton.",
            "dosage": "As needed — max 3-4 times daily",
            "benefits": "Eugenol is a proven dental analgesic — used by dentists worldwide",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Make a paste of turmeric + salt + mustard oil. Apply on affected gum area.",
            "dosage": "2-3 times daily",
            "benefits": "Antimicrobial, anti-inflammatory, reduces gum swelling",
            "evidence_level": "Traditional"
        },
        {
            "herb": "peppermint",
            "preparation": "Apply peppermint tea bag (used, warm) on the affected area for 20 min.",
            "dosage": "As needed",
            "benefits": "Numbing effect, cooling relief, mild antimicrobial",
            "evidence_level": "Traditional"
        },
    ],

    # ── Stress & Fatigue ─────────────────────────────────────────────────────
    "fatigue": [
        {
            "herb": "ashwagandha",
            "preparation": "Take 300-500 mg ashwagandha extract with warm milk in the morning.",
            "dosage": "300-500 mg daily for 8-12 weeks",
            "benefits": "Reduces cortisol, increases energy, improves physical performance",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "amla",
            "preparation": "Eat 1-2 amla fruits daily or take 500 mg extract.",
            "dosage": "Daily",
            "benefits": "Rich in vitamin C and iron — addresses nutritional deficiency fatigue",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "tulsi",
            "preparation": "Drink tulsi tea 2-3 times daily.",
            "dosage": "2-3 cups daily",
            "benefits": "Adaptogenic — helps body cope with physical and mental stress",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "shatavari",
            "preparation": "Take 500 mg Shatavari root extract with warm milk.",
            "dosage": "500 mg daily",
            "benefits": "Rejuvenating tonic, energy boost, endurance enhancement",
            "evidence_level": "Traditional"
        },
    ],

    # ── More Conditions ──────────────────────────────────────────────────────
    "sore throat": [
        {
            "herb": "mulethi",
            "preparation": "Chew a piece of mulethi root slowly. Or gargle with mulethi-infused warm water.",
            "dosage": "3-4 times daily",
            "benefits": "Demulcent — coats and soothes inflamed throat tissue",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "honey",
            "preparation": "Mix 1 tbsp honey with warm water and a squeeze of lemon. Sip slowly.",
            "dosage": "3-4 times daily",
            "benefits": "Soothes throat, antimicrobial, reduces irritation",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "turmeric",
            "preparation": "Gargle with warm water + ½ tsp turmeric + pinch of salt. Also drink turmeric milk.",
            "dosage": "Gargle 3-4 times daily",
            "benefits": "Anti-inflammatory, antimicrobial, reduces swelling",
            "evidence_level": "Preliminary Studies"
        },
    ],

    "nausea": [
        {
            "herb": "ginger",
            "preparation": "Chew thin slices of fresh ginger, or sip ginger tea slowly.",
            "dosage": "Small sips or slices as needed, up to 1 g ginger daily",
            "benefits": "Gold-standard natural antiemetic — proven for pregnancy, chemo, and motion sickness",
            "evidence_level": "Well-Studied"
        },
        {
            "herb": "peppermint",
            "preparation": "Inhale peppermint essential oil or sip peppermint tea.",
            "dosage": "As needed",
            "benefits": "Reduces nausea through olfactory pathway, antispasmodic",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "cardamom",
            "preparation": "Chew 1-2 cardamom pods slowly. The aroma and flavour reduce nausea.",
            "dosage": "As needed, 3-4 pods daily",
            "benefits": "Carminative, antiemetic, freshens breath",
            "evidence_level": "Traditional"
        },
    ],

    "dimorphic hemmorhoids(piles)": [
        {
            "herb": "triphala",
            "preparation": "Take 1 tsp Triphala powder in warm water at bedtime to ensure soft stools.",
            "dosage": "1 tsp at bedtime",
            "benefits": "Prevents straining, gentle laxative, anti-inflammatory",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "aloe_vera",
            "preparation": "Apply pure aloe vera gel topically on external haemorrhoids for soothing relief.",
            "dosage": "Apply 3-4 times daily",
            "benefits": "Cooling, anti-inflammatory, reduces itching and burning",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "turmeric",
            "preparation": "Take 500 mg curcumin supplement daily. Also apply turmeric + coconut oil paste locally.",
            "dosage": "500 mg oral + topical application",
            "benefits": "Reduces inflammation, promotes healing of haemorrhoidal tissue",
            "evidence_level": "Traditional"
        },
    ],

    "(vertigo) paroymsal positional vertigo": [
        {
            "herb": "ginger",
            "preparation": "Chew fresh ginger root or drink ginger tea during vertigo episodes.",
            "dosage": "At onset, repeat every 4 hours",
            "benefits": "Reduces dizziness and nausea associated with vertigo",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "brahmi",
            "preparation": "Take 300 mg Brahmi extract daily.",
            "dosage": "300 mg daily for 4-8 weeks",
            "benefits": "Neuroprotective, may improve vestibular function",
            "evidence_level": "Traditional"
        },
    ],

    "cervical spondylosis": [
        {
            "herb": "turmeric",
            "preparation": "Take 500 mg curcumin with black pepper daily. Apply turmeric paste on neck area.",
            "dosage": "500-1000 mg curcumin daily",
            "benefits": "Anti-inflammatory, reduces nerve compression pain",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "ashwagandha",
            "preparation": "Take 500 mg ashwagandha extract twice daily.",
            "dosage": "500 mg twice daily for 12 weeks",
            "benefits": "Anti-inflammatory, muscle relaxant, reduces pain",
            "evidence_level": "Preliminary Studies"
        },
        {
            "herb": "ginger",
            "preparation": "Apply warm ginger oil massage on neck and shoulders.",
            "dosage": "Massage twice daily",
            "benefits": "Warming, anti-inflammatory, improves blood circulation to area",
            "evidence_level": "Traditional"
        },
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_herbal_remedies(disease: str, severity: str = "mild") -> dict[str, Any]:
    """
    Get herbal remedies for a disease.

    Returns a dict with:
      - remedies: list of remedy dicts (or empty if severe)
      - severity_warning: str | None
      - doctor_referral: bool
      - disclaimer: str
    """
    norm = _normalise(disease)
    result: dict[str, Any] = {
        "disease": disease,
        "remedies": [],
        "severity_warning": None,
        "doctor_referral": False,
        "disclaimer": DISCLAIMER,
    }

    # Severity guard — block herbal advice for serious conditions
    if norm in SEVERE_CONDITIONS or severity.lower() == "severe":
        result["severity_warning"] = (
            f"'{disease}' is a serious medical condition that requires professional "
            "medical treatment. Herbal remedies are NOT appropriate as primary "
            "treatment. Please consult a doctor immediately."
        )
        result["doctor_referral"] = True
        return result

    if norm in MODERATE_CONDITIONS or severity.lower() == "moderate":
        result["severity_warning"] = (
            f"'{disease}' may require medical attention. The following herbal "
            "remedies may be used as COMPLEMENTARY support alongside professional "
            "treatment — NOT as a replacement."
        )
        result["doctor_referral"] = True
        # Still return remedies for moderate, but with warning

    # Look up remedies
    raw_remedies = HERBAL_REMEDIES.get(norm, [])
    enriched = []
    for r in raw_remedies:
        herb_key = r["herb"]
        herb_info = HERB_DATABASE.get(herb_key, {})
        enriched.append({
            "herb_name": herb_info.get("common_name", herb_key.replace("_", " ").title()),
            "scientific_name": herb_info.get("scientific_name", ""),
            "emoji": herb_info.get("emoji", "🌿"),
            "parts_used": herb_info.get("parts_used", ""),
            "preparation": r["preparation"],
            "dosage": r["dosage"],
            "benefits": r["benefits"],
            "precautions": herb_info.get("contraindications", ""),
            "evidence_level": r.get("evidence_level", herb_info.get("evidence_level", "Traditional")),
        })

    result["remedies"] = enriched
    return result


def search_herbs(query: str) -> list[dict[str, Any]]:
    """
    Search the herb database by name, property, or use.
    Returns a list of matching herb profiles.
    """
    q = _normalise(query)
    results = []

    for key, herb in HERB_DATABASE.items():
        # Search in common name, scientific name, uses, properties
        searchable = " ".join([
            herb.get("common_name", ""),
            herb.get("scientific_name", ""),
            herb.get("traditional_uses", ""),
            " ".join(herb.get("therapeutic_properties", [])),
            herb.get("description", ""),
        ]).lower()

        if q in searchable:
            results.append({
                "key": key,
                **herb,
            })

    return results


def get_herb_profile(herb_name: str) -> Optional[dict[str, Any]]:
    """Get detailed profile for a specific herb."""
    norm = _normalise(herb_name).replace(" ", "_")

    # Try exact key match first
    if norm in HERB_DATABASE:
        return {"key": norm, **HERB_DATABASE[norm]}

    # Fuzzy search by common name
    for key, herb in HERB_DATABASE.items():
        if norm in herb.get("common_name", "").lower() or norm in key:
            return {"key": key, **herb}

    return None


def get_all_herbs() -> list[dict[str, Any]]:
    """Return all herbs in the database with their profiles."""
    return [
        {"key": key, **herb}
        for key, herb in HERB_DATABASE.items()
    ]


def search_diseases(query: str) -> list[str]:
    """Search diseases that have herbal remedies available."""
    q = _normalise(query)
    return [
        disease for disease in HERBAL_REMEDIES
        if q in disease
    ]


def get_all_diseases() -> list[str]:
    """Return all diseases that have herbal remedies."""
    return sorted(HERBAL_REMEDIES.keys())
