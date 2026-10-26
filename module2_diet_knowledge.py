"""
diet_knowledge.py
=================
Disease-specific Diet Knowledge Base for VitalSense AI — Module 2.

Provides structured "What to Eat" and "Foods to Avoid" lists for every
condition in the dataset, replacing the generic flat-list from diets.csv.

Structure
---------
DIET_DB : dict[str, dict]
    Keys: disease name (title-cased, matching dataset_manager conventions)
    Values:
        {
          "eat":   list[str],   # foods & dietary habits to follow
          "avoid": list[str],   # foods & habits to strictly avoid
        }
"""

from __future__ import annotations

DIET_DB: dict[str, dict] = {

    "Fungal Infection": {
        "eat": [
            "Probiotic-rich foods (plain yogurt, kefir, buttermilk)",
            "Fresh garlic and onion (natural antifungal properties)",
            "Coconut oil (contains lauric acid — antifungal)",
            "Turmeric milk (curcumin has antifungal activity)",
            "Leafy green vegetables (spinach, kale, broccoli)",
            "Lean proteins (eggs, chicken, legumes)",
            "Plenty of water (2–3 litres/day to flush toxins)",
            "Ginger tea (anti-inflammatory support)",
        ],
        "avoid": [
            "Sugar and sugary foods (feeds fungal growth)",
            "White bread, white rice, and refined flour products",
            "Alcohol (suppresses immune response)",
            "Processed and packaged snacks",
            "Fermented foods that are not probiotic (vinegar, soy sauce)",
            "Fruit juices and soft drinks (high sugar content)",
            "Mushrooms (may cross-react with fungal antigens)",
        ],
    },

    "Allergy": {
        "eat": [
            "Quercetin-rich foods (apples, onions, berries) — natural antihistamine",
            "Omega-3 fatty acids (fatty fish, flaxseeds, walnuts)",
            "Vitamin C-rich foods (citrus, bell peppers, kiwi)",
            "Turmeric and ginger (reduce inflammatory mediators)",
            "Probiotic foods (yogurt, kefir, fermented vegetables)",
            "Local raw honey (may build pollen tolerance gradually)",
            "Warm broths and herbal teas to soothe irritated airways",
        ],
        "avoid": [
            "Identified allergen triggers (pollen-associated foods if relevant)",
            "Highly processed packaged foods with artificial additives",
            "Alcohol (triggers histamine release and worsens symptoms)",
            "Fermented foods (cheese, wine, vinegar — high in histamine)",
            "Cold dairy products during active nasal allergy flare-ups",
            "Spicy foods that can worsen nasal congestion",
            "Raw shellfish and peanuts (common cross-reactive allergens)",
        ],
    },

    "Gerd": {
        "eat": [
            "Oatmeal and whole-grain cereals (absorbs stomach acid)",
            "Non-citrus fruits (bananas, melons, apples, pears)",
            "Lean proteins (grilled chicken, turkey, fish)",
            "Steamed or boiled vegetables (broccoli, peas, green beans)",
            "Low-fat dairy (if tolerated — milk, yogurt)",
            "Ginger tea (natural anti-inflammatory for gut lining)",
            "Alkaline water and herbal teas (chamomile, licorice root)",
            "Small, frequent meals every 3–4 hours",
        ],
        "avoid": [
            "Citrus fruits and juices (lemon, orange, grapefruit)",
            "Tomatoes and tomato-based sauces (pizza, pasta sauce)",
            "Chocolate and mint (relax the esophageal sphincter)",
            "Fried, greasy, and fatty foods",
            "Coffee, tea, and carbonated beverages",
            "Spicy foods and raw onions or garlic",
            "Alcohol (relaxes lower esophageal sphincter)",
            "Large meals before bedtime (stop eating 3 hours before sleep)",
        ],
    },

    "Chronic Cholestasis": {
        "eat": [
            "Low-fat, easily digestible foods (boiled vegetables, rice, dal)",
            "High-fiber foods (oats, legumes, fruits) to support bile flow",
            "Lean proteins (egg whites, fish, skinless chicken)",
            "Fat-soluble vitamin supplements (A, D, E, K — as prescribed)",
            "Plenty of water (2–2.5 litres/day) for renal bile acid clearance",
            "Cooked and peeled fruits (bananas, papayas, apples)",
        ],
        "avoid": [
            "All alcohol (even trace amounts)",
            "High-fat foods (fried foods, butter, full-fat dairy, red meat)",
            "Processed meats and packaged salty snacks",
            "Unverified herbal supplements (can worsen cholestasis)",
            "Large fatty meals that trigger biliary colic",
            "Whole milk and high-fat dairy products",
            "Raw or undercooked foods (infection risk with compromised liver)",
        ],
    },

    "Drug Reaction": {
        "eat": [
            "Plenty of water and oral rehydration fluids (flush drug metabolites)",
            "Fresh fruits rich in Vitamin C (kiwi, guava, strawberries)",
            "Anti-inflammatory foods (turmeric, ginger, green tea)",
            "Bland, easily digestible foods (khichdi, oatmeal, toast)",
            "Probiotic yogurt (restore gut microbiome if antibiotics involved)",
            "Coconut water for electrolyte replacement",
        ],
        "avoid": [
            "The offending drug and chemically related compounds",
            "Alcohol (worsens drug hypersensitivity reactions)",
            "Highly spicy or acidic foods (irritate compromised gut/skin)",
            "Shellfish and other known allergens",
            "Caffeine and energy drinks during active reaction",
            "Any new, unverified supplements or herbal remedies",
        ],
    },

    "Peptic Ulcer Disease": {
        "eat": [
            "Probiotic-rich foods (yogurt, kefir) — help suppress H. pylori",
            "High-fiber fruits and vegetables (carrots, broccoli, apples)",
            "Cabbage juice (contains glutamine — promotes mucosal healing)",
            "Turmeric (curcumin reduces H. pylori activity)",
            "Honey (antibacterial and mucosal-protective properties)",
            "Lean proteins (egg whites, chicken, fish)",
            "Oatmeal and whole grains (soothing to the stomach lining)",
            "Small frequent meals every 3–4 hours",
        ],
        "avoid": [
            "Coffee, tea, and caffeinated beverages (stimulate acid secretion)",
            "Alcohol (directly damages gastric mucosa)",
            "Spicy foods, chili, and hot sauces",
            "NSAIDs/Aspirin (Ibuprofen, Diclofenac, Naproxen — strictly avoid)",
            "Citrus fruits and acidic juices",
            "Carbonated soft drinks",
            "Fried and high-fat foods",
            "Skipping meals (empty stomach worsens ulcer pain)",
        ],
    },

    "Aids": {
        "eat": [
            "High-protein foods (eggs, legumes, lean meat, fish, tofu)",
            "Calorie-dense nutrient-rich foods to prevent wasting",
            "Thoroughly cooked vegetables and grains",
            "Omega-3 rich foods (salmon, flaxseeds, chia seeds)",
            "Fortified foods rich in Vitamins B12, D, and zinc",
            "Boiled or bottled safe drinking water",
            "Pasteurized dairy products only",
            "Fresh fruits (washed and peeled) for antioxidants",
        ],
        "avoid": [
            "Raw or undercooked meat, poultry, eggs, and seafood (infection risk)",
            "Unpasteurized milk, cheese, and juices",
            "Alcohol (suppresses immune function and interacts with ART drugs)",
            "Street food and foods from unknown hygienic sources",
            "Raw sprouts, unwashed salads, and unpasteurized fermented foods",
            "Grapefruit juice (interferes with many ART medications)",
            "High-sodium processed foods that stress kidneys",
        ],
    },

    "Diabetes": {
        "eat": [
            "Low-glycemic index foods (oats, barley, legumes, lentils)",
            "Non-starchy vegetables (spinach, cucumber, broccoli, cauliflower)",
            "Whole fruits in moderate portions (guava, apple, pear, berries)",
            "Lean proteins (chicken breast, fish, tofu, eggs)",
            "Healthy fats (avocado, nuts, olive oil)",
            "High-fiber foods to slow glucose absorption",
            "Cinnamon and fenugreek (help with insulin sensitivity)",
            "Green tea (antioxidant support)",
        ],
        "avoid": [
            "Sugary foods and drinks (sodas, fruit juices, sweets, cakes)",
            "White bread, white rice, and maida-based foods",
            "Fried and deep-fried snacks (samosa, chips, pakoda)",
            "Full-fat dairy and high-fat red meats",
            "Alcohol (causes dangerous blood sugar fluctuations)",
            "High-sodium processed foods (increases cardiovascular risk)",
            "Honey, jaggery, and artificial sweeteners in excess",
            "Skipping meals (leads to hypoglycemia)",
        ],
    },

    "Gastroenteritis": {
        "eat": [
            "Oral rehydration salts (ORS) — sip frequently after every stool",
            "Bananas (BRAT diet — provides potassium, easy to digest)",
            "Plain boiled white rice (starchy, soothing)",
            "Toast and plain crackers (bland, easy on the stomach)",
            "Applesauce or stewed apples",
            "Clear broths and soups",
            "Plain boiled potatoes",
            "Gradually reintroduce soft cooked vegetables as tolerated",
        ],
        "avoid": [
            "Dairy products (lactose intolerance worsens during gastroenteritis)",
            "Fried and fatty foods",
            "Spicy foods and sauces",
            "Caffeine (coffee, tea, energy drinks)",
            "Alcohol",
            "High-fiber raw vegetables and fruits (too hard to digest)",
            "Artificial sweeteners (sorbitol — worsens diarrhea)",
            "Fruit juices (high sugar content aggravates diarrhea)",
        ],
    },

    "Bronchial Asthma": {
        "eat": [
            "Anti-inflammatory foods (turmeric, ginger, garlic)",
            "Omega-3 rich foods (fatty fish, walnuts, flaxseeds)",
            "Vitamin D-rich foods (fortified milk, eggs, mushrooms)",
            "Magnesium-rich foods (dark leafy greens, seeds, legumes)",
            "Fresh fruits and vegetables (antioxidant-rich)",
            "Ginger tea and warm herbal teas",
            "Apples (quercetin may reduce asthma risk)",
        ],
        "avoid": [
            "Sulfite-containing foods (wine, dried fruits, pickles, shrimp)",
            "Known food allergens (shellfish, peanuts, tree nuts, dairy if sensitive)",
            "Cold foods and iced beverages (can trigger bronchospasm)",
            "Alcohol (releases histamine and may trigger attacks)",
            "Highly processed foods and artificial food colorings",
            "Aspirin/NSAIDs if sensitive (aspirin-exacerbated respiratory disease)",
            "Gas-producing foods that worsen breathing (beans, cabbage in large amounts)",
        ],
    },

    "Hypertension": {
        "eat": [
            "DASH Diet: Fruits, vegetables, whole grains, low-fat dairy",
            "Potassium-rich foods (bananas, sweet potatoes, avocado, spinach)",
            "Magnesium-rich foods (nuts, seeds, legumes, leafy greens)",
            "Omega-3 rich fish (salmon, mackerel, sardines)",
            "Garlic (natural vasodilator)",
            "Beetroot juice (dietary nitrates lower blood pressure)",
            "Unsalted nuts and seeds in moderation",
        ],
        "avoid": [
            "High-sodium foods (pickles, canned soups, processed meats, salty snacks)",
            "Added table salt — limit to less than 1,500 mg/day",
            "Alcohol (raises blood pressure acutely and chronically)",
            "Caffeine in excess (coffee, energy drinks)",
            "Saturated fats (red meat, butter, full-fat cheese)",
            "Sugary beverages and refined carbohydrates",
            "Licorice (raises blood pressure in large amounts)",
        ],
    },

    "Migraine": {
        "eat": [
            "Magnesium-rich foods (almonds, spinach, seeds, dark chocolate)",
            "Riboflavin (Vitamin B2) sources (eggs, dairy, leafy greens)",
            "Omega-3 fatty acids (fish, walnuts, chia seeds)",
            "Adequate hydration — at least 2.5 litres of water daily",
            "Small regular meals to maintain stable blood sugar",
            "Ginger tea (anti-nausea and anti-inflammatory)",
            "Coenzyme Q10-rich foods (fatty fish, organ meats, nuts)",
        ],
        "avoid": [
            "Tyramine-rich aged foods (aged cheese, red wine, cured meats, soy sauce)",
            "Alcohol, especially red wine and beer",
            "Caffeine (or sudden caffeine withdrawal after regular use)",
            "Aspartame and artificial sweeteners",
            "MSG (monosodium glutamate) in processed and Chinese food",
            "Nitrite-containing foods (hot dogs, ham, salami)",
            "Skipping meals or prolonged fasting",
            "Bright-colored artificial food dyes",
        ],
    },

    "Cervical Spondylosis": {
        "eat": [
            "Anti-inflammatory foods (turmeric, ginger, berries, olive oil)",
            "Omega-3 rich foods (fatty fish, flaxseeds, walnuts)",
            "Calcium-rich foods (low-fat dairy, fortified plant milk, sesame seeds)",
            "Vitamin D sources (eggs, fortified cereals, mushrooms, sunlight)",
            "Magnesium-rich foods (dark greens, nuts, legumes)",
            "Collagen-boosting foods (bone broth, Vitamin C-rich fruits)",
            "Green tea (anti-inflammatory polyphenols)",
        ],
        "avoid": [
            "Pro-inflammatory foods (fried foods, trans fats, processed snacks)",
            "Excess red meat and processed meats",
            "Refined sugars and high-glycemic foods",
            "Alcohol (promotes inflammation and impairs bone metabolism)",
            "Excessive caffeine (depletes calcium from bones)",
            "High-sodium processed foods (increase inflammatory markers)",
            "Cold drinks and iced foods during acute pain flare-ups",
        ],
    },

    "Paralysis (Brain Hemorrhage)": {
        "eat": [
            "Brain-healthy Omega-3s (fatty fish, flaxseeds, chia seeds)",
            "Antioxidant-rich foods (blueberries, walnuts, dark leafy greens)",
            "High-fiber foods (whole grains, legumes, fruits)",
            "Lean proteins for muscle recovery (eggs, chicken, fish)",
            "Low-sodium foods to control blood pressure",
            "Fortified foods with Vitamin B6, B12, and folate",
            "Soft, easily swallowed foods if dysphagia is present",
        ],
        "avoid": [
            "High-sodium and salty foods (worsen hypertension — a key stroke risk)",
            "Saturated and trans fats (red meat, butter, fried foods)",
            "Alcohol (increases re-bleeding risk)",
            "Hard-to-swallow foods if dysphagia is present (choking hazard)",
            "Caffeine in excess (raises blood pressure acutely)",
            "Vitamin K-rich foods in large amounts if on anticoagulant therapy",
            "Sugary foods and refined carbohydrates",
        ],
    },

    "Jaundice": {
        "eat": [
            "Easily digestible carbohydrates (rice porridge, boiled potatoes, toast)",
            "Ripe papaya and bananas (gentle on the liver)",
            "Fresh sugarcane juice (from hygienic sources)",
            "Plenty of boiled or purified water (2.5–3 litres/day)",
            "Coconut water (liver-friendly electrolytes)",
            "Amla (Indian gooseberry) — rich in Vitamin C and antioxidants",
            "Buttermilk and diluted yogurt (easily digestible protein)",
        ],
        "avoid": [
            "All alcohol (can precipitate acute liver failure)",
            "High-fat and fried foods (burden on the inflamed liver)",
            "Raw or undercooked foods (salmonella and hepatitis risk)",
            "Excess protein initially (if encephalopathy risk — doctor's guidance)",
            "Street food and foods of unknown hygiene",
            "Paracetamol and over-the-counter pain relievers",
            "Spicy foods and heavy meals",
            "Salty processed foods and pickles",
        ],
    },

    "Malaria": {
        "eat": [
            "Oral rehydration salts (ORS) and electrolyte fluids",
            "Soft, high-calorie foods (khichdi, porridge, mashed potatoes)",
            "Fresh fruit juices (orange, pomegranate — rich in iron and Vitamin C)",
            "Coconut water (natural electrolyte replenishment)",
            "Lean proteins (eggs, boiled chicken, lentils — to rebuild strength)",
            "Ginger tea and warm soups",
            "Iron-rich foods (spinach, lentils, beetroot — to counter anemia)",
        ],
        "avoid": [
            "Aspirin and ibuprofen (risk of bleeding with malarial platelet drop)",
            "Fried and heavy fatty foods (hard to digest during high fever)",
            "Alcohol (immune suppression and interaction with antimalarials)",
            "Raw and contaminated foods and water",
            "Excessive spicy foods (worsen nausea and vomiting)",
            "Cold and iced foods (may worsen chills and rigors)",
        ],
    },

    "Chicken Pox": {
        "eat": [
            "Plenty of warm fluids (water, warm soups, herbal teas)",
            "Cold yogurt and ice pops (soothe oral blisters if present)",
            "Soft fruits (bananas, papayas, melons — easy to swallow)",
            "Vitamin A-rich foods (carrots, sweet potatoes, eggs)",
            "Zinc-rich foods (pumpkin seeds, legumes, tofu)",
            "Protein-rich foods for tissue repair (eggs, milk, fish)",
            "Oatmeal (both as a food and a soothing bath)",
        ],
        "avoid": [
            "Spicy foods (irritate blisters in mouth and throat)",
            "Salty and acidic foods (citrus, vinegar — sting on oral lesions)",
            "Crunchy, hard-textured foods if oral blisters are present",
            "Aspirin (NEVER give to children — risk of Reye's syndrome)",
            "Alcohol (adult patients — impairs immune recovery)",
            "Arginine-rich foods (chocolate, nuts — may feed the virus)",
            "Highly sugary foods (suppress immune function)",
        ],
    },

    "Dengue": {
        "eat": [
            "Papaya leaf extract juice (clinically shown to raise platelets)",
            "Coconut water (natural electrolytes and hydration)",
            "Pomegranate juice (iron and platelet-supportive antioxidants)",
            "Kiwi fruit (Vitamin C and platelet health)",
            "Fresh orange and sweet lime juice (Vitamin C boosts immunity)",
            "Protein-rich soft foods (eggs, dal, soft chicken)",
            "ORS and warm soups (3–4 litres total fluids daily)",
        ],
        "avoid": [
            "Aspirin, Ibuprofen, and NSAIDs (risk of severe internal bleeding)",
            "Spicy, acidic, and oily foods",
            "Dark-colored foods (coffee, cola, chocolate — make blood in stool hard to detect)",
            "Alcohol (suppresses immunity and affects platelets)",
            "Carbonated beverages",
            "High-fiber raw vegetables during the acute phase",
        ],
    },

    "Typhoid": {
        "eat": [
            "High-calorie bland foods (rice porridge, khichdi, boiled pasta)",
            "Boiled and mashed potatoes",
            "Soft ripe bananas and stewed fruits",
            "Egg whites (easily digestible protein)",
            "Steamed fish (light protein)",
            "Boiled or bottled safe water and ORS",
            "Custard, pudding, and plain yogurt",
        ],
        "avoid": [
            "Raw salads, unpeeled fruits, and raw vegetables",
            "High-fiber and roughage foods (raw whole grains, bran)",
            "Spicy curries and hot sauces",
            "Fried and oily foods",
            "Street food and foods of uncertain hygiene",
            "Alcohol",
            "Carbonated and sugary beverages",
            "Hard-to-digest legumes in the acute phase",
        ],
    },

    "Hepatitis A": {
        "eat": [
            "High-carbohydrate easy-to-digest foods (rice, toast, oatmeal)",
            "Boiled or purified water only",
            "Ripe bananas and papayas",
            "Honey (hepatoprotective and easy energy source)",
            "Coconut water for hydration",
            "Boiled vegetables (pumpkin, carrots, zucchini)",
            "Herbal teas (dandelion, milk thistle — support liver)",
        ],
        "avoid": [
            "All alcohol (even during recovery — complete 6-month abstinence)",
            "Fried, fatty, and greasy foods",
            "Raw or undercooked foods and street food",
            "Paracetamol and over-the-counter pain killers",
            "Processed and salty packaged foods",
            "Shellfish and raw seafood",
            "Herbal supplements not cleared by doctor",
        ],
    },

    "Hepatitis B": {
        "eat": [
            "Mediterranean-style diet (olive oil, vegetables, fish, whole grains)",
            "Antioxidant-rich foods (berries, green tea, nuts)",
            "High-fiber foods (oats, beans, leafy greens)",
            "Lean proteins (chicken, fish, tofu, legumes)",
            "Coffee (2 cups/day of filtered coffee — research suggests liver benefit)",
            "Plenty of water (2–3 litres/day)",
        ],
        "avoid": [
            "All alcohol (dramatically accelerates fibrosis and cirrhosis)",
            "High-fat foods (red meat, butter, full-fat dairy)",
            "Aflatoxin-contaminated foods (mouldy grains, nuts — risk of liver cancer)",
            "Excess salt and sodium (ascites risk)",
            "Raw shellfish and seafood",
            "Unverified herbal and traditional liver remedies",
            "Iron supplements unless prescribed (excess iron is hepatotoxic)",
        ],
    },

    "Hepatitis C": {
        "eat": [
            "Mediterranean-style anti-inflammatory diet",
            "High-fiber foods (beans, lentils, oats, whole grains)",
            "Lean proteins (fish, tofu, chicken, legumes)",
            "Antioxidant-rich berries, green vegetables, and nuts",
            "Coffee (filtered, 2 cups/day — associated with reduced liver stiffness)",
            "Omega-3 rich fatty fish (salmon, sardines, mackerel)",
        ],
        "avoid": [
            "All alcohol — even small amounts accelerate liver fibrosis",
            "High-fat processed and fried foods",
            "Red and processed meats",
            "Excess iron-containing supplements (hepatotoxic in HCV)",
            "Grapefruit juice (interferes with DAA antiviral medications)",
            "Raw shellfish and sushi",
            "High-sodium processed foods",
        ],
    },

    "Hepatitis D": {
        "eat": [
            "Low-sodium, liver-friendly whole foods diet",
            "High-quality lean proteins (fish, eggs, tofu)",
            "Anti-inflammatory foods (turmeric, ginger, olive oil)",
            "High-fiber foods to prevent constipation (important in liver disease)",
            "Vitamin E-rich foods (almonds, sunflower seeds, avocado)",
            "Fresh fruits and vegetables rich in antioxidants",
        ],
        "avoid": [
            "All alcohol — permanently and strictly",
            "High-sodium foods (worsen ascites and portal hypertension)",
            "Processed and fatty meats",
            "Raw and potentially contaminated seafood",
            "Unverified herbal liver remedies",
            "High-protein diet if hepatic encephalopathy is a risk",
        ],
    },

    "Hepatitis E": {
        "eat": [
            "Easily digestible high-carbohydrate foods (rice, oats, toast)",
            "Boiled and thoroughly cooked foods only",
            "Coconut water and ORS for hydration",
            "Ripe bananas and cooked fruits",
            "Lean, well-cooked proteins (eggs, chicken, fish)",
            "Herbal broths and warm teas",
        ],
        "avoid": [
            "All alcohol — mandatory during HEV infection",
            "Raw or undercooked pork, venison, and shellfish",
            "Unboiled or unfiltered water and ice",
            "Street food and foods from unsanitary sources",
            "Fried and oily foods",
            "Excess paracetamol and NSAIDs",
        ],
    },

    "Alcoholic Hepatitis": {
        "eat": [
            "High-protein, high-calorie meals (eggs, legumes, lean meats as tolerated)",
            "Thiamine (Vitamin B1) and B-complex supplement foods",
            "Soft, easily digestible foods (porridge, soups, khichdi)",
            "Fruits and vegetables for antioxidant vitamins",
            "Low-sodium foods (less than 2g/day) to manage ascites",
            "Milk thistle tea (hepatoprotective herb — after doctor clearance)",
        ],
        "avoid": [
            "All alcohol — zero tolerance, immediately and permanently",
            "High-sodium foods (pickles, salty snacks, canned foods)",
            "Paracetamol and ibuprofen (hepatotoxic with compromised liver)",
            "High-fat and fried foods",
            "Raw shellfish and undercooked foods",
            "Unverified herbal remedies and supplements",
        ],
    },

    "Tuberculosis": {
        "eat": [
            "High-protein foods (eggs, milk, lentils, legumes, meat — to combat wasting)",
            "High-calorie nutritious meals to restore lost body weight",
            "Vitamins A, C, and E-rich fruits and vegetables",
            "Zinc-rich foods (pumpkin seeds, legumes, whole grains)",
            "Vitamin D-rich foods and safe sun exposure",
            "Iron-rich foods (spinach, lentils, pomegranate — treat TB-related anemia)",
            "Probiotic foods (yogurt) to counteract antibiotic effect on gut flora",
        ],
        "avoid": [
            "Alcohol (causes serious hepatotoxicity with TB medications)",
            "Smoking and tobacco in any form",
            "High-fat and hard-to-digest fried foods",
            "Refined sugars and sweetened beverages",
            "Caffeine in excess (interferes with medication absorption)",
            "Tyramine-rich foods (aged cheese, fermented foods, red wine) if on Isoniazid",
        ],
    },

    "Common Cold": {
        "eat": [
            "Warm chicken or vegetable soup/broth (anti-inflammatory and decongestant)",
            "Hot ginger tea with honey and lemon",
            "Garlic (allicin — natural antiviral and antimicrobial)",
            "Vitamin C-rich foods (oranges, kiwi, guava, strawberries)",
            "Spicy foods in moderation (capsaicin clears nasal congestion)",
            "Turmeric milk (anti-inflammatory)",
            "Plenty of warm fluids (2.5–3 litres/day)",
            "Zinc-rich foods (pumpkin seeds, oysters, lentils — may shorten cold duration)",
        ],
        "avoid": [
            "Cold dairy products (may temporarily thicken mucus)",
            "Alcohol (dehydrates and impairs immune response)",
            "Sugary foods and drinks (suppress white blood cell function)",
            "Fried and greasy foods",
            "Caffeine in excess (dehydrating effect)",
            "Ice-cold beverages and cold desserts",
        ],
    },

    "Pneumonia": {
        "eat": [
            "High-calorie, protein-rich foods for immune repair (eggs, chicken, fish)",
            "Warm broths and clear soups",
            "Vitamin C-rich foods (citrus fruits, bell peppers, guava)",
            "Vitamin D foods (fortified milk, fatty fish, eggs)",
            "Zinc-rich foods (nuts, seeds, legumes)",
            "Warm herbal teas (honey, ginger, turmeric)",
            "Plenty of warm water (2.5–3 litres/day — thins mucus)",
        ],
        "avoid": [
            "Cold foods and iced beverages",
            "Alcohol (severely suppresses immune response in lung infections)",
            "Smoking and exposure to smoke",
            "High-fat fried and processed foods",
            "Sugary foods and drinks",
            "Dairy products that increase mucus production (if problematic)",
        ],
    },

    "Dimorphic Hemmorhoids(Piles)": {
        "eat": [
            "High-fiber foods (30–35g/day: whole grains, oats, bran, psyllium husk)",
            "Prunes and figs (natural stool softeners)",
            "Fresh fruits and vegetables (especially leafy greens)",
            "Legumes and lentils (fiber and protein)",
            "Plenty of water (3+ litres/day — softens stool)",
            "Papaya and banana (digestive regulators)",
            "Flaxseeds and chia seeds (fiber and anti-inflammatory omega-3s)",
        ],
        "avoid": [
            "Low-fiber refined foods (white bread, white rice, processed snacks)",
            "Spicy foods (irritate the anal mucosa and worsen itching)",
            "Alcohol (causes dehydration and harder stools)",
            "Red meat (low in fiber, hard on digestion)",
            "Caffeinated beverages (dehydrating)",
            "Dairy products in excess (can cause constipation)",
            "Processed and salty packaged foods",
        ],
    },

    "Heart Attack": {
        "eat": [
            "Heart-healthy Mediterranean diet (olive oil, fish, nuts, legumes)",
            "Omega-3 rich fatty fish (salmon, tuna, sardines — 2x per week)",
            "Whole grains (oats, brown rice, whole wheat)",
            "Fruits and vegetables rich in antioxidants",
            "Low-fat dairy (calcium without saturated fat)",
            "Walnuts and almonds (heart-protective fats)",
            "Garlic and leafy greens (lower cholesterol naturally)",
        ],
        "avoid": [
            "Saturated and trans fats (red meat, butter, ghee, full-fat dairy, fried foods)",
            "High-sodium foods (raises blood pressure — risk of re-infarction)",
            "Alcohol (arrhythmia and blood pressure concerns)",
            "Sugary foods and refined carbohydrates",
            "Processed and packaged snacks",
            "Caffeine in excess (can trigger arrhythmias)",
            "Large heavy meals (increases cardiac workload — eat smaller portions)",
        ],
    },

    "Varicose Veins": {
        "eat": [
            "High-fiber foods (prevent constipation and straining)",
            "Flavonoid-rich berries (blueberries, blackberries — strengthen vein walls)",
            "Vitamin C-rich foods (collagen production for vessel integrity)",
            "Omega-3 fatty acids (reduce inflammation)",
            "Water (2–3 litres/day — reduces blood viscosity)",
            "Ginger and turmeric (anti-inflammatory)",
            "Leafy greens rich in Vitamin K (supports blood flow regulation)",
        ],
        "avoid": [
            "High-sodium foods (cause fluid retention and swelling)",
            "Alcohol (dilates veins and worsens congestion)",
            "Refined sugars and processed carbohydrates",
            "Saturated fats and trans fats",
            "Large meals that cause bloating and abdominal pressure",
            "Caffeine in excess (dehydration and vascular stress)",
        ],
    },

    "Hypothyroidism": {
        "eat": [
            "Iodine-rich foods (iodized salt, seafood, dairy) — thyroid hormone synthesis",
            "Selenium-rich foods (Brazil nuts 1–2/day, tuna, eggs)",
            "Zinc-rich foods (oysters, pumpkin seeds, lentils)",
            "Tyrosine-rich lean proteins (chicken, fish, eggs, tofu)",
            "Whole grains and high-fiber foods for metabolism",
            "Vitamin D sources (egg yolks, fortified milk, sunlight)",
        ],
        "avoid": [
            "Goitrogens in raw form (raw cabbage, broccoli, cauliflower, kale — cooking neutralizes)",
            "Soy products (interfere with thyroid hormone absorption)",
            "Gluten (may worsen autoimmune thyroiditis in susceptible individuals)",
            "Highly processed and high-sugar foods",
            "Excess dietary fiber at the time of medication (impairs levothyroxine absorption)",
            "Coffee and calcium supplements within 4 hours of thyroid medication",
        ],
    },

    "Hyperthyroidism": {
        "eat": [
            "Calcium-rich foods (dairy, fortified plant milks, tofu, sesame)",
            "Vitamin D sources (eggs, mushrooms, fortified foods)",
            "Cruciferous vegetables (broccoli, cauliflower, cabbage — naturally reduce thyroid activity)",
            "Balanced protein sources (chicken, legumes, fish)",
            "Low-iodine diet during radioiodine treatment (as directed by doctor)",
            "Nutrient-dense antioxidant foods",
        ],
        "avoid": [
            "Iodine-rich foods (seafood, seaweed, iodized salt — stimulate thyroid hormone excess)",
            "Caffeine (worsens palpitations, anxiety, and tremors)",
            "Alcohol",
            "Highly processed foods and stimulants",
            "Energy drinks and stimulant supplements",
            "Kelp and seaweed supplements (extremely high iodine content)",
        ],
    },

    "Hypoglycemia": {
        "eat": [
            "Complex carbohydrates (oats, whole grain bread, brown rice)",
            "Small, frequent meals every 2–3 hours",
            "Protein-rich snacks (boiled eggs, cheese, peanut butter, Greek yogurt)",
            "High-fiber foods (slow glucose absorption and stabilize blood sugar)",
            "Healthy fats (avocado, nuts — slow digestion and maintain energy)",
            "Fast-acting glucose during episodes (fruit juice, glucose tablets, honey)",
        ],
        "avoid": [
            "Skipping meals or fasting (primary cause of hypoglycemia)",
            "Refined sugars and simple carbohydrates (cause rapid blood sugar spikes then crashes)",
            "Alcohol (blocks glucose release from the liver)",
            "Caffeinated beverages (mask hypoglycemia symptoms)",
            "Very large meals spaced far apart",
            "High-sugar foods followed by inactivity",
        ],
    },

    "Osteoarthristis": {
        "eat": [
            "Anti-inflammatory omega-3 rich foods (salmon, sardines, flaxseeds, walnuts)",
            "Turmeric and ginger (powerful anti-inflammatory)",
            "Vitamin C-rich foods (collagen synthesis for cartilage repair)",
            "Calcium and Vitamin D for bone health",
            "Green tea (antioxidants may protect cartilage)",
            "Cherries and berries (anthocyanins reduce joint inflammation)",
            "Avocado (contains unsaponifiables that inhibit cartilage breakdown)",
        ],
        "avoid": [
            "Refined sugars and high-glycemic foods (promote inflammation)",
            "Saturated and trans fats (red meat, fried foods, margarine)",
            "Omega-6 fatty acid excess (vegetable oils like corn or sunflower in large amounts)",
            "Alcohol (worsens joint inflammation and gout risk)",
            "High-salt foods (retain fluids and worsen joint swelling)",
            "Processed and packaged snack foods",
        ],
    },

    "Arthritis": {
        "eat": [
            "Omega-3 rich anti-inflammatory foods (fatty fish, walnuts, flaxseeds)",
            "Colorful fruits and vegetables (anthocyanins, Vitamin C, beta-carotene)",
            "Olive oil (oleocanthal — NSAID-like anti-inflammatory property)",
            "Turmeric and ginger (curcumin reduces inflammatory cytokines)",
            "Low-fat dairy or fortified alternatives (calcium and Vitamin D)",
            "Legumes and pulses (high protein, low fat)",
        ],
        "avoid": [
            "Refined sugars and high-glycemic foods (AGE compounds worsen joint damage)",
            "Saturated and trans fats (trigger inflammatory processes)",
            "Salt and MSG in excess",
            "Alcohol (especially in gout-related arthritis — raises uric acid)",
            "Purine-rich foods in gout (red meat, organ meats, shellfish, beer)",
            "Processed and ultra-processed foods",
        ],
    },

    "(Vertigo) Paroymsal Positional Vertigo": {
        "eat": [
            "Low-sodium diet (less than 1,500 mg/day — reduces inner ear fluid pressure)",
            "Potassium-rich foods (bananas, sweet potatoes, avocado)",
            "Magnesium-rich foods (nuts, seeds, dark chocolate)",
            "Vitamin D-rich foods (eggs, fortified milk, mushrooms)",
            "Ginger tea (natural anti-nausea remedy for vertigo episodes)",
            "Adequate hydration — water, herbal teas",
        ],
        "avoid": [
            "High-sodium foods (pickles, salty chips, processed meats — worsen Meniere's-related vertigo)",
            "Alcohol (directly disrupts inner ear fluid balance)",
            "Caffeine (coffee, energy drinks — can trigger vertigo attacks)",
            "Tobacco and smoking",
            "Tyramine-rich foods if associated with vestibular migraines",
            "Dehydration (worsens vestibular symptoms)",
        ],
    },

    "Acne": {
        "eat": [
            "Low-glycemic diet (whole grains, legumes, non-starchy vegetables)",
            "Omega-3 rich foods (fatty fish, walnuts, flaxseeds — reduce skin inflammation)",
            "Zinc-rich foods (pumpkin seeds, lentils, hemp seeds — regulate sebum)",
            "Probiotic foods (yogurt, kefir — gut-skin axis support)",
            "Vitamin A-rich foods (sweet potatoes, carrots, leafy greens)",
            "Green tea (polyphenols reduce sebum oxidation)",
            "Antioxidant-rich berries and colorful vegetables",
        ],
        "avoid": [
            "High-glycemic foods (white bread, white rice, sugary snacks, sodas — spike insulin and sebum)",
            "Dairy products (whey protein and hormones in milk may worsen acne)",
            "Chocolate and cocoa in large amounts",
            "Fried and fast foods (trans fats promote inflammation)",
            "Refined sugars and processed carbohydrates",
            "Whey protein supplements (elevates IGF-1 and sebum production)",
        ],
    },

    "Urinary Tract Infection": {
        "eat": [
            "Plenty of water (2–3 litres/day — flushes bacteria from urinary tract)",
            "Cranberry juice (unsweetened — contains proanthocyanidins, reduce bacterial adhesion)",
            "Probiotic foods (yogurt, kefir — restore healthy urinary microbiome)",
            "Vitamin C-rich foods (acidify urine, inhibit bacterial growth)",
            "Garlic (natural antimicrobial allicin)",
            "Parsley tea (natural diuretic — aids urinary flushing)",
        ],
        "avoid": [
            "Alcohol (irritates the bladder and worsens symptoms)",
            "Caffeine and carbonated beverages (bladder irritants)",
            "Spicy foods (irritate the urinary tract lining)",
            "Sugary foods and sodas (feed bacterial growth)",
            "Artificial sweeteners (bladder irritants)",
            "Citrus juices in large amounts (can worsen burning sensation)",
        ],
    },

    "Psoriasis": {
        "eat": [
            "Anti-inflammatory omega-3 rich foods (fatty fish, walnuts, chia seeds)",
            "Colorful antioxidant-rich fruits and vegetables",
            "Turmeric (curcumin reduces psoriatic inflammation)",
            "Probiotic-rich foods (yogurt, kimchi, sauerkraut — gut microbiome and immune regulation)",
            "Vitamin D-rich foods and safe sun exposure",
            "Lean proteins (chicken, fish, legumes)",
            "Olive oil (skin-protective oleic acid)",
        ],
        "avoid": [
            "Alcohol (strongly associated with psoriasis flare-ups)",
            "Refined sugars and high-glycemic foods",
            "Red meat and processed meats (arachidonic acid — inflammatory)",
            "Dairy in excess (may worsen inflammation in some patients)",
            "Gluten (if gluten-sensitive psoriasis is suspected)",
            "Nightshade vegetables in sensitive individuals (tomatoes, peppers, eggplant)",
            "Fried and ultra-processed junk foods",
        ],
    },

    "Impetigo": {
        "eat": [
            "Vitamin C-rich foods (kiwi, guava, citrus — supports wound healing)",
            "Zinc-rich foods (legumes, seeds, nuts — immune support and skin repair)",
            "Protein-rich foods (eggs, chicken, fish, legumes — tissue regeneration)",
            "Plenty of water for hydration and toxin elimination",
            "Probiotic foods (to counteract antibiotic effects on gut flora)",
            "Iron-rich foods (spinach, lentils — support immune defense)",
        ],
        "avoid": [
            "Sugary foods (suppress immune response and promote bacterial growth)",
            "Highly processed and packaged foods",
            "Alcohol during antibiotic treatment",
            "Spicy and irritating foods",
            "Foods that trigger contact dermatitis if relevant",
            "Sharing utensils or food (prevent bacterial spread to others)",
        ],
    },
}


def get_disease_diet(disease: str) -> dict | None:
    """
    Return the {eat, avoid} diet dict for a given disease name.

    Matching is case-insensitive and normalised. Returns None if no
    disease-specific diet data exists in the knowledge base.
    """
    import re

    def _normalise(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip()).lower()

    target = _normalise(disease)
    for key, value in DIET_DB.items():
        if _normalise(key) == target:
            return value

    # Partial match fallback (e.g. "Dimorphic Hemmorhoids(Piles)" vs "Dimorphic Hemmorhoids")
    for key, value in DIET_DB.items():
        if target in _normalise(key) or _normalise(key) in target:
            return value

    return None
