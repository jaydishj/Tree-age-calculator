# -*- coding: utf-8 -*-
"""
tree_age_iks_100_tamil.py
AI மர வகை மற்றும் வயது கணிப்பு (100 இனங்கள்) + IKS (தமிழ்) இணைப்பு
Author: Generated for user
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1️⃣ 100 Common Indian Trees & Plants
# ----------------------------
species_names = [
    # Common trees
    "mango","neem","banyan","peepal","teak","sal","sandalwood","rosewood","mahogany","acacia",
    "ashoka","gulmohar","rain tree","eucalyptus","jamun","guava","jackfruit","tamarind","coconut","bamboo",
    # Medicinal & useful plants
    "amla","drumstick","kadamba","pongamia","arjuna","bael","custard apple","indian almond","bottlebrush","silk cotton",
    "fig","albizia","karanja","tulip tree","silver oak","pine","deodar","oak","maple","cedar",
    # Fruit-bearing & spice plants
    "banana","papaya","cashew","sapota","mangosteen","nutmeg","clove","coffee","tea","black pepper",
    # Herbs & medicinal
    "tulsi","mint","basil","lemongrass","rosemary","sage","aloevera","ginger","turmeric","cardamom",
    "coriander","cumin","fenugreek","castor","sunflower","sesame","mustard","linseed","cotton","okra",
    # Vegetables & vines
    "brinjal","tomato","chili","onion","garlic","spinach","amaranthus","cucumber","pumpkin","bottle gourd",
    "ridge gourd","snake gourd","bitter gourd","watermelon","muskmelon","carrot","beetroot","radish","yam","sweet potato",
    # Traditional & sacred
    "thespesia populnea","saraca asoca","madhuca longifolia","dalbergia latifolia","ficus religiosa","ficus benghalensis",
    "santalum album","syzygium cumini","terminalia arjuna","polyalthia longifolia"
]

# Ensure exactly 100 species
species_names = species_names[:100]

# ----------------------------
# 2️⃣ Generate Synthetic Data
# ----------------------------
np.random.seed(42)
species_data = {
    "species": species_names,
    "leaf_shape": np.random.choice(["broad","oval","needle","compound","heart","lanceolate"], len(species_names)),
    "bark_texture": np.random.choice(["smooth","rough","flaky","fibrous","grooved"], len(species_names)),
    "habitat": np.random.choice(["tropical","dry","coastal","hill","plain","rainforest"], len(species_names)),
    "fruit_presence": np.random.choice(["yes","no"], len(species_names)),
    "average_height_m": np.round(np.random.uniform(2, 60, len(species_names)), 2),
    "leaf_size_cm": np.round(np.random.uniform(2, 45, len(species_names)), 2),
    "growth_factor": np.round(np.random.uniform(1.4, 5.0, len(species_names)), 2)
}
df_species = pd.DataFrame(species_data)

# ----------------------------
# 3️⃣ Train Decision Tree Classifier
# ----------------------------
X = pd.get_dummies(df_species[["leaf_shape","bark_texture","habitat","fruit_presence"]])
X["average_height_m"] = df_species["average_height_m"]
X["leaf_size_cm"] = df_species["leaf_size_cm"]
y = df_species["species"]

clf = DecisionTreeClassifier(random_state=42, max_depth=10)
clf.fit(X, y)

# ----------------------------
# 4️⃣ IKS Tamil Knowledge Base
# ----------------------------
IKS_DB_PATH = "iks_tamil_100_db.json"
prepopulated = {
    "mango": {"tamil_name":"மாமரம்","english_name":"Mango","uses_tamil":"பழம், மருந்து, நிழல்.","notes_tamil":"இந்திய பாரம்பரிய மரம்."},
    "neem": {"tamil_name":"வேம்பு","english_name":"Neem","uses_tamil":"மருந்து மற்றும் கிருமிநாசினி.","notes_tamil":"ஆயுர்வேதத்தில் முக்கியம்."},
    "banyan": {"tamil_name":"ஆலமரம்","english_name":"Banyan","uses_tamil":"நிழல், வழிபாட்டு மரம்.","notes_tamil":"நீண்ட ஆயுள் கொண்டது."},
    "peepal": {"tamil_name":"அரசமரம்","english_name":"Peepal","uses_tamil":"வழிபாடு மற்றும் மருந்து.","notes_tamil":"புனிதமான மரம்."},
    "teak": {"tamil_name":"தேக்கு","english_name":"Teak","uses_tamil":"மரப்பணி மற்றும் கட்டிடம்.","notes_tamil":"வலுவான மரம்."},
    "coconut": {"tamil_name":"தென்னை","english_name":"Coconut","uses_tamil":"எண்ணெய், உணவு, மருந்து.","notes_tamil":"வாழ்க்கை மரம்."},
    "amla": {"tamil_name":"நெல்லிக்காய்","english_name":"Amla","uses_tamil":"C வைட்டமின் ஆதாரம்.","notes_tamil":"மருந்து பயன்பாடு."},
    "drumstick": {"tamil_name":"முருங்கை","english_name":"Drumstick","uses_tamil":"இலைகள் மற்றும் காய் ஊட்டச்சத்து.","notes_tamil":"பசுமை உணவு மரம்."},
    "jackfruit": {"tamil_name":"பலாப்பழம்","english_name":"Jackfruit","uses_tamil":"உணவு, மருந்து.","notes_tamil":"வணிகப் பயிர்."},
    "default": {"tamil_name":"","english_name":"","uses_tamil":"இந்த மரத்திற்கான பாரம்பரிய தகவல் இல்லை.","notes_tamil":"புதிய தகவலை சேர்க்கலாம்."}
}
iks_db = prepopulated.copy()

# ----------------------------
# 5️⃣ Tamil Output Function
# ----------------------------
def pretty_tamil_output(species, iks, circ, dia, age):
    t = iks.get("tamil_name","")
    e = iks.get("english_name","")
    print(f"\n🌳 மரம்: {t or species.capitalize()} ({e}) — {species}")
    print(f"📏 சுற்றளவு: {circ} cm")
    print(f"📐 விட்டம்: {dia:.2f} cm")
    print(f"🕰️ கணிக்கப்பட்ட வயது: {age:.1f} ஆண்டு(கள்)")
    print(f"🌿 பயன்பாடு: {iks.get('uses_tamil','-')}")
    print(f"📝 குறிப்புகள்: {iks.get('notes_tamil','-')}\n")

# ----------------------------
# 6️⃣ Main Tamil Interactive
# ----------------------------
def main():
    print("\n🌿 AI மர வகை மற்றும் வயது கணிப்பு (100 இனங்கள்) - IKS தமிழ் இணைப்பு 🌿\n")

    leaf = input("இலை வடிவம் (broad/oval/needle/...): ").strip().lower()
    bark = input("தோல் அமைப்பு (smooth/rough/...): ").strip().lower()
    hab = input("வாழ்விடம் (tropical/dry/...): ").strip().lower()
    fruit = input("பழம் உள்ளதா? (yes/no): ").strip().lower()
    h = float(input("சுமார் உயரம் (மீ): "))
    l = float(input("இலை அளவு (செ.மீ): "))
    c = float(input("மர சுற்றளவு (செ.மீ): "))

    df = pd.DataFrame([{
        "leaf_shape": leaf,
        "bark_texture": bark,
        "habitat": hab,
        "fruit_presence": fruit,
        "average_height_m": h,
        "leaf_size_cm": l
    }])
    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=X.columns, fill_value=0)

    species = clf.predict(df_enc)[0]
    gf = df_species.loc[df_species["species"] == species, "growth_factor"].values[0]
    dia = c / math.pi
    age = dia * gf
    iks = iks_db.get(species, iks_db["default"])
    pretty_tamil_output(species, iks, c, dia, age)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nநீங்கள் செயலியை நிறுத்தினீர்கள். 🌿")
        sys.exit(0)
