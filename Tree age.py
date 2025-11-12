# -*- coding: utf-8 -*-
"""
tree_age_iks_150_tamil.py
AI மர வகை மற்றும் வயது கணிப்பு (150 இனங்கள்) + IKS (தமிழ்) இணைப்பு
Author: Generated for user
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1️⃣ 150 Indian Tree & Plant Species
# ----------------------------
species_names = [
    # Common Indian Trees
    "mango","neem","banyan","peepal","teak","sal","sandalwood","rosewood","mahogany","acacia",
    "babool","ashoka","gulmohar","rain tree","eucalyptus","jamun","guava","jackfruit","tamarind","coconut",
    "areca palm","rubber","casuarina","bamboo","fig","amla","drumstick","kadamba","pongamia","arjuna",
    "bael","custard apple","flame tree","indian almond","bottlebrush","silk cotton","indian coral","kadam","siris","subabul",
    "albizia","karanja","tulip tree","silver oak","pine","deodar","oak","maple","ash","cedar",
    "cypress","willow","poplar","birch","cashew","papaya","banana","mulberry","indian cherry","sapota",
    "mangosteen","clove","nutmeg","coffee","tea","black pepper","cinnamon","palmyra","date palm","white teak",
    # Medicinal & Herbal Plants
    "tulsi","mint","basil","lemongrass","oregano","thyme","rosemary","sage","aloevera","ginger",
    "turmeric","cardamom","fennel","coriander","cumin","fenugreek","castor","sunflower","sesame","mustard",
    "linseed","cotton","okra","brinjal","tomato","chili","onion","garlic","spinach","amaranthus",
    # Gourds and Fruits
    "cauliflower","cabbage","pumpkin","ridge gourd","bottle gourd","bitter gourd","snake gourd","cucumber","watermelon","muskmelon",
    "carrot","beetroot","radish","sweet potato","cassava","yam","arrowroot","betel leaf","vanilla","pomegranate",
    # Forest & Flower Trees
    "star fruit","breadfruit","kiwi","strawberry","blackberry","orange","lemon","lime","gooseberry","lychee",
    "rambutan","fig variant","rose","jasmine","hibiscus","bougainvillea","thespesia populnea","saraca asoca","butea monosperma","madhuca longifolia",
    "azadirachta indica","dalbergia latifolia","terminalia arjuna","terminalia bellirica","terminalia chebula","ficus religiosa","ficus benghalensis","santalum album","syzygium cumini","polyalthia longifolia"
]

# Make sure exactly 150 species
if len(species_names) > 150:
    species_names = species_names[:150]

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

clf = DecisionTreeClassifier(random_state=42, max_depth=12)
clf.fit(X, y)

# ----------------------------
# 4️⃣ Tamil IKS Knowledge Base
# ----------------------------
IKS_DB_PATH = "iks_tamil_150_db.json"
prepopulated = {
    "mango": {"tamil_name":"மாமரம்","english_name":"Mango","uses_tamil":"பழம், மருந்து, நிழல்.","notes_tamil":"இந்திய பாரம்பரிய மரம்."},
    "neem": {"tamil_name":"வேம்பு","english_name":"Neem","uses_tamil":"மருந்து மற்றும் கிருமிநாசினி.","notes_tamil":"முக்கிய ஆயுர்வேத மரம்."},
    "banyan": {"tamil_name":"ஆலமரம்","english_name":"Banyan","uses_tamil":"நிழல், வழிபாட்டு மரம்.","notes_tamil":"பழைய மரங்களின் ஆயுள் நீண்டது."},
    "teak": {"tamil_name":"தேக்கு","english_name":"Teak","uses_tamil":"மரப்பணி, கட்டிடம்.","notes_tamil":"வலுவான மரம்."},
    "amla": {"tamil_name":"நெல்லிக்காய்","english_name":"Amla","uses_tamil":"C வைட்டமின் ஆதாரம், மருந்து.","notes_tamil":"உணவு மற்றும் மருத்துவ பயன்பாடு."},
    "drumstick": {"tamil_name":"முருங்கை","english_name":"Drumstick","uses_tamil":"உணவு, மருந்து, இலைச்சாறு.","notes_tamil":"உணவு மரபில் முக்கியம்."},
    "coconut": {"tamil_name":"தென்னை","english_name":"Coconut","uses_tamil":"பழம், எண்ணெய், நெய்.","notes_tamil":"வாழ்க்கை மரம் என அழைக்கப்படுகிறது."},
    "default": {"tamil_name":"","english_name":"","uses_tamil":"இந்த மரத்திற்கான பாரம்பரிய தகவல் இல்லை.","notes_tamil":"புதிய தகவலை சேர்க்கலாம்."}
}
iks_db = prepopulated.copy()

# ----------------------------
# 5️⃣ Tamil Output Formatter
# ----------------------------
def pretty_tamil_output(species, iks_entry, circumference, diameter, age):
    lines = []
    tamil_name = iks_entry.get("tamil_name") or species.capitalize()
    eng = iks_entry.get("english_name","")
    lines.append(f"🌳 மரம்: {tamil_name} ({eng}) — {species}")
    lines.append(f"📏 சுற்றளவு: {circumference} cm")
    lines.append(f"📐 விட்டம்: {diameter:.2f} cm")
    lines.append(f"🕰️ கணிக்கப்பட்ட வயது: {age:.1f} ஆண்டு(கள்)")
    lines.append(f"\n🌿 பயன்பாடு: {iks_entry.get('uses_tamil','-')}")
    lines.append(f"📝 குறிப்புகள்: {iks_entry.get('notes_tamil','-')}")
    return "\n".join(lines)

# ----------------------------
# 6️⃣ Main Tamil CLI Function
# ----------------------------
def main():
    print("\n🌿 AI மர வகை மற்றும் வயது கணிப்பு (150 இனங்கள்) - IKS தமிழ் இணைப்பு 🌿\n")

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

    print("\n" + pretty_tamil_output(species, iks, c, dia, age) + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nநீங்கள் செயலியை நிறுத்தினீர்கள். 🌿")
        sys.exit(0)
