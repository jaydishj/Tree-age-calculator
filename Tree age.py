# -*- coding: utf-8 -*-
"""
tree_age_iks_200_tamil.py
AI மர வகை மற்றும் வயது கணிப்பு (200 இனங்கள்) + IKS (தமிழ்) இணைப்பு
Author: Generated for user
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1) 200 Indian species list
# ----------------------------
species_names = [
    # Common Trees
    "mango","neem","banyan","peepal","teak","sal","sandalwood","rosewood","mahogany","acacia",
    "babool","ashoka","gulmohar","rain tree","eucalyptus","jamun","guava","jackfruit","tamarind","coconut",
    "areca palm","rubber","casuarina","bamboo","fig","amla","drumstick","kadamba","pongamia","arjuna",
    "bael","custard apple","flame tree","indian almond","bottlebrush","silk cotton","indian coral","kadam","siris","subabul",
    "albizia","karanja","tulip tree","silver oak","pine","deodar","oak","maple","ash","cedar",
    "cypress","willow","poplar","birch","cashew","papaya","banana","mulberry","tendu","indian cherry",
    "sapota","mangosteen","clove","nutmeg","coffee","tea","black pepper","cinnamon","palmyra","date palm",
    "white teak","pungam","champa","plumeria","mahua","red cedar","apple","pear","peach","cherry",
    "almond","walnut","plum","apricot","persimmon","betel nut","wild jack","hibiscus","bougainvillea","jasmine",
    # Medicinal & Herbs
    "marigold","tulsi","mint","basil","lemongrass","oregano","thyme","rosemary","sage","aloevera",
    "ginger","turmeric","galangal","cardamom","fennel","coriander","cumin","fenugreek","castor","sunflower",
    "sesame","mustard","linseed","hemp","cotton","okra","brinjal","tomato","chili","potato",
    "onion","garlic","spinach","amaranthus","cauliflower","cabbage","broccoli","pumpkin","ridge gourd","bottle gourd",
    "bitter gourd","snake gourd","cucumber","watermelon","muskmelon","melon","lettuce","beetroot","carrot","radish",
    # Palms, Forest & Ornamentals
    "yam","sweet potato","cassava","arrowroot","lavender","bamboo palm","fan palm","royal palm","silver date palm","foxtail palm",
    "betel leaf","vanilla","kokum","star fruit","breadfruit","custard pear","dragon fruit","kiwi","pomegranate","blueberry",
    "strawberry","blackberry","raspberry","cranberry","gooseberry","lychee","rambutan","lemon","lime","orange",
    # Timber & Indian IKS species
    "grapefruit","citron","mandarin","noni","baobab","acacia nilotica","dalbergia latifolia","terminalia bellirica",
    "terminalia chebula","emblica officinalis","cassia fistula","delonix regia","bauhinia purpurea","peltophorum pterocarpum",
    "lagerstroemia speciosa","millingtonia hortensis","polyalthia longifolia","ficus benghalensis","ficus religiosa",
    "ficus racemosa","azadirachta indica","swietenia mahagoni","mimusops elengi","syzygium cumini","eucalyptus globulus",
    "grevillea robusta","santalum album","butea monosperma","madhuca longifolia","pithecellobium dulce","cassia siamea",
    "albizia lebbeck","albizia saman","alstonia scholaris","barringtonia acutangula","careya arborea","cochlospermum religiosum",
    "cordia dichotoma","dillenia indica","feronia limonia","morinda tinctoria","tamarindus indica","terminalia arjuna","wrightia tinctoria","syzygium aromaticum","thespesia populnea","saraca asoca"
]

# Ensure exactly 200
if len(species_names) > 200:
    species_names = species_names[:200]

# ----------------------------
# 2) Synthetic Data
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
# 3) Model Training
# ----------------------------
X = pd.get_dummies(df_species[["leaf_shape","bark_texture","habitat","fruit_presence"]])
X["average_height_m"] = df_species["average_height_m"]
X["leaf_size_cm"] = df_species["leaf_size_cm"]
y = df_species["species"]

clf = DecisionTreeClassifier(random_state=42, max_depth=14)
clf.fit(X, y)

# ----------------------------
# 4) Tamil IKS Base
# ----------------------------
IKS_DB_PATH = "iks_tamil_200_db.json"
prepopulated = {
    "mango": {"tamil_name":"மாமரம்","english_name":"Mango","uses_tamil":"பழம் உணவாக, மருந்தாக பயன்படும்.","notes_tamil":"இந்தியாவின் முக்கிய பழமரம்."},
    "neem": {"tamil_name":"வேம்பு","english_name":"Neem","uses_tamil":"மருந்து மற்றும் கிருமிநாசினி பயன்பாடு.","notes_tamil":"பாரம்பரிய மருத்துவத்தில் முக்கியம்."},
    "banyan": {"tamil_name":"ஆலமரம்","english_name":"Banyan","uses_tamil":"நிழல் மற்றும் வழிபாட்டில் பயன்படும்.","notes_tamil":"நீண்ட ஆயுள் மரம்."},
    "teak": {"tamil_name":"தேக்கு","english_name":"Teak","uses_tamil":"மரப்பணி, கட்டிடம்.","notes_tamil":"வலுவான மரம்."},
    "amla": {"tamil_name":"நெல்லிக்காய்","english_name":"Amla","uses_tamil":"C வைட்டமின் மூலமாக மருந்து.","notes_tamil":"மருந்து மற்றும் உணவு பயன்பாடு."},
    "default": {"tamil_name":"","english_name":"","uses_tamil":"இந்த மரத்திற்கான பாரம்பரிய தகவல் இல்லை.","notes_tamil":"புதிய தகவலைச் சேர்க்கலாம்."}
}
iks_db = prepopulated.copy()

# ----------------------------
# 5) Output Function
# ----------------------------
def pretty_tamil_output(species, iks_entry, circ, dia, age):
    lines = []
    tamil_name = iks_entry.get("tamil_name") or species.capitalize()
    eng = iks_entry.get("english_name","")
    lines.append(f"🌳 மரம்: {tamil_name} ({eng}) — {species}")
    lines.append(f"📏 சுற்றளவு: {circ} cm")
    lines.append(f"📐 விட்டம்: {dia:.2f} cm")
    lines.append(f"🕰️ கணிக்கப்பட்ட வயது: {age:.1f} ஆண்டு(கள்)")
    lines.append(f"\n🌿 பயன்பாடு: {iks_entry.get('uses_tamil','-')}")
    lines.append(f"📝 குறிப்புகள்: {iks_entry.get('notes_tamil','-')}")
    return "\n".join(lines)

# ----------------------------
# 6) Tamil Interactive CLI
# ----------------------------
def main():
    print("\n🌿 AI மர வகை மற்றும் வயது கணிப்பு (200 இனங்கள்) - IKS இணைப்பு\n")

    leaf = input("இலை வடிவம் (broad/oval/needle/...): ").strip().lower()
    bark = input("தோல் அமைப்பு (smooth/rough/...): ").strip().lower()
    hab  = input("வாழ்விடம் (tropical/dry/...): ").strip().lower()
    fruit = input("பழம் உள்ளதா? (yes/no): ").strip().lower()
    h = float(input("சுமார் உயரம் (மீ): "))
    l = float(input("இலை அளவு (செ.மீ): "))
    c = float(input("மர சுற்றளவு (செ.மீ): "))

    df = pd.DataFrame([{
        "leaf_shape":leaf,"bark_texture":bark,"habitat":hab,"fruit_presence":fruit,
        "average_height_m":h,"leaf_size_cm":l
    }])
    df_enc = pd.get_dummies(df)
    df_enc = df_enc.reindex(columns=X.columns, fill_value=0)

    species = clf.predict(df_enc)[0]
    gf = df_species.loc[df_species["species"]==species,"growth_factor"].values[0]
    dia = c / math.pi
    age = dia * gf
    iks = iks_db.get(species, iks_db["default"])

    print("\n"+pretty_tamil_output(species, iks, c, dia, age)+"\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nநீங்கள் செயலியை நிறுத்தினீர்கள். 🌿")
        sys.exit(0)
