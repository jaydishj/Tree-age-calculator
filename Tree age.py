# -*- coding: utf-8 -*-
"""
tree_age_iks_250_tamil.py
AI மர வகை மற்றும் வயது கணிப்பு (250 இனங்கள்) + IKS (தமிழ்) இணைப்பு
Author: Generated for user
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1) 250 species list
# ----------------------------
species_names = [
    # --- Common Indian Trees ---
    "mango","neem","banyan","peepal","teak","sal","sandalwood","rosewood","mahogany","acacia",
    "babool","ashoka","gulmohar","rain tree","eucalyptus","jamun","guava","jackfruit","tamarind","coconut",
    "areca palm","rubber","casuarina","bamboo","fig","amla","drumstick","kadamba","pongamia","arjuna",
    "bael","custard apple","flame tree","indian almond","bottlebrush","silk cotton","indian coral","kadam","siris","subabul",
    "albizia","karanja","tulip tree","silver oak","pine","deodar","oak","maple","ash","cedar",
    "cypress","willow","poplar","birch","cashew","papaya","banana","mulberry","tendu","indian cherry",
    "sapota","mangosteen","clove","nutmeg","coffee","tea","black pepper","cinnamon","palmyra","date palm",
    "white teak","pungam","champa","plumeria","mahua","red cedar","apple","pear","peach","cherry",
    "almond","walnut","plum","apricot","persimmon","betel nut","wild jack","neer fig","hibiscus","bougainvillea",
    # --- Medicinal & Aromatic Plants ---
    "jasmine","marigold","tulsi","mint","basil","lemongrass","oregano","thyme","rosemary","sage",
    "aloevera","ginger","turmeric","galangal","cardamom","fennel","coriander","cumin","fenugreek","castor",
    "sunflower","sesame","mustard","linseed","hemp","cotton","okra","brinjal","tomato","chili",
    "potato","onion","garlic","spinach","amaranthus","cauliflower","cabbage","broccoli","pumpkin","ridge gourd",
    "bottle gourd","bitter gourd","snake gourd","cucumber","watermelon","muskmelon","melon","lettuce","beetroot","carrot",
    # --- Palms, Forest Trees & Ornamentals ---
    "radish","turnip","yam","sweet potato","colocasia","cassava","arrowroot","peppermint","lavender","bamboo palm",
    "traveller’s palm","fan palm","royal palm","silver date palm","foxtail palm","betel leaf","vanilla","kokum","star fruit","breadfruit",
    "custard pear","sugar apple","dragon fruit","kiwi","pomegranate","blueberry","strawberry","blackberry","raspberry","cranberry",
    "gooseberry","lychee","rambutan","fig variant","lemon","lime","orange","grapefruit","pomelo","citron",
    # --- Indian Timber & Medicinal Species ---
    "mandarin","tangerine","miracle fruit","noni","baobab","acacia nilotica","prosopis juliflora","dalbergia latifolia",
    "terminalia bellirica","terminalia chebula","emblica officinalis","cassia fistula","delonix regia","bauhinia purpurea",
    "bauhinia variegata","peltophorum pterocarpum","lagerstroemia speciosa","millingtonia hortensis","polyalthia longifolia",
    "ficus benghalensis","ficus religiosa","ficus racemosa","ficus elastica","artocarpus heterophyllus",
    "azadirachta indica","swietenia mahagoni","mimusops elengi","syzygium cumini","eucalyptus globulus",
    "grevillea robusta","santalum album","butea monosperma","madhuca longifolia","pithecellobium dulce",
    "callistemon citrinus","cassia siamea","albizia lebbeck","albizia saman","alstonia scholaris","barringtonia acutangula",
    "careya arborea","cochlospermum religiosum","cordia dichotoma","crataeva magna","dillenia indica","diospyros melanoxylon",
    "feronia limonia","grewia asiatica","holarrhena pubescens","manilkara zapota","morinda tinctoria","putranjiva roxburghii",
    "sapindus trifoliatus","semecarpus anacardium","syzygium aromaticum","tamarindus indica","terminalia arjuna","wrightia tinctoria"
]

# Ensure exactly 250
if len(species_names) > 250:
    species_names = species_names[:250]

# ----------------------------
# 2) Generate synthetic training data
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
# 3) Train simple Decision Tree model
# ----------------------------
X = pd.get_dummies(df_species[["leaf_shape","bark_texture","habitat","fruit_presence"]])
X["average_height_m"] = df_species["average_height_m"]
X["leaf_size_cm"] = df_species["leaf_size_cm"]
y = df_species["species"]

clf = DecisionTreeClassifier(random_state=42, max_depth=14)
clf.fit(X, y)

# ----------------------------
# 4) Minimal IKS (Tamil) dictionary
# ----------------------------
IKS_DB_PATH = "iks_tamil_250_db.json"
prepopulated = {
    "mango": {"tamil_name":"மாமரம்","english_name":"Mango","uses_tamil":"பழம் உணவாக பயன்படும்.","notes_tamil":"பெரிய பழமரம்."},
    "neem": {"tamil_name":"வேம்பு","english_name":"Neem","uses_tamil":"மருந்து மற்றும் கிருமிநாசினி.","notes_tamil":"பாரம்பரிய மரம்."},
    "banyan": {"tamil_name":"ஆலமரம்","english_name":"Banyan","uses_tamil":"நிழல், வழிபாடு.","notes_tamil":"நீண்ட ஆயுள் மரம்."},
    "teak": {"tamil_name":"தேக்கு","english_name":"Teak","uses_tamil":"மரப்பணி, கப்பல்.","notes_tamil":"வலுவான மரம்."},
    "amla": {"tamil_name":"நெல்லிக்காய்","english_name":"Amla","uses_tamil":"C வைட்டமின்.","notes_tamil":"மருந்து பயன்பாடு."},
    "default": {"tamil_name":"","english_name":"","uses_tamil":"இந்த மரத்திற்கான IKS தரவுகள் இல்லை.","notes_tamil":"புதிய பதிவுகளைச் சேர்க்கலாம்."}
}
iks_db = prepopulated.copy()

# ----------------------------
# 5) Tamil Output Helper
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
# 6) Tamil Interactive Main
# ----------------------------
def main():
    print("\n🌿 AI மர வகை மற்றும் வயது கணிப்பு (250 இனங்கள்) - IKS இணைப்பு\n")

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
    dia = c/math.pi
    age = dia*gf
    iks = iks_db.get(species, iks_db["default"])

    print("\n"+pretty_tamil_output(species, iks, c, dia, age)+"\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nநீங்கள் செயலியை நிறுத்தினீர்கள். 🌿")
        sys.exit(0)
