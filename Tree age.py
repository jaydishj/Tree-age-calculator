# -*- coding: utf-8 -*-
"""
tree_age_iks_100.py
AI-Based Tree Species and Age Estimator (100 Indian Species)
Author: JAYDISH KENNEDY J
"""

import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1️⃣ 100 Common Indian Tree Species
# ----------------------------
species_names = [
    "mango","neem","banyan","peepal","teak","sal","sandalwood","rosewood","mahogany","acacia",
    "ashoka","gulmohar","rain tree","eucalyptus","jamun","guava","jackfruit","tamarind","coconut","bamboo",
    "amla","drumstick","kadamba","pongamia","arjuna","bael","custard apple","indian almond","bottlebrush","silk cotton",
    "fig","albizia","karanja","tulip tree","silver oak","pine","deodar","oak","maple","cedar",
    "banana","papaya","cashew","sapota","mangosteen","nutmeg","clove","coffee","tea","black pepper",
    "tulsi","mint","basil","lemongrass","rosemary","sage","aloevera","ginger","turmeric","cardamom",
    "coriander","cumin","fenugreek","castor","sunflower","sesame","mustard","linseed","cotton","okra",
    "brinjal","tomato","chili","onion","garlic","spinach","amaranthus","cucumber","pumpkin","bottle gourd",
    "ridge gourd","snake gourd","bitter gourd","watermelon","muskmelon","carrot","beetroot","radish","yam","sweet potato",
    "thespesia populnea","saraca asoca","madhuca longifolia","dalbergia latifolia","ficus religiosa","ficus benghalensis",
    "santalum album","syzygium cumini","terminalia arjuna","polyalthia longifolia"
]

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
# 4️⃣ Knowledge Base (English)
# ----------------------------
IKS_DB_PATH = "iks_100_db.json"
prepopulated = {
    "mango": {"english_name":"Mango","uses":"Fruit, shade, medicinal.","notes":"Highly valued tropical tree."},
    "neem": {"english_name":"Neem","uses":"Medicinal, antiseptic, skincare.","notes":"Used in traditional medicine."},
    "banyan": {"english_name":"Banyan","uses":"Shade, worship, ecosystem support.","notes":"Long-living sacred tree."},
    "peepal": {"english_name":"Peepal","uses":"Worship, oxygen supplier.","notes":"Sacred tree in Indian culture."},
    "teak": {"english_name":"Teak","uses":"Timber and furniture wood.","notes":"Strong and durable hardwood."},
    "coconut": {"english_name":"Coconut","uses":"Food, oil, crafts.","notes":"Known as the 'Tree of Life'."},
    "amla": {"english_name":"Amla","uses":"Rich in Vitamin C, medicinal.","notes":"Used in Ayurveda."},
    "drumstick": {"english_name":"Drumstick","uses":"Leaves and pods are edible and nutritious.","notes":"Fast-growing tree."},
    "jackfruit": {"english_name":"Jackfruit","uses":"Fruit, timber, fodder.","notes":"World's largest tree-borne fruit."},
    "default": {"english_name":"","uses":"No data available.","notes":"You can add details later."}
}
iks_db = prepopulated.copy()

# ----------------------------
# 5️⃣ Output Function
# ----------------------------
def show_output(species, iks, circ, dia, age):
    print("\n======================================")
    print(f"🌳 Predicted Tree Species: {species.capitalize()}")
    print(f"📏 Circumference: {circ} cm")
    print(f"📐 Diameter: {dia:.2f} cm")
    print(f"🕰️ Estimated Age: {age:.1f} years")
    print(f"🌿 Uses: {iks.get('uses','-')}")
    print(f"📝 Notes: {iks.get('notes','-')}")
    print("======================================\n")

# ----------------------------
# 6️⃣ Main Interactive Console
# ----------------------------
def main():
    print("\n🌳 AI-Based Tree Species & Age Estimator (100 Indian Species)\n")

    leaf = input("Leaf shape (broad/oval/needle/...): ").strip().lower()
    bark = input("Bark texture (smooth/rough/...): ").strip().lower()
    hab = input("Habitat (tropical/dry/...): ").strip().lower()
    fruit = input("Fruit present? (yes/no): ").strip().lower()
    h = float(input("Approx. average height (m): "))
    l = float(input("Leaf size (cm): "))
    c = float(input("Tree circumference (cm): "))

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
    show_output(species, iks, c, dia, age)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess stopped. Goodbye 🌿")
        sys.exit(0)
