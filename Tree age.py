# -*- coding: utf-8 -*-
"""
tree_age_iks_300_tamil.py
AI மர வகை மற்றும் வயது கணிப்பு (300 இனியோட்டப் பெயர்கள்) + IKS (தமிழ்) இணைப்பு
Author: Generated for user
"""

import streamlit as st
import os
import sys
import json
import math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# 1) 300 species list (primary list)
# ----------------------------
species_names = [
    "mango","neem","banyan","peepal","teak","sal","sandalwood","rosewood","mahogany","acacia",
    "babool","ashoka","gulmohar","rain tree","eucalyptus","jamun","guava","jackfruit","tamarind","coconut",
    "areca palm","rubber","casuarina","bamboo","fig","amla","drumstick","kadamba","pongamia","arjuna",
    "bael","custard apple","flame tree","indian almond","bottlebrush","silk cotton","indian coral","kadam","siris","subabul",
    "albizia","karanja","tulip tree","silver oak","pine","deodar","oak","maple","ash","cedar",
    "cypress","willow","poplar","birch","cashew","papaya","banana","mulberry","tendu","indian cherry",
    "sapota","mangosteen","clove","nutmeg","coffee","tea","black pepper","cinnamon","neer maruthu","palmyra",
    "date palm","cork tree","white teak","pungam","champa","plumeria","mahua","red cedar","apple","pear",
    "peach","cherry","almond","walnut","plum","apricot","persimmon","betel nut","wild jack","neer fig",
    "siris tree","arjun variant","custard variant","amla variant","rosewood variant","rain tree variant",
    "ashoka variant","banyan variant","guava variant","neem variant",
    "hibiscus","bougainvillea","jasmine","marigold","tulsi","mint","basil","lemongrass","oregano","thyme",
    "rosemary","sage","aloevera","ginger","turmeric","galangal","cardamom","fennel","coriander","cumin",
    "fenugreek","castor","sunflower","sesame","mustard","linseed","hemp","cotton","okra","brinjal",
    "tomato","chili","potato","onion","garlic","spinach","amaranthus","cauliflower","cabbage","broccoli",
    "pumpkin","ridge gourd","bottle gourd","bitter gourd","snake gourd","cucumber","watermelon","muskmelon","melon","lettuce",
    "beetroot","carrot","radish","turnip","yam","sweet potato","colocasia","cassava","arrowroot","peppermint",
    "sagebrush","lavender","thyme variant","oregano variant","mint variant","bamboo palm","traveller’s palm","fan palm","sago palm","royal palm",
    "silver date palm","foxtail palm","betel leaf","vanilla","kokum","kokum tree","soursop","star fruit","breadfruit","durian",
    "custard pear","sugar apple","dragon fruit","kiwi","pomegranate","blueberry","strawberry","blackberry","raspberry","cranberry",
    "gooseberry","tamarillo","lychee","longan","rambutan","fig variant","mulberry variant","lemon","lime","orange",
    "grapefruit","pomelo","citron","mandarin","tangerine","lemondrop mangosteen","miracle fruit","noni","baobab","acacia nilotica",
    "prosopis juliflora","dalbergia latifolia","terminalia bellirica","terminalia chebula","emblica officinalis","cassia fistula","delonix regia","bauhinia purpurea","bauhinia variegata","peltophorum pterocarpum",
    "lagerstroemia speciosa","millingtonia hortensis","polyalthia longifolia","ficus benghalensis","ficus religiosa","ficus racemosa","ficus elastica","artocarpus heterophyllus","artocarpus altilis","azadirachta indica",
    "swietenia mahagoni","mimusops elengi","syzygium cumini","eucalyptus globulus","grevillea robusta","santalum album","pterospermum acerifolium","butea monosperma","madhuca longifolia","pithecellobium dulce",
    "callistemon citrinus","cassia siamea","cassia javanica","albizia lebbeck","albizia saman","alstonia scholaris","antidesma acidum","barringtonia acutangula","careya arborea","cochlospermum religiosum",
    "cordia dichotoma","croton bonplandianum","crataeva magna","dillenia indica","diospyros melanoxylon","erythrina variegata","feronia limonia","grewia asiatica","holarrhena pubescens","manilkara zapota",
    "morinda tinctoria","polyalthia longifolia pendula","putranjiva roxburghii","sapindus trifoliatus","semecarpus anacardium","sterculia urens","syzygium aromaticum","tamarindus indica","terminalia arjuna","trichilia emetica",
    "wrightia tinctoria","ziziphus mauritiana","ziziphus jujuba","adansonia digitata","hevea brasiliensis","catharanthus roseus","melia dubia","moringa oleifera","melia azedarach","saraca asoca",
    "michelia champaca","magnolia grandiflora","tithonia diversifolia","melastoma malabathricum","thespesia populnea","vetiver","lemongrass variant","camphor tree","guaiacum officinale","annona reticulata",
    "artemisia annua","tulsi krishna","neem hybrid","mahogany hybrid","sandalwood hybrid","rosewood hybrid","amla hybrid","eucalyptus hybrid","gulmohar hybrid","casuarina hybrid"
]

# ensure exactly 300 (if earlier list shorter, extend with generated names)
if len(species_names) < 300:
    idx = len(species_names) + 1
    while len(species_names) < 300:
        species_names.append(f"species_{idx}")
        idx += 1
elif len(species_names) > 300:
    species_names = species_names[:300]

# ----------------------------
# 2) Create minimal synthetic attributes for classifier
#    (kept simple: leaf_shape, bark_texture, habitat, fruit_presence, avg height, leaf size, growth_factor)
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
    # growth_factor used to compute age = (circumference/pi) * growth_factor
    "growth_factor": np.round(np.random.uniform(1.4, 5.0, len(species_names)), 2)
}
df_species = pd.DataFrame(species_data)

# ----------------------------
# 3) Train a Decision Tree on this lightweight synthetic table
# ----------------------------
X = pd.get_dummies(df_species[["leaf_shape","bark_texture","habitat","fruit_presence"]])
X["average_height_m"] = df_species["average_height_m"]
X["leaf_size_cm"] = df_species["leaf_size_cm"]
y = df_species["species"]

clf = DecisionTreeClassifier(random_state=42, max_depth=14)
clf.fit(X, y)

# ----------------------------
# 4) IKS Tamil knowledge base (prepopulated entries for many common species)
#    File path for persistence
# ----------------------------
IKS_DB_PATH = "iks_tamil_300_db.json"

# Prepopulate Tamil IKS entries for commonly known species.
# For many of the 300 species we provide a default placeholder.
prepopulated = {
    # Common examples (Tamil name, uses, notes) - you can expand/edit later
    "mango": {
        "tamil_name": "மாமரம்",
        "english_name": "Mango",
        "uses_tamil": "மாமரப் பழம் உணவாக பயன்படும்; இலைகள், கூழ்கள் பல மருந்து பயன்பாடுகள்.",
        "notes_tamil": "பாரம்பரியமாக இந்தியாவில் மதிப்பிடப்பட்ட மரம்; பல்வேறு பழவகைகள் உண்டு."
    },
    "neem": {
        "tamil_name": "வேம்பு",
        "english_name": "Neem",
        "uses_tamil": "வழக்கமாக கிருமிநாசினியாகவும், தோல் மருத்துவமாகவும் பயன்படும்; பல் பராமரிப்பிலும் பயன்பாடு.",
        "notes_tamil": "பாரம்பரிய மருத்துவத்தில் முக்கியத்துவம் அதிகம்."
    },
    "banyan": {
        "tamil_name": "ஆலமரம்",
        "english_name": "Banyan",
        "uses_tamil": "பரப்பாக்க மகத்தான நிழல்; வழிபாட்டு மற்றும் சமூக சந்த்பிரதிபலனில் முக்கியம்.",
        "notes_tamil": "பழைய மரங்களின் வாழ்நாள் நீண்டது."
    },
    "peepal": {
        "tamil_name": "பீப்பல்",
        "english_name": "Peepal",
        "uses_tamil": "மத வழிபாடு மற்றும் மரபுத்தனத்தின் அடையாளம்; சிலர் மருத்துவபயன்பாட்டு குறிப்புகளை மேற்கோள் செய்கிறார்கள்.",
        "notes_tamil": "வளர்ச்சி மேலாண்மைக்கு பரவலாக பயனுள்ளது."
    },
    "teak": {
        "tamil_name": "தேக்கு",
        "english_name": "Teak",
        "uses_tamil": "முதன்மையாக கட்டிடக்கலை மற்றும் கப்பல் பணிக்குப் பயன்படும் வலுவான மரம்.",
        "notes_tamil": "ஊரக மற்றும் வணிக மர வனம்."
    },
    "coconut": {
        "tamil_name": "தென்னை",
        "english_name": "Coconut",
        "uses_tamil": "பழம், எண்ணெய் மற்றும் பல பாவனைகளில் பயன்படும்; கடற்கரையில் பயன்பாடு அதிகம்.",
        "notes_tamil": "வாழ்க்கை சார்ந்த மரம்; பலமருந்து பயன்பாடுகள்."
    },
    "amla": {
        "tamil_name": "ஆமலா",
        "english_name": "Amla",
        "uses_tamil": "ஆயுர்வேதத்தில் முக்கியம்; C வைட்டமின் ஆதாரம்.",
        "notes_tamil": "மருந்து மற்றும் உணவு பயன்பாடுகள்."
    },
    "drumstick": {
        "tamil_name": "முருங்கை",
        "english_name": "Drumstick (Moringa)",
        "uses_tamil": "முருங்கை இலைகள், காய் மருத்துவ மற்றும் ஊட்டச்சத்து மூலமாக பயன்படும்.",
        "notes_tamil": "குறைந்த நிலங்களில் வளர்ச்சி சிறந்தது."
    },
    # Add a few more explicit entries
    "sandalwood": {
        "tamil_name": "சந்தனம்",
        "english_name": "Sandalwood",
        "uses_tamil": "அருகம்பா வாசனை, மருந்து மற்றும் ஆன்மீகப் பயன்பாடுகள்.",
        "notes_tamil": "மதிப்புமிக்க வெள்ளிமரப்பலு."
    },
    "jackfruit": {
        "tamil_name": "பலாப்பழம்",
        "english_name": "Jackfruit",
        "uses_tamil": "காயும் பழமும் இரண்டுமே உணவாகவும், வணிகவாய்ப்புகளாகவும் இருக்கும்.",
        "notes_tamil": "உணவு மற்றும் விவசாய பயன்பாடுகள் அதிகம்."
    },
    "guava": {
        "tamil_name": "பேழை",
        "english_name": "Guava",
        "uses_tamil": "பழம், மருத்துவ பயன்பாடுகள் (விநோத தொற்று எதிர்ப்பு).",
        "notes_tamil": "குடியரசு தோல் மற்றும் நன்மைகள்."
    },
    # default fallback entry
    "default": {
        "tamil_name": "",
        "english_name": "",
        "uses_tamil": "இந்த மரத்திற்கான பாரம்பரிய தகவல் தரவுத்தளத்தில் இல்லை. புதிய தகவலைச் சேர்க்கலாம்.",
        "notes_tamil": "பயனாளர் IKS பதிவுகளை சேமிக்கலாம்."
    }
}

# Load or create persistent IKS DB
if os.path.exists(IKS_DB_PATH):
    try:
        with open(IKS_DB_PATH, "r", encoding="utf-8") as f:
            iks_db = json.load(f)
    except Exception:
        iks_db = prepopulated.copy()
else:
    iks_db = prepopulated.copy()
    try:
        with open(IKS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(iks_db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # continue even if file write not permitted

# ----------------------------
# Helper functions
# ----------------------------
def save_iks_entry(species_key, tamil_name, english_name, uses_tamil, notes_tamil):
    key = species_key.lower()
    iks_db[key] = {
        "tamil_name": tamil_name,
        "english_name": english_name,
        "uses_tamil": uses_tamil,
        "notes_tamil": notes_tamil
    }
    try:
        with open(IKS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(iks_db, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def get_iks_for_species(species_key):
    key = species_key.lower()
    if key in iks_db:
        return iks_db[key]
    # try normalized matching
    key2 = key.replace(" ", "").replace("’", "").replace("'", "").lower()
    for k in iks_db:
        nk = k.replace(" ", "").replace("’", "").replace("'", "").lower()
        if nk == key2:
            return iks_db[k]
    return iks_db.get("default")

def pretty_tamil_output(species, iks_entry, circumference_cm, diameter_cm, age_years):
    tamil_lines = []
    tamil_name = iks_entry.get("tamil_name") or species.capitalize()
    eng_name = iks_entry.get("english_name") or ""
    tamil_lines.append(f"🌳 மரம்: {tamil_name}  ({eng_name} — {species})")
    tamil_lines.append(f"📏 சுற்றளவு: {circumference_cm} cm")
    tamil_lines.append(f"📐 விட்டம் (அளவு): {diameter_cm:.2f} cm")
    tamil_lines.append(f"🕰️ கணிக்கப்பட்ட வயது: {age_years:.1f} ஆண்டு(கள்)")
    tamil_lines.append("")
    tamil_lines.append("🌿 பாரம்பரிய பயன்பாடுகள்:")
    tamil_lines.append(iks_entry.get("uses_tamil", "தகவல் இல்லை"))
    tamil_lines.append("")
    tamil_lines.append("📝 குறிப்புகள்:")
    tamil_lines.append(iks_entry.get("notes_tamil", "தகவல் இல்லை"))
    return "\n".join(tamil_lines)

# ----------------------------
# 5) Main interactive loop (Tamil I/O friendly)
# ----------------------------
def main():
    print("\n🌳 AI மர வகை மற்றும் வயது கணிப்பு (300 மரங்கள்) - IKS (தமிழ்) இணைப்பு\n")
    print("கால்வெளி: இங்கே நீங்கள் கீழ்கண்ட விவரங்களை தமிழில் உள்ளீடு செய்யலாம்.")
    print("இலை வடிவம்: broad/oval/needle/compound/heart/lanceolate")
    print("தோல் அமைப்பு: smooth/rough/flaky/fibrous/grooved")
    print("வாழ்விடம்: tropical/dry/coastal/hill/plain/rainforest")
    print("பழம்: yes/no")
    print("உதாரணமாக: ஒவல் -> 'oval', கடுமையான தோல் -> 'rough'\n")

    # collect inputs (allow Tamil words mapped to English tokens)
    def map_tamil_to_token(value):
        v = value.strip().lower()
        mapping = {
            "பரந்த": "broad", "பரந்தது": "broad", "ஒவல்": "oval", "முள்": "needle",
            "முழுகு": "needle", "சேர்க்கை": "compound", "இதயம்": "heart", "ஊசி": "lanceolate",
            "அழுக்கு": "rough", "மென்மை": "smooth", "காசு": "flaky", "நார்": "fibrous", "துளை": "grooved",
            "வெப்பமண்டல": "tropical", "வெப்ப": "tropical", "உலர்": "dry", "கடற்கரை": "coastal",
            "மலை": "hill", "புலம்": "plain", "மழைக் காட்டில்": "rainforest", "மழைக்காடு": "rainforest",
            "ஆம்": "yes", "இல்லை": "no", "உள்ளது": "yes", "இல்லாது": "no"
        }
        return mapping.get(v, v)

    leaf_shape_in = input("இலை வடிவம் (தமிழில்/ஆங்கிலத்தில்): ")
    leaf_shape = map_tamil_to_token(leaf_shape_in)

    bark_texture_in = input("தோல் அமைப்பு (தமிழில்/ஆங்கிலத்தில்): ")
    bark_texture = map_tamil_to_token(bark_texture_in)

    habitat_in = input("வாழ்விடம் (தமிழில்/ஆங்கிலத்தில்): ")
    habitat = map_tamil_to_token(habitat_in)

    fruit_in = input("பழம் உள்ளதா? (ஆம்/இல்லை அல்லது yes/no): ")
    fruit_presence = map_tamil_to_token(fruit_in)

    try:
        avg_height = float(input("சுமார் உயரம் (மீட்டர்): ").strip())
    except Exception:
        avg_height = float(np.mean(df_species["average_height_m"]))

    try:
        leaf_size = float(input("இலை அளவு (செ.மீ): ").strip())
    except Exception:
        leaf_size = float(np.mean(df_species["leaf_size_cm"]))

    try:
        circumference = float(input("மர சுற்றளவு (செ.மீ): ").strip())
    except Exception:
        print("தவறு: சுற்றளவை (cm) சரியாக உள்ளிடவும்.")
        return

    # prepare input for classifier
    input_df = pd.DataFrame([{
        "leaf_shape": leaf_shape,
        "bark_texture": bark_texture,
        "habitat": habitat,
        "fruit_presence": fruit_presence
    }])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    input_encoded["average_height_m"] = avg_height
    input_encoded["leaf_size_cm"] = leaf_size

    # predict species
    try:
        predicted_species = clf.predict(input_encoded)[0]
    except Exception as e:
        print("வகைப்படுத்தலில் பிழை:", e)
        predicted_species = species_names[0]

    # lookup growth factor
    gf_row = df_species.loc[df_species["species"] == predicted_species, "growth_factor"]
    growth_factor = float(gf_row.values[0]) if len(gf_row) > 0 else float(np.mean(df_species["growth_factor"]))

    # compute diameter and age
    diameter_cm = circumference / math.pi
    age_years = diameter_cm * growth_factor

    # fetch IKS entry
    iks_entry = get_iks_for_species(predicted_species)

    # display results in Tamil
    print("\n" + "-"*60 + "\n")
    print(pretty_tamil_output(predicted_species, iks_entry, circumference, diameter_cm, age_years))
    print("\n" + "-"*60 + "\n")

    # Ask user if they want to add/edit IKS info for predicted species
    add = input("இந்த மரத்திற்கான IKS தமிழ் தகவலை சேரிக்கவா/தொகு (y/n)? ").strip().lower()
    if add == "y" or add == "ஆம்":
        tamil_name = input("தமிழ் பெயர் (உதா: மாமரம்): ").strip()
        eng_name = input("ஆங்கிலப் பெயர் (optional): ").strip()
        uses = input("பாரம்பரிய / மருத்துவ பயன்பாடுகள் (தமிழில்): ").strip()
        notes = input("குறிப்புகள் (தமிழில்): ").strip()
        ok = save_iks_entry(predicted_species, tamil_name, eng_name, uses, notes)
        if ok:
            print("✅ IKS தகவல் வெற்றிகரமாக சேமிக்கப்பட்டது:", IKS_DB_PATH)
        else:
            print("⚠️ IKS தகவலை சேமிக்க முடியவில்லை (அனுமதி சோதிக்கவும்).")

    # Continue loop?
    again = input("\nமீண்டும் ஒரு மரத்தை மதிப்பிட வேண்டுமா? (y/n): ").strip().lower()
    if again in ("y","yes","ஆம்"):
        main()
    else:
        print("\nநன்றி! வாழ்த்துகள் 🌿")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nநீங்கள் செயலியை நிறுத்தினீர்கள். வணக்கம்!")
        sys.exit(0)
