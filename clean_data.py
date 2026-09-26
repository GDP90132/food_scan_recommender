import sqlite3
import re
import pandas as pd
import torch
from PIL import Image

df = pd.read_csv(
    '/Users/gill/food_scan_recommender/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')


def clean_ingredients_text(text):
    clean_1 = re.sub(r"\d+", "", text)
    clean_2 = re.sub(r"[,.!]", "", clean_1)
    clean_3 = re.sub(r"[\[\]\(\)]", "", clean_2)
    clean_4 = re.sub(r"[½¾¼⅓⅔⅛⅜⅝⅞–—]", "", clean_3)
    clean_5 = re.sub(r"\(.*?\)", "", clean_4)
    clean_6 = re.sub(
        r"\b(tbsp|tsp|pound|pounds|lb|lbs|oz|ounce|ounces|cup|cups|loaf|pinch|clove|cloves|slice|slices|gram|g|kg|ml|l)\b",
        "",
        clean_5,
        flags=re.IGNORECASE,
    )
    clean_7 = re.sub(
        r"\b(chopped|finely|diced|sliced|minced|peeled|divided|melted|ground|freshly|fresh|dry|dried|kosher|small|medium|large|plus|more|total|room|temperature|optional|quality|good)\b",
        "",
        clean_6,
        flags=re.IGNORECASE,
    )
    # remove spaces between words, and remove leading snd trailing spaces, and convert to lower case.
    clean_final = re.sub(r"\s+", " ", clean_7).strip().lower()
    return clean_final


allergen_rules = {
    "contains_dairy": [
        "milk", "cheese", "butter", "cream", "yogurt",
        "ghee", "whey", "casein", "margarine"
    ],
    "contains_eggs": [
        "egg", "eggs", "yolk", "eggnog", "mayonnaise", "mayo"
    ],
    "contains_fish": [
        "fish", "salmon", "tuna", "cod", "tilapia",
        "trout", "halibut", "anchovy", "sardine", "haddock", "mahi"
    ],
    "contains_shellfish": [
        "shrimp", "crab", "lobster", "prawn", "clam",
        "mussel", "oyster", "scallop", "squid"
    ],
    "contains_nuts": [
        "almond", "walnut", "pecan", "cashew", "pistachio",
        "macadamia", "hazelnut", "chestnut", "peanut", "peanuts"
    ],
    "contains_gluten": [
        "wheat", "flour", "barley", "rye", "semolina",
        "farro", "bulgur", "spelt", "bread"
    ],
    "contains_soy": [
        "soy", "soya", "soybean", "tofu", "tempeh", "edamame", "tamari"
    ],
    "contains_sesame": [
        "sesame", "tahini"
    ]}


def allergen_find(text):
    results = {}

    for catagory_name, trigger_word in allergen_rules.items():
        has_allergen = any(word in text for word in trigger_word)
        results[catagory_name] = 1 if has_allergen else 0
    return results


df[list(allergen_rules.keys())] = df["Cleaned_Ingredients"].apply(
    allergen_find).apply(pd.Series)
df['clean_ingredients_text'] = df["Cleaned_Ingredients"].apply(
    clean_ingredients_text)

csv_file = "/Users/gill/food_scan_recommender/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

# name of database file is food_scan_recipes.db
conn = sqlite3.connect("food_scan_recipes.db")
# name of the table is recipes, its inside the database file food_scan_recipes.db
df.to_sql("recipes", conn, if_exists='replace', index=False)

df = pd.read_sql_query("SELECT * FROM recipes", conn)
print(df.head())
conn.close()
