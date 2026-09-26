import sqlite3
from collections import Counter

# Words to filter out so the vision model doesn't look for non-food terms
STOP_WORDS = {
    "tbsp", "tsp", "cup", "cups", "pound", "pounds", "ounce", "ounces",
    "gram", "grams", "chopped", "diced", "sliced", "minced", "fresh",
    "ground", "small", "medium", "large", "optional", "plus", "more",
    "room", "temperature", "and", "the", "for", "with", "into", "cold", "warm",
    "can", "cans", "package", "packages", "clove", "cloves", "taste", "needed"
}


def create_ranked_ingredient_list(db_path="food_scan_recipes.db", top_n=200):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT clean_ingredients_text FROM recipes")
    rows = cursor.fetchall()
    conn.close()

    counts = Counter()

    for row in rows:
        text = row[0]
        if text:
            # Using set() per recipe ensures a single recipe with 3 "peppers"
            # only increments the global pepper score by 1
            recipe_words = set(text.split())
            for word in recipe_words:
                if len(word) > 2 and word not in STOP_WORDS:
                    counts[word] += 1

    # Extract the top N most frequent ingredient strings
    top_ingredients = [word for word, frequency in counts.most_common(top_n)]
    return top_ingredients


if __name__ == "__main__":
    top_80 = create_ranked_ingredient_list()
    print("\n--- COPY AND PASTE THIS LIST INTO DETECT.PY ---\n")
    print(f"CORE_VISION_INGREDIENTS = {top_80}")
