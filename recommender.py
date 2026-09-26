import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_recipes_without_allergen(allergen):

    conn = sqlite3.connect("food_scan_recipes.db")
    if not allergen:
        sql_query = "SELECT * FROM recipes"
    else:
        set_condition = "= 0"
        new_allergen_list = [f"{item} {set_condition}" for item in allergen]
        allergen_condition_query = " AND ".join(new_allergen_list)
        sql_query = f"SELECT * FROM recipes WHERE {allergen_condition_query}"

    data = pd.read_sql_query(sql_query, conn)
    conn.close()
    return data


def find_missing_and_penalize(recipe_str, user_ingredients, staples):
    all_available = set(user_ingredients + staples)
    recipe_words = set(recipe_str.split())
    missing = recipe_words - all_available
    count_missing = len(missing)
    penalty = count_missing * 0.05
    return pd.Series([penalty, list(missing)])


if __name__ == "__main__":
    allergen_request = ["contains_dairy", "contains_eggs"]
    data = get_recipes_without_allergen(allergen_request)

    user_ingredients = ["chicken", "salt", "pepper", "garlic"]

    # list of items that are common in cooking.
    staples = ["water", "salt", "black pepper", "oil", "olive oil",
               "vegetable oil", "sugar", "flour", "butter", "garlic", "onion"]

    result = " ".join(user_ingredients)
    list_result = []
    list_result.append(result)

    vectorizer = TfidfVectorizer()
    # creates dictionary of words from the recipe database and gives importance based on rarity.
    tfidf_matrix = vectorizer.fit_transform(data['clean_ingredients_text'])
    tfid_matrix_test = vectorizer.transform(list_result)

    # calculates the match score between user ingredients and database recipes.
    scores = cosine_similarity(tfid_matrix_test, tfidf_matrix)[0]
    data['match_score'] = scores

    # finds missing ingredients and calculates penalty for each recipe row.
    data[['penalty', 'missing_ingredients']] = data['clean_ingredients_text'].apply(
        find_missing_and_penalize, args=(user_ingredients, staples)
    )

    # subtracts penalty from the base match score.
    data['final_score'] = data['match_score'] - data['penalty']

    # converts the final score into a percentage.
    data['match_percentage'] = (data['final_score'] * 100).round(1)

    # sorts recipes by final score and prints top 5 matches.
    data_sorted = data.sort_values("final_score", ascending=False)
    print(data_sorted[['Title', 'match_percentage',
          'missing_ingredients']].head())
