import tempfile
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import PIL
import os
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from detect import detect_ingredients

# Setup temporary directory for uploaded image files
temp_dir = tempfile.TemporaryDirectory()

st.title("Recipe Recommender")

# Absolute path to your local Kaggle recipe images
IMAGE_FOLDER = "/Users/gill/food_scan_recommender/Food Ingredients and Recipe Dataset with Image Name Mapping"
DB_PATH = "food_scan_recipes.db"

# -------------------------------------------------------------------
# 1. SIDEBAR: Dietary Restrictions & Allergen Filters
# -------------------------------------------------------------------
st.sidebar.header("Dietary Restrictions")
selected_allergens = []
allergen_options = {
    "Dairy-Free": "contains_dairy",
    "Egg-Free": "contains_eggs",
    "Fish-Free": "contains_fish",
    "Shellfish-Free": "contains_shellfish",
    "Nut-Free": "contains_nuts",
    "Gluten-Free": "contains_gluten",
    "Soy-Free": "contains_soy",
    "Sesame-Free": "contains_sesame"
}

for label, col_name in allergen_options.items():
    if st.sidebar.checkbox(label):
        selected_allergens.append(col_name)

# Initialize ingredient storage in session state
if 'detected_ingredients' not in st.session_state:
    st.session_state['detected_ingredients'] = []

# -------------------------------------------------------------------
# 2. INPUT: Camera & File Uploaders
# -------------------------------------------------------------------
camera_uploaded = st.camera_input('ingredients_scan', key=77,
                                  help="Take a photo of your ingredients", label_visibility='visible')
file_uploaded = st.file_uploader("File Upload", type=['jpg', 'jpeg', 'png'], key=78,
                                 accept_multiple_files=False, help="Upload a photo of your ingredients", label_visibility='visible')

# Handle Camera Capture
if camera_uploaded is not None:
    uploaded_photo_name = 'File_provided.jpg'
    uploaded_photo_path = pathlib.Path(temp_dir.name) / uploaded_photo_name
    with open(uploaded_photo_path, 'wb') as output_temporary_file:
        output_temporary_file.write(camera_uploaded.read())
        if os.path.exists(uploaded_photo_path):
            st.success("Temp file was successfully created on disk")
            # Call your YOLO-World vision model from detect.py
            st.session_state['detected_ingredients'] = detect_ingredients(
                uploaded_photo_path)
            st.write(
                f"Items returned: {st.session_state['detected_ingredients']}")

# Handle File Upload
if file_uploaded is not None:
    uploaded_FILE_name = 'File_provided.jpg'
    uploaded_FILE_path = pathlib.Path(temp_dir.name) / uploaded_FILE_name
    with open(uploaded_FILE_path, 'wb') as output_temporary_file:
        output_temporary_file.write(file_uploaded.read())
        if os.path.exists(uploaded_FILE_path):
            st.success("Temp file was successfully created on disk")
            # Call your YOLO-World vision model from detect.py
            st.session_state['detected_ingredients'] = detect_ingredients(
                uploaded_FILE_path)
            st.write(
                f"Items returned: {st.session_state['detected_ingredients']}")

# -------------------------------------------------------------------
# 3. INGREDIENT DISPLAY & EDITING
# -------------------------------------------------------------------
st.write("### Your Stored Ingredients")
if st.session_state['detected_ingredients']:
    # Display editable text box with detected items
    user_ingredients_input = st.text_input(
        "Detected ingredients (you can edit or add items below):",
        value=", ".join(st.session_state['detected_ingredients'])
    )
    active_ingredients = [item.strip().lower()
                          for item in user_ingredients_input.split(",") if item.strip()]
else:
    manual_input = st.text_input(
        "No ingredients detected yet. Type manually (comma separated):", value="chicken, garlic, onion")
    active_ingredients = [item.strip().lower()
                          for item in manual_input.split(",") if item.strip()]

# -------------------------------------------------------------------
# 4. RECOMMENDATION ALGORITHM FUNCTIONS
# -------------------------------------------------------------------


def get_recipes_without_allergen(allergens):
    conn = sqlite3.connect(DB_PATH)
    if not allergens:
        sql_query = "SELECT * FROM recipes"
    else:
        conditions = [f"{item} = 0" for item in allergens]
        sql_query = f"SELECT * FROM recipes WHERE {' AND '.join(conditions)}"
    data = pd.read_sql_query(sql_query, conn)
    conn.close()
    return data


def find_missing_and_penalize(recipe_str, user_ingredients, staples):
    all_available = set(user_ingredients + staples)
    recipe_words = set(str(recipe_str).split())
    missing = recipe_words - all_available
    penalty = len(missing) * 0.05
    return pd.Series([penalty, list(missing)])


# -------------------------------------------------------------------
# 5. RECOMMENDATION EXECUTION & DISPLAY GRID
# -------------------------------------------------------------------
if st.button("Recommend Recipes"):
    data = get_recipes_without_allergen(selected_allergens)

    if data.empty:
        st.warning("No recipes found matching your allergen criteria.")
    else:
        # Common kitchen staples ignored in penalty scoring
        staples = ["water", "salt", "black pepper", "oil", "olive oil",
                   "vegetable oil", "sugar", "flour", "butter", "garlic", "onion"]

        # TF-IDF Vector Space Model
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(
            data['clean_ingredients_text'].fillna(""))
        user_vector = vectorizer.transform([" ".join(active_ingredients)])

        # Cosine Similarity Scoring
        scores = cosine_similarity(user_vector, tfidf_matrix)[0]
        data['match_score'] = scores

        # Calculate Penalties
        data[['penalty', 'missing_ingredients']] = data['clean_ingredients_text'].apply(
            find_missing_and_penalize, args=(active_ingredients, staples)
        )
        data['final_score'] = data['match_score'] - data['penalty']
        data['match_percentage'] = (
            data['final_score'] * 100).clip(lower=0, upper=100).round(1)

        # Sort recipes by top final score
        top_recipes = data.sort_values("final_score", ascending=False).head(6)

        st.subheader("Recommended Recipes")

        # Render cards in a 2-column layout
        cols = st.columns(2)
        for idx, (_, row) in enumerate(top_recipes.iterrows()):
            with cols[idx % 2]:
                with st.container(border=True):
                    # Fetch image from local Kaggle directory
                    image_filename = str(row.get("Image_Name", "")) + ".jpg"
                    img_path = os.path.join(IMAGE_FOLDER, image_filename)

                    if os.path.exists(img_path):
                        st.image(img_path, use_column_width=True)
                    else:
                        st.write("🖼️ *No image available*")

                    st.write(f"### {row['Title']}")
                    st.write(
                        f"**Match Percentage:** {row['match_percentage']}%")

                    missing = row['missing_ingredients']
                    if missing:
                        st.write("**Missing ingredients:**",
                                 ", ".join(missing[:5]))
                    else:
                        st.success("You have all ingredients!")

                    with st.expander("Show Details & Instructions"):
                        st.write("**Full Ingredients:**")
                        st.write(row.get("Ingredients", "N/A"))
                        st.write("**Instructions:**")
                        st.write(row.get("Instructions", "N/A"))
