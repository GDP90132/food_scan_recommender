cat << 'EOF' > README.md
# Food Scan & Recipe Recommender 🥑📸

**An AI-powered computer vision and vector search engine that scans raw food ingredients and recommends matching recipes in real time.**

This project uses a zero-shot vision model (YOLO-World) to detect raw ingredients directly from uploaded or camera-captured images, normalizes the text, and computes recipe matches using TF-IDF vector space modeling and cosine similarity.

---

## Overview

Deciding what to cook based on leftover ingredients often leads to food waste or endless searching. This application automates ingredient inventory by taking a picture of available food items, extracting detected ingredients using state-of-the-art vision models, and ranking recipe matches mathematically based on inventory overlap, staple item logic, and dietary restrictions.

- **Zero-Shot Ingredient Detection** — Uses YOLO-World (`yolov8s-worldv2`) to detect open-vocabulary ingredient items in a single forward pass without re-training.
- **Categorical Vocabulary Mapping** — Dynamically builds class candidate lists across major food categories (vegetables, fruits, herbs, meats, seafood, dairy, grains) to optimize detection precision.
- **SQL-Based Recipe Filtering** — Queries an internal SQLite database (`recipes.db`) to instantly enforce dietary and allergen restrictions (e.g., Nut-Free, Gluten-Free, Vegan)[cite: 2].
- **Vector Space Matching** — Transforms detected ingredients and database recipes into sparse vectors via `TfidfVectorizer` and ranks matches using cosine similarity.
- **Interactive Streamlit Web App** — Simple web interface supporting camera inputs, file uploads, manual inventory editing, and match confidence badges[cite: 2].

---

## Technical Architecture
--- [ Image Input ] ──> ( YOLO-World Zero-Shot Model ) ──> [ Detected Ingredients ]
│
▼
[ User Restrictions ] ──> ( SQLite Database Filter ) ──> [ Candidate Recipes ]
│
▼
( TF-IDF + Cosine Similarity )
│
▼
[ Ranked Recipe Dashboard ]

## Features

### 1. Computer Vision Pipeline (`detect.py`)
- Powered by **Ultralytics YOLO-World** (`yolov8s-worldv2.pt`).
- Uses global model caching (`_MODEL_CACHE`) to retain the vision backbone in memory for sub-second inference speed.
- Dynamic vocabulary set loading (`model.set_classes(...)`) spanning custom category maps.

### 2. Recommendation & Scoring Engine (`recommender.py`)
- **Text Normalization**: Regex-based cleaning and string standardization (stripping units, measurements, and formatting)[cite: 1, 2].
- **Vector Search**: Computes sparse matrix representations using scikit-learn's `TfidfVectorizer` and scores ingredient overlap using `cosine_similarity`[cite: 1, 2].
- **Staple Logic & Penalty System**: Applies dynamic penalty deductions for missing non-staple ingredients while ignoring common pantry items (salt, water, oil)[cite: 2].

### 3. Application UI (`app.py`)
- Multi-page Streamlit application supporting image drag-and-drop and smartphone camera capture (`st.camera_input`)[cite: 2].
- State persistence via `st.session_state` to allow adding or removing detected ingredients manually without triggering full page re-renders[cite: 2].

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **App / UI** | Streamlit[cite: 1, 2] |
| **Computer Vision** | Ultralytics YOLO-World (`yolov8s-worldv2`), PyTorch, PIL[cite: 1, 2] |
| **Natural Language / Vector Math** | scikit-learn (`TfidfVectorizer`, `cosine_similarity`), Regex[cite: 1, 2] |
| **Database** | SQLite (`sqlite3`)[cite: 1, 2] |
| **Data Processing** | Pandas, NumPy[cite: 1] |
| **Environment Management** | Python `venv`[cite: 2] |

---

## Getting Started

### Prerequisites

- **Python** >= 3.10
- **pip**
- **Git**

### Installation

**1. Clone the repository**

```bash
git clone [https://github.com/GDP90132/food_scan_recommender.git](https://github.com/GDP90132/food_scan_recommender.git)
cd food_scan_recommender
