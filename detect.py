from ultralytics import YOLOWorld
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Category mappings for Pass 1 -> Pass 2 filtering
INGREDIENT_GROUPS = {
    "vegetable": [
        "onion", "red onion", "garlic", "shallot", "scallion", "chives",
        "tomato", "cherry tomato", "bell pepper", "jalapeno", "chile pepper",
        "carrot", "celery", "potato", "sweet potato", "zucchini", "eggplant",
        "cucumber", "broccoli", "cauliflower", "cabbage", "spinach",
        "kale", "lettuce", "arugula", "asparagus", "green beans", "peas",
        "mushroom", "corn", "radish", "squash", "pumpkin", "bok choy"
    ],
    "fruit": [
        "lemon", "lime", "orange", "apple", "banana", "avocado", "strawberry",
        "blueberry", "raspberry", "mango", "pineapple", "peach",
        "plum", "pear", "cherry", "grape", "watermelon", "cantaloupe", "kiwi",
        "fig", "pomegranate", "coconut", "cranberry", "raisin"
    ],
    "herb": [
        "parsley", "cilantro", "basil", "thyme", "rosemary", "dill", "mint",
        "oregano", "sage", "bay leaf", "ginger", "lemongrass"
    ],
    "meat": [
        "chicken", "chicken breast", "chicken thigh", "turkey",
        "beef", "ground beef", "steak", "pork", "pork chop", "bacon",
        "sausage", "ham", "lamb"
    ],
    "seafood": [
        "salmon", "tuna", "cod", "tilapia", "shrimp", "prawn",
        "crab", "lobster", "scallop", "clam", "mussel", "squid"
    ],
    "dairy": [
        "egg", "milk", "butter", "heavy cream", "sour cream", "yogurt",
        "cheese", "cheddar cheese", "mozzarella cheese", "parmesan cheese",
        "feta cheese", "cream cheese", "tofu"
    ],
    "grain": [
        "bread", "tortilla", "pasta", "noodle", "spaghetti",
        "rice", "quinoa", "oats", "flour"
    ]
}

# Cache pipeline globally so it stays in RAM
_MODLE_CACHE = None


def get_model():
    # tells python to update the main _MODEL_CACHE sitting outside.
    global _MODLE_CACHE
    if _MODLE_CACHE is None:
        _MODLE_CACHE = YOLOWorld("yolov8s-worldv2.pt")
    return _MODLE_CACHE


def detect_ingredients(image_path):
    model = get_model()

    # Combine all ingredient groups into a single candidate list
    all_candidates = []
    for items in INGREDIENT_GROUPS.values():
        all_candidates.extend(items)
        # remove duplicate words and convert back from a set into a regular list.
    all_candidates = list(set(all_candidates))

    # Tell YOLO-World what labels to search for
    model.set_classes(all_candidates)

    # Predict ingredients directly from the image path (single pass!)
    results = model.predict(image_path, conf=0.10, verbose=False)

    # Extract detected label names
    detected_items = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            detected_items.append(label)

    return list(set(detected_items))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the image path as a command line argument")
    else:
        image_path = sys.argv[1]
        detected = detect_ingredients(image_path)
        print("Detected Ingredients:", detected)
