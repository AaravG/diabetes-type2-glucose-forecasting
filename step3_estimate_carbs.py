"""
STEP 3: Turn each meal description into an estimate of grams of carbohydrate.

WHY?
    The "tap" in our bathtub equation needs a NUMBER: how many grams of carbs
    were eaten. The dataset only gives us words, like:
        "Rice 150 g; Vegetable 100 g; Beef 15 g"
    So for each food we do:
        carbs in that food = its weight x (carbs per 100 g) / 100
    and then add the foods up.
        Rice:      150 g x 26 / 100 = 39 g
        Vegetable: 100 g x  4 / 100 =  4 g
        Beef:       15 g x  0 / 100 =  0 g
        Meal total                  = 43 g

IMPORTANT: THESE ARE ESTIMATES
    There are over 1,800 different food names, so we match on keywords, and the
    carbs-per-100g numbers are approximate. We assume weights are for food as
    eaten (cooked rice, not dry rice). That's why the end of this script checks
    whether the estimates make sense against the real sugar readings.

HOW TO RUN:
    First run step1_build_csv.py (once). Then:   python step3_estimate_carbs.py

WHAT IT CREATES:
    data/t2dm_meals.csv   -> one row per meal, with its estimated carbs
"""

import re                 # "re" = regular expressions, a mini-language for finding patterns in text
from pathlib import Path
import pandas as pd

PROJECT_FOLDER = Path(__file__).parent
READINGS_CSV = PROJECT_FOLDER / "data" / "t2dm_readings.csv"
MEALS_CSV = PROJECT_FOLDER / "data" / "t2dm_meals.csv"

# Some foods have no weight written down (mostly side vegetables like "Carrot").
# We assume a small typical portion for those.
DEFAULT_PORTION_GRAMS = 50

# If we can't recognise a food at all, we assume this many carbs per 100 g.
UNKNOWN_FOOD_CARBS = 10

# Cereal and oats are a special case. 40 g means dry powder (very high in carbs),
# but "Cereal 400 g" or "Cereal 150 ml" can only mean it's already mixed with
# water or milk (mostly liquid). So heavy or liquid portions get a lower number.
DRY_MIX_WORDS = ("cereal", "oat", "meal replacement")
PREPARED_CEREAL_CARBS = 12
HEAVY_CEREAL_GRAMS = 100

# ---------------------------------------------------------------------------
# THE FOOD TABLE: (keyword, grams of carbs per 100 g of food)
# ---------------------------------------------------------------------------
# ORDER MATTERS! We go down the list and stop at the FIRST keyword found.
# So specific names must come before general ones:
#   "rice cake" must come before "rice", or a rice cake would count as rice.
#   "sweet potato leaves" must come before "sweet potato".
# Starchy foods come before meat and vegetables because they matter most:
#   "steamed pork bun" should count as a bun, not as pork.
FOOD_TABLE = [
    # --- Not really food ---
    ("fasting", 0),

    # --- Specific names that would otherwise match the wrong thing ---
    ("rice in soup", 12), ("rice in salted soup", 12), ("rice wine", 5),
    ("rice cake", 40), ("rice noodle", 25), ("rice dumpling", 40),
    ("sweet potato leaves", 3), ("sea cucumber", 1), ("chicken nugget", 15),
    ("bean sprout", 4), ("soybean milk", 2), ("coconut milk", 6),
    ("ice cream", 24), ("corn bread", 47), ("green bean", 7),
    ("green soybean", 10), ("lotus root", 16), ("lotus nut", 60),
    ("red date", 60), ("red bean", 20), ("mung bean", 20), ("broad bean", 15),
    ("bean curd roll", 5), ("tofu skin", 5), ("egg roll", 5), ("milk powder", 40),
    ("protein powder", 10), ("oil immersed tuna", 0), ("luncheon meat", 5),
    ("chestnut", 45), ("ginger", 10), ("porridge", 10), ("congee", 10),

    # --- Spelling mistakes that really appear in the dataset ---
    ("coarase grain", 30), ("coarce grain", 30), ("corase grain", 30),
    ("wontonwonton", 30), ("vegtable", 4), ("raddish", 4), ("beaf", 0), ("prok", 0),

    # --- Sugars and sweets ---
    ("sugar", 100), ("honey", 80), ("candy", 90), ("chocolate", 55),

    # --- Cereals and dry mixes (eaten as a powder weight, so high in carbs) ---
    ("cereal", 70), ("meal replacement", 60), ("oatmeal", 60), ("oat", 60),
    ("lotus root starch", 85), ("biscuit", 70), ("cracker", 70), ("cookie", 65),

    # --- Breads, buns and cakes ---
    ("steamed bread", 47), ("steamed bun", 47), ("bun", 45), ("bread", 48),
    ("toast", 48), ("pancake", 40), ("mooncake", 55), ("cake", 50), ("pastry", 50),
    ("pie", 40), ("roll", 40), ("doughnut", 45), ("fried dough", 50), ("croissant", 45),
    ("hamburger", 30), ("burger", 30), ("sandwich", 30), ("pizza", 30), ("pretzel", 75),
    ("wafer", 65), ("snack", 50), ("chive pocket", 30), ("staple food", 26),

    # --- Noodles and dumplings ---
    ("noodle", 25), ("pasta", 28), ("spaghetti", 28), ("vermicelli", 25), ("wonton", 30),
    ("dumpling", 28),

    # --- Rice ---
    ("rice", 26),

    # --- Other starchy foods ---
    ("coarse grain", 30), ("millet", 23), ("sorghum", 25), ("buckwheat", 20), ("corn", 20),
    ("sweet potato", 20), ("potato", 17), ("taro", 20), ("yam", 14),
    ("pumpkin", 6), ("gluten", 10),

    # --- Fruit ---
    ("apple", 14), ("pear", 14), ("peach", 11), ("orange", 12),
    ("grapefruit", 9), ("kiwi", 15), ("watermelon", 7), ("banana", 23),
    ("grape", 16), ("plum", 11), ("cherry", 14), ("mango", 15),
    ("strawberry", 8), ("pineapple", 13), ("persimmon", 17), ("melon", 8),
    ("lychee", 17), ("blueberry", 14), ("dekopon", 12), ("pitaya", 11), ("fruit", 13),

    # --- Nuts and seeds ---
    ("peanut", 16), ("walnut", 14), ("sesame", 20), ("nut", 20), ("seed", 20),

    # --- Drinks and dairy ---
    ("soymilk", 2), ("milk", 5), ("yogurt", 12), ("cheese", 3), ("beer", 4), ("wine", 3),
    ("liquor", 0), ("latte", 7), ("coffee", 0), ("tea", 0), ("juice", 11), ("coke", 11),
    ("water", 0),

    # --- Soups (mostly water) ---
    ("soup", 4), ("broth", 1), ("hot pot", 5), ("hotpot", 5),

    # --- Protein foods (few carbs) ---
    ("egg", 1), ("tofu", 3), ("bean curd", 3), ("meatball", 8), ("sausage", 5),
    ("ham", 3), ("bacon", 1), ("pork", 0), ("beef", 0), ("mutton", 0),
    ("lamb", 0), ("chicken", 0), ("duck", 0), ("goose", 0), ("pigeon", 0),
    ("pig", 0), ("fish", 0), ("hairtail", 0), ("croaker", 0), ("shrimp", 0),
    ("crab", 0), ("clam", 2), ("squid", 2), ("meat", 0), ("steak", 0), ("chop", 0),
    ("crayfish", 0), ("crawfish", 0), ("pomfret", 0), ("perch", 0), ("bream", 0),
    ("eel", 0), ("cod", 0), ("tuna", 0), ("seafood", 2), ("abalone", 5), ("oyster", 5),
    ("bullfrog", 0),

    # --- Vegetables (a few carbs). Everything else green goes here ---
    ("carrot", 8), ("onion", 9), ("tomato", 4), ("eggplant", 6), ("radish", 4),
    ("mushroom", 4), ("fungus", 6), ("kelp", 6), ("cabbage", 4), ("lettuce", 3),
    ("spinach", 3), ("cucumber", 3), ("cauliflower", 5), ("broccoli", 7),
    ("celery", 3), ("gourd", 4), ("pepper", 5), ("amaranth", 4), ("daisy", 3),
    ("bamboo", 4), ("leek", 5), ("pickle", 5), ("sauerkraut", 3), ("asparagus", 4),
    ("zucchini", 3), ("okra", 7), ("garlic", 15), ("sprout", 4), ("sword bean", 7),
    ("long bean", 7), ("cowpea", 7), ("medicago polymorpha", 4), ("pleurotus eryngii", 4),
    ("bean", 15), ("vegetable", 4),
]

# Turn each keyword into a "whole word" search pattern. Why whole words?
# So that "egg" doesn't match "eggplant", and "pea" doesn't match "peanut".
#   \b      means "the edge of a word"
#   (s|es)? means "optionally followed by s or es", so "bun" also finds "buns"
# This is a "list comprehension": a short way to build a new list from an old one.
SEARCH_PATTERNS = [(re.compile(rf"\b{keyword}(s|es)?\b"), carbs) for keyword, carbs in FOOD_TABLE]

# This pattern splits "Rice 150 g" into the name ("Rice") and the weight ("150").
# (.*?) captures the name, (\d+(?:\.\d+)?) captures a number like 150 or 12.5.
# (g|ml) captures the unit.
ITEM_PATTERN = re.compile(r"^(.*?)\s*(\d+(?:\.\d+)?)\s*(g|ml)\s*$", re.IGNORECASE)


def carbs_per_100g(food_name):
    """Look a food up in the table. Returns None if nothing matches."""
    for pattern, carbs in SEARCH_PATTERNS:
        if pattern.search(food_name):
            return carbs
    return None


def estimate_meal_carbs(meal_text):
    """Takes text like "Rice 150 g; Egg 50 g" and returns two numbers:
    the estimated carbs in grams, and the share of foods we recognised (0 to 1)."""
    # Split the text into separate foods at ";" or ",".
    items = [piece.strip().lower() for piece in re.split(r"[;,]", meal_text) if piece.strip()]

    # Some meals were logged with no details at all.
    # float("nan") means "not a number": we know a meal happened, but not its size.
    items = [item for item in items if item != "data not available"]
    if not items:
        return float("nan"), float("nan")

    total_carbs = 0
    recognised = 0
    for item in items:
        match = ITEM_PATTERN.match(item)
        if match:
            food_name = match.group(1).strip()
            grams = float(match.group(2))
            unit = match.group(3)
        else:
            food_name = item
            grams = DEFAULT_PORTION_GRAMS
            unit = "g"

        # "Sugar free cake" is still cake, so remove those words before looking it up.
        food_name = food_name.replace("sugar free", "").replace("sugar-free", "").strip()

        carbs = carbs_per_100g(food_name)
        if carbs is None:
            carbs = UNKNOWN_FOOD_CARBS
        else:
            recognised += 1

        # The cereal special case explained at the top of this file.
        # any(...) is True if at least one of the words is found in the name.
        is_dry_mix = any(word in food_name for word in DRY_MIX_WORDS)
        if is_dry_mix and carbs >= 60 and (unit == "ml" or grams >= HEAVY_CEREAL_GRAMS):
            carbs = PREPARED_CEREAL_CARBS

        total_carbs += grams * carbs / 100

    return total_carbs, recognised / len(items)


def main():
    readings = pd.read_csv(READINGS_CSV, parse_dates=["time"], low_memory=False)

    # Keep only the rows where a meal was logged. .copy() makes an independent
    # table, so that adding columns below doesn't touch the original.
    meals = readings.loc[readings["meal"].notna(), ["recording_id", "patient_id", "time", "meal", "cgm"]].copy()

    # .apply runs our function once per meal. It gives back pairs of numbers,
    # and .tolist() lets us put those pairs into two new columns.
    results = meals["meal"].apply(estimate_meal_carbs).tolist()
    meals[["carbs_g", "share_recognised"]] = pd.DataFrame(results, index=meals.index)

    meals.to_csv(MEALS_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")

    # ---- Report ----
    print(f"Saved {len(meals):,} meals to {MEALS_CSV.name}")
    print(f"Meals with no details (carbs unknown): {meals['carbs_g'].isna().sum()}")
    print(f"Average share of foods recognised (meals with details): {meals['share_recognised'].mean() * 100:.0f}%")
    print()
    print("Estimated carbs per meal (grams):")
    print(meals["carbs_g"].describe().round(1))

    # ---- Sanity check: do bigger meals really raise sugar more? ----
    # If our estimates are any good, the answer should be yes.
    # For each meal: rise = highest sugar in the next 2 hours - sugar at the meal.
    # groupby splits the readings into one small table per recording.
    by_recording = {name: group for name, group in readings.groupby("recording_id")}

    rises = []
    for _, meal in meals.iterrows():  # iterrows gives us the table one row at a time
        recording = by_recording[meal["recording_id"]]
        after = recording[(recording["time"] > meal["time"]) &
                          (recording["time"] <= meal["time"] + pd.Timedelta(hours=2))]
        if len(after) == 0 or pd.isna(meal["cgm"]):
            rises.append(float("nan"))
        else:
            rises.append(after["cgm"].max() - meal["cgm"])
    meals["sugar_rise"] = rises

    # pd.cut sorts each meal into a size group.
    groups = pd.cut(meals["carbs_g"], bins=[-1, 20, 40, 60, 80, 1000],
                    labels=["0-20 g", "20-40 g", "40-60 g", "60-80 g", "80+ g"])
    print()
    print("Average sugar rise in the 2 hours after a meal, by meal size:")
    print(meals.groupby(groups, observed=True)["sugar_rise"].agg(["count", "mean"]).round(1))
    correlation = meals["carbs_g"].corr(meals["sugar_rise"])
    print(f"\nCorrelation between carbs and sugar rise: {correlation:.2f}")
    print("(0 = no link at all, 1 = perfect link)")


if __name__ == "__main__":
    main()
