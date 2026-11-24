"""
FEATURES: turns the raw data into the inputs a model is allowed to see.

A "feature" is just one input number, like "carbs eaten in the last hour".
The Pattern Learner and the Hybrid both use this file, so they get identical inputs.

THE GOLDEN RULE: a feature may only use information from NOW OR EARLIER.
If the future sneaks in, the model looks brilliant in testing and is useless
in real life. This mistake is called "data leakage".

You don't run this file directly. Other scripts use:
    import features
"""

import re
import pandas as pd
import common

MEALS_CSV = common.PROJECT_FOLDER / "data" / "t2dm_meals.csv"
PATIENTS_CSV = common.PROJECT_FOLDER / "data" / "t2dm_patients.csv"

READINGS_PER_HOUR = 60 // common.MINUTES_PER_READING

# Long-acting insulin works slowly over about a day. All other insulin in this
# dataset (fast or mixed) works within a few hours, so we count them separately.
LONG_ACTING_WORDS = ("degludec", "detemir", "glargine", "glarigine")

# Which columns of the patient table we use, and their short names.
PATIENT_COLUMNS = {
    "Gender (Female=1, Male=2)": "gender",
    "Age (years)": "age",
    "BMI (kg/m2)": "bmi",
    "Duration of diabetes (years)": "years_with_diabetes",
    "HbA1c (mmol/mol)": "hba1c",
    "Fasting Plasma Glucose (mg/dl)": "lab_fasting_glucose",
    "2-hour Postprandial Plasma Glucose (mg/dl)": "lab_2h_glucose",
    "Fasting C-peptide (nmol/L)": "lab_fasting_cpeptide",
    "2-hour Postprandial C-peptide (nmol/L)": "lab_2h_cpeptide",
    "Fasting Insulin (pmol/L)": "lab_fasting_insulin",
    "2-hour Postprandial insulin (pmol/L)": "lab_2h_insulin",
}


def insulin_units(text):
    """Reads text like "insulin glargine, 14 IU, Humulin 70/30, 12 IU".
    Returns two numbers: (fast or mixed units, long-acting units)."""
    fast = 0.0
    long_acting = 0.0
    if pd.isna(text):
        return fast, long_acting

    # Splitting at "IU" leaves pieces like "insulin glargine, 14 ", so the
    # dose is the LAST number in each piece and the name is in the same piece.
    # \b means "whole word only", so the "iu" inside "sodium" doesn't count.
    pieces = re.split(r"\biu\b", str(text).lower())
    for piece in pieces[:-1]:  # [:-1] skips the leftover piece after the final "IU"
        numbers = re.findall(r"\d+(?:\.\d+)?", piece)
        if not numbers:
            continue  # a stray label with no dose in it
        dose = float(numbers[-1])  # [-1] means "the last item"
        if any(word in piece for word in LONG_ACTING_WORDS):
            long_acting += dose
        else:
            fast += dose
    return fast, long_acting


def rolling_sum(values, recording_ids, hours):
    """For each row, adds up the values from the last `hours` hours (including now)."""
    rows = hours * READINGS_PER_HOUR
    return values.groupby(recording_ids).transform(lambda s: s.rolling(rows, min_periods=1).sum())


def build_feature_table():
    """Returns (table, list of feature column names)."""
    table = common.add_future_sugar(common.load_readings())
    table = table.merge(pd.read_csv(common.SPLIT_CSV), on="patient_id")
    table = table.sort_values(["recording_id", "time"]).reset_index(drop=True)
    recording = table["recording_id"]  # used constantly below, so a short name helps
    feature_names = []

    # ---- 1. The last 2 hours of sugar ----
    for step in range(common.HISTORY_READINGS):  # range(8) gives 0, 1, ... 7
        minutes_ago = step * common.MINUTES_PER_READING
        name = f"sugar_{minutes_ago}_min_ago"
        table[name] = table.groupby("recording_id")["cgm"].shift(step)
        feature_names.append(name)

    # ---- 2. How fast is sugar moving? ----
    # These are simple subtractions, but they make the trend easy for the model to find.
    for minutes in [15, 30, 60]:
        name = f"change_last_{minutes}_min"
        table[name] = table["sugar_0_min_ago"] - table[f"sugar_{minutes}_min_ago"]
        feature_names.append(name)

    # ---- 3. Food ----
    meals = pd.read_csv(MEALS_CSV, parse_dates=["time"])[["recording_id", "time", "carbs_g"]]
    table = table.merge(meals, on=["recording_id", "time"], how="left")
    recording = table["recording_id"]

    known_carbs = table["carbs_g"].fillna(0)
    # A meal with no details: we know they ate, but not how much.
    unknown_meal = (table["meal"].notna() & table["carbs_g"].isna()).astype(float)

    for hours in [1, 2, 4]:
        name = f"carbs_last_{hours}h"
        table[name] = rolling_sum(known_carbs, recording, hours)
        feature_names.append(name)
    table["unknown_meals_last_2h"] = rolling_sum(unknown_meal, recording, 2)
    feature_names.append("unknown_meals_last_2h")

    # Time since the last meal. ffill ("forward fill") copies the last known
    # meal time down the column until the next meal appears.
    last_meal_time = table["time"].where(table["meal"].notna()).groupby(recording).ffill()
    table["minutes_since_meal"] = (table["time"] - last_meal_time).dt.total_seconds() / 60
    feature_names.append("minutes_since_meal")

    # ---- 4. Insulin ----
    pen = table["insulin_pen"].apply(insulin_units).tolist()
    drip = table["insulin_drip"].apply(insulin_units).tolist()
    pen_fast = pd.Series([pair[0] for pair in pen], index=table.index)
    pen_long = pd.Series([pair[1] for pair in pen], index=table.index)
    drip_fast = pd.Series([pair[0] for pair in drip], index=table.index)

    # errors="coerce" turns anything that isn't a number into empty.
    pump_bolus = pd.to_numeric(table["pump_bolus"], errors="coerce").fillna(0)
    fast_insulin = pen_fast + drip_fast + pump_bolus
    # Not a model input. The Equations (step 6) need the raw dose at each moment.
    table["fast_insulin_now"] = fast_insulin

    for hours in [1, 2, 4]:
        name = f"fast_insulin_last_{hours}h"
        table[name] = rolling_sum(fast_insulin, recording, hours)
        feature_names.append(name)
    table["long_insulin_last_24h"] = rolling_sum(pen_long, recording, 24)
    feature_names.append("long_insulin_last_24h")

    # The pump's background rate is only written down when it changes,
    # so each value stays in force until the next one (ffill again).
    # "temporarily suspend insulin delivery" means the rate is 0.
    basal = table["pump_basal"].astype("string").str.replace(r"(?i).*suspend.*", "0", regex=True)
    basal = pd.to_numeric(basal, errors="coerce")
    table["pump_basal_rate"] = basal.groupby(recording).ffill().fillna(0)
    feature_names.append("pump_basal_rate")

    # ---- 5. Tablets ----
    took_tablets = table["tablets"].notna().astype(float)
    table["tablet_doses_last_6h"] = rolling_sum(took_tablets, recording, 6)
    feature_names.append("tablet_doses_last_6h")

    # ---- 6. Time of day (13.5 means 1:30 pm) ----
    table["hour_of_day"] = table["time"].dt.hour + table["time"].dt.minute / 60
    feature_names.append("hour_of_day")

    # ---- 7. Facts about the patient ----
    patients = pd.read_csv(PATIENTS_CSV)[["recording_id", *PATIENT_COLUMNS]].rename(columns=PATIENT_COLUMNS)
    for column in PATIENT_COLUMNS.values():
        patients[column] = pd.to_numeric(patients[column], errors="coerce")
    table = table.merge(patients, on="recording_id", how="left")
    feature_names.extend(PATIENT_COLUMNS.values())

    return table, feature_names
