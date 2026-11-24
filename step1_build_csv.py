"""
STEP 1: Combine all the patient Excel files into ONE clean CSV table.

WHY?
    The Shanghai dataset comes as 109 separate Excel files (one per recording),
    and the column names are slightly different from file to file
    (for example "CGM (mg / dl)" in one file and "CGM " in another).
    Working with 109 messy files is painful, so we do the cleaning ONCE here
    and save the result as a single CSV file that every later step can use.

WHAT IS A CSV?
    "Comma-Separated Values". It's just a plain text file where each line is a
    row of a table and the values are separated by commas, like:
        time,patient_id,cgm
        2020-12-30 13:54:00,2000,138.6
    Excel can open it, pandas can read it, and it's easy to share.

HOW TO RUN:
    Open a terminal in this folder and type:   python step1_build_csv.py

WHAT IT CREATES:
    data/t2dm_readings.csv   -> every sugar reading, meal and insulin dose, all patients
    data/t2dm_patients.csv   -> one row per recording: age, BMI, lab tests, etc.
"""

# ---------------------------------------------------------------------------
# IMPORTS: bring in the tools (libraries) we need
# ---------------------------------------------------------------------------
from pathlib import Path  # Path builds file/folder paths that work on any computer
import pandas as pd       # pandas is THE Python library for working with tables

# ---------------------------------------------------------------------------
# 1. WHERE ARE THE FILES?
# ---------------------------------------------------------------------------
# __file__ is the path of THIS script. .parent is the folder it sits in.
# So PROJECT_FOLDER is always "the folder this script is in",
# no matter where you run it from.
PROJECT_FOLDER = Path(__file__).parent

# The "/" operator on Path objects joins folder names together.
EXTRACTED_FOLDER = PROJECT_FOLDER / "data" / "shanghai_raw" / "extracted"
RAW_FOLDER = EXTRACTED_FOLDER / "Shanghai_T2DM"                     # the 109 patient files
SUMMARY_FILE = EXTRACTED_FOLDER / "Shanghai_T2DM_Summary.xlsx"      # one row per patient (sits one folder up)
OUTPUT_FOLDER = PROJECT_FOLDER / "data"


# ---------------------------------------------------------------------------
# 2. A FUNCTION TO GIVE EVERY COLUMN A SHORT, CONSISTENT NAME
# ---------------------------------------------------------------------------
# A "function" is a reusable block of code. You give it an input, it gives
# back an output. This one takes a messy column name and returns a tidy one.
def tidy_column_name(original_name):
    name = str(original_name).strip()  # .strip() removes spaces at the start/end

    if name == "Date":
        return "time"            # when the reading was taken
    if name.startswith("CGM"):
        return "cgm"             # sensor sugar reading, mg/dL, every 15 min  <-- the main thing we predict
    if name.startswith("CBG"):
        return "cbg"             # finger-prick glucometer reading, mg/dL
    if name.startswith("Blood Ketone"):
        return "ketone"          # only measured in emergencies (mostly empty)
    if name == "Dietary intake":
        return "meal"            # what they ate, in English, e.g. "Rice 150 g"
    if name.startswith("Insulin dose - s.c"):
        return "insulin_pen"     # insulin injected with a pen, e.g. "Gansulin 40R, 8 IU"
    if name.startswith("Insulin dose - i.v"):
        return "insulin_drip"    # insulin through a drip (hospital only)
    if name.startswith("Non-insulin"):
        return "tablets"         # diabetes tablets, e.g. metformin
    if "bolus" in name:
        return "pump_bolus"      # insulin pump: dose given before a meal
    if "basal" in name or "基础量" in name:
        return "pump_basal"      # insulin pump: slow background dose (units per hour)

    # Anything else (e.g. the Chinese copy of the meal column) we don't need.
    # "None" means "nothing", and we'll use it to mean "drop this column".
    return None


# ---------------------------------------------------------------------------
# 3. A FUNCTION TO LOAD ONE PATIENT FILE
# ---------------------------------------------------------------------------
def load_one_recording(file_path):
    # Read the Excel file into a pandas "DataFrame" (= a table in Python).
    table = pd.read_excel(file_path)

    # Build a dictionary {old name: new name} for the columns we want to keep.
    # A dictionary is like a real dictionary: you look up a word (key)
    # and get its meaning (value).
    rename_map = {}
    for old_name in table.columns:
        new_name = tidy_column_name(old_name)
        if new_name is not None:
            rename_map[old_name] = new_name

    # Keep only those columns, then rename them.
    table = table[list(rename_map.keys())]
    table = table.rename(columns=rename_map)

    # The file name tells us who this is, e.g. "2000_0_20201230.xlsx"
    #   2000      = patient number
    #   0         = recording period (some patients were recorded twice)
    #   20201230  = start date (30 Dec 2020)
    # file_path.stem is the file name without ".xlsx".
    recording_id = file_path.stem
    patient_id = recording_id.split("_")[0]  # split at "_" and take the first piece

    # Add these as columns at the front of the table (position 0 and 1).
    table.insert(0, "recording_id", recording_id)
    table.insert(1, "patient_id", patient_id)
    return table


# ---------------------------------------------------------------------------
# 4. THE MAIN PROGRAM
# ---------------------------------------------------------------------------
def main():
    # Find every Excel file in the folder (*.xls and *.xlsx), sorted by name.
    # The "Summary" file is different (one row per patient), so we skip it here.
    all_files = sorted(RAW_FOLDER.glob("*.xls*"))
    patient_files = [f for f in all_files if "Summary" not in f.name]
    print(f"Found {len(patient_files)} patient files")

    # Load every file and collect the tables in a list.
    tables = []
    for file_path in patient_files:
        tables.append(load_one_recording(file_path))

    # pd.concat stacks all the tables on top of each other into one big table.
    # If a file is missing a column, pandas fills that spot with NaN
    # ("Not a Number" = empty).
    readings = pd.concat(tables, ignore_index=True)

    # Make sure "time" is understood as a date/time, not just text.
    # format="mixed" copes with files that store times slightly differently,
    # and .dt.round("min") rounds every time to the nearest minute.
    readings["time"] = pd.to_datetime(readings["time"], format="mixed").dt.round("min")

    # Meal descriptions contain line breaks ("Rice 150 g\nVegetable 100 g").
    # Replace them with "; " so each meal fits on one line.
    readings["meal"] = readings["meal"].str.replace("\n", "; ")

    # Sort by recording, then by time, so everything is in order.
    readings = readings.sort_values(["recording_id", "time"])

    # Save it! index=False means "don't save pandas' row numbers as a column".
    readings_path = OUTPUT_FOLDER / "t2dm_readings.csv"
    # date_format makes every time look the same, e.g. "2020-12-30 13:54:00".
    readings.to_csv(readings_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    # ---- The summary sheet: one row per recording ----
    summary = pd.read_excel(SUMMARY_FILE)
    summary.columns = [str(c).strip() for c in summary.columns]  # remove stray spaces
    summary = summary.rename(columns={"Patient Number": "recording_id"})
    patients_path = OUTPUT_FOLDER / "t2dm_patients.csv"
    summary.to_csv(patients_path, index=False)

    # ---- Print a little report so we can check it worked ----
    print(f"Saved {len(readings):,} rows to {readings_path.name}")
    print(f"Saved {len(summary)} recordings to {patients_path.name}")
    print()
    print("Columns:", list(readings.columns))
    print()
    print("How many rows actually have a value in each column:")
    print(readings.count())  # .count() = number of non-empty cells per column
    print()
    print("First 5 rows:")
    print(readings.head())


# This line means: "only run main() when this file is run directly".
# (If another script imports this file, main() won't run automatically.)
if __name__ == "__main__":
    main()
