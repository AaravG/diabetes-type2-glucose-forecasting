"""
STEP 2: Draw a graph of ONE patient's blood sugar, with meals and insulin marked.

WHY?
    Before building any prediction model, you should LOOK at your data.
    The graph shows how sugar rises after meals and falls after insulin,
    which is exactly what our equations (the "bathtub") describe.

HOW TO RUN:
    First run step1_build_csv.py (once). Then:   python step2_plot_patient.py
    To look at a different patient, change RECORDING_ID below.

WHAT IT CREATES:
    outputs/<recording_id>.png   -> the graph, saved as a picture
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt  # matplotlib is Python's most common graph-drawing library

# ---------------------------------------------------------------------------
# SETTINGS: change these to explore
# ---------------------------------------------------------------------------
RECORDING_ID = "2000_0_20201230"  # which recording to draw (see data/t2dm_patients.csv)
DAYS_TO_SHOW = 3                  # how many days to draw (a full recording can be 14 days)

PROJECT_FOLDER = Path(__file__).parent
READINGS_CSV = PROJECT_FOLDER / "data" / "t2dm_readings.csv"
PATIENTS_CSV = PROJECT_FOLDER / "data" / "t2dm_patients.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"

# The healthy "target range" doctors use for CGM sugar: 70 to 180 mg/dL.
LOW_LIMIT = 70
HIGH_LIMIT = 180


def main():
    # ---- 1. Load the CSV made in step 1 ----
    # parse_dates tells pandas the "time" column holds dates/times.
    # low_memory=False makes pandas read the whole file before deciding each
    # column's type (some columns mix numbers and text, which confuses it otherwise).
    readings = pd.read_csv(READINGS_CSV, parse_dates=["time"], low_memory=False)
    patients = pd.read_csv(PATIENTS_CSV)

    # ---- 2. Keep only the rows for our chosen recording ----
    # readings["recording_id"] == RECORDING_ID gives True/False for every row;
    # putting that inside readings[...] keeps only the True rows.
    one = readings[readings["recording_id"] == RECORDING_ID]

    # Keep only the first few days so the graph isn't too crowded.
    start = one["time"].min()
    end = start + pd.Timedelta(days=DAYS_TO_SHOW)
    one = one[one["time"] < end]

    # Look up this patient's details (age, BMI...) from the patients table.
    info = patients[patients["recording_id"] == RECORDING_ID].iloc[0]  # .iloc[0] = first matching row

    # ---- 3. Some simple numbers about this patient ----
    sugar = one["cgm"]
    in_range = ((sugar >= LOW_LIMIT) & (sugar <= HIGH_LIMIT)).mean() * 100
    # (sugar >= 70) & (sugar <= 180) is True/False per reading.
    # .mean() of True/False = the fraction that are True. x100 = percentage.
    print(f"Recording {RECORDING_ID}: age {info['Age (years)']}, BMI {info['BMI (kg/m2)']}")
    print(f"Average sugar: {sugar.mean():.0f} mg/dL")
    print(f"Lowest / highest: {sugar.min():.0f} / {sugar.max():.0f} mg/dL")
    print(f"Time in healthy range (70-180): {in_range:.0f}%")

    # ---- 4. Draw the graph ----
    # fig = the whole picture, ax = the area where we draw.
    fig, ax = plt.subplots(figsize=(14, 5))

    # Shade the healthy range in light green.
    ax.axhspan(LOW_LIMIT, HIGH_LIMIT, color="green", alpha=0.08, label="Healthy range (70-180)")

    # The main line: sugar over time.
    ax.plot(one["time"], one["cgm"], color="tab:blue", linewidth=1.5, label="Sensor sugar (CGM)")

    # Meals: draw an orange dot at the sugar level at the moment they ate.
    # .notna() = "not empty", so this keeps only rows where a meal was logged.
    meals = one[one["meal"].notna()]
    ax.scatter(meals["time"], meals["cgm"], color="tab:orange", s=60, zorder=3, label="Meal")

    # Insulin: draw a red triangle near the bottom of the graph.
    insulin = one[one["insulin_pen"].notna()]
    ax.scatter(insulin["time"], [45] * len(insulin), marker="^", color="tab:red", s=60, label="Insulin injection")

    # Labels and title make the graph readable for your teacher.
    ax.set_title(f"Blood sugar of patient {RECORDING_ID} (Type 2 diabetes), first {DAYS_TO_SHOW} days")
    ax.set_xlabel("Date and time")
    ax.set_ylabel("Blood sugar (mg/dL)")
    ax.set_ylim(30, max(300, sugar.max() + 20))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()  # stops labels from being cut off

    # ---- 5. Save the picture, then show it on screen ----
    OUTPUT_FOLDER.mkdir(exist_ok=True)  # make the "outputs" folder if it doesn't exist
    picture_path = OUTPUT_FOLDER / f"{RECORDING_ID}.png"
    fig.savefig(picture_path, dpi=150)
    print(f"Saved graph to {picture_path}")
    plt.show()


if __name__ == "__main__":
    main()
