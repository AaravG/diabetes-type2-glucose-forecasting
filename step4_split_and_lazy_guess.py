"""
STEP 4: Set up the fair test, then run Contestant 1, the Lazy Guess.

THE LAZY GUESS:
    "Sugar in 30 minutes will be the same as it is now."
    It uses no science at all. Any model that can't beat it is useless, so
    it sets the score every other contestant has to beat.

THE FAIR TEST:
    80 patients are for practice and 20 are hidden for the final test.
    Every contestant is scored on the same rows from those 20 patients.

HOW TO RUN:
    First run step1_build_csv.py (once). Then:   python step4_split_and_lazy_guess.py

WHAT IT CREATES:
    data/patient_split.csv               -> which patients are "train" and which are "test"
    outputs/results.csv                  -> the race scoreboard
    outputs/clarke_lazy_guess.png        -> the Clarke Error Grids for this contestant
"""

import matplotlib.pyplot as plt
import common  # our own file, common.py


def main():
    # ---- 1. Load the data and add the answers we want to predict ----
    readings = common.add_future_sugar(common.load_readings())

    # ---- 2. Split the patients, and save the split so every step uses the same one ----
    split = common.make_patient_split(readings)
    split.to_csv(common.SPLIT_CSV, index=False)

    # .merge joins two tables on a shared column, like matching names on two lists.
    readings = readings.merge(split, on="patient_id")

    # ---- 3. Check that the two groups look similar ----
    print("Are the practice and test groups similar?")
    check = readings.groupby("group").agg(
        patients=("patient_id", "nunique"),
        readings=("cgm", "count"),
        average_sugar=("cgm", "mean"),
    )
    print(check.round(1))
    print()

    # ---- 4. Take only the rows every contestant will be tested on ----
    # This is the ONLY place the Lazy Guess touches the test patients,
    # and it has nothing to learn, so nothing can leak.
    test_rows = readings[(readings["group"] == "test") & readings["usable"]]
    print(f"Testing on {len(test_rows):,} moments from {test_rows['patient_id'].nunique()} hidden patients")
    print()

    # ---- 5. Make the guesses and score them ----
    # The whole model is this one line: the guess is today's sugar.
    lazy_guess = test_rows["cgm"]

    # A dictionary comprehension: builds {30: scores, 60: scores}.
    scores = {horizon: common.evaluate(test_rows[f"future_{horizon}"], lazy_guess)
              for horizon in common.HORIZONS}

    scoreboard = common.save_results("Lazy Guess", scores)
    print("Scoreboard:")
    print(scoreboard.to_string(index=False))

    # ---- 6. Draw the Clarke Error Grids, one per horizon, side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, horizon in zip(axes, common.HORIZONS):
        zone_a = scores[horizon]["zone_A_%"]
        common.draw_clarke_grid(ax, test_rows[f"future_{horizon}"], lazy_guess,
                                f"Lazy Guess, {horizon} min ahead (Zone A: {zone_a:.1f}%)")
    fig.tight_layout()

    common.OUTPUT_FOLDER.mkdir(exist_ok=True)
    picture_path = common.OUTPUT_FOLDER / "clarke_lazy_guess.png"
    fig.savefig(picture_path, dpi=150)
    print(f"\nSaved the grids to {picture_path.name}")


if __name__ == "__main__":
    main()
