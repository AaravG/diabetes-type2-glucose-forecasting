"""
COMMON TOOLS: code that several steps share, so we only write it once.

You don't run this file directly. Other scripts bring it in with:
    import common

WHAT'S IN HERE:
    load_readings()       -> loads the big table from step 1
    add_future_sugar()    -> adds the answers we're trying to predict
    make_patient_split()  -> decides which patients are for practice and which for the test
    clarke_zone()         -> grades one guess as A, B, C, D or E
    evaluate()            -> calculates all the scores for a set of guesses
    save_results()        -> adds a contestant's scores to the results table
    draw_clarke_grid()    -> draws the Clarke Error Grid
"""

from pathlib import Path
import numpy as np     # numpy does fast maths on whole columns of numbers
import pandas as pd

PROJECT_FOLDER = Path(__file__).parent
READINGS_CSV = PROJECT_FOLDER / "data" / "t2dm_readings.csv"
SPLIT_CSV = PROJECT_FOLDER / "data" / "patient_split.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
RESULTS_CSV = OUTPUT_FOLDER / "results.csv"

# The sensor takes one reading every 15 minutes.
MINUTES_PER_READING = 15

# How far ahead we predict, in minutes.
HORIZONS = [30, 60]

# Each contestant is allowed to see the last 2 hours of sugar = 8 readings.
HISTORY_READINGS = 8

# 20 out of 100 patients are kept hidden for the final test.
TEST_PATIENTS = 20

# A "seed" makes random choices repeatable: same seed, same patients every time.
# RULE: never change this after seeing results. Picking the split that makes
# your model look best is cheating, even if you do it by accident.
RANDOM_SEED = 42


def load_readings():
    return pd.read_csv(READINGS_CSV, parse_dates=["time"], low_memory=False)


def add_future_sugar(readings):
    """Adds the columns future_30 and future_60 (the real sugar later on),
    plus "usable", which is True for rows every contestant can be tested on."""
    readings = readings.sort_values(["recording_id", "time"]).copy()

    # groupby makes sure we never look across two different recordings.
    by_recording = readings.groupby("recording_id")

    for minutes in HORIZONS:
        steps = minutes // MINUTES_PER_READING  # "//" divides and drops any remainder
        # shift(-2) means "the value from 2 rows further down", i.e. the future.
        future_sugar = by_recording["cgm"].shift(-steps)
        future_time = by_recording["time"].shift(-steps)

        # The sensor sometimes skips a reading. Then "2 rows down" isn't really
        # 30 minutes later, so we only trust rows where the time matches exactly.
        time_is_right = (future_time - readings["time"]) == pd.Timedelta(minutes=minutes)
        readings[f"future_{minutes}"] = future_sugar.where(time_is_right)

    # Does this row have a complete 2 hours of history behind it?
    # shift(+7) means "the value from 7 rows further up", i.e. the past.
    oldest_time = by_recording["time"].shift(HISTORY_READINGS - 1)
    expected_gap = pd.Timedelta(minutes=(HISTORY_READINGS - 1) * MINUTES_PER_READING)
    history_time_ok = (readings["time"] - oldest_time) == expected_gap

    # rolling(8).count() counts how many of the last 8 readings aren't empty.
    readings_seen = by_recording["cgm"].transform(lambda s: s.rolling(HISTORY_READINGS).count())
    history_complete = history_time_ok & (readings_seen == HISTORY_READINGS)

    # Every contestant gets tested on exactly the same rows, otherwise the race isn't fair.
    future_columns = [f"future_{minutes}" for minutes in HORIZONS]
    readings["usable"] = history_complete & readings[future_columns].notna().all(axis=1)
    return readings


def make_patient_split(readings):
    """Returns a table with one row per patient, labelled "train" or "test".

    We split by PATIENT, not by row. If one person's readings ended up in both
    groups, the model could memorise them, and the test wouldn't be fair."""
    patient_ids = pd.Series(sorted(readings["patient_id"].unique()), name="patient_id")
    test_ids = patient_ids.sample(n=TEST_PATIENTS, random_state=RANDOM_SEED)

    split = patient_ids.to_frame()
    split["group"] = np.where(split["patient_id"].isin(test_ids), "test", "train")
    return split


def clarke_zone(true_sugar, guess):
    """Grades one guess using the Clarke Error Grid rules (Clarke et al., 1987)."""
    r, p = true_sugar, guess

    # A: within 20% of the truth, or both are low.
    if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r):
        return "A"
    # E: the opposite treatment. Really high but the guess says low, or the reverse.
    if (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
        return "E"
    # C: would make someone "fix" a sugar level that was actually fine.
    if (70 <= r <= 290 and p >= r + 110) or (130 <= r <= 180 and p <= (7 / 5) * r - 182):
        return "C"
    # D: misses a real danger.
    if (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or (175 / 3 <= r <= 70 and p >= (6 / 5) * r):
        return "D"
    # B: wrong, but harmless.
    return "B"


def evaluate(true_sugar, guesses):
    """Takes the real values and the guesses, and returns a dictionary of scores."""
    errors = np.asarray(guesses) - np.asarray(true_sugar)

    # MAE = Mean Absolute Error: on average, how many mg/dL off are we?
    mae = np.mean(np.abs(errors))
    # RMSE = Root Mean Squared Error: like MAE, but big mistakes count extra.
    rmse = np.sqrt(np.mean(errors ** 2))

    # zip walks through two lists side by side.
    zones = pd.Series([clarke_zone(r, p) for r, p in zip(true_sugar, guesses)])
    shares = zones.value_counts(normalize=True) * 100  # normalize=True gives fractions

    scores = {"rows": len(errors), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}
    for zone in "ABCDE":
        scores[f"zone_{zone}_%"] = round(shares.get(zone, 0), 2)
    return scores


def save_results(contestant, scores_by_horizon):
    """Adds one contestant's scores to outputs/results.csv.
    scores_by_horizon looks like {30: {...scores...}, 60: {...scores...}}."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    new_rows = pd.DataFrame([{"contestant": contestant, "horizon_min": horizon, **scores}
                             for horizon, scores in scores_by_horizon.items()])

    # If this contestant is already in the file, replace its old scores.
    if RESULTS_CSV.exists():
        old_rows = pd.read_csv(RESULTS_CSV)
        old_rows = old_rows[old_rows["contestant"] != contestant]
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)

    new_rows.to_csv(RESULTS_CSV, index=False)
    return new_rows


def draw_clarke_grid(ax, true_sugar, guesses, title):
    """Draws the grid on a matplotlib drawing area (ax)."""
    # Each dot is one guess. alpha makes dots see-through, so dense areas look darker.
    ax.scatter(true_sugar, guesses, s=3, alpha=0.15, color="tab:blue", linewidths=0)

    # The official zone boundary lines, as ([x start, x end], [y start, y end]).
    ax.plot([0, 400], [0, 400], color="black", linestyle=":", linewidth=1)
    zone_lines = [
        ([0, 175 / 3], [70, 70]), ([175 / 3, 400 / 1.2], [70, 400]), ([70, 70], [84, 400]),
        ([0, 70], [180, 180]), ([70, 290], [180, 400]), ([70, 70], [0, 56]),
        ([70, 400], [56, 320]), ([180, 180], [0, 70]), ([180, 400], [70, 70]),
        ([240, 240], [70, 180]), ([240, 400], [180, 180]), ([130, 180], [0, 70]),
    ]
    for x_points, y_points in zone_lines:
        ax.plot(x_points, y_points, color="black", linewidth=1)

    # Write each zone's letter inside it.
    letters = [("A", 300, 300), ("A", 25, 25), ("B", 200, 285), ("B", 330, 195), ("C", 100, 295),
               ("C", 150, 8), ("D", 18, 120), ("D", 370, 110), ("E", 22, 290), ("E", 370, 20)]
    for letter, x, y in letters:
        ax.text(x, y, letter, fontsize=14, ha="center", va="center", weight="bold")

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect("equal")  # makes the grid a true square
    ax.set_xlabel("Real sugar (mg/dL)")
    ax.set_ylabel("Predicted sugar (mg/dL)")
    ax.set_title(title)
