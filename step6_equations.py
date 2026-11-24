"""
STEP 6: Contestant 2, the Equations (the Bergman "bathtub" model).

THE IDEA:
    Every minute: new sugar = old sugar + food - natural return - insulin's effect.
    Repeat that 30 or 60 times to see the future.

WHAT'S IN THE SIMULATION (each one is a number that changes over time):
    gut_1, gut_2       -> food moving through the stomach, then the gut
    skin_1, skin_2     -> injected insulin slowly moving out from under the skin
    insulin            -> extra insulin in the blood, above the person's normal level
    action             -> how hard insulin is working right now (called X in the lesson)
    sugar              -> blood sugar (called G in the lesson)

    Two "gut" and two "skin" boxes in a row are simply a neat way to make food
    and insulin arrive gradually, rising to a peak and then fading, instead of
    all at once.

THE RULES:
    d(sugar)/dt   = -(p1 + action) x sugar + p1 x usual_sugar + food_arriving / V
    d(action)/dt  = p2 x (SI x insulin - action)
    d(insulin)/dt = -n x insulin + injected_arriving + gamma x (sugar above usual)

HOW THE FORECAST AVOIDS CHEATING:
    Up to the moment of the forecast, the simulation can use everything already
    recorded. After that moment it knows NOTHING new: no future meals, no future
    injections, no future sugar. It just lets what's already inside play out.

WHAT GETS "LEARNED":
    Only a handful of numbers (p1, p2, SI, gamma, tau_meal, meal_strength), shared
    by everyone, chosen to fit the practice patients. Hidden patients are never used.

PERSONALISING WITH LAB TESTS:
    C-peptide is a blood test showing how much insulin the pancreas is making.
    We test whether using it to adjust SI and gamma for each person helps.
    The validation patients decide. It's only kept if it actually works.

HOW TO RUN:
    First run steps 1, 3 and 4. Then:   python step6_equations.py
    It takes a few minutes, because it tries thousands of simulations.

WHAT IT CREATES:
    outputs/results.csv                  -> adds this contestant to the scoreboard
    outputs/clarke_equations.png         -> its Clarke Error Grids
    data/equations_predictions.csv       -> its guesses, which the Hybrid will use
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # tries different numbers to find the best ones
import common
import features

VALIDATION_PATIENTS = 16

# ---------------------------------------------------------------------------
# FIXED NUMBERS (taken as typical human values, not learned)
# ---------------------------------------------------------------------------
GLUCOSE_VOLUME_PER_KG = 1.6    # V: decilitres of body fluid that sugar spreads into, per kg
INSULIN_VOLUME_PER_KG = 120    # millilitres that insulin spreads into, per kg
INSULIN_CLEARANCE = 0.14       # n: share of blood insulin removed each minute
TAU_INSULIN = 55               # minutes for injected insulin to reach its peak
UNITS_TO_MICRO = 1_000_000     # 1 unit of insulin (IU) = 1,000,000 micro-units
LOWEST_SENSOR_VALUE = 39.6     # the sensor can't read below this
HIGHEST_SUGAR = 500            # stops the simulation running away to silly values

# How finely we step through time.
HISTORY_STEPS_PER_READING = 3  # 3 steps of 5 minutes for each 15-minute reading
HISTORY_STEP_MINUTES = common.MINUTES_PER_READING / HISTORY_STEPS_PER_READING

# Starting guesses for the numbers we learn.
STARTING_VALUES = {
    "p1": 0.02,        # how hard the body pulls sugar back to usual, per minute
    "p2": 0.025,       # how quickly insulin's action builds up and fades
    "SI": 3e-4,        # insulin sensitivity
    "gamma": 0.03,     # pancreas strength
    "tau_meal": 40,    # minutes for food to reach its peak
    "meal_strength": 0.5,  # share of the carbs eaten that shows up as blood sugar
}
POSITIVE_NAMES = list(STARTING_VALUES)

# Believable limits for each learned number: (lowest, highest).
# Without these the search wanders off to values no human body could have.
LIMITS = {
    "p1": (1e-4, 0.1),
    "p2": (0.005, 0.2),
    "SI": (1e-6, 2e-3),
    "gamma": (1e-4, 0.5),
    "tau_meal": (15, 120),
    "meal_strength": (0.02, 1.5),
}


def add_model_inputs(table):
    """Adds the columns the simulation needs."""
    table = table.sort_values(["recording_id", "time"]).reset_index(drop=True)

    # Each person's "usual sugar" is the average of their last 24 hours (up to now).
    # 96 readings x 15 minutes = 24 hours. This plays the role of G_b.
    table["usual_sugar"] = table.groupby("recording_id")["cgm"].transform(
        lambda s: s.rolling(96, min_periods=1).mean())

    # A few readings are missing, so we carry the last known value forward.
    table["sugar_filled"] = table.groupby("recording_id")["cgm"].ffill()
    table["row_number"] = table.groupby("recording_id").cumcount()

    # Patient weight sets how far sugar and insulin spread out in the body.
    weights = pd.read_csv(features.PATIENTS_CSV)[["recording_id", "Weight (kg)"]]
    weights = weights.rename(columns={"Weight (kg)": "weight"})
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    return table.merge(weights, on="recording_id", how="left")


def make_cohort(table, typical_meal, lab_medians):
    """Packs a group of recordings into grids of numbers.
    Each grid has one row per 15-minute reading and one column per recording,
    so numpy can do the maths for every recording at once, which is far faster."""
    table = table.copy()
    recording_ids = sorted(table["recording_id"].unique())
    column_of = {rid: i for i, rid in enumerate(recording_ids)}  # recording -> column number
    table["column_number"] = table["recording_id"].map(column_of)

    # A meal with no details still happened, so we assume a typical size.
    carbs = table["carbs_g"].where(table["meal"].notna(), 0).fillna(typical_meal)

    n_rows = table["row_number"].max() + 1
    n_columns = len(recording_ids)

    def to_grid(values):
        grid = np.full((n_rows, n_columns), np.nan)  # start with a grid of "empty"
        grid[table["row_number"], table["column_number"]] = values
        return grid

    # One value per recording: the first row of each recording is enough.
    each = table.groupby("column_number").first().sort_index()
    weight = each["weight"].fillna(each["weight"].median()).to_numpy()

    # How each person's labs compare with a typical patient (1.0 = typical).
    # Anyone missing a lab test just counts as typical.
    pancreas = (each["lab_fasting_cpeptide"] / lab_medians["cpeptide"]).fillna(1.0).to_numpy()
    resistance = (each["lab_fasting_cpeptide"] * each["lab_fasting_glucose"]
                  / lab_medians["resistance"]).fillna(1.0).to_numpy()

    usable = table[table["usable"]]
    return {
        "sugar": to_grid(table["sugar_filled"]),
        "usual": to_grid(table["usual_sugar"]),
        "carbs": np.nan_to_num(to_grid(carbs)),               # empty cells become 0
        "insulin": np.nan_to_num(to_grid(table["fast_insulin_now"])),
        "weight": weight,
        "pancreas": pancreas,
        "resistance": resistance,
        "rows": usable["row_number"].to_numpy(),              # the moments we forecast from
        "columns": usable["column_number"].to_numpy(),
        "truth_30": usable["future_30"].to_numpy(),
        "truth_60": usable["future_60"].to_numpy(),
        "lookup": usable[["recording_id", "time", "patient_id", "cgm"]].reset_index(drop=True),
    }


def advance(state, sugar, usual, settings, dt):
    """Moves everything except sugar forward by dt minutes (Euler's method)."""
    gut_1, gut_2, skin_1, skin_2, insulin, action = state
    SI, gamma, p2, tau_meal = settings

    # Each "box" empties into the next one. The rate is amount / time constant.
    new_gut_1 = gut_1 + dt * (-gut_1 / tau_meal)
    new_gut_2 = gut_2 + dt * ((gut_1 - gut_2) / tau_meal)
    new_skin_1 = skin_1 + dt * (-skin_1 / TAU_INSULIN)
    new_skin_2 = skin_2 + dt * ((skin_1 - skin_2) / TAU_INSULIN)

    injected_arriving = skin_2 / TAU_INSULIN
    # np.maximum(x, 0) replaces negatives with 0: the pancreas only reacts to HIGH sugar.
    pancreas_release = gamma * np.maximum(sugar - usual, 0)
    new_insulin = insulin + dt * (-INSULIN_CLEARANCE * insulin + injected_arriving + pancreas_release)
    new_action = action + dt * (p2 * (SI * insulin - action))

    return new_gut_1, new_gut_2, new_skin_1, new_skin_2, new_insulin, new_action


def forecast(cohort, params, moment_step=1):
    """Runs the model. Returns the 30 and 60 minute guesses."""
    n_rows, n_columns = cohort["sugar"].shape
    weight = cohort["weight"]

    # Each person's own sensitivity and pancreas strength.
    # With both exponents at 0 everything to the power 0 is 1, so nobody is adjusted.
    person_SI = params["SI"] * cohort["resistance"] ** (-params["a_resistance"])
    person_gamma = params["gamma"] * cohort["pancreas"] ** params["a_pancreas"]

    # ---- Part 1: replay everything that has already happened ----
    state = tuple(np.zeros(n_columns) for _ in range(6))
    saved = np.zeros((6, n_rows, n_columns))
    settings = (person_SI, person_gamma, params["p2"], params["tau_meal"])

    for row in range(n_rows):
        gut_1, gut_2, skin_1, skin_2, insulin, action = state
        # Anything eaten or injected at this reading goes in now.
        gut_1 = gut_1 + cohort["carbs"][row] * 1000 * params["meal_strength"]     # grams -> mg
        skin_1 = skin_1 + cohort["insulin"][row] * UNITS_TO_MICRO / (INSULIN_VOLUME_PER_KG * weight)
        state = (gut_1, gut_2, skin_1, skin_2, insulin, action)

        # Remember the situation at this moment, for forecasting later.
        saved[:, row] = np.stack(state)

        # Roll forward to the next reading, using the REAL sugar we measured.
        for _ in range(HISTORY_STEPS_PER_READING):
            state = advance(state, cohort["sugar"][row], cohort["usual"][row], settings, HISTORY_STEP_MINUTES)

    # ---- Part 2: from each moment, play the future forward ----
    rows = cohort["rows"][::moment_step]        # [::4] would mean "every 4th moment"
    columns = cohort["columns"][::moment_step]

    state = tuple(saved[i, rows, columns] for i in range(6))
    sugar = cohort["sugar"][rows, columns]
    usual = cohort["usual"][rows, columns]
    glucose_volume = GLUCOSE_VOLUME_PER_KG * weight[columns]
    settings = (person_SI[columns], person_gamma[columns], params["p2"], params["tau_meal"])

    guesses = {}
    for minute in range(1, 61):
        _, gut_2, _, _, _, action = state
        food_arriving = gut_2 / params["tau_meal"]

        # THE MAIN EQUATION: change = natural return + insulin's effect + food.
        change = -(params["p1"] + action) * sugar + params["p1"] * usual + food_arriving / glucose_volume

        state = advance(state, sugar, usual, settings, dt=1)
        sugar = np.clip(sugar + change, LOWEST_SENSOR_VALUE, HIGHEST_SUGAR)
        if minute in common.HORIZONS:
            guesses[minute] = sugar.copy()
    return guesses


def unpack(x, personalised):
    """The optimiser works with a plain list of numbers; we want named values.
    We store logarithms so that the values can never become negative."""
    params = {name: float(np.exp(x[i])) for i, name in enumerate(POSITIVE_NAMES)}
    params["a_resistance"] = float(x[len(POSITIVE_NAMES)]) if personalised else 0.0
    params["a_pancreas"] = float(x[len(POSITIVE_NAMES) + 1]) if personalised else 0.0
    return params


def average_error(cohort, params, moment_step=1):
    """The MAE, averaged across the two horizons."""
    guesses = forecast(cohort, params, moment_step)
    errors = [np.mean(np.abs(guesses[h] - cohort[f"truth_{h}"][::moment_step])) for h in common.HORIZONS]
    return float(np.mean(errors))


def fit(cohort, personalised, start_x, max_tries, rounds=3):
    """Tries lots of parameter values to find the ones with the lowest error.
    Nelder-Mead is a method that feels its way downhill without needing calculus.
    It can get stuck, so we restart it from its best answer a few times ("rounds")."""
    def objective(x):
        params = unpack(x, personalised)
        # Anything outside the believable limits gets a terrible score, so it's rejected.
        for name, (lowest, highest) in LIMITS.items():
            if not lowest <= params[name] <= highest:
                return 1e6
        return average_error(cohort, params, moment_step=4)

    x = start_x
    for _ in range(rounds):
        result = minimize(objective, x, method="Nelder-Mead",
                          options={"maxfev": max_tries, "xatol": 1e-3, "fatol": 1e-3})
        x = result.x
    return x


def main():
    print("Preparing the data...")
    table, _ = features.build_feature_table()
    table = add_model_inputs(table)

    # The same 64 / 16 / 20 patient split that the Pattern Learner used.
    practice_table = table[table["group"] == "train"]
    usable_practice = practice_table[practice_table["usable"]]
    practice_ids = pd.Series(sorted(usable_practice["patient_id"].unique()))
    validation_ids = practice_ids.sample(n=VALIDATION_PATIENTS, random_state=common.RANDOM_SEED)
    is_validation = practice_table["patient_id"].isin(validation_ids)

    # Typical values are taken from practice patients only.
    known_meals = practice_table.loc[practice_table["carbs_g"].notna(), "carbs_g"]
    typical_meal = float(known_meals.median())
    per_recording = practice_table.groupby("recording_id").first()
    lab_medians = {
        "cpeptide": float(per_recording["lab_fasting_cpeptide"].median()),
        "resistance": float((per_recording["lab_fasting_cpeptide"]
                             * per_recording["lab_fasting_glucose"]).median()),
    }
    print(f"Typical meal assumed for unknown meals: {typical_meal:.0f} g of carbs")

    fit_cohort = make_cohort(practice_table[~is_validation], typical_meal, lab_medians)
    validation_cohort = make_cohort(practice_table[is_validation], typical_meal, lab_medians)
    practice_cohort = make_cohort(practice_table, typical_meal, lab_medians)
    test_cohort = make_cohort(table[table["group"] == "test"], typical_meal, lab_medians)

    # ---- Stage 1: does personalising with lab tests help? ----
    start = [np.log(STARTING_VALUES[name]) for name in POSITIVE_NAMES]

    print("\nFitting the shared version (same numbers for everybody)...")
    shared_x = fit(fit_cohort, personalised=False, start_x=start, max_tries=600)
    shared_error = average_error(validation_cohort, unpack(shared_x, False))
    print(f"  Validation error: {shared_error:.2f} mg/dL")

    print("Fitting the personalised version (adjusted using C-peptide)...")
    # Both versions get the same amount of searching, so the comparison is fair.
    personal_x = fit(fit_cohort, personalised=True, start_x=[*start, 0.3, 0.3], max_tries=600)
    personal_error = average_error(validation_cohort, unpack(personal_x, True))
    print(f"  Validation error: {personal_error:.2f} mg/dL")

    personalised = personal_error < shared_error
    best_x = personal_x if personalised else shared_x
    print(f"Winner on the validation patients: {'personalised' if personalised else 'shared'}")

    # ---- Stage 2: polish the winner using all 80 practice patients ----
    print("\nFinal fit on all 80 practice patients...")
    final_x = fit(practice_cohort, personalised, best_x, max_tries=300, rounds=1)
    params = unpack(final_x, personalised)
    print("Learned values:")
    for name, value in params.items():
        print(f"  {name:13s} {value:.5g}")

    # ---- Stage 3: the one and only test ----
    guesses = forecast(test_cohort, params)
    scores = {h: common.evaluate(test_cohort[f"truth_{h}"], guesses[h]) for h in common.HORIZONS}
    scoreboard = common.save_results("Equations", scores)
    print("\nScoreboard:")
    print(scoreboard.to_string(index=False))

    # Patient-by-patient comparison with the Lazy Guess, as in step 5.
    print("\nPatient by patient, compared with the Lazy Guess:")
    lookup = test_cohort["lookup"]
    for h in common.HORIZONS:
        truth = test_cohort[f"truth_{h}"]
        results = pd.DataFrame({
            "patient_id": lookup["patient_id"],
            "model_error": np.abs(guesses[h] - truth),
            "lazy_error": np.abs(lookup["cgm"].to_numpy() - truth),
        })
        per_patient = results.groupby("patient_id").mean()
        wins = (per_patient["model_error"] < per_patient["lazy_error"]).sum()
        gain = (per_patient["lazy_error"] - per_patient["model_error"]).mean()
        print(f"  {h} min ahead: better on {wins} of {len(per_patient)} patients, "
              f"average improvement {gain:.2f} mg/dL per patient")

    # ---- Save the Clarke grids ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, h in zip(axes, common.HORIZONS):
        common.draw_clarke_grid(ax, test_cohort[f"truth_{h}"], guesses[h],
                                f"Equations, {h} min ahead (Zone A: {scores[h]['zone_A_%']:.1f}%)")
    fig.tight_layout()
    fig.savefig(common.OUTPUT_FOLDER / "clarke_equations.png", dpi=150)

    # ---- Save the guesses for the Hybrid ----
    saved_rows = []
    for cohort in [practice_cohort, test_cohort]:
        predictions = forecast(cohort, params)
        rows = cohort["lookup"][["recording_id", "time"]].copy()
        for h in common.HORIZONS:
            rows[f"equations_{h}"] = predictions[h]
        saved_rows.append(rows)
    pd.concat(saved_rows, ignore_index=True).to_csv(
        common.PROJECT_FOLDER / "data" / "equations_predictions.csv",
        index=False, date_format="%Y-%m-%d %H:%M:%S")
    print("\nSaved the grids and the predictions for the Hybrid.")


if __name__ == "__main__":
    main()
