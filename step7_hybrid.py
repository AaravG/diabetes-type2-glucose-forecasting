"""
STEP 7: Contestant 4, the Hybrid. This is the main question of the project.

THE IDEA:
    The Hybrid is the Pattern Learner with two extra inputs: the sugar change
    the Equations predict for 30 and for 60 minutes ahead. Everything else
    (patients, model settings, moments tested) stays the same, so any
    difference in the score comes from those two inputs.

THE QUESTION:
    For Type 2 diabetes, does adding the equations help a computer predict
    sugar for patients it has never seen?

IS A DIFFERENCE REAL OR JUST LUCK?
    With only 20 test patients, a small difference could be chance. So we use
    a Wilcoxon signed-rank test, which compares two methods patient by patient.
    It gives a "p-value": roughly, how often you'd see a difference this big
    if the two methods were truly equal. Below 0.05 is the usual cut-off for
    calling a difference real. Above that, we can't say either one is better.

HOW TO RUN:
    First run steps 1, 3, 4 and 6. Then:   python step7_hybrid.py

WHAT IT CREATES:
    outputs/results.csv             -> adds this contestant to the scoreboard
    outputs/clarke_hybrid.png       -> its Clarke Error Grids
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import common
import features
import step5_pattern_learner as pattern  # reuses the exact same training code

EQUATIONS_CSV = common.PROJECT_FOLDER / "data" / "equations_predictions.csv"
LOW_SUGAR = 70  # below this counts as dangerously low


def per_patient_error(test, guess, horizon):
    """Returns each patient's average error, as a Series indexed by patient."""
    errors = np.abs(np.asarray(guess) - test[f"future_{horizon}"].to_numpy())
    return pd.Series(errors).groupby(test["patient_id"].to_numpy()).mean()


def main():
    print("Building the inputs...")
    table, feature_names = features.build_feature_table()

    # Attach the Equations' guesses to each moment.
    equations = pd.read_csv(EQUATIONS_CSV, parse_dates=["time"])
    usable = table[table["usable"]].merge(equations, on=["recording_id", "time"], how="left")

    # Every moment must have a guess, or the comparison wouldn't be fair.
    missing = usable["equations_30"].isna().sum()
    if missing > 0:
        raise ValueError(f"{missing} moments have no Equations guess. Run step6_equations.py first.")

    # The two extra inputs: how much the Equations expect sugar to change.
    hybrid_names = [*feature_names]
    for horizon in common.HORIZONS:
        usable[f"equations_change_{horizon}"] = usable[f"equations_{horizon}"] - usable["cgm"]
        hybrid_names.append(f"equations_change_{horizon}")
        usable[f"change_{horizon}"] = usable[f"future_{horizon}"] - usable["cgm"]

    # The same practice, validation and test patients as step 5.
    practice = usable[usable["group"] == "train"]
    test = usable[usable["group"] == "test"]
    practice_ids = pd.Series(sorted(practice["patient_id"].unique()))
    validation_ids = practice_ids.sample(n=pattern.VALIDATION_PATIENTS, random_state=common.RANDOM_SEED)
    is_validation = practice["patient_id"].isin(validation_ids)
    fit_rows = practice[~is_validation]
    validation_rows = practice[is_validation]
    print(f"Testing on {len(test):,} moments from {test['patient_id'].nunique()} hidden patients")

    # ---- Train both models, so we can compare them patient by patient ----
    all_guesses = {"Lazy Guess": {}, "Equations": {}, "Pattern Learner": {}, "Hybrid": {}}
    for horizon in common.HORIZONS:
        print("\nPattern Learner:")
        learner = pattern.train_model(fit_rows, validation_rows, practice, feature_names, horizon)
        print("Hybrid:")
        hybrid = pattern.train_model(fit_rows, validation_rows, practice, hybrid_names, horizon)

        now = test["cgm"].to_numpy()
        all_guesses["Lazy Guess"][horizon] = now
        all_guesses["Equations"][horizon] = test[f"equations_{horizon}"].to_numpy()
        all_guesses["Pattern Learner"][horizon] = now + learner.predict(test[feature_names])
        all_guesses["Hybrid"][horizon] = now + hybrid.predict(test[hybrid_names])

        importance = pd.Series(hybrid.feature_importances_, index=hybrid_names).sort_values(ascending=False)
        print(f"  Hybrid's most useful inputs for {horizon} min ahead:")
        print((importance.head(6) * 100).round(1).to_string())

    # ---- The scoreboard ----
    scores = {h: common.evaluate(test[f"future_{h}"], all_guesses["Hybrid"][h]) for h in common.HORIZONS}
    print("\nScoreboard:")
    print(common.save_results("Hybrid", scores).to_string(index=False))

    # ---- Is the difference real? ----
    print("\nPatient-by-patient comparison (20 hidden patients):")
    for horizon in common.HORIZONS:
        hybrid_error = per_patient_error(test, all_guesses["Hybrid"][horizon], horizon)
        for rival in ["Pattern Learner", "Equations", "Lazy Guess"]:
            rival_error = per_patient_error(test, all_guesses[rival][horizon], horizon)
            wins = (hybrid_error < rival_error).sum()
            gain = (rival_error - hybrid_error).mean()
            p_value = wilcoxon(hybrid_error, rival_error).pvalue
            print(f"  {horizon} min, Hybrid vs {rival:15s}: better on {wins:2d} of {len(hybrid_error)}, "
                  f"average gain {gain:5.2f} mg/dL, p = {p_value:.4f}")

    # ---- How does each contestant do when sugar really is dangerously low? ----
    # "Average bias" is guess minus truth. Positive means it guessed too high,
    # which is the risky direction when sugar is low.
    print(f"\nWhen the real sugar turned out below {LOW_SUGAR} mg/dL:")
    for horizon in common.HORIZONS:
        truth = test[f"future_{horizon}"].to_numpy()
        is_low = truth < LOW_SUGAR
        print(f"  {horizon} min ahead ({is_low.sum()} moments):")
        for name, guesses in all_guesses.items():
            difference = guesses[horizon][is_low] - truth[is_low]
            print(f"    {name:16s} average error {np.mean(np.abs(difference)):6.2f}, "
                  f"average bias {np.mean(difference):+6.2f}")

    # ---- Draw the Clarke Error Grids ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, horizon in zip(axes, common.HORIZONS):
        zone_a = scores[horizon]["zone_A_%"]
        common.draw_clarke_grid(ax, test[f"future_{horizon}"], all_guesses["Hybrid"][horizon],
                                f"Hybrid, {horizon} min ahead (Zone A: {zone_a:.1f}%)")
    fig.tight_layout()
    fig.savefig(common.OUTPUT_FOLDER / "clarke_hybrid.png", dpi=150)
    print("\nSaved the grids to clarke_hybrid.png")


if __name__ == "__main__":
    main()
