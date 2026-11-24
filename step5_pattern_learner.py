"""
STEP 5: Contestant 3, the Pattern Learner (machine learning with XGBoost).

HOW XGBOOST WORKS, IN ONE PARAGRAPH:
    It builds a small decision tree, like a game of 20 Questions:
    "Is sugar rising? Did they eat in the last hour?" That first tree makes
    mistakes. So it builds a second tree whose only job is to fix the mistakes
    of the first, then a third to fix what's left, and so on for hundreds of
    trees. The final guess adds all the trees' answers together.

WE PREDICT THE CHANGE, NOT THE SUGAR ITSELF:
    The model learns "sugar will go up by 18", and we add that to today's sugar.
    Trees can never give an answer outside what they saw in training, and
    changes are a much easier thing to learn than raw sugar levels.

THREE GROUPS OF PATIENTS:
    fitting (64)     -> the model learns from these
    validation (16)  -> tells us when to stop adding trees (too many = memorising)
    test (20)        -> never touched until the single final score

HOW TO RUN:
    First run steps 1, 3 and 4. Then:   python step5_pattern_learner.py

WHAT IT CREATES:
    outputs/results.csv                     -> adds this contestant to the scoreboard
    outputs/clarke_pattern_learner.png      -> its Clarke Error Grids
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import common
import features

VALIDATION_PATIENTS = 16

# The model's settings. These were chosen BEFORE seeing any test score.
# Changing them to improve the test score would be cheating.
MODEL_SETTINGS = {
    "learning_rate": 0.03,      # how big a correction each new tree makes
    "max_depth": 5,             # how many questions each tree may ask
    "min_child_weight": 20,     # don't make a rule from fewer than about 20 examples
    "subsample": 0.8,           # each tree sees a random 80% of the rows
    "colsample_bytree": 0.8,    # each tree sees a random 80% of the features
    "random_state": common.RANDOM_SEED,
}
MAX_TREES = 3000


def train_model(fit_rows, validation_rows, all_practice_rows, feature_names, horizon):
    """Finds the best number of trees, then trains a final model on all 80 patients."""
    target = f"change_{horizon}"

    # Stage 1: keep adding trees until the validation patients stop improving.
    finder = XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=50, **MODEL_SETTINGS)
    finder.fit(fit_rows[feature_names], fit_rows[target],
               eval_set=[(validation_rows[feature_names], validation_rows[target])],
               verbose=False)
    best_trees = finder.best_iteration + 1
    print(f"  {horizon} min ahead: best number of trees = {best_trees}")

    # Stage 2: now we know how many trees, retrain using all 80 practice patients.
    final = XGBRegressor(n_estimators=best_trees, **MODEL_SETTINGS)
    final.fit(all_practice_rows[feature_names], all_practice_rows[target])
    return final


def main():
    print("Building the inputs...")
    table, feature_names = features.build_feature_table()
    print(f"{len(feature_names)} features for each moment")

    # Only moments with full history and known futures can be used.
    usable = table[table["usable"]].copy()

    # What we're teaching the model to predict: how much sugar will change.
    for horizon in common.HORIZONS:
        usable[f"change_{horizon}"] = usable[f"future_{horizon}"] - usable["cgm"]

    practice = usable[usable["group"] == "train"]
    test = usable[usable["group"] == "test"]

    # Set aside 16 of the 80 practice patients for validation.
    practice_ids = pd.Series(sorted(practice["patient_id"].unique()))
    validation_ids = practice_ids.sample(n=VALIDATION_PATIENTS, random_state=common.RANDOM_SEED)
    is_validation = practice["patient_id"].isin(validation_ids)
    fit_rows = practice[~is_validation]  # "~" means "not"
    validation_rows = practice[is_validation]

    print(f"Learning from {len(fit_rows):,} moments, validating on {len(validation_rows):,}, "
          f"testing on {len(test):,}")
    print("\nTraining...")

    scores = {}
    guesses = {}
    for horizon in common.HORIZONS:
        model = train_model(fit_rows, validation_rows, practice, feature_names, horizon)
        predicted_change = model.predict(test[feature_names])
        guesses[horizon] = test["cgm"].to_numpy() + predicted_change
        scores[horizon] = common.evaluate(test[f"future_{horizon}"], guesses[horizon])

        # Which inputs did the model rely on the most?
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        print(f"  Most useful inputs for {horizon} min ahead:")
        print((importance.head(8) * 100).round(1).to_string())
        print()

    scoreboard = common.save_results("Pattern Learner", scores)
    print("Scoreboard:")
    print(scoreboard.to_string(index=False))

    # ---- Is the win real, or just a few lucky patients? ----
    # We score each of the 20 patients separately against the Lazy Guess.
    print("\nPatient by patient, compared with the Lazy Guess:")
    for horizon in common.HORIZONS:
        truth = test[f"future_{horizon}"].to_numpy()
        results = pd.DataFrame({
            "patient_id": test["patient_id"].to_numpy(),
            "model_error": np.abs(guesses[horizon] - truth),
            "lazy_error": np.abs(test["cgm"].to_numpy() - truth),
        })
        per_patient = results.groupby("patient_id").mean()
        wins = (per_patient["model_error"] < per_patient["lazy_error"]).sum()
        gain = (per_patient["lazy_error"] - per_patient["model_error"]).mean()
        print(f"  {horizon} min ahead: better on {wins} of {len(per_patient)} patients, "
              f"average improvement {gain:.2f} mg/dL per patient")

    # ---- Draw the Clarke Error Grids ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, horizon in zip(axes, common.HORIZONS):
        zone_a = scores[horizon]["zone_A_%"]
        common.draw_clarke_grid(ax, test[f"future_{horizon}"], guesses[horizon],
                                f"Pattern Learner, {horizon} min ahead (Zone A: {zone_a:.1f}%)")
    fig.tight_layout()
    fig.savefig(common.OUTPUT_FOLDER / "clarke_pattern_learner.png", dpi=150)
    print("\nSaved the grids to clarke_pattern_learner.png")


if __name__ == "__main__":
    main()
