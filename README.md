# Predicting blood sugar in Type 2 diabetes

A project on **glucometers and rapid test kits**. It explains the physics, chemistry and biology behind a glucometer, and then tests a real question with real patient data:

> Can blood sugar be predicted 30 and 60 minutes ahead for Type 2 patients the computer has never seen, and does adding the body's equations help?

The full write-up is in [`report/report.pdf`](report/report.pdf).

## The four contestants

| Method | What it does |
|---|---|
| **Lazy Guess** | Assumes sugar will not change. Every other method has to beat this. |
| **Equations** | The Bergman minimal model of glucose and insulin, with gut and insulin absorption, fitted to the practice patients. |
| **Pattern Learner** | An XGBoost model that learns from 34 inputs: recent sugar, meals, insulin, tablets, time of day and patient lab results. |
| **Hybrid** | The Pattern Learner plus the Equations' predicted change as two extra inputs. |

All models learned from 80 patients and were tested once on 20 different patients (19,813 moments). Splitting by patient, not by time, stops a model from scoring well just by memorising someone it has already seen.

## Results

Error is the mean absolute error in mg/dL. Zones come from the Clarke Error Grid: Zone A is clinically accurate, and Zone D means a dangerous level was missed.

| Method | 30 min error | 30 min Zone A | 30 min Zone D | 60 min error | 60 min Zone A | 60 min Zone D |
|---|---:|---:|---:|---:|---:|---:|
| Lazy Guess | 12.42 | 90.78% | 0.51% | 21.46 | 74.63% | 1.59% |
| Equations | 11.56 | 92.62% | **0.48%** | 19.22 | 79.09% | **1.48%** |
| Pattern Learner | 9.37 | 95.29% | 0.76% | 17.01 | 81.81% | 1.98% |
| Hybrid | **9.20** | **95.91%** | 0.63% | **16.51** | **82.74%** | 1.96% |

- The Hybrid was the most accurate and beat the Lazy Guess on all 20 test patients.
- Its gain over the Pattern Learner alone was small: 0.17 mg/dL at 30 minutes (better on 14 of 20 patients, Wilcoxon p = 0.027) and 0.47 mg/dL at 60 minutes (15 of 20, p = 0.002).
- The Equations missed the fewest dangerous readings. When sugar was really below 70 mg/dL, the machine-learning models guessed too high on average, which is the risky direction.

The first version of the Equations did worse than the Lazy Guess. It is kept in [`outputs/results.csv`](outputs/results.csv) as "Equations v1 (first attempt)". It was then fixed using the practice patients only, but the 20 test patients were used twice for the Equations, so treat their score with some caution.

## How to run it

1. Install Python 3 and the libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Download `diabetes_datasets.zip` from the ShanghaiT2DM collection on Figshare: <https://doi.org/10.6084/m9.figshare.c.6310860>
3. Unzip it so these paths exist:
   ```
   data/shanghai_raw/extracted/Shanghai_T2DM/            (the 109 patient files)
   data/shanghai_raw/extracted/Shanghai_T2DM_Summary.xlsx
   ```
4. Run the steps in order from this folder:

| Script | What it does |
|---|---|
| `step1_build_csv.py` | Combines the 109 Excel files into clean CSV tables |
| `step2_plot_patient.py` | Plots a few days of one patient's data |
| `step3_estimate_carbs.py` | Estimates the carbohydrates in each meal from its food list |
| `step4_split_and_lazy_guess.py` | Splits patients 80/20 and scores the Lazy Guess |
| `step5_pattern_learner.py` | Trains and scores the Pattern Learner |
| `step6_equations.py` | Fits and scores the Equations |
| `step7_hybrid.py` | Trains the Hybrid and compares all four, patient by patient |

`common.py` and `features.py` hold shared code. The scripts are heavily commented so they can be read as a tutorial. Results and graphs are written to `outputs/`.

To rebuild the report, run `pdflatex report.tex` twice inside `report/`.

## Limitations

- Only 20 test patients, all from one region of China.
- Meal carbohydrates are rough estimates from food descriptions.
- The fitted equation values are not real body measurements.
- **This is a research project, not a medical device.** Do not use it to make treatment decisions.

## Data

This project uses the ShanghaiT2DM dataset from Zhao Q. et al., "Chinese diabetes datasets for data-driven machine learning", *Scientific Data* 10, 35 (2023). The data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is **not** included in this repository, so download it from the link above.

## Licence

**Copyright (c) 2026 Aarav Gupta. All rights reserved.**

No licence is granted for the code, the report, the figures or the results. You may not use, copy, modify or share any part of this repository without the author's prior written permission. Anyone given permission must credit the author. See [LICENSE](LICENSE) for the full terms, and email gupta.aarav0910@gmail.com to ask for permission.

## Acknowledgement of AI use

I used Claude (Anthropic) to aid in writing the code and drafting the report.
