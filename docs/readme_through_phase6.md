# Adaptive Smart Room Occupancy & Energy Controller

## Phase 6 complete: simulated energy and adaptive modes

Open **outputs/phase6/report.html** and **docs/phase6_notes.md**.
The energy layer provides ECO/NORMAL/COMFORT and deterministic AUTO selection, with editable
60 W light / 40 W ventilation assumptions. It integrates only short recorded intervals
and excludes gaps, embargo boundaries and unknown durations.

The audit covers 9,950 intervals (84.68 hours), with 119,400 source/policy simulation rows.
Independent verification passed. On covered later-test intervals, baseline AUTO simulates
0.147 kWh versus the assumed 4.930 kWh full-power comparator, but also requests zero lighting
during 1.55 occupied hours. These are assumption-dependent simulations, not measured savings
or evidence of equal comfort.

~~~bash
python src/phase6.py
python src/verify_phase6.py
~~~

The reusable interface is src/energy_controller.py; assumptions are in config/phase6_energy.json.
Phase 7 dashboard, integration and presentation work has not started.

## Phase 5 complete: Mamdani fuzzy smart room controller

Open **outputs/phase5/report.html** and **docs/phase5_notes.md** for membership functions,
the 34-rule base, response plots, numerical examples and viva explanations.
The pipeline supports both baseline and GA occupancy models and returns lighting/ventilation
requests. Configurable assumptions are in config/phase5_fuzzy.json.

Verification passed for 4,459 input combinations; all rules were exercised.
Forty independent analytical centroid comparisons had maximum error below 0.00035 percentage points.
Both integrations and CSV commands passed. Six plots are supplied as PNG and SVG.

The compact estimator-plus-controller requires five channels because temperature is an
additional controller measurement. An explicit exact-off gate follows fuzzy lighting inference.
Occupancy errors still propagate. Thresholds are illustrative, no devices are actuated,
and energy estimation remains Phase 6.

~~~bash
python src/phase5.py
python src/verify_phase5.py
~~~

## Phase 4 complete: optimized occupancy system and GA evaluation

Open **outputs/phase4/report.html** and **docs/phase4_notes.md** for the formal comparison.
The four-input model was independently retrained with unchanged parameters and exactly reproduces
Phase 3 predictions. Input reduction is 73.33%, but validation macro F1 drops from 0.8608 to 0.7952.
Both models miss 184 of 294 occupied later-test rows as empty. Baseline remains the default demo;
GA is the explicit compact alternative.

The common prediction interface is src/occupancy_system.py, with CLI src/predict_phase4.py.
Run python src/phase4.py and python src/verify_phase4.py to reproduce and verify.
Phase 5 is fuzzy control; it has not started. See docs/project_roadmap.md.

## Phase 3 complete: Genetic Algorithm sensor selection

Open **outputs/phase3/report.html** for the result and **docs/phase3_notes.md** for the method and viva.
Selected 4/15 channels: S1_Light, S5_CO2, S6_PIR, S7_PIR.
Validation macro F1: 0.8608 (baseline) to 0.7952 (selected).
Later-test balanced accuracy: 40.83%; true class 1 is absent.
Selection uses training-only purged blocks. External comparisons reuse previously seen data.
No measured hardware/energy savings or global-optimum claim is made.

~~~bash
python src/phase3.py
python src/verify_phase3.py
~~~

## Phase 2 complete: baseline occupancy prediction

Open **outputs/phase2/report.html** for the new results and **docs/phase2_notes.md** for methods
and viva preparation. The 15-input Random Forest, fixed chronological split, dummy comparison,
per-class metrics, confusion matrices and prediction script are saved.

Validation: **94.96% accuracy**, **0.8608 four-class macro F1**.
Later test: **95.32% accuracy but only 38.99% balanced accuracy** over its three present classes.
The always-empty test accuracy is 94.99%. One-person examples are absent from the test.
The model therefore has substantial later-period occupancy-detection weaknesses.

~~~bash
python -m pip install -r requirements.txt
python src/phase2.py
python src/verify_phase2.py
~~~

The exact protocol is in docs/phase2_protocol.json. The trained model and metadata are in
models/phase2. Split assignments are in data/splits/phase2_split_manifest.csv.
Phase 2 has not started GA, fuzzy control or energy estimation.
The Phase 1 material below describes that earlier phase and remains available.

## Phase 1: dataset preparation and exploration

Open **outputs/phase1_report.html** in a browser for the plots and main findings.
Read **docs/phase1_notes.md** for detailed methods, cleaning decisions and viva questions.

The task is occupancy classification: zero, one, two or three people.
Phase 1 only prepares and explores data. Model training, Genetic Algorithm optimization,
fuzzy control and energy simulations have not been performed.

## Main results

- Original UCI dataset: 10,129 rows and 19 columns.
- 16 sensor-related fields, two time fields and one target.
- No missing cells, exact duplicate rows or duplicate timestamps.
- All rows retained; no imputation, clipping, scaling or balancing.
- Added temperature, light and sound averages and a PIR OR indicator.
- Seven calendar dates with recording gaps and uneven class coverage.
- Recommended Phase 2 baseline: 15 direct sensor inputs.
- Optional compact comparison: 6 inputs.
- Supplied CO2 slope retained but conditional on verifying its window calculation.
- Ten EDA plots in both PNG and SVG, with numerical tables.

## Folder layout

~~~text
sc project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    Original ZIP, CSV and source/checksum manifest
│   └── processed/              Full audit table and two candidate feature tables
├── src/
│   ├── download_data.py        Retrieve once and verify originals
│   ├── phase1.py               Cleaning, features, tables, plots and documentation
│   └── verify_phase1.py        Independent validation against original data
├── outputs/
│   ├── phase1_report.html      Visual report
│   ├── plots/                  10 PNGs and 10 vector SVGs
│   └── tables/                 Profiles, statistics, correlations and quality results
├── docs/
│   ├── phase1_notes.md         Findings, methods and viva explanation
│   ├── data_dictionary.md      Every original and added field
│   ├── candidate_features.md   Exact input lists, reasoning and Phase 2 handoff
│   └── feature_spec.json       Machine-readable input/target specification
└── work/                       Local runtime support and visual QA
~~~

## Reproduce the project

Use Python 3.12. A normal local setup:

~~~bash
cd "$HOME/Desktop/sc project"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/download_data.py
python src/phase1.py
python src/verify_phase1.py
~~~

Internet is needed only if the original raw archive is absent. Later runs verify the saved
originals. Mismatched checksums stop the pipeline; raw data is never silently overwritten.
The analysis regenerates processed data, figures, tables and documentation.
Keep personal edits in a separate notes file because generated docs are overwritten.

The delivered run used the bundled Python runtime plus a plotting library installed inside
this project. Actual library versions are in outputs/tables/environment.json.
The scripts also run in a normal virtual environment using requirements.txt.

## Processed data

**occupancy_cleaned.csv** has 26 columns: all original fields plus timestamp, source row,
recording segment and four aggregate features. It is the complete audit table.

**baseline_features.csv** contains the 15 recommended inputs plus the target.
**compact_features.csv** contains the six alternative inputs plus the target.
All three tables have the same chronological row order.

~~~python
import json
import pandas as pd

spec = json.load(open("docs/feature_spec.json"))
table = pd.read_csv("data/processed/baseline_features.csv")
X = table[spec["recommended_baseline"]]
y = table[spec["target"]]
audit = pd.read_csv("data/processed/occupancy_cleaned.csv", parse_dates=["Timestamp"])
~~~

This example separates inputs and labels only. No split or model is built.
Do not feed the target, dates, source row or segment identifier to the classifier.
Read docs/candidate_features.md before selecting the Phase 2 evaluation strategy.

## How to read the figures

Boxplots compare sensor distributions across classes. Boxes cover the middle 50% and the
line is the median. Dots beyond whiskers are retained unless separately shown to be faults.
PIR bars show motion-detection rates within each class.
Correlation is association, not causation or prediction accuracy.

## Source and attribution

Singh, A. & Chaudhari, S. (2018). Room Occupancy Estimation [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5P605
Dataset: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

No measured electricity labels are supplied. No measured energy savings are claimed.
