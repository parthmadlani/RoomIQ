# RoomIQ — Adaptive Smart Room Occupancy & Energy Controller

**RoomIQ is a local dashboard that estimates room occupancy, explains lighting and ventilation requests, and calculates simulated energy use.**

It is a Soft Computing minor project built in seven phases. You do not need to know machine learning or fuzzy logic to start reading this guide. Begin with the idea below, then follow the phases to understand how we made it.

The website is called **RoomIQ**. The longer name above describes the academic project.

## 1. What problem are we trying to solve?

Imagine a classroom with lights and ventilation running at full power throughout the day. Sometimes it is empty; sometimes several people are inside. Its lighting and ventilation needs change with occupancy, brightness, temperature and CO₂.

Our project asks two questions:

1. Can environmental sensor readings help estimate how many people are in a room?
2. Given that estimate and the room conditions, what lighting and ventilation requests should the system produce?

For example, an empty room may not need lighting. An occupied but already bright room may need little artificial lighting. Elevated CO₂ may call for ventilation even when the occupancy model says the room is empty.

**What we have built:** a software prototype using recorded data and editable examples. It does not currently receive live hardware readings, switch real devices or measure electricity consumption. Its energy results are estimates based on explicit assumptions.

## 2. The whole project in one view

There are two stages: building the system, and using the finished system.

### Development: learn and compare models

~~~text
UCI recorded room data
        ↓
Inspect, clean and explore
        ↓
Separate training, validation and later-test periods
        ↓
Train a baseline occupancy model with 15 inputs
        ↓
Use a Genetic Algorithm to search smaller input subsets
        ↓
Train and compare the selected-input model
        ↓
Save both models and their evaluation results
~~~

### Dashboard use: make one decision

~~~text
Sensor readings
        ↓
Saved occupancy model → predicted people: 0, 1, 2 or 3
        ↓
Predicted occupancy + temperature + light + CO₂
        ↓
Mamdani fuzzy controller
        ↓
Lighting request (%) + ventilation request (%)
        ↓
ECO / NORMAL / COMFORT / AUTO energy policy
        ↓
Final requests → assumed power (W) → simulated energy (kWh)
~~~

The Genetic Algorithm runs during development. It does **not** run each time you click “Run simulation.” Opening the dashboard loads saved models; it does not retrain them.

## 3. What data did we use?

We used the **Room Occupancy Estimation** dataset from the UCI Machine Learning Repository. The original files are preserved in [data/raw](data/raw).

| Item | Meaning |
|---|---|
| Observations | 10,129 recorded rows |
| Original columns | 19 |
| Target | Room_Occupancy_Count: the known number of people |
| Possible target values | 0, 1, 2 and 3 |
| Input measurements | Temperature, light, sound, CO₂ and PIR motion |
| Time information | Date and Time |
| Electricity labels | Not supplied |

An **input feature** is a value available to the model, such as a light reading. The **target** is the answer it learns to predict. The known target must never be passed to the model as an input.

### How to read the sensor names

| Columns | What they represent |
|---|---|
| S1_Temp to S4_Temp | Four temperature readings, in °C |
| S1_Light to S4_Light | Four light readings, in lux |
| S1_Sound to S4_Sound | Four sound-sensor readings, in volts—not decibels |
| S5_CO2 | CO₂ concentration, in ppm |
| S5_CO2_Slope | A supplied rate-of-change feature |
| S6_PIR, S7_PIR | Passive infrared motion indicators, 0 or 1 |
| Date, Time | When a reading was recorded |
| Room_Occupancy_Count | Known occupancy label |

The S-number identifies a sensor node; a node can supply multiple measurement channels. Reducing the number of input columns is therefore different from removing that many physical devices.

The class counts are:

| People | Rows |
|---:|---:|
| 0 | 8,228 |
| 1 | 459 |
| 2 | 748 |
| 3 | 694 |

About **81.23% of all rows describe an empty room**. This imbalance is important when interpreting model accuracy.

The timestamps cover seven calendar dates in December 2017 and January 2018, with recording gaps. We use the actual timestamps rather than assuming every row represents an uninterrupted 30-second interval.

## 4. How we built it: the seven phases

### Phase 1 — Understand and prepare the data

We inspected every column, its data type, missing values, duplicates, basic statistics and relationship with occupancy.

No missing cells, exact duplicate rows or duplicate timestamps were found. All 10,129 rows were retained; no imputation or outlier removal was required. We preserved the raw measurements and created additional fields for analysis:

- **Timestamp:** combines date and time.
- **Source_Row:** links a processed observation back to its original CSV row.
- **Recording_Segment:** separates recordings across long gaps.
- **Avg_Temp, Avg_Light, Avg_Sound:** averages across the four corresponding channels.
- **PIR_Any:** 1 if either motion sensor detects motion.

These aggregates make exploratory plots easier to interpret. They were not silently added to the trained baseline: its inputs are the 15 direct measurement channels.

We excluded the supplied CO₂ slope from modelling until its calculation window can be verified. A feature calculated using future readings would give the model information it would not have at prediction time.

We produced occupancy-distribution plots, sensor-versus-occupancy plots and a correlation view. Correlation shows association; it does not prove causation or predictive reliability.

**Main files:** [src/phase1.py](src/phase1.py), [data dictionary](docs/data_dictionary.md), [candidate features](docs/candidate_features.md), [visual report](outputs/phase1_report.html).

### Phase 2 — Build a baseline occupancy model

A **baseline** is the first reference model against which later changes are compared.

We used a **Random Forest classifier** with all 15 direct inputs. A Random Forest combines many decision trees. Each tree learns rules from training examples; their combined output determines the predicted occupancy class.

Our forest uses 300 trees, a fixed random seed for repeatability and class weighting to help account for unequal class frequencies. Class weighting does not guarantee good detection of occupied rooms.

We split the data chronologically:

| Part | Purpose | Rows |
|---|---|---:|
| Training: December 22 | Fit the model | 1,462 |
| Validation: December 23 after the boundary buffer | Compare development choices | 2,760 |
| Later test: December 24 after the buffer onward | Assess performance on later observations | 5,870 |
| Boundary buffer, called an embargo | Excluded from fitting/evaluation partitions | 37 |

A ten-minute buffer reduces immediate adjacency across boundaries. It does not make the data fully independent. Earlier exploration had already examined the complete dataset, so the later test is held out from model fitting, but is not a completely unseen research dataset.

We also compared against a deliberately simple classifier that always predicts “empty.”

**Main files:** [src/phase2.py](src/phase2.py), [protocol](docs/phase2_protocol.json), [report](outputs/phase2/report.html).

### Phase 3 — Use a Genetic Algorithm to select inputs

A **Genetic Algorithm (GA)** is a search method inspired by evolution. Here, it searches for a useful smaller set of input channels.

Each candidate is a 15-bit **chromosome**: one bit per baseline input.

- 1 means keep that input.
- 0 means leave it out.

A **population** is a collection of candidates. We evaluate candidates, select stronger ones, combine them through **crossover**, and randomly flip some bits through **mutation**. A small number of strong candidates survive unchanged through **elitism**.

The fitness score balances prediction quality against input count:

~~~text
fitness = training-block macro F1 − 0.05 × (selected inputs / 15)
~~~

Higher fitness is better. Macro F1 gives equal weight to the classes; the second term discourages using unnecessary inputs.

Our search used 16 candidates per population, 10 generations, three random seeds, tournament selection, one-point crossover and a mutation probability of 1/15 per bit. Search models used 80 trees; the final selected-input model used 300.

Fitness was computed only within the training period, using four held-out blocks and ten-minute exclusion buffers around those blocks. Training rows could be on both sides of a block: this evaluates within-day interpolation, not future forecasting.

The selected inputs were:

~~~text
S1_Light
S5_CO2
S6_PIR
S7_PIR
~~~

This is **4 out of 15 estimator inputs**, a 73.33% reduction. It is not a measured reduction in electricity, hardware cost or physical device count. A finite GA search also does not guarantee the globally best subset.

**Main files:** [src/phase3.py](src/phase3.py), [GA protocol](docs/phase3_protocol.json), [locked selection](models/phase3/selection_locked.json), [report](outputs/phase3/report.html).

### Phase 4 — Retrain, compare and package the models

We retrained the model using the locked four-input selection and confirmed that it reproduced Phase 3 predictions and probabilities.

We then created a common prediction interface so later components can explicitly choose:

- **Baseline:** 15 estimator inputs.
- **GA:** four estimator inputs.

The saved registry records model paths, expected inputs and checksums. Input checks reject missing or invalid measurements. The dashboard does not silently substitute another model.

The baseline remains the default because it has better validation macro F1. “Optimized” refers to the GA's training objective, not a guaranteed improvement on later data.

**Main files:** [src/occupancy_system.py](src/occupancy_system.py), [model registry](models/phase4/model_registry.json), [report](outputs/phase4/report.html).

### Phase 5 — Add the fuzzy controller

A normal yes/no rule might say “temperature is hot above a fixed cutoff.” **Fuzzy logic** allows gradual membership: a temperature can partly belong to “comfortable” and partly to “hot.”

A membership degree lies between 0 and 1. It expresses how strongly a value belongs to a defined category; it is not the probability that the room is hot.

Our controller receives:

| Controller input | Where it comes from |
|---|---|
| Occupancy | Model prediction, or explicit manual scenario |
| Temperature | S1_Temp |
| Light | S1_Light |
| CO₂ | S5_CO2 |

It uses **34 rules: seven lighting rules and 27 ventilation rules**. Examples include:

- If occupancy is high and light is dark, request high lighting.
- If CO₂ is poor, request high ventilation, including when occupancy is empty.

The Mamdani process is:

1. **Fuzzification:** calculate membership degrees.
2. **Rule evaluation:** use the minimum degree for conditions joined by AND.
3. **Aggregation:** clip each rule's output set and combine contributions using the maximum.
4. **Defuzzification:** calculate the centroid, or balance point, to obtain one output percentage.

A fuzzy “off” set can have a small positive centroid. We therefore apply an explicit extra rule after inference: **zero occupancy means exactly zero lighting**. Bright occupied scenarios can still receive a small nonzero lighting request.

Inputs outside the environmental membership ranges are clipped and disclosed in the trace. Membership shapes and thresholds are illustrative project choices, not validated building or air-quality standards.

The complete GA estimator-plus-controller needs **five** channels: its four selected inputs plus S1_Temp for fuzzy control.

**Main files:** [src/fuzzy_controller.py](src/fuzzy_controller.py), [memberships and rules](config/phase5_fuzzy.json), [report](outputs/phase5/report.html).

### Phase 6 — Add energy policies and estimate consumption

The fuzzy controller produces percentages. The energy layer adjusts them according to a policy:

| Policy | Behaviour |
|---|---|
| ECO | Multiply both fuzzy requests by 0.8 |
| NORMAL | Keep the fuzzy requests |
| COMFORT | Multiply both by 1.15, capped at 100% |
| AUTO | Choose one of these modes from the current conditions |

AUTO chooses COMFORT if CO₂ is at least 1,100 ppm, occupancy is at least two, or the room is occupied and temperature is at least 28 °C. Otherwise it chooses NORMAL when occupied and ECO when empty.

At elevated CO₂, ventilation cannot be reduced below the original fuzzy request, even under ECO. Zero occupancy still forces lighting to zero. These are illustrative rules, not a safety guarantee.

**“Adaptive” means responding to the current inputs using rules. It does not mean the system learns online or updates its model while running.**

For energy calculations, we assume:

~~~text
Full lighting power = 60 W
Full ventilation power = 40 W

Lighting power = 60 × lighting request / 100
Ventilation power = 40 × ventilation request / 100

Energy in kWh = total power in watts × duration in seconds / 3,600,000
~~~

For example, 50% lighting and 25% ventilation would mean 30 W + 10 W = 40 W. For 30 minutes, that is 0.020 kWh under this linear assumption.

The comparison is both loads at full power over the same duration. It is an assumed reference, not measured conventional consumption.

For dataset replay, we hold each reading's request until the next reading only when the interval is at most 60 seconds and stays inside the same data partition. We exclude recording gaps, embargo crossings and unknown final durations. Across the eligible data, this covers 9,950 intervals and about 84.68 hours.

There is no simulated feedback showing ventilation reducing later CO₂ or lighting changing later measured brightness. This is a replay of recorded conditions, not a closed-loop building simulator. HVAC, standby, sensor and processing electricity are excluded.

**Main files:** [src/energy_controller.py](src/energy_controller.py), [power and policy assumptions](config/phase6_energy.json), [report](outputs/phase6/report.html).

### Phase 7 — Build the RoomIQ dashboard and presentation

We connected the saved components through a local Python server and a browser interface built with HTML, CSS and JavaScript. No additional web framework is required.

| Dashboard view | What a newcomer can learn |
|---|---|
| Room simulator | Enter readings, choose a model and inspect a complete result |
| Models & GA | Compare quality, input count and GA evolution |
| Fuzzy rulebook | Inspect membership functions and all 34 rules |
| Energy evidence | Compare replay estimates with occupied zero-light hours |
| Viva walkthrough | Follow a presentation sequence and download the slides |

There are two input modes:

- **Sensor readings:** the saved model predicts occupancy.
- **Manual scenario:** you supply occupancy directly, bypassing the classifier, to explain the fuzzy controller.

The six recorded examples include validation samples and disclosed later-test failures. A known label shown as provenance is not passed into the predictor.

The simulator duration assumes constant conditions for the chosen number of minutes. It is separate from the timestamp-based replay calculations. You can save a result as JSON, a readable structured record of inputs and outputs.

**Main files:** [src/dashboard_server.py](src/dashboard_server.py), [dashboard](dashboard), [Phase 7 notes](docs/phase7_notes.md), [presentation](outputs/phase7/presentation.pptx), [viva guide](docs/presentation_guide.md).

## 5. A complete example you can explain

Choose **Manual scenario → High occupancy + high CO₂**, with AUTO and 30 minutes.

~~~text
Supplied occupancy: 3
Temperature: 32 °C
Light: 20 lux
CO₂: 1,600 ppm
~~~

The classifier is bypassed in this example. The fuzzy rules produce approximately 90% lighting and 90% ventilation. AUTO selects COMFORT; multiplying by 1.15 and capping at 100% gives 100% for both final requests.

That corresponds to 60 W + 40 W = 100 W. Over 30 minutes:

~~~text
100 × 1,800 / 3,600,000 = 0.050 kWh
~~~

Now try **Empty + dark**. The exact-off gate makes lighting zero; AUTO chooses ECO and ventilation remains a small nonzero request. Empty occupancy alone does not mean every load becomes zero.

Finally, load a **recorded false-empty example** in sensor mode. Its known occupancy is three, but the model predicts zero. The lighting gate then switches the simulated lighting request off. This demonstrates why prediction errors matter downstream.

## 6. How to understand the results honestly

### What the metrics mean

| Metric | Beginner explanation |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision for a class | Of predictions saying this class, how many were correct? |
| Recall for a class | Of actual examples in this class, how many did we find? |
| F1 | A score combining precision and recall |
| Macro F1 | Average class F1, giving each class equal weight |
| Balanced accuracy | Average recall across the evaluated classes |
| Confusion matrix | Table showing which true classes became which predictions |
| False empty | An occupied observation incorrectly predicted as empty |

### Saved model comparison

| Result | Baseline | GA selected |
|---|---:|---:|
| Estimator inputs | 15 | 4 |
| Validation accuracy | 94.96% | 92.79% |
| Validation macro F1 | 0.8608 | 0.7952 |
| Later-test accuracy | 95.32% | 95.37% |
| Later-test balanced accuracy | 38.99% | 40.83% |
| Occupied later-test rows predicted empty | 184 of 294 | 184 of 294 |

The later test contains no true one-person examples. Its balanced accuracy concerns the three present classes and cannot demonstrate one-person detection.

An always-empty model achieves about **94.99% test accuracy** because this period is overwhelmingly empty. Therefore, the approximately 95% accuracy of our models is not enough to establish reliable occupancy detection.

The GA reduces input count, but validation macro F1 gets worse. That tradeoff is a project finding, not something to conceal.

### Energy results need the failure figures beside them

For covered later-test intervals under AUTO, both models produce approximately:

| Quantity | Value |
|---|---:|
| Covered duration | 49.30 hours |
| Simulated consumption | 0.147 kWh |
| Assumed full-power reference | 4.930 kWh |
| Occupied time within coverage | 2.48 hours |
| Occupied time receiving zero lighting | 1.55 hours |

The apparent reduction is about 97%, but **we cannot claim 97% measured energy savings or equal comfort**. Much of the room is empty, and missed occupancy also reduces simulated lighting use.

Both models give the same test AUTO energy requests because their remaining occupancy differences fall within the same fuzzy high-occupancy region. Sensor power is not included.

The dashboard also shows a **true-label reference**: it runs the controller using known occupancy labels. This is a counterfactual comparison, not a deployable predictor or proof of optimal control.

## 7. Run RoomIQ

### Get the code

~~~bash
git clone https://github.com/parthmadlani/RoomIQ.git
cd RoomIQ
~~~

The repository already contains the dataset, trained models and saved results, so the dashboard runs without retraining.

### First-time setup and launch

Use Python 3.12. Open a terminal inside the project folder.

macOS/Linux:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/dashboard_server.py --open
~~~

Windows PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/dashboard_server.py --open
~~~

A virtual environment keeps this project's Python packages separate from other projects. The requirements file records the package versions used.

The dashboard opens at [http://127.0.0.1:8765](http://127.0.0.1:8765). Keep the terminal window open, and press Ctrl+C in it to stop the server. Open the server URL rather than opening dashboard/index.html directly, and refresh the browser after changing website files.

### Mac shortcut

After the setup above, you can double-click **Start Dashboard.command** in Finder. It uses the project's .venv; if that environment is missing, it prints the setup commands instead of starting. If macOS blocks it the first time, right-click it and choose **Open**.

Internet is needed for initial dependency installation or retrieving missing raw data. The existing dashboard runs locally without internet once dependencies, data and models are available.

If port 8765 is already occupied:

~~~bash
python src/dashboard_server.py --port 8766 --open
~~~

If a module is missing, install requirements using the same Python environment used to launch the app. If saved models or outputs are missing, restore them or run the reproduction steps below. The browser cannot run predictions if the server terminal has been closed.

The server listens only on this computer. It is not a publicly hosted website.

## 8. Reproduce the work from the preserved data

You do **not** need to retrain to use the dashboard.

To regenerate the analysis, run these commands in order from the project folder, using the environment configured above:

~~~bash
python src/download_data.py
python src/phase1.py
python src/verify_phase1.py
python src/phase2.py
python src/verify_phase2.py
python src/phase3.py
python src/verify_phase3.py
python src/phase4.py
python src/verify_phase4.py
python src/phase5.py
python src/verify_phase5.py
python src/phase6.py
python src/verify_phase6.py
python src/verify_phase7.py
python src/dashboard_server.py --open
~~~

The download step verifies preserved originals when already present. The raw dataset must not be edited. GA search takes longer than the other stages.

Reproduction scripts regenerate derived outputs and some phase documentation. Keep personal notes in separate files. Fixed seeds, saved protocols, versioned dependencies and checksums help make the results repeatable.

The PPTX is a separately authored presentation, not regenerated by the Python phase scripts. Edit the slides directly in PowerPoint or Keynote.

## 9. What is inside the project folder?

~~~text
RoomIQ/
├── README.md                    Start here
├── Start Dashboard.command      Mac launcher
├── requirements.txt             Python package versions
├── config/                      Fuzzy rules and energy assumptions
├── data/
│   ├── raw/                     Original dataset and checksums
│   ├── processed/               Cleaned data and derived features
│   ├── splits/                  Training/validation/test assignments
│   └── examples/                Demo inputs and provenance
├── models/                      Saved estimators and selection metadata
├── src/                         Phase scripts, reusable logic and checks
├── dashboard/                   Website HTML, CSS and JavaScript
├── outputs/                     Reports, plots, tables and presentation
└── docs/                        Detailed phase explanations and viva notes
~~~

Core technologies:

| Technology | Its role |
|---|---|
| Python | Data processing, modelling, control calculations and local server |
| pandas / NumPy | Tables and numerical calculations |
| scikit-learn | Random Forest classifiers and evaluation |
| Custom Python GA | Binary input selection |
| Custom NumPy fuzzy logic | Memberships, Mamdani rules and centroid calculations |
| Matplotlib | Research plots |
| joblib | Saving and loading trained models |
| HTML / CSS / JavaScript | RoomIQ interface |

Install the requirements in a fresh virtual environment on each machine rather than copying an installed environment between computers.

## 10. What has been checked?

The recorded Phase 1–6 verification runs passed. Phase 7 passed **72 integration checks**, including 48 recorded example/model/policy combinations, four teaching scenarios, invalid input rejection and HTTP interface checks.

Evidence is saved in [verification.json](outputs/phase7/verification.json) and [regression.json](outputs/phase7/regression.json).

The presentation passed package, layout/font and import checks; all ten rendered slides were visually inspected. Native PowerPoint execution is not claimed.

A browser check was completed after the Phase 7 handoff, on a desktop-sized window and a 375-pixel phone-width view. It covered all five tabs, both input modes, both models, the false-empty examples, energy filters, JSON result generation and narrow-screen layout. Details are in [browser_verification.json](outputs/phase7/browser_verification.json). Still check readability at your projector's resolution and open the PPTX once in PowerPoint or Keynote before presenting.

All verification scripts also pass in a fresh .venv created from requirements.txt.

**Passing software checks means the implementation behaved as checked. It does not prove good predictions on new rooms, real electricity savings or acceptable occupant comfort.**

## 11. Explain it to your professor in one minute

> “RoomIQ estimates whether a room contains zero, one, two or three people using environmental sensor readings. First, we preserved and analysed the UCI dataset. Then we trained a Random Forest baseline and used a Genetic Algorithm to search for a smaller input subset. The selected model uses four inputs, but its validation score is lower, so we retain the baseline as the default. A Mamdani fuzzy controller uses predicted occupancy, temperature, light and CO₂ to calculate lighting and ventilation requests. A rule-based energy layer adjusts those requests and estimates energy using declared power assumptions. The dashboard exposes the memberships, fired rules, model results and failure cases. Energy is simulated, and our occupied-room detection still needs improvement before any real deployment.”

For a longer demonstration, use the [seven-minute presentation guide](docs/presentation_guide.md) and [10-slide presentation](outputs/phase7/presentation.pptx).

## 12. Limitations and sensible next steps

The dataset comes from one recording setup, with strong class imbalance and limited time coverage. Previously examined data are reused in model comparisons. The GA is a heuristic search; its result is not a global optimum. Fuzzy memberships and energy policies are manually specified. AUTO is stateless and has no temporal smoothing or online learning.

The next useful work is to collect fresh data from other rooms and occupied periods, improve occupied recall, evaluate time-based safeguards against abrupt false-empty decisions, and measure actual device power and comfort. Those changes need fresh evaluation rather than simply trying to make a savings percentage larger.

## Dataset credit and further reading

Singh, A. & Chaudhari, S. (2018). **Room Occupancy Estimation [Dataset]**.
[UCI dataset and DOI](https://doi.org/10.24432/C5P605).
Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Detailed methods are in docs/phase1_notes.md through docs/phase7_notes.md.
The old phase-by-phase README is preserved in docs/readme_through_phase6.md.
Historical documents may describe the stopping point of their own phase; this README explains the integrated project.
