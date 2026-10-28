# Phase 5 — Fuzzy Smart Room Controller

## What was built

A static Mamdani controller transforms occupancy, temperature, light and CO2 into requested
lighting and ventilation percentages. There are seven lighting rules and 27 ventilation
rules. The implementation is handwritten NumPy so every step can be explained.

It supports standalone scenario inputs and integration with either Phase 4 occupancy model.
No electricity calculation, hardware actuation, adaptive mode or final dashboard is included.
Those belong to later phases. All thresholds are illustrative design assumptions; none are
learned from control labels, calibrated against equipment, or represented as standards.

## Inputs, outputs and units

Occupancy: 0–3 people. Temperature: 10–40 degrees C. Light: 0–1000 lux.
CO2: 300–2000 ppm. Both outputs use a 0–100 requested-level scale.
These output percentages are not watts, fan airflow or demonstrated comfort/air-quality outcomes.

In integrated mode, occupancy comes from the chosen classifier.
Temperature is S1_Temp, light is S1_Light, and CO2 is S5_CO2.
These are explicit local measurements, not room-wide averages. Using S1_Light avoids pretending
that a four-input estimator also receives all four light sensors.

The baseline estimator needs its original 15 channels.
The GA estimator needs S1_Light, S5_CO2, S6_PIR and S7_PIR. Its controller additionally needs
S1_Temp. Therefore the compact complete estimator-plus-controller path requires **five channels**,
not four (66.67% fewer input channels than the 15-channel path, not measured power savings).
Temperature and light share source node S1; distinct node dependencies remain four.
A supplied target column is ignored by the prediction interface.

## Membership definitions

Each function is stored as [a, b, c, d]: zero outside [a,d], rises from a to b, stays at one
from b to c, and falls from c to d. b=c gives a triangle. Repeated left/right endpoints give
shoulders without division by zero. The full definitions are:

| kind | variable | unit | term | a | b | c | d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inputs | occupancy | people | empty | 0 | 0 | 0 | 0.5 |
| inputs | occupancy | people | low | 0 | 1 | 1 | 2 |
| inputs | occupancy | people | high | 1 | 2 | 3 | 3 |
| inputs | temperature | degrees C | cool | 10 | 10 | 18 | 23 |
| inputs | temperature | degrees C | comfortable | 20 | 24 | 24 | 28 |
| inputs | temperature | degrees C | hot | 26 | 30 | 40 | 40 |
| inputs | light | lux | dark | 0 | 0 | 50 | 150 |
| inputs | light | lux | adequate | 50 | 200 | 200 | 350 |
| inputs | light | lux | bright | 250 | 400 | 1000 | 1000 |
| inputs | co2 | ppm | fresh | 300 | 300 | 550 | 800 |
| inputs | co2 | ppm | moderate | 600 | 950 | 950 | 1300 |
| inputs | co2 | ppm | poor | 1100 | 1500 | 2000 | 2000 |
| outputs | lighting | percent | off | 0 | 0 | 0 | 10 |
| outputs | lighting | percent | dim | 0 | 15 | 15 | 30 |
| outputs | lighting | percent | medium | 25 | 50 | 50 | 75 |
| outputs | lighting | percent | high | 70 | 100 | 100 | 100 |
| outputs | ventilation | percent | idle | 0 | 0 | 0 | 10 |
| outputs | ventilation | percent | low | 5 | 25 | 25 | 45 |
| outputs | ventilation | percent | medium | 30 | 55 | 55 | 80 |
| outputs | ventilation | percent | high | 70 | 100 | 100 | 100 |

The labels fresh/moderate/poor describe this demonstration's CO2 fuzzy sets, not certified
air-quality categories. Values above/below the supported environmental universes are clipped
and each clipping is recorded. Nonfinite inputs and negative light/CO2 are rejected.
Occupancy outside 0–3 is rejected. Fractional occupancy is permitted for standalone rule
exploration; integrated classifier output is an integer.

## Rules and inference

| id | output | antecedent | consequent |
| --- | --- | --- | --- |
| L01 | lighting | occupancy is empty | off |
| L02 | lighting | occupancy is low AND light is dark | medium |
| L03 | lighting | occupancy is low AND light is adequate | dim |
| L04 | lighting | occupancy is low AND light is bright | off |
| L05 | lighting | occupancy is high AND light is dark | high |
| L06 | lighting | occupancy is high AND light is adequate | medium |
| L07 | lighting | occupancy is high AND light is bright | off |
| V01 | ventilation | occupancy is empty AND co2 is fresh AND temperature is cool | idle |
| V02 | ventilation | occupancy is empty AND co2 is fresh AND temperature is comfortable | idle |
| V03 | ventilation | occupancy is empty AND co2 is fresh AND temperature is hot | idle |
| V04 | ventilation | occupancy is empty AND co2 is moderate AND temperature is cool | medium |
| V05 | ventilation | occupancy is empty AND co2 is moderate AND temperature is comfortable | medium |
| V06 | ventilation | occupancy is empty AND co2 is moderate AND temperature is hot | medium |
| V07 | ventilation | occupancy is empty AND co2 is poor AND temperature is cool | high |
| V08 | ventilation | occupancy is empty AND co2 is poor AND temperature is comfortable | high |
| V09 | ventilation | occupancy is empty AND co2 is poor AND temperature is hot | high |
| V10 | ventilation | occupancy is low AND co2 is fresh AND temperature is cool | low |
| V11 | ventilation | occupancy is low AND co2 is fresh AND temperature is comfortable | low |
| V12 | ventilation | occupancy is low AND co2 is fresh AND temperature is hot | high |
| V13 | ventilation | occupancy is low AND co2 is moderate AND temperature is cool | medium |
| V14 | ventilation | occupancy is low AND co2 is moderate AND temperature is comfortable | medium |
| V15 | ventilation | occupancy is low AND co2 is moderate AND temperature is hot | high |
| V16 | ventilation | occupancy is low AND co2 is poor AND temperature is cool | high |
| V17 | ventilation | occupancy is low AND co2 is poor AND temperature is comfortable | high |
| V18 | ventilation | occupancy is low AND co2 is poor AND temperature is hot | high |
| V19 | ventilation | occupancy is high AND co2 is fresh AND temperature is cool | medium |
| V20 | ventilation | occupancy is high AND co2 is fresh AND temperature is comfortable | medium |
| V21 | ventilation | occupancy is high AND co2 is fresh AND temperature is hot | high |
| V22 | ventilation | occupancy is high AND co2 is moderate AND temperature is cool | high |
| V23 | ventilation | occupancy is high AND co2 is moderate AND temperature is comfortable | high |
| V24 | ventilation | occupancy is high AND co2 is moderate AND temperature is hot | high |
| V25 | ventilation | occupancy is high AND co2 is poor AND temperature is cool | high |
| V26 | ventilation | occupancy is high AND co2 is poor AND temperature is comfortable | high |
| V27 | ventilation | occupancy is high AND co2 is poor AND temperature is hot | high |

Every rule's AND uses MIN. A rule clips its output membership function at its firing strength.
All rule contributions to an output are combined with MAX.
The centroid is integral(x * aggregated_membership) / integral(aggregated_membership).
The implementation uses trapezoidal numerical integration on a grid spaced 0.1 percentage points.
A zero-area result is an error, not an invented fallback.

For lighting, more occupancy or less light can increase the request.
For ventilation, poor CO2 requests high ventilation even when occupancy is empty.
Hot occupied rooms request high ventilation. Empty rooms with fresh CO2 use the idle set,
even if hot; this is an explicit occupancy-based design preference, not a thermal model.
All occupancy/CO2/temperature combinations have a ventilation rule.

A fuzzy set named off still has area above zero and hence a positive centroid (~3.33%).
An **explicit postprocessing gate** sets final lighting to exactly zero when the supplied
occupancy is exactly zero. The trace retains the raw centroid separately. This exact-off
gate is deterministic logic after Mamdani inference and introduces a boundary discontinuity
near occupancy zero. Bright occupied rooms may retain the small off-set centroid.
High ventilation generally defuzzifies near 90%, not 100%; that is a consequence of its shape.
There is no unreported output rounding or physical relay threshold.

## Scenarios

| scenario | occupancy | temperature | light | co2 | lighting_percent | ventilation_percent | raw_lighting_centroid | exact_off | clipped_inputs | fired_rules |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| empty_dark_fresh | 0 | 24 | 0 | 450 | 0 | 3.333 | 3.333 | True | 0 | 2 |
| empty_dark_poor_co2 | 0 | 24 | 0 | 1600 | 0 | 90 | 3.333 | True | 0 | 2 |
| one_person_dark | 1 | 24 | 20 | 450 | 50 | 25 | 50 | False | 0 | 2 |
| three_people_dark | 3 | 24 | 20 | 450 | 90 | 55 | 90 | False | 0 | 2 |
| three_people_bright | 3 | 24 | 700 | 450 | 3.333 | 55 | 3.333 | False | 0 | 2 |
| crowded_hot_poor_co2 | 3 | 32 | 20 | 1600 | 90 | 90 | 90 | False | 0 | 2 |
| one_person_bright | 1 | 24 | 700 | 450 | 3.333 | 25 | 3.333 | False | 0 | 2 |
| empty_hot_fresh | 0 | 35 | 20 | 450 | 0 | 3.333 | 3.333 | True | 0 | 2 |
| occupied_cool_fresh | 2 | 16 | 200 | 450 | 50 | 55 | 50 | False | 0 | 2 |
| overlapping_memberships | 1.5 | 27 | 100 | 1150 | 53.74 | 66.47 | 53.74 | False | 0 | 12 |
| above_domain_clipped | 3 | 45 | 1200 | 2500 | 3.333 | 90 | 3.333 | False | 3 | 2 |

Scenario traces list all input membership degrees, fired rules and their strengths, raw
centroids, clipping, and final requests. The overlapping_memberships case illustrates several
rules firing at once. The above_domain_clipped case makes saturation visible.

## Integration examples and error propagation

| model | Source_Row | true_occupancy | predicted_occupancy | temperature | light | co2 | lighting_percent | ventilation_percent | true_label_reference_lighting | true_label_reference_ventilation | clipped_inputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1483 | 0 | 0 | 25.38 | 0 | 380 | 0 | 3.628 | 0 | 3.628 | 0 |
| baseline | 2834 | 1 | 1 | 25.19 | 93 | 360 | 40.85 | 25 | 40.85 | 25 | 0 |
| baseline | 2950 | 2 | 2 | 25.69 | 123 | 380 | 60.06 | 55 | 60.06 | 55 | 0 |
| baseline | 3497 | 3 | 2 | 26.19 | 124 | 760 | 59.74 | 74.4 | 59.74 | 74.4 | 0 |
| baseline | 8086 | 3 | 0 | 25.5 | 10 | 370 | 0 | 3.674 | 90 | 55 | 0 |
| baseline | 8087 | 3 | 0 | 25.5 | 10 | 370 | 0 | 3.674 | 90 | 55 | 0 |
| ga | 1483 | 0 | 0 | 25.38 | 0 | 380 | 0 | 3.628 | 0 | 3.628 | 0 |
| ga | 2834 | 1 | 1 | 25.19 | 93 | 360 | 40.85 | 25 | 40.85 | 25 | 0 |
| ga | 2950 | 2 | 1 | 25.69 | 123 | 380 | 33.06 | 25 | 60.06 | 55 | 0 |
| ga | 3497 | 3 | 2 | 26.19 | 124 | 760 | 59.74 | 74.4 | 59.74 | 74.4 | 0 |
| ga | 8086 | 3 | 0 | 25.5 | 10 | 370 | 0 | 3.674 | 90 | 55 | 0 |
| ga | 8087 | 3 | 0 | 25.5 | 10 | 370 | 0 | 3.674 | 90 | 55 | 0 |

Examples contain four real validation observations (one per true class) and two later-test
false-empty observations, each run through both estimators.
The true-label reference passes known occupancy through the SAME rule base, solely to
illustrate how occupancy errors propagate. True labels are never passed to the integrated
prediction pipeline. This reference is not a measured optimal action or new performance score.

Earlier evaluation found 184 of 294 occupied later-test rows predicted empty by both models.
This controller does not repair those occupancy estimates. The exact-off gate can therefore
request zero lighting in an occupied room. CO2-triggered ventilation remains available,
but it does not guarantee adequate control, particularly when CO2 is still low.
An error-free fuzzy implementation does not make the full system reliable.

The dataset does not contain desired control-output labels. Therefore no fuzzy-controller
accuracy, validated comfort score or actual energy-saving percentage can be reported.
Static response surfaces illustrate rules; they do not model CO2 dynamics, actuator response,
feedback of lighting into the light sensor, oscillation, hysteresis or real equipment.
The baseline remains the default academic demo; GA is an explicit alternative.
Model probabilities are uncalibrated and are not used as fuzzy occupancy degrees.

## Reproduction and files

~~~bash
python src/phase5.py
python src/verify_phase5.py
python src/predict_phase5.py --model baseline --input data/examples/phase5_baseline_controller.csv --output outputs/phase5/baseline_demo.csv --trace outputs/phase5/baseline_demo_trace.json
python src/predict_phase5.py --model ga --input data/examples/phase5_ga_controller.csv --output outputs/phase5/ga_demo.csv --trace outputs/phase5/ga_demo_trace.json
~~~

Prediction commands require new output filenames. Evaluation reruns regenerate only Phase 5
reports and examples. Earlier data/models remain unchanged.

- config/phase5_fuzzy.json: all membership parameters, rules, mappings and assumptions.
- src/fuzzy_controller.py: FuzzyRoomController and integrated SmartRoomSystem.
- src/phase5.py: reproducible plots, scenarios and real-row demonstrations.
- src/verify_phase5.py: coverage, centroid, rule, integration and interface tests.
- outputs/phase5/report.html: visual report.
- outputs/phase5/scenario_traces.json: inspectable rule-by-rule worked examples.
- outputs/phase5/integrated_traces.json: model output followed by fuzzy reasoning.
- outputs/phase5/tables: memberships, rule base, examples and verification results.
- outputs/phase5/plots: six PNG/SVG figures.

## Two-minute viva explanation

“I used Mamdani fuzzy logic to turn predicted occupancy and environmental readings into
lighting and ventilation requests. I defined overlapping fuzzy sets and 34 explicit rules.
For each input I calculate membership degrees. I use MIN for AND, clip each consequent,
combine with MAX and calculate the centroid to produce a crisp percentage.

For example, an empty dark room receives zero final lighting, while high CO2 can still
request ventilation. The exact-zero lighting is a clearly identified postprocessing rule,
because a fuzzy off set alone can have a small positive centroid.

I can show every fired rule and its strength. I tested overlap cases, input boundaries and
integration with both occupancy estimators. These are illustrative static control requests,
not real actuator commands or measured energy savings. Classification errors still affect
control, and temperature remains an additional controller measurement in the compact system.”

## Viva questions

**Why fuzzy logic?** It combines overlapping linguistic conditions into interpretable continuous requests.
**Membership versus probability?** Membership is compatibility with a linguistic set, not a probability of a class.
**Why MIN and MAX?** They are the chosen Mamdani AND and aggregation operators; the configuration states them.
**Why centroid?** It converts the combined fuzzy output into its centre of area.
**Why can high mean 90% rather than 100%?** The centroid is inside the membership shape, not necessarily at its peak.
**Is zero lighting purely fuzzy?** No; exact empty occupancy activates an explicit postprocessing gate.
**Why ventilate an empty room?** The illustrative rules allow residual elevated CO2 to request ventilation.
**Are thresholds optimal?** No, they are declared design choices and need validation for any real installation.
**Why five compact-path inputs?** The four-channel occupancy estimator still needs S1_Temp for control.
**Does this fix classifier errors?** No. Rule correctness and occupancy accuracy are separate issues.
**Where are energy savings?** Phase 6 will add explicitly simulated power assumptions.

## Reference

Mamdani control example and inference workflow:
https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_tipping_problem_newapi.html
The project implements its own rules and membership definitions, using NumPy rather than adding
a scikit-fuzzy dependency.
