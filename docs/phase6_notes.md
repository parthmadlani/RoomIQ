# Phase 6 — Energy Estimation & Adaptive Decision Layer

## What this phase does

The Phase 5 fuzzy requests now feed an illustrative linear power model and three explicit
modes: ECO, NORMAL and COMFORT. AUTO chooses among those modes using deterministic conditions.
This is rule-based adaptation, not online learning or an adaptive GA.

Every energy number is simulated. The source dataset has no electrical-consumption labels,
no measured desired actuator outputs and no evidence that these requests maintain comfort.
No equipment is actuated, and no tariffs or monetary savings are assumed.

## Power assumptions and equations

Nominal light power: **60 W**. Nominal ventilation power: **40 W**.
P_light = 60 x applied lighting percent / 100.
P_vent = 40 x applied ventilation percent / 100.
P_total = P_light + P_vent.
Energy_kWh = P_total x elapsed_seconds / 3,600,000.
Reduction_percent = 100 x (reference_kWh - simulated_kWh) / reference_kWh.

Example: 30% light request corresponds to 18 W. One hour at 18 W is 0.018 kWh.
Both loads at full power use 100 W, or 0.1 kWh per hour.
Linear scaling represents an assumed average duty-equivalent power; it is not a measured
dimmer/fan-speed power curve. HVAC, standby, sensors and processor/network loads are excluded.
Therefore the estimated savings do not quantify savings from GA sensor removal.

The comparator keeps both modelled loads at 100% over the SAME eligible intervals.
It is an assumed full-power baseline, not observed “conventional” consumption.
Comparing against a deliberately high full-power reference can give large reductions;
those reductions do not establish equal comfort or useful service.

All assumptions are editable in config/phase6_energy.json. After changes, regenerate outputs
and checks. The sensitivity table reuses frozen requests with three alternative load ratios
to show that percentages also depend on the assumed lighting/ventilation mix.

## Mode decisions

- ECO multiplies fuzzy lighting and ventilation requests by 0.8.
- NORMAL leaves both requests unchanged.
- COMFORT multiplies both by 1.15, capped at 100%.
- If CO2 is at least 1100 ppm, ventilation cannot be reduced below its original fuzzy request.
- Predicted occupancy zero still forces lighting to exactly zero in every mode.

AUTO chooses COMFORT if CO2 >= 1100 ppm, occupancy >= 2, or the room is predicted occupied
and temperature >= 28 C. Otherwise it chooses NORMAL if occupied, ECO if predicted empty.
Thresholds and multipliers are illustrative design preferences, not engineering standards.
The ventilation guard is not a safety guarantee or evidence of adequate air quality.
AUTO uses the same clipped environmental values that Phase 5 used, and clipping is counted.

There is no memory, learning, hysteresis, dwell time or physical feedback model.
Mode transitions within contiguous covered intervals are counted; transitions across gaps
are not counted. These choices must not be presented as a validated building-control policy.
Fixed ECO/NORMAL/COMFORT are comparisons, not three extra learned controllers.

## Time accounting

The replay uses the current sample's request until the next timestamp (left-endpoint
zero-order hold), but only if the next sample is within 60 seconds and remains in the same
Phase 2 partition. Intervals touching embargo rows are excluded. The final row has no known
duration. Longer gaps are not filled and are not treated as zero consumption.

Covered intervals: **9,950**.
Accounted duration across all eligible partitions: **84.6806 hours**.
Known first-to-last span: **478.1744 hours**.
Most of the elapsed calendar span is excluded recording gaps.

| status | interval_rows | known_elapsed_seconds | accounted_seconds |
| --- | --- | --- | --- |
| eligible | 9950 | 3.048e+05 | 3.048e+05 |
| recording_gap_over_60s | 141 | 1.415e+06 | 0 |
| touches_embargo | 37 | 1133 | 0 |
| final_row_unknown_duration | 1 | 0 | 0 |

Each source/policy uses exactly the same eligible intervals. Training, validation and test
are reported separately, plus an explicitly combined all_eligible view. The combined view
is descriptive and is not independent evaluation. Daily aggregates use the starting
observation's calendar date. The method assumes the sampled request and occupancy label
persist until the next eligible sample; sub-sample changes remain unknown.

## Later-test AUTO illustration

| source | covered_hours | simulated_kwh | full_power_reference_kwh | simulated_reduction_percent | occupied_zero_lighting_hours | occupied_zero_lighting_percent |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 49.3 | 0.1474 | 4.93 | 97.01 | 1.549 | 62.54 |
| ga | 49.3 | 0.1474 | 4.93 | 97.01 | 1.549 | 62.54 |
| true_label_reference | 49.3 | 0.2738 | 4.93 | 94.45 | 0 | 0 |

These energy percentages must be read alongside the occupied-zero-lighting column.
The baseline estimator's AUTO simulation requests zero lighting for
**1.5489 occupied hours**
(**62.54%** of covered occupied time).

That error is not an efficiency benefit. A lower energy figure can be caused by
misclassifying an occupied room as empty. The energy layer does not repair those mistakes.
No measured comfort or task-illumination threshold is available, so “nonzero lighting” is
also not proof of adequate service.

The true_label_reference path feeds the known occupancy label into the SAME fuzzy and mode
rules. It is solely a counterfactual diagnostic of occupancy-error propagation. It is not an
available production predictor, a measured optimal-control policy, or an independent benchmark.
Ground-truth labels never enter either predicted-occupancy pipeline.

## Why baseline and GA energy can match

In the covered later-test replay, baseline and GA produce identical power requests under
these rules even though some occupancy counts differ. Their disagreements there are between
two and three people, both of which have full membership in the same high-occupancy set
and select the same AUTO mode. Environmental inputs are shared. No sensor-node power is
modelled, so reducing estimator inputs alone creates no simulated energy benefit here.

## Outputs and reproduction

~~~bash
python src/phase6.py
python src/verify_phase6.py
python src/predict_phase6.py --model baseline --policy AUTO --seconds-per-row 30 --input data/examples/phase5_baseline_controller.csv --output outputs/phase6/demo_energy.csv
~~~

The prediction CLI takes an explicitly hypothetical duration for EACH row. It does not
integrate input timestamps or fill gaps. Use phase6.py for the audited dataset replay.
The CLI refuses to overwrite output files. Generated Phase 6 reports/tables are overwritten
on rerun; prior-phase data and models remain unchanged.

- src/energy_controller.py: reusable EnergyDecisionLayer and integrated EnergyRoomSystem.
- config/phase6_energy.json: nominal powers, mode multipliers, thresholds and exclusions.
- outputs/phase6/interval_simulation.csv: every eligible interval/source/policy and power/energy.
- outputs/phase6/tables/interval_audit.csv: every source row and its inclusion/exclusion reason.
- outputs/phase6/tables/energy_summary.csv: component kWh, reference and reduction, service-error proxies.
- outputs/phase6/tables/mode_time.csv: mode durations.
- outputs/phase6/tables/mode_scenarios.csv: Phase 5 scenarios under each policy.
- outputs/phase6/tables/power_assumption_sensitivity.csv: assumed-power sensitivity.
- outputs/phase6/report.html: visual report; six figures are available in PNG and SVG.

## Professor explanation

“I connected the fuzzy controller to a transparent energy simulation. I assumed a 60-watt
light load and a 40-watt ventilation load, multiplied by the requested percentages, and
integrated power over actual short recorded intervals. I excluded long gaps and unknown
durations rather than assuming continuous data.

ECO, NORMAL and COMFORT scale the requests, while AUTO selects a mode from occupancy,
temperature and CO2. These are declared deterministic rules, not learned adaptation.

I compared simulated consumption with both loads continuously at full power over exactly
the same covered intervals. I also report occupied periods where the model asks for zero
lighting, because otherwise classification mistakes could look like savings. The results
are simulation estimates under assumptions, not measured building energy savings.”

## Viva questions

**Power versus energy?** Power is the instantaneous rate in watts. Energy accumulates it over time, in Wh or kWh.
**Why divide by 3,600,000?** Convert watt-seconds to watt-hours, then kilowatt-hours.
**Why not assume every row is 30 seconds?** Recorded intervals vary and long gaps exist.
**Does AUTO learn?** No. It is deterministic rule-based mode selection.
**Does COMFORT guarantee comfort?** No; it is a mode name and scaling preference, not a validated comfort measure.
**Why preserve ventilation at elevated CO2?** To avoid scaling down the original request in that illustrative condition.
**Does GA yield measured energy savings?** No. Sensor-node power is not modelled.
**Why include true-label reference?** To demonstrate how occupancy errors change simulated requests.
**Can larger savings mean a worse controller?** Yes, if it withholds useful service during occupied intervals.
**What remains?** Phase 7 integrates the existing modules into a dashboard and presentation.
