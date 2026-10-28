# Seven-minute presentation and viva guide

Open outputs/phase7/presentation.pptx. Speaker notes contain details and evidence references.
Start the dashboard before presenting; no internet is needed for the demo after setup.

## Talk track
1. **0:00–0:40 — Problem (slides 1–2).** “We estimate occupancy from environmental sensors,
   then translate it into explainable lighting and ventilation requests.”
2. **0:40–1:25 — Data (slide 3).** UCI, preserved originals, no missing rows, four classes.
   Explain chronological splitting and the ten-minute boundary embargo.
3. **1:25–2:10 — Baseline (slide 4).** Contrast 95.32% accuracy with 38.99% balanced accuracy.
   Almost all test rows are empty; an always-empty model reaches 94.99% accuracy.
4. **2:10–3:00 — GA (slide 5).** Each bit selects a sensor channel. Fitness rewards macro F1
   and penalizes input count. Selection, crossover and mutation search subsets.
   Four selected inputs reduce count but validation macro F1 declines.
5. **3:00–4:00 — Fuzzy logic (slide 6).** Membership is a degree of belonging, not probability.
   Explain minimum AND, maximum aggregation and centroid. Mention the explicit exact-off gate.
6. **4:00–5:00 — Energy (slides 7–8).** State 60 W and 40 W assumptions before showing values.
   Explain why occupied zero-light hours must be considered beside simulated reduction.
7. **5:00–6:30 — Live demo (slide 9).** Run the three manual scenarios; show fired rules.
   Then select a recorded later-test false-empty example to expose error propagation.
8. **6:30–7:00 — Conclusion (slide 10).** Reproducible integration is the contribution.
   Future validation needs new rooms, better occupied recall and measured power and comfort.

## Live demo clicks
1. Double-click Start Dashboard.command.
2. Room simulator → Input method → Manual scenario.
3. Empty + dark → Run simulation. Lighting is exactly 0%.
4. High occupancy + high CO2 → Run. AUTO selects COMFORT; both final requests reach 100%.
5. Low occupancy + bright → Run. Lighting is low (3.33%), ventilation 25%.
6. Expand fired rules and membership degrees.
7. Switch to sensor readings; choose a later-test false-empty example.
   Provenance says 3 people, but the model predicts 0.
8. Models & GA → compare validation macro F1, selected channels and evolution.
9. Energy evidence → Later test / AUTO. Read 0.147 kWh alongside 1.55 occupied zero-light hours.
10. Save a result as JSON if the professor wants the complete numerical trace.

## Likely questions
**Why Random Forest with Soft Computing?**
It supplies the occupancy estimate. GA and fuzzy inference are the syllabus-linked
optimization and reasoning components; we do not claim the classifier itself is fuzzy.

**Did GA improve the model?**
It improved its penalized training-block objective and reduced input count.
Validation macro F1 decreased; a global or future-data improvement is not demonstrated.

**Why not use random train/test splitting?**
Nearby observations are highly related. Chronological blocks better expose changes over time.
This still does not substitute for independent rooms or fresh data.

**Why 15 inputs when the source has 16 sensor-related fields?**
The supplied CO2 slope was retained for audit but excluded pending verification of its
window calculation. Metadata and labels are never model inputs.

**Why does the GA system need five readings?**
Four estimate occupancy; S1_Temp is additionally required for fuzzy control.

**Does adaptive mean online learning?**
No. AUTO chooses a mode deterministically from current conditions.

**Why can 'off' be nonzero?**
The centroid of a fuzzy off triangle is positive. The exact-zero lighting gate is a separate
explicit decision, applied only at occupancy zero.

**Are the CO2 thresholds safety limits?**
No. They are illustrative parameters for this project, not validated standards.

**Did you save 97% electricity?**
No. The figure is an assumption-dependent reduction against full-power operation.
No electricity labels exist, and missed occupancy also reduces simulated use.

**What is verified?**
Data/model/controller checks and 72 Phase 7 integration checks. The slide deck was inspected.
Browser visual/mobile testing remains a manual handoff check because automation was unavailable.

**What would you improve first?**
Occupied recall on fresh data, then temporal safeguards, then calibrated device power and
comfort measurements. Avoid optimizing a saving number while ignoring service failures.
