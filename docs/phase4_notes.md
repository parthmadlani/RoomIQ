# Phase 4 — Optimized Occupancy System & GA Evaluation

## Verdict

GA selected four of fifteen inputs, a **73.33% reduction in model input channels**.
It did **not** preserve baseline validation performance:
accuracy changed from 94.96% to 92.79%,
and macro F1 from 0.8608 to 0.7952.
The change in validation macro F1 is -0.0655.

Keep the baseline as the default academic demonstration estimator; expose the four-input
model as an explicit compact alternative. This preference follows four-class validation,
not the slightly better later-test accuracy. Neither model is established as reliable for
actual room control. “GA-optimized” means optimized for its training fitness objective.

## Professor comparison table

| model | input_channels | source_nodes | input_reduction_percent | validation_accuracy_percent | validation_macro_f1 | later_test_accuracy_percent | later_test_balanced_accuracy_percent | later_test_false_empty_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 15 | 7 | 0 | 94.96 | 0.8608 | 95.32 | 38.99 | 62.59 |
| ga | 4 | 4 | 73.33 | 92.79 | 0.7952 | 95.37 | 40.83 | 62.59 |

The later test has true classes 0, 2 and 3 only. Four-class macro F1 is unavailable there.
Source nodes referenced decrease from seven to four, but these are theoretical data
dependencies, not removed hardware or measured savings. S1–S4 originally carry multiple channels.
The 73.33% figure applies only to input columns. It is not a power, price or electricity reduction.

## Exact experiment

1. Freeze the Phase 3 chromosome and its four selected channels: S1_Light, S5_CO2, S6_PIR, S7_PIR.
2. Verify hashes of the Phase 1 data, Phase 2 baseline, Phase 3 model and split manifest.
3. Independently retrain a fresh selected-input forest using exactly the existing settings:
   300 trees, minimum leaf size two, square-root feature sampling, balanced_subsample weights,
   random state 42 and one worker. Other parameters are in the model metadata.
4. Fit only the original 1,462 December 22 training rows.
5. Compare the fresh model with the original selected model: every prediction and probability
   on the full prepared dataset must match. They do match.
6. Reuse the original saved 15-input baseline. Compare on identical validation/test rows.
7. Save metrics, per-class reports, paired outcomes, confusion matrices and false-empty diagnostics.
8. Package both model options behind a common prediction interface.

There is no new GA search, hyperparameter tuning, resampling, train/validation merge,
threshold optimization or retraining on the test. Earlier model files remain unchanged.
The selected model is a genuinely fresh fit; no new independent dataset has been acquired.

Partitions: 1,462 training rows, 2,760 validation rows, 5,870 later-test rows and
37 embargo rows. Ten-minute gaps separate partitions. Phase 1 explored all data, and external
scores were already reported in Phases 2 and 3. These are repeated descriptive comparisons,
not a new blind evaluation. The January subset overlaps the later test.

## Why accuracy is not enough

The later test contains 5,576 empty rows out of 5,870. An always-empty guess achieves 94.99%
accuracy. Its present-class balanced accuracy is 33.33%.

Both models label **184 of 294 occupied later-test
rows as empty (62.59%)**. The small aggregate improvement of the GA model does
not solve that failure. Later-test class 3 recall changes from 2.99% for the baseline to 1.00%
for the GA model; class 2 recall changes from 13.98% to 21.51%.

In validation, the baseline is stronger on each occupied class's recall:
class 1, 100.00% versus 99.21%; class 2, 65.58% versus 59.30%;
class 3, 99.23% versus 73.85%. These results explain the default-model decision.
An erroneous empty prediction could later suppress desired lighting or ventilation, so
a future controller demonstration must visibly retain this limitation.

The false-empty rate uses actual occupied rows as denominator. The false-occupied rate uses
actual empty rows. Count MAE measures average absolute error in people and supplements,
rather than replaces, the multiclass metrics. Paired outcome counts use the same source rows.
No row-independent significance test or confidence interval is claimed because samples
are temporally dependent and come from one collection.

## Metric definitions

Accuracy is the fraction of correct predictions. Per-class precision asks how often
predictions of a class are right. Recall asks how many true instances are found.
F1 is 2TP/(true support + predicted support). Macro F1 gives each evaluated class equal weight.
Balanced accuracy averages recall over true classes present in a partition.
Weighted F1 gives larger classes more weight.

The comparison CSV includes macro precision and macro recall over present true classes.
For macro precision, a class with no predictions contributes zero. Per-class CSVs leave
undefined precision/recall/F1 blank. Four-class macro F1 is blank if any true class is absent.
A high score on the later test cannot verify one-person performance.

## Common occupancy interface

src/occupancy_system.py provides OccupancySystem(model_name="baseline") or
OccupancySystem(model_name="ga"). Each accepts a dictionary for one room observation or
a table of multiple observations, and returns predicted_occupancy and probabilities for 0–3.

The default model requires all 15 baseline channels. The explicit GA option requires only
S1_Light, S5_CO2, S6_PIR and S7_PIR. Extra columns, including a supplied target, are ignored.
Columns are placed in the saved model order. Missing, nonnumeric, nonfinite, invalid PIR
or negative light/sound/CO2 inputs are rejected. The interface never silently substitutes
the GA model when baseline inputs are missing. There is no automatic device actuation.

Probabilities are uncalibrated. Only load trusted project-generated model files.
Model checksums and expected features are stored in models/phase4/model_registry.json.

Example usage from the project directory:
~~~bash
python src/phase4.py
python src/verify_phase4.py
python src/predict_phase4.py --model baseline --input data/examples/phase4_all_sensors.csv --output outputs/phase4/baseline_demo.csv
python src/predict_phase4.py --model ga --input data/examples/phase4_ga_sensors.csv --output outputs/phase4/ga_demo.csv
~~~

The examples contain one real validation observation per class, with provenance saved
separately. They are for demonstrating the interface, not additional evaluation evidence.
Prediction commands refuse to overwrite existing output files. Generated evaluation outputs
are overwritten on a Phase 4 rerun. Both prior phases and raw data are preserved.

## Files to show the professor

- outputs/phase4/report.html — visual comparison and conclusion.
- outputs/phase4/tables/professor_comparison.csv — concise model comparison.
- outputs/phase4/tables/comparison.csv — full partition metrics.
- outputs/phase4/tables/per_class_comparison.csv — precision, recall, F1 and support.
- outputs/phase4/tables/occupancy_error_analysis.csv — false-empty and count-error analysis.
- outputs/phase4/tables/paired_outcomes.csv — which observations each model gets right.
- outputs/phase4/plots — six figures in PNG and SVG.
- models/phase4 — independently retrained model, registry and explicit evaluation decision.

## Two-minute viva explanation

“Phase 4 tests whether GA sensor selection actually helped. I retrained the same Random Forest
using only the four selected inputs and kept the training rows, model settings and evaluation
partitions unchanged. I compared it with the saved 15-input baseline.

The input count fell by 73.33%, but validation accuracy fell from 94.96% to 92.79% and
macro F1 from 0.8608 to 0.7952. Therefore, I do not claim that GA improved prediction overall.
It found a smaller alternative with a measurable performance cost.

I also examined occupied rooms predicted as empty. Both models make that error on 184 of
294 occupied later-test readings. This explains why high overall accuracy is misleading.
I keep the baseline as the default demonstration model and expose the GA model as a compact
alternative. The next phase will demonstrate fuzzy reasoning while clearly retaining these limits.”

## Viva questions

**Why retrain in Phase 4?** To verify the selected representation produces the same result
in a separate, reproducible fit and formalize the baseline comparison.
**Did GA fail?** It improved its training objective and reduced inputs. External validation
shows a performance cost, so it did not deliver an overall predictive improvement.
**Why is the baseline the default?** It has higher four-class validation macro F1 and better
occupied-class validation recall.
**Is 73.33% fewer inputs equal to 73.33% energy saving?** No. No hardware powers or savings were measured.
**Why not tune until the test improves?** Repeated tuning against known test results makes
the evaluation optimistic. Further confirmation needs new held-out recordings.
**Why report false-empty errors?** They are directly relevant to an occupancy-driven controller.
**Have you built fuzzy logic?** No. That is Phase 5 in the original seven-phase plan.

## Phase 5 handoff

Use the common occupancy interface with an explicit model choice. Keep the baseline as the
default demo configuration and allow an explicit compact-model comparison. Fuzzy inputs such
as temperature or room-wide light are separate controller measurements: the four-input
occupancy subset does not automatically eliminate the controller's need for other sensors.
Do not infer a missing temperature from an occupancy prediction. Fuzzy control must remain
a demonstrator until performance is validated on suitable additional recordings.

No fuzzy rules, membership functions, actuator outputs or energy calculations are implemented here.
