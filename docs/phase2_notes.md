# Phase 2: Baseline occupancy prediction

## What was built

A Random Forest using the 15 direct sensor channels from Phase 1 predicts occupancy 0/1/2/3.
An always-empty model supplies a simple comparison. No GA, fuzzy controller or energy calculation
is included. The saved model is exactly the evaluated model, fitted only on December 22.

## Results

Validation accuracy: **94.96%**.
Validation macro F1 across all four classes: **0.8608**.
Validation balanced accuracy: **91.19%**.
Always-empty validation accuracy: **71.67%**.
Later-test accuracy: **95.32%**.
Later-test macro F1 over its three supported classes: **0.4159**.

**The later test reveals weak occupied-room generalization.** Always-empty test accuracy is
94.99%, close to the forest's 95.32%. Forest balanced accuracy is
only 38.99%. It detects 13 of 93 two-person readings
(recall 13.98%) and 6 of 201 three-person readings (recall 2.99%). These results do not justify
deployment or an energy-control claim. The differences suggest a transfer problem across
recording periods, but this experiment does not identify the cause.

Validation is the all-four-class evaluation. Later test has no true class 1 and cannot
establish recall for one-person occupancy. Its high empty-room share can inflate accuracy.
The separate January subset is an additional slice of the same test predictions, not an
independent dataset or an extra model fit.

Full metrics:
~~~text
        model           partition  rows  accuracy  macro_f1_present_classes  macro_f1_all_four  balanced_accuracy_present_classes  weighted_f1 true_classes
random_forest               train  1462    0.9979                    0.9973             0.9973                             0.9969       0.9979      0,1,2,3
random_forest          validation  2760    0.9496                    0.8608             0.8608                             0.9119       0.9529      0,1,2,3
random_forest                test  5870    0.9532                    0.4159                NaN                             0.3899       0.9394        0,2,3
random_forest january_test_subset  2045    0.8655                    0.4047                NaN                             0.3899       0.8277        0,2,3
 always_empty               train  1462    0.4357                    0.1517             0.1517                             0.2500       0.2645      0,1,2,3
 always_empty          validation  2760    0.7167                    0.2087             0.2087                             0.2500       0.5984      0,1,2,3
 always_empty                test  5870    0.9499                    0.3248                NaN                             0.3333       0.9255        0,2,3
 always_empty january_test_subset  2045    0.8562                    0.3075                NaN                             0.3333       0.7899        0,2,3
~~~

## Per-class results

Validation:
~~~text
 class  precision  recall     f1  support  predicted_count
     0     0.9990  0.9995 0.9992     1978             1979
     1     0.9961  1.0000 0.9980      254              255
     2     0.9962  0.6558 0.7909      398              262
     3     0.4886  0.9923 0.6548      130              264
~~~

Later test:
~~~text
 class  precision  recall     f1  support  predicted_count
     0     0.9681  1.0000 0.9838     5576             5760
     1        NaN     NaN    NaN        0                0
     2     0.5417  0.1398 0.2222       93               24
     3     0.0698  0.0299 0.0418      201               86
~~~

On validation, class 2 recall is 65.58% and class 3 precision is 48.86%, so even the stronger
validation result has meaningful confusion between counts. Undefined entries are explained below.

## Evaluation protocol

The calendar boundaries and model settings are recorded in phase2_protocol.json before fitting.
Training: December 22, 2017. Validation: December 23 after 00:10. Test: December 24 after 00:10
through the remaining recordings. First-ten-minute boundary windows are embargoed and never fitted
or scored. Rows are ordered in time, not shuffled. Embargoed records remain in Phase 1 data.

~~~text
               0    1    2    3
partition                      
train        637  205  257  363
validation  1978  254  398  130
test        5576    0   93  201
embargo       37    0    0    0
~~~

The 10-minute buffer is a pragmatic choice to reduce immediate adjacency, not evidence that
samples become independent after ten minutes. No rolling features are used. Slow environmental
effects can persist longer. This is one fixed temporal split from one collection, without
cross-validation or uncertainty estimates. Additional recordings are needed for reliable
generalization conclusions.

All rows were explored in Phase 1. The test is excluded from model fitting and parameter
selection, but is not untouched by earlier descriptive analysis. No hyperparameters were
changed in response to validation or test scores.

## Why Random Forest?

A decision tree asks a sequence of questions about measurements. A forest combines many trees
trained on varied bootstrap samples and feature subsets, which provides a straightforward
nonlinear sensor baseline. It can use different measurement scales without standardization.

Settings: 300 trees, minimum two samples per leaf, square-root feature sampling,
balanced_subsample class weighting, seed 42, one worker. Other settings are saved in model metadata.
Three hundred trees provide averaging, two samples per leaf modestly limit tiny leaves,
and class weighting addresses unequal training supports. These are fixed starting choices,
not optimized settings. Class weights are derived only from training bootstrap samples.

No scaling, imputation, resampling, synthetic rows or feature selection is fitted.
The target, dates, audit IDs and unverified CO2 slope are excluded from inputs.
No aggregate comparison is fitted in Phase 2; the baseline preserves spatial channels.

## Metrics in plain language

- Accuracy: fraction of all predictions that are correct.
- Precision for a class: among predictions of that class, how many are correct?
- Recall for a class: among true examples of that class, how many are detected?
- F1: balances precision and recall, calculated as 2TP/(true count + predicted count).
- Macro F1: average class F1, giving each evaluated class equal weight.
- Balanced accuracy: mean recall over classes actually present in the partition.
- Weighted F1: average class F1 weighted by true class support.
- Confusion matrix: rows are true occupancy, columns are predicted occupancy. The diagonal is correct.

Per-class CSV blanks mean undefined: recall has no denominator for an absent true class,
precision has no denominator when nothing is predicted as that class, and F1 is undefined
if both true and predicted supports are zero. Aggregate sklearn F1 uses zero_division=0.
The four-class macro F1 column is deliberately blank when any true class is absent.
The present-class macro F1 and balanced accuracy must not be interpreted as four-class test scores.

Training scores are fit diagnostics, not evidence of future performance.
Inspect validation per-class reports and off-diagonal errors rather than quoting accuracy alone.

## Feature importance

The plot shows training impurity-based importance. It describes how the fitted forest used
features, not causation or the optimal subset. Correlated sensors can share importance,
and continuous features may be favoured. This is not GA sensor selection and not proof that
low-ranked sensors can be removed safely.

## Files and reproducibility

- src/phase2.py: fixed train/evaluate/save workflow.
- src/predict_occupancy.py: load the saved model and predict from a CSV.
- models/phase2/random_forest_baseline.joblib: evaluated model, not refitted.
- models/phase2/model_metadata.json: feature order, parameters, versions and hashes.
- data/splits/phase2_split_manifest.csv: row-level assignment, including embargo.
- outputs/phase2/tables: per-class reports, confusion matrices, metrics and checks.
- outputs/phase2/predictions.csv: predictions with source row, timestamp and target for auditing.
- outputs/phase2/plots: five figures in PNG and SVG.
- outputs/phase2/report.html: visual report.

From the project directory, install requirements.txt, then run:
~~~bash
python src/phase2.py
python src/verify_phase2.py
python src/predict_occupancy.py --input data/processed/baseline_features.csv --output outputs/phase2/example_predictions.csv
~~~

The last command is an inference demonstration on already-known data, not an additional evaluation.
Model probabilities are not calibrated confidence. Only load model files from a trusted source.
Generated Phase 2 outputs are overwritten on rerun. Phase 1 outputs and raw data stay unchanged.

## Professor explanation

“I used the prepared sensor data to build an ordinary occupancy baseline before optimization.
The model takes 15 sensor readings and predicts zero, one, two or three people.
I trained a Random Forest on the first recording date and evaluated it on later dates,
leaving a ten-minute gap at the boundaries to reduce immediate overlap.

I compared it with an always-empty guess and reported accuracy, per-class precision,
recall, F1 and a confusion matrix. Validation accuracy is 94.96%, with macro F1
0.8608. The later test lacks one-person examples, so I report that limitation
and do not claim it verifies all four classes. The model and exact split are saved so
later sensor-selection experiments can be compared reproducibly.”

## Viva questions

**Why create a baseline?** It gives later optimization a fixed reference.
**Is Random Forest the soft computing contribution?** It is the occupancy estimator; GA and fuzzy control
are later components, not performed here.
**Why not use a random 80/20 split?** Adjacent measurements can be near duplicates in environmental state.
**Why class weighting?** It gives minority training classes greater influence without inventing samples.
**Why no normalization?** Tree splits use thresholds within each feature rather than distances across units.
**Can the final test prove class 1 performance?** No, its true support for class 1 is zero.
**Is feature importance sensor selection?** No, it is a diagnostic of this fitted model.
**What happens in Phase 3?** Sensor selection must optimize training-only temporal validation folds.
The reported test must not become GA fitness data. Final test results have now been seen, so repeated
adaptation to them would weaken the final evaluation; additional held-out recordings are preferable.

## References

Random Forest API: https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Classification metrics: https://scikit-learn.org/1.7/modules/model_evaluation.html
Dataset and provenance: https://doi.org/10.24432/C5P605 and data/raw/source_manifest.json
