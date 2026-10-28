# Phase 3: Genetic Algorithm sensor selection

## Result and interpretation

Selected **4 of 15 sensor channels** (73.3% fewer input columns):
S1_Light, S5_CO2, S6_PIR, S7_PIR.

Chromosome: 000010000000111.
Input order is recorded in outputs/phase3/tables/chromosome_mapping.csv.

Training-only pooled out-of-fold macro F1: **0.7417**.
All-input search reference macro F1: **0.4553**.
Selected fitness: **0.7283**.

With identical 300-tree settings and the same December 22 final training data,
external validation macro F1 decreased from **0.8608** to
**0.7952** (difference -0.0655).
Selected-model validation accuracy: **92.79%**.
Selected-model later-test accuracy: **95.37%**; balanced accuracy across its
three supported classes: **40.83%**.

The GA optimizes its inner training objective; external improvement is not guaranteed.
Do not claim the subset preserves accuracy solely because its inner fitness is higher.
Both validation and test were already reported in Phase 2, so these are descriptive
comparisons, not new independent confirmation. No choices were changed after this comparison.

## Chromosome and fitness

Each bit corresponds to one direct sensor channel in the Phase 2 input order.
1 means include the channel; 0 means omit it. There are 2^15 - 1 = 32,767 nonempty subsets.
A zero chromosome is repaired by activating a random bit.

Fitness = pooled OOF macro F1 - 0.05 x (selected inputs / 15).

Macro F1 values each occupancy class equally. The penalty favours smaller subsets.
At most 0.05 F1-equivalent units are penalized for all 15 inputs; one extra input costs
0.05 / 15 = 0.003333 fitness units. This is a fixed modelling preference, not a measured
economic or energy cost. It was fixed before the search and never tuned on external scores.

OOF means each training row is predicted by an inner model that did not fit that row.
All OOF predictions are pooled, then one four-class macro F1 is computed. We do not average
fold F1 scores, because some validation blocks contain only a subset of the four classes.

## Training-only fold construction

Only the 1,462 Phase 2 training rows from December 22 enter the GA.
They are divided into four contiguous equal-row validation blocks.
For each block, fit rows come from the remaining December 22 data, excluding anything
within ten minutes before or after the held-out block. Every inner fit has all four classes.
The fold membership and class supports are saved for audit.

This is purged blocked within-day interpolation: some fit rows are later than the validation
block. It is **not** a forward-chaining forecast experiment. A ten-minute buffer reduces
immediate adjacency but does not prove independence. Slow environmental effects can last longer.

The external validation/test dates and embargo rows never enter GA fitting or fitness.
The previously unverified CO2 slope, timestamps, target and audit identifiers are not candidate
inputs. The original 15 direct channels are the complete search space.

## GA process and fixed settings

1. Initialize 16 nonempty chromosomes: all inputs, one random singleton and 14 random subsets.
2. Evaluate each new subset using four purged training-only folds.
3. Keep two elite copies unchanged.
4. Select parents by tournaments of three randomly sampled population positions.
5. Apply one-point crossover with probability 0.8 per pair.
6. Flip each bit independently with probability 1/15 per offspring (about one flip on average).
7. Repair empty offspring, evaluate, and repeat for 10 generations.
8. Run seeds 42, 43 and 44. Choose the highest penalized training fitness across the three run winners.
   Ties prefer fewer inputs, then the lexicographically smaller bit string.
9. Save selection_locked.json before predicting on external data.
10. Fit a final 300-tree model on the original training day using only the chosen inputs.

The initial population is generation 0; there are 11 logged populations per run.
The search uses 80-tree Random Forests to make repeated evaluation practical. All candidates,
including the all-input reference, use identical inner folds and estimator settings.
The final comparison uses the original 300-tree settings on both sides, isolating the
representation change as far as this setup permits. The 80-tree proxy can rank subsets
differently from a 300-tree estimator; its inner scores are not final-model held-out scores.
Forest seed is fixed at 42. GA seeds change the evolutionary search, not the data folds.

Repeated chromosomes reuse a deterministic fitness cache, shared across runs.
The search evaluated **318 distinct subsets**.
The cache saves computation without changing fitness. This is a heuristic search,
not exhaustive optimization or proof of the globally best subset.

## Stability across runs

All three seeds chose the same chromosome: **False**.
Different winners indicate search sensitivity. Even agreement across these three seeds
would not establish stability across different days, folds or datasets.

~~~text
 seed      chromosome  selected_count  cv_macro_f1  fitness
   42 000101000001100               4     0.608253 0.594920
   43 000101000001110               5     0.662046 0.645380
   44 000010000000111               4     0.741672 0.728339
~~~

## Full comparison

~~~text
      model           partition  rows  accuracy  macro_f1_present_classes  macro_f1_all_four  balanced_accuracy_present_classes  weighted_f1 true_classes  input_count
baseline_15               train  1462    0.9979                    0.9973             0.9973                             0.9969       0.9979      0,1,2,3           15
baseline_15          validation  2760    0.9496                    0.8608             0.8608                             0.9119       0.9529      0,1,2,3           15
baseline_15                test  5870    0.9532                    0.4159                NaN                             0.3899       0.9394        0,2,3           15
baseline_15 january_test_subset  2045    0.8655                    0.4047                NaN                             0.3899       0.8277        0,2,3           15
ga_selected               train  1462    0.9904                    0.9873             0.9873                             0.9868       0.9904      0,1,2,3            4
ga_selected          validation  2760    0.9279                    0.7952             0.7952                             0.8308       0.9303      0,1,2,3            4
ga_selected                test  5870    0.9537                    0.4369                NaN                             0.4083       0.9399        0,2,3            4
ga_selected january_test_subset  2045    0.8670                    0.4257                NaN                             0.4083       0.8291        0,2,3            4
~~~

Training scores are diagnostics. The January subset overlaps the later test and is not
independent evidence. The later test has no true class 1; four-class macro F1 is left blank.
Present-class macro F1 and balanced accuracy average only classes with actual support.
Per-class undefined precision/recall/F1 values are blank; inspect the saved supports.
Confusion matrices use actual class as rows and predicted class as columns.

## What “sensor reduction” means here

The chromosome selects individual measurement channels. The selected channels involve
4 distinct source nodes: S1, S5, S6, S7.
Nodes S1–S4 each originally carry temperature, light and sound channels.
Dropping one channel does not necessarily remove its node, radio, processor or power supply.
No hardware has been removed and no watts, cost or energy savings have been measured.
The feature count penalty is only a simple complexity proxy. A later hardware-aware
optimization would need node groups, actual costs and deployment constraints.

## Files and reproduction

- src/phase3.py: commented, handwritten GA operators, fitness, training and comparison.
- src/report_phase3.py: reports and six PNG/SVG figures.
- src/verify_phase3.py: independent metrics, fold and GA-history checks.
- src/predict_phase3.py: selected-model prediction command.
- docs/phase3_protocol.json: fixed settings and selection policy.
- models/phase3/selection_locked.json: choice made before external evaluation.
- models/phase3/ga_selected_random_forest.joblib: final selected-input model.
- models/phase3/model_metadata.json: features, hashes and environment.
- data/splits/phase3_inner_folds.csv: every inner fit/validation/purge assignment.
- outputs/phase3/tables: search cache, complete populations, convergence, winners and metrics.
- outputs/phase3/predictions.csv: external comparison predictions with source row identifiers.

From the project directory:
~~~bash
python src/phase3.py
python src/verify_phase3.py
python src/predict_phase3.py --input data/processed/baseline_features.csv --output outputs/phase3/example_predictions.csv
~~~

The prediction example uses known data and is not a new evaluation.
Generated Phase 3 outputs are overwritten on rerun. Earlier phase models and reports are preserved.
Only load trusted model files. Probabilities are not calibrated confidence.

## Professor explanation

“I represented sensor selection as a 15-bit chromosome. A one includes a sensor channel and
a zero excludes it. My fitness rewards four-class macro F1 and subtracts a small input-count
penalty, so the search balances prediction and complexity.

I used tournament selection, one-point crossover, bit-flip mutation and elitism. All fitness
evaluations use purged time blocks inside the original training day. Validation and test
scores are not part of fitness. I ran three seeds to inspect search sensitivity.

The selected chromosome uses 4 inputs. Its validation macro F1 is 0.7952
compared with 0.8608 for all 15 inputs. I report external changes honestly,
because optimizing training fitness does not guarantee improvement on future recordings.
This is channel selection, not measured hardware or energy savings.”

## Viva questions

**Why binary encoding?** Each candidate channel is either included or excluded.
**Why a penalty?** Prediction score alone might keep redundant inputs; the penalty expresses a simplicity preference.
**Why macro F1?** Empty-room rows dominate; all four classes should influence selection.
**Selection versus crossover versus mutation?** Selection favours fit parents, crossover mixes their
choices, and mutation introduces new bit choices.
**What is elitism?** Preserve the best two chromosomes unchanged each generation.
**Why several seeds?** A GA is stochastic. Different seeds can find different subsets.
**Why caching?** A repeated subset has the same deterministic evaluation, so recomputing is unnecessary.
**Can GA guarantee the global optimum?** No, only a fraction of the possible subsets was evaluated.
**Does higher fitness guarantee better test performance?** No. Inner-fold overfitting and time-period
differences can prevent that. Do not retune against the test after seeing the result.
**Why no fuzzy control yet?** This phase concerns sensor selection; control belongs to a later phase.

## Next-phase constraint

Keep the baseline as a reference regardless of this subset's score. Do not use the already
reported later-test score as a new GA objective or claim an independent final confirmation.
Further evidence needs new held-out recordings. GA does not automatically solve the weak
later-period generalization observed in Phase 2.

## Reference

GA operator concepts: https://deap.readthedocs.io/en/master/tutorials/basic/part2.html
This project implements the operators directly for explanation; it does not depend on DEAP.
