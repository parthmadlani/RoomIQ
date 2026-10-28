# Phase 2 candidate inputs

## Recommended baseline: 15 direct sensor channels

| candidate_input | unit | reason |
| --- | --- | --- |
| S1_Temp | degrees C | Local temperature and spatial variation; also sensitive to ambient conditions. |
| S2_Temp | degrees C | Local temperature and spatial variation; also sensitive to ambient conditions. |
| S3_Temp | degrees C | Local temperature and spatial variation; also sensitive to ambient conditions. |
| S4_Temp | degrees C | Local temperature and spatial variation; also sensitive to ambient conditions. |
| S1_Light | lux | Local light information; association may depend on this room's lighting habits. |
| S2_Light | lux | Local light information; association may depend on this room's lighting habits. |
| S3_Light | lux | Local light information; association may depend on this room's lighting habits. |
| S4_Light | lux | Local light information; association may depend on this room's lighting habits. |
| S1_Sound | V | Local activity; amplifier voltage is not decibels. |
| S2_Sound | V | Local activity; amplifier voltage is not decibels. |
| S3_Sound | V | Local activity; amplifier voltage is not decibels. |
| S4_Sound | V | Local activity; amplifier voltage is not decibels. |
| S5_CO2 | ppm | Direct CO2 concentration; may lag arrivals and persist after departures. |
| S6_PIR | 0/1 | Motion coverage at this location; zero does not prove the room is empty. |
| S7_PIR | 0/1 | Motion coverage at this location; zero does not prove the room is empty. |

This is a starting candidate set, not a learned ranking or a claim of optimality.
Keep separate channels initially to preserve spatial information. No feature was removed
solely because its correlation was smaller.

## Optional compact comparison: 6 inputs

Avg_Temp, Avg_Light, Avg_Sound, S5_CO2, S6_PIR, S7_PIR.

Compare this representation with the baseline on identical validation partitions in Phase 2.
Averages summarize four simultaneous readings in the same units. They can conceal local
signals and calibration offsets. Average sound is mean voltage, not mean loudness in dB.
The two PIR channels remain separate because they cover different locations. PIR_Any is an
optional alternative summary, not an additional redundant baseline feature.

Averaging reduces model columns, but still needs all four underlying sensors.
It does not demonstrate reduced hardware cost or energy consumption.

## Conditional feature

S5_CO2_Slope is preserved and plotted, but excluded from the recommended baseline.
The source calls it a sliding-window slope without sufficient detail to verify window
length, alignment, boundary handling or time units. It is not proven to leak, but is not
yet verified as a causal real-time input. Confirm a trailing-only definition before using
it, or construct a new clearly defined trailing CO2 trend separately within each segment.

## Exclusions

Room_Occupancy_Count is the target, never an input. Date, Time, Timestamp, Source_Row and
Recording_Segment are audit fields. The baseline uses neither these fields nor aggregates.
The compact representation is a separate experiment.

## Phase 2 handoff and evaluation plan

Use feature_spec.json for explicit input lists. Separate the target from baseline_features.csv.
All three processed tables have the same chronological row order; occupancy_cleaned.csv
provides the timestamps. No training or split has been performed.

1. Design time-blocked or grouped evaluation and inspect class counts in every partition.
   The last calendar date contains only class 0. Class 1 occurs only on the first two dates.
   A simple late-date holdout cannot evaluate all four classes. Contiguous early-recording
   blocks with a justified time buffer can support an in-dataset comparison. Later-period
   evaluation should be reported separately with its absent classes. Additional recordings
   are needed for a strong future-period, all-class generalization claim.
2. Do not rely on a random row split: adjacent readings share conditions. Justify any buffer
   using temporal dependence and the lookback of future windowed features.
3. Fit scaling, imputation, balancing and feature selection on training folds only.
   This dataset needs no imputation. A tree baseline generally does not need scaling.
4. Report per-class precision/recall/F1, class supports, macro F1, balanced accuracy and
   a confusion matrix, alongside accuracy and an always-empty baseline.
5. All available rows were used for descriptive EDA. No untouched test set is reserved.
   Do not call a later split completely unseen to this exploration, or tune on test results.
