# Phase 1 notes: Adaptive Smart Room Occupancy & Energy Controller

## Scope

This phase supplies verified source data, a clean chronological table, understandable
aggregates, a full column audit, summary statistics, 10 EDA plots and candidate input lists.
No classifier, GA, fuzzy control or energy simulation has been performed.

## Source and problem

Estimate occupancy classes 0, 1, 2 or 3 using sensor evidence.
Room_Occupancy_Count is the true label and is kept out of the inputs.

Singh, A. & Chaudhari, S. (2018). Room Occupancy Estimation [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5P605
Dataset: https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The source contains temperature, light, sound, CO2 and PIR observations with manually recorded
occupancy. UCI describes nominal 30-second reporting and no HVAC operation during collection.
There are no measured electricity or device-power labels. Future energy savings must be
explicit simulations with stated assumptions.

The original ZIP and CSV are preserved in data/raw. source_manifest.json records the source
URL, retrieval time, byte sizes and SHA-256 hashes. Existing raw files are verified on rerun.

## Inspection

- Raw: **10,129 rows x 19 columns**: 16 sensor-related fields, two time fields and one target.
- Processed: **10,129 rows x 26 columns**: seven added fields.
- Missing cells: **0**. Exact duplicate rows: **0**. Duplicate timestamps: **0**.
- Numeric values are finite, PIR is binary, and labels are exactly 0, 1, 2, 3.
- No constant numeric sensor features.
- Recorded range: **2017-12-22 10:49:41** to **2018-01-11 09:00:09**.
- **7 calendar dates**, median interval **31 seconds**.
- **141 intervals over 60 seconds**.
- A descriptive gap threshold of 600 seconds creates **4 segments**.

UCI's four-day description differs from the seven distinct calendar dates in this CSV.
Report the observed dates transparently. Do not rewrite dates or assume a continuous recording.
The timestamps have no supplied timezone.

See data_dictionary.md for every column, dtype, unit, missing count, unique count and range.
raw_numeric_statistics.csv gives count, mean, standard deviation, minimum, quartiles and maximum
for every numeric field. The target is still a four-class label even though it is stored as an integer.

## Cleaning and preprocessing

1. Verify the source checksum, expected shape and schema.
2. Validate missing values, exact duplicates, numeric parsing, finite values, binary PIR,
   target labels and nonnegative light, sound and CO2.
3. Parse Date and Time explicitly using YYYY/MM/DD HH:MM:SS.
4. Preserve source row numbers and sort chronologically. The source was already sorted.
5. Mark gaps over 10 minutes as descriptive recording segments.
6. Add simultaneous means of the four temperature, light and sound channels and a PIR OR indicator.
7. Export the full audit table and separate baseline/compact feature tables.

**No rows removed, no imputation, no clipping, no interpolation, no resampling,
no scaling and no class balancing.** Cleaning includes validating already sound observations.

IQR flags identify unusual values but do not establish errors. Occupied readings can appear
unusual because most rows are empty. Negative slopes can represent falling CO2. Zero light
and zero PIR are valid readings. Calibration cannot be certified from this CSV alone.
No arbitrary universal temperature cutoff is imposed.

## Occupancy distribution

| occupancy | rows | percent |
| --- | --- | --- |
| 0 | 8228 | 81.23 |
| 1 | 459 | 4.532 |
| 2 | 748 | 7.385 |
| 3 | 694 | 6.852 |

An always-empty guess would obtain **81.23% accuracy** across
these rows without detecting any occupied state. This is a descriptive reference, not a
held-out score. Keep the natural class distribution during Phase 1.

Counts measure recorded rows, not a reliable fraction of total elapsed time: intervals vary
and long gaps exist. Do not assume a missing interval means an empty room or multiply every
row by 30 seconds to claim monitored duration.

## Plot interpretations

- 01: class imbalance, shown as row counts and percentages.
- 02: CO2 concentration and the supplied trend by occupancy; overlap limits simple thresholds.
- 03: motion-detection percentage within each class. PIR measures movement, not people.
- 04/05/06: individual light/sound/temperature channels and their averages, preserving spatial comparison.
- 07: Pearson correlations show related or redundant measurements. Both Pearson and Spearman matrices
  are also provided as CSV.
- 08: linear Pearson and rank-based Spearman associations with the integer occupancy label.
  These are descriptive associations, not predictive importance or causal effects.
- 09: the recording timeline breaks lines at gaps longer than 10 minutes.
- 10: occupancy by calendar date shows uneven class coverage.

In boxplots, the box is the middle 50%, the line is the median, and whiskers extend to the
most extreme data within 1.5 IQR of the quartiles. Dots beyond the whiskers are retained.
The plots show all valid observations; rare extremes are not hidden.

Among the recommended channels, **S1_Light** has the largest absolute Pearson
association with occupancy (**r = 0.849**).
This is not a proof of causation or generalization to other rooms. Ambient conditions,
lighting habits, calibration and temporal dependence can contribute to associations.
No significance tests are reported because adjacent rows are not independent replicates.

Median sensor values by class (temperature in degrees C, light in lux, sound in V, CO2 in ppm):

| Room_Occupancy_Count | Avg_Temp | Avg_Light | Avg_Sound | S5_CO2 |
| --- | --- | --- | --- | --- |
| 0 | 25.29 | 0 | 0.065 | 355 |
| 1 | 25.92 | 63.75 | 0.14 | 440 |
| 2 | 26 | 95.38 | 0.44 | 685 |
| 3 | 26.22 | 144.4 | 0.3337 | 870 |

These medians describe typical readings, while the boxplots show substantial overlap and
extremes. Sensor responses need not be strictly monotonic with occupancy. For example,
S4_Light behaves differently from the other light channels. This supports retaining separate
channels for the baseline rather than assuming the room average contains all useful information.

PIR detection proportions (0 to 1):

| Room_Occupancy_Count | S6_PIR | S7_PIR | PIR_Any |
| --- | --- | --- | --- |
| 0 | 0.002431 | 0.001458 | 0.002917 |
| 1 | 0.3442 | 0.04139 | 0.3529 |
| 2 | 0.4786 | 0.4278 | 0.639 |
| 3 | 0.5432 | 0.6556 | 0.7695 |

Class counts by date:

| Timestamp | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 2017-12-22 | 637 | 205 | 257 | 363 |
| 2017-12-23 | 1997 | 254 | 398 | 130 |
| 2017-12-24 | 1064 | 0 | 0 | 0 |
| 2017-12-25 | 1716 | 0 | 0 | 0 |
| 2017-12-26 | 1063 | 0 | 0 | 0 |
| 2018-01-10 | 703 | 0 | 93 | 201 |
| 2018-01-11 | 1048 | 0 | 0 | 0 |

## Final candidate decision

Keep 15 direct sensor channels for the initial baseline. Compare a separate six-input
aggregate representation in Phase 2. The exact names and reasons are in candidate_features.md
and feature_spec.json. The supplied CO2 slope is conditional on verifying its time window.

Averages still use four physical sensors. They do not establish hardware or electricity
savings. This collection cannot by itself establish performance in other rooms or settings.

## Two-minute professor explanation

“I completed the dataset preparation phase for the occupancy controller. I obtained the
original data from UCI and preserved it with checksums. It contains 10,129 observations,
16 sensor-related fields, two time fields and an occupancy label from zero to three.

I checked every column, missing values, duplicates, numeric values and timestamps. No missing
values or duplicates were found, so I did not invent replacement values or remove valid
readings. I kept an auditable chronological copy and added easy-to-explain sensor averages.

My graphs show class imbalance, sensor behaviour and relationships between features.
About 81 percent of rows are empty-room observations, so accuracy alone will not be sufficient.
I also found recording gaps and uneven class coverage across dates, which must guide evaluation.

My handoff is a 15-input baseline candidate list and a six-input aggregate comparison.
I have not trained a model or claimed measured energy savings. Phase 2 will evaluate
occupancy prediction using a carefully designed split and class-wise metrics.”

## Viva questions

**Why this dataset?** Multiple environmental sensor types and actual occupancy labels from 0 to 3.

**Feature versus target?** Features are available measurements. The target is the true occupancy
count we want to predict, never an input.

**What cleaning was needed without missing values?** Schema/type/value checks, timestamp parsing,
traceability, chronological ordering and documented feature creation.

**Why averages?** An interpretable summary of simultaneous same-unit readings. They can hide spatial
information, so original channels are retained and the average representation is optional.

**Why exclude the CO2 slope from the baseline?** Its window alignment is not yet verified as causal.
It remains preserved for investigation.

**Why retain outliers?** Unusual readings may represent rare occupied periods. An IQR flag alone
is not sufficient evidence of a fault.

**Why avoid a random split?** Neighbouring readings can be very similar and make evaluation optimistic.

**Why not use dates as inputs?** The baseline should learn sensor evidence. Dates are useful for
auditing and evaluation but can encode the collection schedule.

**Does correlation prove prediction performance?** No. It describes association here. Evaluate
a model using suitable validation data.

**Where is soft computing?** GA sensor selection and fuzzy control belong to later phases.
This phase establishes their data foundation.

**Have you measured energy savings?** No. No electricity labels are present. Future savings
must be described as simulations under declared power assumptions.
