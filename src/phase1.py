"""Phase 1: inspect, validate, prepare, explore and document. No model training."""
from pathlib import Path
import os
import sys
import json
import platform
import html

ROOT = Path(__file__).resolve().parents[1]
# Optional local dependencies for the delivered run. A normal venv also works.
if (ROOT / "work/python-packages").exists():
    sys.path.append(str(ROOT / "work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from download_data import main as verify_raw

TARGET = "Room_Occupancy_Count"
TEMP = [f"S{i}_Temp" for i in range(1, 5)]
LIGHT = [f"S{i}_Light" for i in range(1, 5)]
SOUND = [f"S{i}_Sound" for i in range(1, 5)]
PIR = ["S6_PIR", "S7_PIR"]
SENSORS = TEMP + LIGHT + SOUND + ["S5_CO2", "S5_CO2_Slope"] + PIR
BASELINE = [c for c in SENSORS if c != "S5_CO2_Slope"]
AGG = ["Avg_Temp", "Avg_Light", "Avg_Sound", "PIR_Any"]
COMPACT = ["Avg_Temp", "Avg_Light", "Avg_Sound", "S5_CO2"] + PIR
CLASSES = [0, 1, 2, 3]
COLORS = ["#5B7186", "#2B8A83", "#DA9439", "#A56185"]
PLOTS = ROOT / "outputs/plots"
TABLES = ROOT / "outputs/tables"
PROCESSED = ROOT / "data/processed"
DOCS = ROOT / "docs"

def md(frame):
    """Make a small Markdown table without an extra dependency."""
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v).replace("|", "/")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |"
              for row in frame.fillna("").itertuples(index=False, name=None)]
    return "\n".join(lines)

def save(fig, name):
    fig.savefig(PLOTS / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

def info(column):
    if column == "Date": return "YYYY/MM/DD", "Recorded date; metadata only"
    if column == "Time": return "HH:MM:SS", "Recorded time; timezone not supplied"
    if column == TARGET: return "people", "Manually recorded occupancy class, 0 to 3"
    if column.endswith("_Temp"): return "degrees C", f"Temperature at node {column[:2]}"
    if column.endswith("_Light"): return "lux", f"Light at node {column[:2]}"
    if column.endswith("_Sound"): return "V", f"Sound amplifier voltage at node {column[:2]}, not decibels"
    if column == "S5_CO2": return "ppm", "CO2 concentration at node S5"
    if column == "S5_CO2_Slope": return "unspecified", "Supplied CO2 window slope; alignment and time units unverified"
    return "0/1", f"Motion detection at node {column[:2]}; not a count of people"

def boxes(data, columns, title, units, name):
    n = len(columns)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, min(3, n), figsize=(13, 3.8 * rows),
                             squeeze=False, layout="constrained")
    for ax, col, unit in zip(axes.flat, columns, units):
        groups = [data.loc[data[TARGET] == c, col].values for c in CLASSES]
        bp = ax.boxplot(groups, tick_labels=CLASSES, patch_artist=True,
                        flierprops={"markersize": 2, "alpha": .3, "marker": "."},
                        medianprops={"color": "#172B40", "linewidth": 1.5})
        for box, color in zip(bp["boxes"], COLORS):
            box.set_facecolor(color)
            box.set_alpha(.65)
        ax.set(title=col, xlabel="Occupancy (people)", ylabel=unit)
        ax.grid(axis="y", alpha=.18)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(title + "\nBox: middle 50%; line: median; dots: beyond 1.5 x IQR whiskers", fontsize=14)
    save(fig, name)

def main():
    for path in (PLOTS, TABLES, PROCESSED, DOCS):
        path.mkdir(parents=True, exist_ok=True)
    verify_raw()
    raw = pd.read_csv(ROOT / "data/raw/Occupancy_Estimation.csv", float_precision="round_trip")
    expected = ["Date", "Time"] + SENSORS + [TARGET]
    if raw.columns.tolist() != expected or raw.shape != (10129, 19):
        raise ValueError("Unexpected dataset version, shape or schema.")
    if raw.isna().any().any() or raw.duplicated().any():
        raise ValueError("Missing values or exact duplicates need explicit investigation.")
    data = raw.copy()
    for col in SENSORS + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if not np.isfinite(data[SENSORS + [TARGET]].to_numpy()).all():
        raise ValueError("Non-finite numeric data.")
    if not data[TARGET].isin(CLASSES).all():
        raise ValueError("Unexpected occupancy label.")
    for col in PIR:
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} is not binary.")
    if (data[LIGHT + SOUND + ["S5_CO2"]] < 0).any().any():
        raise ValueError("Negative measurement in a nonnegative channel.")
    data[TARGET] = data[TARGET].astype("int64")
    data["Timestamp"] = pd.to_datetime(
        data.Date + " " + data.Time, format="%Y/%m/%d %H:%M:%S", errors="raise")
    if data.Timestamp.duplicated().any():
        raise ValueError("Repeated timestamps need inspection.")
    original_order = bool(data.Timestamp.is_monotonic_increasing)
    # CSV row 1 is the header; the first observation is on line 2.
    data["Source_Row"] = np.arange(2, len(data) + 2)
    data = data.sort_values(["Timestamp", "Source_Row"], kind="stable").reset_index(drop=True)
    delta = data.Timestamp.diff().dt.total_seconds()
    # Descriptive groups, not known experimental session identifiers.
    data["Recording_Segment"] = (delta.gt(600).cumsum() + 1).astype("int64")
    for name, cols in [("Temp", TEMP), ("Light", LIGHT), ("Sound", SOUND)]:
        # Only simultaneous readings in the same units; no future or target values.
        data["Avg_" + name] = data[cols].mean(axis=1)
    data["PIR_Any"] = data[PIR].max(axis=1).astype("int64")

    records = []
    for col in expected:
        unit, meaning = info(col)
        s = raw[col]
        records.append({
            "column": col, "role": "target" if col == TARGET else
            ("metadata" if col in ["Date", "Time"] else "sensor_related"),
            "raw_dtype": str(s.dtype), "processed_dtype": str(data[col].dtype),
            "units": unit, "missing": int(s.isna().sum()), "unique_values": int(s.nunique()),
            "min": s.min(), "max": s.max(), "example": s.iloc[0], "meaning": meaning})
    profile = pd.DataFrame(records)
    profile.to_csv(TABLES / "column_profile.csv", index=False)
    raw[SENSORS + [TARGET]].describe().T.to_csv(TABLES / "raw_numeric_statistics.csv", index_label="feature")
    data[SENSORS + AGG + [TARGET]].describe().T.to_csv(TABLES / "processed_numeric_statistics.csv", index_label="feature")
    counts = data[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    classes = counts.rename("rows").rename_axis("occupancy").reset_index()
    classes["percent"] = classes.rows / len(data) * 100
    classes.to_csv(TABLES / "occupancy_distribution.csv", index=False)
    day = pd.crosstab(data.Timestamp.dt.strftime("%Y-%m-%d"), data[TARGET]).reindex(columns=CLASSES, fill_value=0)
    day.to_csv(TABLES / "occupancy_by_date.csv", index_label="date")
    delta.dropna().value_counts().sort_index().rename("count").rename_axis(
        "interval_seconds").to_csv(TABLES / "sampling_intervals.csv")
    gaps = data.loc[delta.gt(60), ["Source_Row", "Timestamp", "Recording_Segment"]].copy()
    gaps["previous_timestamp"] = data.Timestamp.shift(1)[delta.gt(60)]
    gaps["gap_seconds"] = delta[delta.gt(60)]
    gaps.to_csv(TABLES / "gaps_over_60_seconds.csv", index=False)
    segments = data.groupby("Recording_Segment").agg(
        start=("Timestamp", "min"), end=("Timestamp", "max"), rows=(TARGET, "size"),
        classes=(TARGET, lambda x: ",".join(str(i) for i in sorted(x.unique()))))
    segments.to_csv(TABLES / "recording_segments.csv")
    summary = data.groupby(TARGET)[SENSORS + AGG].agg(["count", "mean", "std", "min", "median", "max"])
    summary.columns = [f"{a}__{b}" for a, b in summary.columns]
    summary.to_csv(TABLES / "sensor_statistics_by_occupancy.csv")
    # Flags alone do not prove faults, particularly for rare occupied conditions.
    flags = []
    for col in SENSORS:
        q1, q3 = data[col].quantile([.25, .75])
        low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        flags.append({"feature": col, "lower_fence": low, "upper_fence": high,
                      "flagged_rows": int((data[col].lt(low) | data[col].gt(high)).sum()),
                      "action": "retained"})
    pd.DataFrame(flags).to_csv(TABLES / "iqr_review.csv", index=False)
    columns = SENSORS + AGG + [TARGET]
    pearson = data[columns].corr()
    spearman = data[columns].rank(method="average").corr()
    pearson.to_csv(TABLES / "pearson_correlation.csv")
    spearman.to_csv(TABLES / "spearman_correlation.csv")
    correlations = pd.DataFrame({
        "feature": SENSORS + AGG,
        "pearson_with_occupancy": pearson.loc[SENSORS + AGG, TARGET].values,
        "spearman_with_occupancy": spearman.loc[SENSORS + AGG, TARGET].values})
    correlations["absolute_pearson"] = correlations.pearson_with_occupancy.abs()
    correlations = correlations.sort_values("absolute_pearson", ascending=False)
    correlations.to_csv(TABLES / "feature_target_correlations.csv", index=False)
    pairs = [{"feature_a": a, "feature_b": b, "pearson_r": float(pearson.loc[a,b])}
             for i,a in enumerate(SENSORS) for b in SENSORS[i+1:] if abs(pearson.loc[a,b]) >= .8]
    pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson_r"]).to_csv(
        TABLES / "high_correlation_pairs.csv", index=False)
    pir_rates = data.groupby(TARGET)[PIR + ["PIR_Any"]].mean()
    pir_rates.to_csv(TABLES / "pir_detection_rates.csv")
    data.to_csv(PROCESSED / "occupancy_cleaned.csv", index=False)
    data[BASELINE + [TARGET]].to_csv(PROCESSED / "baseline_features.csv", index=False)
    data[COMPACT + [TARGET]].to_csv(PROCESSED / "compact_features.csv", index=False)
    spec = {
        "target": TARGET, "classes": CLASSES, "recommended_baseline": BASELINE,
        "optional_compact_comparison": COMPACT, "supplied_sensor_features": SENSORS,
        "conditional_feature": {"S5_CO2_Slope": "Exclude until trailing-only window calculation is verified."},
        "excluded_metadata": ["Date", "Time", "Timestamp", "Source_Row", "Recording_Segment"],
        "eda_only_alternative": {"PIR_Any": "OR of the two PIR channels; loses location information."},
        "row_order": "Same chronological row order in all three processed tables.",
        "split_status": "No split or training. All rows used for descriptive EDA."
    }
    (DOCS / "feature_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    checks = {
        "raw_shape": list(raw.shape), "processed_shape": list(data.shape),
        "missing_cells": int(raw.isna().sum().sum()), "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_timestamps": int(data.Timestamp.duplicated().sum()),
        "raw_already_chronological": original_order,
        "rows_removed": 0, "values_imputed": 0, "values_clipped": 0,
        "timestamp_start": str(data.Timestamp.min()), "timestamp_end": str(data.Timestamp.max()),
        "calendar_dates": int(data.Timestamp.dt.date.nunique()),
        "median_interval_seconds": float(delta.median()),
        "gaps_over_60_seconds": int(delta.gt(60).sum()),
        "segments_gap_threshold_600_seconds": int(data.Recording_Segment.nunique()),
        "constant_numeric_features": [c for c in SENSORS if data[c].nunique() == 1],
        "all_numeric_finite": True, "pir_values_binary": True,
        "occupancy_counts": {str(k): int(v) for k,v in counts.items()},
        "empty_room_percent": float(counts.loc[0] / len(data) * 100),
        "baseline_input_count": len(BASELINE), "compact_input_count": len(COMPACT)}
    (TABLES / "quality_summary.json").write_text(json.dumps(checks, indent=2) + "\n")
    make_plots(data, counts, day, pir_rates, pearson, correlations)
    write_docs(data, checks, classes, profile, day, correlations, pir_rates)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    (TABLES / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    print("Report:", ROOT / "outputs/phase1_report.html")

def make_plots(data, counts, day, pir_rates, pearson, correlations):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "svg.hashsalt": "sc-project-phase1"})
    fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
    bars = ax.bar(CLASSES, counts.values, color=COLORS, width=.65)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, count+120,
                f"{count:,}\n{count / len(data):.2%}", ha="center", fontsize=11)
    ax.set(xticks=CLASSES, xlabel="Room occupancy (people)", ylabel="Recorded rows",
           ylim=(0, counts.max()*1.19), title="Empty-room readings dominate the dataset")
    ax.grid(axis="y", alpha=.18)
    save(fig, "01_occupancy_distribution")
    boxes(data, ["S5_CO2", "S5_CO2_Slope"], "CO2 by occupancy",
          ["CO2 (ppm)", "Slope (units unspecified)"], "02_co2_by_occupancy")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for i,col in enumerate(pir_rates):
        ax.bar(np.array(CLASSES)+(i-1)*.24, pir_rates[col]*100, width=.23,
               label=col, color=["#2B8A83", "#496D9B", "#DA9439"][i])
    ax.set(xticks=CLASSES, xlabel="Occupancy (people)", ylabel="Rows with motion detected (%)",
           ylim=(0, 105), title="PIR measures motion, so an occupied room can show zero")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.18)
    save(fig, "03_pir_by_occupancy")
    boxes(data, LIGHT+["Avg_Light"], "Light by occupancy", ["Light (lux)"]*5, "04_light_by_occupancy")
    boxes(data, SOUND+["Avg_Sound"], "Sound by occupancy", ["Amplifier output (V)"]*5, "05_sound_by_occupancy")
    boxes(data, TEMP+["Avg_Temp"], "Temperature by occupancy", ["Temperature (degrees C)"]*5, "06_temperature_by_occupancy")
    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    im = ax.imshow(pearson, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(pearson)), yticks=range(len(pearson)),
           xticklabels=pearson.columns, yticklabels=pearson.columns,
           title="Pearson correlations: associations and redundant measurements")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(len(pearson)):
        for j in range(len(pearson)):
            v = pearson.iloc[i,j]
            ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=6,
                    color="white" if abs(v)>.65 else "#172B40")
    fig.colorbar(im, ax=ax, shrink=.72, label="Pearson r (not causal influence)")
    save(fig, "07_correlation_matrix")
    fig, ax = plt.subplots(figsize=(10,8), layout="constrained")
    cc = correlations.sort_values("absolute_pearson")
    ypos = np.arange(len(cc))
    ax.barh(ypos-.18,cc.pearson_with_occupancy,height=.34,label="Pearson",color="#496D9B")
    ax.barh(ypos+.18,cc.spearman_with_occupancy,height=.34,label="Spearman",color="#2B8A83")
    ax.set(yticks=ypos,yticklabels=cc.feature,xlim=(-1,1),xlabel="Correlation with occupancy count",
           title="Exploratory associations do not establish predictive importance")
    ax.axvline(0,color="#AAAAAA",linewidth=.7);ax.legend(loc="lower left",frameon=False)
    save(fig, "08_feature_target_associations")
    fig, axes = plt.subplots(3,1,figsize=(13,8),sharex=True,layout="constrained")
    for ax,col,label in zip(axes,[TARGET,"S5_CO2","Avg_Light"],["People","CO2 (ppm)","Average light (lux)"]):
        for _,segment in data.groupby("Recording_Segment"):
            ax.plot(segment.Timestamp,segment[col],linewidth=.7,color="#2B8A83")
        ax.set_ylabel(label);ax.grid(alpha=.18)
    axes[0].set_yticks(CLASSES)
    axes[0].set_title("Recording timeline: lines break at gaps longer than 10 minutes")
    axes[-1].set_xlabel("Recorded timestamp (timezone unspecified)")
    fig.autofmt_xdate()
    save(fig, "09_recording_timeline")
    fig,ax=plt.subplots(figsize=(11,5),layout="constrained")
    bottom=np.zeros(len(day))
    for col,color in zip(CLASSES,COLORS):
        ax.bar(day.index,day[col],bottom=bottom,label=f"{col} people",color=color)
        bottom+=day[col].values
    ax.set(xlabel="Recording date",ylabel="Recorded rows",
           title="Some recording dates contain only the empty-room class")
    ax.legend(frameon=False,ncol=4);ax.tick_params(axis="x",rotation=25)
    save(fig, "10_occupancy_by_date")

def write_docs(data, checks, classes, profile, day, correlations, pir_rates):
    added = pd.DataFrame([
        ["Timestamp","datetime64[ns]","Date + Time, strict parsing; timezone unknown","Audit and future splits"],
        ["Source_Row","int64","Original CSV line number starting at 2","Traceability"],
        ["Recording_Segment","int64","New group after gap > 600 seconds","Descriptive grouping, not experimental label"],
        ["Avg_Temp","float64","(S1_Temp + S2_Temp + S3_Temp + S4_Temp) / 4","Optional compact feature, degrees C"],
        ["Avg_Light","float64","(S1_Light + S2_Light + S3_Light + S4_Light) / 4","Optional compact feature, lux"],
        ["Avg_Sound","float64","(S1_Sound + S2_Sound + S3_Sound + S4_Sound) / 4","Optional compact feature, V"],
        ["PIR_Any","int64","max(S6_PIR, S7_PIR)","Logical OR, not a people count"]
    ],columns=["field","dtype","definition","purpose"])
    added.to_csv(TABLES/"derived_feature_dictionary.csv",index=False)
    dictionary = "# Data dictionary\n\nRaw: 10,129 rows x 19 columns. "
    dictionary += "16 sensor-related fields, two time fields and one target. CO2 slope is source-derived.\n\n"
    dictionary += md(profile)+"\n\n## Added fields\n\n"+md(added)
    dictionary += "\n\nCSV does not store dtype metadata. Load Timestamp with parse_dates. "
    dictionary += "Date/Time remain strings; numeric originals keep their values and units. "
    dictionary += "Units follow UCI metadata; the slope time units are unspecified.\n"
    (DOCS/"data_dictionary.md").write_text(dictionary)
    candidates=[]
    for col in BASELINE:
        if col in TEMP: why="Local temperature and spatial variation; also sensitive to ambient conditions."
        elif col in LIGHT: why="Local light information; association may depend on this room's lighting habits."
        elif col in SOUND: why="Local activity; amplifier voltage is not decibels."
        elif col in PIR: why="Motion coverage at this location; zero does not prove the room is empty."
        else: why="Direct CO2 concentration; may lag arrivals and persist after departures."
        candidates.append([col,info(col)[0],why])
    candidates=pd.DataFrame(candidates,columns=["candidate_input","unit","reason"])
    candidates.to_csv(TABLES/"candidate_features.csv",index=False)
    candidate_doc = "# Phase 2 candidate inputs\n\n## Recommended baseline: 15 direct sensor channels\n\n"
    candidate_doc += md(candidates)
    candidate_doc += """

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
"""
    (DOCS/"candidate_features.md").write_text(candidate_doc)
    medians = data.groupby(TARGET)[["Avg_Temp", "Avg_Light", "Avg_Sound", "S5_CO2"]].median().reset_index()
    medians.to_csv(TABLES/"median_sensors_by_occupancy.csv", index=False)
    top=correlations.loc[correlations.feature.isin(BASELINE)].iloc[0]
    notes=f"""# Phase 1 notes: Adaptive Smart Room Occupancy & Energy Controller

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
- Processed: **{data.shape[0]:,} rows x {data.shape[1]} columns**: seven added fields.
- Missing cells: **0**. Exact duplicate rows: **0**. Duplicate timestamps: **0**.
- Numeric values are finite, PIR is binary, and labels are exactly 0, 1, 2, 3.
- No constant numeric sensor features.
- Recorded range: **{checks['timestamp_start']}** to **{checks['timestamp_end']}**.
- **{checks['calendar_dates']} calendar dates**, median interval **{checks['median_interval_seconds']:.0f} seconds**.
- **{checks['gaps_over_60_seconds']} intervals over 60 seconds**.
- A descriptive gap threshold of 600 seconds creates **{checks['segments_gap_threshold_600_seconds']} segments**.

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

{md(classes)}

An always-empty guess would obtain **{checks['empty_room_percent']:.2f}% accuracy** across
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

Among the recommended channels, **{top['feature']}** has the largest absolute Pearson
association with occupancy (**r = {top['pearson_with_occupancy']:.3f}**).
This is not a proof of causation or generalization to other rooms. Ambient conditions,
lighting habits, calibration and temporal dependence can contribute to associations.
No significance tests are reported because adjacent rows are not independent replicates.

Median sensor values by class (temperature in degrees C, light in lux, sound in V, CO2 in ppm):

{md(medians)}

These medians describe typical readings, while the boxplots show substantial overlap and
extremes. Sensor responses need not be strictly monotonic with occupancy. For example,
S4_Light behaves differently from the other light channels. This supports retaining separate
channels for the baseline rather than assuming the room average contains all useful information.

PIR detection proportions (0 to 1):

{md(pir_rates.reset_index())}

Class counts by date:

{md(day.reset_index())}

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
"""
    (DOCS/"phase1_notes.md").write_text(notes)
    write_html(checks, classes, profile, candidates)

def write_html(checks, classes, profile, candidates):
    captions=[
        ("01_occupancy_distribution","Class balance","Empty-room readings dominate; accuracy alone is insufficient."),
        ("02_co2_by_occupancy","CO2 and supplied trend","The supplied slope is excluded from the recommended baseline pending window verification."),
        ("03_pir_by_occupancy","Motion and occupancy","Rates are within each occupancy class. Motion is not a people count."),
        ("04_light_by_occupancy","Light","Separate positions can carry different information."),
        ("05_sound_by_occupancy","Sound","Values are amplifier volts, not decibels. Unusual readings are retained."),
        ("06_temperature_by_occupancy","Temperature","Temperature changes slowly and can reflect ambient conditions."),
        ("07_correlation_matrix","Correlation matrix","Strong sensor-to-sensor association can indicate redundancy."),
        ("08_feature_target_associations","Feature associations","Pearson measures linear association; Spearman measures rank association. Neither is predictive importance."),
        ("09_recording_timeline","Recording timeline","No data are inserted into the gaps."),
        ("10_occupancy_by_date","Class coverage across dates","Class 1 occurs only on the first two dates. A simple late holdout will not evaluate all classes.")]
    cards="".join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
                  f'<img src="plots/{name}.png" alt="{html.escape(title)}"></section>'
                  for name,title,caption in captions)
    page="""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SC Project: Phase 1</title>
<style>body{margin:0;background:#eef2f5;color:#172b40;font:16px/1.6 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:38px 24px 70px}header{padding:28px 0 22px;border-bottom:3px solid #2b8a83}
h1{font-size:36px;line-height:1.2;max-width:900px}h2{font-size:24px}.kicker{font-size:14px;letter-spacing:.12em;color:#2b746f}
.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.metric{background:#fff;padding:20px 24px;flex:1}
.metric strong{display:block;font-size:30px}.metric span{color:#526779}
section{background:white;padding:24px;margin:24px 0;border-radius:8px}section img{width:100%;height:auto;display:block;margin-top:18px}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{border-bottom:1px solid #dee5eb;padding:9px;text-align:left}
th{background:#edf5f4}a{color:#166e75}.scroll{overflow:auto}.note{border-left:4px solid #da9439;padding-left:18px}
@media print{body{background:white}section{break-inside:avoid;padding:8px 0}main{padding:0}.scroll{overflow:visible}}
</style></head><body><main><header><div class="kicker">SOFT COMPUTING MINOR PROJECT / PHASE 1</div>
<h1>Adaptive Smart Room Occupancy &amp; Energy Controller</h1>
<p>Dataset preparation and exploration. No model training, GA, fuzzy control or energy estimates.</p></header>"""
    page+=f"""<div class="metrics"><div class="metric"><strong>10,129</strong><span>recorded observations</span></div>
<div class="metric"><strong>0–3</strong><span>people: target classes</span></div>
<div class="metric"><strong>{checks['empty_room_percent']:.2f}%</strong><span>empty-room rows</span></div>
<div class="metric"><strong>0</strong><span>missing cells or duplicate rows</span></div></div>
<section><h2>What was prepared</h2><p>The original UCI ZIP and CSV are preserved with checksums.
The processed data keep all rows and add timestamp traceability, recording segments and four
aggregate indicators. No values were imputed or clipped.</p>
<p class="note">The CSV spans seven calendar dates with recording gaps. Class coverage differs by
date. A future evaluation must account for both time and the four occupancy classes.</p>
<p><a href="../docs/phase1_notes.md">Detailed notes and viva explanation</a> /
<a href="../docs/candidate_features.md">Candidate feature rationale</a> /
<a href="../docs/data_dictionary.md">Data dictionary</a> /
<a href="../README.md">Run instructions</a></p></section>
<section><h2>Occupancy counts</h2>{classes.round(2).to_html(index=False,border=0)}</section>"""
    page+=cards
    page+="<section><h2>Recommended Phase 2 inputs: 15 direct channels</h2>"
    page+=candidates.to_html(index=False,border=0)
    page+="""<p>Optional six-input comparison: Avg_Temp, Avg_Light, Avg_Sound, S5_CO2,
S6_PIR and S7_PIR. Averages still require all their underlying physical sensors.</p>
<p>The target, timestamp fields and audit IDs are excluded from inputs. No train/test split
has been created. All rows were used for descriptive EDA.</p></section>
<section><h2>Every original column</h2><div class="scroll">"""
    page+=profile.to_html(index=False,border=0)+"</div></section>"
    page+="""<section><h2>Source and limits</h2><p>Singh, A. &amp; Chaudhari, S. (2018).
<a href="https://doi.org/10.24432/C5P605">Room Occupancy Estimation, UCI Machine Learning Repository</a>.
CC BY 4.0. Original source details and hashes are in data/raw/source_manifest.json.</p>
<p>This is occupancy data from one setup. There are no measured electricity labels.
Future energy savings must be explicitly simulated. Performance in other rooms remains
unverified.</p></section></main></body></html>"""
    (ROOT/"outputs/phase1_report.html").write_text(page)

if __name__ == "__main__":
    main()
