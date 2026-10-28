"""Train and evaluate the fixed Phase 2 baseline, keeping Phase 1 intact."""
from pathlib import Path
import os, sys, json, hashlib, platform
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():
    sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from download_data import main as verify_raw

OUT=ROOT/"outputs/phase2"
TABLES=OUT/"tables"
PLOTS=OUT/"plots"
MODELS=ROOT/"models/phase2"
LABELS=[0,1,2,3]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(y,pred,model_name,partition):
    cm=confusion_matrix(y,pred,labels=LABELS)
    support=cm.sum(axis=1)
    predicted=cm.sum(axis=0)
    tp=np.diag(cm)
    recall=np.divide(tp,support,out=np.full(4,np.nan),where=support>0)
    precision=np.divide(tp,predicted,out=np.full(4,np.nan),where=predicted>0)
    # F1 = 2TP/(true support + predicted support). Undefined only if both are absent.
    f1=np.divide(2*tp,support+predicted,out=np.full(4,np.nan),where=(support+predicted)>0)
    table=pd.DataFrame({"class":LABELS,"precision":precision,"recall":recall,"f1":f1,
                        "support":support,"predicted_count":predicted})
    table.to_csv(TABLES/f"{model_name}_{partition}_per_class.csv",index=False)
    pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(TABLES/f"{model_name}_{partition}_confusion.csv",index_label="true_class")
    present=np.array(LABELS)[support>0]
    metrics={"model":model_name,"partition":partition,"rows":len(y),
             "accuracy":accuracy_score(y,pred),
             "macro_f1_present_classes":f1_score(y,pred,labels=present,average="macro",zero_division=0),
             "macro_f1_all_four":f1_score(y,pred,labels=LABELS,average="macro",zero_division=0) if len(present)==4 else np.nan,
             "balanced_accuracy_present_classes":float(np.nanmean(recall)),
             "weighted_f1":f1_score(y,pred,labels=LABELS,average="weighted",zero_division=0),
             "true_classes":",".join(map(str,present))}
    return metrics,cm

def save(fig,name):
    fig.savefig(PLOTS/f"{name}.png",dpi=160,bbox_inches="tight")
    fig.savefig(PLOTS/f"{name}.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)

def plot_confusion(cm,name,title):
    norm=np.divide(cm,cm.sum(axis=1)[:,None],out=np.full((4,4),np.nan),where=cm.sum(axis=1)[:,None]>0)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,values,label in zip(axes,[cm,norm],["Counts","Fraction of true class"]):
        ax.imshow(np.nan_to_num(values),cmap="Blues",vmin=0,vmax=max(float(np.nanmax(values)),1e-9))
        for i in LABELS:
            for j in LABELS:
                value=values[i,j]
                text="n/a" if np.isnan(value) else (f"{value:.1%}" if label!="Counts" else str(int(value)))
                ax.text(j,i,text,ha="center",va="center",
                        color="white" if np.isfinite(value) and value>np.nanmax(values)*.5 else "#172b40")
        ax.set(xticks=LABELS,yticks=LABELS,xlabel="Predicted people",ylabel="Actual people",title=label)
    fig.suptitle(title,fontsize=14)
    save(fig,name)

def main():
    for p in [TABLES,PLOTS,MODELS,ROOT/"data/splits"]:
        p.mkdir(parents=True,exist_ok=True)
    verify_raw()
    spec=json.loads((ROOT/"docs/feature_spec.json").read_text())
    protocol=json.loads((ROOT/"docs/phase2_protocol.json").read_text())
    features=spec["recommended_baseline"]
    target=spec["target"]
    source=ROOT/"data/processed/occupancy_cleaned.csv"
    d=pd.read_csv(source,parse_dates=["Timestamp"],float_precision="round_trip")
    raw=pd.read_csv(ROOT/"data/raw/Occupancy_Estimation.csv",float_precision="round_trip")
    # Independently ensure features and labels still match each referenced raw row.
    original=raw.iloc[d.Source_Row.to_numpy()-2][features+[target]].reset_index(drop=True)
    pd.testing.assert_frame_equal(d[features+[target]],original)
    assert d.Timestamp.is_monotonic_increasing and d.Source_Row.is_unique
    assert len(features)==15 and target not in features and not set(features)&set(spec["excluded_metadata"])
    assert "S5_CO2_Slope" not in features and np.isfinite(d[features].to_numpy()).all()
    t=d.Timestamp
    partition=pd.Series("embargo",index=d.index)
    partition[t<pd.Timestamp(protocol["train_before"])]="train"
    partition[(t>=pd.Timestamp(protocol["validation_start"]))&(t<pd.Timestamp(protocol["validation_before"]))]="validation"
    partition[t>=pd.Timestamp(protocol["test_start"])]="test"
    masks={name:partition.eq(name) for name in ["train","validation","test"]}
    for name in ["train","validation"]:
        assert set(d.loc[masks[name],target])==set(LABELS),f"{name} needs all classes"
    assert t[masks["train"]].max()+pd.Timedelta(minutes=10)<t[masks["validation"]].min()
    assert t[masks["validation"]].max()+pd.Timedelta(minutes=10)<t[masks["test"]].min()
    manifest=d[["Source_Row","Timestamp",target]].copy()
    manifest["partition"]=partition
    manifest.to_csv(ROOT/"data/splits/phase2_split_manifest.csv",index=False)
    coverage=pd.crosstab(partition,d[target]).reindex(index=["train","validation","test","embargo"],columns=LABELS,fill_value=0)
    coverage.to_csv(TABLES/"split_class_counts.csv",index_label="partition")
    ranges=d.assign(partition=partition).groupby("partition").agg(rows=(target,"size"),start=("Timestamp","min"),end=("Timestamp","max"))
    ranges.to_csv(TABLES/"split_ranges.csv")
    X=d[features];y=d[target]
    models={"random_forest":RandomForestClassifier(**protocol["estimator"]),
            "always_empty":DummyClassifier(strategy="constant",constant=0)}
    results=[];predictions=[];matrices={}
    # Every fit uses the exact same training mask; no data-dependent tuning.
    for name,model in models.items():
        model.fit(X.loc[masks["train"]],y.loc[masks["train"]])
        for part,mask in {**masks,"january_test_subset":masks["test"]&t.ge("2018-01-01")}.items():
            pred=model.predict(X.loc[mask])
            metrics,cm=evaluate(y.loc[mask],pred,name,part)
            results.append(metrics);matrices[(name,part)]=cm
            if part in ["train","validation","test"]:
                rows=d.loc[mask,["Source_Row","Timestamp",target]].copy()
                rows["partition"]=part;rows["model"]=name;rows["prediction"]=pred
                if name=="random_forest":
                    probs=model.predict_proba(X.loc[mask])
                    for i,cls in enumerate(model.classes_): rows[f"probability_{cls}"]=probs[:,i]
                predictions.append(rows)
    metrics=pd.DataFrame(results)
    metrics.to_csv(TABLES/"metrics.csv",index=False)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(OUT/"predictions.csv",index=False)
    forest=models["random_forest"]
    model_path=MODELS/"random_forest_baseline.joblib"
    joblib.dump(forest,model_path,compress=3)
    loaded=joblib.load(model_path)
    assert np.array_equal(loaded.predict(X.loc[masks["test"]]),forest.predict(X.loc[masks["test"]]))
    assert np.allclose(loaded.predict_proba(X.iloc[:25]).sum(axis=1),1)
    importance=pd.DataFrame({"feature":features,"impurity_importance":forest.feature_importances_}).sort_values("impurity_importance")
    importance.to_csv(TABLES/"training_feature_importance.csv",index=False)
    metadata={"model":"RandomForestClassifier","features":features,"target":target,"classes":LABELS,
              "trained_on":"2017-12-22 only; not refitted on validation or test",
              "parameters":forest.get_params(),"processed_data_sha256":digest(source),
              "protocol_sha256":digest(ROOT/"docs/phase2_protocol.json"),"model_sha256":digest(model_path),
              "environment":{"python":platform.python_version(),"numpy":np.__version__,
                             "pandas":pd.__version__,"scikit_learn":sklearn.__version__,"joblib":joblib.__version__},
              "limitations":["Test lacks true class 1","All data explored in Phase 1","Single room and one fixed temporal split",
                             "Forest probabilities are not calibrated confidence"]}
    (MODELS/"model_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.hashsalt":"sc-phase2"})
    for part in ["validation","test"]:
        plot_confusion(matrices[("random_forest",part)],f"{part}_confusion",f"Random Forest: {part}"+(" (no true class 1)" if part=="test" else " (all four classes)"))
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    bottom=np.zeros(4)
    for cls,color in zip(LABELS,["#5b7186","#2b8a83","#da9439","#a56185"]):
        ax.bar(coverage.index,coverage[cls],bottom=bottom,label=f"{cls} people",color=color)
        bottom+=coverage[cls].to_numpy()
    ax.set(ylabel="Rows",title="Chronological partition coverage")
    ax.legend(frameon=False);save(fig,"split_coverage")
    fig,ax=plt.subplots(figsize=(10,7),layout="constrained")
    ax.barh(importance.feature,importance.impurity_importance,color="#2b8a83")
    ax.set(xlabel="Training impurity importance (sums to 1)",title="Model usage of features, not causal influence")
    save(fig,"feature_importance")
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    keys=["accuracy","macro_f1_present_classes","balanced_accuracy_present_classes"]
    for ax,part in zip(axes,["validation","test"]):
        for i,name in enumerate(models):
            row=metrics[(metrics.model==name)&(metrics.partition==part)].iloc[0]
            ax.bar(np.arange(3)+(i-.5)*.36,[row[k] for k in keys],width=.35,label=name.replace("_"," "))
        ax.set(xticks=range(3),xticklabels=["Accuracy","Macro F1\npresent classes","Balanced accuracy\npresent classes"],
               ylim=(0,1.1),title=part+(" (classes 0, 2, 3)" if part=="test" else " (classes 0, 1, 2, 3)"))
        ax.legend(frameon=False,fontsize=9)
    save(fig,"baseline_comparison")
    write_report(metrics,coverage,ranges,importance,metadata)
    checks={"status":"passed","training_rows":int(masks["train"].sum()),
            "validation_rows":int(masks["validation"].sum()),"test_rows":int(masks["test"].sum()),
            "embargo_rows":int(partition.eq("embargo").sum()),"disjoint_chronological_partitions":True,
            "at_least_ten_minutes_between_partitions":True,"raw_feature_and_label_match":True,
            "saved_model_predictions_match":True,"probabilities_sum_to_one":True,
            "no_target_or_timestamp_in_model_inputs":True}
    (TABLES/"verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(coverage.to_string());print(metrics.to_string(index=False));print(json.dumps(checks,indent=2))

def write_report(metrics,coverage,ranges,importance,metadata):
    val=metrics[(metrics.model=="random_forest")&(metrics.partition=="validation")].iloc[0]
    test=metrics[(metrics.model=="random_forest")&(metrics.partition=="test")].iloc[0]
    dummy=metrics[(metrics.model=="always_empty")&(metrics.partition=="validation")].iloc[0]
    test_dummy=metrics[(metrics.model=="always_empty")&(metrics.partition=="test")].iloc[0]
    val_classes=pd.read_csv(TABLES/"random_forest_validation_per_class.csv")
    test_classes=pd.read_csv(TABLES/"random_forest_test_per_class.csv")
    def text_table(df): return df.to_string(index=False,float_format=lambda v:f"{v:.4f}")
    notes=f"""# Phase 2: Baseline occupancy prediction

## What was built

A Random Forest using the 15 direct sensor channels from Phase 1 predicts occupancy 0/1/2/3.
An always-empty model supplies a simple comparison. No GA, fuzzy controller or energy calculation
is included. The saved model is exactly the evaluated model, fitted only on December 22.

## Results

Validation accuracy: **{val.accuracy:.2%}**.
Validation macro F1 across all four classes: **{val.macro_f1_all_four:.4f}**.
Validation balanced accuracy: **{val.balanced_accuracy_present_classes:.2%}**.
Always-empty validation accuracy: **{dummy.accuracy:.2%}**.
Later-test accuracy: **{test.accuracy:.2%}**.
Later-test macro F1 over its three supported classes: **{test.macro_f1_present_classes:.4f}**.

**The later test reveals weak occupied-room generalization.** Always-empty test accuracy is
{test_dummy.accuracy:.2%}, close to the forest's {test.accuracy:.2%}. Forest balanced accuracy is
only {test.balanced_accuracy_present_classes:.2%}. It detects 13 of 93 two-person readings
(recall 13.98%) and 6 of 201 three-person readings (recall 2.99%). These results do not justify
deployment or an energy-control claim. The differences suggest a transfer problem across
recording periods, but this experiment does not identify the cause.

Validation is the all-four-class evaluation. Later test has no true class 1 and cannot
establish recall for one-person occupancy. Its high empty-room share can inflate accuracy.
The separate January subset is an additional slice of the same test predictions, not an
independent dataset or an extra model fit.

Full metrics:
~~~text
{text_table(metrics)}
~~~

## Per-class results

Validation:
~~~text
{text_table(val_classes)}
~~~

Later test:
~~~text
{text_table(test_classes)}
~~~

On validation, class 2 recall is 65.58% and class 3 precision is 48.86%, so even the stronger
validation result has meaningful confusion between counts. Undefined entries are explained below.

## Evaluation protocol

The calendar boundaries and model settings are recorded in phase2_protocol.json before fitting.
Training: December 22, 2017. Validation: December 23 after 00:10. Test: December 24 after 00:10
through the remaining recordings. First-ten-minute boundary windows are embargoed and never fitted
or scored. Rows are ordered in time, not shuffled. Embargoed records remain in Phase 1 data.

~~~text
{coverage.to_string()}
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
recall, F1 and a confusion matrix. Validation accuracy is {val.accuracy:.2%}, with macro F1
{val.macro_f1_all_four:.4f}. The later test lacks one-person examples, so I report that limitation
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
"""
    (ROOT/"docs/phase2_notes.md").write_text(notes)
    cards="".join(f'<section><h2>{title}</h2><img src="plots/{name}.png" alt="{title}"></section>'
                  for name,title in [("split_coverage","Class coverage"),("baseline_comparison","Compare with always empty"),
                                     ("validation_confusion","Validation: all four classes"),
                                     ("test_confusion","Later test: class 1 absent"),
                                     ("feature_importance","Training feature importance")])
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC Project: Phase 2</title><style>
body{{background:#eef2f5;color:#172b40;font:16px/1.6 system-ui;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}
h1{{font-size:34px}}section{{background:white;border-radius:8px;padding:24px;margin:24px 0}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{padding:9px;border-bottom:1px solid #dce5e9;text-align:left}}th{{background:#e8f2f1}}
.scroll{{overflow:auto}}a{{color:#166e75}}.note{{border-left:4px solid #da9439;padding-left:18px}}
</style><body><main><p>SOFT COMPUTING MINOR PROJECT / PHASE 2</p><h1>Baseline occupancy prediction</h1>
<p>15 sensor inputs. Random Forest. Chronological evaluation.</p>
<section><h2>Validation accuracy: {val.accuracy:.2%}</h2><p>Four-class macro F1: <strong>{val.macro_f1_all_four:.4f}</strong>.
Always-empty validation accuracy: {dummy.accuracy:.2%}.</p>
<p class="note">Later-test accuracy is {test.accuracy:.2%}, but the test has no true class 1.
Its scores cannot establish one-person detection.</p>
<p class="note"><strong>Later-period occupied detection is weak:</strong> balanced accuracy is
{test.balanced_accuracy_present_classes:.2%}, versus 33.33% for always empty.
Two-person recall is 13.98% and three-person recall is 2.99%. Always-empty accuracy is
{test_dummy.accuracy:.2%}. High overall accuracy hides these failures.</p><p>All data were explored in Phase 1;
the test is excluded from fitting and tuning.</p>
<p><a href="../../docs/phase2_notes.md">Methods and viva notes</a> /
<a href="../../docs/phase2_protocol.json">Fixed protocol</a> /
<a href="tables/metrics.csv">Full metrics</a></p></section>
<section><h2>Results</h2><div class="scroll">{metrics.round(4).to_html(index=False,border=0,na_rep="n/a")}</div>
<p>Macro F1 and balanced accuracy labelled “present classes” average only classes with true support.
Training scores are diagnostics. January is a subset of the later test, not independent evidence.</p></section>
{cards}<section><h2>What is saved</h2><p>Model, input specification, source hashes, split manifest,
row-level predictions, per-class metrics and plots. The saved model uses only December 22 training data.</p>
<p>Feature importance is a training diagnostic. No GA, fuzzy controller or measured energy-savings
claim is included.</p></section></main></body></html>"""
    (OUT/"report.html").write_text(page)

if __name__=="__main__":
    main()
