"""Formal fixed-model comparison. No tuning, reselection, fuzzy logic or energy layer."""
from pathlib import Path
import sys,os,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import phase2
from phase1 import md
OUT=ROOT/"outputs/phase4";TABLES=OUT/"tables";PLOTS=OUT/"plots";MODELS=ROOT/"models/phase4"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    for p in [OUT,TABLES,PLOTS,MODELS,ROOT/"data/examples"]:p.mkdir(parents=True,exist_ok=True)
    meta2=json.loads((ROOT/"models/phase2/model_metadata.json").read_text())
    meta3=json.loads((ROOT/"models/phase3/model_metadata.json").read_text())
    lock=json.loads((ROOT/"models/phase3/selection_locked.json").read_text())
    source=ROOT/"data/processed/occupancy_cleaned.csv"
    splitpath=ROOT/"data/splits/phase2_split_manifest.csv"
    baselinepath=ROOT/"models/phase2/random_forest_baseline.joblib"
    selectedpath3=ROOT/"models/phase3/ga_selected_random_forest.joblib"
    assert sha(source)==meta3["processed_data_sha256"]==meta2["processed_data_sha256"]
    assert sha(splitpath)==meta3["split_manifest_sha256"]
    assert sha(baselinepath)==meta2["model_sha256"] and sha(selectedpath3)==meta3["model_sha256"]
    assert sha(ROOT/"models/phase3/selection_locked.json")==meta3["selection_sha256"]
    data=pd.read_csv(source,parse_dates=["Timestamp"],float_precision="round_trip")
    split=pd.read_csv(splitpath)
    assert data.Source_Row.tolist()==split.Source_Row.tolist()
    train=split.partition.eq("train")
    features=lock["features"];all_features=meta2["features"]
    assert features==meta3["features"]
    params=meta3["parameters"]
    assert params==meta2["parameters"],"Comparison must use identical RF hyperparameters"
    # Fresh fit; no use of validation/test labels or scores.
    selected=RandomForestClassifier(**params)
    selected.fit(data.loc[train,features],data.loc[train,"Room_Occupancy_Count"])
    previous=joblib.load(selectedpath3)
    assert np.array_equal(selected.predict(data[features]),previous.predict(data[features]))
    assert np.allclose(selected.predict_proba(data[features]),previous.predict_proba(data[features]),rtol=0,atol=0)
    selectedpath=MODELS/"ga_occupancy_retrained.joblib"
    joblib.dump(selected,selectedpath,compress=3)
    registry={"default_model":"baseline","reason":"Higher four-class validation macro F1; GA remains a compact comparison.",
              "status":"Academic demonstration; later-period occupied detection is weak.",
              "probabilities":"Uncalibrated class probabilities, not a confidence guarantee.",
              "models":{
                "baseline":{"path":str(baselinepath.relative_to(ROOT)),"sha256":sha(baselinepath),"features":all_features},
                "ga":{"path":str(selectedpath.relative_to(ROOT)),"sha256":sha(selectedpath),"features":features}},
              "classes":[0,1,2,3],"training_rows":int(train.sum()),
              "processed_data_sha256":sha(source),"split_sha256":sha(splitpath),
              "phase3_selection_sha256":sha(ROOT/"models/phase3/selection_locked.json"),
              "phase4_protocol_sha256":sha(ROOT/"docs/phase4_protocol.json"),
              "environment":meta3["environment"],"refit_matches_phase3_predictions_and_probabilities":True}
    (MODELS/"model_registry.json").write_text(json.dumps(registry,indent=2)+"\n")
    baseline=joblib.load(baselinepath)
    phase2.TABLES=TABLES;phase2.PLOTS=PLOTS
    results=[];predictions=[];errors=[];perclass=[]
    for name,model,cols in [("baseline",baseline,all_features),("ga",selected,features)]:
        for part in ["train","validation","test","january_test_subset"]:
            mask=split.partition.eq("test")&data.Timestamp.ge("2018-01-01") if part=="january_test_subset" else split.partition.eq(part)
            rows=data.loc[mask];truth=rows.Room_Occupancy_Count.to_numpy()
            prediction=model.predict(rows[cols])
            metric,cm=phase2.evaluate(truth,prediction,name,part)
            support=cm.sum(axis=1);predicted=cm.sum(axis=0);tp=np.diag(cm);present=support>0
            precision=np.divide(tp,predicted,out=np.zeros(4),where=predicted>0)
            metric.update({"input_channels":len(cols),"source_nodes":len({c.split("_")[0] for c in cols}),
                           "input_reduction_percent":100*(15-len(cols))/15,
                           "macro_precision_present_classes":float(precision[present].mean()),
                           "macro_recall_present_classes":metric["balanced_accuracy_present_classes"]})
            results.append(metric)
            occupied=truth>0;empty=truth==0
            errors.append({"model":name,"partition":part,"rows":len(rows),
                           "occupied_rows":int(occupied.sum()),
                           "occupied_predicted_empty":int(((prediction==0)&occupied).sum()),
                           "false_empty_rate":float(np.mean(prediction[occupied]==0)) if occupied.any() else np.nan,
                           "empty_rows":int(empty.sum()),
                           "empty_predicted_occupied":int(((prediction>0)&empty).sum()),
                           "false_occupied_rate":float(np.mean(prediction[empty]>0)) if empty.any() else np.nan,
                           "count_mae":float(np.mean(np.abs(truth-prediction))),
                           "occupied_count_mae":float(np.mean(np.abs(truth[occupied]-prediction[occupied]))) if occupied.any() else np.nan})
            pc=pd.read_csv(TABLES/f"{name}_{part}_per_class.csv")
            pc.insert(0,"partition",part);pc.insert(0,"model",name);perclass.append(pc)
            if part!="january_test_subset":
                r=rows[["Source_Row","Timestamp","Room_Occupancy_Count"]].copy()
                r["model"]=name;r["partition"]=part;r["prediction"]=prediction
                probability=model.predict_proba(rows[cols])
                for i,cls in enumerate(model.classes_):r[f"probability_{cls}"]=probability[:,i]
                predictions.append(r)
    metrics=pd.DataFrame(results);error=pd.DataFrame(errors);pred=pd.concat(predictions,ignore_index=True)
    metrics.to_csv(TABLES/"comparison.csv",index=False)
    error.to_csv(TABLES/"occupancy_error_analysis.csv",index=False)
    pd.concat(perclass,ignore_index=True).to_csv(TABLES/"per_class_comparison.csv",index=False)
    pred.to_csv(OUT/"predictions.csv",index=False)
    # Paired comparison uses exactly the same observations, no independent-sample claim.
    paired=[]
    for part in ["validation","test"]:
        a=pred[(pred.model=="baseline")&(pred.partition==part)].set_index("Source_Row")
        b=pred[(pred.model=="ga")&(pred.partition==part)].set_index("Source_Row").loc[a.index]
        ac=a.prediction.eq(a.Room_Occupancy_Count);bc=b.prediction.eq(b.Room_Occupancy_Count)
        paired.append({"partition":part,"both_correct":int((ac&bc).sum()),
                       "baseline_only_correct":int((ac&~bc).sum()),"ga_only_correct":int((~ac&bc).sum()),
                       "both_wrong":int((~ac&~bc).sum()),"predictions_differ":int((a.prediction!=b.prediction).sum())})
    pd.DataFrame(paired).to_csv(TABLES/"paired_outcomes.csv",index=False)
    daily=[]
    for (name,part,day,cls),g in pred.assign(date=pred.Timestamp.dt.strftime("%Y-%m-%d")).groupby(
        ["model","partition","date","Room_Occupancy_Count"]):
        daily.append({"model":name,"partition":part,"date":day,"true_class":cls,"rows":len(g),
                      "correct":int(g.prediction.eq(cls).sum()),"recall":float(g.prediction.eq(cls).mean()),
                      "predicted_empty":int(g.prediction.eq(0).sum())})
    pd.DataFrame(daily).to_csv(TABLES/"daily_class_errors.csv",index=False)
    summary=[]
    for name in ["baseline","ga"]:
        val=metrics[(metrics.model==name)&(metrics.partition=="validation")].iloc[0]
        test=metrics[(metrics.model==name)&(metrics.partition=="test")].iloc[0]
        err=error[(error.model==name)&(error.partition=="test")].iloc[0]
        summary.append({"model":name,"input_channels":int(val.input_channels),"source_nodes":int(val.source_nodes),
                        "input_reduction_percent":val.input_reduction_percent,
                        "validation_accuracy_percent":100*val.accuracy,"validation_macro_f1":val.macro_f1_all_four,
                        "later_test_accuracy_percent":100*test.accuracy,
                        "later_test_balanced_accuracy_percent":100*test.balanced_accuracy_present_classes,
                        "later_test_false_empty_percent":100*err.false_empty_rate})
    summary=pd.DataFrame(summary);summary.to_csv(TABLES/"professor_comparison.csv",index=False)
    decision={"default_demo_model":"baseline","alternative":"ga","ga_input_reduction_percent":100*(15-len(features))/15,
              "validation_macro_f1_difference_ga_minus_baseline":float(summary.iloc[1].validation_macro_f1-summary.iloc[0].validation_macro_f1),
              "conclusion":"GA reduces inputs but does not preserve the baseline's validation performance. Neither model resolves later-period false-empty errors.",
              "do_not_claim":["GA universally improves accuracy","73.33% measured energy/cost savings","Independent new test confirmation","Reliable automatic control"]}
    (MODELS/"evaluation_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    # Four real validation examples (one per label). Provenance separated from pure sensor input.
    examples=data.loc[split.partition.eq("validation")].groupby("Room_Occupancy_Count",sort=True).head(1)
    examples[all_features].to_csv(ROOT/"data/examples/phase4_all_sensors.csv",index=False)
    examples[features].to_csv(ROOT/"data/examples/phase4_ga_sensors.csv",index=False)
    examples[["Source_Row","Timestamp","Room_Occupancy_Count"]].to_csv(ROOT/"data/examples/phase4_example_provenance.csv",index=False)
    make_plots(metrics,error)
    write_notes(summary,metrics,error,features,registry,decision)
    print(summary.to_string(index=False))
    print(json.dumps(decision,indent=2))

def make_plots(metrics,error):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.hashsalt":"sc-phase4"})
    colors=["#496d9b","#2b8a83"]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,part in zip(axes,["validation","test"]):
        for i,name in enumerate(["baseline","ga"]):
            r=metrics[(metrics.model==name)&(metrics.partition==part)].iloc[0]
            ax.bar(np.arange(3)+(i-.5)*.36,[r.accuracy,r.macro_f1_present_classes,r.balanced_accuracy_present_classes],
                   width=.35,label=name,color=colors[i])
        ax.set(xticks=range(3),xticklabels=["Accuracy","Macro F1\npresent classes","Balanced accuracy\npresent classes"],
               ylim=(0,1.08),title=part+(" (classes 0, 2, 3)" if part=="test" else " (classes 0, 1, 2, 3)"))
        ax.legend(frameon=False)
    phase2.save(fig,"performance_comparison")
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    axes[0].bar(["Baseline","GA"],[15,4],color=colors);axes[0].set(ylabel="Input channels",title="73.33% fewer model inputs",ylim=(0,17))
    axes[1].bar(["Baseline","GA"],[7,4],color=colors);axes[1].set(ylabel="Distinct source nodes referenced",title="No measured hardware or power saving",ylim=(0,8))
    phase2.save(fig,"input_reduction")
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    for ax,part in zip(axes,["validation","test"]):
        for i,name in enumerate(["baseline","ga"]):
            pc=pd.read_csv(TABLES/f"{name}_{part}_per_class.csv")
            ax.bar(np.arange(4)+(i-.5)*.35,pc.recall,width=.34,label=name,color=colors[i])
        if part=="test":ax.text(1,.48,"no true\nclass 1",ha="center",color="#526779")
        ax.set(xticks=range(4),xlabel="Actual occupancy",ylabel="Class recall",ylim=(0,1.1),title=part)
        ax.legend(frameon=False)
    phase2.save(fig,"per_class_recall")
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    for i,name in enumerate(["baseline","ga"]):
        values=[100*error[(error.model==name)&(error.partition==p)].iloc[0].false_empty_rate for p in ["validation","test"]]
        bars=ax.bar(np.arange(2)+(i-.5)*.35,values,width=.34,label=name,color=colors[i])
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{v:.2f}%",ha="center")
    ax.set(xticks=[0,1],xticklabels=["Validation","Later test"],ylabel="Occupied rows predicted empty (%)",ylim=(0,80),
           title="The key occupancy-control limitation remains");ax.legend(frameon=False)
    phase2.save(fig,"false_empty_comparison")
    for part in ["validation","test"]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
        for ax,name in zip(axes,["baseline","ga"]):
            cm=pd.read_csv(TABLES/f"{name}_{part}_confusion.csv",index_col=0).to_numpy()
            ax.imshow(cm,cmap="Blues",vmin=0,vmax=cm.max())
            for i in range(4):
                for j in range(4):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>.5*cm.max() else "#172b40")
            ax.set(xticks=range(4),yticks=range(4),xlabel="Predicted people",ylabel="Actual people",title=name)
        fig.suptitle(part+(" — class 1 absent" if part=="test" else " — all four classes"))
        phase2.save(fig,f"{part}_confusion_comparison")

def write_notes(summary,metrics,error,features,registry,decision):
    b,g=summary.iloc[0],summary.iloc[1]
    err=error[(error.model=="ga")&(error.partition=="test")].iloc[0]
    notes=f"""# Phase 4 — Optimized Occupancy System & GA Evaluation

## Verdict

GA selected four of fifteen inputs, a **73.33% reduction in model input channels**.
It did **not** preserve baseline validation performance:
accuracy changed from {b.validation_accuracy_percent:.2f}% to {g.validation_accuracy_percent:.2f}%,
and macro F1 from {b.validation_macro_f1:.4f} to {g.validation_macro_f1:.4f}.
The change in validation macro F1 is {decision['validation_macro_f1_difference_ga_minus_baseline']:+.4f}.

Keep the baseline as the default academic demonstration estimator; expose the four-input
model as an explicit compact alternative. This preference follows four-class validation,
not the slightly better later-test accuracy. Neither model is established as reliable for
actual room control. “GA-optimized” means optimized for its training fitness objective.

## Professor comparison table

{md(summary.round(4))}

The later test has true classes 0, 2 and 3 only. Four-class macro F1 is unavailable there.
Source nodes referenced decrease from seven to four, but these are theoretical data
dependencies, not removed hardware or measured savings. S1–S4 originally carry multiple channels.
The 73.33% figure applies only to input columns. It is not a power, price or electricity reduction.

## Exact experiment

1. Freeze the Phase 3 chromosome and its four selected channels: {", ".join(features)}.
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

Both models label **{int(err.occupied_predicted_empty)} of {int(err.occupied_rows)} occupied later-test
rows as empty ({err.false_empty_rate:.2%})**. The small aggregate improvement of the GA model does
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
"""
    (ROOT/"docs/phase4_notes.md").write_text(notes)
    images=[("input_reduction","Inputs and source nodes"),("performance_comparison","Performance comparison"),
            ("per_class_recall","Recall of each occupancy class"),("false_empty_comparison","Occupied rooms predicted empty"),
            ("validation_confusion_comparison","Validation confusion matrices"),("test_confusion_comparison","Later-test confusion matrices")]
    cards="".join(f'<section><h2>{title}</h2><img src="plots/{name}.png" alt="{title}"></section>' for name,title in images)
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC Project: Phase 4</title><style>
body{{background:#eef2f5;color:#172b40;font:16px/1.6 system-ui;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}
section{{background:white;border-radius:8px;padding:24px;margin:24px 0}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dde4e8;text-align:left}}
th{{background:#e8f2f1}}.scroll{{overflow:auto}}a{{color:#166e75}}.note{{border-left:4px solid #da9439;padding-left:18px}}
</style><body><main><p>SOFT COMPUTING MINOR PROJECT / PHASE 4</p><h1>Does GA improve the occupancy system?</h1>
<section><h2>Fewer inputs, lower validation performance</h2><p>Inputs: <strong>15 → 4 (73.33% reduction)</strong>.
Validation macro F1: <strong>{b.validation_macro_f1:.4f} → {g.validation_macro_f1:.4f}</strong>.
Baseline remains the default demonstration model; GA is an explicit compact alternative.</p>
<p class="note">Both models predict empty on {int(err.occupied_predicted_empty)} of {int(err.occupied_rows)} occupied
later-test observations ({err.false_empty_rate:.2%}). Input reduction does not resolve this limitation.
The later test lacks class 1. These previously seen data are not a new independent confirmation.</p>
<p><a href="../../docs/phase4_notes.md">Methods and viva notes</a> /
<a href="../../models/phase4/evaluation_decision.json">Evaluation decision</a> /
<a href="../../docs/project_roadmap.md">Seven-phase roadmap</a></p></section>
<section><h2>Formal comparison</h2><div class="scroll">{summary.round(4).to_html(index=False,border=0)}</div>
<p>Accuracy and reduction columns are percentages; macro F1 is a fraction.
Node counts describe data dependencies, not physical removal or power savings.</p></section>
{cards}<section><h2>Reproducible occupancy system</h2><p>A fresh four-input, 300-tree fit reproduces
Phase 3 predictions and probabilities. The unchanged baseline and fresh GA model are available through
one interface with explicit input validation and model choice. No fuzzy control or energy layer
has been implemented.</p></section></main></body></html>"""
    (OUT/"report.html").write_text(page)
    readme=ROOT/"README.md";text=readme.read_text();heading="## Phase 4 complete: optimized occupancy system and GA evaluation"
    block=f"""{heading}

Open **outputs/phase4/report.html** and **docs/phase4_notes.md** for the formal comparison.
The four-input model was independently retrained with unchanged parameters and exactly reproduces
Phase 3 predictions. Input reduction is 73.33%, but validation macro F1 drops from 0.8608 to 0.7952.
Both models miss 184 of 294 occupied later-test rows as empty. Baseline remains the default demo;
GA is the explicit compact alternative.

The common prediction interface is src/occupancy_system.py, with CLI src/predict_phase4.py.
Run python src/phase4.py and python src/verify_phase4.py to reproduce and verify.
Phase 5 is fuzzy control; it has not started. See docs/project_roadmap.md.

"""
    if heading not in text:
        text=text.replace("## Phase 3 complete: Genetic Algorithm sensor selection",block+"## Phase 3 complete: Genetic Algorithm sensor selection")
        readme.write_text(text)

if __name__=="__main__":main()
