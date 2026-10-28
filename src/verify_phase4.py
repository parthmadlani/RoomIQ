"""Independent Phase 4 metric, preservation, refit and interface checks."""
from pathlib import Path
import sys,json,hashlib,math,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
import pandas as pd
import numpy as np
import joblib
from occupancy_system import OccupancySystem
OUT=ROOT/"outputs/phase4";TABLES=OUT/"tables"

def main():
    reg=json.loads((ROOT/"models/phase4/model_registry.json").read_text())
    for path,key in [(ROOT/"data/processed/occupancy_cleaned.csv","processed_data_sha256"),
                     (ROOT/"data/splits/phase2_split_manifest.csv","split_sha256"),
                     (ROOT/"models/phase3/selection_locked.json","phase3_selection_sha256"),
                     (ROOT/"docs/phase4_protocol.json","phase4_protocol_sha256")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reg[key]
    data=pd.read_csv(ROOT/"data/processed/occupancy_cleaned.csv",float_precision="round_trip").set_index("Source_Row")
    split=pd.read_csv(ROOT/"data/splits/phase2_split_manifest.csv")
    pred=pd.read_csv(OUT/"predictions.csv")
    metrics=pd.read_csv(TABLES/"comparison.csv")
    errors=pd.read_csv(TABLES/"occupancy_error_analysis.csv")
    prior=pd.read_csv(ROOT/"outputs/phase3/predictions.csv")
    def equal(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)
    for name,oldname in [("baseline","baseline_15"),("ga","ga_selected")]:
        path=ROOT/reg["models"][name]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reg["models"][name]["sha256"]
        p=pred[pred.model.eq(name)]
        assert p.Source_Row.is_unique
        expected=split[~split.partition.eq("embargo")]
        joined=p.merge(expected,on="Source_Row",validate="one_to_one",suffixes=("_pred","_split"))
        assert len(joined)==len(expected)==10092
        assert (joined.partition_pred==joined.partition_split).all()
        assert (joined.Room_Occupancy_Count_pred==joined.Room_Occupancy_Count_split).all()
        previous=prior[prior.model.eq(oldname)].sort_values("Source_Row")
        current=p.sort_values("Source_Row")
        assert np.array_equal(previous.prediction,current.prediction)
        system=OccupancySystem(name)
        result=system.predict(data.loc[p.Source_Row])
        assert np.array_equal(result.predicted_occupancy,p.prediction)
        cols=[f"probability_{i}" for i in range(4)]
        assert np.allclose(result[cols],p[cols],rtol=0,atol=1e-14)
        assert np.allclose(result[cols].sum(axis=1),1,rtol=0,atol=1e-14)
        for part in ["train","validation","test","january_test_subset"]:
            r=p[p.partition.eq("test") & pd.to_datetime(p.Timestamp).ge("2018-01-01")] if part=="january_test_subset" else p[p.partition.eq(part)]
            truth=r.Room_Occupancy_Count.to_numpy();guess=r.prediction.to_numpy()
            cm=np.zeros((4,4),int)
            for a,b in zip(truth,guess):cm[int(a),int(b)]+=1
            assert np.array_equal(cm,pd.read_csv(TABLES/f"{name}_{part}_confusion.csv",index_col=0).to_numpy())
            true_count=cm.sum(axis=1);pred_count=cm.sum(axis=0);tp=np.diag(cm);present=true_count>0
            denom=true_count+pred_count
            f1=np.divide(2*tp,denom,out=np.zeros(4),where=denom>0)
            prec=np.divide(tp,pred_count,out=np.zeros(4),where=pred_count>0)
            m=metrics[(metrics.model==name)&(metrics.partition==part)].iloc[0]
            for key,v in {"accuracy":tp.sum()/cm.sum(),"macro_f1_present_classes":f1[present].mean(),
                          "weighted_f1":(f1*true_count).sum()/true_count.sum(),
                          "balanced_accuracy_present_classes":(tp[present]/true_count[present]).mean(),
                          "macro_precision_present_classes":prec[present].mean(),
                          "macro_recall_present_classes":(tp[present]/true_count[present]).mean()}.items():equal(m[key],v)
            if present.all():equal(m.macro_f1_all_four,f1.mean())
            else:assert pd.isna(m.macro_f1_all_four)
            pc=pd.read_csv(TABLES/f"{name}_{part}_per_class.csv")
            for i,row in pc.iterrows():
                assert row.support==true_count[i] and row.predicted_count==pred_count[i]
                for key,value,defined in [("precision",prec[i],pred_count[i]>0),
                                         ("recall",tp[i]/true_count[i] if true_count[i]>0 else 0,true_count[i]>0),
                                         ("f1",f1[i],denom[i]>0)]:
                    if defined:equal(row[key],value)
                    else:assert pd.isna(row[key])
            e=errors[(errors.model==name)&(errors.partition==part)].iloc[0]
            occupied=truth>0;empty=truth==0
            assert int(e.occupied_predicted_empty)==int(((guess==0)&occupied).sum())
            assert int(e.empty_predicted_occupied)==int(((guess>0)&empty).sum())
            equal(e.false_empty_rate,np.mean(guess[occupied]==0))
            equal(e.false_occupied_rate,np.mean(guess[empty]>0))
            equal(e.count_mae,np.mean(np.abs(truth-guess)))
            equal(e.occupied_count_mae,np.mean(np.abs(truth[occupied]-guess[occupied])))
        # Required feature order, ignored target and single-vs-batch behaviour.
        X=data[system.features].iloc[:4].copy()
        standard=system.predict(X)
        reordered=X[list(reversed(X.columns))].copy()
        reordered["Room_Occupancy_Count"]=[999]*len(X)
        assert np.array_equal(system.predict(reordered).predicted_occupancy,standard.predicted_occupancy)
        equal(system.predict(X.iloc[0].to_dict()).predicted_occupancy.iloc[0],standard.predicted_occupancy.iloc[0])
        invalid=[]
        invalid.append(X.drop(columns=[system.features[0]]))
        z=X.copy();z.iloc[0,0]=np.nan;invalid.append(z)
        z=X.copy();z.iloc[0,0]=np.inf;invalid.append(z)
        z=X.copy();z["S6_PIR"]=2;invalid.append(z)
        z=X.copy();z["S5_CO2"]=-1;invalid.append(z)
        z=X.copy();z["S5_CO2"]="not a number";invalid.append(z)
        invalid.append(X.iloc[:0])
        for bad in invalid:
            try:system.predict(bad)
            except (ValueError,TypeError):pass
            else:raise AssertionError("Invalid input was accepted")
        (ROOT/"work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"work",prefix="phase4-check-") as tmp:
            tmp=Path(tmp);X.to_csv(tmp/"input.csv",index=False)
            command=[sys.executable,str(ROOT/"src/predict_phase4.py"),"--model",name,"--input",str(tmp/"input.csv"),"--output",str(tmp/"output.csv")]
            subprocess.run(command,check=True,capture_output=True,text=True)
            assert np.array_equal(pd.read_csv(tmp/"output.csv").predicted_occupancy,standard.predicted_occupancy)
            repeat=subprocess.run(command,capture_output=True,text=True)
            assert repeat.returncode!=0 and "Output already exists" in repeat.stderr
    try:OccupancySystem("unknown")
    except ValueError:pass
    else:raise AssertionError("Unknown model accepted")
    assert OccupancySystem().model_name==reg["default_model"]=="baseline"
    ga_only=pd.read_csv(ROOT/"data/examples/phase4_ga_sensors.csv")
    try:OccupancySystem().predict(ga_only)
    except ValueError as e:assert "Missing required sensors" in str(e)
    else:raise AssertionError("Baseline silently accepted incomplete sensor set")
    summary=pd.read_csv(TABLES/"professor_comparison.csv")
    for row in summary.itertuples():
        features=reg["models"][row.model]["features"]
        assert row.input_channels==len(features) and row.source_nodes==len({c.split("_")[0] for c in features})
        equal(row.input_reduction_percent,100*(15-len(features))/15)
        val=metrics[(metrics.model==row.model)&(metrics.partition=="validation")].iloc[0]
        test=metrics[(metrics.model==row.model)&(metrics.partition=="test")].iloc[0]
        equal(row.validation_accuracy_percent,100*val.accuracy);equal(row.validation_macro_f1,val.macro_f1_all_four)
        equal(row.later_test_accuracy_percent,100*test.accuracy)
        equal(row.later_test_balanced_accuracy_percent,100*test.balanced_accuracy_present_classes)
    paired=pd.read_csv(TABLES/"paired_outcomes.csv")
    for r in paired.itertuples():
        a=pred[(pred.model=="baseline")&(pred.partition==r.partition)].set_index("Source_Row")
        b=pred[(pred.model=="ga")&(pred.partition==r.partition)].set_index("Source_Row").loc[a.index]
        ac=a.prediction==a.Room_Occupancy_Count;bc=b.prediction==b.Room_Occupancy_Count
        assert r.both_correct==int((ac&bc).sum())
        assert r.baseline_only_correct==int((ac&~bc).sum())
        assert r.ga_only_correct==int((~ac&bc).sum())
        assert r.both_wrong==int((~ac&~bc).sum())
        assert r.predictions_differ==int((a.prediction!=b.prediction).sum())
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ("src","href") and not v.startswith(("https:","http:","#")):assert (OUT/v).exists(),v
    Links().feed((OUT/"report.html").read_text())
    assert len(list((OUT/"plots").glob("*.png")))==6
    result={"status":"passed","checks":[
        "Data, split, selected subset and registered model hashes match",
        "Fresh GA model and unchanged baseline reproduce previous predictions",
        "All multiclass, false-empty, count-error and paired metrics independently recalculated",
        "Summary input counts and channel-reduction calculation verified",
        "Both interfaces match saved predictions and class probabilities",
        "Missing, nonnumeric, nonfinite, negative and invalid-PIR inputs rejected",
        "Column order and extra target do not affect prediction; incomplete baseline never falls back",
        "Both CLIs tested, including overwrite refusal",
        "Report links and six figures verified"]}
    (TABLES/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
