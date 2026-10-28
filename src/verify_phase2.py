"""Independently recompute reported metrics and test the saved prediction interface."""
from pathlib import Path
import sys,json,math,hashlib,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists(): sys.path.append(str(ROOT/"work/python-packages"))
import pandas as pd
import numpy as np
import joblib

def main():
    out=ROOT/"outputs/phase2"
    meta=json.loads((ROOT/"models/phase2/model_metadata.json").read_text())
    for path,key in [(ROOT/"models/phase2/random_forest_baseline.joblib","model_sha256"),
                     (ROOT/"data/processed/occupancy_cleaned.csv","processed_data_sha256"),
                     (ROOT/"docs/phase2_protocol.json","protocol_sha256")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[key],key
    manifest=pd.read_csv(ROOT/"data/splits/phase2_split_manifest.csv",parse_dates=["Timestamp"])
    pred=pd.read_csv(out/"predictions.csv")
    metrics=pd.read_csv(out/"tables/metrics.csv")
    assert len(manifest)==10129 and manifest.Source_Row.is_unique
    assert set(manifest.partition)=={"train","validation","test","embargo"}
    for a,b in [("train","validation"),("validation","test")]:
        left=manifest.loc[manifest.partition.eq(a),"Timestamp"]
        right=manifest.loc[manifest.partition.eq(b),"Timestamp"]
        assert (right.min()-left.max()).total_seconds()>600
    for model in ["random_forest","always_empty"]:
        rows=pred[pred.model.eq(model)]
        assert rows.Source_Row.is_unique
        expected=manifest[~manifest.partition.eq("embargo")]
        joined=rows.merge(expected,on="Source_Row",validate="one_to_one",suffixes=("_pred","_split"))
        assert len(joined)==len(expected)==10092
        assert (joined.partition_pred==joined.partition_split).all()
        assert (joined.Room_Occupancy_Count_pred==joined.Room_Occupancy_Count_split).all()
        for part in ["train","validation","test","january_test_subset"]:
            p=rows[rows.partition.eq("test") & pd.to_datetime(rows.Timestamp).ge("2018-01-01")] if part=="january_test_subset" else rows[rows.partition.eq(part)]
            cm=np.zeros((4,4),dtype=int)
            for actual,guess in zip(p.Room_Occupancy_Count,p.prediction): cm[int(actual),int(guess)]+=1
            saved=pd.read_csv(out/f"tables/{model}_{part}_confusion.csv",index_col=0).to_numpy()
            assert np.array_equal(cm,saved)
            support=cm.sum(axis=1); predicted=cm.sum(axis=0);tp=np.diag(cm)
            present=support>0
            f1=np.divide(2*tp,support+predicted,out=np.zeros(4),where=support+predicted>0)
            expected_metrics={"accuracy":tp.sum()/cm.sum(),
                "macro_f1_present_classes":f1[present].mean(),
                "balanced_accuracy_present_classes":(tp[present]/support[present]).mean(),
                "weighted_f1":(f1*support).sum()/support.sum()}
            report=metrics[(metrics.model==model)&(metrics.partition==part)].iloc[0]
            for key,value in expected_metrics.items():
                assert math.isclose(report[key],value,rel_tol=1e-12,abs_tol=1e-12),(model,part,key)
            if present.all(): assert math.isclose(report.macro_f1_all_four,f1.mean(),abs_tol=1e-12)
            else: assert pd.isna(report.macro_f1_all_four)
    data=pd.read_csv(ROOT/"data/processed/occupancy_cleaned.csv",float_precision="round_trip").set_index("Source_Row")
    model=joblib.load(ROOT/"models/phase2/random_forest_baseline.joblib")
    forest=pred[pred.model.eq("random_forest")]
    inputs=data.loc[forest.Source_Row,meta["features"]]
    assert model.feature_names_in_.tolist()==meta["features"]
    assert np.array_equal(model.predict(inputs),forest.prediction.to_numpy())
    assert np.allclose(model.predict_proba(inputs),forest[[f"probability_{i}" for i in range(4)]].to_numpy(),rtol=0,atol=1e-14)
    # Exercise the actual CLI on one validation row of each true class.
    sample=manifest[manifest.partition.eq("validation")].groupby("Room_Occupancy_Count").first()
    (ROOT/"work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/"work",prefix="phase2-check-") as tmp:
        tmp=Path(tmp)
        X=data.loc[sample.Source_Row,meta["features"]]
        X.to_csv(tmp/"input.csv",index=False)
        command=[sys.executable,str(ROOT/"src/predict_occupancy.py"),"--input",str(tmp/"input.csv"),"--output",str(tmp/"output.csv")]
        subprocess.run(command,check=True,capture_output=True,text=True)
        actual=pd.read_csv(tmp/"output.csv")
        assert np.array_equal(actual.predicted_occupancy,model.predict(X))
        X.drop(columns=[meta["features"][0]]).to_csv(tmp/"missing.csv",index=False)
        bad=command.copy();bad[3]=str(tmp/"missing.csv");bad[5]=str(tmp/"bad-output.csv")
        result=subprocess.run(bad,capture_output=True,text=True)
        assert result.returncode!=0 and "Missing required sensors" in result.stderr
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ("src","href") and not v.startswith(("https:","http:","#")):
                    assert (out/v).exists(),v
    Links().feed((out/"report.html").read_text())
    result={"status":"passed","independent_checks":[
        "Source, model and protocol hashes match",
        "Every scored row matches its disjoint partition and true label",
        "All confusion matrices and aggregate metrics recalculated from predictions",
        "Saved-model predictions and probabilities match all scored records",
        "Prediction CLI handles a four-class example and rejects a missing sensor",
        "Report local links resolve"
    ]}
    (out/"tables/independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
