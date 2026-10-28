"""Independent GA audit, inner-score reproduction and external metric verification."""
from pathlib import Path
import sys,json,hashlib,math,random,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from phase3 import offspring,repair
OUT=ROOT/"outputs/phase3";T=OUT/"tables"

def macro_f1(y,p):
    cm=np.zeros((4,4),int)
    for a,b in zip(y,p):cm[int(a),int(b)]+=1
    denom=cm.sum(axis=0)+cm.sum(axis=1)
    return np.divide(2*np.diag(cm),denom,out=np.zeros(4),where=denom>0).mean()

def main():
    config=json.loads((ROOT/"docs/phase3_protocol.json").read_text())
    lock=json.loads((ROOT/"models/phase3/selection_locked.json").read_text())
    meta=json.loads((ROOT/"models/phase3/model_metadata.json").read_text())
    for path,key in [
        (ROOT/"models/phase3/ga_selected_random_forest.joblib","model_sha256"),
        (ROOT/"models/phase2/random_forest_baseline.joblib","phase2_model_sha256"),
        (ROOT/"data/processed/occupancy_cleaned.csv","processed_data_sha256"),
        (ROOT/"models/phase3/selection_locked.json","selection_sha256"),
        (ROOT/"docs/phase3_protocol.json","protocol_sha256"),
        (ROOT/"data/splits/phase2_split_manifest.csv","split_manifest_sha256")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[key],key
    data=pd.read_csv(ROOT/"data/processed/occupancy_cleaned.csv",parse_dates=["Timestamp"],float_precision="round_trip").set_index("Source_Row")
    split=pd.read_csv(ROOT/"data/splits/phase2_split_manifest.csv")
    train_ids=set(split.loc[split.partition.eq("train"),"Source_Row"])
    folds=pd.read_csv(ROOT/"data/splits/phase3_inner_folds.csv")
    assert set(folds.Source_Row)==train_ids
    valid_ids=[]
    for fold,g in folds.groupby("fold"):
        assert len(g)==1462 and g.Source_Row.is_unique
        fit=g.loc[g.role.eq("fit"),"Source_Row"].to_numpy()
        val=g.loc[g.role.eq("validation"),"Source_Row"].to_numpy()
        a,b=data.loc[val,"Timestamp"].min(),data.loc[val,"Timestamp"].max()
        expected_fit={i for i in train_ids if data.loc[i,"Timestamp"]<a-pd.Timedelta(minutes=10) or data.loc[i,"Timestamp"]>b+pd.Timedelta(minutes=10)}
        assert set(fit)==expected_fit and not set(fit)&set(val)
        assert set(data.loc[fit,"Room_Occupancy_Count"])=={0,1,2,3}
        valid_ids.extend(val)
    assert len(valid_ids)==len(set(valid_ids))==len(train_ids)
    cache=pd.read_csv(T/"evaluated_subsets.csv",dtype={"chromosome":str})
    assert cache.chromosome.is_unique and len(cache)==lock["unique_subsets_evaluated"]
    for r in cache.itertuples():
        assert len(r.chromosome)==15 and set(r.chromosome)<=set("01") and "1" in r.chromosome
        assert r.selected_count==r.chromosome.count("1")
        assert math.isclose(r.fitness,r.cv_macro_f1-.05*r.selected_count/15,abs_tol=1e-12)
    best=cache.sort_values(["fitness","selected_count","chromosome"],ascending=[False,True,True]).iloc[0]
    assert best.chromosome==lock["winner"]["chromosome"]
    assert meta["features"]==[f for f,b in zip(lock["all_features"],best.chromosome) if b=="1"]
    hist=pd.read_csv(T/"convergence.csv",dtype={"chromosome":str})
    pop=pd.read_csv(T/"population_history.csv",dtype={"chromosome":str})
    assert len(pop)==3*11*16
    for seed in config["seeds"]:
        h=hist[hist.seed==seed]
        assert h.generation.tolist()==list(range(11))
        assert (h.fitness.diff().dropna()>=-1e-12).all()
        previous=None
        for gen,g in pop[pop.seed==seed].groupby("generation"):
            assert len(g)==16
            actual=g.sort_values(["fitness","selected_count","chromosome"],ascending=[False,True,True]).iloc[0]
            logged=h[h.generation==gen].iloc[0]
            assert actual.chromosome==logged.chromosome
            if previous is not None:
                # Elite copies must appear with multiplicity, including ties/duplicates.
                elite=previous.sort_values(["fitness","selected_count","chromosome"],ascending=[False,True,True]).head(2)
                for chrom,n in elite.chromosome.value_counts().items():
                    assert (g.chromosome==chrom).sum()>=n
            previous=g
    # Refit both reference and selected inner models to reproduce stored OOF predictions.
    oof=pd.read_csv(T/"training_oof_predictions.csv").set_index("Source_Row")
    for cols,column in [(lock["all_features"],"all_inputs_prediction"),(lock["features"],"selected_prediction")]:
        for fold,g in folds.groupby("fold"):
            fit=g.loc[g.role.eq("fit"),"Source_Row"];valid=g.loc[g.role.eq("validation"),"Source_Row"]
            model=RandomForestClassifier(**lock["search_estimator"])
            model.fit(data.loc[fit,cols].to_numpy(),data.loc[fit,"Room_Occupancy_Count"])
            guess=model.predict(data.loc[valid,cols].to_numpy())
            assert np.array_equal(guess,oof.loc[valid,column].to_numpy())
        score=macro_f1(oof.Room_Occupancy_Count,oof[column])
        expected=lock["reference"] if column=="all_inputs_prediction" else lock["winner"]
        assert math.isclose(score,expected["cv_macro_f1"],abs_tol=1e-12)
    # Operator invariants: parent copies, valid bitstrings, empty repair and seeded reproducibility.
    a=[1,0,1,0,1]*3;b=[0,1,0,1,0]*3
    old_a=a.copy();old_b=b.copy()
    c=offspring(a,b,random.Random(5),config)
    assert c==offspring(a,b,random.Random(5),config) and a==old_a and b==old_b
    assert all(len(x)==15 and set(x)<=set([0,1]) and any(x) for x in c)
    assert sum(repair([0]*15,random.Random(2)))==1
    force={**config,"crossover_probability":0,"mutation_per_bit":1}
    c=offspring(a,b,random.Random(3),force)
    assert c==([1-x for x in a],[1-x for x in b])
    pred=pd.read_csv(OUT/"predictions.csv")
    metrics=pd.read_csv(T/"comparison_metrics.csv")
    for name in ["baseline_15","ga_selected"]:
        rows=pred[pred.model.eq(name)]
        assert rows.Source_Row.is_unique
        expected=split[~split.partition.eq("embargo")]
        joined=rows.merge(expected,on="Source_Row",suffixes=("_pred","_split"),validate="one_to_one")
        assert len(joined)==len(expected)
        assert (joined.partition_pred==joined.partition_split).all()
        assert (joined.Room_Occupancy_Count_pred==joined.Room_Occupancy_Count_split).all()
        for part in ["train","validation","test","january_test_subset"]:
            r=rows[rows.partition.eq("test") & pd.to_datetime(rows.Timestamp).ge("2018-01-01")] if part=="january_test_subset" else rows[rows.partition.eq(part)]
            cm=np.zeros((4,4),int)
            for y,p in zip(r.Room_Occupancy_Count,r.prediction):cm[int(y),int(p)]+=1
            assert np.array_equal(cm,pd.read_csv(T/f"{name}_{part}_confusion.csv",index_col=0).to_numpy())
            support=cm.sum(axis=1);tp=np.diag(cm);present=support>0
            denom=support+cm.sum(axis=0);f1=np.divide(2*tp,denom,out=np.zeros(4),where=denom>0)
            m=metrics[(metrics.model==name)&(metrics.partition==part)].iloc[0]
            for key,val in {"accuracy":tp.sum()/cm.sum(),"macro_f1_present_classes":f1[present].mean(),
                            "balanced_accuracy_present_classes":np.mean(tp[present]/support[present]),
                            "weighted_f1":np.sum(f1*support)/support.sum()}.items():
                assert math.isclose(m[key],val,abs_tol=1e-12)
            if present.all():assert math.isclose(m.macro_f1_all_four,f1.mean(),abs_tol=1e-12)
            else:assert pd.isna(m.macro_f1_all_four)
    # Baseline predictions must exactly reproduce Phase 2, not a new all-input fit.
    old=pd.read_csv(ROOT/"outputs/phase2/predictions.csv")
    old=old[old.model.eq("random_forest")].sort_values("Source_Row")
    base=pred[pred.model.eq("baseline_15")].sort_values("Source_Row")
    assert np.array_equal(old.prediction,base.prediction)
    model=joblib.load(ROOT/"models/phase3/ga_selected_random_forest.joblib")
    chosen=pred[pred.model.eq("ga_selected")]
    X=data.loc[chosen.Source_Row,meta["features"]]
    assert model.feature_names_in_.tolist()==meta["features"]
    assert np.array_equal(model.predict(X),chosen.prediction)
    assert np.allclose(model.predict_proba(X),chosen[[f"probability_{i}" for i in range(4)]],rtol=0,atol=1e-14)
    sample=X.iloc[:5]
    (ROOT/"work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/"work",prefix="phase3-check-") as tmp:
        tmp=Path(tmp);sample.to_csv(tmp/"input.csv",index=False)
        command=[sys.executable,str(ROOT/"src/predict_phase3.py"),"--input",str(tmp/"input.csv"),"--output",str(tmp/"output.csv")]
        subprocess.run(command,check=True,capture_output=True,text=True)
        assert np.array_equal(pd.read_csv(tmp/"output.csv").predicted_occupancy,model.predict(sample))
        sample.drop(columns=[meta["features"][0]]).to_csv(tmp/"missing.csv",index=False)
        command[3]=str(tmp/"missing.csv");command[5]=str(tmp/"bad.csv")
        failed=subprocess.run(command,capture_output=True,text=True)
        assert failed.returncode!=0 and "Missing required sensors" in failed.stderr
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,v in attrs:
                if key in ("src","href") and not v.startswith(("https:","http:","#")):
                    assert (OUT/v).exists(),v
    Links().feed((OUT/"report.html").read_text())
    result={"status":"passed","checks":[
        "Earlier baseline and data hashes preserved",
        "All GA folds confined to training day with 10-minute purges and exact OOF coverage",
        "All cached fitness values, selected winner, population sizes and elite preservation verified",
        "Selected and reference inner scores independently reproduced from eight fresh fold fits",
        "Operator mutation, copying, empty repair and seeded reproducibility checked",
        "External metrics recomputed; baseline predictions match Phase 2",
        "Selected model predictions and probabilities verified",
        "Selected-input inference works and rejects a missing selected feature",
        "Report links resolve"]}
    (T/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
