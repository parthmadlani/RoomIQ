"""Handwritten binary GA with training-only purged blocked fitness."""
from pathlib import Path
import sys,os,json,hashlib,random
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import phase2

OUT=ROOT/"outputs/phase3"
TABLES=OUT/"tables"
MODELS=ROOT/"models/phase3"

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(mask):return "".join(map(str,mask))
def rank(record):return (-record["fitness"],record["selected_count"],record["chromosome"])

def repair(mask,rng):
    # The empty subset cannot train a model; activate one random bit.
    mask=list(mask)
    if not any(mask):mask[rng.randrange(len(mask))]=1
    return mask

def offspring(a,b,rng,config):
    a,b=list(a),list(b)  # copies protect parents and elites
    if rng.random()<config["crossover_probability"]:
        point=rng.randrange(1,len(a))
        a,b=a[:point]+b[point:],b[:point]+a[point:]
    for child in [a,b]:
        for i in range(len(child)):
            if rng.random()<config["mutation_per_bit"]:child[i]=1-child[i]
    return repair(a,rng),repair(b,rng)

def make_folds(training,config):
    times=training.Timestamp
    folds=[];assignments=[];coverage=[]
    for fold,valid in enumerate(np.array_split(np.arange(len(training)),config["folds"])):
        start,end=times.iloc[valid[0]],times.iloc[valid[-1]]
        purge=pd.Timedelta(minutes=config["purge_minutes"])
        fit=np.flatnonzero(((times<start-purge)|(times>end+purge)).to_numpy())
        assert not set(fit)&set(valid)
        assert set(training.iloc[fit].Room_Occupancy_Count)=={0,1,2,3}
        folds.append((fit,valid))
        role=np.full(len(training),"purged",dtype=object);role[fit]="fit";role[valid]="validation"
        for i,row in training.iterrows():
            assignments.append({"fold":fold,"Source_Row":int(row.Source_Row),"role":role[i]})
        for label,idx in [("fit",fit),("validation",valid),("purged",np.flatnonzero(role=="purged"))]:
            c=training.iloc[idx].Room_Occupancy_Count.value_counts()
            coverage.append({"fold":fold,"role":label,"rows":len(idx),
                             **{f"class_{k}":int(c.get(k,0)) for k in range(4)}})
    pd.DataFrame(assignments).to_csv(ROOT/"data/splits/phase3_inner_folds.csv",index=False)
    pd.DataFrame(coverage).to_csv(TABLES/"inner_fold_coverage.csv",index=False)
    return folds

def search(training,features,config,params):
    folds=make_folds(training,config)
    X=training[features].to_numpy();y=training.Room_Occupancy_Count.to_numpy()
    cache={};oof_cache={};history=[];populations=[];runs=[]
    search_params={**params,"n_estimators":config["search_trees"]}
    def score(mask):
        chromosome=bits(mask)
        if chromosome not in cache:
            selected=np.flatnonzero(mask);pred=np.full(len(y),-1,dtype=int)
            assert len(selected)>0
            for fit,valid in folds:
                model=RandomForestClassifier(**search_params)
                model.fit(X[fit][:,selected],y[fit])
                pred[valid]=model.predict(X[valid][:,selected])
            assert (pred>=0).all()
            f1=f1_score(y,pred,labels=[0,1,2,3],average="macro",zero_division=0)
            record={"chromosome":chromosome,"selected_count":len(selected),"cv_macro_f1":float(f1),
                    "fitness":float(f1-config["sensor_penalty"]*len(selected)/len(features))}
            cache[chromosome]=record;oof_cache[chromosome]=pred
        return cache[chromosome]
    all_mask=[1]*len(features)
    reference=score(all_mask)
    for seed in config["seeds"]:
        rng=random.Random(seed)
        population=[all_mask.copy(),repair([0]*len(features),rng)]
        population += [repair([rng.randrange(2) for _ in features],rng) for _ in range(config["population"]-2)]
        for generation in range(config["generations"]+1):
            scored=[score(m) for m in population]
            order=sorted(range(len(population)),key=lambda i:rank(scored[i]))
            best=scored[order[0]]
            history.append({"seed":seed,"generation":generation,**best,
                            "mean_fitness":float(np.mean([s["fitness"] for s in scored])),
                            "population_diversity":len({bits(m) for m in population}),
                            "unique_evaluations_so_far":len(cache)})
            for pos,record in enumerate(scored):
                populations.append({"seed":seed,"generation":generation,"position":pos,**record})
            print(f"Seed {seed}, generation {generation:02d}: best F1={best['cv_macro_f1']:.4f}, sensors={best['selected_count']}, unique subsets={len(cache)}",flush=True)
            if generation==config["generations"]:
                runs.append({"seed":seed,**best});break
            next_pop=[population[i].copy() for i in order[:config["elites"]]]
            def tournament():
                candidates=[rng.randrange(len(population)) for _ in range(config["tournament_size"])]
                winner=min(candidates,key=lambda i:rank(scored[i]))
                return population[winner].copy()
            while len(next_pop)<config["population"]:
                children=offspring(tournament(),tournament(),rng,config)
                next_pop.extend(children[:config["population"]-len(next_pop)])
            population=next_pop
        pd.DataFrame(history).to_csv(TABLES/"convergence.csv",index=False)
        pd.DataFrame(list(cache.values())).to_csv(TABLES/"evaluated_subsets.csv",index=False)
    winner=min(runs,key=rank)
    selected=[f for f,b in zip(features,winner["chromosome"]) if b=="1"]
    lock={"features":selected,"all_features":features,"winner":winner,"reference":reference,
          "seeds":runs,"unique_subsets_evaluated":len(cache),
          "selection_rule":config["selection"],"search_estimator":search_params,
          "protocol_sha256":digest(ROOT/"docs/phase3_protocol.json"),
          "training_source_rows_sha256":hashlib.sha256(training.Source_Row.to_numpy().tobytes()).hexdigest(),
          "holdout_access":"Selected and saved before downstream validation/test prediction.",
          "selected_nodes":sorted({f.split("_")[0] for f in selected})}
    # Save the choice before fitting/evaluating the final external comparison.
    (MODELS/"selection_locked.json").write_text(json.dumps(lock,indent=2)+"\n")
    pd.DataFrame(runs).to_csv(TABLES/"run_winners.csv",index=False)
    pd.DataFrame(populations).to_csv(TABLES/"population_history.csv",index=False)
    oof=training[["Source_Row","Timestamp","Room_Occupancy_Count"]].copy()
    oof["all_inputs_prediction"]=oof_cache[bits(all_mask)]
    oof["selected_prediction"]=oof_cache[winner["chromosome"]]
    oof.to_csv(TABLES/"training_oof_predictions.csv",index=False)
    pd.DataFrame({"bit_position":np.arange(1,len(features)+1),"feature":features,
                  "selected":[int(f in selected) for f in features]}).to_csv(TABLES/"chromosome_mapping.csv",index=False)
    return lock

def main():
    for p in [TABLES,OUT/"plots",MODELS,ROOT/"data/splits"]:p.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/"docs/phase3_protocol.json").read_text())
    meta=json.loads((ROOT/"models/phase2/model_metadata.json").read_text())
    source=ROOT/"data/processed/occupancy_cleaned.csv"
    assert digest(source)==meta["processed_data_sha256"],"Phase 1 data changed"
    baseline_path=ROOT/"models/phase2/random_forest_baseline.joblib"
    assert digest(baseline_path)==meta["model_sha256"],"Baseline changed"
    manifest=pd.read_csv(ROOT/"data/splits/phase2_split_manifest.csv")
    data=pd.read_csv(source,parse_dates=["Timestamp"],float_precision="round_trip")
    assert data.Source_Row.tolist()==manifest.Source_Row.tolist()
    features=meta["features"]
    train_ids=manifest.loc[manifest.partition.eq("train"),"Source_Row"]
    training=data[data.Source_Row.isin(train_ids)].reset_index(drop=True)
    assert len(training)==1462 and training.Timestamp.dt.strftime("%Y-%m-%d").eq("2017-12-22").all()
    params=json.loads((ROOT/"docs/phase2_protocol.json").read_text())["estimator"]
    lock=search(training,features,config,params)
    selected=lock["features"]
    final=RandomForestClassifier(**params)
    final.fit(training[selected],training.Room_Occupancy_Count)
    model_path=MODELS/"ga_selected_random_forest.joblib"
    joblib.dump(final,model_path,compress=3)
    # Reuse Phase 2 metric code, explicitly directing every output to Phase 3.
    phase2.TABLES=TABLES
    baseline=joblib.load(baseline_path)
    results=[];predictions=[]
    for name,model,cols in [("baseline_15",baseline,features),("ga_selected",final,selected)]:
        for part in ["train","validation","test","january_test_subset"]:
            mask=manifest.partition.eq("test")&data.Timestamp.ge("2018-01-01") if part=="january_test_subset" else manifest.partition.eq(part)
            rows=data.loc[mask];pred=model.predict(rows[cols])
            metric,_=phase2.evaluate(rows.Room_Occupancy_Count,pred,name,part)
            metric["input_count"]=len(cols);results.append(metric)
            if part!="january_test_subset":
                r=rows[["Source_Row","Timestamp","Room_Occupancy_Count"]].copy()
                r["model"]=name;r["partition"]=part;r["prediction"]=pred
                for i,c in enumerate(model.classes_):r[f"probability_{c}"]=model.predict_proba(rows[cols])[:,i]
                predictions.append(r)
    pd.DataFrame(results).to_csv(TABLES/"comparison_metrics.csv",index=False)
    pd.concat(predictions).to_csv(OUT/"predictions.csv",index=False)
    modelmeta={"features":selected,"classes":[0,1,2,3],"target":"Room_Occupancy_Count",
               "parameters":final.get_params(),"model_sha256":digest(model_path),
               "processed_data_sha256":digest(source),"phase2_model_sha256":digest(baseline_path),
               "selection_sha256":digest(MODELS/"selection_locked.json"),
               "protocol_sha256":digest(ROOT/"docs/phase3_protocol.json"),
               "split_manifest_sha256":digest(ROOT/"data/splits/phase2_split_manifest.csv"),
               "environment":meta["environment"],"training_rows":1462,
               "notes":"Selection uses December 22 only. Final model trained on that same day with 300 trees."}
    (MODELS/"model_metadata.json").write_text(json.dumps(modelmeta,indent=2)+"\n")
    import report_phase3
    report_phase3.main()
    print("Locked selection:",selected,flush=True)
    print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=="__main__":main()
