"""Replay simulated modes and energy only over eligible recorded intervals."""
from pathlib import Path
import sys,os,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import pandas as pd,numpy as np
from fuzzy_controller import FuzzyRoomController
from occupancy_system import OccupancySystem
from energy_controller import EnergyDecisionLayer
OUT=ROOT/"outputs/phase6";TABLES=OUT/"tables";PLOTS=OUT/"plots"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    for path in [TABLES,PLOTS]:path.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/"config/phase6_energy.json").read_text())
    registry=json.loads((ROOT/"models/phase4/model_registry.json").read_text())
    source=ROOT/"data/processed/occupancy_cleaned.csv"
    assert sha(source)==registry["processed_data_sha256"]
    assert sha(ROOT/"data/splits/phase2_split_manifest.csv")==registry["split_sha256"]
    data=pd.read_csv(source,parse_dates=["Timestamp"],float_precision="round_trip")
    split=pd.read_csv(ROOT/"data/splits/phase2_split_manifest.csv")
    assert data.Source_Row.tolist()==split.Source_Row.tolist()
    assert data.Timestamp.is_monotonic_increasing and data.Timestamp.is_unique
    intervals=data[["Source_Row","Timestamp","Room_Occupancy_Count"]].copy()
    intervals["next_timestamp"]=data.Timestamp.shift(-1)
    intervals["partition"]=split.partition
    intervals["next_partition"]=split.partition.shift(-1)
    intervals["elapsed_seconds"]=(intervals.next_timestamp-intervals.Timestamp).dt.total_seconds()
    reasons=[]
    for r in intervals.itertuples():
        if pd.isna(r.elapsed_seconds):reason="final_row_unknown_duration"
        elif r.elapsed_seconds<=0:raise ValueError("Nonpositive source interval")
        elif r.elapsed_seconds>config["max_interval_seconds"]:reason="recording_gap_over_60s"
        elif r.partition=="embargo" or r.next_partition=="embargo":reason="touches_embargo"
        elif r.partition!=r.next_partition:reason="crosses_partition"
        else:reason="eligible"
        reasons.append(reason)
    intervals["status"]=reasons;intervals["eligible_seconds"]=np.where(intervals.status.eq("eligible"),intervals.elapsed_seconds,0)
    intervals.to_csv(TABLES/"interval_audit.csv",index=False)
    coverage=intervals.groupby("status",sort=False).agg(interval_rows=("Source_Row","size"),
             known_elapsed_seconds=("elapsed_seconds","sum"),accounted_seconds=("eligible_seconds","sum")).reset_index()
    coverage.to_csv(TABLES/"coverage.csv",index=False)
    mask=intervals.status.eq("eligible")
    rows=data.loc[mask].reset_index(drop=True)
    audit=intervals.loc[mask].reset_index(drop=True)
    predictions={}
    for model in ["baseline","ga"]:
        predictions[model]=OccupancySystem(model).predict(rows).predicted_occupancy.to_numpy()
    predictions["true_label_reference"]=rows.Room_Occupancy_Count.to_numpy()
    fuzzy=FuzzyRoomController();layer=EnergyDecisionLayer();cache={};records=[]
    policies=["ECO","NORMAL","COMFORT","AUTO"]
    for model,occupancy in predictions.items():
        for i,(r,a) in enumerate(zip(rows.itertuples(index=False),audit.itertuples(index=False))):
            o=int(occupancy[i]);key=(o,r.S1_Temp,r.S1_Light,r.S5_CO2)
            if key not in cache:cache[key]=fuzzy.evaluate(*key,trace=False)
            f=cache[key];used=f["inputs_used"]
            for policy in policies:
                e=layer.apply(f["lighting_percent"],f["ventilation_percent"],o,used["temperature"],used["co2"],policy)
                records.append({"source":model,"policy":policy,"mode":e["mode"],"mode_reason":e["reason"],
                    "Source_Row":r.Source_Row,"Timestamp":r.Timestamp,"next_timestamp":a.next_timestamp,
                    "partition":a.partition,"duration_seconds":a.eligible_seconds,"true_occupancy":int(r.Room_Occupancy_Count),
                    "supplied_occupancy":o,"temperature":used["temperature"],"co2":used["co2"],
                    "fuzzy_lighting_percent":f["lighting_percent"],"fuzzy_ventilation_percent":f["ventilation_percent"],
                    "lighting_percent":e["lighting_percent"],"ventilation_percent":e["ventilation_percent"],
                    "ventilation_guard_applied":e["ventilation_guard_applied"],
                    "input_clipping_count":len(f["input_clipping"]),
                    "lighting_w":e["lighting_w"],"ventilation_w":e["ventilation_w"],"total_w":e["total_w"],
                    "lighting_kwh":layer.energy_kwh(e["lighting_w"],a.eligible_seconds),
                    "ventilation_kwh":layer.energy_kwh(e["ventilation_w"],a.eligible_seconds),
                    "energy_kwh":layer.energy_kwh(e["total_w"],a.eligible_seconds),
                    "reference_kwh":layer.energy_kwh(e["reference_full_power_w"],a.eligible_seconds)})
            if i and i%3000==0:print(f"{model}: {i}/{len(rows)} eligible intervals processed",flush=True)
    replay=pd.DataFrame(records).sort_values(["source","policy","Timestamp"]).reset_index(drop=True)
    replay["occupied_seconds"]=np.where(replay.true_occupancy.gt(0),replay.duration_seconds,0)
    replay["false_empty_seconds"]=np.where(replay.true_occupancy.gt(0)&replay.supplied_occupancy.eq(0),replay.duration_seconds,0)
    replay["occupied_zero_lighting_seconds"]=np.where(replay.true_occupancy.gt(0)&replay.lighting_percent.eq(0),replay.duration_seconds,0)
    grouped=replay.groupby(["source","policy"],sort=False)
    contiguous=replay.Timestamp.eq(grouped.next_timestamp.shift())&replay.partition.eq(grouped.partition.shift())
    replay["mode_change"]=contiguous&replay["mode"].ne(grouped["mode"].shift())
    replay.to_csv(OUT/"interval_simulation.csv",index=False)
    summary_rows=[]
    for part in ["train","validation","test","all_eligible"]:
        subset=replay if part=="all_eligible" else replay[replay.partition.eq(part)]
        for (source_name,policy),g in subset.groupby(["source","policy"]):
            energy=g.energy_kwh.sum();reference=g.reference_kwh.sum();occupied=g.occupied_seconds.sum()
            summary_rows.append({"partition":part,"source":source_name,"policy":policy,"intervals":len(g),
                "covered_hours":g.duration_seconds.sum()/3600,"lighting_kwh":g.lighting_kwh.sum(),
                "ventilation_kwh":g.ventilation_kwh.sum(),"simulated_kwh":energy,"full_power_reference_kwh":reference,
                "simulated_reduction_kwh":reference-energy,"simulated_reduction_percent":100*(reference-energy)/reference if reference else np.nan,
                "occupied_hours":occupied/3600,"false_empty_hours":g.false_empty_seconds.sum()/3600,
                "occupied_zero_lighting_hours":g.occupied_zero_lighting_seconds.sum()/3600,
                "occupied_zero_lighting_percent":100*g.occupied_zero_lighting_seconds.sum()/occupied if occupied else np.nan,
                "mode_changes_within_contiguous_coverage":int(g.mode_change.sum()),
                "clipped_intervals":int(g.input_clipping_count.gt(0).sum())})
    summary=pd.DataFrame(summary_rows);summary.to_csv(TABLES/"energy_summary.csv",index=False)
    modes=replay.groupby(["partition","source","policy","mode"]).agg(seconds=("duration_seconds","sum"),intervals=("Source_Row","size"),
              energy_kwh=("energy_kwh","sum")).reset_index()
    modes["hours"]=modes.seconds/3600;modes.to_csv(TABLES/"mode_time.csv",index=False)
    daily=replay.assign(date=replay.Timestamp.dt.strftime("%Y-%m-%d")).groupby(["source","policy","date"]).agg(
        hours=("duration_seconds",lambda x:x.sum()/3600),simulated_kwh=("energy_kwh","sum"),
        full_power_kwh=("reference_kwh","sum")).reset_index()
    daily.to_csv(TABLES/"daily_energy.csv",index=False)
    # Same frozen requests, alternative hypothetical load ratios (no tuning).
    sensitivity=[]
    for lw,vw in [(60,40),(30,80),(100,20)]:
        for (source_name,policy),g in replay[replay.partition.eq("test")].groupby(["source","policy"]):
            e=((lw*g.lighting_percent/100+vw*g.ventilation_percent/100)*g.duration_seconds/3_600_000).sum()
            ref=(lw+vw)*g.duration_seconds.sum()/3_600_000
            sensitivity.append({"lighting_max_w":lw,"ventilation_max_w":vw,"source":source_name,"policy":policy,
                "test_simulated_kwh":e,"test_reference_kwh":ref,"test_reduction_percent":100*(ref-e)/ref})
    pd.DataFrame(sensitivity).to_csv(TABLES/"power_assumption_sensitivity.csv",index=False)
    scenarios=pd.read_csv(ROOT/"outputs/phase5/tables/scenario_results.csv");sr=[]
    for r in scenarios.itertuples():
        f=fuzzy.evaluate(r.occupancy,r.temperature,r.light,r.co2,trace=False)
        for policy in policies:
            e=layer.apply(f["lighting_percent"],f["ventilation_percent"],r.occupancy,
                f["inputs_used"]["temperature"],f["inputs_used"]["co2"],policy)
            sr.append({"scenario":r.scenario,**e,"one_hour_simulated_kwh":layer.energy_kwh(e["total_w"],3600)})
    scenario=pd.DataFrame(sr);scenario.to_csv(TABLES/"mode_scenarios.csv",index=False)
    manifest={"config_sha256":sha(ROOT/"config/phase6_energy.json"),
              "fuzzy_config_sha256":sha(ROOT/"config/phase5_fuzzy.json"),
              "phase4_registry_sha256":sha(ROOT/"models/phase4/model_registry.json"),
              "data_sha256":sha(source),"split_sha256":sha(ROOT/"data/splits/phase2_split_manifest.csv"),
              "eligible_intervals":int(mask.sum()),"covered_hours":float(audit.eligible_seconds.sum()/3600),
              "known_span_hours":float(intervals.elapsed_seconds.sum()/3600),
              "source_rows":len(data),"simulation_rows":len(replay),
              "numpy":np.__version__,"pandas":pd.__version__}
    (TABLES/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    import report_phase6
    report_phase6.main()
    print(summary[(summary.partition=="test")&(summary.policy=="AUTO")].to_string(index=False),flush=True)
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=="__main__":main()
