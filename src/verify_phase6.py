"""Independent checks of every interval, mode, energy row and aggregate."""
from pathlib import Path
import sys,json,hashlib,math,tempfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
import pandas as pd,numpy as np
from energy_controller import EnergyDecisionLayer,EnergyRoomSystem
from fuzzy_controller import FuzzyRoomController
from occupancy_system import OccupancySystem
OUT=ROOT/"outputs/phase6";T=OUT/"tables"

def main():
    cfg=json.loads((ROOT/"config/phase6_energy.json").read_text())
    manifest=json.loads((T/"run_manifest.json").read_text())
    for p,key in [("config/phase6_energy.json","config_sha256"),("config/phase5_fuzzy.json","fuzzy_config_sha256"),
                  ("models/phase4/model_registry.json","phase4_registry_sha256"),("data/processed/occupancy_cleaned.csv","data_sha256"),
                  ("data/splits/phase2_split_manifest.csv","split_sha256")]:
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==manifest[key]
    data=pd.read_csv(ROOT/"data/processed/occupancy_cleaned.csv",parse_dates=["Timestamp"],float_precision="round_trip")
    split=pd.read_csv(ROOT/"data/splits/phase2_split_manifest.csv")
    audit=pd.read_csv(T/"interval_audit.csv",parse_dates=["Timestamp","next_timestamp"])
    replay=pd.read_csv(OUT/"interval_simulation.csv",parse_dates=["Timestamp","next_timestamp"])
    summary=pd.read_csv(T/"energy_summary.csv")
    modes=pd.read_csv(T/"mode_time.csv")
    delta=(data.Timestamp.shift(-1)-data.Timestamp).dt.total_seconds()
    independent=(delta>0)&(delta<=cfg["max_interval_seconds"])&split.partition.ne("embargo")&split.partition.eq(split.partition.shift(-1))
    assert np.array_equal(independent,audit.status.eq("eligible"))
    assert audit.Source_Row.tolist()==data.Source_Row.tolist()
    assert np.allclose(audit.elapsed_seconds,delta,equal_nan=True)
    assert np.allclose(audit.eligible_seconds,np.where(independent,delta,0))
    assert math.isclose(delta.sum(),(data.Timestamp.iloc[-1]-data.Timestamp.iloc[0]).total_seconds(),abs_tol=1e-8)
    assert audit.status.iloc[-1]=="final_row_unknown_duration"
    assert math.isclose(audit.eligible_seconds.sum()/3600,manifest["covered_hours"],abs_tol=1e-12)
    valid=data.loc[independent].set_index("Source_Row")
    durations=audit.loc[independent].set_index("Source_Row")
    assert len(valid)==manifest["eligible_intervals"]
    assert len(replay)==manifest["simulation_rows"]==len(valid)*12
    assert not replay.duplicated(["source","policy","Source_Row"]).any()
    expected_predictions={name:OccupancySystem(name).predict(valid).predicted_occupancy.to_numpy() for name in ["baseline","ga"]}
    expected_predictions["true_label_reference"]=valid.Room_Occupancy_Count.to_numpy()
    total_rated=sum(cfg["loads"].values())
    for (source,policy),g in replay.groupby(["source","policy"]):
        g=g.sort_values("Source_Row")
        assert g.Source_Row.tolist()==valid.index.tolist()
        assert np.array_equal(g.supplied_occupancy,expected_predictions[source])
        assert np.array_equal(g.true_occupancy,valid.Room_Occupancy_Count)
        assert np.allclose(g.duration_seconds,durations.eligible_seconds)
        assert (g.duration_seconds>0).all() and (g.duration_seconds<=60).all()
        if policy=="AUTO":
            high=(g.co2>=1100)|(g.supplied_occupancy>=2)|((g.supplied_occupancy>0)&(g.temperature>=28))
            mode=np.where(high,"COMFORT",np.where(g.supplied_occupancy>0,"NORMAL","ECO"))
        else:mode=np.full(len(g),policy)
        assert np.array_equal(g["mode"],mode)
        scale=np.array([cfg["modes"][m]["lighting_scale"] for m in mode])
        vscale=np.array([cfg["modes"][m]["ventilation_scale"] for m in mode])
        light=np.where(g.supplied_occupancy.eq(0),0,np.minimum(100,g.fuzzy_lighting_percent*scale))
        proposed=np.minimum(100,g.fuzzy_ventilation_percent*vscale)
        vent=np.where(g.co2>=1100,np.maximum(proposed,g.fuzzy_ventilation_percent),proposed)
        assert np.allclose(g.lighting_percent,light,rtol=0,atol=1e-12)
        assert np.allclose(g.ventilation_percent,vent,rtol=0,atol=1e-12)
        assert np.array_equal(g.ventilation_guard_applied,(g.co2>=1100)&(proposed<g.fuzzy_ventilation_percent))
        lw=cfg["loads"]["lighting_max_w"]*light/100
        vw=cfg["loads"]["ventilation_max_w"]*vent/100
        for col,values in {"lighting_w":lw,"ventilation_w":vw,"total_w":lw+vw,
                           "lighting_kwh":lw*g.duration_seconds/3_600_000,
                           "ventilation_kwh":vw*g.duration_seconds/3_600_000,
                           "energy_kwh":(lw+vw)*g.duration_seconds/3_600_000,
                           "reference_kwh":total_rated*g.duration_seconds/3_600_000}.items():
            assert np.allclose(g[col],values,rtol=0,atol=1e-12),col
        assert (g.total_w>=0).all() and (g.total_w<=total_rated+1e-10).all()
        assert np.allclose(g.false_empty_seconds,np.where((g.true_occupancy>0)&(g.supplied_occupancy==0),g.duration_seconds,0))
        assert np.allclose(g.occupied_zero_lighting_seconds,np.where((g.true_occupancy>0)&(g.lighting_percent==0),g.duration_seconds,0))
        expected_changes=g.Timestamp.eq(g.next_timestamp.shift())&g.partition.eq(g.partition.shift())&g["mode"].ne(g["mode"].shift())
        assert np.array_equal(expected_changes,g.mode_change)
    for r in summary.itertuples():
        g=replay[(replay.source==r.source)&(replay.policy==r.policy)]
        if r.partition!="all_eligible":g=g[g.partition==r.partition]
        assert len(g)==r.intervals
        occupied=g.loc[g.true_occupancy>0,"duration_seconds"].sum()
        values={"covered_hours":g.duration_seconds.sum()/3600,"simulated_kwh":g.energy_kwh.sum(),
                "lighting_kwh":g.lighting_kwh.sum(),"ventilation_kwh":g.ventilation_kwh.sum(),
                "full_power_reference_kwh":g.reference_kwh.sum(),"simulated_reduction_kwh":g.reference_kwh.sum()-g.energy_kwh.sum(),
                "simulated_reduction_percent":100*(1-g.energy_kwh.sum()/g.reference_kwh.sum()),
                "occupied_hours":occupied/3600,"false_empty_hours":g.false_empty_seconds.sum()/3600,
                "occupied_zero_lighting_hours":g.occupied_zero_lighting_seconds.sum()/3600,
                "occupied_zero_lighting_percent":100*g.occupied_zero_lighting_seconds.sum()/occupied}
        for k,v in values.items():assert math.isclose(getattr(r,k),v,rel_tol=1e-10,abs_tol=1e-10),(r.source,r.policy,r.partition,k)
        assert r.mode_changes_within_contiguous_coverage==g.mode_change.sum()
    for r in modes.itertuples():
        g=replay[(replay.source==r.source)&(replay.policy==r.policy)&(replay.partition==r.partition)&(replay["mode"]==r.mode)]
        assert r.intervals==len(g) and math.isclose(r.seconds,g.duration_seconds.sum(),abs_tol=1e-8)
        assert math.isclose(r.energy_kwh,g.energy_kwh.sum(),abs_tol=1e-10)
    sensitivity=pd.read_csv(T/"power_assumption_sensitivity.csv")
    for r in sensitivity.itertuples():
        g=replay[(replay.partition=="test")&(replay.source==r.source)&(replay.policy==r.policy)]
        e=((r.lighting_max_w*g.lighting_percent+r.ventilation_max_w*g.ventilation_percent)/100*g.duration_seconds/3_600_000).sum()
        ref=(r.lighting_max_w+r.ventilation_max_w)*g.duration_seconds.sum()/3_600_000
        assert math.isclose(r.test_simulated_kwh,e,abs_tol=1e-10)
        assert math.isclose(r.test_reduction_percent,100*(1-e/ref),abs_tol=1e-9)
    c=FuzzyRoomController()
    for r in replay.sample(40,random_state=42).itertuples():
        raw=data.loc[data.Source_Row.eq(r.Source_Row)].iloc[0]
        f=c.evaluate(r.supplied_occupancy,raw.S1_Temp,raw.S1_Light,raw.S5_CO2,trace=False)
        assert math.isclose(r.fuzzy_lighting_percent,f["lighting_percent"],abs_tol=1e-10)
        assert math.isclose(r.fuzzy_ventilation_percent,f["ventilation_percent"],abs_tol=1e-10)
    layer=EnergyDecisionLayer()
    assert layer.energy_kwh(100,3600)==.1 and layer.energy_kwh(18,3600)==.018
    assert layer.energy_kwh(100,0)==0
    assert layer.apply(30,0,1,24,450,"NORMAL")["lighting_w"]==18
    assert layer.apply(30,50,1,24,1100,"ECO")["ventilation_percent"]==50
    assert layer.apply(30,50,1,24,1100,"ECO")["ventilation_guard_applied"]
    assert layer.apply(99,99,2,24,450,"COMFORT")["total_w"]==100
    assert layer.apply(30,50,0,24,450,"AUTO")["mode"]=="ECO"
    assert layer.apply(30,50,1,24,450,"AUTO")["mode"]=="NORMAL"
    assert layer.apply(30,50,1,28,450,"AUTO")["mode"]=="COMFORT"
    assert layer.apply(30,50,0,24,1100,"AUTO")["mode"]=="COMFORT"
    for power,seconds in [(-1,30),(30,-1),(np.inf,30),(30,np.nan)]:
        try:layer.energy_kwh(power,seconds)
        except ValueError:pass
        else:raise AssertionError("Invalid integration accepted")
    for args in [(101,50,1,24,450,"AUTO"),(30,50,4,24,450,"AUTO"),(30,50,1,np.nan,450,"AUTO"),(30,50,1,24,450,"bad")]:
        try:layer.apply(*args)
        except ValueError:pass
        else:raise AssertionError("Invalid decision input accepted")
    for model in ["baseline","ga"]:
        sensors=pd.read_csv(ROOT/f"data/examples/phase5_{model}_controller.csv").head(2)
        system=EnergyRoomSystem(model,"AUTO")
        (ROOT/"work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"work",prefix="phase6-check-") as tmp:
            tmp=Path(tmp);sensors.to_csv(tmp/"input.csv",index=False)
            cmd=[sys.executable,str(ROOT/"src/predict_phase6.py"),"--model",model,"--policy","AUTO","--seconds-per-row","30",
                 "--input",str(tmp/"input.csv"),"--output",str(tmp/"output.csv")]
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            out=pd.read_csv(tmp/"output.csv")
            for i,row in sensors.iterrows():
                expected=system.evaluate(row.to_dict(),30,trace=False)
                assert math.isclose(out.iloc[i].simulated_energy_kwh,expected["simulated_energy_kwh"],abs_tol=1e-12)
            repeat=subprocess.run(cmd,capture_output=True,text=True)
            assert repeat.returncode!=0 and "Choose a new output path" in repeat.stderr
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ("src","href") and not v.startswith(("https:","http:","#")):assert (OUT/v).exists(),v
    Links().feed((OUT/"report.html").read_text())
    assert len(list((OUT/"plots").glob("*.png")))==6
    result={"status":"passed","intervals_verified":len(valid),"simulation_rows_verified":len(replay),
            "summary_rows_verified":len(summary),"checks":[
            "Every interval independently classified; gaps, embargo and final unknown duration excluded",
            "All 12 source/policy runs use identical unique eligible source rows",
            "Both model predictions and true-label reference verified",
            "Every mode, guard, applied request, power and interval kWh independently recalculated",
            "All summary, mode-duration, transition, error-proxy and sensitivity values reconciled",
            "Known watt-hour examples and boundary/invalid inputs tested",
            "Both integrated CSV interfaces exercised and overwrite refusal tested",
            "Earlier configs/data/models remain referenced by verified hashes; report links resolve"]}
    (T/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
