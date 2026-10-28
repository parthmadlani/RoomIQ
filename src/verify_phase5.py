"""Independent analytical centroid, coverage, scenarios and integration checks."""
from pathlib import Path
import sys,json,math,itertools,tempfile,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
import numpy as np,pandas as pd
from fuzzy_controller import FuzzyRoomController,SmartRoomSystem,membership
OUT=ROOT/"outputs/phase5"

def scalar_mu(x,p):
    a,b,c,d=p
    if x<a or x>d:return 0.
    if b<=x<=c:return 1.
    if x<b:return (x-a)/(b-a)
    return (d-x)/(d-c)

def exact_centroid(config,values,output):
    # Recreate firing strengths independently, then integrate the exact piecewise-linear MAX envelope.
    active=[]
    for rule in config["rules"]:
        if rule["output"]!=output:continue
        strength=min(scalar_mu(values[k],config["inputs"][k]["sets"][v]) for k,v in rule["if"].items())
        if strength>0:active.append((config["outputs"][output]["sets"][rule["then"]],strength))
    knots={0.,100.}
    for p,s in active:
        a,b,c,d=p;knots.update(p);knots.add(a+s*(b-a));knots.add(d-s*(d-c))
    initial=sorted(knots)
    def clipped(x,term):return min(term[1],scalar_mu(x,term[0]))
    # Include intersections where the identity of the max curve can switch.
    for left,right in zip(initial[:-1],initial[1:]):
        for a,b in itertools.combinations(active,2):
            da=clipped(left,a)-clipped(left,b);db=clipped(right,a)-clipped(right,b)
            if da*db<0:knots.add(left+(right-left)*(-da)/(db-da))
    area=moment=0.
    points=sorted(knots)
    for x0,x1 in zip(points[:-1],points[1:]):
        y0=max(clipped(x0,a) for a in active);y1=max(clipped(x1,a) for a in active)
        dx=x1-x0;area+=dx*(y0+y1)/2
        moment+=dx*x0*(y0+y1)/2+dx*dx*(y0/6+y1/3)
    assert area>0
    return moment/area

def main():
    c=FuzzyRoomController();cfg=c.config
    manifest=json.loads((OUT/"tables/run_manifest.json").read_text())
    assert hashlib.sha256((ROOT/"config/phase5_fuzzy.json").read_bytes()).hexdigest()==manifest["config_sha256"]
    assert hashlib.sha256((ROOT/"models/phase4/model_registry.json").read_bytes()).hexdigest()==manifest["phase4_registry_sha256"]
    assert len(cfg["rules"])==34 and len({r["id"] for r in cfg["rules"]})==34
    for kind in ["inputs","outputs"]:
        for var,spec in cfg[kind].items():
            x=np.linspace(*spec["range"],1001)
            curves=[]
            for label,p in spec["sets"].items():
                assert p==sorted(p)
                mu=membership(x,p);assert np.isfinite(mu).all() and ((mu>=0)&(mu<=1)).all()
                assert np.allclose(mu,[scalar_mu(v,p) for v in x],atol=1e-14)
                curves.append(mu)
            assert (np.max(curves,axis=0)>0).all(),var
    grid=itertools.product(np.linspace(0,3,13),[10,18,23,24,28,30,40],
                           [0,50,100,200,300,400,1000],[300,550,700,950,1200,1500,2000])
    count=0;fired=set()
    for o,t,l,co in grid:
        r=c.evaluate(o,t,l,co)
        for name in ["lighting_percent","ventilation_percent"]:assert np.isfinite(r[name]) and 0<=r[name]<=100
        if o==0:assert r["lighting_percent"]==0
        fired.update(rule["id"] for rule in r["fired_rules"]);count+=1
    assert fired=={r["id"] for r in cfg["rules"]}
    rng=np.random.default_rng(42);max_error=0.
    cases=[dict(occupancy=o,temperature=24,light=l,co2=co) for o,l,co in [(0,0,450),(0,0,1600),(1.5,100,1150),(3,700,1600)]]
    cases += [dict(occupancy=float(rng.uniform(0,3)),temperature=float(rng.uniform(10,40)),
                   light=float(rng.uniform(0,1000)),co2=float(rng.uniform(300,2000))) for _ in range(36)]
    for values in cases:
        r=c.evaluate(**values)
        for output in cfg["outputs"]:
            reference=exact_centroid(cfg,values,output)
            error=abs(reference-r["mamdani_centroids"][output]);max_error=max(max_error,error)
            assert error<.005,(values,output,error)
    assert c.evaluate(0,24,0,450)["lighting_percent"]==0
    assert c.evaluate(0,24,0,450)["mamdani_centroids"]["lighting"]>0
    for o in [0,1,2,3]:
        assert c.evaluate(o,24,0,1600)["ventilation_percent"]>85
    for o in [1,2,3]:
        assert c.evaluate(o,24,20,450)["lighting_percent"]>c.evaluate(o,24,700,450)["lighting_percent"]
    assert c.evaluate(3,24,20,450)["lighting_percent"]>c.evaluate(1,24,20,450)["lighting_percent"]
    clipped=c.evaluate(3,45,1200,2500)
    assert len(clipped["input_clipping"])==3
    assert clipped["inputs_used"]==dict(occupancy=3.,temperature=40.,light=1000.,co2=2000.)
    bads=[(-1,24,0,450),(4,24,0,450),(1,np.nan,0,450),(1,24,-1,450),(1,24,0,-1),(1,24,0,np.inf),("invalid",24,0,450)]
    for args in bads:
        try:c.evaluate(*args)
        except ValueError:pass
        else:raise AssertionError("Invalid input accepted")
    scenarios=pd.read_csv(OUT/"tables/scenario_results.csv")
    for r in scenarios.itertuples():
        actual=c.evaluate(r.occupancy,r.temperature,r.light,r.co2)
        assert math.isclose(actual["lighting_percent"],r.lighting_percent,abs_tol=1e-12)
        assert math.isclose(actual["ventilation_percent"],r.ventilation_percent,abs_tol=1e-12)
    examples=pd.read_csv(OUT/"tables/integrated_examples.csv")
    data=pd.read_csv(ROOT/"data/processed/occupancy_cleaned.csv",float_precision="round_trip").set_index("Source_Row")
    for model in ["baseline","ga"]:
        system=SmartRoomSystem(model)
        for row in examples[examples.model.eq(model)].itertuples():
            sensors=data.loc[row.Source_Row].to_dict()
            result=system.evaluate(sensors)
            assert result["predicted_occupancy"]==row.predicted_occupancy
            assert math.isclose(result["lighting_percent"],row.lighting_percent,abs_tol=1e-12)
            assert math.isclose(result["ventilation_percent"],row.ventilation_percent,abs_tol=1e-12)
            # Arbitrary targets cannot change inference.
            sensors["Room_Occupancy_Count"]=999
            assert system.evaluate(sensors)["predicted_occupancy"]==row.predicted_occupancy
        sample=pd.read_csv(ROOT/f"data/examples/phase5_{model}_controller.csv",float_precision="round_trip")
        (ROOT/"work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"work",prefix="phase5-check-") as tmp:
            tmp=Path(tmp);sample.head(2).to_csv(tmp/"input.csv",index=False)
            command=[sys.executable,str(ROOT/"src/predict_phase5.py"),"--model",model,"--input",str(tmp/"input.csv"),
                     "--output",str(tmp/"output.csv"),"--trace",str(tmp/"trace.json")]
            subprocess.run(command,check=True,capture_output=True,text=True)
            saved=pd.read_csv(tmp/"output.csv")
            assert len(saved)==2 and len(json.loads((tmp/"trace.json").read_text()))==2
            for i,r in saved.iterrows():
                expected=system.evaluate(sample.iloc[i].to_dict())
                assert math.isclose(r.lighting_percent,expected["lighting_percent"],abs_tol=1e-12)
            repeat=subprocess.run(command,capture_output=True,text=True)
            assert repeat.returncode!=0 and "Use new output paths" in repeat.stderr
    ga=SmartRoomSystem("ga")
    sensors=pd.read_csv(ROOT/"data/examples/phase5_ga_controller.csv").iloc[0].to_dict()
    del sensors["S1_Temp"]
    try:ga.evaluate(sensors)
    except ValueError as e:assert "Missing controller measurements" in str(e)
    else:raise AssertionError("GA path inferred missing temperature")
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ("src","href") and not v.startswith(("http:","https:","#")):assert (OUT/v).exists(),v
    Links().feed((OUT/"report.html").read_text())
    assert len(list((OUT/"plots").glob("*.png")))==6
    result={"status":"passed","grid_cases":count,"rules_activated":len(fired),
            "independent_centroid_cases":len(cases),"max_centroid_error_percentage_points":max_error,
            "centroid_tolerance_percentage_points":.005,
            "checks":["All membership terms independently matched; input/output universes covered",
                      "Every rule activated; no empty aggregate or out-of-range request on test grid",
                      "Centroids compared with independent exact piecewise-linear envelope integration",
                      "Empty-off, high-CO2 ventilation and dark/bright scenarios verified",
                      "Clipping disclosed; invalid values rejected",
                      "Both occupancy integration paths and CSV/trace CLIs verified",
                      "Ground-truth column ignored; missing GA temperature rejected",
                      "Earlier registry unchanged; report links and figures present"]}
    (OUT/"tables/verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
