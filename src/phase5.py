"""Build Phase 5 demonstrations, rule tables, plots and notes."""
from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fuzzy_controller import FuzzyRoomController,SmartRoomSystem,membership
from phase1 import md
OUT=ROOT/"outputs/phase5";PLOTS=OUT/"plots";TABLES=OUT/"tables"

def save(fig,name):
    fig.savefig(PLOTS/f"{name}.png",dpi=160,bbox_inches="tight")
    fig.savefig(PLOTS/f"{name}.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)

def main():
    for path in [PLOTS,TABLES]:path.mkdir(parents=True,exist_ok=True)
    c=FuzzyRoomController();cfg=c.config
    scenario_specs=[
        ("empty_dark_fresh",0,24,0,450),
        ("empty_dark_poor_co2",0,24,0,1600),
        ("one_person_dark",1,24,20,450),
        ("three_people_dark",3,24,20,450),
        ("three_people_bright",3,24,700,450),
        ("crowded_hot_poor_co2",3,32,20,1600),
        ("one_person_bright",1,24,700,450),
        ("empty_hot_fresh",0,35,20,450),
        ("occupied_cool_fresh",2,16,200,450),
        ("overlapping_memberships",1.5,27,100,1150),
        ("above_domain_clipped",3,45,1200,2500)]
    rows=[];traces={}
    for name,o,t,l,co in scenario_specs:
        r=c.evaluate(o,t,l,co,curves=False)
        traces[name]=r
        rows.append({"scenario":name,"occupancy":o,"temperature":t,"light":l,"co2":co,
                     "lighting_percent":r["lighting_percent"],"ventilation_percent":r["ventilation_percent"],
                     "raw_lighting_centroid":r["mamdani_centroids"]["lighting"],
                     "exact_off":r["lighting_exact_off_applied"],"clipped_inputs":len(r["input_clipping"]),
                     "fired_rules":len(r["fired_rules"])})
    scenarios=pd.DataFrame(rows);scenarios.to_csv(TABLES/"scenario_results.csv",index=False)
    (OUT/"scenario_traces.json").write_text(json.dumps(traces,indent=2)+"\n")
    rules=pd.DataFrame([{"id":r["id"],"output":r["output"],"antecedent":" AND ".join(f"{k} is {v}" for k,v in r["if"].items()),"consequent":r["then"]} for r in cfg["rules"]])
    rules.to_csv(TABLES/"rule_base.csv",index=False)
    terms=pd.DataFrame([{"kind":kind,"variable":name,"unit":spec.get("unit","percent"),"term":label,
                        "a":p[0],"b":p[1],"c":p[2],"d":p[3]} for kind in ["inputs","outputs"]
                       for name,spec in cfg[kind].items() for label,p in spec["sets"].items()])
    terms.to_csv(TABLES/"membership_parameters.csv",index=False)
    # Real sensor examples: four validation labels, then examples of false-empty test errors.
    data=pd.read_csv(ROOT/"data/processed/occupancy_cleaned.csv",float_precision="round_trip")
    preds=pd.read_csv(ROOT/"outputs/phase4/predictions.csv")
    validation=preds[(preds.model=="baseline")&(preds.partition=="validation")]
    ids=validation.groupby("Room_Occupancy_Count",sort=True).Source_Row.first().tolist()
    false_empty=preds[(preds.model=="baseline")&(preds.partition=="test")&
                      (preds.Room_Occupancy_Count>0)&(preds.prediction==0)]
    ids+=false_empty.Source_Row.head(2).tolist()
    observations=data.set_index("Source_Row").loc[ids]
    examples=[];integration_traces=[]
    for model in ["baseline","ga"]:
        system=SmartRoomSystem(model)
        for source,row in observations.iterrows():
            r=system.evaluate(row.to_dict())
            reference=c.evaluate(int(row.Room_Occupancy_Count),row.S1_Temp,row.S1_Light,row.S5_CO2,trace=False)
            examples.append({"model":model,"Source_Row":source,"true_occupancy":int(row.Room_Occupancy_Count),
                "predicted_occupancy":r["predicted_occupancy"],"temperature":row.S1_Temp,"light":row.S1_Light,"co2":row.S5_CO2,
                "lighting_percent":r["lighting_percent"],"ventilation_percent":r["ventilation_percent"],
                "true_label_reference_lighting":reference["lighting_percent"],
                "true_label_reference_ventilation":reference["ventilation_percent"],
                "clipped_inputs":len(r["input_clipping"])})
            integration_traces.append({"Source_Row":int(source),"true_occupancy":int(row.Room_Occupancy_Count),**r})
    integration=pd.DataFrame(examples);integration.to_csv(TABLES/"integrated_examples.csv",index=False)
    (OUT/"integrated_traces.json").write_text(json.dumps(integration_traces,indent=2)+"\n")
    # Save an actual minimal GA+controller input example: one extra temperature channel.
    observations[["S1_Light","S5_CO2","S6_PIR","S7_PIR","S1_Temp"]].to_csv(ROOT/"data/examples/phase5_ga_controller.csv",index=False)
    baseline_features=json.loads((ROOT/"models/phase4/model_registry.json").read_text())["models"]["baseline"]["features"]
    observations[baseline_features].to_csv(ROOT/"data/examples/phase5_baseline_controller.csv",index=False)
    observations[["Date","Time","Room_Occupancy_Count"]].to_csv(ROOT/"data/examples/phase5_example_provenance.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"phase5"})
    for kind,name,ncols in [("inputs","input_memberships",2),("outputs","output_memberships",2)]:
        fig,axes=plt.subplots(int(np.ceil(len(cfg[kind])/ncols)),ncols,figsize=(12,7 if kind=="inputs" else 3.8),squeeze=False,layout="constrained")
        for ax,(var,spec) in zip(axes.flat,cfg[kind].items()):
            x=np.linspace(*spec["range"],1001)
            for label,p in spec["sets"].items():ax.plot(x,membership(x,p),label=label)
            ax.set(title=var,xlabel=spec.get("unit","requested level (%)"),ylabel="Membership degree",ylim=(-.02,1.08))
            ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
        save(fig,name)
    light=np.linspace(0,1000,61);occ=np.linspace(0,3,25)
    response=np.array([[c.evaluate(o,24,l,450,trace=False)["lighting_percent"] for l in light] for o in occ])
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    im=ax.pcolormesh(light,occ,response,shading="auto",vmin=0,vmax=100,cmap="viridis")
    ax.set(xlabel="Light (lux)",ylabel="Supplied occupancy",title="Lighting request (temperature 24 C, CO2 450 ppm)")
    fig.colorbar(im,ax=ax,label="Lighting (%)");save(fig,"lighting_response")
    co=np.linspace(300,2000,45);temp=np.linspace(10,40,31)
    response=np.array([[c.evaluate(2,t,100,v,trace=False)["ventilation_percent"] for v in co] for t in temp])
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    im=ax.pcolormesh(co,temp,response,shading="auto",vmin=0,vmax=100,cmap="viridis")
    ax.set(xlabel="CO2 (ppm)",ylabel="Temperature (C)",title="Ventilation request (occupancy 2, light 100 lux)")
    fig.colorbar(im,ax=ax,label="Ventilation (%)");save(fig,"ventilation_response")
    r=c.evaluate(1.5,27,100,1150,curves=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    for ax,out in zip(axes,["lighting","ventilation"]):
        x=c.universe;curve=np.array(r["curves"][out])
        for rule in r["fired_rules"]:
            if rule["output"]==out:
                ax.plot(x,np.minimum(rule["strength"],c.output_mfs[out][rule["then"]]),alpha=.35,lw=.8)
        ax.fill_between(x,curve,alpha=.35,color="#2b8a83",label="MAX aggregation")
        ax.axvline(r["mamdani_centroids"][out],color="#d18325",label=f"centroid {r['mamdani_centroids'][out]:.2f}%")
        ax.set(title=out,xlabel="Requested level (%)",ylabel="Aggregated membership",ylim=(0,1.02));ax.legend(frameon=False)
    save(fig,"worked_defuzzification")
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout="constrained")
    for ax,col in zip(axes,["lighting_percent","ventilation_percent"]):
        ax.barh(scenarios.scenario,scenarios[col],color="#2b8a83")
        ax.set(xlim=(0,100),xlabel="Requested level (%)",title=col.replace("_"," "))
    save(fig,"scenario_comparison")
    write_notes(c,scenarios,rules,terms,integration)
    manifest={"config_sha256":hashlib.sha256((ROOT/"config/phase5_fuzzy.json").read_bytes()).hexdigest(),
              "phase4_registry_sha256":hashlib.sha256((ROOT/"models/phase4/model_registry.json").read_bytes()).hexdigest(),
              "rules":len(cfg["rules"]),"synthetic_scenarios":len(scenarios),"integrated_example_rows":len(integration),
              "numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__}
    (TABLES/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(scenarios.to_string(index=False))
    print("Saved Phase 5 report and explanatory traces.")

def write_notes(c,scenarios,rules,terms,integration):
    notes=f"""# Phase 5 — Fuzzy Smart Room Controller

## What was built

A static Mamdani controller transforms occupancy, temperature, light and CO2 into requested
lighting and ventilation percentages. There are seven lighting rules and 27 ventilation
rules. The implementation is handwritten NumPy so every step can be explained.

It supports standalone scenario inputs and integration with either Phase 4 occupancy model.
No electricity calculation, hardware actuation, adaptive mode or final dashboard is included.
Those belong to later phases. All thresholds are illustrative design assumptions; none are
learned from control labels, calibrated against equipment, or represented as standards.

## Inputs, outputs and units

Occupancy: 0–3 people. Temperature: 10–40 degrees C. Light: 0–1000 lux.
CO2: 300–2000 ppm. Both outputs use a 0–100 requested-level scale.
These output percentages are not watts, fan airflow or demonstrated comfort/air-quality outcomes.

In integrated mode, occupancy comes from the chosen classifier.
Temperature is S1_Temp, light is S1_Light, and CO2 is S5_CO2.
These are explicit local measurements, not room-wide averages. Using S1_Light avoids pretending
that a four-input estimator also receives all four light sensors.

The baseline estimator needs its original 15 channels.
The GA estimator needs S1_Light, S5_CO2, S6_PIR and S7_PIR. Its controller additionally needs
S1_Temp. Therefore the compact complete estimator-plus-controller path requires **five channels**,
not four (66.67% fewer input channels than the 15-channel path, not measured power savings).
Temperature and light share source node S1; distinct node dependencies remain four.
A supplied target column is ignored by the prediction interface.

## Membership definitions

Each function is stored as [a, b, c, d]: zero outside [a,d], rises from a to b, stays at one
from b to c, and falls from c to d. b=c gives a triangle. Repeated left/right endpoints give
shoulders without division by zero. The full definitions are:

{md(terms)}

The labels fresh/moderate/poor describe this demonstration's CO2 fuzzy sets, not certified
air-quality categories. Values above/below the supported environmental universes are clipped
and each clipping is recorded. Nonfinite inputs and negative light/CO2 are rejected.
Occupancy outside 0–3 is rejected. Fractional occupancy is permitted for standalone rule
exploration; integrated classifier output is an integer.

## Rules and inference

{md(rules)}

Every rule's AND uses MIN. A rule clips its output membership function at its firing strength.
All rule contributions to an output are combined with MAX.
The centroid is integral(x * aggregated_membership) / integral(aggregated_membership).
The implementation uses trapezoidal numerical integration on a grid spaced 0.1 percentage points.
A zero-area result is an error, not an invented fallback.

For lighting, more occupancy or less light can increase the request.
For ventilation, poor CO2 requests high ventilation even when occupancy is empty.
Hot occupied rooms request high ventilation. Empty rooms with fresh CO2 use the idle set,
even if hot; this is an explicit occupancy-based design preference, not a thermal model.
All occupancy/CO2/temperature combinations have a ventilation rule.

A fuzzy set named off still has area above zero and hence a positive centroid (~3.33%).
An **explicit postprocessing gate** sets final lighting to exactly zero when the supplied
occupancy is exactly zero. The trace retains the raw centroid separately. This exact-off
gate is deterministic logic after Mamdani inference and introduces a boundary discontinuity
near occupancy zero. Bright occupied rooms may retain the small off-set centroid.
High ventilation generally defuzzifies near 90%, not 100%; that is a consequence of its shape.
There is no unreported output rounding or physical relay threshold.

## Scenarios

{md(scenarios.round(3))}

Scenario traces list all input membership degrees, fired rules and their strengths, raw
centroids, clipping, and final requests. The overlapping_memberships case illustrates several
rules firing at once. The above_domain_clipped case makes saturation visible.

## Integration examples and error propagation

{md(integration.round(3))}

Examples contain four real validation observations (one per true class) and two later-test
false-empty observations, each run through both estimators.
The true-label reference passes known occupancy through the SAME rule base, solely to
illustrate how occupancy errors propagate. True labels are never passed to the integrated
prediction pipeline. This reference is not a measured optimal action or new performance score.

Earlier evaluation found 184 of 294 occupied later-test rows predicted empty by both models.
This controller does not repair those occupancy estimates. The exact-off gate can therefore
request zero lighting in an occupied room. CO2-triggered ventilation remains available,
but it does not guarantee adequate control, particularly when CO2 is still low.
An error-free fuzzy implementation does not make the full system reliable.

The dataset does not contain desired control-output labels. Therefore no fuzzy-controller
accuracy, validated comfort score or actual energy-saving percentage can be reported.
Static response surfaces illustrate rules; they do not model CO2 dynamics, actuator response,
feedback of lighting into the light sensor, oscillation, hysteresis or real equipment.
The baseline remains the default academic demo; GA is an explicit alternative.
Model probabilities are uncalibrated and are not used as fuzzy occupancy degrees.

## Reproduction and files

~~~bash
python src/phase5.py
python src/verify_phase5.py
python src/predict_phase5.py --model baseline --input data/examples/phase5_baseline_controller.csv --output outputs/phase5/baseline_demo.csv --trace outputs/phase5/baseline_demo_trace.json
python src/predict_phase5.py --model ga --input data/examples/phase5_ga_controller.csv --output outputs/phase5/ga_demo.csv --trace outputs/phase5/ga_demo_trace.json
~~~

Prediction commands require new output filenames. Evaluation reruns regenerate only Phase 5
reports and examples. Earlier data/models remain unchanged.

- config/phase5_fuzzy.json: all membership parameters, rules, mappings and assumptions.
- src/fuzzy_controller.py: FuzzyRoomController and integrated SmartRoomSystem.
- src/phase5.py: reproducible plots, scenarios and real-row demonstrations.
- src/verify_phase5.py: coverage, centroid, rule, integration and interface tests.
- outputs/phase5/report.html: visual report.
- outputs/phase5/scenario_traces.json: inspectable rule-by-rule worked examples.
- outputs/phase5/integrated_traces.json: model output followed by fuzzy reasoning.
- outputs/phase5/tables: memberships, rule base, examples and verification results.
- outputs/phase5/plots: six PNG/SVG figures.

## Two-minute viva explanation

“I used Mamdani fuzzy logic to turn predicted occupancy and environmental readings into
lighting and ventilation requests. I defined overlapping fuzzy sets and 34 explicit rules.
For each input I calculate membership degrees. I use MIN for AND, clip each consequent,
combine with MAX and calculate the centroid to produce a crisp percentage.

For example, an empty dark room receives zero final lighting, while high CO2 can still
request ventilation. The exact-zero lighting is a clearly identified postprocessing rule,
because a fuzzy off set alone can have a small positive centroid.

I can show every fired rule and its strength. I tested overlap cases, input boundaries and
integration with both occupancy estimators. These are illustrative static control requests,
not real actuator commands or measured energy savings. Classification errors still affect
control, and temperature remains an additional controller measurement in the compact system.”

## Viva questions

**Why fuzzy logic?** It combines overlapping linguistic conditions into interpretable continuous requests.
**Membership versus probability?** Membership is compatibility with a linguistic set, not a probability of a class.
**Why MIN and MAX?** They are the chosen Mamdani AND and aggregation operators; the configuration states them.
**Why centroid?** It converts the combined fuzzy output into its centre of area.
**Why can high mean 90% rather than 100%?** The centroid is inside the membership shape, not necessarily at its peak.
**Is zero lighting purely fuzzy?** No; exact empty occupancy activates an explicit postprocessing gate.
**Why ventilate an empty room?** The illustrative rules allow residual elevated CO2 to request ventilation.
**Are thresholds optimal?** No, they are declared design choices and need validation for any real installation.
**Why five compact-path inputs?** The four-channel occupancy estimator still needs S1_Temp for control.
**Does this fix classifier errors?** No. Rule correctness and occupancy accuracy are separate issues.
**Where are energy savings?** Phase 6 will add explicitly simulated power assumptions.

## Reference

Mamdani control example and inference workflow:
https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_tipping_problem_newapi.html
The project implements its own rules and membership definitions, using NumPy rather than adding
a scikit-fuzzy dependency.
"""
    (ROOT/"docs/phase5_notes.md").write_text(notes)
    cards="".join(f'<section><h2>{title}</h2><img src="plots/{name}.png" alt="{title}"></section>' for name,title in [
        ("input_memberships","Input membership functions"),("output_memberships","Output membership functions"),
        ("lighting_response","Lighting response surface"),("ventilation_response","Ventilation response surface"),
        ("worked_defuzzification","Worked clipping, aggregation and centroids"),("scenario_comparison","Scenario results")])
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC Project: Phase 5</title><style>
body{{background:#eef2f5;color:#172b40;font:16px/1.6 system-ui;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}
section{{background:white;border-radius:8px;padding:24px;margin:24px 0}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dde4e8;text-align:left}}
th{{background:#e8f2f1}}.scroll{{overflow:auto}}a{{color:#166e75}}.note{{border-left:4px solid #da9439;padding-left:18px}}
</style><body><main><p>SOFT COMPUTING MINOR PROJECT / PHASE 5</p><h1>Mamdani fuzzy room controller</h1>
<section><h2>Four environmental inputs, two control requests</h2><p>Occupancy, temperature, light and CO2 feed
34 explainable rules. MIN combines conditions, clipping applies rules, MAX combines outputs,
and centroid defuzzification produces lighting and ventilation percentages.</p>
<p class="note">Illustrative thresholds and static requests only. No actual equipment control or measured
energy savings. Occupancy errors still propagate. The compact estimator-plus-controller needs five
channels, including an additional temperature measurement.</p>
<p><a href="../../docs/phase5_notes.md">Method and viva notes</a> /
<a href="../../config/phase5_fuzzy.json">Memberships and rules</a> /
<a href="scenario_traces.json">Worked rule traces</a></p></section>
<section><h2>Scenarios</h2><div class="scroll">{scenarios.round(3).to_html(index=False,border=0)}</div>
<p>The raw lighting centroid remains visible. Exact zero at occupancy zero is a separate
postprocessing gate. Environmental clipping is counted rather than hidden.</p></section>
{cards}<section><h2>Real-row integration examples</h2><div class="scroll">{integration.round(3).to_html(index=False,border=0)}</div>
<p>The true-label reference is the same rules with known occupancy for illustration, not a measured
correct control action. These selected examples are not an independent evaluation.</p></section>
<section><h2>Rule base</h2>{rules.to_html(index=False,border=0)}</section>
<section><h2>Next phase</h2><p>Phase 6 adds simulated energy estimates using declared appliance powers.
No energy calculations or adaptive modes are included in Phase 5.</p></section></main></body></html>"""
    (OUT/"report.html").write_text(page)

if __name__=="__main__":main()
