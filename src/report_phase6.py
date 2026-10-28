"""Generate Phase 6 plots and professor notes from replay results."""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase1 import md
OUT=ROOT/"outputs/phase6";T=OUT/"tables";P=OUT/"plots"
def save(fig,name):
    fig.savefig(P/f"{name}.png",dpi=160,bbox_inches="tight")
    fig.savefig(P/f"{name}.svg",bbox_inches="tight",metadata={"Date":None});plt.close(fig)

def main():
    summary=pd.read_csv(T/"energy_summary.csv")
    coverage=pd.read_csv(T/"coverage.csv")
    manifest=json.loads((T/"run_manifest.json").read_text())
    replay=pd.read_csv(OUT/"interval_simulation.csv")
    modes=pd.read_csv(T/"mode_time.csv")
    sensitivity=pd.read_csv(T/"power_assumption_sensitivity.csv")
    scenarios=pd.read_csv(T/"mode_scenarios.csv")
    test=summary[summary.partition.eq("test")]
    auto=test[test.policy.eq("AUTO")]
    baseline=auto[auto.source.eq("baseline")].iloc[0]
    config=json.loads((ROOT/"config/phase6_energy.json").read_text())
    policies=["ECO","NORMAL","COMFORT","AUTO"];sources=["baseline","ga","true_label_reference"]
    colors=["#496d9b","#2b8a83","#a56185"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"phase6"})
    fig,ax=plt.subplots(figsize=(11,5),layout="constrained")
    for i,source in enumerate(sources):
        g=test[test.source.eq(source)].set_index("policy").loc[policies]
        ax.bar(np.arange(4)+(i-1)*.25,g.simulated_kwh,width=.24,label=source,color=colors[i])
    ax.axhline(baseline.full_power_reference_kwh,color="#d18325",ls="--",label="Assumed full-power operation")
    ax.set(xticks=range(4),xticklabels=policies,ylabel="Simulated kWh",title="Later-test energy over identical covered intervals")
    ax.legend(frameon=False,fontsize=8);save(fig,"energy_comparison")
    fig,ax=plt.subplots(figsize=(11,5),layout="constrained")
    for i,source in enumerate(sources):
        g=test[test.source.eq(source)].set_index("policy").loc[policies]
        ax.bar(np.arange(4)+(i-1)*.25,g.simulated_reduction_percent,width=.24,label=source,color=colors[i])
    ax.set(xticks=range(4),xticklabels=policies,ylim=(0,100),ylabel="Simulated reduction vs full power (%)",
           title="Savings depend on the comparator; no comfort equivalence established")
    ax.legend(frameon=False,fontsize=8,loc="center left",bbox_to_anchor=(1.01,.5));save(fig,"simulated_reduction")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    g=auto.set_index("source").loc[sources]
    bars=ax.bar(sources,g.occupied_zero_lighting_hours,color=colors)
    for b,value in zip(bars,g.occupied_zero_lighting_hours):ax.text(b.get_x()+b.get_width()/2,value+.02,f"{value:.3f} h",ha="center")
    ax.set(ylabel="Hours with true occupancy > 0 and lighting = 0",title="Lower energy can conceal missed occupancy (AUTO, later test)")
    save(fig,"occupied_zero_lighting")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    pivot=modes[(modes.partition=="test")&(modes.policy=="AUTO")].pivot(index="source",columns="mode",values="hours").reindex(sources).fillna(0)
    bottom=np.zeros(3)
    for mode,color in zip(["ECO","NORMAL","COMFORT"],["#2b8a83","#496d9b","#da9439"]):
        vals=pivot[mode] if mode in pivot else np.zeros(3)
        ax.bar(sources,vals,bottom=bottom,label=mode,color=color);bottom+=vals
    ax.set(ylabel="Covered hours",title="Rule-selected AUTO mode durations");ax.legend(frameon=False,loc="center left",bbox_to_anchor=(1.01,.5));save(fig,"auto_modes")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    for source in ["baseline","true_label_reference"]:
        g=replay[(replay.partition=="test")&(replay.policy=="AUTO")&(replay.source==source)].sort_values("Source_Row")
        ax.plot(g.duration_seconds.cumsum()/3600,g.energy_kwh.cumsum(),label=source)
        if source=="baseline":ax.plot(g.duration_seconds.cumsum()/3600,g.reference_kwh.cumsum(),ls="--",label="full-power reference")
    ax.set(xlabel="Accumulated covered hours (excluded gaps removed)",ylabel="Cumulative simulated kWh",title="Energy accumulation, not a continuous calendar timeline")
    ax.legend(frameon=False);save(fig,"cumulative_energy")
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    ratios=[(60,40),(30,80),(100,20)]
    for i,source in enumerate(["baseline","ga","true_label_reference"]):
        vals=[sensitivity[(sensitivity.source==source)&(sensitivity.policy=="AUTO")&(sensitivity.lighting_max_w==lw)&(sensitivity.ventilation_max_w==vw)].iloc[0].test_reduction_percent for lw,vw in ratios]
        ax.bar(np.arange(3)+(i-1)*.25,vals,width=.24,label=source,color=colors[i])
    ax.set(xticks=range(3),xticklabels=["Light 60 W\nVent 40 W","Light 30 W\nVent 80 W","Light 100 W\nVent 20 W"],
           ylabel="Simulated reduction (%)",ylim=(0,100),title="Sensitivity to assumed load ratings (AUTO, later test)")
    ax.legend(frameon=False,fontsize=8,loc="center left",bbox_to_anchor=(1.01,.5));save(fig,"power_sensitivity")
    compact=auto[["source","covered_hours","simulated_kwh","full_power_reference_kwh","simulated_reduction_percent","occupied_zero_lighting_hours","occupied_zero_lighting_percent"]]
    notes=f"""# Phase 6 — Energy Estimation & Adaptive Decision Layer

## What this phase does

The Phase 5 fuzzy requests now feed an illustrative linear power model and three explicit
modes: ECO, NORMAL and COMFORT. AUTO chooses among those modes using deterministic conditions.
This is rule-based adaptation, not online learning or an adaptive GA.

Every energy number is simulated. The source dataset has no electrical-consumption labels,
no measured desired actuator outputs and no evidence that these requests maintain comfort.
No equipment is actuated, and no tariffs or monetary savings are assumed.

## Power assumptions and equations

Nominal light power: **60 W**. Nominal ventilation power: **40 W**.
P_light = 60 x applied lighting percent / 100.
P_vent = 40 x applied ventilation percent / 100.
P_total = P_light + P_vent.
Energy_kWh = P_total x elapsed_seconds / 3,600,000.
Reduction_percent = 100 x (reference_kWh - simulated_kWh) / reference_kWh.

Example: 30% light request corresponds to 18 W. One hour at 18 W is 0.018 kWh.
Both loads at full power use 100 W, or 0.1 kWh per hour.
Linear scaling represents an assumed average duty-equivalent power; it is not a measured
dimmer/fan-speed power curve. HVAC, standby, sensors and processor/network loads are excluded.
Therefore the estimated savings do not quantify savings from GA sensor removal.

The comparator keeps both modelled loads at 100% over the SAME eligible intervals.
It is an assumed full-power baseline, not observed “conventional” consumption.
Comparing against a deliberately high full-power reference can give large reductions;
those reductions do not establish equal comfort or useful service.

All assumptions are editable in config/phase6_energy.json. After changes, regenerate outputs
and checks. The sensitivity table reuses frozen requests with three alternative load ratios
to show that percentages also depend on the assumed lighting/ventilation mix.

## Mode decisions

- ECO multiplies fuzzy lighting and ventilation requests by 0.8.
- NORMAL leaves both requests unchanged.
- COMFORT multiplies both by 1.15, capped at 100%.
- If CO2 is at least 1100 ppm, ventilation cannot be reduced below its original fuzzy request.
- Predicted occupancy zero still forces lighting to exactly zero in every mode.

AUTO chooses COMFORT if CO2 >= 1100 ppm, occupancy >= 2, or the room is predicted occupied
and temperature >= 28 C. Otherwise it chooses NORMAL if occupied, ECO if predicted empty.
Thresholds and multipliers are illustrative design preferences, not engineering standards.
The ventilation guard is not a safety guarantee or evidence of adequate air quality.
AUTO uses the same clipped environmental values that Phase 5 used, and clipping is counted.

There is no memory, learning, hysteresis, dwell time or physical feedback model.
Mode transitions within contiguous covered intervals are counted; transitions across gaps
are not counted. These choices must not be presented as a validated building-control policy.
Fixed ECO/NORMAL/COMFORT are comparisons, not three extra learned controllers.

## Time accounting

The replay uses the current sample's request until the next timestamp (left-endpoint
zero-order hold), but only if the next sample is within 60 seconds and remains in the same
Phase 2 partition. Intervals touching embargo rows are excluded. The final row has no known
duration. Longer gaps are not filled and are not treated as zero consumption.

Covered intervals: **{manifest['eligible_intervals']:,}**.
Accounted duration across all eligible partitions: **{manifest['covered_hours']:.4f} hours**.
Known first-to-last span: **{manifest['known_span_hours']:.4f} hours**.
Most of the elapsed calendar span is excluded recording gaps.

{md(coverage.round(3))}

Each source/policy uses exactly the same eligible intervals. Training, validation and test
are reported separately, plus an explicitly combined all_eligible view. The combined view
is descriptive and is not independent evaluation. Daily aggregates use the starting
observation's calendar date. The method assumes the sampled request and occupancy label
persist until the next eligible sample; sub-sample changes remain unknown.

## Later-test AUTO illustration

{md(compact.round(5))}

These energy percentages must be read alongside the occupied-zero-lighting column.
The baseline estimator's AUTO simulation requests zero lighting for
**{baseline.occupied_zero_lighting_hours:.4f} occupied hours**
(**{baseline.occupied_zero_lighting_percent:.2f}%** of covered occupied time).

That error is not an efficiency benefit. A lower energy figure can be caused by
misclassifying an occupied room as empty. The energy layer does not repair those mistakes.
No measured comfort or task-illumination threshold is available, so “nonzero lighting” is
also not proof of adequate service.

The true_label_reference path feeds the known occupancy label into the SAME fuzzy and mode
rules. It is solely a counterfactual diagnostic of occupancy-error propagation. It is not an
available production predictor, a measured optimal-control policy, or an independent benchmark.
Ground-truth labels never enter either predicted-occupancy pipeline.

## Why baseline and GA energy can match

In the covered later-test replay, baseline and GA produce identical power requests under
these rules even though some occupancy counts differ. Their disagreements there are between
two and three people, both of which have full membership in the same high-occupancy set
and select the same AUTO mode. Environmental inputs are shared. No sensor-node power is
modelled, so reducing estimator inputs alone creates no simulated energy benefit here.

## Outputs and reproduction

~~~bash
python src/phase6.py
python src/verify_phase6.py
python src/predict_phase6.py --model baseline --policy AUTO --seconds-per-row 30 --input data/examples/phase5_baseline_controller.csv --output outputs/phase6/demo_energy.csv
~~~

The prediction CLI takes an explicitly hypothetical duration for EACH row. It does not
integrate input timestamps or fill gaps. Use phase6.py for the audited dataset replay.
The CLI refuses to overwrite output files. Generated Phase 6 reports/tables are overwritten
on rerun; prior-phase data and models remain unchanged.

- src/energy_controller.py: reusable EnergyDecisionLayer and integrated EnergyRoomSystem.
- config/phase6_energy.json: nominal powers, mode multipliers, thresholds and exclusions.
- outputs/phase6/interval_simulation.csv: every eligible interval/source/policy and power/energy.
- outputs/phase6/tables/interval_audit.csv: every source row and its inclusion/exclusion reason.
- outputs/phase6/tables/energy_summary.csv: component kWh, reference and reduction, service-error proxies.
- outputs/phase6/tables/mode_time.csv: mode durations.
- outputs/phase6/tables/mode_scenarios.csv: Phase 5 scenarios under each policy.
- outputs/phase6/tables/power_assumption_sensitivity.csv: assumed-power sensitivity.
- outputs/phase6/report.html: visual report; six figures are available in PNG and SVG.

## Professor explanation

“I connected the fuzzy controller to a transparent energy simulation. I assumed a 60-watt
light load and a 40-watt ventilation load, multiplied by the requested percentages, and
integrated power over actual short recorded intervals. I excluded long gaps and unknown
durations rather than assuming continuous data.

ECO, NORMAL and COMFORT scale the requests, while AUTO selects a mode from occupancy,
temperature and CO2. These are declared deterministic rules, not learned adaptation.

I compared simulated consumption with both loads continuously at full power over exactly
the same covered intervals. I also report occupied periods where the model asks for zero
lighting, because otherwise classification mistakes could look like savings. The results
are simulation estimates under assumptions, not measured building energy savings.”

## Viva questions

**Power versus energy?** Power is the instantaneous rate in watts. Energy accumulates it over time, in Wh or kWh.
**Why divide by 3,600,000?** Convert watt-seconds to watt-hours, then kilowatt-hours.
**Why not assume every row is 30 seconds?** Recorded intervals vary and long gaps exist.
**Does AUTO learn?** No. It is deterministic rule-based mode selection.
**Does COMFORT guarantee comfort?** No; it is a mode name and scaling preference, not a validated comfort measure.
**Why preserve ventilation at elevated CO2?** To avoid scaling down the original request in that illustrative condition.
**Does GA yield measured energy savings?** No. Sensor-node power is not modelled.
**Why include true-label reference?** To demonstrate how occupancy errors change simulated requests.
**Can larger savings mean a worse controller?** Yes, if it withholds useful service during occupied intervals.
**What remains?** Phase 7 integrates the existing modules into a dashboard and presentation.
"""
    (ROOT/"docs/phase6_notes.md").write_text(notes)
    cards="".join(f'<section><h2>{title}</h2><img src="plots/{name}.png" alt="{title}"></section>' for name,title in [
        ("energy_comparison","Later-test simulated energy"),("simulated_reduction","Reduction against the assumed comparator"),
        ("occupied_zero_lighting","Read energy together with occupancy errors"),("auto_modes","AUTO mode durations"),
        ("cumulative_energy","Accumulation over covered hours"),("power_sensitivity","Power-assumption sensitivity")])
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC Project: Phase 6</title><style>
body{{background:#eef2f5;color:#172b40;font:16px/1.6 system-ui;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}
section{{background:white;border-radius:8px;padding:24px;margin:24px 0}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dde4e8;text-align:left}}
th{{background:#e8f2f1}}.scroll{{overflow:auto}}a{{color:#166e75}}.note{{border-left:4px solid #da9439;padding-left:18px}}
</style><body><main><p>SOFT COMPUTING MINOR PROJECT / PHASE 6</p><h1>Simulated energy and adaptive modes</h1>
<section><h2>Declared loads: 60 W lighting + 40 W ventilation</h2><p>ECO, NORMAL and COMFORT adjust the
fuzzy requests. AUTO selects among them using explicit occupancy, temperature and CO2 rules.</p>
<p class="note">All energy is simulated. The full-power comparator is assumed, not measured.
Baseline AUTO requests zero lighting for {baseline.occupied_zero_lighting_hours:.3f} occupied hours in
covered test intervals. Do not treat withheld service as a demonstrated efficiency gain.</p>
<p>{manifest['eligible_intervals']:,} eligible intervals cover {manifest['covered_hours']:.3f} hours.
Long gaps, embargo intervals and the unknown final duration are excluded.</p>
<p><a href="../../docs/phase6_notes.md">Methods and viva notes</a> /
<a href="../../config/phase6_energy.json">Power and mode assumptions</a> /
<a href="tables/interval_audit.csv">Interval audit</a></p></section>
<section><h2>AUTO comparison on identical later-test intervals</h2><div class="scroll">{compact.round(5).to_html(index=False,border=0)}</div>
<p>True-label reference is counterfactual, not a deployable predictor or measured correct control.
Present data have already been explored; these are descriptive simulations.</p></section>
{cards}<section><h2>Accounting coverage</h2>{coverage.round(3).to_html(index=False,border=0)}
<p>No HVAC, standby, sensor-node or processor/network energy is included. Real fan/dimmer dynamics and
physical environmental feedback are not modelled. Phase 7 dashboard integration has not started.</p></section></main></body></html>"""
    (OUT/"report.html").write_text(page)

if __name__=="__main__":main()
