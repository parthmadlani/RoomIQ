"""Create figures and professor notes from completed Phase 3 results."""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"work/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phase2
OUT=ROOT/"outputs/phase3";TABLES=OUT/"tables";PLOTS=OUT/"plots"

def main():
    lock=json.loads((ROOT/"models/phase3/selection_locked.json").read_text())
    metrics=pd.read_csv(TABLES/"comparison_metrics.csv")
    history=pd.read_csv(TABLES/"convergence.csv")
    pool=pd.read_csv(TABLES/"evaluated_subsets.csv",dtype={"chromosome":str})
    runs=pd.read_csv(TABLES/"run_winners.csv",dtype={"chromosome":str})
    mapping=pd.read_csv(TABLES/"chromosome_mapping.csv")
    config=json.loads((ROOT/"docs/phase3_protocol.json").read_text())
    phase2.PLOTS=PLOTS
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.hashsalt":"sc-phase3"})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for seed,group in history.groupby("seed"):
        axes[0].plot(group.generation,group.fitness,marker="o",markersize=3,label=f"seed {seed}")
        axes[1].plot(group.generation,group.selected_count,marker="o",markersize=3,label=f"seed {seed}")
    axes[0].axhline(lock["reference"]["fitness"],ls="--",color="gray",label="all 15 inputs")
    axes[0].set(xlabel="Generation (0 = initial population)",ylabel="Best penalized fitness",title="Evolution of fitness")
    axes[1].set(xlabel="Generation",ylabel="Selected inputs in generation winner",title="Evolution of subset size",ylim=(0,16))
    for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.15)
    phase2.save(fig,"convergence")
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    ax.scatter(pool.selected_count,pool.cv_macro_f1,alpha=.45,s=22,label="evaluated subsets")
    ax.scatter([lock["winner"]["selected_count"]],[lock["winner"]["cv_macro_f1"]],marker="*",s=230,color="#d18325",label="locked winner")
    ax.scatter([15],[lock["reference"]["cv_macro_f1"]],marker="D",s=60,color="#a56185",label="all-input reference")
    ax.set(xlabel="Number of sensor inputs",ylabel="Training-only pooled OOF macro F1",
           title="Observed search trade-off; not an exhaustive frontier",xticks=range(1,16))
    ax.legend(frameon=False);phase2.save(fig,"subset_tradeoff")
    fig,ax=plt.subplots(figsize=(10,6),layout="constrained")
    ax.barh(mapping.feature,mapping.selected,color=["#2b8a83" if v else "#cbd5df" for v in mapping.selected])
    for i,row in mapping.iterrows():ax.text(.04,i,"selected" if row.selected else "excluded",va="center",color="#172b40")
    ax.set(xlim=(0,1.2),xticks=[],title=f"Locked chromosome: {lock['winner']['chromosome']}")
    phase2.save(fig,"selected_sensors")
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,part in zip(axes,["validation","test"]):
        for i,name in enumerate(["baseline_15","ga_selected"]):
            r=metrics[(metrics.model==name)&(metrics.partition==part)].iloc[0]
            ax.bar(np.arange(3)+(i-.5)*.36,[r.accuracy,r.macro_f1_present_classes,r.balanced_accuracy_present_classes],width=.35,label=name)
        ax.set(xticks=range(3),xticklabels=["Accuracy","Macro F1\npresent classes","Balanced accuracy\npresent classes"],ylim=(0,1.08),
               title=part+(" (true classes 0, 2, 3)" if part=="test" else " (all four classes)"))
        ax.legend(frameon=False)
    phase2.save(fig,"external_comparison")
    for part in ["validation","test"]:
        cm=pd.read_csv(TABLES/f"ga_selected_{part}_confusion.csv",index_col=0).to_numpy()
        phase2.plot_confusion(cm,f"selected_{part}_confusion","GA subset: "+part+(" (no true class 1)" if part=="test" else ""))
    def row(name,part):return metrics[(metrics.model==name)&(metrics.partition==part)].iloc[0]
    base=row("baseline_15","validation");sel=row("ga_selected","validation");test=row("ga_selected","test")
    selected=lock["features"];count=len(selected);nodes=lock["selected_nodes"]
    validation_delta=sel.macro_f1_all_four-base.macro_f1_all_four
    conclusion=("improved" if validation_delta>0 else "decreased" if validation_delta<0 else "matched")
    stable=len(set(runs.chromosome))==1
    notes=f"""# Phase 3: Genetic Algorithm sensor selection

## Result and interpretation

Selected **{count} of 15 sensor channels** ({100*(15-count)/15:.1f}% fewer input columns):
{", ".join(selected)}.

Chromosome: {lock['winner']['chromosome']}.
Input order is recorded in outputs/phase3/tables/chromosome_mapping.csv.

Training-only pooled out-of-fold macro F1: **{lock['winner']['cv_macro_f1']:.4f}**.
All-input search reference macro F1: **{lock['reference']['cv_macro_f1']:.4f}**.
Selected fitness: **{lock['winner']['fitness']:.4f}**.

With identical 300-tree settings and the same December 22 final training data,
external validation macro F1 {conclusion} from **{base.macro_f1_all_four:.4f}** to
**{sel.macro_f1_all_four:.4f}** (difference {validation_delta:+.4f}).
Selected-model validation accuracy: **{sel.accuracy:.2%}**.
Selected-model later-test accuracy: **{test.accuracy:.2%}**; balanced accuracy across its
three supported classes: **{test.balanced_accuracy_present_classes:.2%}**.

The GA optimizes its inner training objective; external improvement is not guaranteed.
Do not claim the subset preserves accuracy solely because its inner fitness is higher.
Both validation and test were already reported in Phase 2, so these are descriptive
comparisons, not new independent confirmation. No choices were changed after this comparison.

## Chromosome and fitness

Each bit corresponds to one direct sensor channel in the Phase 2 input order.
1 means include the channel; 0 means omit it. There are 2^15 - 1 = 32,767 nonempty subsets.
A zero chromosome is repaired by activating a random bit.

Fitness = pooled OOF macro F1 - 0.05 x (selected inputs / 15).

Macro F1 values each occupancy class equally. The penalty favours smaller subsets.
At most 0.05 F1-equivalent units are penalized for all 15 inputs; one extra input costs
0.05 / 15 = 0.003333 fitness units. This is a fixed modelling preference, not a measured
economic or energy cost. It was fixed before the search and never tuned on external scores.

OOF means each training row is predicted by an inner model that did not fit that row.
All OOF predictions are pooled, then one four-class macro F1 is computed. We do not average
fold F1 scores, because some validation blocks contain only a subset of the four classes.

## Training-only fold construction

Only the 1,462 Phase 2 training rows from December 22 enter the GA.
They are divided into four contiguous equal-row validation blocks.
For each block, fit rows come from the remaining December 22 data, excluding anything
within ten minutes before or after the held-out block. Every inner fit has all four classes.
The fold membership and class supports are saved for audit.

This is purged blocked within-day interpolation: some fit rows are later than the validation
block. It is **not** a forward-chaining forecast experiment. A ten-minute buffer reduces
immediate adjacency but does not prove independence. Slow environmental effects can last longer.

The external validation/test dates and embargo rows never enter GA fitting or fitness.
The previously unverified CO2 slope, timestamps, target and audit identifiers are not candidate
inputs. The original 15 direct channels are the complete search space.

## GA process and fixed settings

1. Initialize 16 nonempty chromosomes: all inputs, one random singleton and 14 random subsets.
2. Evaluate each new subset using four purged training-only folds.
3. Keep two elite copies unchanged.
4. Select parents by tournaments of three randomly sampled population positions.
5. Apply one-point crossover with probability 0.8 per pair.
6. Flip each bit independently with probability 1/15 per offspring (about one flip on average).
7. Repair empty offspring, evaluate, and repeat for 10 generations.
8. Run seeds 42, 43 and 44. Choose the highest penalized training fitness across the three run winners.
   Ties prefer fewer inputs, then the lexicographically smaller bit string.
9. Save selection_locked.json before predicting on external data.
10. Fit a final 300-tree model on the original training day using only the chosen inputs.

The initial population is generation 0; there are 11 logged populations per run.
The search uses 80-tree Random Forests to make repeated evaluation practical. All candidates,
including the all-input reference, use identical inner folds and estimator settings.
The final comparison uses the original 300-tree settings on both sides, isolating the
representation change as far as this setup permits. The 80-tree proxy can rank subsets
differently from a 300-tree estimator; its inner scores are not final-model held-out scores.
Forest seed is fixed at 42. GA seeds change the evolutionary search, not the data folds.

Repeated chromosomes reuse a deterministic fitness cache, shared across runs.
The search evaluated **{lock['unique_subsets_evaluated']} distinct subsets**.
The cache saves computation without changing fitness. This is a heuristic search,
not exhaustive optimization or proof of the globally best subset.

## Stability across runs

All three seeds chose the same chromosome: **{stable}**.
Different winners indicate search sensitivity. Even agreement across these three seeds
would not establish stability across different days, folds or datasets.

~~~text
{runs.to_string(index=False)}
~~~

## Full comparison

~~~text
{metrics.to_string(index=False,float_format=lambda v:f'{v:.4f}')}
~~~

Training scores are diagnostics. The January subset overlaps the later test and is not
independent evidence. The later test has no true class 1; four-class macro F1 is left blank.
Present-class macro F1 and balanced accuracy average only classes with actual support.
Per-class undefined precision/recall/F1 values are blank; inspect the saved supports.
Confusion matrices use actual class as rows and predicted class as columns.

## What “sensor reduction” means here

The chromosome selects individual measurement channels. The selected channels involve
{len(nodes)} distinct source nodes: {", ".join(nodes)}.
Nodes S1–S4 each originally carry temperature, light and sound channels.
Dropping one channel does not necessarily remove its node, radio, processor or power supply.
No hardware has been removed and no watts, cost or energy savings have been measured.
The feature count penalty is only a simple complexity proxy. A later hardware-aware
optimization would need node groups, actual costs and deployment constraints.

## Files and reproduction

- src/phase3.py: commented, handwritten GA operators, fitness, training and comparison.
- src/report_phase3.py: reports and six PNG/SVG figures.
- src/verify_phase3.py: independent metrics, fold and GA-history checks.
- src/predict_phase3.py: selected-model prediction command.
- docs/phase3_protocol.json: fixed settings and selection policy.
- models/phase3/selection_locked.json: choice made before external evaluation.
- models/phase3/ga_selected_random_forest.joblib: final selected-input model.
- models/phase3/model_metadata.json: features, hashes and environment.
- data/splits/phase3_inner_folds.csv: every inner fit/validation/purge assignment.
- outputs/phase3/tables: search cache, complete populations, convergence, winners and metrics.
- outputs/phase3/predictions.csv: external comparison predictions with source row identifiers.

From the project directory:
~~~bash
python src/phase3.py
python src/verify_phase3.py
python src/predict_phase3.py --input data/processed/baseline_features.csv --output outputs/phase3/example_predictions.csv
~~~

The prediction example uses known data and is not a new evaluation.
Generated Phase 3 outputs are overwritten on rerun. Earlier phase models and reports are preserved.
Only load trusted model files. Probabilities are not calibrated confidence.

## Professor explanation

“I represented sensor selection as a 15-bit chromosome. A one includes a sensor channel and
a zero excludes it. My fitness rewards four-class macro F1 and subtracts a small input-count
penalty, so the search balances prediction and complexity.

I used tournament selection, one-point crossover, bit-flip mutation and elitism. All fitness
evaluations use purged time blocks inside the original training day. Validation and test
scores are not part of fitness. I ran three seeds to inspect search sensitivity.

The selected chromosome uses {count} inputs. Its validation macro F1 is {sel.macro_f1_all_four:.4f}
compared with {base.macro_f1_all_four:.4f} for all 15 inputs. I report external changes honestly,
because optimizing training fitness does not guarantee improvement on future recordings.
This is channel selection, not measured hardware or energy savings.”

## Viva questions

**Why binary encoding?** Each candidate channel is either included or excluded.
**Why a penalty?** Prediction score alone might keep redundant inputs; the penalty expresses a simplicity preference.
**Why macro F1?** Empty-room rows dominate; all four classes should influence selection.
**Selection versus crossover versus mutation?** Selection favours fit parents, crossover mixes their
choices, and mutation introduces new bit choices.
**What is elitism?** Preserve the best two chromosomes unchanged each generation.
**Why several seeds?** A GA is stochastic. Different seeds can find different subsets.
**Why caching?** A repeated subset has the same deterministic evaluation, so recomputing is unnecessary.
**Can GA guarantee the global optimum?** No, only a fraction of the possible subsets was evaluated.
**Does higher fitness guarantee better test performance?** No. Inner-fold overfitting and time-period
differences can prevent that. Do not retune against the test after seeing the result.
**Why no fuzzy control yet?** This phase concerns sensor selection; control belongs to a later phase.

## Next-phase constraint

Keep the baseline as a reference regardless of this subset's score. Do not use the already
reported later-test score as a new GA objective or claim an independent final confirmation.
Further evidence needs new held-out recordings. GA does not automatically solve the weak
later-period generalization observed in Phase 2.

## Reference

GA operator concepts: https://deap.readthedocs.io/en/master/tutorials/basic/part2.html
This project implements the operators directly for explanation; it does not depend on DEAP.
"""
    (ROOT/"docs/phase3_notes.md").write_text(notes)
    cards="".join(f'<section><h2>{title}</h2><img src="plots/{name}.png" alt="{title}"></section>' for name,title in [
        ("convergence","GA convergence across three seeds"),("subset_tradeoff","Prediction and subset-size trade-off"),
        ("selected_sensors","Selected channels"),("external_comparison","Comparison using unchanged external partitions"),
        ("selected_validation_confusion","Selected-model validation errors"),("selected_test_confusion","Selected-model later-test errors")])
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC Project: Phase 3</title><style>
body{{background:#eef2f5;color:#172b40;font:16px/1.6 system-ui;margin:0}}main{{max-width:1100px;margin:auto;padding:35px 24px}}
section{{background:white;border-radius:8px;padding:24px;margin:24px 0}}img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dde4e8;text-align:left}}
th{{background:#e8f2f1}}.scroll{{overflow:auto}}a{{color:#166e75}}.note{{border-left:4px solid #da9439;padding-left:18px}}
</style><body><main><p>SOFT COMPUTING MINOR PROJECT / PHASE 3</p><h1>Genetic Algorithm sensor selection</h1>
<section><h2>{count} of 15 input channels selected</h2><p>{", ".join(selected)}</p>
<p>Chromosome: <strong>{lock['winner']['chromosome']}</strong>. Three GA runs; {lock['unique_subsets_evaluated']} distinct subsets evaluated.</p>
<p>Validation macro F1 {conclusion}: {base.macro_f1_all_four:.4f} → {sel.macro_f1_all_four:.4f}.
Selected-model accuracy: {sel.accuracy:.2%}.</p>
<p class="note">Later-test balanced accuracy: {test.balanced_accuracy_present_classes:.2%}.
No true class 1 occurs in that test. External comparisons reuse data seen in earlier phases;
they are not independent confirmation. Fewer inputs do not establish hardware or energy savings.</p>
<p><a href="../../docs/phase3_notes.md">Method and viva notes</a> /
<a href="../../docs/phase3_protocol.json">Fixed protocol</a> /
<a href="../../models/phase3/selection_locked.json">Locked selection</a></p></section>
<section><h2>How selection works</h2><p>Fitness = training-only pooled OOF macro F1 − 0.05 × inputs / 15.
Four purged time blocks from December 22 supply the predictions. Tournament selection,
one-point crossover, bit-flip mutation and two elites evolve 16 chromosomes for 10 generations.</p>
<p>The inner blocks allow training on either side of a held-out time block; this evaluates
within-day interpolation, not forecasting. Eighty trees are used during search and 300 for final comparison.</p></section>
{cards}<section><h2>Full metrics</h2><div class="scroll">{metrics.round(4).to_html(index=False,border=0,na_rep="n/a")}</div>
<p>Present-class metrics cover only supported true classes. Training scores are diagnostics.
The January subset overlaps the test.</p></section></main></body></html>"""
    (OUT/"report.html").write_text(page)
    readme=ROOT/"README.md";text=readme.read_text()
    header="## Phase 3 complete: Genetic Algorithm sensor selection"
    block=f"""{header}

Open **outputs/phase3/report.html** for the result and **docs/phase3_notes.md** for the method and viva.
Selected {count}/15 channels: {", ".join(selected)}.
Validation macro F1: {base.macro_f1_all_four:.4f} (baseline) to {sel.macro_f1_all_four:.4f} (selected).
Later-test balanced accuracy: {test.balanced_accuracy_present_classes:.2%}; true class 1 is absent.
Selection uses training-only purged blocks. External comparisons reuse previously seen data.
No measured hardware/energy savings or global-optimum claim is made.

~~~bash
python src/phase3.py
python src/verify_phase3.py
~~~

"""
    if header not in text:
        text=text.replace("## Phase 2 complete: baseline occupancy prediction",block+"## Phase 2 complete: baseline occupancy prediction")
        readme.write_text(text)

if __name__=="__main__":main()
