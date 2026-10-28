"use strict";
let data,lastResult,edited=false;
const $=id=>document.getElementById(id);
const fmt=(v,n=3)=>Number(v).toFixed(n);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const table=(heads,rows)=>'<table><thead><tr>'+heads.map(x=>'<th>'+escape(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+escape(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
const manual={empty:{occupancy:0,temperature:24,light:0,co2:450},high:{occupancy:3,temperature:32,light:20,co2:1600},bright:{occupancy:1,temperature:24,light:700,co2:450},"empty-co2":{occupancy:0,temperature:24,light:0,co2:1600}};
function clearResult(){lastResult=null;$("result").hidden=true;$("result-empty").hidden=false;$("error").textContent=""}
function fields(){
 const isManual=$("input-mode").value==="manual";
 $("sensor-options").hidden=isManual;$("manual-options").hidden=!isManual;
 const preset=data.presets[Number($("preset").value)||0];
 const values=isManual?manual[$("scenario").value]:preset.sensors;
 const names=isManual?Object.keys(values):[...new Set([...data.registry.models[$("model").value].features,"S1_Temp"])];
 $("inputs").innerHTML=names.map(name=>{const label=name==="temperature"||name.includes("Temp")?name+" (°C)":name==="co2"||name==="S5_CO2"?name+" (ppm)":name==="light"||name.includes("Light")?name+" (lux)":name.includes("Sound")?name+" (V)":name;
 return '<label>'+escape(label)+'<input data-field="'+name+'" type="number" step="'+(name.includes("PIR")?"1":"any")+'" '+(name.includes("PIR")?'min="0" max="1"':name==="occupancy"?'min="0" max="3"':name==="temperature"||name.includes("Temp")?"":'min="0"')+' value="'+values[name]+'" required></label>'}).join('');
 $("provenance").textContent=Object.entries(preset.provenance).map(([k,v])=>k+": "+v).join(" · ");
 edited=false;clearResult();
}
function plot(series,domain,yLabel){
 const W=650,H=240,L=48,R=20,T=20,B=38;
 const max=Math.max(1,...series.flatMap(s=>s.points.map(p=>p[1])));
 const x=v=>L+(v-domain[0])/(domain[1]-domain[0])*(W-L-R),y=v=>H-B-v/max*(H-B-T);
 return '<svg role="img" aria-label="'+escape(yLabel)+'" viewBox="0 0 '+W+' '+H+'"><line x1="'+L+'" x2="'+(W-R)+'" y1="'+(H-B)+'" y2="'+(H-B)+'" stroke="#b7c9bc"/>'+[0,.5,1].map(v=>'<text x="3" y="'+(y(v*max)+4)+'" font-size="12" fill="#61786c">'+fmt(v*max,2)+'</text>').join('')+series.map((s,i)=>'<polyline fill="none" stroke="'+["#14765d","#ce8b32","#527fa1","#8a629a"][i%4]+'" stroke-width="2.5" points="'+s.points.map(p=>x(p[0])+","+y(p[1])).join(" ")+'"/>').join('')+'<text x="'+L+'" y="'+(H-10)+'" font-size="12">'+domain[0]+'</text><text x="'+(W-R-30)+'" y="'+(H-10)+'" font-size="12">'+domain[1]+'</text></svg><p class="small">'+series.map((s,i)=>'<span style="color:'+["#14765d","#ce8b32","#527fa1","#8a629a"][i%4]+'">● '+escape(s.name)+'</span>').join(' &nbsp; ')+'</p>';
}
function evidence(){
 $("comparison").innerHTML=table(["Model","Estimator inputs","Validation macro F1","Test accuracy","Test balanced accuracy"],data.comparison.map(r=>[r.model,r.input_channels,fmt(r.validation_macro_f1,4),fmt(r.later_test_accuracy_percent,2)+"%",fmt(r.later_test_balanced_accuracy_percent,2)+"%"]));
 $("selected").innerHTML=data.registry.models.ga.features.map(x=>'<span class="pill">'+x+'</span>').join('');
 $("ga-chart").innerHTML=plot([42,43,44].map(seed=>({name:"Seed "+seed,points:data.convergence.filter(r=>Number(r.seed)===seed).map(r=>[Number(r.generation),Number(r.fitness)])})),[0,10],"Best penalized fitness by generation");
 $("memberships").innerHTML=Object.entries(data.fuzzy.inputs).map(([name,spec])=>'<div><h3>'+escape(name)+'</h3>'+plot(Object.entries(spec.sets).map(([term,p])=>({name:term,points:[...new Set([spec.range[0],...p,spec.range[1]])].sort((a,b)=>a-b).map(x=>[x,x<p[0]||x>p[3]?0:Math.min(p[0]===p[1]?1:Math.max(0,Math.min(1,(x-p[0])/(p[1]-p[0]))),p[2]===p[3]?1:Math.max(0,Math.min(1,(p[3]-x)/(p[3]-p[2]))))])})),spec.range,name+" membership degrees")+'</div>').join('');
 $("rules").innerHTML=table(["Rule","IF (AND)","THEN"],data.fuzzy.rules.map(r=>[r.id,Object.entries(r.if).map(([k,v])=>k+" is "+v).join(" AND "),r.output+" is "+r.then]));
 energyTable();
}
function energyTable(){
 const list=data.energy.filter(r=>r.partition===$("partition").value&&r.policy===$("energy-policy").value);
 $("energy-table").innerHTML=table(["Source","Hours covered","Simulated kWh","Full-power kWh","Reduction","Occupied zero-light hours"],list.map(r=>[r.source,fmt(r.covered_hours,2),fmt(r.simulated_kwh),fmt(r.full_power_reference_kwh),fmt(r.simulated_reduction_percent,2)+"%",fmt(r.occupied_zero_lighting_hours,2)]))+'<p class="small">Compare over '+fmt(list[0]?.occupied_hours,2)+' occupied hours. Counterfactual reference uses true labels, not predictions.</p>';
}
function labelCheck(r,known){
 // The recorded label is provenance only: it is compared after prediction, never sent to the model.
 if(r.input_mode!=="sensors"||known===null)return '';
 const predicted=r.fuzzy.predicted_occupancy;
 if(known===predicted)return '<p class="small">Recorded label: '+known+' people, matching the prediction. The label is not a model input.</p>';
 return '<div class="notice"><strong>Prediction error: recorded label '+known+' people, predicted '+predicted+'.</strong> '+(predicted===0?'This is a false-empty case, so lighting is switched off in an occupied room.':'The controller acts on the predicted occupancy, not the recorded label.')+'</div>';
}
function render(r,known=null){
 const f=r.fuzzy,e=r.energy_request;
 const occ=r.input_mode==="manual"?f.inputs_used.occupancy:f.predicted_occupancy;
 $("result").innerHTML='<div class="panel"><span class="badge">'+escape(e.mode)+' · '+escape(e.reason)+'</span>'+labelCheck(r,known)+'<div class="metrics"><div><p class="small">'+(r.input_mode==="manual"?"Supplied occupancy":"Predicted occupancy")+'</p><div class="metric">'+occ+' <small>people</small></div></div><div><p class="small">Simulated load</p><div class="metric">'+fmt(e.total_w,1)+' <small>W</small></div></div><div><p class="small">Lighting request</p><div class="metric">'+fmt(e.lighting_percent,1)+'<small>%</small></div></div><div><p class="small">Ventilation request</p><div class="metric">'+fmt(e.ventilation_percent,1)+'<small>%</small></div></div></div><hr><p><strong>'+fmt(r.simulated_energy_kwh,4)+' kWh</strong> over '+fmt(r.duration_seconds/60,1)+' minutes, compared with '+fmt(r.full_power_reference_kwh,4)+' kWh at full power.</p><p class="small">Assumed constant conditions. Actual use and comfort are unmeasured.</p>'+
(f.occupancy_probabilities?'<details><summary>Class probabilities (uncalibrated)</summary>'+table(["People","Model probability"],Object.entries(f.occupancy_probabilities).map(([k,v])=>[k,fmt(v*100,2)+"%"]))+'</details>':'<p class="small">Manual scenario: the occupancy estimator was bypassed.</p>')+
'<details open><summary>How the request was formed</summary>'+table(["Stage","Lighting %","Ventilation %"],[["Mamdani centroid",fmt(f.mamdani_centroids.lighting,2),fmt(f.mamdani_centroids.ventilation,2)],["After occupancy gate",fmt(f.lighting_percent,2),fmt(f.ventilation_percent,2)],["After "+e.mode,fmt(e.lighting_percent,2),fmt(e.ventilation_percent,2)]])+'<p class="small">Exact-off gate: '+f.lighting_exact_off_applied+'. CO₂ ventilation floor applied: '+e.ventilation_guard_applied+'.</p></details>'+
'<details><summary>Fired fuzzy rules ('+f.fired_rules.length+')</summary>'+table(["Rule","Antecedents → consequent","Strength"],f.fired_rules.map(x=>[x.id,Object.entries(x.if).map(([k,v])=>k+"="+v).join(", ")+" → "+x.output+"="+x.then,fmt(x.strength)]))+'</details>'+
'<details><summary>Membership degrees & input clipping</summary><pre>'+escape(JSON.stringify({memberships:f.memberships,input_clipping:f.input_clipping,inputs_used:f.inputs_used},null,2))+'</pre></details><button type="button" id="download" class="secondary">Save this result as JSON</button></div>';
 $("result").hidden=false;$("result-empty").hidden=true;
 const top=$("result").getBoundingClientRect().top;
 if(top<0||top>innerHeight-120)$("result").scrollIntoView({behavior:"smooth",block:"start"});
 $("download").onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(lastResult,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="roomiq-simulation.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
}
document.querySelectorAll("[data-tab]").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tab").forEach(t=>t.hidden=t.id!==b.dataset.tab);document.querySelectorAll("[data-tab]").forEach(x=>x.classList.toggle("active",x===b));$("page-title").textContent=b.textContent.replace(/^\d+\s*/,"").trim();scrollTo(0,0)});
$("simulation").onsubmit=async event=>{event.preventDefault();$("error").textContent="";$("run").disabled=true;const values=Object.fromEntries([...document.querySelectorAll("[data-field]")].map(x=>[x.dataset.field,Number(x.value)]));const payload={input_mode:$("input-mode").value,model:$("model").value,policy:$("policy").value,duration_minutes:Number($("duration").value),sensors:values,conditions:values};const known=edited?null:Number(data.presets[Number($("preset").value)||0].provenance.Room_Occupancy_Count);try{const response=await fetch("/api/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const r=await response.json();if(!response.ok)throw new Error(r.error);lastResult={request:payload,...r};render(r,known)}catch(e){clearResult();$("error").textContent=e.message}finally{$("run").disabled=false}};
["input-mode","model","preset","scenario"].forEach(id=>$(id).onchange=fields);
["partition","energy-policy"].forEach(id=>$(id).onchange=energyTable);
$("simulation").addEventListener("input",event=>{if(event.target.dataset.field)edited=true;clearResult()});
fetch("/api/bootstrap").then(r=>{if(!r.ok)throw Error("Cannot load saved evidence.");return r.json()}).then(d=>{data=d;$("preset").innerHTML=d.presets.map((r,i)=>'<option value="'+i+'">'+(i<4?"Validation · actual class "+i:"Later test · false-empty example "+(i-3))+'</option>').join('');fields();evidence()}).catch(e=>{$("error").textContent=e.message;$("run").disabled=true});
