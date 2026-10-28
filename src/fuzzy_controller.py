"""Transparent Mamdani inference using NumPy only."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
import numpy as np

def membership(x,points):
    """Trapezoid [a,b,c,d]; b=c is triangular, repeated ends form shoulders."""
    a,b,c,d=map(float,points)
    x=np.asarray(x,dtype=float)
    rising=np.ones_like(x) if a==b else np.clip((x-a)/(b-a),0,1)
    falling=np.ones_like(x) if c==d else np.clip((d-x)/(d-c),0,1)
    return np.where((x<a)|(x>d),0,np.minimum(rising,falling))

class FuzzyRoomController:
    def __init__(self,config_path=None):
        path=Path(config_path) if config_path else ROOT/"config/phase5_fuzzy.json"
        self.config=json.loads(path.read_text())
        step=self.config["output_grid_step"]
        self.universe=np.linspace(0,100,round(100/step)+1)
        self.output_mfs={name:{label:membership(self.universe,p) for label,p in spec["sets"].items()}
                         for name,spec in self.config["outputs"].items()}
    def evaluate(self,occupancy,temperature,light,co2,trace=True,curves=False):
        supplied={"occupancy":occupancy,"temperature":temperature,"light":light,"co2":co2}
        try:supplied={k:float(v) for k,v in supplied.items()}
        except (ValueError,TypeError):raise ValueError("All fuzzy inputs must be numeric.")
        if not all(np.isfinite(v) for v in supplied.values()):raise ValueError("Fuzzy inputs must be finite.")
        if not 0<=supplied["occupancy"]<=3:raise ValueError("Occupancy must be between 0 and 3.")
        if supplied["light"]<0 or supplied["co2"]<0:raise ValueError("Light and CO2 cannot be negative.")
        used={};clipped=[]
        for name,value in supplied.items():
            lo,hi=self.config["inputs"][name]["range"]
            used[name]=float(np.clip(value,lo,hi))
            if used[name]!=value:clipped.append({"input":name,"supplied":value,"used":used[name]})
        memberships={name:{label:float(membership(used[name],points)) for label,points in spec["sets"].items()}
                     for name,spec in self.config["inputs"].items()}
        aggregate={name:np.zeros_like(self.universe) for name in self.output_mfs}
        fired=[]
        for rule in self.config["rules"]:
            strength=min(memberships[var][label] for var,label in rule["if"].items())
            if strength>0:
                curve=np.minimum(strength,self.output_mfs[rule["output"]][rule["then"]])
                aggregate[rule["output"]]=np.maximum(aggregate[rule["output"]],curve)
                fired.append({**rule,"strength":strength})
        centroids={}
        for output,curve in aggregate.items():
            area=np.trapezoid(curve,self.universe)
            if area<=0:raise ValueError(f"No rule coverage for {output}.")
            centroids[output]=float(np.trapezoid(curve*self.universe,self.universe)/area)
        # A fuzzy 'off' triangle has a small positive centroid, so exact off is explicit.
        exact_off=used["occupancy"]==0
        result={"lighting_percent":0.0 if exact_off else centroids["lighting"],
                "ventilation_percent":centroids["ventilation"],
                "mamdani_centroids":centroids,"lighting_exact_off_applied":exact_off,
                "input_clipping":clipped,"inputs_used":used}
        if trace:result.update({"memberships":memberships,"fired_rules":fired})
        if curves:result["curves"]={"universe":self.universe.tolist(),**{k:v.tolist() for k,v in aggregate.items()}}
        return result

class SmartRoomSystem:
    """Estimate occupancy, then feed explicit environmental measurements to the controller."""
    def __init__(self,model_name="baseline"):
        from occupancy_system import OccupancySystem
        self.estimator=OccupancySystem(model_name)
        self.controller=FuzzyRoomController()
    def evaluate(self,sensors,trace=True):
        import pandas as pd
        frame=pd.DataFrame([sensors])
        prediction=self.estimator.predict(frame).iloc[0]
        mapping=self.controller.config["mapping"]
        missing=set(mapping.values())-set(sensors)
        if missing:raise ValueError(f"Missing controller measurements: {sorted(missing)}")
        conditions={key:sensors[value] for key,value in mapping.items()}
        result=self.controller.evaluate(int(prediction.predicted_occupancy),**conditions,trace=trace)
        result.update({"occupancy_model":self.estimator.model_name,
                       "predicted_occupancy":int(prediction.predicted_occupancy),
                       "occupancy_probabilities":{str(i):float(prediction[f"probability_{i}"]) for i in range(4)},
                       "controller_measurement_sources":mapping})
        return result
