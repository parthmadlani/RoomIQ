"""Phase 6 mode adjustment and transparent linear energy accounting."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))

class EnergyDecisionLayer:
    def __init__(self,config_path=None):
        self.config=json.loads((Path(config_path) if config_path else ROOT/"config/phase6_energy.json").read_text())
        for value in self.config["loads"].values():
            if not math.isfinite(value) or value<=0:raise ValueError("Nominal load powers must be finite and positive.")
        for mode in self.config["modes"].values():
            if any(not math.isfinite(v) or v<0 for v in mode.values()):raise ValueError("Invalid mode scale.")
    def apply(self,lighting_percent,ventilation_percent,occupancy,temperature,co2,policy="AUTO"):
        values=list(map(float,[lighting_percent,ventilation_percent,occupancy,temperature,co2]))
        if not all(math.isfinite(v) for v in values):raise ValueError("Inputs must be finite.")
        light,vent,occ,temp,gas=values
        if not(0<=light<=100 and 0<=vent<=100 and 0<=occ<=3 and gas>=0):
            raise ValueError("Requests must be in [0,100], occupancy in [0,3], CO2 nonnegative.")
        if policy not in ["ECO","NORMAL","COMFORT","AUTO"]:raise ValueError("Unknown policy.")
        thresholds=self.config["auto_policy"]
        if policy=="AUTO":
            if gas>=thresholds["co2_comfort_threshold_ppm"]:mode,reason="COMFORT","elevated CO2"
            elif occ>=thresholds["high_occupancy_threshold"]:mode,reason="COMFORT","high predicted occupancy"
            elif occ>0 and temp>=thresholds["occupied_hot_threshold_c"]:mode,reason="COMFORT","occupied and hot"
            elif occ>0:mode,reason="NORMAL","lower occupied demand"
            else:mode,reason="ECO","predicted empty without elevated CO2"
        else:mode,reason=policy,"explicit fixed mode"
        scale=self.config["modes"][mode]
        applied_light=min(100.,light*scale["lighting_scale"])
        proposed_vent=min(100.,vent*scale["ventilation_scale"])
        guard=gas>=thresholds["co2_comfort_threshold_ppm"] and proposed_vent<vent
        applied_vent=max(proposed_vent,vent) if gas>=thresholds["co2_comfort_threshold_ppm"] else proposed_vent
        # Preserve Phase 5's explicit occupancy-zero lighting gate.
        if occ==0:applied_light=0.
        lighting_w=self.config["loads"]["lighting_max_w"]*applied_light/100
        ventilation_w=self.config["loads"]["ventilation_max_w"]*applied_vent/100
        full=sum(self.config["loads"].values())
        return {"policy":policy,"mode":mode,"reason":reason,"lighting_percent":applied_light,
                "ventilation_percent":applied_vent,"ventilation_guard_applied":guard,
                "lighting_w":lighting_w,"ventilation_w":ventilation_w,"total_w":lighting_w+ventilation_w,
                "reference_full_power_w":full}
    @staticmethod
    def energy_kwh(power_w,duration_seconds):
        power=float(power_w);seconds=float(duration_seconds)
        if not math.isfinite(power) or not math.isfinite(seconds) or power<0 or seconds<0:
            raise ValueError("Power and duration must be finite and nonnegative.")
        return power*seconds/3_600_000

class EnergyRoomSystem:
    def __init__(self,model_name="baseline",policy="AUTO"):
        from fuzzy_controller import SmartRoomSystem
        if policy not in ["ECO","NORMAL","COMFORT","AUTO"]:raise ValueError("Unknown policy.")
        self.room=SmartRoomSystem(model_name);self.layer=EnergyDecisionLayer();self.policy=policy
    def evaluate(self,sensors,duration_seconds=None,trace=True):
        fuzzy=self.room.evaluate(sensors,trace=trace)
        # Conditions used by the fuzzy system include disclosed environmental clipping.
        values=fuzzy["inputs_used"]
        result=self.layer.apply(fuzzy["lighting_percent"],fuzzy["ventilation_percent"],
                                fuzzy["predicted_occupancy"],values["temperature"],values["co2"],self.policy)
        answer={"fuzzy":fuzzy,"energy_request":result,"simulation_only":True}
        if duration_seconds is not None:
            answer["duration_seconds"]=float(duration_seconds)
            answer["simulated_energy_kwh"]=self.layer.energy_kwh(result["total_w"],duration_seconds)
            answer["full_power_reference_kwh"]=self.layer.energy_kwh(result["reference_full_power_w"],duration_seconds)
        return answer
