"""Explicit-duration energy demonstration; no inference of gaps or equipment control."""
from pathlib import Path
import argparse,json
from energy_controller import EnergyRoomSystem
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",choices=["baseline","ga"],default="baseline")
    p.add_argument("--policy",choices=["ECO","NORMAL","COMFORT","AUTO"],default="AUTO")
    p.add_argument("--seconds-per-row",type=float,required=True,help="Explicit hypothetical duration for every input row; not dataset timeline integration")
    args=p.parse_args()
    if args.output.exists() or args.output.resolve()==args.input.resolve():raise FileExistsError("Choose a new output path.")
    df=pd.read_csv(args.input,float_precision="round_trip")
    if df.empty:raise ValueError("Input has no observations.")
    system=EnergyRoomSystem(args.model,args.policy)
    results=[]
    for i,sensors in enumerate(df.to_dict("records")):
        r=system.evaluate(sensors,args.seconds_per_row,trace=False)
        results.append({"input_csv_row":i+2,"predicted_occupancy":r["fuzzy"]["predicted_occupancy"],
                        **r["energy_request"],"duration_seconds":r["duration_seconds"],
                        "simulated_energy_kwh":r["simulated_energy_kwh"],
                        "reference_energy_kwh":r["full_power_reference_kwh"]})
    args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_csv(args.output,index=False)
    print("Saved hypothetical per-row simulation; source timestamp gaps were not integrated.")

if __name__=="__main__":main()
