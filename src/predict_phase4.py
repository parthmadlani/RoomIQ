"""Run an explicit baseline or GA occupancy model on a CSV."""
from pathlib import Path
import argparse
from occupancy_system import OccupancySystem
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",choices=["baseline","ga"],default="baseline")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("Output already exists; choose a new filename.")
    if args.input.resolve()==args.output.resolve():raise ValueError("Do not overwrite input data.")
    data=pd.read_csv(args.input,float_precision="round_trip")
    result=OccupancySystem(args.model).predict(data)
    result.insert(0,"input_csv_row",range(2,len(result)+2))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f"Saved {len(result)} {args.model} predictions to {args.output}")

if __name__=="__main__":main()
