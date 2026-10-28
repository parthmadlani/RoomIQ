"""CSV -> occupancy prediction -> fuzzy lighting/ventilation requests. No actuation."""
from pathlib import Path
import argparse,json
from fuzzy_controller import SmartRoomSystem
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=["baseline","ga"],default="baseline")
    parser.add_argument("--trace",type=Path,help="Optional full rule traces as JSON")
    args=parser.parse_args()
    paths=[args.output]+([args.trace] if args.trace else [])
    if len({p.resolve() for p in paths})!=len(paths):raise ValueError("Output and trace paths must differ.")
    for path in paths:
        if path.exists() or path.resolve()==args.input.resolve():raise FileExistsError("Use new output paths.")
    df=pd.read_csv(args.input,float_precision="round_trip")
    if df.empty:raise ValueError("Input CSV is empty.")
    system=SmartRoomSystem(args.model)
    traces=[system.evaluate(row,trace=True) for row in df.to_dict("records")]
    rows=[{"input_csv_row":i+2,"occupancy_model":args.model,
           "predicted_occupancy":r["predicted_occupancy"],
           "lighting_percent":r["lighting_percent"],"ventilation_percent":r["ventilation_percent"],
           "raw_lighting_centroid":r["mamdani_centroids"]["lighting"],
           "exact_off_applied":r["lighting_exact_off_applied"],
           "clipped_input_count":len(r["input_clipping"])} for i,r in enumerate(traces)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)
    if args.trace:
        args.trace.parent.mkdir(parents=True,exist_ok=True)
        args.trace.write_text(json.dumps(traces,indent=2)+"\n")
    print(f"Saved {len(rows)} simulated control requests.")

if __name__=="__main__":main()
