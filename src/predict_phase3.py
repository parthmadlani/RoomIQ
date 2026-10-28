"""Predict with the saved Phase 3 model. Extra CSV columns are ignored."""
from pathlib import Path
import sys, argparse, json
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists(): sys.path.append(str(ROOT/"work/python-packages"))
import pandas as pd
import numpy as np
import joblib

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve(): raise ValueError("Input and output must differ.")
    if args.output.exists(): raise FileExistsError("Choose a new output filename.")
    meta=json.loads((ROOT/"models/phase3/model_metadata.json").read_text())
    df=pd.read_csv(args.input,float_precision="round_trip")
    missing=set(meta["features"])-set(df)
    if missing: raise ValueError(f"Missing required sensors: {sorted(missing)}")
    X=df[meta["features"]].apply(pd.to_numeric,errors="raise")
    if len(X)==0 or not np.isfinite(X.to_numpy()).all(): raise ValueError("Need finite, nonempty sensor rows.")
    for col in [c for c in ["S6_PIR","S7_PIR"] if c in X.columns]:
        if not X[col].isin([0,1]).all(): raise ValueError(f"{col} must be binary.")
    model=joblib.load(ROOT/"models/phase3/ga_selected_random_forest.joblib")
    result=pd.DataFrame({"input_csv_row":np.arange(2,len(df)+2),"predicted_occupancy":model.predict(X)})
    probability=model.predict_proba(X)
    for i,cls in enumerate(model.classes_): result[f"probability_{cls}"]=probability[:,i]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f"Saved {len(result)} predictions to {args.output}")

if __name__=="__main__": main()
