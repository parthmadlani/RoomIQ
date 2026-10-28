"""Common, explicit baseline/GA prediction interface for later project phases."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/"work/python-packages").exists():sys.path.append(str(ROOT/"work/python-packages"))
import numpy as np
import pandas as pd
import joblib

class OccupancySystem:
    def __init__(self,model_name="baseline"):
        registry=json.loads((ROOT/"models/phase4/model_registry.json").read_text())
        if model_name not in registry["models"]:
            raise ValueError("model_name must be 'baseline' or 'ga'")
        self.model_name=model_name
        self.spec=registry["models"][model_name]
        self.features=self.spec["features"]
        path=ROOT/self.spec["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.spec["sha256"]:
            raise ValueError("Model checksum mismatch; regenerate the verified model registry.")
        # Load only project-generated, trusted model artifacts.
        self.model=joblib.load(path)
        if self.model.feature_names_in_.tolist()!=self.features:
            raise ValueError("Model feature order does not match registry.")
    def predict(self,rows):
        frame=pd.DataFrame([rows]) if isinstance(rows,dict) else pd.DataFrame(rows)
        missing=set(self.features)-set(frame.columns)
        if missing:raise ValueError(f"Missing required sensors: {sorted(missing)}")
        X=frame[self.features].apply(pd.to_numeric,errors="raise")
        if X.empty or not np.isfinite(X.to_numpy()).all():
            raise ValueError("Provide nonempty finite sensor measurements.")
        for col in self.features:
            if col.endswith("_PIR") and not X[col].isin([0,1]).all():
                raise ValueError(f"{col} must contain 0 or 1.")
            if col.endswith(("_Light","_Sound")) or col=="S5_CO2":
                if X[col].lt(0).any():raise ValueError(f"{col} cannot be negative.")
        out=pd.DataFrame({"model":self.model_name,"predicted_occupancy":self.model.predict(X)},index=frame.index)
        probs=self.model.predict_proba(X)
        for i,cls in enumerate(self.model.classes_):out[f"probability_{cls}"]=probs[:,i]
        return out
