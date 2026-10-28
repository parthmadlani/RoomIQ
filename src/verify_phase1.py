"""Independent standard-library verification of the Phase 1 outputs."""
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def main():
    raw_dir = ROOT / "data/raw"
    manifest = json.loads((raw_dir / "source_manifest.json").read_text())
    for name, info in manifest["files"].items():
        path = raw_dir / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], name
        assert path.stat().st_size == info["bytes"], name
    with zipfile.ZipFile(raw_dir / "room_occupancy_estimation.zip") as z:
        assert z.read("Occupancy_Estimation.csv") == (raw_dir / "Occupancy_Estimation.csv").read_bytes()
    raw = read_csv(raw_dir / "Occupancy_Estimation.csv")
    clean = read_csv(ROOT / "data/processed/occupancy_cleaned.csv")
    baseline = read_csv(ROOT / "data/processed/baseline_features.csv")
    compact = read_csv(ROOT / "data/processed/compact_features.csv")
    spec = json.loads((ROOT / "docs/feature_spec.json").read_text())
    target = spec["target"]
    assert len(raw) == len(clean) == len(baseline) == len(compact) == 10129
    assert len(raw[0]) == 19 and len(clean[0]) == 26
    assert list(baseline[0]) == spec["recommended_baseline"] + [target]
    assert list(compact[0]) == spec["optional_compact_comparison"] + [target]
    assert len(spec["recommended_baseline"]) == 15
    assert len(spec["optional_compact_comparison"]) == 6
    assert not set(spec["recommended_baseline"]) & set(spec["excluded_metadata"] + [target, "S5_CO2_Slope"])
    assert len({r["Source_Row"] for r in clean}) == len(raw)
    for i, row in enumerate(clean):
        original = raw[int(row["Source_Row"]) - 2]
        for col in original:
            if col in ["Date", "Time"]:
                assert row[col] == original[col]
            else:
                assert float(row[col]) == float(original[col]), (i, col)
        for family in ["Temp", "Light", "Sound"]:
            mean = math.fsum(float(original[f"S{k}_{family}"]) for k in range(1,5))/4
            assert math.isclose(float(row["Avg_"+family]), mean, abs_tol=1e-12)
        assert int(row["PIR_Any"]) == int(bool(int(original["S6_PIR"]) or int(original["S7_PIR"])))
        for table in (baseline,compact):
            for col,value in table[i].items():
                assert float(value) == float(row[col]), (i,col)
    timestamps = [r["Timestamp"] for r in clean]
    assert timestamps == sorted(timestamps) and len(set(timestamps)) == len(timestamps)
    counts = Counter(int(r[target]) for r in raw)
    assert counts == {0: 8228, 1: 459, 2: 748, 3: 694}
    distribution = read_csv(ROOT / "outputs/tables/occupancy_distribution.csv")
    assert {int(r["occupancy"]):int(r["rows"]) for r in distribution} == dict(counts)
    assert math.isclose(sum(float(r["percent"]) for r in distribution), 100, abs_tol=1e-10)
    profile = read_csv(ROOT / "outputs/tables/column_profile.csv")
    assert len(profile) == 19 and {r["column"] for r in profile} == set(raw[0])
    assert all(int(r["missing"]) == 0 for r in profile)
    assert len(list((ROOT / "outputs/plots").glob("*.png"))) == 10
    assert len(list((ROOT / "outputs/plots").glob("*.svg"))) == 10
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("src", "href") and not value.startswith(("https:", "http:", "#")):
                    assert (ROOT / "outputs" / value).exists(), value
    Links().feed((ROOT / "outputs/phase1_report.html").read_text())
    result = {"status":"passed","verified":[
        "Raw files match the source manifest and archive bytes",
        "All 10129 records and original sensor values preserved",
        "All aggregates independently recalculated",
        "Model tables match the audit table and explicit input lists exclude the target",
        "Class counts, schema, column profiles and timestamp order verified",
        "Ten PNG and ten SVG plots exist; report links resolve"]}
    (ROOT / "outputs/tables/verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
