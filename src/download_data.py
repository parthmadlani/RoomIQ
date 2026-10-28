"""Preserve the original UCI archive and CSV, checking hashes on every rerun."""
from pathlib import Path
import datetime
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
URL = "https://archive.ics.uci.edu/static/public/864/room%2Boccupancy%2Bestimation.zip"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "room_occupancy_estimation.zip"
    csv_path = RAW / "Occupancy_Estimation.csv"
    manifest_path = RAW / "source_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        for name, info in previous["files"].items():
            path = RAW / name
            if path.exists() and sha256(path) != info["sha256"]:
                raise ValueError(f"Raw file changed: {name}. Restore before proceeding.")
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        if previous and hashlib.sha256(data).hexdigest() != previous["files"][archive.name]["sha256"]:
            raise ValueError("Downloaded version differs from the original manifest.")
        archive.write_bytes(data)
    with zipfile.ZipFile(archive) as zipped:
        data = zipped.read("Occupancy_Estimation.csv")
    if csv_path.exists():
        if csv_path.read_bytes() != data:
            raise ValueError("CSV differs from ZIP; refusing overwrite.")
    else:
        csv_path.write_bytes(data)
    if not previous:
        manifest = {
            "source_url": URL,
            "dataset_page": "https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation",
            "doi": "10.24432/C5P605", "license": "CC BY 4.0",
            "retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in (archive, csv_path)}
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Original archive and CSV verified.")

if __name__ == "__main__":
    main()
