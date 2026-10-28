"""Phase 7 local dashboard. Serves saved evidence and calls the real Python pipeline."""
from pathlib import Path
import sys, json, csv, math, argparse, webbrowser, mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, unquote
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from energy_controller import EnergyRoomSystem, EnergyDecisionLayer
from fuzzy_controller import FuzzyRoomController

def rows(path):
    with (ROOT/path).open() as f: return list(csv.DictReader(f))

def bootstrap():
    registry=json.loads((ROOT/"models/phase4/model_registry.json").read_text())
    presets=rows("data/examples/phase5_baseline_controller.csv")
    provenance=rows("data/examples/phase5_example_provenance.csv")
    return dict(registry=registry, presets=[dict(sensors={k:float(v) for k,v in r.items()},provenance=p) for r,p in zip(presets,provenance)],
        comparison=rows("outputs/phase4/tables/professor_comparison.csv"),
        convergence=rows("outputs/phase3/tables/convergence.csv"),
        energy=rows("outputs/phase6/tables/energy_summary.csv"),
        fuzzy=json.loads((ROOT/"config/phase5_fuzzy.json").read_text()),
        assumptions=json.loads((ROOT/"config/phase6_energy.json").read_text()))

def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(name+" must be a finite number.")
    return float(value)

def evaluate(payload):
    if not isinstance(payload,dict): raise ValueError("Expected a JSON object.")
    mode=payload.get("input_mode","sensors")
    policy=payload.get("policy","AUTO")
    if policy not in ("AUTO","ECO","NORMAL","COMFORT"): raise ValueError("Unknown energy policy.")
    seconds=number(payload.get("duration_minutes",30),"Duration")*60
    if not 0<seconds<=86400: raise ValueError("Duration must be greater than 0 and at most 1440 minutes.")
    if mode=="sensors":
        model=payload.get("model","baseline")
        if model not in ("baseline","ga"): raise ValueError("Unknown occupancy model.")
        sensors=payload.get("sensors")
        if not isinstance(sensors,dict): raise ValueError("Supply a sensor object.")
        sensors={k:number(v,k) for k,v in sensors.items()}
        result=EnergyRoomSystem(model,policy).evaluate(sensors,seconds,trace=True)
    elif mode=="manual":
        values=payload.get("conditions")
        if not isinstance(values,dict) or set(values)!={"occupancy","temperature","light","co2"}:
            raise ValueError("Manual scenarios need occupancy, temperature, light and co2.")
        values={k:number(v,k) for k,v in values.items()}
        fuzzy=FuzzyRoomController().evaluate(**values,trace=True)
        used=fuzzy["inputs_used"]
        layer=EnergyDecisionLayer()
        request=layer.apply(fuzzy["lighting_percent"],fuzzy["ventilation_percent"],used["occupancy"],used["temperature"],used["co2"],policy)
        result=dict(fuzzy=fuzzy,energy_request=request,simulation_only=True,duration_seconds=seconds,
                    simulated_energy_kwh=layer.energy_kwh(request["total_w"],seconds),
                    full_power_reference_kwh=layer.energy_kwh(request["reference_full_power_w"],seconds))
    else: raise ValueError("Unknown input mode.")
    result["input_mode"]=mode
    result["duration_assumption"]="Constant conditions for the chosen hypothetical duration."
    return result

class Handler(BaseHTTPRequestHandler):
    def reply(self,status,body,content_type="application/json"):
        if not isinstance(body,bytes): body=json.dumps(body,allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=="/api/bootstrap": return self.reply(200,bootstrap())
        if path=="/api/health": return self.reply(200,{"status":"ok","simulation_only":True})
        if path=="/": file=ROOT/"dashboard/index.html"
        elif path in ("/app.js","/style.css"): file=ROOT/"dashboard"/path[1:]
        elif path.startswith("/evidence/"):
            file=(ROOT/"outputs"/path[len("/evidence/"):]).resolve()
            if not file.is_relative_to((ROOT/"outputs").resolve()) or file.suffix not in (".png",".svg",".html",".csv",".pptx"):
                return self.reply(404,{"error":"Not found"})
        else: return self.reply(404,{"error":"Not found"})
        if not file.is_file(): return self.reply(404,{"error":"Not found"})
        self.reply(200,file.read_bytes(),mimetypes.guess_type(str(file))[0] or "application/octet-stream")
    def do_POST(self):
        if self.path!="/api/evaluate": return self.reply(404,{"error":"Not found"})
        origin=self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}",f"http://localhost:{self.server.server_port}"):
            return self.reply(403,{"error":"Use this local dashboard to submit requests."})
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=32768: raise ValueError("Invalid request size.")
            if self.headers.get("Content-Type","").split(";")[0]!="application/json": raise ValueError("Send JSON.")
            payload=json.loads(self.rfile.read(length))
            result=evaluate(payload)
        except (ValueError,TypeError,KeyError) as e: return self.reply(400,{"error":str(e)})
        except Exception:
            import traceback
            traceback.print_exc()
            return self.reply(500,{"error":"The saved pipeline could not run. See the local terminal."})
        self.reply(200,result)
    def log_message(self,*args): pass

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--open",action="store_true")
    args=parser.parse_args()
    # Check the models before opening the UI; never retrain from a web request.
    for model in ("baseline","ga"): EnergyRoomSystem(model)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    url=f"http://127.0.0.1:{server.server_port}"
    print(f"RoomIQ dashboard: {url}\nSimulation only. Press Ctrl+C to stop.",flush=True)
    if args.open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
if __name__=="__main__": main()
