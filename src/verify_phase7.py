"""Integration checks for saved models, manual scenarios and the local HTTP interface."""
import json, threading, urllib.request, urllib.error, math
from pathlib import Path
from dashboard_server import evaluate,bootstrap,Handler,ThreadingHTTPServer,ROOT
from energy_controller import EnergyRoomSystem
def main():
    checks=0
    data=bootstrap()
    for preset in data["presets"]:
        for model in ("baseline","ga"):
            for policy in ("AUTO","ECO","NORMAL","COMFORT"):
                payload=dict(input_mode="sensors",sensors=preset["sensors"],model=model,policy=policy,duration_minutes=30)
                actual=evaluate(payload)
                expected=EnergyRoomSystem(model,policy).evaluate(preset["sensors"],1800,True)
                assert actual["fuzzy"]==expected["fuzzy"]
                assert actual["energy_request"]==expected["energy_request"]
                assert actual["simulated_energy_kwh"]==expected["simulated_energy_kwh"]
                checks+=1
    cases=[(0,24,0,450,0),(3,32,20,1600,100),(1,24,700,450,10/3),(0,24,0,1600,0)]
    for occ,temp,light,gas,expected_light in cases:
        result=evaluate(dict(input_mode="manual",conditions=dict(occupancy=occ,temperature=temp,light=light,co2=gas)))
        assert math.isclose(result["energy_request"]["lighting_percent"],expected_light,abs_tol=.001)
        if gas==1600: assert result["energy_request"]["ventilation_percent"]==100
        checks+=1
    bad=[{},{"input_mode":"no"}, {"input_mode":"manual","conditions":{}},
         {"duration_minutes":-1},{"duration_minutes":float("nan")},{"duration_minutes":True},
         {"policy":"bad"},{"model":"unknown"},{"sensors":{"S1_Temp":"25"}}]
    for payload in bad:
        try:evaluate(payload)
        except ValueError:checks+=1
        else:raise AssertionError(f"Accepted invalid payload {payload}")
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f"http://127.0.0.1:{server.server_port}"
    def get(path):
        with urllib.request.urlopen(base+path) as r:return r.status,r.read()
    try:
        assert json.loads(get("/api/health")[1])["status"]=="ok"
        assert len(json.loads(get("/api/bootstrap")[1])["presets"])==6
        assert b"Room simulator" in get("/")[1]
        assert b"function render" in get("/app.js")[1]
        checks+=4
        payload=dict(input_mode="sensors",sensors=data["presets"][0]["sensors"])
        req=urllib.request.Request(base+"/api/evaluate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req) as r: assert json.load(r)["fuzzy"]["predicted_occupancy"]==0
        checks+=1
        for path in ["/evidence/../models/phase2/random_forest_baseline.joblib","/../README.md","/unknown"]:
            try:get(path)
            except urllib.error.HTTPError as e:assert e.code==404;checks+=1
            else:raise AssertionError("Unexpected file access")
        for body,headers,status in [(b"[]",{"Content-Type":"application/json"},400),(b'{"x":NaN}',{"Content-Type":"application/json"},400),
             (b"{}",{"Content-Type":"application/json","Origin":"https://example.com"},403)]:
            req=urllib.request.Request(base+"/api/evaluate",data=body,headers=headers)
            try:urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:assert e.code==status;checks+=1
            else:raise AssertionError("Invalid HTTP request accepted")
    finally:server.shutdown();server.server_close();worker.join()
    report={"status":"passed","checks":checks,"recorded_model_policy_cases":48,"manual_scenarios":4,
        "scope":"Pipeline parity, known scenarios, invalid inputs, HTTP routes and local-origin restriction.",
        "limitation":"These checks verify implementation, not model generalization or real-world energy savings."}
    target=ROOT/"outputs/phase7/verification.json";target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
if __name__=="__main__":main()
