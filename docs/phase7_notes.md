# Phase 7 — Final dashboard, integration and presentation

## What was added
A local browser dashboard calls the existing Python occupancy, fuzzy and energy pipeline.
It uses the standard-library HTTP server and plain HTML/CSS/JavaScript so no new application
framework is required. This is the local implementation of the planned dashboard, rather
than a hosted service. No model training, deployment or device actuation occurs.

Five views cover:
- Room simulator: six recorded examples, editable sensor fields, baseline/GA choice,
  policy choice, hypothetical duration and JSON download.
- Models & GA: saved comparison metrics, selected channels and all three convergence runs.
- Fuzzy rulebook: membership diagrams generated from the saved configuration and all 34 rules.
- Energy evidence: partition/policy filters over saved interval replay summaries, including
  occupied zero-light hours beside apparent reduction.
- Viva walkthrough: teaching sequence and downloadable editable presentation.

The baseline needs 15 estimator channels. The GA needs four estimator channels plus
S1_Temp for the controller. Extra example fields are not presented as GA estimator inputs.
Recorded labels and timestamps are provenance only, not classifier inputs.

## Two distinct demo modes
Sensor mode predicts occupancy using the selected saved model. Manual mode supplies occupancy
directly to fuzzy inference; it explicitly bypasses the classifier. A scenario's occupancy
can be fractional to explore fuzzy overlap, although the trained classifier predicts integers.
Model probabilities are uncalibrated, not a guarantee of confidence.
Changing a field clears the prior displayed result.

The duration in the simulator assumes constant conditions for 0–1440 minutes (strictly positive).
It is unrelated to replay gap handling. Replay evidence uses the Phase 6 audited intervals.
The UI shows raw centroids, the occupancy-zero gate and mode-adjusted requests separately.
Environmental clipping is visible in the expanded trace.

## Teaching cases (AUTO, 30 minutes)
| Scenario | Inputs: occupancy, temperature, light, CO2 | Final lighting | Final ventilation |
|---|---|---:|---:|
| Empty + dark | 0, 24 C, 0 lux, 450 ppm | 0% | 2.67% |
| High occupancy + high CO2 | 3, 32 C, 20 lux, 1600 ppm | 100% | 100% |
| Low occupancy + bright | 1, 24 C, 700 lux, 450 ppm | 3.33% | 25% |
| Empty + high CO2 | 0, 24 C, 0 lux, 1600 ppm | 0% | 100% |

Final percentages include energy mode adjustment. The fuzzy 'off' centroid is positive;
lighting is made exactly zero only by the explicit occupancy-zero gate.
CO2 and temperature thresholds are project assumptions, not health/building standards.

## Testing performed
src/verify_phase7.py checks 48 recorded example/model/policy combinations for parity
with the reusable energy pipeline, four teaching scenarios, malformed input rejection,
HTTP routes, file path restrictions and local-origin handling: 72 checks passed.
JavaScript syntax was checked with Node.
See outputs/phase7/verification.json for the machine-readable result.
Regression results for phases 1–6 are recorded separately in outputs/phase7/regression.json.

Browser automation could not complete during the original handoff. A later interactive
browser check passed on a desktop-sized window and at 375-pixel phone width: five tabs,
both models, manual and sensor modes, false-empty examples, energy filters, JSON result
generation and narrow layout. See outputs/phase7/browser_verification.json.
Before presenting, still verify readability at your projector resolution.

Post-check dashboard polish: switching tabs returns to the top; the result scrolls into view
after a run; a recorded example shows its known label beside the prediction (display only,
never a model input, and hidden once a reading is edited); the JSON file is named
roomiq-simulation.json.

The 10-slide PPTX passed structural, layout/font and import checks. Every rendered slide
was visually inspected. Native PowerPoint execution is not claimed.
The final PPTX is in outputs/phase7.

## What Phase 7 does not change
The training split, selected features, model files, GA objective and energy assumptions remain
as established in earlier phases. Test data are previously examined, not a fresh blind holdout.
High test accuracy does not resolve poor occupied recall. No real saving or comfort guarantee
follows from the simulated energy reduction. New rooms, occupied data, temporal safeguards
and real power/comfort measurements are needed before considering physical deployment.

## Viva explanation
“I connected all previous phases into a local dashboard. The model estimates occupancy,
the Mamdani controller explains the request through memberships and fired rules, and the
energy layer calculates a transparent estimate from assumed power. I also show failure cases,
because a low energy estimate can result from incorrectly treating an occupied room as empty.”
