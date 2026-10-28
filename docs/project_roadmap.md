# Seven-phase project roadmap

This follows the seven-phase plan in the original SC conversation.

1. Dataset understanding and preprocessing — completed.
2. Baseline occupancy prediction model — completed.
3. Genetic Algorithm sensor selection — completed.
4. Optimized occupancy system and GA evaluation — completed; formal retraining, comparison and prediction interface.
5. Fuzzy smart room controller — completed; 34 Mamdani rules, centroid outputs, occupancy integration and tests.
6. Energy estimation and adaptive decision layer — completed; audited interval simulation, explicit power assumptions and rule-based modes.
7. Final dashboard, integration tests and presentation — implemented; browser visual QA remains a manual handoff check.

Phase 3 included a preliminary fitted-model comparison. Phase 4 independently retrains the
fixed selected-input model and formalizes the evidence, operational error analysis and common
prediction interface. The term “optimized” refers to the GA objective, not a guaranteed improvement
on external data. Phase 5 now provides the static fuzzy controller; Phase 6 now adds explicit simulated power assumptions and mode decisions; Phase 7 now integrates the saved pipeline into a local dashboard and presentation.
