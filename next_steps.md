# Next Steps

Open items surfaced across this project's documentation, not yet acted on.

| Next Step | Rationale |
|---|---|
| **Validate the Dockerfile** | Build and run the container locally to confirm the API behaves identically inside Docker as it does when run directly with `uvicorn`. Not yet verified in this environment (`api/README.md`). |
| **Gather more data and re-confirm the winning model** | The current dataset (1,000 rows) is small relative to what tree ensembles typically need to outperform linear regression. With more data, retrain and re-compare all 5 models (baseline, advanced regression, Random Forest, XGBoost, LightGBM) to check whether the linear model still wins, or whether the added data changes the outcome (`model_notes.md`). The current conclusion is conditional on the current sample size and should not be treated as permanent. |
| **Test the untested `Weather × Traffic_Level` interaction** | Identified as an open question in `strategic_reflections.md` ("Model Failure"): the confirmed underestimation bias in `Foggy` weather and `High` traffic conditions was never tested for a compounding interaction effect between the two. |
| **Investigate the 4 unexplained large-error cases** | 4 of the 5 largest test-set errors did not correspond to known EDA outliers and were left as an open question (`error_insights.md`). Worth revisiting with additional data or features if available. |
| **Add automated tests** | Unit tests for the preprocessing pipeline (`model_pipeline/`) and integration tests for the API endpoint — identified as out of scope in the "Going to Production" reflection (`strategic_reflections.md`) but a natural next step. |
| **Add basic prediction logging** | Store predictions made through the API alongside a way to later compare them against actual outcomes, as a foundation for monitoring and future retraining. |
| **Validate on a local sample before any new-market deployment** | Directly tied to the Mumbai → São Paulo transferability discussion (`strategic_reflections.md`): recalibrate or validate the model against local data before trusting it in a new market. |
