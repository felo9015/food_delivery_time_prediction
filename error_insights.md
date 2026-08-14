# Error Insights

## Observations

Two findings point in the same direction — prediction error is not uniform across `Delivery_Time_min`, and grows for the slower, more extreme deliveries:

- **From the EDA** (`EDA_report.md`): `Weather = Snowy` has the largest categorical effect on delivery time. Open question: does the selected model systematically *underestimate* delivery time under `Snowy`/`Rainy` conditions beyond what the `Weather` coefficient already corrects for, or does the coefficient fully absorb the effect?
- **From the OLS diagnostics** (`model_notes.md`): Breusch-Pagan rejects constant residual variance, and residuals-vs-fitted shows a one-sided scatter of large *positive* residuals with no comparable negative-side cluster — the largest misses are under-predictions of unusually slow deliveries, concentrated among the same rows flagged as IQR outliers in the EDA.

## To Be Completed

A segment-level error breakdown (`Weather`, `Traffic_Level`, distance bucket, etc.) was deferred until the final candidate set was available, so the analysis would not need to be redone per model. Model selection is now complete (`model_notes.md`) — the baseline linear model is the one in production — so this breakdown, if pursued, should now target that single model rather than the full comparison set.
