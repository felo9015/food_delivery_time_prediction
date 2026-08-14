# Error Insights

## Initial Observations

A full per-segment error breakdown is deferred until all planned models are trained (see "To Be Completed"). Two findings already point in the same direction — prediction error is not uniform across `Delivery_Time_min`, and grows for the slower, more extreme deliveries:

- **From the EDA** (`EDA_report.md`): `Weather = Snowy` has the largest categorical effect on delivery time. Open question for the segment-level analysis: does the model systematically *underestimate* delivery time under `Snowy`/`Rainy` conditions beyond what the `Weather` coefficient already corrects for, or does the coefficient fully absorb the effect?
- **From the OLS diagnostics** (`model_notes.md`): Breusch-Pagan rejects constant residual variance in both linear models, and residuals-vs-fitted shows a one-sided scatter of large *positive* residuals with no comparable negative-side cluster — the largest misses are under-predictions of unusually slow deliveries, concentrated among the same rows flagged as IQR outliers in the EDA.

## To Be Completed

Segment-level error analysis (`Weather`, `Traffic_Level`, distance bucket, etc.) across all trained models, deferred until the final candidate set is available so the analysis isn't redone per model.
