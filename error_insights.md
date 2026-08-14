# Error Insights

## Initial Observations

This section collects what has already surfaced from the EDA and the first two regression models. A full per-segment error breakdown is deferred until every planned model (including the tree-based ones) has been trained — see "To Be Completed," below.

**From the EDA (`EDA_report.md`):** `Weather = Snowy` showed the highest average delivery time of any category (67.11 min vs. 53.08 min for `Clear`, a ~14 min gap — the largest categorical effect found). `Rainy` was also elevated (59.79 min), though by a smaller margin than `Snowy`. This is a hypothesis worth checking once per-segment error analysis is possible: does the model systematically *underestimate* delivery time specifically under `Snowy` (and, to a lesser extent, `Rainy`) conditions, beyond what the `Weather` coefficient already corrects for — or does the coefficient fully absorb the effect?

**From the baseline and advanced model diagnostics (`model_notes.md`):**

- **Breusch-Pagan:** both models reject the null of constant residual variance (baseline LM p ≈ 0.011; advanced LM p ≈ 0.037) — heteroscedasticity is present in both, not resolved by adding the interaction terms.
- **Residuals vs. fitted:** both models show a dense, roughly symmetric band of residuals near zero, plus a one-sided scatter of large *positive* residuals (up to +60 minutes) with no comparable cluster on the negative side — errors are not evenly spread, and the model's largest misses are concentrated among under-predictions of unusually slow deliveries, rather than over-predictions of fast ones.

Together, these two findings point in the same direction: prediction error is not uniform across the range of `Delivery_Time_min` — it grows for the slower, more extreme deliveries, the same handful flagged as IQR outliers in the EDA.

## To Be Completed

A proper segment-level error analysis — error broken down by `Weather`, `Traffic_Level`, distance bucket, and similar slices, across all trained models — is deferred until every planned model is available for comparison, so the analysis reflects the final set of candidates rather than being redone per model.
