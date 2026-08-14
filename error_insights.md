# Error Insights

Residuals (`predicted − actual`) on the test set, from the selected linear model (`model_notes.md`), segmented by `Weather`, `Traffic_Level`, and `Distance_km` (`notebooks/model_exploration.ipynb`, Step 13). A negative mean residual means the model underestimates delivery time for that segment.

## Residuals by Segment

| Weather | Mean residual | Std | n |
|---|---|---|---|
| Foggy | -5.75 | 15.65 | 21 |
| Windy | -1.83 | 9.90 | 16 |
| Snowy | +1.11 | 7.80 | 20 |
| Rainy | +1.48 | 8.67 | 40 |
| Clear | +2.77 | 6.36 | 95 |

| Traffic_Level | Mean residual | Std | n |
|---|---|---|---|
| High | -3.51 | 12.92 | 38 |
| Medium | +1.07 | 6.71 | 79 |
| Low | +2.26 | 8.06 | 72 |

| Distance_km bucket | Mean residual | Std | n |
|---|---|---|---|
| (0, 5] | -0.14 | 9.26 | 51 |
| (5, 10] | +1.25 | 7.38 | 50 |
| (10, 15] | +3.22 | 7.85 | 47 |
| (15, 20] | +0.08 | 10.88 | 52 |

Overall mean residual: +1.06 (a mild overestimation bias, small relative to the model's test MAE reported in `model_notes.md`).

## Weather Hypothesis: Not Confirmed

**The EDA-driven hypothesis — that the model systematically underestimates under `Snowy`/`Rainy` — does not hold; if anything, the opposite.** Both categories show a slightly *positive* mean residual (mild overestimation, not underestimation). The `Weather_Snowy` and `Weather_Rainy` coefficients (`model_notes.md`, both p < 0.001) already absorb most of the raw effect the EDA found, leaving little systematic bias behind.

**The real systematic underestimation shows up under `Foggy` weather and `High` traffic instead** — neither part of the original hypothesis, but concrete, model-relative biases. `Foggy` also has by far the largest residual spread of any weather category: the bias is concentrated in a handful of extreme cases (see below), not spread evenly across every `Foggy` order.

`Distance_km` buckets show no consistent directional trend — error does not systematically grow or shrink with distance once the model's own `Distance_km` term already accounts for it.

## Top 5 Highest-Error Orders

| Order_ID | Weather | Distance (km) | Actual | Predicted | Residual | Top SHAP driver |
|---|---|---|---|---|---|---|
| 773 | Foggy | 2.99 | 90.00 | 36.76 | -53.24 | Distance_km (-21.27) |
| 150 | Foggy | 5.93 | 106.00 | 72.38 | -33.62 | Distance_km (-12.48) |
| 428 | Windy | 17.81 | 122.00 | 89.45 | -32.55 | Distance_km (+23.07) |
| 729 | Foggy | 4.43 | 63.00 | 33.05 | -29.95 | Distance_km (-16.97) |
| 732 | Rainy | 16.38 | 90.00 | 67.50 | -22.50 | Distance_km (+18.79) |

**`Distance_km` is the top SHAP contributor in all 5 of the largest errors** — either pushing the prediction down for a short order that still took a long time, or pushing it up for a long order that took even longer than that push accounted for. `Preparation_Time_min` and `Traffic_Level` are consistently the next-largest contributors (`explainability.md`). Every case is the model's two most important features (`Distance_km`, `Preparation_Time_min`) reaching their limit on an unusually extreme order, not a feature the model ignores. 3 of the 5 are `Foggy` orders, reinforcing the segment-level finding above. Full SHAP breakdown of the single largest error: `explainability.md`.

## Outlier Cross-Check

**Only 1 of the 5 largest errors is an EDA-flagged IQR outlier.** `EDA_report.md` flagged 6 rows as IQR outliers, all on `Delivery_Time_min` (> 116 min) — `Distance_km` and `Preparation_Time_min` had 0 outliers each, so an order could only match here through its actual delivery time.

| Order_ID | Actual (min) | Residual | EDA outlier? |
|---|---|---|---|
| 773 | 90.00 | -53.24 | No |
| 150 | 106.00 | -33.62 | No |
| 428 | 122.00 | -32.55 | **Yes** — `Delivery_Time_min` |
| 729 | 63.00 | -29.95 | No |
| 732 | 90.00 | -22.50 | No |

- **Order 428 is one of the 6 rows the EDA deliberately kept rather than removing or capping** (`EDA_report.md`, "Outliers"). This is the direct, observable effect of that decision: the model still under-predicts an order that was already flagged, at the EDA stage, as unusually extreme.
- **The other 4 are not outliers by the IQR method on any of the three columns checked**, including `Delivery_Time_min` itself (all below the 116 min threshold). Their large errors do not come from an extreme, easily-flagged input — they are ordinary-looking orders (moderate distance, typical prep time) that still took far longer than the model's features predict. This is an open question this analysis does not resolve: something specific to these deliveries drove an unusually long delivery time without leaving a trace in any of the three EDA-checked columns — a candidate for further investigation if additional data becomes available (e.g. a specific courier or a restaurant-side delay not captured in this dataset).
