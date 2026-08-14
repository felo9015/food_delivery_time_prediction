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

| Weather | Distance (km) | Actual | Predicted | Residual | Top SHAP driver |
|---|---|---|---|---|---|
| Foggy | 2.99 | 90.00 | 36.76 | -53.24 | Distance_km (-21.27) |
| Foggy | 5.93 | 106.00 | 72.38 | -33.62 | Distance_km (-12.48) |
| Windy | 17.81 | 122.00 | 89.45 | -32.55 | Distance_km (+23.07) |
| Foggy | 4.43 | 63.00 | 33.05 | -29.95 | Distance_km (-16.97) |
| Rainy | 16.38 | 90.00 | 67.50 | -22.50 | Distance_km (+18.79) |

**`Distance_km` is the top SHAP contributor in all 5 of the largest errors** — either pushing the prediction down for a short order that still took a long time, or pushing it up for a long order that took even longer than that push accounted for. `Preparation_Time_min` and `Traffic_Level` are consistently the next-largest contributors (`explainability.md`). None of the 5 errors trace back to a feature the model ignores — every case is the model's two most important features (`Distance_km`, `Preparation_Time_min`) reaching their limit on an unusually extreme order, consistent with the IQR outliers on `Delivery_Time_min` already flagged in the EDA. 3 of the 5 are `Foggy` orders, reinforcing the segment-level finding above. Full SHAP breakdown of the single largest error: `explainability.md`.
