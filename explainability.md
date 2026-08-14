# Explainability

## Method

The selected model (`model_notes.md`) is a linear regression, so `shap.LinearExplainer` applies exactly — SHAP values for a linear model reduce directly to `coefficient × (feature value − background mean)`, with no sampling approximation. The explainer uses the full training set as background and is evaluated on the test set, so it explains genuinely out-of-sample predictions (`notebooks/model_exploration.ipynb`, Step 12).

## Global Importance

| Feature | Mean \|SHAP\| | Coefficient |
|---|---|---|
| Distance_km | 14.54 | 2.99 |
| Preparation_Time_min | 6.02 | 0.97 |
| Traffic_Level | 3.44 | 4.92 |
| Weather_Snowy | 1.64 | 9.27 |
| Weather_Rainy | 1.62 | 5.02 |
| Courier_Experience_yrs | 1.46 | -0.63 |
| Weather_Foggy | 1.16 | 6.26 |
| Time_of_Day_Unknown | 0.45 | 7.69 |

- **`Distance_km` is the top feature by SHAP**, confirming the EDA's strongest correlation (r ≈ 0.78) directly through the model's predictions — even though its raw coefficient is smaller than several `Weather` dummy coefficients.
- **This is not an inconsistency — it is exactly what SHAP is supposed to correct for.** SHAP is exact for a linear model here (verified directly: every SHAP value matches `coefficient × (value − background mean)` with zero deviation), but a *raw* coefficient is not comparable across features on different scales. `Distance_km` varies continuously over ~20 km; the `Weather` dummies are mostly 0. SHAP's mean |value| accounts for that variability, which is why its ranking differs from raw coefficient magnitude while remaining mathematically derived from those same coefficients — both readings tell the same story, just at different levels (coefficient = effect per unit, SHAP = realized impact given how much that unit actually varies in the data).

## Local Explainability

Three test-set orders, chosen to cover the largest error, a near-perfect prediction, and a case where the dominant driver points the right direction but the model still falls short:

| Example | Distance | Weather | Predicted | Actual | Residual | Top SHAP driver |
|---|---|---|---|---|---|---|
| Largest error | 2.99 km | Foggy | 36.76 | 90.00 | -53.24 | Distance_km (-21.27) |
| Best-predicted | 2.58 km | Clear | 36.95 | 37.00 | -0.05 | Distance_km (-22.50) |
| Distance-driven miss | 17.81 km | Windy | 89.45 | 122.00 | -32.55 | Distance_km (+23.07) |

- **Largest error:** a short, high-traffic, low-prep order — `Distance_km` and `Preparation_Time_min` correctly push the prediction down, but the actual delivery still far exceeds it. Nothing in this model's feature set explains that gap; the delay traces back to something not captured here.
- **Best-predicted:** SHAP attributes the below-base prediction almost entirely to `Distance_km` and `Preparation_Time_min` — a case where the linear structure captures the order well.
- **Distance-driven miss:** `Distance_km` correctly pushes the prediction far above the base value, in the right direction, but the actual delivery still exceeds it by a wide margin — the model gets the dominant driver right without capturing the full magnitude of an unusually slow long-distance order.

All three, plus the global summary, confirm the same point: SHAP is a direct, exact decomposition of the linear model's own coefficients here, not new information beyond them — its value is making each individual prediction traceable, and validating that the coefficient-based reading (`model_notes.md`) and the SHAP-based reading tell the same story.
