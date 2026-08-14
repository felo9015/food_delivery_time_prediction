# Model Notes

This document records the modeling logic, the assumptions behind it, and the reasoning tested in `notebooks/model_exploration.ipynb`. The preprocessing pipeline lives in `model_pipeline/`; this file covers the modeling decisions built on top of it.

## Modeling Approach

Two `statsmodels.OLS` regressions were built and compared on the same 80/20 train/test split (`random_state=42`) and the same preprocessed features from `model_pipeline/data_preprocessing.py` (median imputation for numeric columns; `"Unknown"` imputation plus ordinal/one-hot encoding for the categoricals, with `OneHotEncoder(drop="first")`):

- **Baseline:** the 15 preprocessed features plus an intercept, no interaction or polynomial terms.
- **Advanced:** the baseline plus 12 interaction terms (`Distance_km × Traffic_Level`, `Distance_km × Weather`, `Courier_Experience_yrs × Weather`, `Courier_Experience_yrs × Traffic_Level`).

`statsmodels.OLS` was used instead of `sklearn`'s linear regression specifically so that p-values, adjusted R², and confidence intervals per coefficient are available directly from `.summary()`.

## OLS Assumptions

`statsmodels.OLS` rests on 5 classical assumptions. Each was evaluated using the reusable functions in `model_pipeline/model_diagnostics.py`, applied to both the baseline and the advanced model.

1. **Linearity.** Evaluated with `plot_residuals_vs_fitted`. Both models show a dense band of residuals centered near zero with no systematic curve or trend, but a one-sided scatter of large positive residuals. No evidence of a systematic nonlinear relationship; the asymmetric outliers are a separate issue (see normality, below).

2. **Independence of the errors.** Each row is a single, independent order; there is no repeated-courier panel structure in this dataset the way there was for the trend analysis in Part I (SQL), where multiple deliveries per courier over time made autocorrelation a real concern worth testing formally with a window function. Without an identifier linking rows into groups, or a meaningful time ordering, there is nothing here for a Durbin-Watson-style test to actually check. `statsmodels`' `.summary()` prints a Durbin-Watson statistic by default regardless (≈ 2.03 for both models), but it is not treated as diagnostic here, for the same reason. Independence is assumed based on the data's design (one row = one independent order), not tested.

3. **Homoscedasticity.** Evaluated formally with `breusch_pagan_test`. Both models reject the null of constant residual variance at the 5% level (baseline LM p ≈ 0.011; advanced LM p ≈ 0.037). This assumption does not hold cleanly in either model.

4. **Normality of the residuals.** Evaluated with `qq_plot`, backed by the `Omnibus`/`Jarque-Bera` block in `.summary()`. Both models show a clearly non-normal, right-skewed, heavy-tailed residual distribution (skew ≈ 2.4, kurtosis ≈ 13 in both, vs. 0 and 3 for a normal distribution). Clearly violated, driven by the same handful of extreme `Delivery_Time_min` values flagged as IQR outliers in `EDA_report.md`.

5. **No severe multicollinearity.** Evaluated with `compute_vif`, using the common VIF > 5 (moderate) / VIF > 10 (severe) rule of thumb. Clean for the baseline (every VIF ≤ 1.46) after fixing an initial dummy-variable-trap issue with `OneHotEncoder(drop="first")` (the first pass at the baseline had `VIF = inf` on every one-hot column, from encoding every category of each nominal feature without dropping one, which made the design matrix exactly rank-deficient alongside the intercept). Moderately elevated for the advanced model instead: `Traffic_Level` (≈ 6.9), several `Weather` dummies (5.1-9.2), and `Distance_km`/`Courier_Experience_yrs` (≈ 3.2-3.3) all rise once they have interaction partners, since the continuous variables were not centered before being multiplied — a well-known side effect of uncentered interactions, not a sign that the underlying variables are redundant with each other (their own pairwise correlations remain near zero). Nothing crosses the severe (> 10) threshold.

**Net assessment:** linearity and (assumed) independence hold reasonably; homoscedasticity and normality do not, in either model, and are not improved by the interaction terms; multicollinearity is clean in the baseline and moderately elevated in the advanced model. None of this was corrected in this pass — it is documented to inform the next modeling step, since violated assumptions do not make a model useless, but they do mean its p-values and confidence intervals should be read with that caveat.

## Interaction Terms: Rationale and Related Work

The four interaction terms in the advanced model — courier-level information crossed with contextual conditions (`Weather`, `Traffic_Level`) — are adapted from:

> Fu, G., Chi, Y., Zheng, L., & Shen, Z. J. M. (2025). A Deep & Cross Network-based framework for online food delivery time prediction with driver-specific information. *Omega*.

That paper finds value in modeling interactions between driver-specific information (age, rating) and contextual variables (traffic, weather) using a Deep & Cross Network (DCN) framework, trained on real operational data from a food delivery platform.

**This project adapts that idea; it does not replicate it.** The dataset available here has no courier identifier and none of the paper's original driver-level variables (age, rating, city, order date, a multi-delivery indicator). `Courier_Experience_yrs` is used as the closest available proxy for "driver-specific information," and `Traffic_Level`/`Weather` as the equivalent contextual variables — the same conceptual pairing (driver information × context), applied to the variables this dataset actually has.

**The paper's core architecture (a Deep & Cross Network / deep learning with embeddings) is deliberately not implemented, for three reasons:**

1. **Scale.** The paper works with large-scale real operational data (Zomato); this dataset has 1,000 rows — far too little to justify a deep learning architecture without a high risk of overfitting.
2. **Problem fit.** The DCN's purpose in the paper is capturing heterogeneity across thousands of individual couriers via their ID (a high-cardinality categorical feature that benefits from embeddings). This dataset has no courier ID column, so that specific problem does not apply here.
3. **Signal already captured by a simpler model.** The EDA already found a strong, roughly linear relationship between `Distance_km` and the target (r ≈ 0.78), suggesting that a linear model already captures most of the available signal — consistent with the advanced model's own result below, where the added interaction complexity did not clearly outperform the simpler baseline.

## Model Comparison: Baseline vs. Advanced

| Metric | Baseline | Advanced |
|---|---|---|
| Adj. R² | 0.761 | 0.763 |
| AIC | 6106.94 | 6112.96 |
| Test MAE (min) | 6.185 | 6.239 |
| Test RMSE (min) | 9.054 | 9.024 |

**The interactions do not earn their added complexity.** By AIC — which explicitly penalizes extra parameters — the baseline is preferred (lower by ≈ 6 points, despite the advanced model fitting the training data slightly better). Test RMSE improves only marginally; test MAE gets slightly worse. Adj. R² moves by less than a hundredth. Of the 12 interaction terms, only 1 (`Courier_Experience_yrs × Weather_Unknown`, p = 0.040) is individually significant, and it involves the originally-missing-data category rather than a real weather condition, making it more plausibly a small-subgroup artifact than a genuine effect. Both models are kept for the record; **the baseline is the stronger candidate to carry forward.**

## Tree-Based Models

Three tree ensembles were trained next — `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor` — using the same `build_preprocessing_pipeline()` (no scaling needed for trees, but the same validated imputation and encoding, for consistency with the linear models) and the same 15-feature `X_train`/`X_test`, without the interaction terms built for the advanced linear model (trees can learn interactions and nonlinearities on their own). Default, reasonable hyperparameters were used (`n_estimators=100`, a fixed `random_state`, no other tuning) — a hyperparameter search was deliberately left out of this pass. Each model was evaluated with 5-fold cross-validation on the training set (for a sense of stability across subsets) and with `compute_test_metrics` on the held-out test set. AIC is a likelihood-based metric for parametric models and does not apply to tree ensembles, so it is reported as `N/A` for all three.

| Model | CV MAE (mean ± std) | Train MAE | Test MAE | Test RMSE | Train→Test MAE Gap |
|---|---|---|---|---|---|
| Random Forest | 7.90 ± 0.59 | 2.95 | 6.92 | 10.03 | 3.97 |
| XGBoost | 8.73 ± 0.41 | 0.57 | 7.63 | 10.33 | 7.05 |
| LightGBM | 7.78 ± 0.51 | 3.97 | 7.13 | 9.93 | 3.17 |

**All three overfit, `XGBoost` severely.** Its train MAE (0.57 min) indicates it has essentially memorized the training set, against a test MAE more than 13x larger (7.63 min) — and it also has the worst mean CV MAE, despite the tightest CV std, meaning it overfits consistently across folds rather than unstably. `Random Forest` and `LightGBM` overfit more moderately but still clearly, with train→test MAE gaps of ≈ 3-4 minutes. With only 800 training rows, unregularized default hyperparameters give these ensembles more capacity than the data supports without a validation-driven stopping rule or explicit regularization.

## Model Comparison: All 5 Models

| Model | Test MAE (min) | Test RMSE (min) | AIC | Note |
|---|---|---|---|---|
| Baseline (linear) | 6.185 | 9.054 | 6106.94 | Best test MAE and lowest AIC overall |
| Advanced (interactions) | 6.239 | 9.024 | 6112.96 | Best test RMSE overall, by a razor-thin margin; AIC still favors the baseline |
| Random Forest | 6.923 | 10.032 | N/A | Best of the 3 tree models; still trails both linear models by ~0.7+ min MAE |
| XGBoost | 7.626 | 10.327 | N/A | Worst performer overall; most severely overfit |
| LightGBM | 7.132 | 9.928 | N/A | Second-best tree model; still behind both linear models |

**First-pass conclusion: the linear baseline remains the strongest candidate, and no tree model is a clear enough improvement to justify carrying forward as-is.** This runs against the usual expectation that tree ensembles beat linear models on tabular data, but it is consistent with the reasoning already given above for not using a deep learning architecture: with only 1,000 rows total and a target whose strongest driver (`Distance_km`) is already close to linear (r ≈ 0.78 in the EDA), there is limited room for higher-capacity models to find real structure beyond what the linear models already capture — their extra capacity instead goes toward overfitting, exactly as the train/test gaps above show.

## Next Steps

Neither linear model resolves the heteroscedasticity or non-normal residuals found in the OLS diagnostics. The tree models, which do not carry those same assumptions, currently underperform them on raw defaults — whether a hyperparameter search would close that gap (most plausibly for `Random Forest` or `LightGBM`, the two tree models that overfit the least) is worth deciding together before investing more effort, given the linear models' current lead. See `explainability.md` for how model interpretability will be handled across both model families, and `error_insights.md` for the error patterns already observed that motivate this direction.
