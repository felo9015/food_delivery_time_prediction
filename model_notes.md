# Model Notes

Records the modeling logic, assumptions, and reasoning tested in `notebooks/model_exploration.ipynb`. The preprocessing pipeline lives in `model_pipeline/`; this file covers the modeling decisions built on top of it.

## Modeling Approach

Two `statsmodels.OLS` regressions, same 80/20 split (`random_state=42`) and same preprocessing (`model_pipeline/data_preprocessing.py`: median imputation for numeric columns; `"Unknown"` imputation plus ordinal/one-hot encoding for categoricals, `OneHotEncoder(drop="first")`):

- **Baseline** — 15 preprocessed features + intercept, no interactions.
- **Advanced** — baseline + 12 interaction terms (`Distance_km`/`Courier_Experience_yrs` × `Weather`/`Traffic_Level`).

`statsmodels.OLS` (rather than `sklearn`) was used specifically for its per-coefficient p-values, adjusted R², and confidence intervals via `.summary()`.

## OLS Assumptions

Evaluated with the reusable functions in `model_pipeline/model_diagnostics.py`, applied to both models:

| Assumption | Method | Verdict | Why |
|---|---|---|---|
| Linearity | Residuals vs. fitted | Holds | Residual band centered near zero, no systematic curve; the one-sided large-positive-residual scatter is a separate (normality) issue |
| Independence | Design-based | Assumed, not tested | One row = one independent order; no repeated-courier or time structure to test. Durbin-Watson (≈2.03, both models) is printed by `.summary()` but is not diagnostic here for the same reason |
| Homoscedasticity | Breusch-Pagan | Violated (both) | Rejects the constant-variance null at 5% (baseline p≈0.011, advanced p≈0.037) |
| Normality | Q-Q plot + Jarque-Bera | Violated (both) | Right-skewed, heavy-tailed (skew≈2.4, kurtosis≈13 vs. 0/3 expected), driven by the same extreme `Delivery_Time_min` values flagged as IQR outliers in `EDA_report.md` |
| No severe multicollinearity | VIF (>5 moderate, >10 severe) | Clean (baseline); moderate (advanced) | Baseline: all VIF ≤ 1.46, after fixing an initial VIF=inf dummy-variable-trap bug via `OneHotEncoder(drop="first")`. Advanced: several dummies rise to the 5–9 range because the interaction terms use uncentered continuous variables, not because the underlying features are redundant (their pairwise correlations stay near zero); nothing crosses the severe (>10) threshold |

**Net assessment:** linearity and (assumed) independence hold; homoscedasticity and normality do not, in either model, and the interactions don't fix them; multicollinearity is clean in the baseline and moderately elevated in the advanced model. Not corrected in this pass — documented so the next step's p-values/CIs are read with this caveat, since violated assumptions don't make a model useless.

## Interaction Terms: Rationale and Related Work

The four interaction terms (courier-level information × context) are adapted from:

> Fu, G., Chi, Y., Zheng, L., & Shen, Z. J. M. (2025). A Deep & Cross Network-based framework for online food delivery time prediction with driver-specific information. *Omega*.

That paper models interactions between driver-specific variables (age, rating) and context (traffic, weather) with a Deep & Cross Network (DCN), on real operational data. **This project adapts the idea, not the architecture:** the dataset has no courier identifier or the paper's driver-level variables, so `Courier_Experience_yrs` stands in as the closest available proxy, paired with `Traffic_Level`/`Weather`.

The DCN itself is deliberately not implemented:

- **Scale** — the paper uses large-scale operational data; this dataset has 1,000 rows, too little to justify a deep architecture without a high overfitting risk.
- **Problem fit** — the DCN's role is capturing heterogeneity across thousands of courier IDs via embeddings; there is no courier ID column here, so that problem doesn't apply.
- **Signal already captured** — `Distance_km` already has a strong, roughly linear relationship with the target (r≈0.78 in the EDA), suggesting a linear model captures most of the signal — confirmed below, where the interactions don't outperform the baseline.

## Model Comparison: Baseline vs. Advanced

| Metric | Baseline | Advanced |
|---|---|---|
| Adj. R² | 0.761 | 0.763 |
| AIC | 6106.94 | 6112.96 |
| Test MAE (min) | 6.185 | 6.239 |
| Test RMSE (min) | 9.054 | 9.024 |

**The interactions don't earn their complexity.** AIC favors the baseline despite the advanced model's slightly better training fit; test RMSE improves marginally, test MAE gets slightly worse. Of the 12 interaction terms, only 1 (`Courier_Experience_yrs × Weather_Unknown`, p=0.040) is individually significant, and it involves the missing-data category rather than a real weather condition — more plausibly a small-subgroup artifact than a genuine effect. Both models are kept for the record; **the baseline is the stronger candidate.**

## Tree-Based Models (Untuned)

`RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`, same preprocessing (no scaling needed for trees) and same 15 features, without the linear models' interaction terms (trees can learn interactions on their own). Default hyperparameters (`n_estimators=100`, fixed `random_state`), no tuning in this pass — evaluated with 5-fold CV plus `compute_test_metrics` on the test set. AIC does not apply to tree ensembles (`N/A`).

| Model | CV MAE (mean ± std) | Train MAE | Test MAE | Test RMSE | Train→Test Gap |
|---|---|---|---|---|---|
| Random Forest | 7.90 ± 0.59 | 2.95 | 6.92 | 10.03 | 3.97 |
| XGBoost | 8.73 ± 0.41 | 0.57 | 7.63 | 10.33 | 7.05 |
| LightGBM | 7.78 ± 0.51 | 3.97 | 7.13 | 9.93 | 3.17 |

**All three overfit, `XGBoost` severely.** Its train MAE indicates it has essentially memorized the training set, and it also has the worst mean CV MAE of the three despite the tightest CV std — it overfits consistently across folds, not unstably. `Random Forest` and `LightGBM` overfit more moderately. With only 800 training rows, unregularized defaults give these ensembles more capacity than the data supports.

## Light Tuning: Random Forest & LightGBM

**XGBoost is excluded from tuning.** Its train/test gap (0.57 vs. 7.63 min MAE, ≈7.05 min) is too large to close with moderate hyperparameter tuning — it signals that the default capacity (depth, `n_estimators`) is fundamentally misaligned with a training set of 800 rows, not a fine-tuning problem. Closing that gap would require such a conservative search that the result would functionally converge toward Random Forest or LightGBM, so tuning effort is invested in those two instead, where overfitting is already moderate and correctable.

**Search space prioritizes capacity-reducing parameters** (`max_depth`, `min_samples_leaf`/`min_child_samples`, L1/L2 regularization in LightGBM) over capacity-increasing ones. `Distance_km` already explains most of the variance (r≈0.78, `EDA_report.md`), and the linear models already capture that signal plus the relevant interactions — leaving little residual nonlinear structure for a tree to exploit. In this regime, the highest-impact tuning move is restricting capacity so the model stops fitting noise, not searching for a better fit.

**Why tune at all, rather than dismiss trees outright:** linear models are expected to remain the safer choice given the sample size, but asserting that without empirical verification would be premature. Verifying with evidence is more defensible than reasoning from theory alone, so this comparison is documented as a legitimate part of model selection regardless of outcome.

`RandomizedSearchCV`, 5-fold CV, `scoring="neg_mean_absolute_error"`, 30 iterations:

| Model | Best CV MAE | Train MAE | Test MAE | Test RMSE | Train→Test Gap | Gap vs. Untuned |
|---|---|---|---|---|---|---|
| Random Forest (tuned) | 7.535 | 5.537 | 6.967 | 9.905 | 1.430 | −64% (from 3.97) |
| LightGBM (tuned) | 7.301 | 6.210 | 6.466 | 9.342 | 0.256 | −92% (from 3.17) |

Tuning substantially closes the overfitting gap for both models — most dramatically for LightGBM, whose train/test gap nearly disappears. **Neither tuned model surpasses the linear models**, though: LightGBM (tuned) test MAE comes within ≈0.28 min of the baseline but does not beat it; Random Forest (tuned) remains further behind.

**This confirms the stated hypothesis:** with a dominant linear signal (`Distance_km` r≈0.78) and a small dataset (800 training rows), added tree complexity — even after capacity-restricting tuning — does not translate into better generalization than the simpler linear models. Tuning fixed the overfitting, but fixing overfitting is not the same as gaining predictive power beyond what the linear signal already captures. This is reported as a valid finding, not a shortfall — it is exactly what the hypothesis predicted.

## Final Model Comparison

| Model | Test MAE (min) | Test RMSE (min) | AIC | Note |
|---|---|---|---|---|
| Baseline (linear) | 6.185 | 9.054 | 6106.94 | Best test MAE and lowest AIC overall |
| Advanced (interactions) | 6.239 | 9.024 | 6112.96 | Best test RMSE, by a razor-thin margin; AIC still favors the baseline |
| LightGBM (tuned) | 6.466 | 9.342 | N/A | Strongest non-linear candidate; closest tree model to the linear models |
| Random Forest (tuned) | 6.967 | 9.905 | N/A | Overfitting reduced 64% by tuning; still behind both linear models |
| XGBoost (not tuned) | 7.626 | 10.327 | N/A | Excluded from tuning — misaligned default capacity, not a fine-tuning problem; worst performer overall |

**The linear baseline remains the strongest candidate.** Tuning closed most of the trees' overfitting gap but did not change the ranking: with 1,000 total rows and a target whose strongest driver (`Distance_km`) is already close to linear, there is limited room for higher-capacity models to find real structure beyond what the linear models capture.

## Next Steps

The heteroscedasticity and non-normal residuals from the OLS diagnostics remain unresolved by either linear model. See `explainability.md` for how interpretability is handled across model families, and `error_insights.md` for the error patterns motivating that direction.
