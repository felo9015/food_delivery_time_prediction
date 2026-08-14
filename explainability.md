# Explainability

## Planned Approach

**SHAP** (SHapley Additive exPlanations) will be used for interpretability: each feature's contribution to a prediction is computed by averaging its marginal effect across all possible feature orderings (Shapley values from cooperative game theory), giving a consistent, theoretically grounded attribution.

| Model family | Method | Why |
|---|---|---|
| Tree-based (Random Forest, LightGBM, XGBoost) | TreeSHAP | Exact, efficient SHAP variant designed for tree ensembles |
| Linear (baseline, advanced OLS) | Standardized coefficients | Already a direct, interpretable measure of each feature's effect — no approximation needed |

Cross-family comparison uses test-set **MAE and RMSE** (see `model_notes.md`), since AIC is a likelihood-based metric that does not apply to the tree ensembles.

SHAP values are not yet computed — planned as a next step, alongside the segment-level error analysis in `error_insights.md`.
