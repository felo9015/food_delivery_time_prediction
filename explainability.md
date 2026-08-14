# Explainability

## Planned Approach

Interpretability will be handled with **SHAP** (SHapley Additive exPlanations), based on Shapley values from cooperative game theory: each feature's contribution to a given prediction is computed by averaging its marginal effect across all possible orderings of the features, which gives a consistent, theoretically grounded way to attribute a prediction to its inputs.

The specific SHAP method will depend on the model family:

- **Tree-based models** (planned next, after the linear models documented in `model_notes.md`): **TreeSHAP**, the exact, efficient SHAP variant designed for tree ensembles.
- **Linear models** (the baseline and advanced OLS regressions already built): standardized coefficients as the reference measure of feature importance, since a linear model's coefficients are already a direct, interpretable measure of each feature's effect, without needing an approximation method.

**AIC does not apply to the tree-based models** — it is a metric for parametric models fit by (log-)likelihood, which tree ensembles are not. Comparison across every model in this project, linear and tree-based alike, will therefore use test-set **MAE and RMSE** as the common criterion, the same metrics already computed for the baseline and advanced regressions in `model_notes.md`.

SHAP values themselves are not computed yet — that follows once the tree-based models are trained.
