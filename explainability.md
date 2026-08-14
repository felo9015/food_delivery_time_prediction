# Explainability

## Approach for the Selected Model

The baseline linear model is the one used in production (see `model_notes.md`, "Selected Model and Production Pipeline"). Its `statsmodels.OLS` coefficients (`notebooks/model_exploration.ipynb`, Step 6) are already a complete, exact attribution of each feature's effect on the prediction — for a linear model, the coefficient *is* the effect, not an approximation of it, so no post-hoc method is needed to explain a prediction or the model as a whole.

## Why SHAP Was Not Pursued

SHAP (SHapley Additive exPlanations) was originally scoped for this project on the possibility that a tree-based model would end up in production, where TreeSHAP would be the natural exact, efficient choice. That did not happen: Random Forest, LightGBM, and XGBoost were all evaluated, including a light hyperparameter-tuning pass for the two moderately-overfit candidates, and none outperformed the linear baseline (`model_notes.md`). With no tree model in production, TreeSHAP has nothing to explain, and applying a SHAP approximation to the linear model itself would add computational overhead without adding information beyond what its coefficients already give directly.

Cross-family comparison during model selection used test-set **MAE and RMSE**, since AIC (used to compare the two linear models against each other) is a likelihood-based metric that does not apply to the tree ensembles.
