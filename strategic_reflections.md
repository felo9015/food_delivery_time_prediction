# Strategic Reflections

## 1. Model Failure

Before addressing a reported model failure, the first step is to verify that the failure is real rather than assumed. In this project, the specific scenario posed — underestimation of delivery time on rainy days — was tested directly against the error analysis in `error_insights.md` and was not confirmed: both `Rainy` and `Snowy` weather show a slightly positive mean residual (overestimation, not underestimation), because the `Weather_Rainy`/`Weather_Snowy` coefficients already absorb most of the raw effect observed during the EDA. The actual underestimation bias in this dataset appears under `Foggy` weather (-5.75 min) and `High` traffic (-3.51 min) conditions instead.

Given this confirmed (rather than assumed) failure, the appropriate fix depends on which of three explanations holds:

- If the failure reflects missing structure the model should be able to capture (e.g., a `Weather × Traffic_Level` interaction, which was not tested in this project — fog combined with heavy traffic may compound rather than simply add), the fix belongs in the **model**.
- If the failure stems from information the model was never given (e.g., real-time visibility conditions beyond a categorical weather label), the fix belongs in the **data**.
- If the bias is confirmed but small relative to overall error (-5.75 min against an overall test MAE of 6.185 min), the more cost-effective fix may be adjusting the ETA shown to the customer for these specific conditions rather than retraining — a **business-expectations** fix.

This project did not test a `Weather × Traffic_Level` interaction term, which would be the natural next step before choosing between a model fix and a business-expectations fix.

## 2. Transferability

Deploying this model from Mumbai to São Paulo requires distinguishing between what is likely to generalize structurally and what is specific to the original market. `Distance_km` is the dominant predictor (r ≈ 0.78 in the EDA, the leading SHAP contributor), and its role as the primary driver of delivery time is a structural relationship likely to hold across markets. In contrast, the coefficients on `Weather` and `Traffic_Level` are context-specific: São Paulo's climate does not include snow, so the `Weather_Snowy` coefficient learned in the original market would be irrelevant or poorly calibrated; typical delivery distances, traffic patterns, and the mix of `Vehicle_Type` also differ by city. `Courier_Experience_yrs` may not transfer cleanly either, since the composition of the gig-economy delivery workforce can vary meaningfully between markets.

Recommended approach: do not assume direct transfer of the trained coefficients. Validate on a local sample before serving predictions, monitor for distribution drift in the input features (delivery distances, weather category frequencies, traffic patterns), and choose between recalibrating (refitting the same linear specification with local data) versus a full model rebuild, depending on how much local data is available. Because the final model is a simple, low-capacity linear regression rather than a tuned black-box ensemble, recalibration is inexpensive — the same feature specification can be refit on local data without repeating the full model-selection process.

## 3. GenAI Disclosure

Generative AI tools (Claude, used for project planning and strategy discussion, and Claude Code for implementation) were used throughout this project. SQL query construction has its own disclosure in `sql/sql_insights.md`; this section focuses on the EDA and modeling stages, where the specific direction behind each decision — and not just the fact that AI was used — is the relevant part of the disclosure.

### EDA

- The missingness-mechanism classification (MCAR/MAR/MNAR per column, `EDA_report.md`) followed a specific methodology that was directed rather than left to Claude Code's default judgment: compare `Delivery_Time_min` between null and non-null rows per column, instead of classifying missingness by assumption or by column name alone.
- The imputation strategy — a constant `"Unknown"` category for `Weather`/`Traffic_Level`/`Time_of_Day` and the median for `Courier_Experience_yrs` — was a specific, directed decision, not Claude Code's default choice.
- The decision to keep the 6 IQR outliers on `Delivery_Time_min` at the EDA stage, rather than removing or capping them, and to defer that call to the modeling stage, was directed rather than assumed. Its consequence was verified later, independently, in the error analysis (`error_insights.md`): only 1 of the 5 largest test-set errors turned out to be one of those same outliers.

### Model

- **Tuning scope and reasoning:** the requirement not to tune XGBoost, along with the specific argument for why (its train/test MAE gap reflects a capacity mismatch with the training set size, not an under-tuned model), was specified in advance — not a conclusion Claude Code reached and then justified afterward. The same is true for prioritizing capacity-reducing hyperparameters (`max_depth`, `min_samples_leaf`/`min_child_samples`, L1/L2 regularization) over capacity-increasing ones in the Random Forest/LightGBM search space, and for the requirement to report explicitly whether the tuning outcome confirmed or contradicted the hypothesis that added complexity would not improve generalization here.
- **Production pipeline design:** using `sklearn.LinearRegression` instead of continuing with `statsmodels.OLS`, running a 5-fold CV stability check before freezing the model, and retraining the final pipeline on 100% of the data before serializing it — including the specific requirement to document that the reported metrics remain the ones obtained honestly on the held-out split, not new numbers from the 100%-data fit — were all specified requirements.
- **SHAP methodology:** the choice of `shap.LinearExplainer` specifically (exact, not approximate, for a linear model) and the required deliverables — a global summary plot cross-checked against the model's own coefficients, plus 2-3 local examples including the single largest test-set error — were specified before any SHAP code was written.
- **Error analysis scope:** the segmentation variables (`Weather`, `Traffic_Level`, `Distance_km`), and the requirement to explicitly test — and report the outcome of, whichever way it went — the rainy/snowy underestimation hypothesis raised in the EDA, were specified in advance.
- **Outlier cross-check:** whether the 5 largest test-set errors coincided with the outliers already flagged by the IQR method in the EDA was a check requested directly; it was not something Claude Code had already planned, and it surfaced the finding that only 1 of the 5 did.
- **Documentation accuracy:** staleness was flagged and correction requested directly on multiple occasions — for example, `explainability.md`'s SHAP plan no longer matched the model that was actually selected, and `model_exploration.ipynb`'s markdown cells had grown too long — rather than Claude Code identifying and fixing these unprompted.
- Interaction terms inspired by an external paper (Fu, Chi, Zheng & Shen, 2025, *Omega*) were explicitly adapted, not replicated, because the source paper's key variables (driver ID, age, rating) do not exist in this dataset — and the paper's core Deep & Cross Network architecture was deliberately not implemented, for reasons documented in `model_notes.md`.

## 4. Your Signature Insight

The most notable insight from this project is not a single number but a pattern: the discipline of testing hypotheses against evidence rather than accepting a plausible-sounding narrative. Two examples illustrate this. First, on model selection: despite the common assumption that ensemble/tree-based models outperform linear regression, none of Random Forest, XGBoost, or LightGBM — even after hyperparameter tuning aimed at reducing overfitting — beat the baseline linear regression on test MAE. With roughly 1,000 rows and a dataset dominated by one strongly linear predictor (`Distance_km`), the added flexibility of tree ensembles had little genuine structure to exploit and mostly increased overfitting instead (XGBoost showed a training-to-test MAE gap of roughly 7 minutes). Second, on error analysis: the EDA-stage hypothesis that the model underestimates on rainy/snowy days was directly tested and rejected — the real underestimation bias appeared under foggy weather and high traffic instead. In both cases, the less flattering or less "clean" result was reported as-is rather than adjusted to fit an expected narrative.

## 5. Going to Production

Moving this model to production involves several components. Two are prioritized as the concrete next step for this repository, to be implemented as working code rather than only described:

- **Serving layer:** a FastAPI endpoint (`api/main.py`, currently a placeholder) will wrap `predict.py`, validate incoming request payloads against the expected schema using Pydantic models (rejecting, for example, an unrecognized `Weather` category rather than silently mis-encoding it), and load the serialized `model.joblib` once at startup rather than on every request.
- **Containerization:** a minimal Dockerfile will package the API for consistent deployment.

Beyond that, a full production deployment would additionally require:

- **Model versioning:** tagging each serialized artifact with a version and the data/commit it was trained on, so predictions can be traced back to a specific model version.
- **Input/feature monitoring:** tracking the distribution of incoming request features over time to detect drift (directly relevant given the Mumbai-to-São Paulo scenario above) — for example, alerting if the proportion of `Weather` categories or the average `Distance_km` shifts meaningfully from the training distribution.
- **Prediction logging and feedback loop:** storing predictions alongside eventual actual delivery times to support periodic retraining and ongoing error analysis, following the same segmented-error approach used in `error_insights.md`.
- **Testing:** unit tests on the preprocessing pipeline and integration tests on the API endpoint.
- **CI/CD:** automated testing and deployment on each change to the pipeline or model.

Given the scope of this project, the serving layer and containerization are planned to be implemented as working code, while versioning, monitoring, logging, and CI/CD are described here rather than built out, since standing up that infrastructure without a real operating environment to run it in would add complexity without adding genuine demonstration value.
