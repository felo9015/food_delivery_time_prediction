"""Reusable diagnostic functions for fitted statsmodels OLS models.

Each function takes an already-fitted statsmodels results object (or the
data needed alongside it) so the same functions can be reused across
different model versions -- baseline, with interactions, etc. -- without
being tied to any one model's specifics.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_residuals_vs_fitted(model):
    """Scatter plot of residuals against fitted values, with a zero line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(model.fittedvalues, model.resid, alpha=0.4, s=15)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted")
    return fig, ax


def qq_plot(model):
    """Q-Q plot of the model's residuals against a normal distribution."""
    fig = qqplot(model.resid, line="45", fit=True)
    fig.suptitle("Q-Q Plot of Residuals")
    return fig


def breusch_pagan_test(model):
    """Breusch-Pagan test for heteroscedasticity in the model's residuals.

    Returns the Lagrange multiplier statistic and p-value (plus the
    equivalent F-statistic and its p-value) as a dict. A small p-value is
    evidence against the null of homoscedasticity (constant residual
    variance).
    """
    lm_stat, lm_p_value, f_stat, f_p_value = het_breuschpagan(model.resid, model.model.exog)
    return {
        "lm_statistic": lm_stat,
        "lm_p_value": lm_p_value,
        "f_statistic": f_stat,
        "f_p_value": f_p_value,
    }


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per column of X.

    A constant is added internally for the auxiliary regressions (the
    standard convention VIF is defined under) and then dropped from the
    returned table, since a VIF for the intercept itself is not
    meaningful.
    """
    X_with_const = sm.add_constant(X, has_constant="add")
    vif_table = pd.DataFrame({
        "feature": X_with_const.columns,
        "VIF": [
            variance_inflation_factor(X_with_const.values, i)
            for i in range(X_with_const.shape[1])
        ],
    })
    return vif_table[vif_table["feature"] != "const"].reset_index(drop=True)


def get_aic(model) -> float:
    """AIC of the fitted model, for comparing different model versions."""
    return model.aic


def compute_test_metrics(y_true, y_pred) -> dict:
    """MAE and RMSE between true and predicted values on a held-out set."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}
