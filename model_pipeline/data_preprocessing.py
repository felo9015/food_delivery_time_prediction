"""Preprocessing pipeline: imputation and encoding for the raw features.

Validated in notebooks/model_exploration.ipynb against the strategy
decided in EDA_report.md.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from model_pipeline.config import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the ColumnTransformer used to impute and encode the raw features.

    Numeric features are median-imputed. Traffic_Level (ordinal) and the
    nominal categoricals are imputed with a constant "Unknown" category,
    then Traffic_Level is ordinal-encoded and the nominal features are
    one-hot encoded. The transformer is returned unfit; the caller is
    responsible for fitting it only on training data.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(categories=ORDINAL_CATEGORIES)),
    ])

    # drop="first" avoids the dummy variable trap: with a constant term already
    # in the model, encoding every category would make each nominal column's
    # dummies sum to exactly 1, an exact linear dependency with the intercept
    # (perfect multicollinearity). Dropping one category per column makes it
    # the implicit reference level, which is also what makes the remaining
    # dummy coefficients and p-values interpretable.
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ("nominal", nominal_pipeline, NOMINAL_FEATURES),
    ])
