# EDA Report

This report consolidates the exploratory data analysis performed on `data/Food_Delivery_Times.csv` in `notebooks/eda_exploration.ipynb`. It summarizes the key patterns, data-quality issues, and decisions that carry forward into preprocessing and modeling.

## Data Overview

- **Shape:** 1,000 rows × 9 columns.
- **Data types:** `Order_ID` (int), `Distance_km` (float), `Weather` (categorical), `Traffic_Level` (categorical), `Time_of_Day` (categorical), `Vehicle_Type` (categorical), `Preparation_Time_min` (int), `Courier_Experience_yrs` (float), `Delivery_Time_min` (int, target).
- **Duplicates:** 0 duplicate rows.

## Missing Data

Four columns have nulls, each with exactly 30 missing values (3.0% of rows): `Weather`, `Traffic_Level`, `Time_of_Day`, and `Courier_Experience_yrs`. The remaining five columns, including the target, have no missing values.

**Overlap across the 4 columns is minimal.** 117 rows have a null in at least one of the four columns, but 0 rows are null in all four simultaneously: 114 rows have exactly one null, and only 3 rows have two nulls at once. Missingness in these columns is effectively independent rather than concentrated in a shared subset of rows.

**Missingness classification, by column:**

| Column | Classification | Evidence |
|---|---|---|
| `Weather` | MCAR | `Delivery_Time_min` is nearly identical between null and non-null rows (54.2 vs. 56.8 min). The null rate is flat across `Traffic_Level` (2.3-3.8%) and `Vehicle_Type` (1.8-4.6%), with no consistent split. |
| `Traffic_Level` | MAR (leaning) | Rows with a null average 62.1 min vs. 56.6 min for the rest — a gap tied to an observed variable. The null rate also roughly doubles across `Vehicle_Type`, from `Bike` (2.0%) to `Scooter` (4.3%). |
| `Time_of_Day` | MAR | The clearest signal of the four: null rows average 66.2 min vs. 56.4 min for the rest, with much higher variance (std ≈ 33.7 vs. 21.6). The null rate is also elevated under `Windy` weather (6.2% vs. 2-3% for other conditions) and `Low` traffic (4.2% vs. 2.0-2.3%). |
| `Courier_Experience_yrs` | MCAR | `Delivery_Time_min` is virtually unchanged between null and non-null rows (56.4 vs. 56.7 min). Category-level null-rate spread (e.g. `Vehicle_Type`: `Car` 1.5% vs. `Scooter` 4.0%) is inconsistent across candidates and looks like noise rather than a real pattern. |

For `Traffic_Level` and `Time_of_Day`, MNAR cannot be fully ruled out (e.g. a value failing to be logged specifically because the underlying condition was severe), but this cannot be tested directly, since the true missing values are unobserved. The evidence collected only supports dependence on observed variables, which is the working definition of MAR used here.

**Treatment strategy.** `Weather`, `Traffic_Level`, and `Time_of_Day` will be imputed with a constant `"Unknown"` category; `Courier_Experience_yrs` will be imputed with the simple median, with no missingness-indicator column added. The `"Unknown"` category is a reasonable default across both the MCAR (`Weather`) and MAR-leaning (`Traffic_Level`, `Time_of_Day`) columns: it does not assume a specific true category, and — since it becomes its own level in the encoded feature — it implicitly preserves any signal carried by the missingness itself (relevant for the two MAR columns) without requiring a separate indicator. For `Courier_Experience_yrs`, no such signal was found (MCAR), so a plain median fill is the simpler, equally defensible choice.

This imputation is **not** implemented in this notebook. It is deferred to a `scikit-learn` `Pipeline`/`ColumnTransformer` in the preprocessing stage, where the median (and any other fitted statistic) is computed only on the training split and then applied to validation/test data — computing it on the full dataset here, before the train/test split exists, would leak information from validation/test rows into training.

## Key Patterns

**Univariate distributions.** `Distance_km` (skew ≈ 0.04), `Preparation_Time_min` (≈ 0.03), and `Courier_Experience_yrs` (≈ -0.03) are all close to symmetric. `Delivery_Time_min`, the target, has a moderate right skew (≈ 0.51): its mean (56.7 min) sits above its median (55.5 min), with a longer tail toward slow deliveries.

**Numeric variables vs. target.** `Distance_km` has by far the strongest relationship with `Delivery_Time_min` (r ≈ 0.78). `Preparation_Time_min` is moderately correlated (r ≈ 0.31). `Courier_Experience_yrs` has a weak negative correlation (r ≈ -0.09) — more experienced couriers trend very slightly faster, but the effect is practically negligible.

**Categorical variables vs. target:**

- **`Weather`** shows the largest categorical effect: `Snowy` has the highest average delivery time (67.11 min) vs. `Clear` (53.08 min), a difference of ~14 min. `Rainy` is also elevated (59.79 min, ~6.7 min above `Clear`), though by a smaller margin than `Snowy`.
- **`Traffic_Level`** shows a clear, ordered gradient: `High` (64.81 min) > `Medium` (56.02 min) > `Low` (52.89 min), an ~11.9 min spread.
- **`Time_of_Day`** and **`Vehicle_Type`** show no notable differences between categories — all group means fall within a ~2.3 min band for `Time_of_Day` (55.21-57.48) and a ~2.2 min band for `Vehicle_Type` (56.05-58.20).

## Outliers

Using the IQR method (values below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR`):

- `Distance_km`: 0 outliers.
- `Preparation_Time_min`: 0 outliers.
- `Delivery_Time_min`: 6 outliers, all above the upper bound (~116 min).

None of these were removed or modified at this stage. Their presence is reported for downstream use — the decision on how to handle them (keep, cap, or exclude) belongs to the modeling stage, once it is clear whether they correspond to plausible combinations of the other variables or look more like data issues.

## Correlation Summary

The three numeric features are essentially uncorrelated with each other (|r| ≤ 0.03 for every pair among `Distance_km`, `Preparation_Time_min`, and `Courier_Experience_yrs`). This means there is no multicollinearity concern among them going into modeling — each contributes largely independent information.

## Assumptions

- The dataset is treated as a complete, self-contained snapshot; no external data was joined, and `Order_ID` is assumed to be a plain identifier with no predictive value.
- The MCAR/MAR classification above is descriptive, based on the full dataset before any train/test split. It informs the imputation *strategy* chosen, but the actual fitted values (e.g. the median for `Courier_Experience_yrs`) must still be computed only on the training split, inside the preprocessing pipeline.
- The 6 IQR outliers in `Delivery_Time_min` are assumed to be plausible extreme deliveries rather than confirmed data errors, since no evidence (e.g. impossible values) was found to suggest otherwise; they are kept as-is pending a modeling-stage decision.
- `Time_of_Day` is treated as a categorical variable (four labeled periods) rather than a continuous time value, consistent with how it is provided in the source data — no assumption is made about the exact clock time each label corresponds to.
