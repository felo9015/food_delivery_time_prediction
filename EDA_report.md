# EDA Report

Consolidates the exploratory analysis of `data/Food_Delivery_Times.csv` (`notebooks/eda_exploration.ipynb`) that drives the preprocessing and modeling decisions downstream.

## Data Overview

| | |
|---|---|
| Shape | 1,000 rows × 9 columns |
| Target | `Delivery_Time_min` (int) |
| Features | `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs` (numeric); `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type` (categorical); `Order_ID` (identifier, no predictive value) |
| Duplicates | 0 |

## Missing Data

Four columns have nulls, each at exactly 3.0% (30/1,000): `Weather`, `Traffic_Level`, `Time_of_Day`, `Courier_Experience_yrs`. Overlap is minimal — 117 rows have at least one null, 0 rows have all four, only 3 rows have two at once — so missingness is not concentrated in a shared subset of rows.

| Column | Classification | Evidence |
|---|---|---|
| `Weather` | MCAR | Target mean nearly identical for null vs. non-null rows (54.2 vs. 56.8 min); null rate flat across `Traffic_Level` and `Vehicle_Type` |
| `Traffic_Level` | MAR (leaning) | Null rows average well above non-null rows; null rate roughly doubles across `Vehicle_Type` (`Bike` → `Scooter`) |
| `Time_of_Day` | MAR | Clearest signal of the four: null rows average ~10 min higher with higher variance; elevated null rate under `Windy` weather and `Low` traffic |
| `Courier_Experience_yrs` | MCAR | Target mean virtually unchanged between null and non-null rows; category-level null-rate spread looks like noise, not a pattern |

MNAR cannot be fully ruled out for `Traffic_Level`/`Time_of_Day` (e.g., a value withheld because the underlying condition was severe), but this is untestable since the true missing values are unobserved — the MAR label here reflects only the dependence on observed variables that could actually be checked.

**Treatment:** `Weather`, `Traffic_Level`, `Time_of_Day` → constant `"Unknown"` category; `Courier_Experience_yrs` → median, no missingness indicator.

- `"Unknown"` works across both the MCAR and MAR-leaning columns: it assumes no specific true category, and, by becoming its own encoded level, implicitly preserves any signal carried by the missingness itself (relevant for the two MAR columns) without needing a separate indicator.
- `Courier_Experience_yrs` showed no such signal (MCAR), so a plain median fill is simpler and equally defensible.
- Imputation itself is **not** performed here — it is deferred to the `scikit-learn` pipeline so fitted statistics (e.g. the median) are computed only on the training split, avoiding leakage from validation/test rows.

## Key Patterns

**Distributions.** `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs` are all close to symmetric (|skew| ≤ 0.04). `Delivery_Time_min` has moderate right skew (≈0.51): mean sits above median, with a tail toward slow deliveries.

**Numeric correlations with target:**

| Feature | r vs. `Delivery_Time_min` |
|---|---|
| `Distance_km` | ≈ 0.78 |
| `Preparation_Time_min` | ≈ 0.31 |
| `Courier_Experience_yrs` | ≈ -0.09 (negligible) |

**Categorical effects on target:**

- `Weather` — largest effect: `Snowy` far above `Clear` (~14 min gap); `Rainy` also elevated, by a smaller margin.
- `Traffic_Level` — ordered gradient: `High` > `Medium` > `Low` (~12 min spread).
- `Time_of_Day`, `Vehicle_Type` — no notable effect (group means within a ~2 min band for both).

## Outliers

IQR method (below `Q1-1.5×IQR` or above `Q3+1.5×IQR`): 0 outliers in `Distance_km` and `Preparation_Time_min`; 6 in `Delivery_Time_min`, all above the upper bound (~116 min). None removed at this stage — kept for the modeling stage to decide once it's clear whether they reflect plausible variable combinations or data issues.

## Correlation Summary

The three numeric features are essentially uncorrelated with each other (|r| ≤ 0.03 for every pair), so there is no multicollinearity concern among them going into modeling.

## Assumptions

- Dataset treated as a complete, self-contained snapshot; no external joins; `Order_ID` assumed to be a plain identifier.
- The MCAR/MAR classification is descriptive (full dataset, pre-split); it informs the imputation *strategy*, but fitted values (e.g. the median) are still computed only on the training split, inside the pipeline.
- The 6 IQR outliers in `Delivery_Time_min` are assumed plausible extreme deliveries, not confirmed errors — no evidence (e.g. impossible values) suggests otherwise.
- `Time_of_Day` is treated as categorical (four labeled periods), consistent with the source data — no assumption is made about the exact clock time each label covers.
