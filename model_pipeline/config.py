"""Shared constants for the preprocessing and modeling pipeline.

Feature groups and the split settings validated in
notebooks/model_exploration.ipynb and documented in EDA_report.md.
"""

NUMERIC_FEATURES = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]

# Traffic_Level is genuinely ordinal (Low < Medium < High), but "Unknown" is
# missing information, not a fourth traffic level. It is placed after High
# rather than before Low or in the middle because rows with a missing
# Traffic_Level had a higher average delivery time (62.1 min) than the
# overall average (56.6 min) in the EDA -- closer in direction to High
# traffic than to Low. This is a judgment call, not a fact derived from the
# missing values themselves (which are, by definition, unobserved).
ORDINAL_FEATURES = ["Traffic_Level"]
ORDINAL_CATEGORIES = [["Low", "Medium", "High", "Unknown"]]

NOMINAL_FEATURES = ["Weather", "Time_of_Day", "Vehicle_Type"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
