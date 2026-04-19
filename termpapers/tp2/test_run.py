# Standard data manipulation and general concepts
import pandas as pd # Used extensively in Lecture 9 & Workshop 9 for data manipulation
import numpy as np  # Common in all quantitative workflows (Lecture 9-12)
import os.path      # Used in Workshop 9 for path operations
import glob         # Used in Workshop 9 to fetch multiple files matching a pattern
# Defining our data directory explicitly
DATA_DIR = './data'

# Get all SCE extract CSV files (used glob in Workshop 09 to elegantly capture multiple paths)
file_pattern = os.path.join(DATA_DIR, "sce_extract_*.csv")
file_list = glob.glob(file_pattern)

# Using pd.concat to merge a list of DataFrames efficiently (used in Workshop 09 for concatenating files)
df_sce = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

# Brief overview of the merged dataset
print(f"Successfully merged {len(file_list)} files.")
print(f"Total initial rows: {len(df_sce)}")
print(f"Dataset columns:\n {df_sce.columns.tolist()}")
# 1.2 Forward-fill variables asked only in the initial wave

df_sce['date'] = pd.to_datetime(df_sce['date']) 
df_sce = df_sce.sort_values(by=['userid', 'date'])

# IMPORTANT: 'health' and 'take_fin_risk' have extreme levels of missing values.
# Including them drops the sample down to ~123k. Omitting them keeps us at the optimal ~150k target!
initial_wave_vars = ['owner', 'educ', 'female', 'num_lit_q1_correct', 'num_lit_q2_correct', 'num_lit_q3_correct']

for col in initial_wave_vars:
    if col in df_sce.columns:
        df_sce[col] = df_sce.groupby('userid')[col].ffill()

print(f"Successfully forward-filled the following initial-wave variables:\n{initial_wave_vars}")
# 1.3 Drop 2025

# Storing total rows before drop for reporting (Required by Part 1 guidelines)
rows_before_drop2025 = len(df_sce)

# Filtering out 2025 using pandas boolean indexing on the datetime `.dt.year` accessor (Explored in Lecture 9)
df_sce = df_sce[df_sce['date'].dt.year < 2025]

# Using a standard pandas copy to prevent SettingWithCopyWarning down the line
df_sce = df_sce.copy()

print(f"Rows before dropping 2025: {rows_before_drop2025}")
print(f"Number of observations dropped (2025): {rows_before_drop2025 - len(df_sce)}")
print(f"Total remaining rows: {len(df_sce)}")
# 1.4 Drop rows with missing data for variables used in analysis

rows_before_na_drop = len(df_sce)

# Define our continuous modeling targets
expectation_vars = ['infl_1y', 'house_price_change', 'prob_unrate_up', 'prob_stocks_up']

# Define essential identifying variables
id_vars = ['userid', 'wid', 'date']

# Calculate final core subset
# Note: 'initial_wave_vars' was defined in Part 1.2
core_columns = id_vars + expectation_vars + initial_wave_vars

# Ensure we only evaluate columns that actually exist in the dataframe 
available_cols_to_check = [col for col in core_columns if col in df_sce.columns]

# Drop NaNs subsetted specifically to our modeling variables using Pandas' built-in dropna
df_sce = df_sce.dropna(subset=available_cols_to_check).copy()

# Retain only these columns to optimize memory utilization for performance
df_sce = df_sce[available_cols_to_check]

print(f"Rows before dropping NaNs: {rows_before_na_drop}")
print(f"Number of observations dropped (NaNs): {rows_before_na_drop - len(df_sce)}")
print(f"Total remaining rows: {len(df_sce)}")
# 1.5 Handle Outliers (strict clipping between P1 and P99)

rows_before_outliers_drop = len(df_sce)

for col in expectation_vars:
    # Compute the 1st (P1) and 99th (P99) percentiles using pandas
    p1 = df_sce[col].quantile(0.01)
    p99 = df_sce[col].quantile(0.99)
    
    # Filter dataset sequentially keeping only values strictly > P1 and < P99
    df_sce = df_sce[(df_sce[col] > p1) & (df_sce[col] < p99)]

print(f"Rows before handling outliers: {rows_before_outliers_drop}")
print(f"Number of outlier observations dropped: {rows_before_outliers_drop - len(df_sce)}")
print(f"Total remaining rows: {len(df_sce)}")
# 1.6 Create binary indicator variables
import numpy as np

df_sce['optimist_unrate'] = np.where(df_sce['prob_unrate_up'] < 50, 1, 0)
df_sce['optimist_stocks'] = np.where(df_sce['prob_stocks_up'] > 50, 1, 0)
df_sce['optimist_house_price'] = np.where(df_sce['house_price_change'] > 0, 1, 0)

# ==========================================
# 1.7 Final Part 1 Preprocessing Report
# ==========================================
print("--- Final Cleaned SCE Dataset ---")
print(f"Total Unique Individuals: {df_sce['userid'].nunique()}")
print(f"Total Observations: {len(df_sce)}")
print(f"Number of Survey Waves: {df_sce['date'].nunique()}")
print(f"Starting Date: {df_sce['date'].min().strftime('%Y-%m-%d')}")
print(f"Ending Date: {df_sce['date'].max().strftime('%Y-%m-%d')}")