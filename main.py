import csv
import pandas as pd
import numpy as np
import sksurv
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_ipcw
import scipy.stats as st
from scipy.stats import sem
from sksurv.metrics import concordance_index_censored


# from sksurv.metrics import concordance_index_censored
# pip install pymatch

# ****************************************************************************************************
# *    DATA PREPROCESSING                                                                                              *
# *                                                                                                  *
# ****************************************************************************************************



# save the file in dataframe in order to be able to deal with it
df = pd.read_csv('war.csv', header=0)
# ignore unnessary  columns and columns that have missing values that are more that %50 of the data
df = df.drop(columns=['gwno_a','gwno_a_2nd', 'gwno_b', 'gwno_b_2nd', 'gwno_loc','version','side_a_2nd','side_b_2nd','ep_end_prec','territory_name'])
# knowing the missing values
missing_data_summary = df.isnull().sum()
# print(missing_data_summary)

#  count the duration column
df["start_date"] = pd.to_datetime(df["start_date"])
df["ep_end_date"] = pd.to_datetime(df["ep_end_date"])
df["duration"]  =((df['ep_end_date'].dt.year - df['start_date'].dt.year) * 12 +
    (df['ep_end_date'].dt.month - df['start_date'].dt.month))

# select only was
data = df[df['intensity_level'] == 2]

# haandle missing data
missing_data_summary = data.isnull().sum()
# print(missing_data_summary)

# data.to_excel('war_data.xlsx', index=False)

# ****************************************************************************************************
# *    FEATURE SELETION                                                                                          *
# *                                                                                                  *
# ****************************************************************************************************
correlation, p_value = spearmanr(data[ 'location'], data['duration'])
# print(f"Spearman correlation coefficient (r_s): {correlation}")
# print(f"P-value: {p_value}")

# ****************************************************************************************************
# *   know which data is actually missing and which one in censored                                                                                      *
# *                                                                                                  *
# ****************************************************************************************************
# missing_ones =  data[data['ep_end_date'].isna()]
# missing_ones.to_excel('output.xlsx', index=False)
# print(missing_ones)
# filtered_df = data[data['ep_end_date'].notna()]
# filtered_df .to_excel('not_missing.xlsx', index=False)
# # ****************************************************************************************************
# # *  use the final data after cleaninig it and modifying it manually                                                                                        *
# # *                                                                                                  *
# # ****************************************************************************************************


# # Read the data
real_data = pd.read_csv('real2.csv', header=0)


# # Convert ep_end_date to datetime

real_data['ep_end_date'] = pd.to_datetime(real_data['ep_end_date'], errors='coerce')

# # Replace NaT values with a default date '31/12/2023'
real_data['ep_end_date'] = real_data['ep_end_date'].fillna(pd.to_datetime('31/12/2023'))

# # Ensure start_date is also in datetime format
real_data['start_date'] = pd.to_datetime(real_data['start_date'], errors='coerce', format='%Y-%m-%d')
# real_data['start_date'] = pd.to_datetime(real_data['start_date'], errors='coerce', format=None)

# # Calculate duration in months
real_data["duration"] = ((real_data['ep_end_date'].dt.year - real_data['start_date'].dt.year) * 12 +
                         (real_data['ep_end_date'].dt.month - real_data['start_date'].dt.month))



# print(real_data['duration'].head(20))
print(real_data)


# ****************************************************************************************************
# *    testing and evaluating                                                                                         *
# *                                                                                                  *
# ****************************************************************************************************

# Assuming 'df' is your DataFrame

# Define the columns to exclude from features

real_data['region'] = real_data['region'].apply(lambda x: int(x.split(',')[0]) if isinstance(x, str) and ',' in x else int(x) if str(x).isdigit() else 0)

real_data['region'] = real_data['region'].astype(int)
exclude_columns = ['duration', 'censored', 'conflict_id', 'start_date2', 'start_date','start_prec2','conflict_id','location','side_a','side_b','side_b_id','side_a_id','ep_end_date','year','ep_end','cumulative_intensity','intensity_level','lose_of_leadership']

# Select feature columns (X) by dropping the exclude_columns
X = real_data.drop(columns=exclude_columns)
non_numeric_columns = X.select_dtypes(include=['object']).columns
print(non_numeric_columns)

# Select target columns (y)
real_data['censored'] = 1 - real_data['censored']
y = real_data[['duration', 'censored']]
# y is a dataframe

# Convert the target columns to a structured array
y = Surv.from_arrays(event=y['censored'], time=y['duration'])

# Initialize the Random Survival Forest model
rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
# Set up K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
c_indices = []
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model on the training data
    rsf.fit(X_train, y_train)

    # Predict on the test data

    y_pred = rsf.predict(X_test)
    c_index = concordance_index_ipcw(y_train, y_test, y_pred)
    c_indices.append(c_index[0])



result = permutation_importance(rsf, X_test, y_test, n_repeats=10, random_state=42)

# Display feature importance
print("Feature importances:")
for i, importance in enumerate(result.importances_mean):
    print(f"{X.columns[i]}: {importance:.4f}")
	 # Calculate C-index for each fold
event = y_test['event']  # Access the 'event' field
time = y_test['time']

# Calculate the concordance index for the current fold


# Calculate the average concordance index across all folds
mean_c_index = np.mean(c_indices)
print(f'Mean Concordance Index: {mean_c_index}')
from scipy.stats import sem, t

# Your existing code where you calculate mean C-index
mean_c_index = np.mean(c_indices)

from scipy.stats import t
import numpy as np
from scipy.stats import sem

# Calculate the standard error of the mean for c_indices
mean_c_index = np.mean(c_indices)
print(f'Concordance Indices: {c_indices}')

# Ensure there are enough data points
if len(c_indices) > 1:
    standard_error = sem(c_indices)  # Standard error
    df = len(c_indices) - 1  # Degrees of freedom

    # Set alpha for 95% confidence interval (alpha = 0.05)
    alpha = 0.05

    # Calculate the confidence interval using t-distribution
    ci_low, ci_high = t.interval(1 - alpha, df, loc=mean_c_index, scale=standard_error)

    print(f'95% Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})')
else:
    print("Not enough data points for a valid confidence interval.")
train_c_index = concordance_index_censored(y_train['event'], y_train['time'], rsf.predict(X_train))[0]

# Validation performance
val_c_index = concordance_index_censored(
    y_test['event'], y_test['time'], rsf.predict(X_test)
)[0]

print(f"Training C-Index: {train_c_index:.3f}")
print(f"Validation C-Index: {val_c_index:.3f}")
# import pandas as pd

# # Assuming 'rsf' is your trained Random Survival Forest model
# # And 'new_data' is the new observation in the same format as your training data
train_columns = X.columns
print(train_columns)
new_data = pd.DataFrame({
	'incompatibility': [2],
    'type_of_conflict': [2],
    'start_prec': [5],
    'region': [1],
	'agreement_or_ceasfire': [0],
	'victory_of_one_side':[1],
	'external_intervention': [1],

})

# Predict the duration for the new row (event duration)
predicted_duration = rsf.predict(new_data)
print(f"Predicted Duration: {predicted_duration}")













# If needed, convert X and y to NumPy arrays

 # Replace 'accuracy' with your preferred metric















