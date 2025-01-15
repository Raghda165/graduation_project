import csv
import pandas as pd
import numpy as np
import sksurv
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


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
df["duration"]  =( (df['ep_end_date'].dt.year - df['start_date'].dt.year) * 12 +
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
real_data = pd.read_csv('real1.csv', header=0)


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
exclude_columns = ['duration', 'censored', 'conflict_id', 'start_date2', 'start_date','start_prec2','conflict_id','location','side_a','side_b','side_b_id','side_a_id','ep_end','ep_end_date','region']

# Select feature columns (X) by dropping the exclude_columns
X = real_data.drop(columns=exclude_columns)
non_numeric_columns = X.select_dtypes(include=['object']).columns
print(non_numeric_columns)

# Select target columns (y)
y = real_data[['duration', 'censored']]

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

    # Calculate the concordance index for the current fold
    c_index = concordance_index_ipcw(y_test, y_pred)
    c_indices.append(c_index)

# Calculate the average concordance index across all folds
mean_c_index = np.mean(c_indices)
print(f'Mean Concordance Index: {mean_c_index}')






# If needed, convert X and y to NumPy arrays

 # Replace 'accuracy' with your preferred metric















