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
from scipy.stats import t
import numpy as np
from scipy.stats import sem


# ****************************************************************************************************
# *    DATA PREPROCESSING                                                                                              *
# *                                                                                                  *
# ****************************************************************************************************



# save the file in dataframe in order to be able to deal with it
df = pd.read_csv('war.csv', header=0)
# ignore unnessary  columns and columns that have missing values that are more that %50 of the data
df = df.drop(columns=['gwno_a','gwno_a_2nd', 'gwno_b', 'gwno_b_2nd', 'gwno_loc','version','side_a_2nd','side_b_2nd','ep_end_prec','territory_name'])

#  count the duration column
df["start_date"] = pd.to_datetime(df["start_date"])
df["ep_end_date"] = pd.to_datetime(df["ep_end_date"])
df["duration"]  =((df['ep_end_date'].dt.year - df['start_date'].dt.year) * 12 +
    (df['ep_end_date'].dt.month - df['start_date'].dt.month))


data = df[df['intensity_level'] == 2]

# haandle missing data
missing_data_summary = data.isnull().sum()
# print(missing_data_summary)
# working on data manually
# data.to_excel('war_data.xlsx', index=False)

# # ****************************************************************************************************
# # *  use the final data after cleaninig it and modifying it manually                                                                                        *
# # *                                                                                                  *
# # ****************************************************************************************************


# # Convert ep_end_date to datetime
real_data = pd.read_csv('real2.csv', header=0)
real_data['ep_end_date'] = pd.to_datetime(real_data['ep_end_date'], errors='coerce')

# # Replace NaT values with a default date '31/12/2023'
real_data['ep_end_date'] = real_data['ep_end_date'].fillna(pd.to_datetime('31/12/2023'))

# # Ensure start_date is also in datetime format
real_data['start_date'] = pd.to_datetime(real_data['start_date'], errors='coerce', format='%Y-%m-%d')
real_data['start_date'] = pd.to_datetime(real_data['start_date'], errors='coerce', format=None)

# # Calculate duration in months
real_data["duration"] = ((real_data['ep_end_date'].dt.year - real_data['start_date'].dt.year) * 12 +
                         (real_data['ep_end_date'].dt.month - real_data['start_date'].dt.month))
# ****************************************************************************************************
# *    FEATURE SELETION                                                                                          *
# *                                                                                                  *
# ****************************************************************************************************

print(real_data.columns)
correlation, p_value = spearmanr(real_data['victory_of_one_side'], real_data['duration'])
print(f"Spearman correlation coefficient (r_s): {correlation}")
print(f"P-value: {p_value}")


# # print(real_data['duration'].head(20))
# print(real_data)


# # ****************************************************************************************************
# # *    testing and evaluating and train the model                                                                                      *
# # *    and model evaluation                                                                                              *
# # ****************************************************************************************************

# ****** prepare the predictors(x)*******
exclude_columns = ['duration', 'censored', 'conflict_id', 'start_date2', 'start_date','start_prec2','conflict_id','location','side_a','side_b','side_a_id','side_b_id','ep_end_date','year','ep_end','cumulative_intensity','intensity_level','lose_of_leadership','incompatibility','start_prec']
real_data['region'] = real_data['region'].apply(lambda x: int(x.split(',')[0]) if isinstance(x, str) and ',' in x else int(x) if str(x).isdigit() else 0)
real_data['side_b_id'] = real_data['side_b_id'].apply(lambda x: int(x.split(',')[0]) if isinstance(x, str) and ',' in x else int(x) if str(x).isdigit() else 0)
real_data['region'] = real_data['region'].astype(int)
real_data['side_b_id'] = real_data['side_b_id'].astype(int)
X = real_data.drop(columns=exclude_columns)

# ****** prepare the response(y)*******
real_data['censored'] = 1 - real_data['censored']
y = real_data[['duration', 'censored']]
y = Surv.from_arrays(event=y['censored'], time=y['duration'])

#*********Initialize the Random Survival Forest model*****
rsf = RandomSurvivalForest(n_estimators=100, random_state=42)
#*********Initialize the folds and prepare the arrays that will hold the c_index values for evluation*****
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
c_indices = []
train_c_indices = []
val_c_indices = []

#*********train the Random Survival Forest model using k_fold*****
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rsf.fit(X_train, y_train)
    y_pred = rsf.predict(X_test)
    y_train_pred = rsf.predict(X_train)
    y_test_pred = rsf.predict(X_test)
    c_index = concordance_index_ipcw(y_train, y_test, y_pred)
    train_c_index = concordance_index_censored(y_train['event'], y_train['time'], y_train_pred)[0]
    val_c_index = concordance_index_censored(y_test['event'], y_test['time'], y_test_pred)[0]
    train_c_indices.append(train_c_index)
    val_c_indices.append(val_c_index)
    c_indices.append(c_index[0])


# **********************determine the importance for each feature*****
result = permutation_importance(rsf, X_test, y_test, n_repeats=10, random_state=42)
for i, importance in enumerate(result.importances_mean):
    print(f"{X.columns[i]}: {importance:.4f}")


#*************************the evaluation***********
mean_c_index = np.mean(c_indices)
print(f'Mean Concordance Index: {mean_c_index}')
avg_train_c_index = np.mean(train_c_indices)
avg_val_c_index = np.mean(val_c_indices)

#*************************** Assess overfitting***************
if avg_train_c_index - avg_val_c_index > 0.05:
    print("The model might be overfitting. Consider regularization or tuning hyperparameters.")
else:
    print("The model generalizes well across the folds.")

#**************************  ❤️ predict for sudan ❤️  ***************
new_data = pd.DataFrame({
    'type_of_conflict': [2],
    'region': [1],
	'agreement_or_ceasfire': [0],
	'victory_of_one_side':[1],
	'external_intervention': [1],
})
predicted_duration = rsf.predict(new_data)
print(f"Predicted Duration for Sudan war: {predicted_duration} months")
print(f"Average Training C-Index: {avg_train_c_index:.3f}")
print(f"Average Validation C-Index: {avg_val_c_index:.3f}")

