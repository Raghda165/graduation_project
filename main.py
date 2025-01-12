import csv
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
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
df["duration"]  = df['ep_end_date'].dt.year - df['start_date'].dt.year

# select only was
data = df[df['intensity_level'] == 2]

# haandle missing data
missing_data_summary = data.isnull().sum()
# print(missing_data_summary)

# select only the rows that the end data is existing for them
basic_data = data[data['ep_end_date'].notna()]
# print(basic_data['type_of_conflict'])

# ****************************************************************************************************
# *    FEATURE SELETION                                                                                          *
# *                                                                                                  *
# ****************************************************************************************************
correlation, p_value = spearmanr(basic_data[ 'location'], basic_data['duration'])
# print(f"Spearman correlation coefficient (r_s): {correlation}")
# print(f"P-value: {p_value}")


# ****************************************************************************************************
# *    testing and evaluating                                                                                         *
# *                                                                                                  *
# ****************************************************************************************************














