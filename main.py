import csv
import pandas as pd


# save the file in dataframe in order to be able to deal with it
df = pd.read_csv('war.csv', header=0)
df = df.drop(columns=['gwno_a','gwno_a_2nd', 'gwno_b', 'gwno_b_2nd', 'gwno_loc','version'])

df["start_date"] = pd.to_datetime(df["start_date"])
df["ep_end_date"] = pd.to_datetime(df["ep_end_date"])
df = df.drop_duplicates()
df["duration"] = df["duration"] = df['ep_end_date'].dt.year - df['start_date'].dt.year
# print(df['ep_end_date'])
# print(df['start_date'])
filtered_data = df[df
['intensity_level'] == 2]
print(filtered_data)
# unique_values = df['conflict_id'].unique()






# print(df.head(10))





