import csv
import pandas as pd

# save the file in dataframe in order to be able to deal with it
df = pd.read_csv('interstate.csv', header=0)
# select only the rows that have war
war_data = df[df['hostlev'] == 5]
# create the time colume
war_data['duartion'] = war_data['endyear']  - war_data['styear']
print(war_data['duartion'])  # Correct attribute to access column names in a Pandas DataFrame




