# import openpyxl
# import csv
# from groq import Groq

# client = Groq(api_key="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")

# def read_all_rows(file_path):
#     """Read all rows from the given Excel file."""
#     # Load the workbook and select the active sheet
#     size = 3
#     wb = openpyxl.load_workbook(file_path)
#     sheet = wb.active

#     # Initialize a list to store all rows
#     all_rows = []

#     # Iterate through each row in the sheet
#     for row in sheet.iter_rows(min_row=1, max_row=sheet.max_row, max_col=sheet.max_column, values_only=True):
#         if size <= 0:
#             break
#         size -= 1
#         all_rows.append(list(row))

#     # Close the workbook
#     wb.close()

#     return all_rows

# def generate_new_row(prompt):
#     """Generate a new row using the OpenAI API (gpt-3.5-turbo) based on the given prompt."""
#     response = client.chat.completions.create(
#         model="llama3-70b-8192",
#         messages=[
#             {"role": "system", "content": "You are a highly efficient assistant for generating rows of data for a spreadsheet. Provide realistic and accurate entries using dates no later than 2023. Ensure all fields are separated by commas without unnecessary quotes or formatting. Respond only with the one raw of data, adhering strictly to the format provided in the example. Each row should represent a unique conflict, aligning with historical accuracy and logical coherence in the data, rember just print relavent data separated by commas without unnecessary quotes"},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=100,
#         n=1
#     )
#     generated_text = response.choices[0].message.content.split("\n\n")[-1]
#     return generated_text.split(",")  # Assuming output is comma-separated values

# def append_new_row(file_path, new_row):
#     """Append the new row to the Excel file."""
#     wb = openpyxl.load_workbook(file_path)
#     sheet = wb.active
#     sheet.append(new_row)
#     wb.save(file_path)
#     wb.close()

# def main():
#     file_path = 'real2.xlsx'
#     last_row_data = read_all_rows(file_path)
#     print(last_row_data)
#     n = int(input("Enter the number of new rows to generate: "))
#     for i in range(n):
#         last_row_data = read_all_rows(file_path)
#         # print(last_row_data)
#         prompt = f"Generate a new row based on this data: {last_row_data}"
#         new_row = generate_new_row(prompt)
#         print(f"New row generated: {new_row}")
#         append_new_row(file_path, new_row)
#         print(f"Row {i+1} appended successfully.")

#     print(f"{n} new rows have been added to {file_path}.")

# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sdv.tabular import CTGAN
from DataSynthesizer.DataTransformers import CategoricalTransformer

# Load data from Excel file
file_path = "your_file.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path)

# Display first few rows of the data
print("Original Data:")
print(df.head())

# Option 1: SMOTE for Synthetic Data Generation (useful for imbalanced datasets)
def generate_data_smote(df):
    X = df.drop(columns=["target_column"])  # Replace "target_column" with your target column name
    y = df["target_column"]

    smote = SMOTE(sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)

    synthetic_df = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_df["target_column"] = y_resampled

    return synthetic_df

# Option 2: CTGAN for Synthetic Data Generation (for complex structured data)
def generate_data_ctgan(df):
    model = CTGAN()
    model.fit(df)
    synthetic_data = model.sample(len(df))
    return synthetic_data

# Option 3: DataSynthesizer for Synthetic Data Generation (handles structured tabular data)
def generate_data_datasynthesizer(df):
    transformer = CategoricalTransformer()
    transformer.fit(df)
    synthetic_data = transformer.generate_data(len(df))
    return synthetic_data

# Choose which method to use:
# 1. Uncomment the following line to use SMOTE
# synthetic_data = generate_data_smote(df)

# 2. Uncomment the following line to use CTGAN
synthetic_data = generate_data_ctgan(df)

# 3. Uncomment the following line to use DataSynthesizer
# synthetic_data = generate_data_datasynthesizer(df)

# Save the generated synthetic data to a new Excel file
output_file = "synthetic_data.xlsx"  # You can change the file name
synthetic_data.to_excel(output_file, index=False)

print(f"Synthetic data has been saved to {output_file}")

