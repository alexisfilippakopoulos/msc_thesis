import os
import ast
import pandas as pd

# Set the base path and the specific CSV files to parse
base_path = 'results/mmgpt_debug'
specific_files = ['results_5.csv', 'results_6.csv']
prefixes = \
    [
     'mmgpt-mosei-v0',
     'mmgpt-mosei-v1',
     'mmgpt-mosei-v2',
     'mmgpt-mosei-v3'
     ]


# Function to parse the tuple and return the mean
def parse_mean(value):
    try:
        return ast.literal_eval(value)[0]
    except:
        return None  # Returns None if the value cannot be parsed


# Define the function to filter the DataFrame based on given conditions and file type
def filter_results(df, file_name):
    if file_name == 'results_5.csv':
        df['Non0_acc_2_Avg5'] = df['Non0_acc_2_Avg5'].apply(parse_mean)
        df['Mult_acc_7_Avg5'] = df['Mult_acc_7_Avg5'].apply(parse_mean)
        return df[(df['Non0_acc_2_Avg5'] >= 86.10) & (df['Mult_acc_7_Avg5'] >= 54.00)]
    elif file_name == 'results_6.csv':
        df['Non0_acc_2_Avg6'] = df['Non0_acc_2_Avg6'].apply(parse_mean)
        df['Mult_acc_7_Avg6'] = df['Mult_acc_7_Avg6'].apply(parse_mean)
        return df[(df['Non0_acc_2_Avg6'] >= 86.10) & (df['Mult_acc_7_Avg6'] >= 54.00)]


# Search through the directories starting at the base path
for root, dirs, files in os.walk(base_path):
    # Check if the current directory name starts with any of the specified prefixes
    if any(root.split(os.sep)[-1].startswith(prefix) for prefix in prefixes):
        # For each specific file, check if it exists in the 'normal' subdirectory
        for file_name in specific_files:
            file_path = os.path.join(root, 'normal', file_name)
            if os.path.isfile(file_path):  # Check if the file exists
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    # Check if required columns are present based on file type
                    required_columns = {'Non0_acc_2_Avg5', 'Mult_acc_7_Avg5'} if file_name == 'results_5.csv' else {'Non0_acc_2_Avg6', 'Mult_acc_7_Avg6'}
                    if not required_columns.issubset(df.columns):
                        print(f"Header missing in file: {file_path}")
                        continue
                    # Filter the DataFrame based on the conditions
                    filtered_df = filter_results(df, file_name)
                    # Print the results if there are any rows after filtering
                    if not filtered_df.empty:
                        print(f"Model Folder: {root}")
                        print(filtered_df)
                        print("\n")
                except pd.errors.EmptyDataError:
                    print(f"No data in file: {file_path}")

# Note: replace '<model_name_variable>' with the actual variable or parameter defining the model names.
