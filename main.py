# ----------------------------------------------------------------------------------------------------------------
# Task 2: Identifying alternative materials
# Author: Simon Markthaler
# Date: 2025-07-14
#
# main:
# > Main script for data preprocessing and training the algorithm.
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------Data Preprocessing-----------------------------------------------
def load_data():
    """   
    Load the dataset.
    """
    # Load the datasets and concatenate them
    #TODO: string paths from config file
    print("Load data...")
    df_data = pd.read_csv('./data/Fuse.csv', delimiter=';', header=0)

    print(df_data.head())

    print(f'...Data shape: {df_data.shape}')

    print("Extract data and information from table with redundant data...")
    fuse_data = {}
    fuse_data['Blow Characteristic'] = df_data['Blow Characteristic'] + df_data['Acting']
    fuse_data['Dimensions (B/H/L)'] = df_data['Body Breadth (mm)'] + df_data['Body Height (mm)'] + df_data['Body Length or Diameter (mm)'] + df_data['Fuse Size'] + df_data['Physical Dimension'] + df_data['Product Diameter'] + df_data['Product Length']

    
    # Check if there are discprepancies for reduntant data and select the most probable one

    df_data_part1 = df_data_part1.drop_duplicates()
    df_data_part2 = df_data_part2.drop_duplicates()

    print(f'...Shape data_part1 without duplicates: {df_data_part1.shape}')
    print(f'...Shape data_part2 without duplicates: {df_data_part2.shape}')

    df_data_part1 = df_data_part1.sort_values(by='id')
    df_data_part2 = df_data_part2.sort_values(by='id')

    # print(df_data_part1.head())
    # print(df_data_part2.head())

    # Check for duplicates in 'id', but with different data (not dropped before)
    print(f"...Duplicates in ids of data_part1: {df_data_part1['id'].duplicated().sum()}")
    print(f"...Duplicates in ids of data_part2: {df_data_part2['id'].duplicated().sum()}")

    # Check which ids are not present in the other dataset
    ids1 = set(df_data_part1['id'])
    ids2 = set(df_data_part2['id'])
    print(f"...IDs in part1 not in part2: {len(ids1 - ids2)}")
    print(f"...IDs in part2 not in part1: {len(ids2 - ids1)}")

    # Merge datasets on the "id" column
    df_data = pd.merge(df_data_part1, df_data_part2, on='id')

    # Move 'id' column to the end
    cols = [c for c in df_data.columns if c != 'id'] + ['id']
    df_data = df_data[cols]

    print(f'...Shape of merged data: {df_data.shape}')

    # for col in df_data.columns:
    #     print(col, ":", df_data[col].dtype)
    #     print(pd.api.types.is_numeric_dtype(df_data[col]))
    #     print(isinstance(df_data[col], pd.CategoricalDtype))
    #     print(df_data[col].dtype == object)'

    # print(df_data.head())  # Display the first few rows of the merged dataset

    # Save the cleaned merged data to a new CSV file
    df_data.to_csv('./data/Training_merged.csv', index=False, sep=';')

    print("...Data preprocessing completed and merged data saved to 'Training_merged.csv'.\n")


if __name__ == "__main__":
    print("\n-----------------------------------------------------------------------------")
    print("Task 2: Identifying alternative materials")
    print("-----------------------------------------------------------------------------\n")
    
    load_data()



    print("\nFinished training and evaluation of the binary classifier.")
    print("-----------------------------------------------------------------------------")










## Description of the data preprocessing steps:
# 0. Review knowledge about fuses + Use LLM to get additional information about the data columns. (e.g., JESD-609 Codes for Lead Finishes)
# 1. Open the two CSV files in an editor for manual inspection. (e.g., VSCode) (Check for missing data, periodical features, etc.)
# Lot's of missing data for 'Part Description' and 'Materials'.
# Reduntant data: 'Part Description' + Current rating, Rated current + Physical dimensions + Acting, Blow Characteristics + Rated Voltage
# 'Part descriptiion' (feature for material prediction) does sometimes not contain all information or various information
# COnnection between Acting, Blow Characteristic and Pre-arcing time-Min (ms), if so discrepancy? 
# Which material is the target: only Fuse Materials or also JESD-609 codes (Lead Finishes)
# Measures:
# - Remove Redundant data
# - Add all relevant information to 'Part Description'
# - Fill in gaps: (e.g., by Data Imputation)
# - Expect that Fuse Materials are the target, but discuss JESD-609 codes with experts.
# - Data shows no correlation/ relation between Acting, Blow Characteristics and Pre-arcing time-Min (ms), treat as independent features, but discuss this with experts.
# 2. Load the datasets from CSV files.
# 3. Check the shapes
# 4. Drop duplicates in each dataset.
# 5. Sort both datasets by the 'id' column to ensure they are in the same order.
# 6. Check for dataset shape 
# 7. Check for duplicates in the 'id' column which contain different data.
# 8. Merge the datasets on the 'id' column.
# 9. Check the shape and store the data into a csv file.