# ----------------------------------------------------------------------------------------------------------------
# Task 2: Identifying alternative materials
# Author: Simon Markthaler
# Date: 2025-07-19
# Version: 0.0.1
#
# preprocessing:
# > Contains classes and functions for preprocessing.
# ----------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from src.utils import plot_histogram, plot_missing_value_count, plot_correlation_histogram

class Preprocessing:
    """Preprocessing class for loading and preprocessing data sets."""
    def __init__(self, Config):
        self.Config = Config

        self.path_fuse = self.Config.path + self.Config.data_path_fuse
        self.path_nan = self.Config.path + self.Config.data_path_nan
        self.path_cor =self.Config.path + self.Config.data_path_cor
        self.path_plot = self.Config.path + self.Config.data_path_plot

        self.fuse_material_list = None  # List with all fuse materials in the 'Fuse Material' column

        print("Start data preprocessing...")
        
    def load_data(self):
        """   
        Load the dataset and drop duplicates.
        """
        # Load the dataset
        print("...Load data")
        df_data = pd.read_csv(self.path_fuse, delimiter=',', header=0)

        print(f'...Data shape: {df_data.shape}')

        # Visualize the number of missing values per column
        plot_missing_value_count(df_data, self.path_nan)

        # Drop duplicates from the list
        df_data = df_data.drop_duplicates(subset=df_data.columns.difference(['PART_ID']))
        print(f'...Data shape without duplicates: {df_data.shape}')

        # Compute and plot the correlation.
        # Be aware that the results does not account for mixed formats.
        cramers_matrix = self.create_correlation(df_data)
        plot_correlation_histogram(cramers_matrix, self.path_cor)
        
        return df_data

    def matching(self, df_data):
        """
        Match 'Fuse Material' with 'PART_DESCRIPTION' and infer missing values.
        """

        print("...Matching 'Fuse Materials' with 'PART_DESCRIPTION'")

        # Harmonize text data
        df_data['Fuse Material'] = df_data['Fuse Material'].str.lower()
        df_data['PART_DESCRIPTION'] = df_data['PART_DESCRIPTION'].str.lower()
        
        self.fuse_material_list = df_data['Fuse Material'].dropna().unique()     # List with all fuse materials in the 'Fuse Material' column
        
        # Identify Indexes with missing fuse materials where 'PART_DESCRIPTION' contains material information
        # Add these materials to the 'Fuse Material' column
        df_data['Fuse Material'] = df_data.apply(self.infer_fuse_material, axis=1)
        # Preliminary investigations also analyzed the 'PART_DESCRIPTION' column for materials not present in the fuse_material_list using a GPT model.
        # This was done to ensure that all relevant materials are captured, even if they are not explicitly listed in the 'Fuse Material' column.

        # Visualize the distribution of inferred fuse materials
        plot_histogram(df_data['Fuse Material'], self.path_plot)

        # If 'PART_DESCRIPTION' is missing material details, supplement it using the inferred 'Fuse Material' values
        df_data['PART_DESCRIPTION'] = df_data.apply(self.add_material_if_missing, axis=1)

        # Identify Indexes with missing Part Descriptions
        missing_mask = df_data['PART_DESCRIPTION'].isna() | (df_data['PART_DESCRIPTION'].str.strip() == '')

        condition = missing_mask & df_data['Fuse Material'].notna()
        print(f"...Number of missing PART_DESCRIPTION values with non-missing Fuse Material: {condition.sum()}")  

        # Optional: If PART_DESCRIPTION is missing, generate a pseudo-description based on other fields (e.g. by a rule-based approach and concetation of relevant fields)
        # df_data['PART_DESCRIPTION'] = df_data.apply(self.rule_based_imputation, axis=1)
        # However, this may weaken the effectiveness of TF-IDF and cosine similarity.
        # Alternatively, use rule-based imputation together with sentence_transformers or other embedding methods, 
        # could be used to generate embeddings for the PART_DESCRIPTION column.
    
        return df_data, self.fuse_material_list
    
    def infer_fuse_material(self, row):
        """
        Infer missing 'Fuse Material' based on 'PART_DESCRIPTION'.
        """
        # Check if the Part Description contains a Fuse Material of the Fuse Material List in the text of the missing material values -> Add it to Fuse Materials
        if pd.isna(row['Fuse Material']):
            description = str(row['PART_DESCRIPTION'])
            for material in self.fuse_material_list:
                if material in description:
                    print(f"     Missing 'Fuse Material' in PART_ID:{row['PART_ID']} -> Filled with information from 'PART_DESCRIPTION': 'Fuse Material' = {material}")
                    return material
            return np.nan  # Still missing if no match found
        else:
            return row['Fuse Material']  # Already filled
        
            # Function to add fuse material if missing

    def add_material_if_missing(self, row):
        """
        Add fuse material to PART_DESCRIPTION if missing.
        """
        material = row['Fuse Material']
        description = row['PART_DESCRIPTION']
        if material in self.fuse_material_list:
            if not pd.isna(description) or str(description).strip() == '':
                # Check if any material from the full list is already in the description
                if not any(mat in description for mat in self.fuse_material_list):
                    # Append the material if none found
                    new_description = description + ', ' + material
                    print(f"     Missing material information in 'PART_DESCRIPTION' in PART_ID:{row['PART_ID']} -> Added information from 'Fuse Material' = {material}")
                    return new_description
                return description
            else:
                return description
        else:
            return description
        
    def rule_based_imputation(row):
        """        
        Generate a pseudo-description based on other fields if PART_DESCRIPTION is missing.
        """
        if pd.isna(row['PART_DESCRIPTION']) or str(row['PART_DESCRIPTION']).strip() == '':
            # Extract fields, fallback to default or blank if missing
            acting = str(row.get('Acting', '')).strip()
            current = str(row.get('Rated Current (A)', '')).replace('A', '').strip()
            voltage = str(row.get('Rated Voltage (V)', '') or row.get('Maximum AC Voltage Rating', '')).replace('V', '').strip()
            mounting = str(row.get('Mounting', '')).strip()
            fuse_size = str(row.get('Fuse Size', '')).strip()
            material = str(row.get('Fuse Material', '')).strip()

            # Ensure values are not empty
            current = current + 'A' if current else ''
            voltage = voltage + 'V' if voltage else ''
            acting = acting if acting else 'Fuse'
            mounting = mounting if mounting else ''
            fuse_size = fuse_size if fuse_size else ''
            material = material if material else ''

            # Build description
            parts = [
                "Fuse",
                acting,
                current,
                voltage,
                mounting,
                "Cartridge",
                fuse_size,
                material,
                "Electric Fuse"
            ]

            # Filter out empty strings and join
            return ', '.join([p for p in parts if p])
        else:
            return row['PART_DESCRIPTION']

    @staticmethod    
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        if confusion_matrix.size == 0:
            return np.nan
        chi2, p, dof, expected = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        if min(kcorr - 1, rcorr - 1) == 0:
            return np.nan
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    
    def create_correlation(self, df_data):

        df_copy = df_data.copy()

        # Exclude certain columns
        excluded_columns = ['PART_ID', 'PART_DESCRIPTION']
        df_categorical = df_copy.drop(columns=excluded_columns)

        # Ensure all columns are treated as categorical
        df_categorical = df_categorical.astype('category')

        # Create the Cramér's V matrix
        columns = df_categorical.columns
        cramers_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), 
                                    index=columns, columns=columns)

        # Fill in the matrix
        for col1 in columns:
            for col2 in columns:
                if col1 == col2:
                    cramers_matrix.loc[col1, col2] = 1.0
                else:
                    val = self.cramers_v(df_categorical[col1], df_categorical[col2])
                    cramers_matrix.loc[col1, col2] = val
        
        return cramers_matrix 



