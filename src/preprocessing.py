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

from src.utils import plot_histogram

class Preprocessing:
    """Preprocessing class for loading and preprocessing data sets."""
    def __init__(self, Config):
        self.Config = Config

        self.path_fuse = self.Config.path + self.Config.data_path_fuse
        self.path_plot = self.Config.path + self.Config.data_path_plot

        print("Start data preprocessing...")
        
    def load_data(self):
        """   
        Load the dataset and drop duplicates.
        """
        # Load the dataset
        print("...Load data")
        df_data = pd.read_csv(self.path_fuse, delimiter=';', header=0)

        print(f'...Data shape: {df_data.shape}')

        # Drop duplicates from the list
        df_data = df_data.drop_duplicates(subset=df_data.columns.difference(['PART_ID']))
        print(f'...Data shape without duplicates: {df_data.shape}')

        self.fuse_material_list = None  # List with all fuse materials in the 'Fuse Material' column

        return df_data

    def matching(self, df_data):

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
