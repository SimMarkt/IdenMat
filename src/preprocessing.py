"""
---------------------------------------------------------------------------------------------------
IdenMat: Identifying alternative battery electrode materials
         via unsupervised similarity matching (NLP)
GitHub Repository: https://github.com/SimMarkt/IdenMat.git

preprocessing:
> Contains classes and functions for preprocessing.
---------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from src.utils import Configuration, plot_histogram, plot_missing_value_count, plot_correlation_histogram

class Preprocessing:
    """Preprocessing class for loading and preprocessing data sets."""
    def __init__(self, config: Configuration) -> None:
        """
        :param config: Configuration object containing environment variables.
        """
        self.config = config

        self.path_bat = self.config.path + self.config.data_path_bat
        self.path_nan = self.config.path + self.config.data_path_nan
        self.path_cor =self.config.path + self.config.data_path_cor
        self.path_plot = self.config.path + self.config.data_path_plot

        # List with all electrode materials in the 'ELECTRODE_MATERIAL' column
        self.bat_material_list = None

        print("Start data preprocessing...")

    def load_data(self) -> pd.DataFrame:
        """   
        Load the dataset and drop duplicates.
        :return df_data: DataFrame containing the cleaned data.
        """
        # Load the dataset
        print("...Load data")
        df_data = pd.read_csv(self.path_bat, delimiter=',', header=0)

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

    def matching(self, df_data: pd.DataFrame):
        """
        Match 'ELECTRODE_MATERIAL' with 'PART_DESCRIPTION' and infer missing values.
        :param df_data: DataFrame containing the cleaned data.
        :return df_data: DataFrame with matched and inferred values.
        :return bat_material_list: List of all electrode materials in 
                                   the 'ELECTRODE_MATERIAL' column.
        """

        print("...Matching 'ELECTRODE_MATERIALs' with 'PART_DESCRIPTION'")

        # Harmonize text data
        df_data['ELECTRODE_MATERIAL'] = df_data['ELECTRODE_MATERIAL'].str.lower()
        df_data['PART_DESCRIPTION'] = df_data['PART_DESCRIPTION'].str.lower()

        # List with all electrode materials in the 'ELECTRODE_MATERIAL' column
        self.bat_material_list = df_data['ELECTRODE_MATERIAL'].dropna().unique()

        # Identify Indexes with missing electrode materials where 'PART_DESCRIPTION' contains
        # material information
        # Add these materials to the 'ELECTRODE_MATERIAL' column
        df_data['ELECTRODE_MATERIAL'] = df_data.apply(self.infer_bat_material, axis=1)
        # Preliminary investigations also analyzed the 'PART_DESCRIPTION' column for materials
        # not present in the bat_material_list using an LLM.
        # This was done to ensure that all relevant materials are captured, even if they are not
        # explicitly listed in the 'ELECTRODE_MATERIAL' column.

        # Visualize the distribution of inferred electrode materials
        plot_histogram(df_data['ELECTRODE_MATERIAL'], self.path_plot)

        # If 'PART_DESCRIPTION' is missing material details, supplement it using the inferred
        # 'ELECTRODE_MATERIAL' values
        df_data['PART_DESCRIPTION'] = df_data.apply(self.add_material_if_missing, axis=1)

        # Identify Indexes with missing Part Descriptions
        missing_mask = (
            df_data['PART_DESCRIPTION'].isna() |
            (df_data['PART_DESCRIPTION'].str.strip() == '')
        )

        condition = missing_mask & df_data['ELECTRODE_MATERIAL'].notna()
        print("...Number of missing PART_DESCRIPTION values with non-missing"
              f" ELECTRODE_MATERIAL: {condition.sum()}")

        # Optional: If PART_DESCRIPTION is missing, generate a pseudo-description based on other
        # fields (e.g. by a rule-based approach and concetation of relevant fields)
        # df_data['PART_DESCRIPTION'] = df_data.apply(self.rule_based_imputation, axis=1)
        # However, this may weaken the effectiveness of TF-IDF and cosine similarity.
        # Alternatively, use rule-based imputation together with sentence_transformers or other
        # embedding methods, could be used to generate embeddings for the PART_DESCRIPTION column.

        return df_data, self.bat_material_list

    def infer_bat_material(self, row):
        """
        Infer missing 'ELECTRODE_MATERIAL' based on 'PART_DESCRIPTION'.
        :param row: DataFrame row.
        :return: Inferred or original 'ELECTRODE_MATERIAL'.
        """
        # Check if the Part Description contains a ELECTRODE_MATERIAL of the ELECTRODE_MATERIAL
        # list in the text of the missing material values -> Add it to ELECTRODE_MATERIALs
        if pd.isna(row['ELECTRODE_MATERIAL']):
            description = str(row['PART_DESCRIPTION'])
            for material in self.bat_material_list:
                if material in description:
                    print(f"     Missing 'ELECTRODE_MATERIAL' in PART_ID:{row['PART_ID']} ->"
                          " Filled with information from 'PART_DESCRIPTION': 'ELECTRODE_MATERIAL'"
                          f" = {material}")
                    return material
            return np.nan  # Still missing if no match found
        else:
            return row['ELECTRODE_MATERIAL']  # Already filled

            # Function to add ELECTRODE_MATERIAL if missing

    def add_material_if_missing(self, row):
        """
        Add ELECTRODE_MATERIAL to PART_DESCRIPTION if missing.
        """
        material = row['ELECTRODE_MATERIAL']
        description = row['PART_DESCRIPTION']
        if material in self.bat_material_list:
            if not pd.isna(description) or str(description).strip() == '':
                # Check if any material from the full list is already in the description
                if not any(mat in description for mat in self.bat_material_list):
                    # Append the material if none found
                    new_description = description + ', ' + material
                    print("     Missing material information in 'PART_DESCRIPTION' in"
                          f" PART_ID:{row['PART_ID']} -> Added information from 'ELECTRODE_MATERIAL"
                          f"' = {material}")
                    return new_description
                return description
            else:
                return description
        else:
            return description

    @staticmethod
    def rule_based_imputation(row):
        """        
        Generate a pseudo-description based on other fields if PART_DESCRIPTION is missing.
        :param row: DataFrame row.
        :return: Imputed or original 'PART_DESCRIPTION'.
        """
        if pd.isna(row['PART_DESCRIPTION']) or str(row['PART_DESCRIPTION']).strip() == '':
            # Extract fields, fallback to default or blank if missing
            acting = str(row.get('Acting', '')).strip()
            current = str(row.get('Rated Current (A)', '')).replace('A', '').strip()
            voltage = str(row.get('Rated Voltage (V)', '') or row.get('Maximum AC Voltage Rating', '')).replace('V', '').strip()
            mounting = str(row.get('Mounting', '')).strip()
            bat_size = str(row.get('bat Size', '')).strip()
            material = str(row.get('ELECTRODE_MATERIAL', '')).strip()

            # Ensure values are not empty
            current = current + 'A' if current else ''
            voltage = voltage + 'V' if voltage else ''
            acting = acting if acting else 'bat'
            mounting = mounting if mounting else ''
            bat_size = bat_size if bat_size else ''
            material = material if material else ''

            # Build description
            parts = [
                "bat",
                acting,
                current,
                voltage,
                mounting,
                "Cartridge",
                bat_size,
                material,
                "Electric bat"
            ]

            # Filter out empty strings and join
            return ', '.join([p for p in parts if p])
        else:
            return row['PART_DESCRIPTION']

    @staticmethod
    def cramers_v(x, y):
        """
        Calculate Cramér's V statistic for categorical-categorical association.
        :param x: First categorical variable (Pandas Series).
        :param y: Second categorical variable (Pandas Series).
        :return: Cramér's V statistic.
        """
        confusion_matrix = pd.crosstab(x, y)
        if confusion_matrix.size == 0:
            return np.nan
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
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
        """
        Create a Cramér's V correlation matrix for categorical variables.
        :param df_data: DataFrame containing the cleaned data.
        :return cramers_matrix: Cramér's V correlation matrix.
        """

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
