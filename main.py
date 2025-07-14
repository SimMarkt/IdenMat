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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# -----------------------------------------------Data Preprocessing-----------------------------------------------
def load_data():
    """   
    Load the dataset.
    """
    # Load the datasets and concatenate them
    #TODO: string paths from config file
    print("Load data...")
    df_data = pd.read_csv('./data/Fuse.csv', delimiter=';', header=0)

    # print(df_data.head())
    # print(df_data.iloc[:, 23:28])

    print(f'...Data shape: {df_data.shape}')

    print("...Extract data and information from table with redundant data")

    # 0. Drop duplicates from the list
    df_data = df_data.drop_duplicates(subset=df_data.columns.difference(['PART_ID']))
    print(f'...Data shape without duplicates: {df_data.shape}')

    # 1. Get missing Part Descriptions and Fuse Materials
    df_data['Fuse Material'] = df_data['Fuse Material'].str.lower()
    df_data['PART_DESCRIPTION'] = df_data['PART_DESCRIPTION'].str.lower()
    # 1.0. Get Fuse Material list + Histogramm plot
    fuse_material_list = df_data['Fuse Material'].dropna().unique()
    
    plot_histogram(df_data['Fuse Material'])
    
    # 1.1. Identify Indexes with missing Fuse Materials
    invalid_fuse_materials = ~df_data['Fuse Material'].isin(fuse_material_list)

    missing_indexes = df_data[invalid_fuse_materials].index

    # 1.2. Check if the Part Description contains a Fuse Material of the Fuse Material List in the text of the missing material values -> Add it to Fuse Materials
    def infer_fuse_material(row):
        if pd.isna(row['Fuse Material']):
            description = str(row['PART_DESCRIPTION'])
            for mat in fuse_material_list:
                if mat in description:
                    print(f"   Missing Fuse Material in PART_ID:{row['PART_ID']} -> Filled with information from PART_DESCRIPTION: Fuse Material = {mat}")
                    return mat
            return np.nan  # Still missing if no match found
        else:
            return row['Fuse Material']  # Already filled

    # Apply the function to each row
    df_data['Fuse Material'] = df_data.apply(infer_fuse_material, axis=1)

    plot_histogram(df_data['Fuse Material'])

    # 1.3. Check if Part Description misses Material information and add this if its given in 'Fuse Materials'
    # Function to add fuse material if missing
    def add_material_if_missing(row):
        material = row['Fuse Material']
        description = row['PART_DESCRIPTION']
        if material in fuse_material_list:
            # Check if any material from the full list is already in the description
            if not any(mat in description for mat in fuse_material_list):
                # Append the material if none found
                new_description = description + material
                print(f"   Missing material information in PART_DESCRIPTION in PART_ID:{row['PART_ID']} -> Filled with information from Fuse Material: PART_DESCRIPTION = {new_description}")
                return new_description
        else:
            return description

    df_data['PART_DESCRIPTION'] = df_data.apply(add_material_if_missing, axis=1)
    
    # 1.3. Identify Indexes with missing Part Descriptions
    missing_mask = df_data['PART_DESCRIPTION'].isna() | (df_data['PART_DESCRIPTION'].str.strip() == '')

    missing_indexes = df_data[missing_mask].index

    # 1.4. Extract Part Description from other Data in that row so that it looks similar to other Part descriptions
    # Extract information and omit redundancy
    
    # Generate Pseudo-descriptions with rule-based approach (SHould be sufficient for the similarity task - for more accurate descriptions: Use kNN or TF-IDF)
    def generate_description(row):
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
    
    df_data['PART_DESCRIPTION'] = df_data.apply(generate_description, axis=1)

    # 2. Train the TFDFT + similarity function
    # Train/Validation/Test set
    # 2.1 lower all strings


    return df_data

def plot_histogram(df):
    
    # Count values including NaN
    value_counts = df.value_counts(dropna=False)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    value_counts.plot(kind='bar', ax=ax, color="darkblue", edgecolor='black')

    # Customize
    ax.set_ylabel('Count')
    ax.set_xlabel(df)
    ax.tick_params(axis='x', rotation=45)

    # Save or show
    plt.tight_layout()
    plt.savefig("./plots/fuse_material_histogram.png", dpi=300)
    plt.show()
    plt.close(fig)



#     print("...Analyzing feature types")
#     feature_types = analyze_feature_types(df_data)

#     # Impute missing numerical values is not necessary, since XGBoost can handle missing values.

#     # Plot histograms for categorical features to check for missing values, distribution, ordinality, etc.
#     print("...Plotting histograms and stripplots for categorical and numerical features")
#     plot_categorical_histograms(df_data, feature_types)
#     plot_categorical_id_stripplots(df_data, feature_types)
#     plot_numerical_histograms(df_data, feature_types)
#     plot_numerical_id_scatter(df_data, feature_types)







#     fuse_data = {}
#     fuse_data['Blow'] = {}
#     fuse_data['Blow']['Blow Characteristic'] = df_data['Blow Characteristic']
#     fuse_data['Blow']['Acting'] = df_data['Acting']
#     fuse_data['Pre-arcing time-Min (ms)'] = df_data['Pre-arcing time-Min (ms)']
#     fuse_data['Dimensions (B/H/L)'] = {}
#     fuse_data['Dimensions (B/H/L)']['Body Breadth (mm)'] = df_data['Body Breadth (mm)'] 
#     fuse_data['Dimensions (B/H/L)']['Body Height (mm)'] = df_data['Body Height (mm)'] 
#     fuse_data['Dimensions (B/H/L)']['Body Length or Diameter (mm)'] = df_data['Body Length or Diameter (mm)'] 
#     fuse_data['Dimensions (B/H/L)']['Fuse Size'] = df_data['Fuse Size'] 
#     fuse_data['Dimensions (B/H/L)']['Physical Dimension'] = df_data['Physical Dimension'] 
#     fuse_data['Dimensions (B/H/L)']['Product Diameter'] = df_data['Product Diameter'] 
#     fuse_data['Dimensions (B/H/L)']['Product Length'] = df_data['Product Length']
#     fuse_data['Current Rating'] = {}
#     fuse_data['Current Rating']['Current Rating'] = df_data['Current Rating']
#     fuse_data['Current Rating']['Rated Current (A)'] = df_data['Rated Current (A)']
#     fuse_data['Voltage Rating'] = {}
#     fuse_data['Voltage Rating']['Maximum AC Voltage Rating'] = df_data['Maximum AC Voltage Rating']
#     fuse_data['Voltage Rating']['Maximum DC Voltage Rating'] = df_data['Maximum DC Voltage Rating']
#     fuse_data['Voltage Rating']['Rated Voltage (V)'] = df_data['Rated Voltage (V)']
#     fuse_data['Voltage Rating']['Rated Voltage(AC) (V)'] = df_data['Rated Voltage(AC) (V)']
#     fuse_data['Voltage Rating']['Rated Voltage(DC) (V)'] = df_data['Rated Voltage(DC) (V)']
#     fuse_data['Rated Breaking Capacity (A)'] = df_data['Rated Breaking Capacity (A)'] 
#     fuse_data['Fuse Material'] = df_data['Fuse Material'] 
#     fuse_data['JESD-609 Code'] = df_data['JESD-609 Code'] 
#     fuse_data['Joule-integral-Nom (J)'] = df_data['Joule-integral-Nom (J)']
#     fuse_data['LC Risk'] = df_data['LC Risk']
#     fuse_data['Maximum Power Dissipation'] = df_data['Maximum Power Dissipation']
#     fuse_data['Mounting'] = {}
#     fuse_data['Mounting']['Mounting'] = df_data['Mounting']
#     fuse_data['Mounting']['Mounting Feature'] = df_data['Mounting Feature']
#     fuse_data['Number of Terminals'] = df_data['Number of Terminals']
#     fuse_data['Operating Temperature-Max (Cel)'] = df_data['Operating Temperature-Max (Cel)']
#     fuse_data['Operating Temperature-Min (Cel)'] = df_data['Operating Temperature-Min (Cel)']
#     fuse_data['Application'] = df_data['Application']
#     fuse_data['Additional Feature'] = df_data['Additional Feature']

#     print("...Data preprocessing completed.\n")

#     return df_data

# def analyze_feature_types(df, exclude_cols=[], periodic_nominals=[]):
#     feature_types = {}
#     for col in df.columns:
#         if col in exclude_cols:
#             continue
#         if pd.api.types.is_numeric_dtype(df[col]):
#             feature_types[col] = 'numerical'
#         elif isinstance(df[col], pd.CategoricalDtype) or df[col].dtype == object:
#             feature_types[col] = 'categorical'
#         else:
#             feature_types[col] = 'unknown'
#     return feature_types

# def plot_categorical_histograms(df, feature_types):

#     categorical_cols = [col for col, ftype in feature_types.items() if ftype.startswith('categorical')]
#     n_cols = 3  # Number of subplots per row
#     n = len(categorical_cols)
#     n_rows = (n + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
#     axes = axes.flatten()

#     for i, col in enumerate(categorical_cols):
#         df[col].value_counts(dropna=False).plot(kind='bar', ax=axes[i], label=col, color='darkblue', edgecolor='black')
#         axes[i].tick_params(axis='x', rotation=45)
#         # axes[i].set_xlabel('Value')
#         axes[i].set_ylabel('Count')
#         axes[i].legend(loc='upper right')

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.savefig('./plots/categorical_histograms.png', dpi=300)
#     plt.close(fig)

# def transform_numerical_nans(df, numerical_cols):
#     min_values = {}
#     max_values = {}
#     nan_values = {}
#     for col in numerical_cols:
#         min_values[col] = df[col].min()
#         max_values[col] = df[col].max()
#         nan_values[col] = min_values[col] - (max_values[col] - min_values[col]) /10
#         df[col] = df[col].fillna(nan_values[col])  # Fill NaNs with a placeholder value for plotting 

#     return df, min_values, max_values, nan_values

# def plot_numerical_histograms(df, feature_types):

#     numerical_cols = [col for col, ftype in feature_types.items() if ftype == 'numerical']
#     n_cols = 3  # Number of subplots per row
#     n = len(numerical_cols)
#     n_rows = (n + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
#     axes = axes.flatten()

#     for i, col in enumerate(numerical_cols):
#         df[col].hist(ax=axes[i], bins=30, label=col, color='darkblue', edgecolor='black')
#         # axes[i].set_xlabel('Value')
#         axes[i].set_ylabel('Count')
#         axes[i].legend(title=f'NaN={df[col].isnull().sum()}', loc='upper right')

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.savefig(f'./plots/numerical_histograms.png', dpi=300)
#     plt.close(fig) 

# def plot_categorical_id_stripplots(df, feature_types):
#     """
#     For each categorical column, plot a strip plot (or scatter) with:
#     - x-axis: value of the categorical column
#     - y-axis: id
#     """

#     categorical_cols = [col for col, ftype in feature_types.items() if ftype == 'categorical']
#     n_cols = 3  # Number of subplots per row
#     n = len(categorical_cols)
#     n_rows = (n + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
#     axes = axes.flatten()

#     for col in categorical_cols:
#         df[col] = df[col].fillna('nan')

#     for i, col in enumerate(categorical_cols):
#         sns.stripplot(x=df.index, y=df[col], ax=axes[i], alpha=0.7, jitter=True, color='darkblue')
#         axes[i].set_ylabel(col)
#         axes[i].tick_params(axis='x')

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.savefig('./plots/categorical_stripplot.png', dpi=300)
#     plt.close(fig) 

# def plot_numerical_id_scatter(df, feature_types):
#     """
#     For each numerical column, plot its value along 'id' on the x-axis.
#     """

#     numerical_cols = [col for col, ftype in feature_types.items() if ftype == 'numerical']
#     n_cols = 3  # Number of subplots per row
#     n = len(numerical_cols)
#     n_rows = (n + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
#     axes = axes.flatten()

#     df, min_values, max_values, nan_values = transform_numerical_nans(df, numerical_cols)

#     for i, col in enumerate(numerical_cols):
#         # Highlight range around the NaN values
#         buffer = (max_values[col] - min_values[col]) / 20
#         nan_val = nan_values[col]
#         axes[i].axhspan(nan_val - buffer, nan_val + buffer,
#                         color='red', alpha=0.5, label='NaN')
#         axes[i].scatter(df.index, df[col], alpha=0.7, color='darkblue', edgecolor='black')
#         axes[i].set_xlabel('id')
#         axes[i].set_ylabel(col)

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.tight_layout()
#     plt.savefig('./plots/numerical_scatter.png', dpi=300)
#     plt.close(fig)


# def vectorize_part_descriptions(df_data):

#     # Vectorize the part descriptions
#     print("Vectorize the part descriptions...")
#     tfidf = TfidfVectorizer(stop_words='english')       # Computes: [Frequency of a term (TF) in a description] x [How rare the term is across all descriptions]
#     tfidf_matrix = tfidf.fit_transform(df_data['PART_DESCRIPTION'].fillna(""))

#     # print(tfidf_matrix)

#     print(tfidf.get_feature_names_out())

#     df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
#     print(df_tfidf)

#     print("...Completed vectorization.\n")

#     # print("...Data preprocessing completed and merged data saved to 'Training_merged.csv'.\n")

#     print("Similarity analysis...")

#     cosine_sim = cosine_similarity(tfidf_matrix)

#     # Store indices of top 5 similar items for each row
#     top_n = 5
#     similarities = []

#     for idx, row in enumerate(cosine_sim):
#         # Get indices of top 5 most similar items (excluding itself)
#         similar_indices = np.argsort(row)[::-1][1:top_n+1]
#         similarities.append(similar_indices)

#     # Convert to DataFrame if needed
#     similar_df = pd.DataFrame(similarities, columns=[f"Alternative_{i+1}" for i in range(top_n)])

#     for i in range(top_n):
#         similar_df[f"Alternative_{i+1}_Desc"] = similar_df[f"Alternative_{i+1}"].apply(lambda x: df_data['PART_DESCRIPTION'].iloc[x])

#     # Check 
#     query = df_data['PART_DESCRIPTION'].iloc[135]

#     query_vec = tfidf.transform([query])
#     query_sim = cosine_similarity(query_vec, tfidf_matrix)

#     top_matches = query_sim[0].argsort()[::-1][1:6]
#     for idx in top_matches:
#         print(df_data['PART_DESCRIPTION'].iloc[idx])





if __name__ == "__main__":
    print("\n-----------------------------------------------------------------------------")
    print("Task 2: Identifying alternative materials")
    print("-----------------------------------------------------------------------------\n")
    
    df_data = load_data()

    # tfidf_matrix = vectorize_part_descriptions(df_data)



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
# Mixed formats in fields (e.g., "250V", "250VAC", etc.).
# SLight discrepancies in words: A11 22.5mm	5.4mm = 5 X 20mm
# Also capilization and non capitalization of the strings -> lower everything
# The "PART_DESCRIPTION" is a long, free-text field. -> Difficult to directly compare, cluster, or analyze with standard methods.
# Measures:
# - Use NLP like TF-IDF vectorization to turn text into vectors
# - Remove Redundant data
# - Add all relevant information to 'Part Description'
# - Fill in gaps: (e.g., by Data Imputation)
# - Expect that Fuse Materials are the target, but discuss JESD-609 codes with experts.
# - Data shows no correlation/ relation between Acting, Blow Characteristics and Pre-arcing time-Min (ms), treat as independent features, but discuss this with experts.
# - Principal component analysis for all features for material forecast
# 2. Load the datasets from CSV files.
# 3. Check the shapes
# 4. Drop duplicates in each dataset.
# 5. Sort both datasets by the 'id' column to ensure they are in the same order.
# 6. Check for dataset shape 
# 7. Check for duplicates in the 'id' column which contain different data.
# 8. Merge the datasets on the 'id' column.
# 9. Check the shape and store the data into a csv file.