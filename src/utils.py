"""
---------------------------------------------------------------------------------------------------
IdenMat: Identifying alternative battery electrode materials
         via unsupervised similarity matching (NLP)
GitHub Repository: https://github.com/SimMarkt/IdenMat.git

utils:
> Contains utility/helper functions.
---------------------------------------------------------------------------------------------------
"""

import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Configuration:
    """Configuration class for loading environment variables from a YAML file."""
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config.yaml", "r", encoding="utf-8") as env_file:
            agent_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(agent_config)

        # OS path to the project folder
        self.path = None

        self.print_init()
        
    @staticmethod
    def print_init():
        print("\n-----------------------------------------------------------------------------")
        print("Task 2: Identifying alternative materials")
        print("-----------------------------------------------------------------------------\n")

def plot_histogram(df, path, title="ELECTRODE_MATERIAL Counts Histogram"):
    """
    Plot a histogram of the counts of unique values in a DataFrame column.
    """
    # Count values including NaN
    value_counts = df.value_counts(dropna=False)

    # Create the plot
    plt.figure(figsize=(8, 5))
    value_counts.plot(kind='bar', color="darkblue", edgecolor='black')
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(df.name if df.name else 'Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_missing_value_count(df, path, title="Number of Missing (NaN) Values per Column"):
    """
    Plot a bar chart with the number of missing values per column.
    """
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]  # only show columns with NaNs

    # Sort for better readability (optional)
    nan_counts = nan_counts.sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    nan_counts.plot(kind='bar', color='darkblue', edgecolor='black')
    plt.title(title)
    plt.ylabel("Count of NaNs")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(path, dpi=300)
    plt.close()

def plot_cosine_similarity_heatmap(df, path, title="Cosine Similarity Heatmap"):
    """
    Plot a heatmap of cosine similarity values.
    """
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='Blues', annot=True, fmt=".2f", square=True, cbar_kws={"label": "Cosine Similarity"})
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_correlation_histogram(cramers_matrix, path, title="Cramér's V Heatmap (Categorical Associations)"):
    """
    Plot a heatmap of Cramér's V correlation.
    """

    cramers_matrix = cramers_matrix.astype(float)
    # Plot the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(cramers_matrix, annot=False, fmt=".2f", cmap='Blues', 
                square=True, cbar_kws={"label": "Cramér's V"}, center=0, linewidths=0.5)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()





