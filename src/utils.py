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
        """Print initialization message."""
        print("\n-----------------------------------------------------------------------------")
        print("ItenMat: Identifying alternative battery electrode materials")
        print("-----------------------------------------------------------------------------\n")

def plot_histogram(df, path, title="CATHODE_MATERIAL Counts Histogram"):
    """
    Plot a histogram of the counts of unique values in a DataFrame column.
    :param df: DataFrame column (Series) to plot.
    :param path: Path to save the plot.
    :param title: Title of the plot.
    """
    # Count values including NaN
    value_counts = df.value_counts(dropna=False)
    print(value_counts)
    # Create the plot
    plt.figure(figsize=(8, 5))
    value_counts.plot(kind='bar', color="darkblue", edgecolor='black')
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(df.name if df.name else 'Value')
    plt.xticks(rotation=45, ha="right")  # rotate and right-align
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_missing_value_count(df, path, title="Number of Missing (NaN) Values per Column"):
    """
    Plot a bar chart with the number of missing values per column.
    :param df: DataFrame to analyze.
    :param path: Path to save the plot.
    :param title: Title of the plot.
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
    :param df: DataFrame containing cosine similarity values.
    :param path: Path to save the plot.
    :param title: Title of the plot.
    """
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='Blues', annot=True, fmt=".2f", square=True,
                cbar_kws={"label": "Cosine Similarity"})
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_correlation_histogram(cramers_matrix, path, title="Cramér's V Heatmap (Categorical Associations)"):
    """
    Plot a heatmap of Cramér's V correlation.
    :param cramers_matrix: Cramér's V correlation matrix.
    :param path: Path to save the plot.
    :param title: Title of the plot.
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
