# ----------------------------------------------------------------------------------------------------------------
# Task 2: Identifying alternative materials
# Author: Simon Markthaler
# Date: 2025-07-19
# Version: 0.0.1
#
# preprocessing:
# > Contains utility/helper functions.
# ----------------------------------------------------------------------------------------------------------------

import yaml
import matplotlib.pyplot as plt

class Configuration:
    """Configuration class for loading environment variables from a YAML file."""
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config.yaml", "r") as env_file:
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

def plot_histogram(df, path, title="Fuse Material Counts Histogram"):
    # Count values including NaN
    value_counts = df.value_counts(dropna=False)

    # Create the plot
    plt.figure(figsize=(8, 5))
    value_counts.plot(kind='bar', color="darkblue", edgecolor='black')

    # Customize
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(df.name if df.name else 'Value')
    plt.xticks(rotation=45)

    # Save or show
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()