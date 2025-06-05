# visualisation.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_missing_data(df):
    """Plot missing data as a heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    return plt

def plot_outliers(df, column):
    """Plot outliers for a specific column."""
    if df[column].dtype in [np.float64, np.int64]:  # Ensure column is numeric
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f"Outliers Detection for {column}")
        plt.tight_layout()
        return plt
    else:
        raise ValueError(f"Column {column} is not numeric.")

def plot_duplicates(df):
    """Plot duplicate records in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df.duplicated(), kde=False)
    plt.title("Duplicate Records Distribution")
    plt.tight_layout()
    return plt
