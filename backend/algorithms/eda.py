import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_sample_data():
    """Load a sample dataset for EDA practice"""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv"
    return pd.read_csv(url)

def basic_exploration(df):
    """Perform basic exploration of a dataset"""
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nSummary Statistics:")
    print(df.describe())
    return df

def check_missing_values(df):
    """Check for missing values in the dataset"""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    print("Missing Values Analysis:")
    print(missing_data[missing_data['Missing Values'] > 0])
    
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    
    return missing_data

def visualize_distributions(df, columns=None):
    """Visualize distributions of numerical columns"""
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns[:5]
    
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()

def correlation_analysis(df):
    """Perform correlation analysis on numerical columns"""
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numerical_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    
    return corr_matrix

