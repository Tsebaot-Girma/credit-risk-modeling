#scripts/eda_functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def overview(data):
    """Display the number of rows, columns, and data types."""
    print("Shape of the dataset:", data.shape)
    print("\nData Types:\n", data.dtypes)

def summary_statistics(data):
    """Display summary statistics for numerical features."""
    return data.describe()

def plot_numerical_distribution(data, features):
    """Visualize the distribution of numerical features."""
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

def plot_categorical_distribution(data, features):
    """Visualize the distribution of categorical features."""
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=data[feature])
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
        plt.show()
def correlation_analysis(data):
    """Display the correlation matrix."""
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=['int64', 'float64'])
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def identify_missing_values(data):
    """Identify missing values in the dataset."""
    return data.isnull().sum()

def detect_outliers(data, features):
    """Use box plots to identify outliers."""
    for feature in features:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[feature])
        plt.title(f'Box plot of {feature}')
        plt.show()