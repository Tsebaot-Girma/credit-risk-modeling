from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

class Aggregator(BaseEstimator, TransformerMixin):
    """Includes transaction time features needed for RFMS"""
    def __init__(self, customer_id='CustomerId'):
        self.customer_id = customer_id

    def fit(self, X, y=None):
        # Store feature names for sklearn >= 1.0 compatibility
        self.feature_names = list(X.columns)
        return self

    def transform(self, X):
        # Extract transaction times
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], utc=True)
        
        # Aggregate with time features
        agg_df = X.groupby(self.customer_id).agg({
            'Amount': ['sum', 'mean', 'count', 'std'],
            'TransactionStartTime': ['min', 'max']  # Keep first/last transaction times
        })
        agg_df.columns = [
            'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount',
            'FirstTransaction', 'LastTransaction'
        ]
        return agg_df.reset_index()
    


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Extract time-based features."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['TransactionHour'] = pd.to_datetime(X['TransactionStartTime']).dt.hour
        X['TransactionDay'] = pd.to_datetime(X['TransactionStartTime']).dt.day
        return X

# Example usage in notebook:
# aggregator = Aggregator()
# df_agg = aggregator.transform(df)