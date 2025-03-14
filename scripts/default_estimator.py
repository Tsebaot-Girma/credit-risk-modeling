import scorecardpy as sc
import pandas as pd
import numpy as np

def calculate_rfms(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Calculate RFMS metrics with score calculation"""
    df_rfms = df_agg.copy()
    
    # Convert to datetime
    df_rfms['FirstTransaction'] = pd.to_datetime(df_rfms['FirstTransaction'], utc=True)
    df_rfms['LastTransaction'] = pd.to_datetime(df_rfms['LastTransaction'], utc=True)
    current_time = pd.Timestamp.now(tz='UTC')
    
    # Calculate metrics
    df_rfms['Recency'] = (current_time - df_rfms['LastTransaction']).dt.days
    df_rfms['Frequency'] = df_rfms['TransactionCount']
    df_rfms['Monetary'] = df_rfms['TotalAmount']
    df_rfms['Stability'] = (df_rfms['LastTransaction'] - df_rfms['FirstTransaction']).dt.days

    # Add debug statements
    print("Recency values:", df_rfms['Recency'])
    print("Monetary values:", df_rfms['Monetary'])

    # Add scoring
    try:
        df_rfms['RecencyScore'] = pd.qcut(df_rfms['Recency'], q=4, labels=False)
        df_rfms['MonetaryScore'] = pd.qcut(df_rfms['Monetary'], q=4, labels=False)
        df_rfms['RFM_Score'] = df_rfms['RecencyScore'] + df_rfms['MonetaryScore']
    except Exception as e:
        print("Error in scoring:", e)

    return df_rfms

def assign_labels(df_rfms: pd.DataFrame) -> pd.DataFrame:
    """Create valid binary labels"""
    df_labeled = df_rfms.copy()
    
    # Ensure numeric RFM_Score
    df_labeled['RFM_Score'] = df_labeled['RFM_Score'].astype(int)
    
    # Create labels (adjust threshold if needed)
    df_labeled['Label'] = np.where(
        df_labeled['RFM_Score'] > df_labeled['RFM_Score'].median(),
        'Good', 
        'Bad'
    )
    
    return df_labeled

def woe_binning(df: pd.DataFrame, target: str, features: list) -> dict:
    """Perform WoE binning using scorecardpy."""
    bins = sc.woebin(df, y=target, x=features)
    return bins