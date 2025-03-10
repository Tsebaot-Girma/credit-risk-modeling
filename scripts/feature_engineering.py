import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_aggregate_features(df):
    """Create aggregate features for each customer."""
    aggregate_features = df.groupby('CustomerId').agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('TransactionId', 'count'),
        Std_Dev_Transaction_Amount=('Amount', 'std')
    ).reset_index()
    
    return df.merge(aggregate_features, on='CustomerId', how='left')


def extract_time_features(df):
    """Extract time-based features from the TransactionStartTime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
    df['Transaction_Day'] = df['TransactionStartTime'].dt.day
    df['Transaction_Month'] = df['TransactionStartTime'].dt.month
    df['Transaction_Year'] = df['TransactionStartTime'].dt.year
    
    return df


def encode_categorical_variables(df):
    """Encode categorical variables using One-Hot and Label Encoding."""
    df = pd.get_dummies(df, columns=['CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId'], drop_first=True)
    
    label_encoder = LabelEncoder()
    df['PricingStrategy'] = label_encoder.fit_transform(df['PricingStrategy'])
    
    return df


def handle_missing_values(df):
    """Handle missing values by imputation."""
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        # Fill missing values with the mean and reassign to the DataFrame
        df[column] = df[column].fillna(df[column].mean())
    
    for column in df.select_dtypes(include=['object']).columns:
        # Fill missing values with the mode and reassign to the DataFrame
        df[column] = df[column].fillna(df[column].mode()[0])
    
    return df


def normalize_standardize_features(df):
    """Normalize and standardize numerical features."""
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

def treat_outliers(df, numerical_columns):
    """Treat outliers in numerical features using IQR method."""
    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip outliers to the bounds
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df