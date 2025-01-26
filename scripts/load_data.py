import pandas as pd

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame with basic information.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - dict: Contains the DataFrame, column names, and row count.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        return {
            "data": df,
            "column_names": df.columns.tolist(),
            "row_count": len(df)
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")