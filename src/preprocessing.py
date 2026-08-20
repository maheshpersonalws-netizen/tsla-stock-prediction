import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans stock dataframe. Specifically:
    - Normalizes column names (handles yfinance MultiIndex headers).
    - Converts Date column to datetime.
    - Sorts by Date.
    - Eliminates basic nulls if any (though yfinance rarely has any before features).
    
    Parameters:
    - df (pd.DataFrame): Input raw DataFrame.
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame with flat column headers.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")
        
    df_clean = df.copy()
    
    # Flatten MultiIndex columns if present
    if isinstance(df_clean.columns, pd.MultiIndex):
        df_clean.columns = [col[0] for col in df_clean.columns]
        
    # Ensure Date column exists
    if 'Date' not in df_clean.columns:
        # Check if Date is the index
        if df_clean.index.name == 'Date' or 'date' in str(df_clean.index.name).lower():
            df_clean = df_clean.reset_index()
        else:
            raise KeyError("Date column is not present in the dataset.")
            
    # Normalize column names to title case to match expectation (e.g. Date, Close, High, Low, Open, Volume)
    # yfinance columns are usually Capitalized (Open, High, Low, Close, Adj Close, Volume)
    rename_dict = {}
    for col in df_clean.columns:
        col_str = str(col)
        if col_str.lower() == 'date':
            rename_dict[col] = 'Date'
        elif col_str.lower() == 'close':
            rename_dict[col] = 'Close'
        elif col_str.lower() == 'open':
            rename_dict[col] = 'Open'
        elif col_str.lower() == 'high':
            rename_dict[col] = 'High'
        elif col_str.lower() == 'low':
            rename_dict[col] = 'Low'
        elif col_str.lower() == 'volume':
            rename_dict[col] = 'Volume'
            
    df_clean.rename(columns=rename_dict, inplace=True)
    
    # Convert Date to datetime object
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Sort chronologically by Date
    df_clean.sort_values('Date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    # Ensure required columns exist after renaming
    required_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    for col in required_cols:
        if col not in df_clean.columns:
            raise ValueError(f"Required column '{col}' is missing after cleanup.")
            
    # Convert prices to numeric
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Drop rows where critical price values are missing
    df_clean.dropna(subset=['Close', 'Volume'], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean
