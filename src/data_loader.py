import os
import pandas as pd
import yfinance as yf

def load_data(csv_path="data/TSLA.csv", symbol="TSLA", start_date="2015-01-01", end_date="2024-12-31"):
    """
    Loads TSLA historical stock data.
    If the local CSV file exists, it loads data from it.
    Otherwise, it downloads the data programmatically via Yahoo Finance (yfinance)
    and caches it locally as a CSV file.
    
    Parameters:
    - csv_path (str): The local file path to the CSV dataset.
    - symbol (str): The stock ticker symbol.
    - start_date (str): Start date for download (YYYY-MM-DD).
    - end_date (str): End date for download (YYYY-MM-DD).
    
    Returns:
    - pd.DataFrame: Cleaned stock market DataFrame.
    """
    # Create parent directory of CSV if it doesn't exist
    dir_name = os.path.dirname(csv_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    df = None
    
    # Try reading from local CSV first
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            # We want to read it and check if it's valid
            df = pd.read_csv(csv_path)
            # If the CSV has multi-level columns stored, pandas might load them with header issues.
            # Usually index is saved as column 0 or we parse it
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif df.index.name == 'Date':
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Print feedback
            print(f"Loaded dataset from cached local CSV: {csv_path}")
        except Exception as e:
            print(f"Failed to read local CSV: {e}. Re-downloading from Yahoo Finance...")
            df = None
            
    # Download from yfinance if not loaded
    if df is None:
        try:
            print(f"Downloading {symbol} data from Yahoo Finance ({start_date} to {end_date})...")
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                raise ValueError("Downloaded DataFrame is empty.")
                
            # Reset index so Date becomes a column
            df.reset_index(inplace=True)
            
            # Save a copy to local CSV for caching
            df.to_csv(csv_path, index=False)
            print(f"Dataset successfully downloaded and cached to: {csv_path}")
        except Exception as e:
            raise IOError(f"Error downloading data from yfinance: {e}")
            
    # Validate column requirements
    required_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # In newer yfinance versions, columns could be MultiIndex. We must normalize them.
    # We will do normalization in preprocessing, but let's check core presence here.
    # If df.columns has level 1, we look at the level 0 names.
    if isinstance(df.columns, pd.MultiIndex):
        col_names = [col[0] for col in df.columns]
    else:
        col_names = list(df.columns)
        
    missing = [c for c in required_cols if c not in col_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
        
    return df
