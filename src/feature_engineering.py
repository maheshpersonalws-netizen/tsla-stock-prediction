import pandas as pd

def generate_features(df):
    """
    Generates lag and moving average features matching the original notebook.
    Also creates the Target column and drops rows with missing values due to lag/rolling.
    
    Parameters:
    - df (pd.DataFrame): Cleaned stock DataFrame.
    
    Returns:
    - pd.DataFrame: DataFrame containing engineered features and the target variable.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")
        
    df_features = df.copy()
    
    # 1. Lag features
    df_features['Close_Lag1'] = df_features['Close'].shift(1)
    df_features['Close_Lag2'] = df_features['Close'].shift(2)
    
    # 2. Technical indicators (Moving averages)
    df_features['MA7'] = df_features['Close'].rolling(window=7).mean()
    df_features['MA21'] = df_features['Close'].rolling(window=21).mean()
    
    # 3. Create target variable
    df_features['Target'] = df_features['Close']
    
    # 4. Drop rows with NaN (introduced by shifts and rolling means)
    df_features.dropna(inplace=True)
    df_features.reset_index(drop=True, inplace=True)
    
    return df_features
