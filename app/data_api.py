import requests
import pandas as pd

# Financial Modeling Prep API settings
API_KEY = 'mkquVjFKrSfxSUcKrg4oUggKmW7q7Y1s'

def get_train_data(ticker: str) -> pd.DataFrame:

    url = f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={API_KEY}'
    
    # Make the API request
    response = requests.get(url)
    data = response.json()
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    
    # Check if data is available
    if df.empty:
        raise ValueError("No data found for the specified ticker.")
    
    # Select and rename the necessary columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    # refactor datetime to the same timezone
    df['Datetime'] = pd.to_datetime(df['date'])


    df = df[['Datetime', 'open', 'close', 'high', 'low', 'volume']]
    df.columns = ['Datetime', 'Open', 'Close', 'High', 'Low', 'Volume']
    
    
    # Keep only the last 1440 minutes
    df = df.head(1440)
    df  = df.reset_index()
    # Extract the first and last candle datetime
    start_date = df['Datetime'].iloc[-1]  # Last record in the sorted DataFrame (oldest among the 1440)
    end_date = df['Datetime'].iloc[0]   # First record in the sorted DataFrame (newest among the 1440)
    
    return df,start_date, end_date

def get_predict_data(ticker: str, seq_length: int) -> pd.DataFrame:

    url = f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={API_KEY}'
    
    # Make the API request
    response = requests.get(url)
    data = response.json()
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(data)
    
    # Check if data is available
    if df.empty:
        raise ValueError("No data found for the specified ticker.")
    
    # Select and rename the necessary columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    # refactor datetime to the same timezone
    df['Datetime'] = pd.to_datetime(df['date'])


    df = df[['Datetime', 'open', 'close', 'high', 'low', 'volume']]
    df.columns = ['Datetime', 'Open', 'Close', 'High', 'Low', 'Volume']

    
    # Keep only the last `seq_length` minutes
    df = df.head(seq_length)
    df  = df.reset_index()
    return df