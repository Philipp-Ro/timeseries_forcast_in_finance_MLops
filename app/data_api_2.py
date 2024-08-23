import requests
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from typing import List
# Financial Modeling Prep API settings
API_KEY = 'mkquVjFKrSfxSUcKrg4oUggKmW7q7Y1s'

def get_train_data(ticker: str, features_list:List) -> pd.DataFrame:

    url = f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={API_KEY}'
    
    # Make the API request
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Check if data is available
        if df.empty:
            raise ValueError("No data found for the specified ticker.")
        
        # change to pandas datime
        df['Datetime'] = pd.to_datetime(df['date'])
        # select necesarry columns and rename 
        df = df[['Datetime', 'open', 'close', 'high', 'low', 'volume']]
        df.columns = ['Datetime', 'Open', 'Close', 'High', 'Low', 'Volume']

        # Initialize MinMaxScaler for normalization
        scaler = MinMaxScaler()
        df[features_list] = pd.DataFrame(scaler.fit_transform(df[features_list]))
        
        # Keep only the last 1440 minutes
        df = df.head(1440)

        df  = df.reset_index(drop=True)
        # Extract the first and last candle datetime
        start_date = df['Datetime'].iloc[-1]  # Last record in the sorted DataFrame (oldest among the 1440)
        end_date = df['Datetime'].iloc[0]   # First record in the sorted DataFrame (newest among the 1440)
        
        return df[features_list],start_date, end_date
    
    elif response.status_code == 429:
        # if requestlimit for API is full
        return None, None, None



def get_predict_data(ticker: str, seq_length: int, features_list:List) -> pd.DataFrame:

    url = f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={API_KEY}'
    
    # Make the API request
    response = requests.get(url)
    print(response.status_code)
    if response.status_code == 200:
        data = response.json()
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
   
        # Check if data is available
        if df.empty:

            return None
        
        # change to pandas datime
        df['Datetime'] = pd.to_datetime(df['date'])
        # select necesarry columns and rename 
        df = df[['Datetime', 'open', 'close', 'high', 'low', 'volume']]
        df.columns = ['Datetime', 'Open', 'Close', 'High', 'Low', 'Volume']

        # get target scaler 
        target_scaler = MinMaxScaler()
        target_scaler.fit(df[['Close']])

        # norm the data
        scaler = MinMaxScaler()
        df[features_list] = pd.DataFrame(scaler.fit_transform(df[features_list]))


        # Keep only the last `seq_length` minutes
        df = df.head(seq_length)
        df  = df.reset_index(drop=True)

        predict_date = df['Datetime'].iloc[0]  

        # . values to convert the DF to an array
        return df[features_list].values, predict_date, target_scaler
    
    elif response.status_code == 429:

        return None, None, None
    


def create_sequences(data, seq_length:int, predict_length:int, target_column:str):

    sequences_norm = []
    targets_norm  = []

    targets_all = data[[target_column]]
    #for i in range(3):
    for i in range(len(data) - seq_length-predict_length ):    
        # extract sequence of length seq_length 
        sequence = data[i:i + seq_length].reset_index(drop=True)
        
        # get target_price which is the Close price n = predict_length steps after the sequence 
        # Get target price which is the target_column n = predict_length steps after the sequence
        target_price = targets_all.iloc[i + seq_length + predict_length].values[0]

        # . values to convert the DF to an array
        sequences_norm.append(sequence.values)
        targets_norm .append(target_price)

    return sequences_norm, targets_norm
