import requests
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from typing import List
# Financial Modeling Prep API settings
API_KEY = 'mkquVjFKrSfxSUcKrg4oUggKmW7q7Y1s'

def get_train_data(ticker: str) -> pd.DataFrame:

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
        
        # Select and rename the necessary columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        # refactor datetime to the same timezone
        df['Datetime'] = pd.to_datetime(df['date'])


        df = df[['Datetime', 'open', 'close', 'high', 'low', 'volume']]
        df.columns = ['Datetime', 'Open', 'Close', 'High', 'Low', 'Volume']
        
        
        # Keep only the last 1440 minutes
        df = df.head(1440)
        df= df.dropna()
        df  = df.reset_index()
        # Extract the first and last candle datetime
        start_date = df['Datetime'].iloc[-1]  # Last record in the sorted DataFrame (oldest among the 1440)
        end_date = df['Datetime'].iloc[0]   # First record in the sorted DataFrame (newest among the 1440)
        
        return df,start_date, end_date
    
    elif response.status_code == 429:
        # if requestlimit for API is full
        return None, None, None



def get_predict_data(ticker: str, seq_length: int) -> pd.DataFrame:

    url = f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={API_KEY}'
    
    # Make the API request
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        
        # Check if data is available
        if df.empty:

            return None
        
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
    
    elif response.status_code == 429:

        return None
    
def create_sequences(df, seq_length:int, predict_length:int, target_column:str):
    print(df)
    sequences = []
    targets = []
   
    
    data =df.values
    for i in range(len(data) - seq_length - predict_length ):
        sequence = data[i:i + seq_length]
        target = data[target_column][i + seq_length+predict_length]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def create_sequences_2(df, seq_length:int, predict_length:int, target_column:str, features_list:List):
    print(df)
    sequences = []
    targets = []

    targets_all = df[[target_column]]

    # Initialize MinMaxScaler for normalization
    scaler = MinMaxScaler()

    #for i in range(len(df) - seq_length-predict_length ):
    for i in range(5):
        # extract sequence of length seq_length 
        sequence = df[i:i + seq_length].reset_index()
        print(sequence)
        
        datetime_col = df[['Datetime']]

        # Fit and transform the features
        sequence_normalized= scaler.fit_transform(sequence[features_list])
        #print(sequence_normalized)
        # Convert scaled features back to DataFrame
        df = pd.DataFrame(sequence_normalized, columns=features_list)

        # Concatenate 'Datetime' column back with the scaled features
        df= pd.concat([datetime_col, df], axis=1)
        # extract sequence of length seq_length 
        sequence = df[i:i + seq_length].reset_index()

        # get target_price which is the Close price n = predict_length steps after the sequence 
        # Get target price which is the target_column n = predict_length steps after the sequence
        target_price = targets_all.iloc[i + seq_length + predict_length].values[0]
        
        # Extract the last closing price in the current sequence
        last_close = df['Close'].iloc[i + seq_length - 1]
        # target is the factor we multiply the last closing price in sequence to get the target price 
        # target_price = target * last_close 
        target =last_close/target_price
        sequences.append(sequence_normalized)
        targets.append(target)

    return sequences, targets
