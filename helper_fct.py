import os 
import pickle
import yaml
import yfinance as yf

def load_model(ticker:str, model_name:str, ticker_dir:str):
    
            
    model_name_path = os.path.join(ticker_dir, model_name )

    model_path = os.path.join(model_name_path  , 'model.pkl')
    scaler_X_path = os.path.join(model_name_path  , 'scaler_X.pkl')
    scaler_Y_path = os.path.join(model_name_path  , 'scaler_Y.pkl')
    yaml_path = os.path.join(model_name_path  , 'params.yaml')

            
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load Scalers           
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
                
    with open(scaler_Y_path, 'rb') as f:
        scaler_Y = pickle.load(f)

    # Load training parameters from YAML file
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    params = params

    return params, model, scaler_X, scaler_Y


def get_predict_data(ticker:str, interval: str, seq_length:int):

        stock = yf.Ticker(ticker)

        # Define the period to fetch enough data points
        interval_to_period = {
            '1m': '1d',
            '5m': '5d',
            '15m': '10d',
            '1h': '60d',
            '1d': 'max'
                }
        period = interval_to_period.get(interval, '1d')  # Default to '1d' if interval not found

        # Fetch historical data
        data = stock.history(period=period, interval=interval)
        # Get the last x candles
        last_x_candles = data.tail(seq_length)
            
        return last_x_candles