
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from typing import List, Optional


import RNN_model_class
import helper_fct
import pandas as pd
import io
import matplotlib.pyplot as plt
import logging
import os
import yaml
from fastapi import FastAPI,Form,Query,Depends
from fastapi.responses import JSONResponse
import pickle
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles



app = FastAPI()

# set up logger for debugging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#set baser_dir to current path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#set static_dir for html and js file 
STATIC_DIR = os.path.join(BASE_DIR , "static")

# set up path for model DB
MODEL_DB_DIR =os.path.join(BASE_DIR,"Model_DB")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")




# set up basemodels
class Ticker(BaseModel):
    symbol: str


class PredictRequest(BaseModel):
    ticker: str
    checked_models: List[str]


class TrainRequest(BaseModel):
    ticker: str
    num_epochs: int
    forecast_len: int

# define global variable for older predictions 
all_preds = {'Datetime': [], 'Prediction': [], 'Model_name': []}


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# predict and plot the data 
@app.post("/predict/")
async def predict_stock(request: PredictRequest):
    ticker = request.ticker
    checked_models = request.checked_models

    global all_preds
    logger.debug(f"Received ticker for predict: {ticker}")
    logger.debug(f"Received model names for predict: {checked_models}")

    
    # load prediction data 
    data = helper_fct.get_predict_data(ticker=ticker, interval='1m', seq_length=60)
    plot_data = data['Close'].reset_index()
    plot_data['Datetime'] = pd.to_datetime(plot_data['Datetime'])

    # Initialize all_preds if empty

    ticker_dir = os.path.join(MODEL_DB_DIR,ticker)
    for name in checked_models :
        params, model, scaler_X, scaler_Y = helper_fct.load_model(ticker, name, ticker_dir)
        model = RNN_model_class.RNN_model(params=params, model=model, scaler_X=scaler_X, scaler_Y=scaler_Y)
        prediction = model.predict(data)
        all_preds['Datetime'].append(prediction['Datetime'])
        all_preds['Prediction'].append(prediction['Prediction'])
        all_preds['Model_name'].append(name)

        

    df = pd.DataFrame(all_preds)
    df = df.drop_duplicates(subset=['Datetime', 'Model_name'])

    # Refactor df so that it has the desired structure
    pivoted_df = df.pivot(index='Datetime', columns='Model_name', values='Prediction').reset_index()
    plot_data = pd.merge(pivoted_df, plot_data, on="Datetime", how='outer').sort_values(by='Datetime').reset_index(drop=True)
    buf = plot_predictions(plot_data, checked_models)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/fetchmodels")
async def list_models(ticker: Ticker):
    #stock_symbol = ticker.symbol
    stock_symbol = ticker.symbol
    logger.debug(f"Fetching models for: {stock_symbol}")
    ticker_dir = os.path.join(MODEL_DB_DIR,stock_symbol)

    if os.path.exists(ticker_dir):
        logger.debug(f"Available models are : {os.listdir(ticker_dir)}")
        models = os.listdir(ticker_dir)
        
    else:
        logger.debug(f"No models found for: {stock_symbol}")
        models =[]
        
    return JSONResponse(content=models, status_code=200)  


        


@app.post("/train_new_model/")
def train_model(request:TrainRequest):
    ticker = request.ticker
    forecast_len = request.forecast_len
    num_epochs = request.num_epochs
    # Load training parameters from YAML file
    yaml_path = 'train_params.yaml'
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    logger.debug(f"Training model for: {ticker}")
    logger.debug(f"Model is trained with {num_epochs} Epochs")
    logger.debug(f"Model has a forcast length of : {forecast_len}")
    params['data_params']['ticker'] = ticker
    params['train_params']['num_epochs'] = num_epochs
    params['model_params']['output_size'] = forecast_len

    model = None
    scaler_X = None
    scaler_Y = None
    
    RNN_model = RNN_model_class.RNN_model(params, model, scaler_X, scaler_Y)
    model, scaler_X, scaler_Y, params = RNN_model.train_model()

    save_training(model, scaler_X, scaler_Y, params)

    return JSONResponse(content={"message": "Model trained successfully!"}, status_code=200)
   

def plot_predictions(plot_data, model_names):
   
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8.5)

    ax.scatter(plot_data['Datetime'], plot_data['Close'], label='Closing Price', color='blue')
    ax.plot(plot_data['Datetime'], plot_data['Close'], color='blue')


    if model_names:
        for name in model_names:
            ax.scatter(plot_data['Datetime'], plot_data[name], label=name)
            ax.plot(plot_data['Datetime'], plot_data[name])

    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return buf

def save_training(model, scaler_X, scaler_Y, params):
        ticker_dir = os.path.join(MODEL_DB_DIR,params['data_params']['ticker'])
        logger.debug(f"ticker dir  : {ticker_dir}")
        os.makedirs(ticker_dir, exist_ok=True)

        model_name = params['model_params']['model_name']

        models_directory=os.path.join(ticker_dir,model_name)
        os.makedirs(models_directory, exist_ok=True)
        
        
        with open(f"{models_directory}/model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        with open(f"{models_directory}/scaler_X.pkl", 'wb') as f:
            pickle.dump(scaler_X, f)
        
        with open(f"{models_directory}/scaler_Y.pkl", 'wb') as f:
            pickle.dump(scaler_Y, f)

        # Save training parameters to YAML
        yaml_filename = f"{models_directory}/params.yaml"
        with open(yaml_filename, 'w') as f:
            yaml.dump(params, f)


if __name__ == "__main__":
    import uvicorn
 
    uvicorn.run(app, host="127.0.0.1", port=8000)