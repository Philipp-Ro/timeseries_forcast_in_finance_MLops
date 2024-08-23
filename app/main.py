
from fastapi.responses import  JSONResponse, FileResponse
from typing import List
from pathlib import Path
import model_classes.RNN_model_class as RNN_model_class
import model_classes.MLP_model_class as MLP_model_class
import pandas as pd
import logging
import os
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pickle
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error as MAE
import shutil
import data_api

app = FastAPI()

# set up logger for debugging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#set baser_dir to current path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#set static_dir for html and js file 
STATIC_DIR = os.path.join(BASE_DIR , "static")

# set up path for model DB
MODEL_DB_DIR =os.path.join(BASE_DIR, '..', 'Model_DB')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# set up basemodels
class Ticker(BaseModel):
    symbol: str

class PlotRequest(BaseModel):
    ticker: str
    model_list: List[str]

class TrainRequest(BaseModel):
    ticker: str
    num_epochs: int
    forecast_len: int

class DeleteModelRequest(BaseModel):
    ticker: str
    model: str

# define global variable for older predictions 
all_preds = {'Datetime': [], 'Prediction': [], 'Model_name': []}


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# predict and plot the data 
@app.post("/generate_plot/")
async def predict_stock(request: PlotRequest):
    ticker = request.ticker
    model_list = request.model_list
    logger.debug(f"Model_list:${model_list}")
    global all_preds

    mae_dict = {}
    # load prediction data 
    data = data_api.get_predict_data(ticker=ticker, seq_length=60)
    
    if data.empty:
        logger.debug(f"REQUEST LIMIT from financialmodelingprep.com is full")
        response_content ={"figure": None,
        "MAE": None}
        return JSONResponse(content=response_content, status_code=429)
    
    plot_data = data[['Datetime','Close']]

    # refactor datetime to the same timezone for merging 
    #plot_data['Datetime'] = pd.to_datetime(plot_data['Datetime'], utc=True)
    #plot_data['Datetime'] = plot_data['Datetime'].dt.tz_convert('America/New_York')

    for model_name in model_list :
        # predict with the model
        prediction = predict_model(data=data, model_name=model_name)

        all_preds['Datetime'].append(prediction['Datetime'])
        all_preds['Prediction'].append(prediction['Prediction'])
        all_preds['Model_name'].append(model_name)


    # refactor df so that it has the structure:
    # Datetime      Model_name_A            Model_name_B              Close
    # date          prediction_model A      prediction_model B        actual_price
    df_plot = pd.DataFrame(all_preds)

    # refactor datetime to the same timezone for merging 
    #df_plot['Datetime'] = pd.to_datetime(df_plot['Datetime'], utc=True)
    #df_plot['Datetime'] = df_plot['Datetime'].dt.tz_convert('America/New_York')

    # merging both dataframes 
    df_plot = df_plot.drop_duplicates(subset=['Datetime', 'Model_name'])               
    pivoted_df = df_plot.pivot(index='Datetime', columns='Model_name', values='Prediction').reset_index()
    plot_data = pd.merge(pivoted_df, plot_data, on="Datetime", how='outer').sort_values(by='Datetime').reset_index(drop=True)


    # Iterate over model columns
    for model_col in [col for col in plot_data.columns if col in model_list]:

        # Drop rows where 'Close' or the current model column are NaN
        df_clean = plot_data.dropna(subset=['Close', model_col])

        # Extract true values and predicted values
        y_true = df_clean['Close']
        y_pred = df_clean[model_col]
        
        if df_clean.empty:
            mae = None
            
        else :
            # Calculate the absolute error
            mae = MAE(y_true,y_pred)

        mae_dict[model_col] = mae
    
 

    logger.debug(f"PLOT DATA HERE ")
    fig = plot_predictions(plot_data, model_list)
    logger.debug(f"FIGURE CREATED ")
    fig_json = fig.to_json()

    # Create the response content
    response_content = {
        "figure": fig_json,
        "MAE": mae_dict
        }
            
    return JSONResponse(content=response_content, status_code=200)


def predict_model( data, model_name):
    if "RNN" in model_name:
        # set folder dir for model
        model = RNN_model_class.RNN_model()
        model.load_model(model_name=model_name,model_db_dir=MODEL_DB_DIR)
        prediction = model.predict(data)
    if "MLP" in model_name:
        model = MLP_model_class.MLP_model()
        model.load_model(model_name=model_name,model_db_dir=MODEL_DB_DIR)
        prediction = model.predict(data)
        
    return prediction




@app.post("/fetchmodels")
async def list_models(ticker: Ticker):
    #reset all_preds for new ticker 
    stock_symbol = ticker.symbol
    logger.debug(f"fetching models for : {stock_symbol}")
    # check if ticker sybol is valid 
    logger.debug(f"Fetching models for: {stock_symbol}")

    
    # List all folders in the base directory that start with the ticker string
    models = [folder for folder in os.listdir(MODEL_DB_DIR)
                       if os.path.isdir(os.path.join(MODEL_DB_DIR, folder)) and folder.startswith(stock_symbol)]
    

    mae_dict = {model: None for model in models}

    response_content = {
        "ticker":ticker.symbol,
        "model_list": models,
        "mae_dict": mae_dict
        }
    logger.debug(f"CONTENT: {response_content}")         
    return JSONResponse(content=response_content, status_code=200)
     


        


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


    
    RNN_model = RNN_model_class.RNN_model()
    model, scaler_X, scaler_Y, params = RNN_model.train_model(train_params=params)

    save_training(model, scaler_X, scaler_Y, params)

    return JSONResponse(content={"message": "Model trained successfully!"}, status_code=200)
   


def plot_predictions(plot_data, model_names):
    plot_data = plot_data.tail(60)
    # Create a Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the actual closing price line and scatter
    fig.add_trace(go.Scatter(
        x=plot_data['Datetime'],
        y=plot_data['Close'],
        mode='lines+markers',
        name='Closing Price',
        line=dict(color='blue'),
        marker=dict(color='blue')
    ))

    # Add the predicted values for each model
    if model_names:
        for name in model_names:
            fig.add_trace(go.Scatter(
                x=plot_data['Datetime'],
                y=plot_data[name],
                mode='lines+markers',
                name=name
            ))

    # Set x-axis and y-axis titles
    fig.update_layout(
        title="Stock Price Predictions",
        xaxis_title="Time",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white",
        autosize=True,  
        height=None,
        width=None
    )

    # Customize x-axis ticks (rotate labels)
    fig.update_xaxes(tickangle=45)

    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    return fig
   

def save_training(model, scaler_X, scaler_Y, params):
        # before saving a new model check if the Model_db has more than 10 entrys if so delete all models before saving the trained one 
        manage_folders(MODEL_DB_DIR)      
        model_name = params['model_params']['model_name']

        models_directory=os.path.join(MODEL_DB_DIR,model_name)
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




@app.post("/delete_model/")
async def delete_model(request: DeleteModelRequest):
    ticker = request.ticker
    model = request.model
    ticker_dir = os.path.join(MODEL_DB_DIR,ticker)
    model_dir = os.path.join(ticker_dir, model)

    logger.debug(f"MODEL DIR   : {model_dir}")
    success = delete_folder(model_dir)
    
   
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")

    return JSONResponse(content=success, status_code=200)


def delete_folder(folder_path):
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted the folder and all its contents: {folder_path}")
            success = True
        except FileNotFoundError:
            print(f"Folder not found: {folder_path}")
            success = False
        except PermissionError:
            print(f"Permission denied: {folder_path}")
            success = False
        except OSError as e:
            print(f"Error: {e.strerror}")
            success = False
    else:
        print(f"The path is not a directory: {folder_path}")
        success = False
    return success

def manage_folders(base_dir):
    
    base_path = Path(base_dir)
    # Get a list of all directories in the base directory
    folders = [folder for folder in base_path.iterdir() if folder.is_dir()]
    
    # Check if the number of folders exceeds 10
    if len(folders) >= 10:
        print(f"Number of folders exceeds 10. Total folders: {len(folders)}")
        for folder in folders:
            delete_folder(folder)  # Delete each folder
        print("All folders have been deleted.")
    else:
        print(f"Number of folders is within limit. Total folders: {len(folders)}")


if __name__ == "__main__":
    import uvicorn
 
    uvicorn.run("main:app", host="127.0.0.1", port=8000)