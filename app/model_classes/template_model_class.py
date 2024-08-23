import os
import pickle
import yaml
import torch
import torch.nn as nn
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import torch.optim as optim
import pandas as pd
import data_api

class torch_model(nn.Module):
    def __init__(self, model_specs):
        super(torch_model, self).__init__()
        # init toch model 
    
    def forward(self, x):
        # code forward pass
        out = None
        return out
    
###### structure of params ###########
# data_params:
# - ticker (string) -> ticker name as 'AAPL'
# - interval (str) -> string for candle width here 1m for one minute
# - num_days(int) -> number of days back in the training set 
# - seq_len (int) -> length of sequence for prediction here 60
# - predict_length (int) -> length of prediction here 1 but you can also increase 
# - target_column (string) -> name of prediction column
#
# train_params:
# - lr (float) -> learning rate in float 0.001
# - num_epochs (int) -> number of epochs for train 100  
#
# model_params:
# - input_size (int) -> dimension of input size here 5 ['Open', 'High', 'Low', 'Volume', 'Close']
# - hidden_size (int) -> number hof hidden neurons her 50 
# - num_layers (int) -> number of layers number of layers in RNN here 2
# - output_size (int) -> dimension of output here 1 ['Close']
class name_model():
    def __init__(self, params=None, model=None, scaler_X=None, scaler_Y=None):
        super(name_model, self).__init__()  
        # inti the model parameters
        self.params = params
        self.model = model 
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y 

    
    def predict(self, data_df):
        # Get features
        features = data_df[self.params['model_params']["features"]]
        
        # Scale features

        # add manual feature extraction if nesserary
        
        # Prepare for prediction
        features = features.values
        features = torch.tensor(features, dtype=torch.float32)
        features = features.unsqueeze(0)
        
        self.model.eval()

        with torch.no_grad():
            # Predict
            new_prediction = self.model(features)
            # Inverse scale
            new_prediction = self.scaler_Y.inverse_transform(new_prediction.numpy())
            
        # Get timecode for prediction
        latest_row = data_df.iloc[0]  
        current_time = pd.to_datetime(latest_row['Datetime'])  
        
        # Adjust for different candle size
        predict_time = current_time + dt.timedelta(minutes=self.params['model_params']['output_size'])
        
        # Return just the last value in the forecast list
        # This is the value corresponding to the timestamp
        return {'Datetime': predict_time, 'Prediction': new_prediction[0][-1]}
        

        
    def train_model(self, train_params):
   
        df,start_date_train, end_date_train = data_api.get_train_data(ticker=train_params['data_params']['ticker'])
      

        #Create sequences
        sequences, targets = data_api.create_sequences(df, seq_length=train_params['data_params']['seq_length'], predict_length=train_params['data_params']['predict_length'], target_column=train_params['data_params']['target_column'])
        
        # add feature engineering if model requires it 

        # Split into training and testing sets 80% Train 20% Test
        train_size = int(len(sequences) * 0.8)
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        test_sequences = sequences[train_size:]
        test_targets = targets[train_size:]

        # Convert to PyTorch tensors
        X_train= torch.tensor(train_sequences, dtype=torch.float32)
        Y_train = torch.tensor(train_targets, dtype=torch.float32)
        X_test= torch.tensor(test_sequences, dtype=torch.float32)
        Y_test= torch.tensor(test_targets, dtype=torch.float32)
       
        # intt model optimizer and lossfct
        model = torch_model(train_params['model_params']['input_size'], train_params['model_params']['hidden_size'], train_params['model_params']['num_layers'], train_params['model_params']['output_size'])
        optimizer = optim.Adam(model.parameters(), lr=train_params['train_params']['lr'])
        criterion = nn.MSELoss()

        # create variables to save basic train result 
        train_loss_per_epoch = []
        test_loss_per_epoch = []
        
        # Training loop
        num_epochs = train_params['train_params']['num_epochs']
        model.train()
        for epoch in range(num_epochs):
            # forward pass
            outputs = model(X_train)

            #calculate loss 
            train_loss = criterion(outputs[:,-1], Y_train)
            train_loss_per_epoch.append(train_loss.item())

            #backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Evaluating the model after each epoch
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs[:,-1], Y_test)
                test_loss_per_epoch.append(test_loss.item())

                
        # safe train and test eval 
        train_params['train_params']['train_loss_per_epoch'] = train_loss_per_epoch
        train_params['train_params']['test_loss_per_epoch'] = test_loss_per_epoch

        # Convert to string and save the timestamp for first and last candle
        train_params['train_params']['start_date_train'] = start_date_train.isoformat()
        train_params['train_params']['end_date_train'] = end_date_train.isoformat()

        # create model name
        self.params =train_params
        model_name =train_params['data_params']['ticker']+f"_model_name_{dt.datetime.now().strftime('%H.%M')}"
        self.params['model_params']['model_name'] = model_name

        #set model and scalers in object 
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        
        return model, scaler_X, scaler_Y, self.params
    

    def load_model( self, model_name:str, model_db_dir:str):
     
        model_name_path = os.path.join(model_db_dir, model_name )

        model_path = os.path.join(model_name_path  , 'model.pkl')
        scaler_X_path = os.path.join(model_name_path  , 'scaler_X.pkl')
        scaler_Y_path = os.path.join(model_name_path  , 'scaler_Y.pkl')
        yaml_path = os.path.join(model_name_path  , 'params.yaml')

                
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load Scalers           
        with open(scaler_X_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
                    
        with open(scaler_Y_path, 'rb') as f:
            self.scaler_Y = pickle.load(f)

        # Load training parameters from YAML file
        with open(yaml_path, 'r') as file:
            self.params = yaml.safe_load(file)









