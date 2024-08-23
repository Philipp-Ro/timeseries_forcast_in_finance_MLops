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
import data_api_2

class RNN_price_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN_price_predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
###### structure of params ###########
# data_params:
# - ticker (string) -> ticker name as 'AAPL'
# - interval (str) -> string for candle width here 1m for one imnute
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
class RNN_model():
    def __init__(self, params=None, model=None, target_scaler=None):
        super(RNN_model, self).__init__()  
        # init the model parameters
        self.params = params
        self.model = model 
  
    
    def predict(self, data, pred_date, target_scaler):
        data = data[self.params['model_params']['features']].values
        features = torch.tensor(data, dtype=torch.float32)
        features = features.unsqueeze(0)
        
        self.model.eval()

        with torch.no_grad():
            # Predict
            prediction = self.model(features)

            
            pred_price = target_scaler.inverse_transform(prediction)

        current_time = pd.to_datetime(pred_date)  
        
        # Adjust for different candle size
        predict_time = current_time + dt.timedelta(minutes=self.params['model_params']['output_size'])
        
        # Return just the last value in the forecast list
        # This is the value corresponding to the timestamp
        return {'Datetime': predict_time, 'Prediction': pred_price[0][-1]}
        

        
    def train_model(self, train_params):
   
        df,start_date_train, end_date_train = data_api_2.get_train_data(ticker=train_params['data_params']['ticker'],
                                                                        features_list=train_params['model_params']["features"])

        # load RNN model params
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        yaml_path = os.path.join(current_file_dir,'RNN_params.yaml')
        with open(yaml_path, 'r') as file:
            model_params = yaml.safe_load(file)

        train_params['model_params'] = model_params['model_params']
      

     
        #Create sequences
        sequences, targets = data_api_2.create_sequences(df, 
                                                         seq_length=train_params['data_params']['seq_length'], 
                                                         predict_length=train_params['data_params']['predict_length'], 
                                                         target_column=train_params['data_params']['target_column'] )
        
        # Split into training and testing sets 80% Train 20% Test
        train_size = int(len(sequences) * 0.8)
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        test_sequences = sequences[train_size:]
        test_targets = targets[train_size:]

        # Convert to PyTorch tensors
        # . values to convert the DF to an array
        X_train= torch.tensor(train_sequences, dtype=torch.float32)
        Y_train = torch.tensor(train_targets, dtype=torch.float32)
        X_test= torch.tensor(test_sequences, dtype=torch.float32)
        Y_test= torch.tensor(test_targets, dtype=torch.float32)
       
        # intt model optimizer and lossfct
        model = RNN_price_predictor(train_params['model_params']['input_size'], train_params['model_params']['hidden_size'], train_params['model_params']['num_layers'], train_params['model_params']['output_size'])
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
                
        # save train data 
        train_params['train_params']['train_loss_per_epoch'] = train_loss_per_epoch
        train_params['train_params']['test_loss_per_epoch'] = test_loss_per_epoch

        # Convert to string and save the timestamp for first and last candle
        train_params['train_params']['start_date_train'] = start_date_train.isoformat()
        train_params['train_params']['end_date_train'] = end_date_train.isoformat()

        # create model name
        self.params = train_params
        model_name = train_params['data_params']['ticker']+f"_RNN_{dt.datetime.now().strftime('%H.%M')}"
        self.params['model_params']['model_name'] = model_name

        #set model and scalers in object 
        self.model = model

        
        return model, self.params
    

    def load_model( self, model_name:str, model_db_dir:str):
     
        model_name_path = os.path.join(model_db_dir, model_name )

        #set up paths 
        model_path = os.path.join(model_name_path  , 'model.pkl')
        yaml_path = os.path.join(model_name_path  , 'params.yaml')

                
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load training parameters from YAML file
        with open(yaml_path, 'r') as file:
            self.params = yaml.safe_load(file)









