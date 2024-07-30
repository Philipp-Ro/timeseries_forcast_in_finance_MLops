import os
import pickle
import yaml
import yfinance as yf
import torch
import torch.nn as nn
import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import torch.optim as optim


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
# -lr (float) -> learning rate in float 0.001
# -num_epochs (int) -> number of epochs for train 100  
#
# model_params:
# -input_size (int) -> dimension of input size here 5 ['Open', 'High', 'Low', 'Volume', 'Close']
# -hidden_size (int) -> number hof hidden neurons her 50 
# -num_layers (int) -> number of layers number of layers in RNN here 2
# -output_size (int) -> dimension of output here 1 ['Close']
class RNN_model():
    def __init__(self, params, model, scaler_X, scaler_Y):
        super(RNN_model, self).__init__()  
        # inti the model parameters
        self.params = params
        #self.model_params =params['model_params']
        #self.data_params = params['data_params']
        #self.train_params = params['train_params']

        self.model = model 
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y 

    
    def predict(self, data_df):
        #get features 
        features = data_df[self.params['model_params']["features"]]
        #scale features
        features[features.columns] = self.scaler_X.transform(features[features.columns])

        #prepare for prediction 
        features = features.values
        features = torch.tensor(features, dtype=torch.float32)
        features = features.unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            # predict
            new_prediction = self.model(features)
            # inverse scale 
            new_prediction = self.scaler_Y.inverse_transform(new_prediction.numpy())
            # get the last element since models with output size n predict n minutes into the future

        # get timecode for prediction   
        current_time = data_df.index[-1]

        # adjust here ifyou choose differend candle size 
        predict_time = current_time + dt.timedelta(minutes=self.params['model_params']['output_size'])
        
        #return just the last value in the forecast list 
        # this is the value corresponding to the timestamp 
        return {'Datetime': predict_time, 'Prediction': new_prediction[0][-1]}
    

        
    def train_model(self):
        df,start_date_train, end_date_train = get_train_data(ticker=self.params['data_params']['ticker'],interval=self.params['data_params']['interval'])

        # normalizing the output
        df_target = df[['Close']]
        scaler_Y = MinMaxScaler()
        scaler_Y = scaler_Y.fit(df_target[df_target.columns])

        # normalizing the input 
        df = df[self.params['model_params']["features"]]
        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(df[df.columns])
        df[df.columns]= scaler_X.transform(df[df.columns])

        #Create sequences
        sequences, targets = create_sequences(df, seq_length=self.params['data_params']['seq_length'], predict_length=self.params['data_params']['predict_length'], target_column=self.params['data_params']['target_column'])
        
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
        model = RNN_price_predictor(self.params['model_params']['input_size'], self.params['model_params']['hidden_size'], self.params['model_params']['num_layers'], self.params['model_params']['output_size'])
        optimizer = optim.Adam(model.parameters(), lr=self.params['train_params']['lr'])
        criterion = nn.MSELoss()

        # create variables to save basic train result 
        train_loss_per_epoch = []
        test_loss_per_epoch = []
        train_results={}

        # Training loop
        num_epochs = self.params['train_params']['num_epochs']
        model.train()
        for epoch in range(num_epochs):
            # forward pass
            outputs = model(X_train)


            #optimizer.zero_grad()
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

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train_Loss: {train_loss.item():.4f}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Test_Loss: {test_loss.item():.4f}')
                

        self.params['train_params']['train_loss_per_epoch'] = train_loss_per_epoch
        self.params['train_params']['test_loss_per_epoch'] = test_loss_per_epoch
        self.params['train_params']['start_date_train'] = start_date_train
        self.params['train_params']['end_date_train'] = end_date_train

        # create model name 
        model_name=f"RNN_{dt.datetime.now().strftime('%d.%m.%Y__%H.%M')}"
        self.params['model_params']['model_name'] = model_name

        #set model and scalers in object 
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        return model, scaler_X, scaler_Y, self.params
    


def get_train_data(ticker: str, interval:str )-> pd.DataFrame:
    if interval =='1m':
        num_days = 3
    elif interval =='5m':
        num_days = 7
    elif interval =='1h':
        num_days =30
    else:
        return None
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=num_days)  
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    return stock_data, start_date, end_date


def create_sequences(df, seq_length, predict_length, target_column):
    sequences = []
    targets = []
   
    target_idx = df.columns.get_loc(target_column)
    data =df.values
    for i in range(len(data) - seq_length - predict_length ):
        sequence = data[i:i + seq_length]
        target = data[i + seq_length+predict_length, target_idx]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)




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