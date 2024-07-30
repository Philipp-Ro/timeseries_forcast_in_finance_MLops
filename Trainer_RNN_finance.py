import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


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
###### structure of all the inputs ###########
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

class RNN_timeseries_forecast():
    def __init__(self, data_params, train_params, model_params):
        super(RNN_timeseries_forecast, self).__init__()  
        self.data_params = data_params
        self.train_params = train_params
        self.model_params = model_params
        self.criterion = nn.MSELoss()
     
    def prepare_data(self):
        df = fetch_data(ticker=self.data_params['ticker'], interval=self.data_params['interval'], num_days=self.data_params['num_days'])

        # normalizing the output
        df_target = df[[ 'Close']]
        scaler_Y = MinMaxScaler()
        scaler_Y = scaler_Y.fit(df_target[df_target.columns])

        # normalizing the input 
        df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(df[df.columns])
        df[df.columns]= scaler_X.transform(df[df.columns])
        
     
        #Create sequences
        sequences, targets = create_sequences(df, seq_length=self.data_params['seq_length'], predict_length=self.data_params['predict_length'], target_column=self.data_params['target_column'])

        
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

        return X_train,Y_train, X_test,Y_test, scaler_X, scaler_Y

    def init_model(self):
        model = RNN_price_predictor(self.model_params['input_size'], self.model_params['hidden_size'], self.model_params['num_layers'], self.model_params['output_size'])
        optimizer = optim.Adam(model.parameters(), lr=self.train_params['lr'])
        return model, optimizer
    

    def fit(self, X_train,Y_train, X_test,Y_test):
        model, optimizer = self.init_model()
        # Loss and optimizer
        criterion = nn.MSELoss()
        train_loss = []
        # Training loop
        num_epochs = self.train_params['num_epochs']
        model.train()
        for epoch in range(num_epochs):
          
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, Y_train)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluating the model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test)
            print(f'Test Loss: {test_loss.item():.4f}')

        return model, train_loss, test_loss
    

def create_sequences(df, seq_length, predict_length, target_column):
    sequences = []
    targets = []
   
    target_idx = df.columns.get_loc(target_column)
    data =df.values
    for i in range(len(data) - seq_length - predict_length + 1):
        sequence = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + predict_length, target_idx]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def fetch_data(ticker: str, interval:str ,num_days:int) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)  
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

