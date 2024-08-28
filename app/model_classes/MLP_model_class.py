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

class MLP_predictor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(MLP_predictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    

class MLP_model():
    def __init__(self, params=None, model=None):
        super(MLP_model, self).__init__()  
        # inti the model parameters
        self.params = params
        self.model = model 

        
    ###### structure of  model_params.yaml ######
    # model_params:
    # - model_name('str') -> modelname default None 
    # - features(List) -> list of columns relevant for the model from the raw data [['Close','Volume']]
    # - input_size (int) -> dimension of input size here 5 ['Open', 'High', 'Low', 'Volume', 'Close']
    # - output_size (int) -> dimension of output here 1 ['Close']
    # - hidden_size (int) -> number hof hidden neurons her 50 

    def predict(self, data, pred_date, target_scaler):
        # predicting on the given data 
        # input:
        # data : Dataframe with length 60 and the columns ['Datetime', 'Open', 'High', 'Low', 'Volume', 'Close']
        # pred_date: pandas dateime obj of the last candle in the prediction 
        # target_scaler: a sklearn_scaler to inversesclae the prediction of the model
        # output 
        # dictionary : {'Datetime': predict_time, 'Prediction': pred_price} 
        data = data[self.params['model_params']['features']].values
      
        features = extract_features(data)

        features = torch.tensor(list(features.values()), dtype=torch.float32)
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
        
    ###### structure of  training_params.yaml ######
    # data_params:
    # - ticker (string) -> ticker name as 'AAPL'
    # - interval (str) -> string for candle width here 1m for one minute
    # - num_days(int) -> number of days back in the training set 
    # - seq_len (int) -> length of sequence for prediction here 60
    # - predict_length (int) -> length of prediction here 1 but you can also increase 
    # - target_column (string) -> name of prediction column
    # train_params:
    # - lr (float) -> learning rate in float 0.001
    # - num_epochs (int) -> number of epochs for train 100  
        
    def train_model(self, train_params):
        # train a new model 
        # input:
        # train_params.yaml 
        # output 
        # trained model ,  self.params 
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_file_dir,'MLP_params.yaml')
        with open(yaml_path, 'r') as file:
            model_params = yaml.safe_load(file)

        train_params['model_params'] =model_params['model_params']

        # get train data :
        # df : Dataframe with length 60 and the columns ['Datetime', 'Open', 'High', 'Low', 'Volume', 'Close']
        # start_date_train: pandas dateime obj of the first candle in the training 
        # end_date_train: pandas dateime obj of the last candle in the training 
   
        df,start_date_train, end_date_train = data_api_2.get_train_data(ticker=train_params['data_params']['ticker'],
                                                                        features_list=train_params['model_params']["features"])
       
        #Create sequences
        sequences, targets = data_api_2.create_sequences(df, 
                                                         seq_length=train_params['data_params']['seq_length'], 
                                                         predict_length=train_params['data_params']['predict_length'], 
                                                         target_column=train_params['data_params']['target_column'])
        feature_matrix = []

        for x in sequences:
            new_features = extract_features(x) 
            feature_matrix.append(list(new_features.values()))
        
        # Split into training and testing sets 80% Train 20% Test
        train_size = int(len(feature_matrix) * 0.8)
        train_feature_matrix = feature_matrix[:train_size]
        train_targets = targets[:train_size]
        test_feature_matrix = feature_matrix[train_size:]
        test_targets = targets[train_size:]

        # Convert to PyTorch tensors
        X_train= torch.tensor(train_feature_matrix, dtype=torch.float32)
        Y_train = torch.tensor(train_targets, dtype=torch.float32)
        X_test= torch.tensor(test_feature_matrix, dtype=torch.float32)
        Y_test= torch.tensor(test_targets, dtype=torch.float32)
       
        # intt model optimizer and lossfct

        model = MLP_predictor(model_params['model_params']['input_size'], 
                              model_params['model_params']['hidden_size'])
        
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
        model_name =train_params['data_params']['ticker']+f"_MLP_{dt.datetime.now().strftime('%H.%M')}"
        self.params['model_params']['model_name'] = model_name

        #set model and scalers in object 
        self.model = model

        return model, self.params
    

    def load_model( self, model_name:str, model_db_dir:str):
        # safe the training in the Model_DB
        # the model and the params are safed in a new folder with the model name
        # model_name = <ticker>_<model_type>_< h:min trainstart>
        # model_name = AAPL_MLP_10.43
        
        model_name_path = os.path.join(model_db_dir, model_name )

        model_path = os.path.join(model_name_path  , 'model.pkl')
        yaml_path = os.path.join(model_name_path  , 'params.yaml')

                
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load training parameters from YAML file
        with open(yaml_path, 'r') as file:
            self.params = yaml.safe_load(file)


def extract_features(price_data):

    price_data = price_data.flatten()

    # Ensure the sequence is a Pandas Series for easier calculation
    seq = pd.Series(price_data)
    
    # Initialize a dictionary to store the features
    features = {}

    # Calculate Simple Moving Averages (SMA)
    features['SMA_5'] = seq.rolling(window=5).mean().iloc[-1]
    features['SMA_20'] = seq.rolling(window=20).mean().iloc[-1]
    features['SMA_50'] = seq.rolling(window=50).mean().iloc[-1]

    # Calculate Bollinger Bands
    rolling_mean = seq.rolling(window=20).mean().iloc[-1]
    rolling_std = seq.rolling(window=20).std().iloc[-1]
    features['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    features['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    # Calculate Relative Strength Index (RSI)
    delta = seq.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean().iloc[-1]
    avg_loss = pd.Series(loss).rolling(window=14).mean().iloc[-1]
    rs = avg_gain / avg_loss
    features['RSI_14'] = 100 - (100 / (1 + rs))

    return features





