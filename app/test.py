import data_api_2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import yaml
import model_classes.MLP_model_class as MLP_model_class
import model_classes.RNN_model_class as RNN_model_class
ticker ='AAPL'
features = ['Close']
features_1 =['Open','High','Low','Volume','Close']
seq_length = 60
predict_length = 20
#df_train,start_date_train, end_date_train = data_api_2.get_train_data(ticker=ticker,features_list=features_1 )
#df_predict, pred_date, target_scaler= data_api_2.get_predict_data(ticker=ticker, seq_length=seq_length, features_list=features_1)



forecast_len = predict_length
num_epochs = 60

#Create sequences


# Load training parameters from YAML file
yaml_path = 'train_params.yaml'

with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)

params['data_params']['ticker'] = ticker
params['train_params']['num_epochs'] = num_epochs
params['model_params']['output_size'] = forecast_len
params['model_params']['features'] = features_1




#feature_matrix = []
MLP_model = MLP_model_class.MLP_model()
model, params = MLP_model.train_model(train_params=params)
df_predict, pred_date, target_scaler = data_api_2.get_predict_data(ticker=ticker, seq_length=seq_length)
print(df_predict)
pred = MLP_model.predict(data=df_predict,pred_date=pred_date, target_scaler=target_scaler)

#RNN_model = RNN_model_class.RNN_model()
#model, params = RNN_model.train_model(train_params=params)
#df_predict, pred_date, target_scaler= data_api_2.get_predict_data(ticker=ticker, seq_length=seq_length, features_list=features_1)
#pred = RNN_model.predict(df_predict,pred_date=pred_date, target_scaler=target_scaler)
print(pred)
print(pred_date)