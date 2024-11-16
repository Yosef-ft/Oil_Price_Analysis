import argparse
import yaml
import os
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.data_stage import Data
from scripts.Logger import LOGGER
logger = LOGGER

class Featurize(Data):
    def __init__(self, config_path):
        super().__init__(config_path=config_path)

        with open(self.config_path) as conf_file:
            self.config = yaml.safe_load(conf_file) 

    def scale_data(self):

        scaler = MinMaxScaler(feature_range=(0, 1))
        train: pd.DataFrame = pd.read_csv(self.config['data']['train_split'], index_col= 'Date')
        test: pd.DataFrame = pd.read_csv(self.config['data']['test_split']  , index_col= 'Date')
        train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
        test_scaled = scaler.transform(test.values.reshape(-1, 1))    

        logger.info("Successfully scaled train, test split.")

        return train_scaled, test_scaled    

    def create_sequence(self):

        data = super().data_load()
        time_steps = self.config['data']['time_steps']

        train_scaled, test_scaled = self.scale_data()
        
        X_train, y_train = [], []
        for i in range(len(train_scaled) - time_steps):
            X_train.append(train_scaled[i:(i + time_steps)])
            y_train.append(train_scaled[i + time_steps])

        X_test, y_test = [], []
        for i in range(len(test_scaled) - time_steps):
            X_test.append(test_scaled[i:(i + time_steps)])
            y_test.append(test_scaled[i + time_steps])     

        logger.info("Successfully created sequence for LSTM model")                   

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config', required=True)
    args = args_parser.parse_args()
    
    feature_stage = Featurize(config_path=args.config)
    feature_stage.create_sequence()