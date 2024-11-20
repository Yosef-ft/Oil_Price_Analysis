import os
import sys
import yaml
import time
import argparse

import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, MaxPooling1D, Flatten,Reshape, SimpleRNN, RNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from tensorflow.keras.metrics import Accuracy, Precision, F1Score, Recall
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from scripts.Logger import LOGGER
from src.features.featurize import Featurize
logger = LOGGER

class ModelTrain:
    def __init__(self, config_path):    
        self.config_path = config_path
        self.feature_stage = Featurize(config_path=config_path)     
        
        with open(self.config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)    


    def setUp_mlflow(self):
        '''
        This function is used to setup mlflow by creating experiment Price_Models 
        '''
        logger.info("Setting up Mlflow")
        client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

        experiment_description = (
            "This is the oil price prediction project. "
            "This experiment contains the produce models for predicting oil price."
        )


        experiment_tags = {
            "project_name": "oil-price-Data-forecasting",
            "mlflow.note.content": experiment_description,
        }

        existing_experiment = {}

        all_experiments = client.search_experiments()
        for experiment in all_experiments:
            project_name = experiment.tags.get('project_name')
            if project_name == 'oil-price-Data-forecasting':
                existing_experiment['oil-price-Data-forecasting'] = experiment

        try:
            price_experiment = existing_experiment['oil-price-Data-forecasting']
            logger.info("Found existing experiment name: oil-price-Data-forecasting")

        except:
            price_experiment = client.create_experiment(
                name="Price_Models", tags=experiment_tags
            )
            logger.info("Creating new experiment name: oil-price-Data-forecasting")


        return client, price_experiment
    

    def model_config(self):

        model = Sequential()
        for layer_config in self.config['model']['layers']:
            if layer_config['name'] == 'InputLayer':
                model.add(InputLayer(shape=layer_config['shape']))
            elif layer_config['name'] == 'LSTM':
                model.add(LSTM(units=layer_config['units'], activation=layer_config['activation'], return_sequences=layer_config['return_sequences']))
            elif layer_config['name'] == 'Dense':
                model.add(Dense(units=layer_config['units'], activation=layer_config['activation']))
            elif layer_config['name'] == 'OutputLayer':
                model.add(Dense(units=layer_config['units'], activation=layer_config['activation']))

       
        model.compile(optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
                       loss=self.config['training']['loss'],
                         metrics=[metric for metric in self.config['training']['metrics']])
        model.summary()

        return model
    
    def train_model(self):

        model_name = 'LSTM'

        start_time = time.time()
        logger.info(f"Start training {model_name} model...")

        callback = EarlyStopping(
            monitor=self.config['training']['early_stopping']['monitor'],
            patience=self.config['training']['early_stopping']['patience'],
            mode=self.config['training']['early_stopping']['mode']
        )        

        X_train, y_train, X_test, y_test = self.feature_stage.create_sequence()
        client, price_experiment = self.setUp_mlflow()
        price_experiments =mlflow.set_experiment("Price_Models") 

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model = self.model_config()
        mlflow.keras.autolog()
        with mlflow.start_run(run_name=f'Oil_{model_name}'):
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=self.config['base']['epochs'], callbacks=[ callback],
                            batch_size = self.config['base']['batch_size'], verbose=self.config['base']['verbose'])

        end_time = time.time()

        logger.info(f"Training {model_name} took {round(end_time - start_time, 2)} seconds")  

        model.save(filepath=self.config['report']['model_path'])

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(self.config['data']['history_df'])

        return history, model           



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config', required=True)
    args = args_parser.parse_args()
    
    model_stage = ModelTrain(config_path=args.config)
    model_stage.train_model()