from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

from Logger import LOGGER
logger = LOGGER

class ModelUtils:

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



    def train_neurals(self, model_name: str,X_train, y_train, X_val,y_val):
        '''
        This funcion is used to train neural networks

        Parameter:
        ----------
            model_name(str): The name of the model like LSTM, CNN, RNN
            X_train, y_train, X_val,y_val
        '''

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        times = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')



        price_experiments =mlflow.set_experiment("Price_Models") 

        if model_name == 'LSTM':
            model1 = Sequential()
            model1.add(InputLayer(shape=(30, 1)))

            model1.add(LSTM(64, activation='tanh', return_sequences=False))
            model1.add(Dense(8, activation='relu'))
            model1.add(Dense(1, activation='linear'))

            model1.compile(
                loss=MeanSquaredError(),
                optimizer=Adam(learning_rate=0.01),
                metrics=[MeanAbsoluteError(), RootMeanSquaredError(), MeanAbsoluteError()]
            )

        start_time = time.time()
        logger.info(f"Start training {model_name} model...")

        mlflow.keras.autolog()
        with mlflow.start_run(run_name=f'Oil_{model_name}'):
            history = model1.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, callbacks=[ callback], batch_size = 32, verbose=0)

        end_time = time.time()

        logger.info(f"Training {model_name} took {round(end_time - start_time, 2)} seconds")  

        return model1          


    def plot_evaluate_neurons(self, metrics: str):
        '''
        This function is used to evaluate trained neural network models

        Parameter:
        ----------
            metrics(str): 
        
        '''


        data = pd.read_csv(f'../report/{metrics}.csv')
        val_data = pd.read_csv(f'../report/val_{metrics}.csv')
        
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18,8))

        sns.set_style("whitegrid")

        sns.lineplot(data=data, x='step', y='value', hue='Run', palette='pastel', linewidth=2.5, ax=axes[0])
        sns.lineplot(data=val_data, x='step', y='value', hue='Run', palette='pastel', linewidth=2.5, ax=axes[1])

        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel(f'{metrics}', fontsize=12)
        axes[0].set_title(f'{metrics} vs. Epochs by model', fontsize=14)

        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel(f'{metrics}', fontsize=12)
        axes[1].set_title(f'val_{metrics} vs. Epochs by model', fontsize=14)

        plt.show();        