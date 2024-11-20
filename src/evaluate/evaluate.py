import os
import sys
import yaml
import json
import argparse

import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from scripts.Logger import LOGGER
from src.features.featurize import Featurize
logger = LOGGER


def evaluate_metrics(config_path: str):
    
    with open(config_path) as file:
        config = yaml.safe_load(file)

    model = load_model(config['report']['model_path'])

    feature_stage = Featurize('params.yaml')
    X_train, y_train, X_test, y_test = feature_stage.create_sequence()

    forecast = model.predict(X_test)
    actual_values = y_test

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_values, forecast)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual_values, forecast))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100

    # Print the results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}%")   

    metrics = {
    'mae': mae,
    'rmse': rmse,
    'mape' : mape
    }

    with open(config["report"]["metrics_file"], 'w') as mf:
        json.dump(
            obj=metrics,
            fp=mf,
            indent=4
        )  


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()

    evaluate_metrics(config_path=args.config)