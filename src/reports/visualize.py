import os
import sys
import yaml
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'../../'))
from scripts.Logger import LOGGER
from src.features.featurize import Featurize
logger = LOGGER
sns.set_theme('notebook')

def plot_metrics(config_path: str):
    
    with open(config_path) as file:
        config = yaml.safe_load(file)

    history = pd.read_csv(config['data']['history_df'])        

    metrics = ['loss', 'mae', 'mse', 'RootMeanSquaredError']
    for metric in metrics:
        plt.figure(figsize=(12,7))
        sns.lineplot(data=history[f'{metric}'], label='Train', color='blue')
        sns.lineplot(data=history[f'val_{metric}'], label='Validation', color='orange')

        plt.title(f"{metric}", fontdict={'fontweight': 'bold', 'fontsize': 14})
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric}')
        plt.legend(loc='upper right')
        loss = plt.gcf()

        loss.savefig(config['report'][f'metrics_{metric}'])    

    logger.info('Successfully saved all images')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()

    plot_metrics(config_path=args.config)        