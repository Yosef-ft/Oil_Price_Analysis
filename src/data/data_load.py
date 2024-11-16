import yaml
import argparse
import pandas as pd
import os
import sys
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scripts.Logger import LOGGER
logger = LOGGER


class Data:
    def __init__(self, config_path):
        self.config_path = config_path


    def data_load(self):

        with open(self.config_path) as conf_file:
            config = yaml.safe_load(conf_file)    

        # Create necessary directory if they don't exist
        if not os.path.exists(config['output']['report_dir']):
            os.makedirs(config['output']['report_dir'])

        if not os.path.exists(config['output']['model_dir']):
            os.makedirs(config['output']['model_dir'])    


        start_time = time.time()
        data = pd.read_csv(config['data']['dataset_csv'], index_col='Date')
        end_time = time.time()

        logger.info(f"Loading data took {round(end_time - start_time,2)}")


        return data

    def data_split(self, data: pd.DataFrame):

        with open(self.config_path) as conf_file:
            config = yaml.safe_load(conf_file)    

        test_size = config['data']['test_size']
        train = data[:-test_size]
        test = data[-test_size:]

        logger.info("Data successfully splited")
        return train, test
        



if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',dest='config', required=True)
    args = args_parser.parse_args()
    
    data_stage = Data(config_path=args.config)
    data_stage.data_load()


