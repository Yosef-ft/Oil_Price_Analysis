import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays


log_dir = os.path.join(os.path.split(os.getcwd())[0], 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'Info.log')
log_file_error = os.path.join(log_dir, 'Error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s :: %(message)s',
                              datefmt="%Y-%m-%d %H:%M")
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)


# ANSI Escape code to make the printing more appealing
ANSI_ESC = {
    "PURPLE": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "ITALICS" :"\033[3m"
}


class DataUtils:
    # def __init__(self, data):
    #     self.data = data


    def load_data(self, file_name: str)->pd.DataFrame:
        '''
        Load the file name from the data directory

        Parameters:
            file_name(str): name of the file

        Returns:
            pd.DataFrame
        '''
        logger.debug("Loading data from file...")
        try:
            data = pd.read_csv(f"../data/{file_name}", low_memory=False)
            return data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
        


    def data_info(self, data) -> pd.DataFrame:
        '''
        Provides detailed information about the data, including:
            - Percentage of missing values per column
            - Number of missing values per column
            - Data types of the columns
        It also highlights:
            - The total number of rows and columns in the dataset
            - Columns with the most missing values
            - Columns with more than 50% missing values

        Parameters:
            data(pd.DataFrame): The dataset 
        
        Returns:
            info_df(pd.DataFrame)
        '''
        
        missing_values = data.isna().sum()
        missing_percent = round(data.isna().mean() * 100, 2)
        data_types = data.dtypes
        
        info_df = pd.DataFrame({
            "Missing Values": missing_values,
            "Missing Percentage": missing_percent,
            "Data Types": data_types
        })


        info_df = info_df[missing_percent > 0]
        info_df = info_df.sort_values(by='Missing Percentage', ascending=False)

        max_na_col = list(info_df.loc[info_df['Missing Values'] == info_df['Missing Values'].max()].index)
        more_than_half_na = list(info_df.loc[info_df['Missing Percentage'] > 50].index)
        

        print(f"\n{ANSI_ESC['BOLD']}Dataset Overview{ANSI_ESC['ENDC']}")
        print(f"---------------------")
        print(f"- {ANSI_ESC['ITALICS']}Total rows{ANSI_ESC['ENDC']}: {data.shape[0]}")
        print(f"- {ANSI_ESC['ITALICS']}Total columns{ANSI_ESC['ENDC']}: {data.shape[1]}\n")

        duplicated_rows = int(data.duplicated().sum())
        if duplicated_rows == 0:
            print(f"{ANSI_ESC['GREEN']}No Duplicated data found in the dataset.{ANSI_ESC['ENDC']}\n")
        else:
             print(f"- {ANSI_ESC['RED']}Number of duplicated rows are{ANSI_ESC['ENDC']}: {duplicated_rows}")
        
        if info_df.shape[0] > 0:
            print(f"{ANSI_ESC['BOLD']}Missing Data Summary{ANSI_ESC['ENDC']}")
            print(f"------------------------")
            print(f"- {ANSI_ESC['ITALICS']}Columns with missing values{ANSI_ESC['ENDC']}: {info_df.shape[0]}\n")
            
            print(f"- {ANSI_ESC['ITALICS']}Column(s) with the most missing values{ANSI_ESC['ENDC']}: `{', '.join(max_na_col)}`")
            print(f"- {ANSI_ESC['RED']}Number of columns with more than 50% missing values{ANSI_ESC['ENDC']}: `{len(more_than_half_na)}`\n")


            if more_than_half_na:
                print(f"{ANSI_ESC['BOLD']}Columns with more than 50% missing values:{ANSI_ESC['ENDC']}")
                for column in more_than_half_na:
                    print(f"   - `{column}`")
            else:
                print(f"{ANSI_ESC['GREEN']}No columns with more than 50% missing values.{ANSI_ESC['ENDC']}")
        else:
            print(f"{ANSI_ESC['GREEN']}No missing data found in the dataset.{ANSI_ESC['ENDC']}")

        print(f"\n{ANSI_ESC['BOLD']}Detailed Missing Data Information{ANSI_ESC['ENDC']}")
        print(info_df)

        return info_df
    

    def holiday_generator(self, data):
        logger.debug("Adding holiday column...")
        try:
            us_holidays = holidays.US()
            data['Holiday'] = data['Date'].apply(lambda x: us_holidays[x] if x in us_holidays else 'Not Holiday')
            return data
        
        except Exception as e:
            logger.error(f"Error in adding holiday column: {e}")
            return data
        

    def add_features(self, data):
        '''
        Funtion used to add features

        Parameter:
        ---------
            data(pd.DataFrmae)
        '''
        data['SMA_week'] = data['Price'].rolling(window=7).mean()  
        data['SMA_month'] = data['Price'].rolling(window=30).mean()  
        data['SMA_quarter'] = data['Price'].rolling(window=90).mean()  
        data['SMA_semi_yearly'] = data['Price'].rolling(window=180).mean()  
        data['SMA_yearly'] = data['Price'].rolling(window=360).mean()  

        data['std_week'] = data['Price'].rolling(window=7).std()  
        data['std_month'] = data['Price'].rolling(window=30).std()  
        data['std_quarter'] = data['Price'].rolling(window=90).std()  
        data['std_semi_yearly'] = data['Price'].rolling(window=180).std() 
        data['std_yearly'] =data['Price'].rolling(window=360).std()  

        return data        
    

    def convert_date_event(self, date_str):
        input_format = "%b%d.%Y"
        parsed_date = datetime.strptime(date_str, input_format)
        output_format = "%b %d, %Y"
        return parsed_date.strftime(output_format)    
    

    def convert_date_price(self, date_str):
        try:
            input_format = "%d-%b-%y"
            parsed_date = datetime.strptime(date_str, input_format)
        except:
            parsed_date = datetime.strptime(date_str, "%b %d, %Y")
                
        output_format = "%b %d, %Y"
        return parsed_date.strftime(output_format)
