import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf



class Plots:

    def plot_seasonal_decompose(self, data_limiter: int, timeframe: int, data: pd.DataFrame):
        '''
        This function is used to plot seasonal decompose for price data

        Parameter:
        ---------
            data_limiter(int): a numerical value to limit the amount of data to be displayed
            timeframe(int): the time period for the plot
            data(pd.DataFrame)

        Return:
        ------
            matplotlip.pyplot object
        '''

        series = data['Price'][data_limiter:]
        result = seasonal_decompose(series, model='additive', period=timeframe)
        result.plot()


    def plot_data(self, data: pd.DataFrame):
        '''
        This funcion is used to plot line plot of the data

        Parameter:
        ----------
            data(pd.Dataframe)

        Return:
        ------
            matplotlip.pyplot object            
        '''

        sns.set_theme(style="whitegrid")  

        plt.figure(figsize=(12, 6)) 
        sns.lineplot(data=data, x=data.index, y='Price', color='b') 
        plt.title('Price Trend Over Time')  
        plt.xlabel('')  
        plt.ylabel('Price') 

        plt.xticks(rotation=45)  
        plt.tight_layout()  

        plt.show()         


    def plot_changePoint(self, price):
        price_array = price['Price'].values
        model = "rbf"
        algo = rpt.Pelt(model=model).fit(price_array)
        change_points = algo.predict(pen=20)

        plt.figure(figsize=(14, 7))
        plt.plot(price.index, price['Price'], label='Brent Oil Price')
        for cp in change_points[:-1]:
            plt.axvline(x=price.index[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title('Brent Oil Prices with Detected Change Points')
        plt.legend()
        plt.show()            

