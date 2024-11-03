import pandas as pd
from pandas.plotting import lag_plot
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


    def plot_distribution(self, price):
        '''
        Function to plot the distrution of price

        Parameter:
        ---------
            Price(pd.Dataframe): price data
        '''
        sns.set_style("whitegrid")  
        sns.set_palette("pastel")    

        plt.figure(figsize=(15, 5))

        sns.histplot(data=price, x='Price', kde=True, bins=30, color='skyblue', alpha=0.7)

        plt.title('Distribution of Price', fontsize=18, fontweight='bold')
        plt.xlabel('Price', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)


        plt.xlim(price['Price'].min(), price['Price'].max())  
        plt.ylim(0, 1800)  

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()        


    def lag_plot(self, price_data: pd.DataFrame, period: int):
        '''
        Function to plot lag

        Paramter:
        --------
            price_data(pd.DataFrmae)
            period(int): lag period
        '''
        # Plot lag plot
        plt.figure(figsize=(8, 8))
        lag_plot(price_data['Price'], lag=period)
        plt.title(f'Lag Plot (Lag={period})')
        plt.xlabel('Price(t)')
        plt.ylabel('Price(t-1)')
        plt.grid()
        plt.show()  


