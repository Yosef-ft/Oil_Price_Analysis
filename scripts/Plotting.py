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


    def plot_changePoint(self, price, model:str):
        '''
        The model is used to plot changePoint

        Parameter:
        ---------
            price(pd.DataFrame)
            model(str): rbf, l1, l2
        '''
        price_array = price['Price'].values
        model = model
        algo = rpt.Pelt(model=model).fit(price_array)
        change_points = algo.predict(pen=20)

        plt.figure(figsize=(14, 7))
        plt.plot(price.index, price['Price'], label='Brent Oil Price')
        for cp in change_points[:-1]:
            plt.axvline(x=price.index[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title('Brent Oil Prices with Detected Change Points')
        plt.grid()
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


    def plot_ma(self, price: pd.DataFrame, sma: bool):
        '''
        Function to plot moving average and the std

        Parameter:
        ---------
            price(pd.DataFrame)
            sma(bool): if sma is true it plot the sma and if false it plots the STD 
        '''
        plt.figure(figsize=(14, 7))
        if sma:
            sns.lineplot(x=price.index, y='SMA_week', data=price, label='SMA Week')
            sns.lineplot(x=price.index, y='SMA_month', data=price, label='SMA Month')
            sns.lineplot(x=price.index, y='SMA_yearly', data=price, label='SMA Yearly')

            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title('Brent Oil Prices and SMAs Over Time')
            plt.legend(title='Simple Moving Averages')

        else:
            sns.lineplot(x=price.index, y='std_week', data=price, label='STD Week')
            sns.lineplot(x=price.index, y='std_month', data=price, label='STD Month')
            sns.lineplot(x=price.index, y='std_yearly', data=price, label='STD Yearly')

            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.title('Brent Oil Prices and STD Over Time')
            plt.legend(title='Standard deviation Moving Averages')
                    
        plt.show()        


    def plot_changePoint_with_event(self, data:pd.DataFrame):
        '''
        This function is used to plot Change point with evetns

        data(pd.DataFrame)
        text(bool): to show the events on the plot
        '''
        event_data = {
            '1999-10-01': 'OPEC production cuts take effect',
            '2004-07-01': 'Hurricane Ivan impacts U.S. production',
            '2005-08-01': 'Hurricane Katrina causes severe supply disruptions',
            '2007-05-01': "OPEC’s output policies influence prices",
            '2008-04-01': 'Global financial crisis begins affecting demand',
            '2008-10-01': 'Economic downturn leads to reduced demand',
            '2009-04-01': 'Early signs of economic recovery',
            '2011-01-03': 'Political unrest in the Middle East (Arab Spring)',
            '2014-10-01': 'OPEC decides not to cut production',
            '2015-01-02': 'Oil price crash due to oversupply',
            '2015-10-01': 'Global economic concerns affect demand',
            '2017-07-03': 'Increased U.S. inventories impact prices',
            # '2018-01-02': 'Strong global economic growth boosts demand',
            '2020-04-01': 'COVID-19 pandemic leads to historic price drop',
            '2020-07-01': 'Recovery in demand as economies reopen',
            '2021-01-04': 'Vaccine rollout boosts global economic outlook',
            '2022-01-03': 'Geopolitical tensions over Ukraine',
        }        

        fig, axs = plt.subplots(2, 1, figsize=(18, 15))

        # Plot with events
        plt.sca(axs[1])
        plt.plot(data.index, data['Price'], label='Brent Oil Price', color='blue')
        plt.title('Brent Oil Prices Over Time with event')
        plt.xlabel('Date')
        plt.ylabel('Price (USD per Barrel)')
        plt.legend()
        plt.grid()


        for date, event in event_data.items():
            event_date = pd.to_datetime(date)
            if event_date in data.index:
                price = data.loc[event_date, 'Price']
                plt.axvline(event_date, color='red', linestyle='--', alpha=0.6)
                plt.text(event_date, price, event, rotation=90, verticalalignment='center', fontsize=8, color='black', fontweight='bold')

        # Plot without events
        plt.sca(axs[0])
        plt.plot(data.index, data['Price'], label='Brent Oil Price', color='blue')
        plt.title('Brent Oil Prices Over Time without event')
        plt.xlabel('Date')
        plt.ylabel('Price (USD per Barrel)')
        plt.legend()
        plt.grid()


        for date, event in event_data.items():
            event_date = pd.to_datetime(date)
            if event_date in data.index:
                plt.axvline(event_date, color='red', linestyle='--', alpha=0.6)

        plt.tight_layout()  
        plt.show()


    def compare_indicator_price(self, event: pd.DataFrame, price_with_indicators,year =2012, full_year = False):

        event_data = price_with_indicators.loc[price_with_indicators['title'] == event] 

        if full_year:
            event_data = event_data.loc[event_data['Date'].dt.year >= year]  
        
        else:
            event_data = event_data.loc[event_data['Date'].dt.year == year]   
            
        # unemplyment, cpi
        event_data.loc[:, 'actual'] = event_data['actual'].apply(lambda x: x.replace("%", ''))
        event_data.loc[:, 'forecast'] = event_data['forecast'].apply(lambda x: x.replace("%", ''))
        event_data.loc[:, 'previous'] = event_data['previous'].apply(lambda x: x.replace("%", ''))  

        # storage
        event_data.loc[:, 'actual'] = event_data['actual'].apply(lambda x: x.replace("B", ''))
        event_data.loc[:, 'forecast'] = event_data['forecast'].apply(lambda x: x.replace("B", ''))
        event_data.loc[:, 'previous'] = event_data['previous'].apply(lambda x: x.replace("B", ''))      

        # inventory
        event_data.loc[:, 'actual'] = event_data['actual'].apply(lambda x: x.replace("M", ''))
        event_data.loc[:, 'forecast'] = event_data['forecast'].apply(lambda x: x.replace("M", ''))
        event_data.loc[:, 'previous'] = event_data['previous'].apply(lambda x: x.replace("M", ''))
    

        event_data.set_index('Date', inplace=True)    

        fig, ax1 = plt.subplots(figsize=(10, 8))
        event_data['actual'] = event_data['actual'].astype(float)
        sns.lineplot(data=event_data, x=event_data.index, y='actual', ax=ax1, color="b")
        ax1.set_ylabel(event, color="b")
        ax1.tick_params(axis='y', labelcolor="b")

        # Create a second y-axis sharing the same x-axis for 'Price'
        ax2 = ax1.twinx()
        sns.lineplot(data=event_data, x=event_data.index, y='Price', ax=ax2, color="r")
        ax2.set_ylabel('Price', color="r")
        ax2.tick_params(axis='y', labelcolor="r")

        # Add titles and labels
        ax1.set_xlabel("Date")
        plt.title(f"{event} vs Price Over Time")

        plt.show()          