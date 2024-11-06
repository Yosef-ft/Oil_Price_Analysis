import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter as smooth
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, \
                            mean_absolute_error, max_error

import os
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import pmdarima as pm

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.seasonal.sarim
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style='whitegrid')
sns.set_context(context='paper',font_scale=1.5)

from scipy.signal import savgol_filter as smooth
import scipy.stats as stats

from Logger import LOGGER

logger = LOGGER


class Utils:
    def detrend_series(self, series):
        """Detrends a series using a linear regression model, returning slope and intercept."""
        
        m, b = np.polyfit(np.arange(len(series)), series, 1)
        return m, b    

    def read_data(self, csv_file, window):
        """
        Reads and preprocesses price data for time series analysis.

        Parameters:
        ----------
        csv_file : str
            Path to the CSV file containing price data with a 'Date' column.
        window : int
            Window size parameter for smoothing (smoothing window = 2 * window + 1).

        Returns:
        --------
        df : pd.DataFrame
            DataFrame containing detrended and smoothed price data.
        m : float
            Slope of the trend in the log-transformed data.
        b : float
            Intercept of the trend in the log-transformed data.
        ref : float
            Reference price for log transformation.
        """
        
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        
        
        logger.info(f'Time range: from {df.index.min().strftime('%d-%m-%Y')} to {df.index.max().strftime('%d-%m-%Y')}')

        # Log-transform the 'Price' column
        ref = df['Price'].iloc[0]
        df['log_price'] = np.log(df['Price'] / ref)

        # Detrend the log-transformed series
        m, b = self.detrend_series(df['log_price'])
        df['detrended_log_price'] = df['log_price'] - (b + np.arange(len(df)) * m)

        # Apply smoothing to the detrended data
        smooth_window = 2 * window + 1
        df[f'smoothed_detrended_log_price_w{window}'] = smooth(df['detrended_log_price'].values, smooth_window, 3)

        # Drop intermediate columns if not needed
        df.drop(columns=['log_price'], inplace=True)
    
        return df, m, b, ref        

    
    def mkfolder(self, folder: str) -> None:
        """Create a directory if it does not exist."""
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f'Error creating folder {folder}: {e}')

    
    def sliding_windows(self, ts: pd.Series, n_inputs: int, n_outputs: int, shift: int = 0) -> tuple:
        """Generate sliding windows from the time series data.

        Args:
            ts (pd.Series): The time series data.
            n_inputs (int): Number of input points for each window.
            n_outputs (int): Number of output points for each window.
            shift (int): Number of shift points for the output.

        Returns:
            tuple: Lists of input and output data windows.
        """
        x, y = [], []
        for i in range(0, len(ts) - n_inputs - n_outputs, n_outputs):
            _x = ts.iloc[i - shift:(i + n_inputs) - shift]
            _y = ts.iloc[i + n_inputs - shift:i + n_inputs + n_outputs - shift]
            x.append(_x)
            y.append(_y)
        return x, y

    
    def inverse_transform(self, data: np.ndarray, m: float, b: float, ref: float, start_index: int = 0) -> np.ndarray:
        """Perform inverse transformation on leveled and log-transformed data.

        Args:
            data (np.ndarray): The log-transformed data.
            m (float): Slope for the transformation.
            b (float): Intercept for the transformation.
            ref (float): Reference value for the transformation.
            start_index (int): Starting index for reshaping.

        Returns:
            np.ndarray: The inverse transformed data.
        """
        index = np.arange(start_index, len(data) + start_index).reshape(-1, 1)
        return np.exp(np.array(data).reshape(-1, 1) + (b + index * m)) * ref


    
    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the Mean Absolute Percentage Error (MAPE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: MAPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    
    def metric_df(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Generate a DataFrame containing model evaluation metrics.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            pd.DataFrame: DataFrame with evaluation metrics.
        """
        return pd.DataFrame({
            'MAPE': self.mape(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'ME': max_error(y_true, y_pred),
            'R2 score': r2_score(y_true, y_pred)
        }, index=[0])

    
    def metric_df_size(self, y_true: np.ndarray, y_pred: np.ndarray, _input: int, _shift: int, loss: float) -> pd.DataFrame:
        """Generate a DataFrame containing metrics with input size and shift.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            _input (int): Number of input points.
            _shift (int): Number of shift points.
            loss (float): Loss value.

        Returns:
            pd.DataFrame: DataFrame with evaluation metrics.
        """
        return pd.DataFrame({
            'Input': _input,
            'Shift': _shift,
            'MAPE': Utils.mape(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'ME': max_error(y_true, y_pred),
            'R2 score': r2_score(y_true, y_pred),
            'Loss': loss
        }, index=[0])

    
    def metric_summary(self, df: pd.DataFrame, column: str, alpha: float, folder: str, model_name: str, figname: str, figsize=(7, 3)) -> None:
        """Generate a summary of metrics and save the plot.

        Args:
            df (pd.DataFrame): DataFrame containing metrics.
            column (str): Column name for the metrics to summarize.
            alpha (float): Confidence level for the interval.
            folder (str): Directory to save the plot.
            model_name (str): Name of the model.
            figname (str): Figure name for the plot.
            figsize (tuple): Size of the figure.
        """
        # Compute 68% confidence interval for MAPE
        _mean = df[column].mean()
        _std = df[column].std()
        conf_int = stats.t.interval(confidence=alpha, 
                                     df=len(df) - 1, 
                                     loc=_mean, 
                                     scale=_std)

        delta = conf_int[1] - _mean

        # Plot MAPE distribution for different train-test splits
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        df[column].hist(bins=20, ax=ax)
        ax.set_title(f'Distribution of {figname} for {model_name} model: {column} = {round(_mean, 2)} +/- {round(delta, 2)}')
        ax.set_ylabel('Count')
        ax.set_xlabel(column)

        plt.tight_layout()
        plt.show()
        plt.close(fig)



class Timeseries:

    def fit_arima(self, params, X, Y, target, n_inputs, n_outputs, df_coeff, exog_x=None, exog_y=None, add_exog=False):
        """
        Fits an ARIMA model with optional exogenous variables, predicts future values, and stores coefficients.

        Parameters:
            params (tuple): ARIMA parameters (p, d, q).
            X (pd.DataFrame): Historical time series data for training.
            Y (pd.DataFrame): Target time series data for prediction.
            target (str): Column name of the target variable in X.
            n_inputs (int): Number of input observations.
            n_outputs (int): Number of output observations to predict.
            df_coeff (pd.DataFrame): DataFrame to store model coefficients.
            exog_x (pd.Series or np.ndarray, optional): Exogenous variables for training. Default is None.
            exog_y (pd.Series or np.ndarray, optional): Exogenous variables for prediction. Default is None.
            add_exog (bool): Whether to add exogenous variables. Default is False.

        Returns:
            y_pred (pd.Series): Predicted mean values.
            y_pred_cf (pd.DataFrame): Confidence intervals for predictions.
            d (int): Differencing order.
            df_coeff (pd.DataFrame): Updated DataFrame with model coefficients.
        """

        # Unpack ARIMA parameters
        p, d, q = params

        # Prepare exogenous variables if requested
        if add_exog and exog_x is not None and exog_y is not None:
            exog_x = exog_x.to_numpy().reshape(-1, 1) if isinstance(exog_x, pd.Series) else exog_x
            exog_y = exog_y.to_numpy().reshape(-1, 1) if isinstance(exog_y, pd.Series) else exog_y

        # Define and fit the SARIMAX model
        model = SARIMAX(
            X.reset_index()[target],
            exog=exog_x if add_exog else None,
            order=(p, d, q),
            seasonal_order=(1,1,1,7)
        )
        fitted_model = model.fit(maxiter=300, disp=False)

        # Store model coefficients
        df_coeff = pd.concat([df_coeff, pd.DataFrame(fitted_model.params).T], ignore_index=True)

        # Predict with confidence interval
        pred = fitted_model.get_prediction(
            exog=exog_y if add_exog else None,
            start=d,
            end=n_inputs + n_outputs - 1,
            dynamic=False
        )
        y_pred = pred.predicted_mean

        # Set original datetime index for prediction results
        combined_index = X.index.union(Y.index)[d:]
        y_pred.index = combined_index
        y_pred_cf = pred.conf_int()
        y_pred_cf.index = combined_index

        return y_pred, y_pred_cf, d, df_coeff


    def cv_plot(self, X_true, Y_true, y_pred, y_pred_cf, model_name, folder, img_num, slice_num=0):
        '''
        Function used to plot the train data, test data and predicted values
        '''
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Train data
        X_true.plot(marker='o', linewidth=0, markersize=3, color='black', label='Train Data', ax=ax)
        
        # Test data
        Y_true.plot(marker='o', linewidth=0, markersize=3, color='forestgreen', label='Test Data', ax=ax)
        
        # Predicted values
        y_pred.plot(linewidth=2.5, color='crimson', label='Predicted', ax=ax)
        ax.fill_between(y_pred_cf.index,
                        y_pred_cf.iloc[:, 0],
                        y_pred_cf.iloc[:, 1], 
                        color='lightgrey', alpha=0.5, label='95% Confidence Interval')
        
        fs = 14
        ax.set_ylabel('Transformed Price', fontsize=fs)
        ax.set_xlabel('Date', fontsize=fs)
        
        ax.legend(loc='upper left', fontsize=fs)
        ax.set_title(f"Crude Oil Price Prediction Using {model_name}", fontsize=fs + 2, fontweight='bold')
        
        ax.set_ylim(-1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        

        # Only show the first plot
        if slice_num != 1:
            plt.close(fig)
        else:
            plt.show()

    def cv_arima(self, df, cv_num, n_inputs, n_outputs, folder, 
                 target, params=(4, 1, 1), model_name='ARIMA',
                 add_exog=False, exog_col='detrended_log_price'):
        """This function does cv_num CV for fitted ARIMA model
        Exogenous variable can be introduced."""
        
        utils = Utils()

        # Define path to store results
        path_res = f'/{target}_train_{n_inputs}_test_{n_outputs}_cv_{cv_num}'
        
        if add_exog:
            path_res += f'{exog_col}'

        # Make folders
        utils.mkfolder(folder)
        folder += path_res
        utils.mkfolder(folder)
        utils.mkfolder(folder + '/results/')
        print(folder)
        
        # Generate slices of data for CV
        x_sm, y_sm = utils.sliding_windows(df[target], n_inputs=n_inputs, n_outputs=n_outputs) 
        x_true, y_true = utils.sliding_windows(df[target], n_inputs=n_inputs, n_outputs=n_outputs) 
        
        # Calculate lagged exogenous variable
        x_ex, y_ex = utils.sliding_windows(df[exog_col], shift=n_outputs, n_inputs=n_inputs, n_outputs=n_outputs)  
        
        # Define metric DataFrames
        df_metrics = pd.DataFrame()
        df_coeff = pd.DataFrame()
        df_true_pred = pd.DataFrame()
        
        print(f'Use {n_inputs} points to predict {n_outputs}.')
        
        # Select last cv_num slices to perform CV test
        _ind = len(x_true) - cv_num
        img = _ind  # Number of a slice to name an image
        
        for X, Y, X_true, Y_true, exog_x, exog_y in zip(x_sm[_ind:], y_sm[_ind:], 
                                                         x_true[_ind:], y_true[_ind:], 
                                                         x_ex[_ind:], y_ex[_ind:]):
            if not add_exog:
                exog_x = None
                exog_y = None
            
            # Fit the model and predict
            y_pred, y_pred_cf, d, df_coeff = self.fit_arima(params, X, Y, target, n_inputs, n_outputs,
                                                         df_coeff, exog_x, exog_y, add_exog)
            
            # Calculate the metric with respect to true values
            metrics_data = utils.metric_df(Y_true, y_pred[n_inputs - d:])
            df_metrics = pd.concat([df_metrics, metrics_data], ignore_index=True)
            
            # Save Y_true and y_pred
            true_pred_data = pd.concat([Y_true, y_pred[n_inputs - d:]], axis=1)
            df_true_pred = pd.concat([df_true_pred, true_pred_data])
                    
            # Show progress every 10th slice
            slice_num = img - _ind + 1
            if slice_num % 10 == 0:
                print(f'Done with slice number {slice_num} out of {cv_num}.')
            img += 1  
            
            # Plot train-test intervals
            self.cv_plot(X_true, Y_true, y_pred, y_pred_cf, model_name, folder, img, slice_num)

        # Save ARIMA coefficients
        df_coeff.index = np.arange(len(df_coeff))
        df_coeff.to_csv(f'{folder}/coeff.csv', index=False)
        
        # Plot coefficients
        # fig = df_coeff.plot()
        # plt.ylabel('coefficients')
        # plt.xlabel('slice number')
        # plt.savefig(f'{folder}/coeff.png')
        
        # Inverse transform to actual units
        utils = Utils()
        df, m, b, ref = utils.read_data(csv_file = '../data/BrentOilPrices.csv', 
                              window = 7)
        df_true_pred.columns = ['y_true', 'y_pred']
        start_index = df.reset_index()[df.reset_index().Date == df_true_pred.index.min()].index[0]
        df_true_pred.y_true = utils.inverse_transform(df_true_pred.y_true, m, b, ref, start_index)
        df_true_pred.y_pred = utils.inverse_transform(df_true_pred.y_pred, m, b, ref, start_index)
        
        # Plot true vs predicted
        fig = df_true_pred.plot(figsize=(12, 6), lw=3, color=['k', 'r'])
        plt.legend(('True', 'Predicted'))
        plt.ylabel('Price ($)')
        plt.title(f'Brent oil price prediction using ARIMA for {n_outputs} days.')
        plt.savefig(f'{folder}/compare.png')
        
        # Compute actual metrics based on data in actual units
        mape_actual = []
        rmse_actual = []

        for i in np.arange(0, len(df_true_pred), n_outputs):
            y_true = df_true_pred.y_true.iloc[i:i + n_outputs].values
            y_pred = df_true_pred.y_pred.iloc[i:i + n_outputs].values

            mape_actual.append(utils.mape(y_true, y_pred))
            rmse_actual.append(np.sqrt(mean_squared_error(y_true, y_pred)))

        df_metrics_actual = pd.DataFrame(dict(MAPE=mape_actual, RMSE=rmse_actual))

        ### Metric summary ###
        # alpha = 0.68
        # model_name = 'ARIMA'
        # utils.metric_summary(df_metrics_actual, 'MAPE', alpha, folder, model_name, 'MAPE')
        # utils.metric_summary(df_metrics, 'MAPE', alpha, folder, model_name, 'MAPE_tf')
        # utils.metric_summary(df_metrics_actual, 'RMSE', alpha, folder, model_name, 'RMSE')
        # utils.metric_summary(df_metrics, 'RMSE', alpha, folder, model_name, 'RMSE_tf')
        
        # Save metrics and summary
        df_metrics.to_csv(f'{folder}/metrics_transformed.csv', index=False)
        df_metrics_actual.to_csv(f'{folder}/metrics.csv', index=False)
        df_true_pred.to_csv(f'{folder}/results/y_true_y_pred.csv', index=True)   
        
        return df_metrics, df_coeff, df_true_pred            