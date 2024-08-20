import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from II_Data_visualization import plot_after_smoothing, plot_comparison, plot_decomposed_components
import Constants as const
from scipy.signal import savgol_filter


def load_and_preprocess_data(lag, df, target):
    """
    Load and preprocess data: clean, impute missing values, and create lagged features.
    
    Args:
    - k (int): Number of past periods to create lag features.
    - df (DataFrame): Raw DataFrame with the initial data.
    
    Returns:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    """
    output_file = f'spreadsheet/preproc_lag_{lag}.xlsx'
    
    if os.path.exists(output_file):
        return pd.read_excel(output_file)

    print('Preprocessing ... ...')
    print(f'Size of the initial dataset : {df.shape}')

    print('     - Imputing')
    impute_missing_values(df)


    print('     - Smoothing')
    if const.ACTION["time_series_smoothing"]:
        df = time_series_smoothing(const.SMOOTH_WINDOW, df)

    print('     - New feature creation')
    df = new_features_creation(df, target)


    print('     - Shifting')
    if const.ACTION["shifting"]:
        df = create_lag_features(df, lag, target)


    print('     - Multi-step Forecasting Features')
    if const.ACTION["Multi-step"]:
        df = create_multi_step_features(df, target, const.FORECAST_WEEKS)

    print(f'Size of the dataset after preprocessing : {df.shape}')

    # Sauvegarde du DataFrame final
    df.to_excel(output_file, index=False)
    print(f"Le fichier prétraité a été sauvegardé sous {output_file}.")

    return df


def new_features_creation(df,target):

    df['yield_per_supplier'] = df['litres'] / df['num_suppliers']
    df['cos_week'] = np.cos(df['Week'] * (2 * np.pi / 52))
    df['past_values'] = df[target].expanding().mean()


    return df

def impute_missing_values(df):
    # Separate columns with missing values
    columns_with_nan = df.columns[df.isna().any()].tolist()
    
    if not columns_with_nan:
        print("No missing values found in the DataFrame.")
        return df

    # Handle remaining missing values with IterativeImputer
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=100, random_state=42, tol=1e-3)
    df[columns_with_nan] = iterative_imputer.fit_transform(df[columns_with_nan])

    return df

def time_series_smoothing(window, data):
    """
    Perform time series analysis: handle missing values, decompose series, and create features.
    
    Args:
    - window (int): Number of periods for moving average and lag features.
    - data (DataFrame): DataFrame with time series data.
    - colonne_cible (str): Target column name.
    
    Returns:
    - data_final (DataFrame): DataFrame with time series features.
    """
    prep_data = pd.DataFrame(data)
    prep_data = prep_data.copy()

    exclude_cols = ['Date', 'Week', 'Grass_growth', 'litres', 'num_suppliers']
    impute_cols = prep_data.columns.difference(exclude_cols)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    prep_data[impute_cols] = imputer.fit_transform(prep_data[impute_cols])

    feature_cols = data.columns.difference(exclude_cols)

    if const.ACTION["compare lifting methods"]:

        data_MA = create_MA(data[feature_cols], past_time=window)
        data_WMA = create_WMA(data[feature_cols], window_size=20)
        data_EW = create_exponential_smoothing(data[feature_cols], span=window)

    if const.ACTION["time_series_smoothing"]:
        data_SG = create_savgol_smoothing(data[feature_cols], window_length=window, polyorder=2)
        data_SG = pd.concat([prep_data[exclude_cols], data_SG], axis=1)
    else:
        data_SG = data
    
    if const.ACTION["compare lifting methods"]:
        plot_comparison(data, data_MA, data_WMA, data_EW, data_SG, "Ireland_Milk_price" )

    return data_SG

def create_MA(data, past_time):
    new_data = data.copy() 
    
    ma_cols = []
    for col in new_data.columns:
        ma_col = new_data[col].rolling(window=past_time, min_periods=1).mean()
        ma_col.name = f'{col}_MA'
        ma_cols.append(ma_col)
    
    ma_data = pd.concat(ma_cols, axis=1)
    
    return ma_data

def create_savgol_smoothing(data, window_length, polyorder):
    new_data = data.copy()
    sg_cols = []

    for col in new_data.columns:
        smoothed_col = savgol_filter(new_data[col].bfill(), window_length=window_length, polyorder=polyorder, mode='interp')
        smoothed_col = pd.Series(smoothed_col, index=new_data.index, name=f'{col}')
        sg_cols.append(smoothed_col)
    
    
    sg_data = pd.concat(sg_cols, axis=1)
    
    return sg_data

def create_WMA(data, window_size):
    """
    Create Weighted Moving Average (WMA) features.
    
    Args:
    - data (DataFrame): DataFrame with time series data.
    - window_size (int): The size of the moving window.
    
    Returns:
    - new_data (DataFrame): DataFrame with WMA columns added.
    """
    new_data = data.copy()
    wma_cols = []
    
    for col in new_data.columns:
        weights = np.arange(1, window_size + 1)
        wma_col = new_data[col].rolling(window=window_size).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
        wma_col.name = f'{col}_WMA'
        wma_cols.append(wma_col)
    
    # Concatenate WMA columns with the original data
    wma_data = pd.concat(wma_cols, axis=1)
    new_data = wma_data.dropna()
    
    return new_data

def create_exponential_smoothing(data, span):
    """
    Create Exponential Smoothing features.
    
    Args:
    - data (DataFrame): DataFrame with time series data.
    - span (int): The span for the exponential smoothing.
    
    Returns:
    - new_data (DataFrame): DataFrame with exponential smoothing columns added.
    """
    new_data = data.copy()
    ewm_cols = []
    
    for col in new_data.columns:
        ewm_col = new_data[col].ewm(span=span).mean()
        ewm_col.name = f'{col}_EW'
        ewm_cols.append(ewm_col)
    
    # Concatenate EWM columns with the original data
    ewm_data = pd.concat(ewm_cols, axis=1)
    
    return ewm_data


def create_lag_features(data, lag, target):
    lagged_cols = []
    for col in data.columns:
        if col != "Date" and col != "Week" and col != target:
            for i in range(1, lag + 1):
                lagged_col = data[col].shift(i)
                lagged_col.name = f'{col}-{i}'
                lagged_cols.append(lagged_col)

    lagged_data = pd.concat(lagged_cols, axis=1)
    data = pd.concat([data, lagged_data], axis=1)


    return data

def preprocess_arima_data(df, forecast_weeks):
    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df.drop(columns=[col for col in df.columns if 'litres' in col])

    # Split exog_data into exog_train and exog_future
    exog_train = exog_data.iloc[:-forecast_weeks]
    exog_future = exog_data.iloc[-forecast_weeks:]

    # Ensure the number of columns in exog_future matches exog_train
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0)

    return exog_train, exog_future

def determine_optimum_lags(df, target_column, max_lag=40):
    """
    Determine the number of lags based on the autocorrelation function (ACF) and partial autocorrelation function (PACF).
    
    Args:
    df (pd.DataFrame): DataFrame containing the time series data.
    target_column (str): The name of the target column in the DataFrame.
    max_lag (int): The maximum number of lags to consider.
    
    Returns:
    int: The optimal number of lags.
    """
    series = df[target_column]
    
    # Plot ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=max_lag, ax=ax[0])
    plot_pacf(series, lags=max_lag, ax=ax[1])
    
    ax[0].set_title('Autocorrelation Function (ACF)')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.savefig(f"visualization/Autocorrelation_{target_column}.png")
    plt.close(fig)
    
    # Determine the optimal number of lags using acf function
    acf_values = acf(series, nlags=max_lag, alpha=0.05)[0]
    
    for lag in range(1, max_lag + 1):
        if abs(acf_values[lag]) < 1.96/np.sqrt(len(series)):
            optimal_lag = lag
            break
    else:
        optimal_lag = max_lag
    
    print(f"Optimal number of lags: {optimal_lag}")
    return optimal_lag

def create_multi_step_features(df, target_column, n_steps):
    """
    Create features for multi-step forecasting.
    
    Args:
    - df (DataFrame): The DataFrame containing the data.
    - target_column (str): The name of the target column.
    - n_steps (int): The number of steps to forecast into the future.
    
    Returns:
    - df (DataFrame): The DataFrame with new features for multi-step forecasting.
    """
    for step in range(1, n_steps + 1):
        df[f'{target_column}_step_{step}'] = df[target_column].shift(-step)
    
    df.dropna(inplace=True)
    return df
