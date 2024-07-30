import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from II_Data_visualization import plot_after_smoothing, plot_decomposed_components
import Constants as const


def load_and_preprocess_data(k, full_df):
    """
    Load and preprocess data: clean, impute missing values, and create lagged features.
    
    Args:
    - k (int): Number of past periods to create lag features.
    - full_df (DataFrame): Raw DataFrame with the initial data.
    
    Returns:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    """
    
    # Clean your data
    columns_to_drop = ['year_week', 'Week', 'EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price']
    full_df = full_df.drop(columns=columns_to_drop)

    # Define columns to impute
    columns_to_impute = ['yield_per_supplier']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    full_df[columns_to_impute] = imputer.fit_transform(full_df[columns_to_impute])

    # Drop columns with any remaining NaNs
    columns_with_nan = full_df.columns[full_df.isna().any()].tolist()
    full_df = full_df.drop(columns=columns_with_nan)

    # Create lagged features
    df = full_df.copy()
    for i in range(1, const.FORECAST_WEEKS + 1):
        df[f'{const.TARGET_COLUMN}_next_{i}weeks'] = df[const.TARGET_COLUMN].shift(-i)

    df = df.dropna()

    past_time = k

    if 'Date' in df.columns:
        df = df.set_index('Date')

    df = time_series_analysis(past_time, df, const.TARGET_COLUMN)
    df = df.reset_index()

    df.to_excel('spreadsheet/lagged_results.xlsx', index=False)
    return df

def preprocess_arima_data(df, forecast_weeks):
    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df.drop(columns=[col for col in df.columns if 'litres' in col])

    # Split exog_data into exog_train and exog_future
    exog_train = exog_data.iloc[:-forecast_weeks]
    exog_future = exog_data.iloc[-forecast_weeks:]

    # Ensure the number of columns in exog_future matches exog_train
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0)

    return exog_train, exog_future



def determine_lags(df, target_column, max_lag=40):
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



def create_lag_features(data, lag):
    new_data = data.copy()  

    lagged_cols = []
    for col in data.columns:
        # Create lagged columns for each original column
        for i in range(1, lag + 1):
            lagged_col = data[col].shift(i)
            lagged_col.name = f'{col}-{i}'
            lagged_cols.append(lagged_col)

    lagged_data = pd.concat(lagged_cols, axis=1)
    data = pd.concat([data, lagged_data], axis=1)

    data = data.iloc[lag:]

    return data


def create_MA(data, past_time):
    new_data = data.copy() 
    
    ma_cols = []
    for col in new_data.columns:
        # Create a moving average column for each original column
        ma_col = new_data[col].rolling(window=past_time).mean()
        ma_col.name = f'{col}_MA_{past_time}'
        ma_cols.append(ma_col)
    
    # Concatenate MA columns with the original data
    ma_data = pd.concat(ma_cols, axis=1)
    new_data = pd.concat([new_data, ma_data], axis=1)

    # Drop the first 'past_time' rows to avoid NaN values
    new_data = new_data.iloc[past_time:]
    
    return new_data

def time_series_analysis(past_time, data, colonne_cible):
    """
    Perform time series analysis: handle missing values, decompose series, and create features.
    
    Args:
    - past_time (int): Number of periods for moving average and lag features.
    - data (DataFrame): DataFrame with time series data.
    - colonne_cible (str): Target column name.
    
    Returns:
    - data_final (DataFrame): DataFrame with time series features.
    """
    
    # Ensure a copy of the data to avoid SettingWithCopyWarning
    data = data.copy()

    # Identify date and non-date columns
    date_col = data.select_dtypes(include=[np.datetime64]).columns
    other_cols = data.columns.difference(date_col)

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data[other_cols] = imputer.fit_transform(data[other_cols])

    # Keep the date column and the imputed data
    data = pd.concat([data[date_col], data[other_cols]], axis=1)

    # Create time series for the target column
    data_ireland = data[[colonne_cible]].copy()

    # Create a complete date range
    all_periods = pd.date_range(start=data_ireland.index.min(), end=data_ireland.index.max(), freq='MS')
    data_all_periods = pd.DataFrame(index=all_periods)

    # Merge to ensure all periods are included
    data_ireland = data_ireland.merge(data_all_periods, how='outer', left_index=True, right_index=True).fillna(0)
    
    # Decompose the time series
    decomposition = seasonal_decompose(data_ireland[colonne_cible], model='additive', period=12)

    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    decomposition.observed.plot(ax=ax1, title='Original Time Series: Milk Price in Ireland')
    decomposition.trend.plot(ax=ax2, title='Trend: Milk Price in Ireland')
    decomposition.seasonal.plot(ax=ax3, title='Seasonality: Milk Price in Ireland')
    decomposition.resid.plot(ax=ax4, title='Residual: Milk Price in Ireland')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'visualization/time_series_analysis_{colonne_cible}.png')
    plt.close(fig)

    # Create moving averages and lag features
    data_MA = create_MA(data, past_time)
    data_lagged = create_lag_features(data, past_time)

    # Concatenate the data
    data_final = pd.concat([data_lagged, data_MA], axis=1)
    data_final = data_final.loc[:, ~data_final.columns.duplicated()]
    data_final = data_final.reindex(sorted(data_final.columns), axis=1)

    return data_final
