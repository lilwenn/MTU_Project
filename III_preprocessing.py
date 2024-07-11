from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from II_visualization import plot_after_smoothing, plot_decomposed_components
from sklearn.ensemble import RandomForestRegressor


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from II_visualization import plot_after_smoothing, plot_decomposed_components
from sklearn.ensemble import RandomForestRegressor



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


    date_col = data.select_dtypes(include=[np.datetime64]).columns
    other_cols = data.columns.difference(date_col)

    data[other_cols] = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data[other_cols])
    data[other_cols] = pd.DataFrame(data[other_cols], columns=other_cols, index=data.index)

    data = pd.concat([data[date_col], data[other_cols]], axis=1)

    data_ireland = data[[colonne_cible]].copy()

    all_periods = pd.date_range(start=data_ireland.index.min(), end=data_ireland.index.max(), freq='MS')
    data_all_periods = pd.DataFrame(index=all_periods)


    data_ireland = data_ireland.merge(data_all_periods, how='outer', left_index=True, right_index=True).fillna(0)
    decomposition = seasonal_decompose(data_ireland[colonne_cible], model='additive', period=12)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    decomposition.observed.plot(ax=ax1, title='Original Time Series: Milk Price in Ireland')
    decomposition.trend.plot(ax=ax2, title='Trend: Milk Price in Ireland')
    decomposition.seasonal.plot(ax=ax3, title='Seasonality: Milk Price in Ireland')
    decomposition.resid.plot(ax=ax4, title='Residual: Milk Price in Ireland')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'visualization/time series analysis_{colonne_cible}.png')

    # Lissage

    data_MA = create_MA(data, past_time)
    data_lagged = create_lag_features(data, past_time)


    data_final = pd.concat([data_lagged, data_MA], axis=1)


    data_final = data_final.loc[:,~data_final.columns.duplicated()]
    data_final = data_final.reindex(sorted(data_final.columns), axis=1)

    return data_final



def feature_importance(X, y):
    """
    Calculate feature importances using a RandomForestRegressor.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Features matrix.
        y (numpy.ndarray or pandas.Series): Target variable.

    Returns:
        numpy.ndarray: Feature importances.
    """
    model = RandomForestRegressor()
    model.fit(X, y)

    return model.feature_importances_


def pearson_corr(X, y):
    """
    Calculate Pearson correlation coefficients between features and target variable.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Features matrix.
        y (numpy.ndarray or pandas.Series): Target variable.

    Returns:
        numpy.ndarray: Pearson correlation coefficients.
    """
    corr_matrix = np.corrcoef(X, y, rowvar=False)
    corr_with_target = np.abs(corr_matrix[:-1, -1])  
    return corr_with_target
