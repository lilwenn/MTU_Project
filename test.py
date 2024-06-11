import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar for training loops

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse

def Performance(y_test, predictions, name, metrics_data, predictions_data):

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    metrics_data.append({'Model': name,
                         'R^2': round(r2,3),
                         'MSE': round(mse,3),
                         'MAE': round(mae,3)})

    predictions_data[name] = predictions


def time_series_to_tabular(data):
    target_col = 'Ireland_Milk_Price'  # The column in data we want to forecast
    loopback = 6  # This is how far back we want to look for features
    horizon = 3  # This is how far forward we want to forecast

    # Separate datetime columns from others
    datetime_cols = data.select_dtypes(include=[np.datetime64]).columns
    other_cols = data.columns.difference(datetime_cols)

    # Fill in missing values for non-datetime columns
    data_non_datetime = data[other_cols]
    data_non_datetime = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data_non_datetime)
    data_non_datetime = pd.DataFrame(data_non_datetime, columns=other_cols, index=data.index)

    # Recombine data with datetime columns
    data = pd.concat([data[datetime_cols], data_non_datetime], axis=1)

    def create_lag_features(data, target, lag):
        """Create features for our ML model (X matrix).

        :param pd.DataFrame data: DataFrame
        :param str target: Name of target column (int)
        :param int lag: Lookback window (int)
        """
        for col in data.columns:
            for i in range(1, lag + 1):
                data[f'{col}-{i}'] = data[col].shift(i)

            # Drop non-target values (we only keep historical feature values)
            if col != target:
                data = data.drop(col, axis=1)

        # OPTIONAL: Drop first N rows where N = lag
        # Alternatively, we could impute the missing data
        data = data.iloc[lag:]
        return data

    def create_future_values(data, target, horizon):
        """Create target columns for horizons greater than 1"""
        targets = [target]
        for i in range(1, horizon):
            col_name = f'{target}+{i}'
            data[col_name] = data[target].shift(-i)
            targets.append(col_name)

        # Optional: Drop rows missing future target values
        data = data[data[targets[-1]].notna()]
        return data, targets

    print('\nInitial data shape:', data.shape)

    # Create feature data (X)
    data = create_lag_features(data, target_col, loopback)
    print('\ndata shape with feature columns:', data.shape)

    # Create targets to forecast (y)
    data, targets = create_future_values(data, target_col, horizon)
    print('\ndata shape with target columns:', data.shape)

    # Separate features (X) and targets (y)
    y = data[targets]
    X = data.drop(targets, axis=1)
    print('\nShape of X (features):', X.shape)
    print('Shape of y (target(s)):', y.shape)

    # Adding temporal features
    try:
        X['hour'] = X.index.hour
        X['sin_hour'] = np.sin(2 * np.pi * X['hour'].astype(int) / 24.0)
        X['cos_hour'] = np.cos(2 * np.pi * X['hour'].astype(int) / 24.0)
        X.drop(columns=['hour'], inplace=True)  # Optional
    except AttributeError as e:
        print(f"Index attribute error: {e}. Ensure the index is a DateTimeIndex.")

    # Saving the features and targets to CSV
    X.to_csv('features.csv')
    y.to_csv('targets.csv')
    return X, y


data = pd.read_excel('Data.xlsx')
colonne_cible = 'Ireland_Milk_Price'
data = data.iloc[:, 1:]  
data = data.sort_values('Date')

time_series_to_tabular(data)

# 80% of data for training
date_split_index = int(0.8 * len(data))
print(date_split_index)

train = data.iloc[:date_split_index]
test = data.iloc[date_split_index:]

X_train = train.drop(columns=['Date', colonne_cible])
y_train = train[colonne_cible]
X_test = test.drop(columns=['Date', colonne_cible])
y_test = test[colonne_cible]


model = LinearRegression()
model.fit(X_train, y_train)


# List to store the performance metrics
metrics_data = []
predictions_data = {}
predictions = model.predict(X_test)

print(predictions)


# Calculer les métriques d'évaluation
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Afficher les métriques
print("R^2 score:", r2)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

