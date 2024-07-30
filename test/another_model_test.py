import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from II_Data_visualization import plot_correlation_matrix  
from III_preprocessing import time_series_analysis, determine_lags 
import Constants as const

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
#import autosklearn.classification
#from flaml import AutoML

from sklearn.metrics import r2_score
import sklearn.model_selection
import sklearn.datasets
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def iteration_mean(mean_results, result, iteration):
        
    for model_name in result.keys():
        mape_lists = []
        prediction_lists = []

        # Collect all MAPE and Prediction lists from all iterations
        for i in range(iteration + 1):
            mape_lists.append(result[model_name][i]['MAPE'])
            prediction_lists.append(result[model_name][i]['Prediction'])

        # Transpose lists to calculate mean for each week
        transposed_mape_lists = list(zip(*mape_lists))
        transposed_prediction_lists = list(zip(*prediction_lists))

        mean_mape = [np.mean(week_mape) for week_mape in transposed_mape_lists]
        mean_prediction = [np.mean(week_prediction) for week_prediction in transposed_prediction_lists]

        mean_results[model_name] = {
            'MAPE': mean_mape,
            'Prediction': mean_prediction
        }

    return mean_results


def train_and_predict(df, features, target_col, model_pipeline):
    X = df[features]
    y = df[target_col]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the RandomForestRegressor model
    model = RandomForestRegressor()

    # Update the pipeline with the instantiated model
    model_pipeline.steps[-1] = ('model', model)

    # Train the model pipeline
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)

    # Evaluate the model using MAPE and MAE
    mape_score = mape(y_test, y_pred)
    mae_score = mae(y_test, y_pred)

    # Get the selected features from SelectKBest
    selected_features = model_pipeline.named_steps['selectkbest'].get_support(indices=True)
    selected_feature_names = [features[i] for i in selected_features]

    # Predict future values
    last_row = df.iloc[-1][features]
    future = pd.DataFrame([last_row])
    prediction = model_pipeline.predict(future)

    return mape_score, mae_score, prediction[0], selected_feature_names

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

    optimal_lag = 2

    return optimal_lag


def load_and_preprocess_data():
    """
    Load data from Excel file, perform data cleaning by dropping specified columns,
    impute missing values in specified columns, and create lagged features.

    Args:
    - file_path (str): Path to the Excel file containing the data.
    - columns_to_drop (list): List of columns to drop from the DataFrame.
    - columns_to_impute (list): List of columns to impute missing values.

    Returns:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    - result_file (str): File path where lagged results will be saved.
    """
    print("Start preprocessing ... ...")
    # Load your data
    full_df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

    # Clean your data (if necessary)
    full_df = full_df.drop(columns=['Week','EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price'])

    # Define columns to impute
    columns_to_impute = ['yield_per_supplier']

    # Utilize SimpleImputer to impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    full_df[columns_to_impute] = imputer.fit_transform(full_df[columns_to_impute])

    columns_with_nan = full_df.columns[full_df.isna().any()].tolist()
    full_df = full_df.drop(columns=columns_with_nan)

    df = full_df

    for i in range(1, const.forecast_weeks + 1):
        df.loc[:, f'{const.target_column}_next_{i}weeks'] = df[const.target_column].shift(-i)

    df = df.dropna()

    past_time = determine_lags(df, const.target_column, max_lag=40)

    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)

    df = time_series_analysis(past_time, df, const.target_column)
    df.reset_index(inplace=True)

    df.to_excel('spreadsheet/lagged_results.xlsx', index=False)
    return df


def train_and_predict_pytorch(df, features, target_col, model, criterion, optimizer, epochs=50, batch_size=32):
    X = df[features].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        y_test = y_test_tensor.numpy()

    # Evaluate the model using MAPE and MAE
    mape_score = mape(y_test, y_pred)
    mae_score = mae(y_test, y_pred)

    # Predict future values
    last_row = df.iloc[-1][features].values
    last_row_tensor = torch.tensor(last_row, dtype=torch.float32).view(1, -1)
    prediction = model(last_row_tensor).item()

    return mape_score, mae_score, prediction


def train_and_predict_arima(df):
    # Check if 'Date' is in the columns
    if 'Date' in df.columns:
        # Set the date as index
        df_copy = df.set_index('Date')
    else:
        # Assume 'Date' is the index
        df_copy = df.copy()

    # Infer the frequency
    inferred_freq = pd.infer_freq(df_copy.index)

    # Resample if not weekly on Sundays
    if inferred_freq != 'W-SUN':
        df_copy = df_copy.resample('W-SUN').mean().fillna(df_copy.mean())

    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df_copy.drop(columns=[col for col in df_copy.columns if 'litres' in col])

    # Repeat the last row to match the number of forecast steps
    last_row_exog_data = np.tile(exog_data.iloc[[-1]].values, (const.forecast_weeks, 1))

    # Train the ARIMA model
    model = ARIMA(df_copy['litres'], exog=exog_data, order=(1, 1, 1), dates=df_copy.index, freq='W-SUN')
    model_fit = model.fit()

    # Predict the next const.forecast_weeks weeks
    predictions = model_fit.forecast(steps=const.forecast_weeks, exog=last_row_exog_data)

    # Get the last date in the training data
    last_date = df_copy.index.max()

    # Reset the index
    df_copy = df_copy.reset_index()

    return predictions, last_date


def find_best_combination(model_data):
    best_mape = float('inf')
    best_combination = None
    for scaler_name, scaler_data in model_data.items():
        for scoring_name, metrics in scaler_data.items():
            if 'MAPE' in metrics:
                mape = sum(metrics["MAPE"].values()) / len(metrics["MAPE"])
                if mape < best_mape:
                    best_mape = mape
                    best_combination = (scaler_name, scoring_name, metrics)
    return best_combination

def preprocess_exog_data(df, forecast_weeks):
    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df.drop(columns=[col for col in df.columns if 'litres' in col])

    # Split exog_data into exog_train and exog_future
    exog_train = exog_data.iloc[:-forecast_weeks]
    exog_future = exog_data.iloc[-forecast_weeks:]

    # Ensure the number of columns in exog_future matches exog_train
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0)

    return exog_train, exog_future

def train_and_predict_arima_corrected(df, exog_future, order=(1, 1, 1)):
    # Check if 'Date' is in the columns
    if 'Date' in df.columns:
        # Set the date as index
        df_copy = df.set_index('Date')
    else:
        # Assume 'Date' is the index
        df_copy = df.copy()

    # Infer the frequency
    inferred_freq = pd.infer_freq(df_copy.index)

    # Resample if not weekly on Sundays
    if inferred_freq != 'W-SUN':
        df_copy = df_copy.resample('W-SUN').mean().fillna(df_copy.mean())

    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df_copy.drop(columns=[col for col in df_copy.columns if 'litres' in col])

    # Ensure the number of columns in exog_future matches exog_data
    exog_future = exog_future.reindex(columns=exog_data.columns, fill_value=0)

    # Train the ARIMA model
    model = ARIMA(df_copy['litres'], exog=exog_data, order=order, dates=df_copy.index, freq='W-SUN')
    model_fit = model.fit()

    # Predict the next const.forecast_weeks weeks
    predictions = model_fit.forecast(steps=const.forecast_weeks, exog=exog_future)

    # Get the last date in the training data
    last_date = df_copy.index.max()

    # Reset the index
    df_copy = df_copy.reset_index()

    return predictions, last_date

def calculate_mape(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    return (abs(y_true - y_pred) / y_true).mean() * 100

def train_models(df, model_name):
    """
    Train multiple machine learning models on preprocessed data, evaluate their performance,
    and save results including MAPE, MAE, and predictions for each model, scaler, and scoring method combination.

    Args:
    - df (DataFrame): Preprocessed DataFrame.
    - model_name (str): Name of the model to train.
    - const (object): Object containing constants such as models, scalers, scoring methods, etc.

    Outputs:
    - Saves JSON files with evaluation metrics ('result/week_without_scale_weeks.json').
    - Saves correlation matrices for selected features as PNG files in 'visualization/correlation/'.
    """

    def save_results(results, model_name, forecast_weeks, suffix=""):
        with open(f'result/by_model/{model_name}_{forecast_weeks}week{suffix}.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)

    def calculate_metrics_and_save(predictions, actual_values, result, model_name, forecast_weeks):
        mae_score = mean_absolute_error(actual_values, predictions)
        mape_score = calculate_mape(actual_values, predictions)
        result[model_name]['Predictions'] = predictions.tolist()
        result[model_name]['MAE'] = mae_score
        result[model_name]['MAPE'] = mape_score
        return mae_score, mape_score

    # Define features (use all columns except 'Date' and target columns)
    features = [col for col in df.columns if not col.startswith(f'{const.target_column}_next_') and col != 'Date']
    result = {model_name: {}}

    if model_name == 'ARIMA':
        exog_train, exog_future = preprocess_exog_data(df, const.forecast_weeks)
        df_copy = df.set_index('Date') if 'Date' in df.columns else df.copy()
        best_order = (1, 1, 1)  # Hardcoding the ARIMA order as (1, 1, 1)

        predictions, last_date = train_and_predict_arima_corrected(df, exog_future, best_order)
        actual_values = df_copy['litres'].iloc[-const.forecast_weeks:].values

        mae_score, mape_score = calculate_metrics_and_save(predictions, actual_values, result, model_name, const.forecast_weeks)
        result[model_name]['Last_Date'] = last_date.strftime('%Y-%m-%d')
        save_results(result[model_name], model_name, const.forecast_weeks)
        print(f'MAPE for {model_name} with {const.target_column}: {mape_score:.2f}')

    else:
        for scaler_name, scaler in const.scalers.items():
            result[model_name][scaler_name] = {}

            for scoring_name, scoring_func in const.scoring_methods.items():
                result[model_name][scaler_name][scoring_name] = {}
                result[model_name][scaler_name][scoring_name]['MAPE'] = {}
                result[model_name][scaler_name][scoring_name]['MAE'] = {}
                result[model_name][scaler_name][scoring_name]['Prediction'] = {}

                # Instantiate RandomForestRegressor() here
                model = RandomForestRegressor()

                pipeline = Pipeline([
                    ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else f_regression, k=const.k_values.get(scoring_name, 5))),
                    ('scaler', scaler),
                    ('model', model)  # Use the instantiated model object
                ])

                selected_features_set = set()

                for week in range(1, const.forecast_weeks + 1):
                    target_col = f'{const.target_column}_next_{week}weeks'
                    mape_score, mae_score, prediction, selected_features = train_and_predict(df, features, target_col, pipeline)
                    print(f'MAPE for {model_name} with {target_col}, {scaler_name} and scoring {scoring_name}: {mape_score:.2f}')

                    result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
                    result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
                    result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

                    selected_features_set.update(selected_features)

                plot_correlation_matrix(df, const.target_column , list(selected_features_set), output_file=f'visualization/correlation/{model_name}_{scaler_name}_{scoring_name}_{week}.png')

                # Save results for each combination
                #save_results(result[model_name], model_name, const.forecast_weeks)

    return result

    return result
# Load data
df = load_and_preprocess_data()

# Train models
#train_models(df, 'ARIMA')
train_models(df, 'RandomForestRegressor')


#train_models(df, 'NeuralNetworkPyTorch')
