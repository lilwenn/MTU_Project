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
from II_visualization import plot_correlation_matrix  
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


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



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


def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape'])
    return model

def train_models(df, model_name):
    """
    Train multiple machine learning models on preprocessed data, evaluate their performance,
    and save results including MAPE, MAE, and predictions for each model, scaler, and scoring method combination.

    Args:
    - df (DataFrame): Preprocessed DataFrame obtained from load_and_preprocess_data().
    - result_file (str): File path where lagged results were saved.
    - features (list): List of feature columns for model training.
    - const (object): Object containing constants such as models, scalers, scoring methods, etc.

    Outputs:
    - Saves JSON files with evaluation metrics ('result/week_without_scale_weeks.json').
    - Saves correlation matrices for selected features as PNG files in 'visualization/correlation/'.
    """

    # Define features (use all columns except 'Date' and target columns)
    features = [col for col in df.columns if not col.startswith(f'{const.target_column}_next_') and col != 'Date']

    result = {}
    result[model_name] = {}

    if model_name == 'ARIMA':
        result[model_name] = {}

        # Call the function to get predictions and last date
        predictions, last_date = train_and_predict_arima(df)

        # Store predictions and last date in results
        result[model_name]['Predictions'] = predictions.tolist()
        result[model_name]['Last_Date'] = last_date.strftime('%Y-%m-%d')

        # Print the predictions and the last date for verification
        print("Predictions for the next 52 weeks:")
        print(predictions)
        print("Last date in the training data:", last_date)

        """        # Loop through forecast weeks and store each week's metrics
                for week in range(1, const.forecast_weeks + 1):
                    target_col = f'{const.target_column}_next_{week}weeks'
                    mape_score, mae_score, prediction, selected_features = train_and_predict(df, features, target_col, pipeline)
                    print(f'MAPE for ARIMA with {target_col}: {mape_score:.2f}')

                    # Store metrics in results
                    result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
                    result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
                    result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

                    # Plot correlation matrix for ARIMA
                    plot_correlation_matrix(df, const.target_column, [], output_file=f'visualization/correlation/ARIMA_{week}.png')

                    # Save results to JSON
                    with open(f'result/by_model/ARIMA_{week}week.json', 'w') as json_file:
                        json.dump(result[model_name], json_file, indent=4)"""


    elif model_name == 'NeuralNetworkTensorflow':
        for week in range(1, const.forecast_weeks + 1):
            target_col = f'{const.target_column}_next_{week}weeks'
            X = df[features]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Build the neural network model
            model = build_nn_model(X_train.shape[1])

            # Train the model
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

            # Evaluate the model
            mape_score, mae_score, prediction, selected_features = train_and_predict(df, features, target_col, model)
            print(f'MAPE for NeuralNetwork with {target_col}: {mape_score:.2f}')

            result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
            result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
            result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

            selected_features_set.update(selected_features)

        # Plot correlation matrix for NeuralNetwork
        plot_correlation_matrix(df, const.target_column , list(selected_features_set), output_file=f'visualization/correlation/NeuralNetwork_{week}.png')

    elif model_name == 'NeuralNetworkPyTorch':
        for week in range(1, const.lag + 1):
            target_col = f'{const.target_column}_next_{week}weeks'
            model = NeuralNetwork(input_dim=len(features))
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            mape_score, mae_score, prediction = train_and_predict_pytorch(df, features, target_col, model, criterion, optimizer)
            print(f'MAPE for NeuralNetworkPyTorch with {target_col}: {mape_score:.2f}')

            result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
            result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
            result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

            selected_features_set.update(selected_features)

        # Plot correlation matrix for NeuralNetworkPyTorch
        plot_correlation_matrix(df, const.target_column , list(selected_features_set), output_file=f'visualization/correlation/NeuralNetworkPyTorch_{week}.png')


    else:
    
        for scaler_name, scaler in const.scalers.items():
            result[model_name][scaler_name] = {}

            for scoring_name, scoring_func in const.scoring_methods.items():
                result[model_name][scaler_name][scoring_name] = {}
                result[model_name][scaler_name][scoring_name]['MAPE'] = {}
                result[model_name][scaler_name][scoring_name]['MAE'] = {}
                result[model_name][scaler_name][scoring_name]['Prediction'] = {}

                # Default scoring function and k value if scoring_func is None
                default_scoring_func = f_regression
                default_k = 5

                pipeline = Pipeline([
                    ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func, k=const.k_values.get(scoring_name, default_k))),
                    ('scaler', scaler),
                    ('model', model)
                ])

                selected_features_set = set()

                for week in range(1, const.forecast_weeks + 1):
                    target_col = f'{const.target_column}_next_{week}weeks'
                    mape_score, mae_score, prediction, selected_features = train_and_predict(df, features, target_col, pipeline)
                    print(f'MAPE for {model_name} with {target_col} and scoring {scoring_name}: {mape_score:.2f}')

                    result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
                    result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
                    result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

                    selected_features_set.update(selected_features)

                plot_correlation_matrix(df, const.target_column , list(selected_features_set), output_file=f'visualization/correlation/{model_name}_{week}{scaler_name}_{scoring_name}.png')

                with open(f'result/by_model/{model_name}_{week}week.json', 'w') as json_file:
                    json.dump(result[model_name], json_file, indent=4)


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


if __name__ == "__main__":
    """
    df = load_and_preprocess_data()
        #Entrainement des modeles
        df = load_and_preprocess_data()
        for model_name, model in const.models.items():
            train_models(df, model)
    
    df = pd.read_excel('spreadsheet/lagged_results.xlsx')

    train_models(df, 'ARIMA')
    #train_models(df, 'NeuralNetworkPyTorch')
    #train_models(df, 'NeuralNetworkTensorflow')

    """
    

if False: # Test autosklearn

    #pip install auto-sklearn
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120, # Temps limite en secondes (ici 2 minutes)
    per_run_time_limit=30, # Temps limite par modèle en secondes
    tmp_folder='/tmp/autosklearn_classification_example_tmp',
    output_folder='/tmp/autosklearn_classification_example_out',
    delete_tmp_folder_after_terminate=True,
    delete_output_folder_after_terminate=True,
    )
    automl.fit(X_train, y_train)

    y_hat = automl.predict(X_test)
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, y_hat))


    # autre maniere



    # Install FLAML library using a subprocess call, ignoring errors
    subprocess.call([sys.executable, "-m", "pip", "install", "flaml"])

    # Create a FLAML AutoML instance
    automl = AutoML()

    # Set the AutoML settings
    settings = {
        "time_budget": 60,  # total running time in seconds
        "metric": 'r2',  # primary metric for evaluating model performance
        "estimator_list": ['lgbm', 'rf', 'xgboost'], # list of ML learners; we exclude 'catboost' as it is not compatible with FLAML
        "task": 'regression',  # task type
        "log_file_name": 'flaml.log',  # log file name
        "seed": 1, # random seed
    }

    # Train the AutoML model
    automl.fit(X_train=X_train, y_train=y_train, **settings)

    # Make predictions on the test set
    y_pred = automl.predict(X_test)

    # Evaluate the model's performance
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared on test set: {r2:.2f}')

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class Const:
    forecast_weeks = 52
    target_column = 'litres'

const = Const()

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

def train_models(df, model_name):
    """
    Train multiple machine learning models on preprocessed data, evaluate their performance,
    and save results including MAPE, MAE, and predictions for each model, scaler, and scoring method combination.

    Args:
    - df (DataFrame): Preprocessed DataFrame obtained from load_and_preprocess_data().
    - result_file (str): File path where lagged results were saved.
    - features (list): List of feature columns for model training.
    - const (object): Object containing constants such as models, scalers, scoring methods, etc.

    Outputs:
    - Saves JSON files with evaluation metrics ('result/week_without_scale_weeks.json').
    - Saves correlation matrices for selected features as PNG files in 'visualization/correlation/'.
    """

    # Define features (use all columns except 'Date' and target columns)
    features = [col for col in df.columns if not col.startswith(f'{const.target_column}_next_') and col != 'Date']

    result = {}
    result[model_name] = {}

    if model_name == 'ARIMA':
        result[model_name] = {}

        # Preprocess exogenous data
        exog_train, exog_future = preprocess_exog_data(df, const.forecast_weeks)

        # Check if 'Date' is in the columns
        if 'Date' in df.columns:
            # Set the date as index
            df_copy = df.set_index('Date')
        else:
            # Assume 'Date' is the index
            df_copy = df.copy()

        best_order = (1, 1, 1)  # Hardcoding the ARIMA order as (1, 1, 1)

        # Call the function to get predictions and last date
        predictions, last_date = train_and_predict_arima_corrected(df, exog_future, best_order)

        # Store predictions and last date in results
        result[model_name]['Predictions'] = predictions.tolist()
        result[model_name]['Last_Date'] = last_date.strftime('%Y-%m-%d')

        # Print the predictions and the last date for verification
        print("Predictions for the next 52 weeks:")
        print(predictions)
        print("Last date in the training data:", last_date)

    return result

# Load data
df = pd.read_excel('spreadsheet/lagged_results.xlsx')

# Train models
train_models(df, 'ARIMA')
