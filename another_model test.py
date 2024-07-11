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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import autosklearn.classification
from flaml import AutoML
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


def train_and_predict_arima(df, target_col, p=1, d=1, q=1):
    """
    Trains an ARIMA model on time series data and makes a one-step-ahead prediction.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): Name of the column containing the target variable.
        p (int): Order of the autoregressive (AR) term. Default is 1.
        d (int): Degree of differencing. Default is 1.
        q (int): Order of the moving average (MA) term. Default is 1.

    Returns:
        tuple: A tuple containing:
            - MAPE score (float)
            - MAE score (float)
            - One-step-ahead forecast (float)
            - Residuals (list)
    """

    series = df[target_col]

    # Test for stationarity (optional, but recommended)
    adfuller_result = adfuller(series)
    if adfuller_result[1] > 0.05:  # If p-value > 0.05, the series is likely non-stationary
        print("Warning: Time series is likely non-stationary. Consider adjusting 'd'.")

    # Fit ARIMA model
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=1)[0]

    # Calculate in-sample predictions for error calculation
    in_sample_predictions = model_fit.predict()

    # Evaluate model
    mape_score = mean_absolute_percentage_error(series, in_sample_predictions)
    mae_score = mean_absolute_error(series, in_sample_predictions)

    # Get residuals
    residuals = model_fit.resid.tolist()

    return mape_score, mae_score, forecast, residuals

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

        for week in range(1, const.forecast_weeks + 1):
            target_col = f'{const.target_column}_next_{week}weeks'
            mape_score, mae_score, prediction, selected_features = train_and_predict_arima(df, target_col)
            print(f'MAPE for ARIMA with {target_col}: {mape_score:.2f}')

            result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
            result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
            result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

            # Plot correlation matrix for ARIMA
            plot_correlation_matrix(df, const.target_column, [], output_file=f'visualization/correlation/ARIMA_{week}.png')

            with open(f'result/by_model/ARIMA_{week}week.json', 'w') as json_file:
                json.dump(result[model_name], json_file, indent=4)

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


    #Entrainement des modeles
    df = load_and_preprocess_data()
    for model_name, model in const.models.items():
        train_models(df, model)

    train_models(df, 'ARIMA')
    train_models(df, 'NeuralNetworkPyTorch')
    train_models(df, 'NeuralNetworkTensorflow')
    

    # Faire la moyenne de chaque itération

    # Trouver la meilleur combinaison pour chaque modele

    with open('result/week_without_scale_52weeks.json', 'r') as json_file:
        data = json.load(json_file)


    best_combinations = {}

    folder_path = 'result/by_model'
    file_list = os.listdir(folder_path)

    for file_name in file_list:

        model_name = file_name.split('_')[0] 
        model_data = data.get(model_name, {}) 
        best_combination = find_best_combination(model_data)
        best_combinations[model_name] = {
            'best_combination': best_combination[:2],  
            'best_mape': best_combination[2]['MAPE']  
        }

    with open("result/best_week_with_scale_52weeks.json", 'w') as json_file:
        json.dump(best_combinations, json_file, indent=4)


    print(json.dumps(data, indent=4))

    best_combinations = {}
    for model, model_data in data.items():
        best_combinations[model] = find_best_combination(model_data)


    best_combinations_json = []
    for model, (scaler_name, best_method, metrics) in best_combinations.items():
        best_combinations_json.append({
            "Model": model,
            "Scaler": scaler_name,
            "Best Feature Selection": best_method,
            "MAPE": metrics["MAPE"],
            "MAE": metrics["MAE"],
            "Prediction": metrics["Prediction"]
        })


    with open("result/best_week_with_scale_52weeks.json", 'w') as json_file:
        json.dump(best_combinations_json, json_file, indent=4)


    # Mettre les résultats dans un Tableau
    with open('result/best_week_with_scale_52weeks.json', 'r') as json_file:
        data = json.load(json_file)

   
    df = pd.DataFrame(data)
    df['MAPE'] = df['MAPE'].apply(lambda x: x['week_1'])
    df['MAE'] = df['MAE'].apply(lambda x: x['week_1'])

    df = df.sort_values(by='MAPE')
    df = df[['Model', 'Scaler', 'MAPE', 'MAE', 'Best Feature Selection', 'Prediction']]
    df.to_excel('result/best_models_sorted_with_scale.xlsx', index=False)


    # Tracer les courbes de MAPE et précision

    with open('result/best_week_with_scale_52weeks.json', 'r') as json_file:
        data = json.load(json_file)

    # Initialiser les figures et axes pour les trois graphiques
    fig_mape, ax_mape = plt.subplots(figsize=(10, 6))
    fig_mae, ax_mae = plt.subplots(figsize=(10, 6))
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))

    # Parcourir chaque modèle pour tracer les courbes
    for model_data in data:
        model = model_data["Model"]
        weeks = list(model_data["MAPE"].keys())
        weeks_int = [int(week.split('_')[1]) for week in weeks]  # Convertir les semaines en entiers

        # Récupérer les valeurs de MAPE, MAE et Prediction
        mape_values = list(model_data["MAPE"].values())
        mae_values = list(model_data["MAE"].values())
        pred_values = list(model_data["Prediction"].values())

        # Tracer les courbes
        ax_mape.plot(weeks_int, mape_values, label=model)
        ax_mae.plot(weeks_int, mae_values, label=model)
        ax_pred.plot(weeks_int, pred_values, label=model)

    # Ajouter des titres et des légendes
    ax_mape.set_title('MAPE par Modèle au fil du temps')
    ax_mape.set_xlabel('Semaine')
    ax_mape.set_ylabel('MAPE')
    ax_mape.legend()

    ax_mae.set_title('MAE par Modèle au fil du temps')
    ax_mae.set_xlabel('Semaine')
    ax_mae.set_ylabel('MAE')
    ax_mae.legend()

    ax_pred.set_title('Prédictions par Modèle au fil du temps')
    ax_pred.set_xlabel('Semaine')
    ax_pred.set_ylabel('Prédiction')
    ax_pred.legend()

    # Afficher les graphiques
    #plt.show()


if True: # Test autosklearn

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

