import os
import json
import joblib
import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import NBEATSModel, NaiveSeasonal, ExponentialSmoothing
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split as darts_train_test_split

from prophet import Prophet

from II_Data_visualization import plot_correlation_matrix  
from III_Preprocessing import time_series_analysis, determine_lags, load_and_preprocess_data, preprocess_arima_data
import Constants as const

from tpot import TPOTRegressor
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima import model_selection

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_darts_modelquifonctionne(df, forecast_weeks):
    """
    Train Darts models on the time series data and evaluate their performance.

    Args:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    - forecast_weeks (int): Number of weeks to forecast.

    Returns:
    - None
    """
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove duplicate dates
    df = df.drop_duplicates(subset='Date')

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Set the Date column as the index
    df.set_index('Date', inplace=True)

    # Reindex the DataFrame to fill in missing dates with a specified frequency (e.g., 'W' for weekly)
    df = df.asfreq('W', method='ffill')

    # Convert the DataFrame to a TimeSeries object, only using the target column
    series = TimeSeries.from_dataframe(df[[const.TARGET_COLUMN]], fill_missing_dates=True)

    # Split the data into training and validation sets
    train, val = darts_train_test_split(series, test_size=forecast_weeks)

    # Optionally scale the time series
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)

    # Define models to train
    models = {
        "NBEATSModel": NBEATSModel(input_chunk_length=24, output_chunk_length=forecast_weeks),
        "NaiveSeasonal": NaiveSeasonal(K=52),  # Example of a simple seasonal naive model
        "ExponentialSmoothing": ExponentialSmoothing()
    }

    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        model.fit(train_transformed)
        prediction = model.predict(len(val))
        mape_score = darts_mape(val_transformed, prediction)

        # Inverse transform the prediction to get actual scale
        prediction = transformer.inverse_transform(prediction)

        results[model_name] = {
            "MAPE": mape_score,
            "Prediction": prediction.values().flatten().tolist()
        }

        print(f"{model_name} - MAPE: {mape_score:.2f}")

    # Save the results to a JSON file
    os.makedirs('result', exist_ok=True)
    with open(f'result/darts_models_{forecast_weeks}weeks.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results


def train_darts_model(df, forecast_weeks):
    # Convert the Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    # Remove duplicate dates and sort the DataFrame by date
    df = df.drop_duplicates(subset='Date').sort_values(by='Date')
    # Set the Date column as the index
    df.set_index('Date', inplace=True)
    # Fill missing dates with the specified frequency
    df = df.asfreq('W', method='ffill')
    # Convert the DataFrame to a TimeSeries object
    series = TimeSeries.from_dataframe(df[[const.TARGET_COLUMN]], fill_missing_dates=True)

    # Split the data into training and validation sets
    train, val = darts_train_test_split(series, test_size=forecast_weeks)
    # Optionally scale the time series
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)

    # Define models to train
    models = {
        "NaiveSeasonal": NaiveSeasonal(K=52),
    }
    
    for model_name, model in models.items():
        # Fit the model on the transformed training data
        model.fit(train_transformed)
        # Predict the future values
        prediction = model.predict(len(val_transformed))
        # Inverse transform the prediction to get actual scale
        prediction = transformer.inverse_transform(prediction)

    # Return the predictions as a flattened list
    return prediction.values().flatten().tolist()


def train_prophet_model(df, forecast_weeks):
    """
    Train Prophet models on the time series data and evaluate their performance.

    Args:
    - df (DataFrame): Preprocessed DataFrame ready for model training.
    - forecast_weeks (int): Number of weeks to forecast.

    Returns:
    - dict: Results with MAPE and predictions from Prophet.
    """
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove duplicate dates
    df = df.drop_duplicates(subset='Date')

    # Sort the DataFrame by date
    df = df.sort_values(by='Date')

    # Set the Date column as the index
    df = df.set_index('Date')

    # Reindex the DataFrame to fill in missing dates with a specified frequency (e.g., 'W' for weekly)
    df = df.asfreq('W', method='ffill')

    # Prepare data for Prophet
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', const.TARGET_COLUMN: 'y'})

    # Split the data into training and validation sets
    train = df_prophet[:-forecast_weeks]
    val = df_prophet[-forecast_weeks:]

    # Define and train the Prophet model
    model = Prophet()
    model.fit(train)

    # Create a DataFrame for future dates
    future = model.make_future_dataframe(periods=forecast_weeks, freq='W')

    # Predict future values
    forecast = model.predict(future)

    # Extract predictions and ensure alignment with validation data
    predictions = forecast[['ds', 'yhat']].tail(forecast_weeks).set_index('ds').rename(columns={'yhat': 'Prediction'})

    # Ensure that predictions and actual values align
    actual = val.set_index('ds')['y']

    # Align predictions with actual values
    aligned_predictions = predictions.reindex(actual.index)

    # Calculate MAPE
    mape = np.mean(np.abs((actual - aligned_predictions['Prediction']) / actual)) * 100

    # Prepare results for JSON (no dates, only predictions and MAPE)
    predictions_list = aligned_predictions['Prediction'].tolist()

    results = {
        "MAPE": mape,
        "Predictions": predictions_list
    }

    print(f"Prophet - MAPE: {mape:.2f}")

    # Save the results to a JSON file
    os.makedirs('result', exist_ok=True)
    with open(f'result/prophet_model_{forecast_weeks}weeks.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

def train_TPOT_model(df, TARGET_COLUMN):

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    # Préparation des données
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    # Utiliser TimeSeriesSplit pour diviser les données
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_score = float('inf')
    best_model_performance = {}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Utiliser TPOT pour entraîner un modèle
        tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2, random_state=42, cv=5, scoring='neg_mean_squared_error')
        tpot.fit(X_train, y_train)

        # Prédictions
        y_pred = tpot.predict(X_test)

        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        score_mape = mape(y_test, y_pred)

        print(f'Erreur quadratique moyenne (MSE): {mse}')
        print(f'Coefficient de détermination (R^2): {r2}')
        print(f'Erreur absolue moyenne (MAE): {mae}')
        print(f'Erreur absolue moyenne en pourcentage (MAPE): {score_mape}')

        # Mémoriser le meilleur modèle basé sur MSE
        if mse < best_score:
            best_score = mse
            best_model = tpot.fitted_pipeline_

            # Enregistrer les performances du meilleur modèle
            best_model_performance = {
                'MSE': mse,
                'R2': r2,
                'MAE': mae,
                'MAPE': score_mape
            }

            # Enregistrer le meilleur modèle
            joblib.dump(best_model, 'best_tpot_model.pkl')

    # Charger le meilleur modèle
    best_model = joblib.load('best_tpot_model.pkl')

    # Entraîner le meilleur modèle sur toutes les données
    best_model.fit(X, y)

    # Faire des prédictions pour les 52 semaines
    if len(X) >= 52:
        y_pred_52_weeks = best_model.predict(X.head(52))
        print('Prédictions pour les 52 semaines:', y_pred_52_weeks)
    else:
        print("Pas assez de données pour les prédictions sur 52 semaines")

    # Faire des prédictions sur l'ensemble des données
    y_pred = best_model.predict(X)

    # Calculer MAE et MAPE pour les prédictions sur l'ensemble des données
    mae = mean_absolute_error(y, y_pred)
    score_mape = mape(y, y_pred)

    print(f'Erreur absolue moyenne (MAE): {mae}')
    print(f'Erreur absolue moyenne en pourcentage (MAPE): {score_mape}')

    # Afficher les performances du meilleur modèle
    print("Performances du meilleur modèle :")
    print(best_model_performance)

def train_and_predict_arima(df, exog_future, order=(1, 1, 1)):
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

    # Predict the next const.FORECAST_WEEKS weeks
    predictions = model_fit.forecast(steps=const.FORECAST_WEEKS, exog=exog_future)

    # Get the last date in the training data
    last_date = df_copy.index.max()

    # Reset the index
    df_copy = df_copy.reset_index()

    return predictions, last_date

def train_pmdarima_model(df, target_column, forecast_weeks):
    # Préparation des données
    data = df[target_column]

    # Diviser les données en ensemble d'entraînement et de test
    train_size = len(data) - forecast_weeks
    train, test = data[:train_size], data[train_size:]

    # Ajuster le modèle auto_arima avec différents paramètres
    model = auto_arima(train, seasonal=True, m=12, stepwise=True, trace=True)
    
    # Faire des prévisions
    forecast = model.predict(n_periods=len(test))
    return forecast, test

def calc_scores(actual: np.ndarray, predicted: np.ndarray, execution_time):
    pred_dict = {}
    mae_dict = {}
    mae2_dict = {}
    mape_dict = {}
    me_dict = {}
    mse_dict = {}
    rmse_dict = {}

    for week in range(1, const.FORECAST_WEEKS + 1):
        actual_week = actual[week - 1:week]
        predicted_week = predicted[week - 1:week]
        pred_dict[f'week_{week}'] = predicted_week[0]

        # Calculate metrics with safety checks

        mae_dict[f'week_{week}'] = mean_absolute_error(actual_week, predicted_week)
        mae2_dict[f'week_{week}'] = median_absolute_error(actual_week, predicted_week)
        mse = mean_squared_error(actual_week, predicted_week)
        mse_dict[f'week_{week}'] = mean_squared_error(actual_week, predicted_week)
        rmse_dict[f'week_{week}'] = math.sqrt(mse)
        
        mape_dict[f'week_{week}'] = mean_absolute_percentage_error(actual_week, predicted_week)
        print(mape_dict[f'week_{week}'])
        me_dict[f'week_{week}'] = np.mean(actual_week - predicted_week)

    results = {
        'Predictions': pred_dict,
        'Execution Time': execution_time,
        'MAE': mae_dict,
        'MAE2': mae2_dict,
        'MAPE': mape_dict,
        'ME': me_dict,
        'MSE': mse_dict,
        'RMSE': rmse_dict,
    }

    return results

def train_and_predict(df, features, target_col, model_pipeline):
    X = df[features]
    y = df[target_col]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model pipeline
    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time

    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)

    # Get the selected features from SelectKBest
    selected_features = model_pipeline.named_steps['selectkbest'].get_support(indices=True)
    selected_feature_names = [features[i] for i in selected_features]

    # Predict future values
    last_row = df.iloc[-1][features]
    future = pd.DataFrame([last_row])
    prediction = model_pipeline.predict(future)

    return y_test, prediction[0]

def train_models(df, model_name, model, w):
    features = [col for col in df.columns if not col.startswith(f'{const.TARGET_COLUMN}_next_') and col != 'Date']
    result = {}

    if model_name == 'ARIMA':
        exog_train, exog_future = preprocess_arima_data(df, const.FORECAST_WEEKS)
        df_copy = df.set_index('Date') if 'Date' in df.columns else df.copy()
        best_order = (1, 1, 1)
        start_time = time.time()
        predictions, last_date = train_and_predict_arima(df, exog_future, best_order)
        end_time = time.time()
        execution_time = end_time - start_time
        actual_values = df_copy['litres'].iloc[-const.FORECAST_WEEKS:].values
        result = calc_scores(actual_values, predictions, execution_time)

    elif model_name == 'Pmdarima':
        start_time = time.time()
        predictions, test = train_pmdarima_model(df, const.TARGET_COLUMN, const.FORECAST_WEEKS)
        predictions = predictions.tolist()
        end_time = time.time()
        execution_time = end_time - start_time
        actual_values = df[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values
        result = calc_scores(actual_values, predictions, execution_time)

    elif model_name == 'Darts':
        start_time = time.time()
        predictions = train_darts_model(df, const.FORECAST_WEEKS)
        end_time = time.time()
        execution_time = end_time - start_time
        actual_values = df[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values
        result = calc_scores(actual_values, predictions, execution_time)

    else:
        for scaler_name, scaler in const.SCALERS.items():
            result[scaler_name] = {}

            for scoring_name, scoring_func in const.SCORING_METHODS.items():
                result[scaler_name][scoring_name] = {}
                default_scoring_func = f_regression
                default_k = 5

                pipeline = Pipeline([
                    ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func, k=const.K_VALUES.get(scoring_name, default_k))),
                    ('scaler', scaler),
                    ('model', model)
                ])

                selected_features_set = set()
                total_execution_time = 0   
                pred_list = []  

                for week in range(1, const.FORECAST_WEEKS + 1):
                    target_col = f'{const.TARGET_COLUMN}_next_{week}weeks'
                    start_time = time.time()
                    y_test ,prediction = train_and_predict(df, features, target_col, pipeline)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    total_execution_time += execution_time
                    pred_list.append(prediction)

                           
                result[scaler_name][scoring_name] = calc_scores( y_test, pred_list, total_execution_time)


    print(f"All combinations of {model_name} for window ={w} saved in json")
    with open(f'result/by_model/{model_name}_{w}_model_{const.FORECAST_WEEKS}.json', 'w') as f:
        json.dump(result, f, indent=4)

def find_best_model_configs(model_name, window_list, Non_ml):
    """
    Analyzes JSON result files to find the best model configurations for each model based on the best MAPE across all iterations.
    """
    best_mape = float('inf')
    best_combination = None
    best_iteration = None



    if model_name not in Non_ml :
        for w in window_list: 
            file_path = os.path.join(f'result/by_model/{model_name}_{w}_model_{const.FORECAST_WEEKS}.json')

            if not os.path.exists(file_path):
                print(f"Result file for iteration {w} does not exist: {file_path}")
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)
            
        for scaler_name, scaler_data in data.items():
            for scoring_name, metrics in scaler_data.items():
                weeks = list(metrics['MAPE'].keys())
                last_week = max(weeks, key=lambda x: int(x.split('_')[1]))
                mape = metrics['MAPE'][last_week]
                total_execution_time = sum(metrics['Execution Time'][week] for week in weeks)
                predictions = metrics.get('Prediction', {})

                if mape < best_mape:
                    best_mape = mape
                    best_iteration = w
                    best_combination = {
                        'MAPE': metrics['MAPE'],
                        'MAE': metrics['MAE'],
                        'MAE2': metrics['MAE2'],
                        'ME': metrics['ME'],
                        'MSE': metrics['MAE2'], 
                        'RMSE': metrics['RMSE'],
                        'Execution Time': total_execution_time,
                        'Scaler': None,
                        'Scoring': None,
                        'Predictions': predictions,
                        'Iteration': w
                    }


                        
    else:
        w = 0
        file_path = os.path.join(f'result/by_model/{model_name}_{w}_model_{const.FORECAST_WEEKS}.json')

        if not os.path.exists(file_path):
            print(f"Result file for iteration {w} does not exist: {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)    

        weeks = list(data['MAPE'].keys())
        last_week = max(weeks, key=lambda x: int(x.split('_')[1]))
        mape = data['MAPE'][last_week]
        if mape < best_mape:
            best_mape = mape
            best_iteration = w
            best_combination = {
                'MAPE': data ['MAPE'],
                'MAE': data ['MAE'],
                'MAE2':data ['MAE2'],
                'ME':data ['ME'],
                'MSE': data ['MAE2'],
                'RMSE': data ['RMSE'],
                'Execution Time': data['Execution Time'],
                'Scaler': None,
                'Scoring': None,
                'Predictions': data['Predictions'],
                'Iteration': w
            }
            

    if best_combination:
        output_file_path = f'result/by_model/{model_name}_best_model.json'
        with open(output_file_path, 'w') as f:
            json.dump(best_combination, f, indent=4)

        print(f"Best combination for {model_name} across all iterations saved (Iteration {best_iteration})")

    print(best_combination['MAPE'])
    return best_combination

def update_global_results(model_name):
    """
    Updates a global JSON file with the best results of a specific model.
    """
    # Define the path for the best model results file
    best_model_result_file = f'result/by_model/{model_name}_best_model.json'
    global_results_file = 'result/global_results.json'

    if not os.path.exists(best_model_result_file):
        print(f"Best model result file {best_model_result_file} does not exist.")
        return

    with open(best_model_result_file, 'r') as f:
        best_model_data = json.load(f)

    if os.path.exists(global_results_file):
        with open(global_results_file, 'r') as f:
            global_data = json.load(f)
    else:
        global_data = {}

    global_data[model_name] = best_model_data

    with open(global_results_file, 'w') as f:
        json.dump(global_data, f, indent=4)

    print(f"Global JSON results updated")

def save_results_to_excel(output_excel_file, output_json_file, week_predict):
    """
    Sauvegarde les meilleurs résultats du fichier JSON global dans un fichier Excel,
    triés par le MAPE le plus bas, et les enregistre également dans un fichier JSON.
    """
    suffix = f'_best_model .json'
    results = []  # Initialiser la liste pour stocker les résultats pour Excel
    JSON_results = []  # Initialiser la liste pour stocker les résultats pour JSON

    for file_name in os.listdir('result/by_model'):
        if file_name.endswith(suffix):
            model_name = file_name.replace(suffix, '') 

            with open(f'result/by_model/{file_name}', 'r') as f:
                model_data = json.load(f)

            last_mape = model_data['MAPE'][f"week_{week_predict}"]

            results.append({
                'Model Name': model_name,
                'MAPE': model_data['MAPE'],
                'MAE': model_data['MAE'],
                'Execution Time': model_data['Execution Time'],
                'Scaler': model_data['Scaler'],
                'Scoring': model_data['Scoring']
            })

            JSON_results.append({
                'Model Name': model_name,
                'MAPE': last_mape,
                'MAE': model_data['MAE'],
                'MAE2':model_data['MAE2'],
                'ME':model_data['ME'],
                'MSE': model_data['MAE2'],
                'RMSE': model_data['RMSE'],
                'Predictions': model_data['Predictions'],
                'Execution Time': model_data['Execution Time'],
                'Scaler': model_data['Scaler'],
                'Scoring': model_data['Scoring']
            })

    # Sauvegarder les résultats dans un fichier Excel
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='MAPE', ascending=True)
        df_results.to_excel(output_excel_file, index=False)
        print(f"Results saved to {output_excel_file}.")
    else:
        print("No results to save.")

    # Sauvegarder les résultats dans un fichier JSON
    if JSON_results:
        with open(output_json_file, 'w') as f:
            json.dump(JSON_results, f, indent=4)
        print(f"JSON results saved to {output_json_file}.")
    else:
        print("No JSON results to save.")


if __name__ == "__main__":
    print('----------------------- START----------------------')

    Non_ml = ['Pmdarima', 'Darts' , 'ARIMA']

    window_list = [9, 10, 11, 12, 13, 14, 15, 16,17]

    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')
    for model_name, model in const.MODELS.items():

        print(f' MODEL : {model_name}')
        if model_name not in Non_ml:
            for w in window_list:
                lagged_df = load_and_preprocess_data(w, df)
                train_models(lagged_df, model_name, model,w)
        else:
            train_models(df, model_name, model,0)


    # Find the best configurations for each model
    for model_name in const.MODELS.keys():
        find_best_model_configs(model_name, window_list, Non_ml)
        update_global_results(model_name)

    week_predict = 1
    output_excel_file = f'result/{week_predict}w_global_model_results.xlsx' 
    output_json_file = f'result/{week_predict}w_global_model_results.json' 
    #save_results_to_excel(output_excel_file, output_json_file, week_predict)





