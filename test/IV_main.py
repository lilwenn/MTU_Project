import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statsmodels.tsa.arima.model import ARIMA

from sklearn.model_selection import train_test_split
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
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from II_Data_visualization import plot_correlation_matrix  
from III_Preprocessing import time_series_analysis, determine_lags 
import Constants as const

from darts import TimeSeries
from darts.metrics import mape as darts_mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import NBEATSModel, NaiveSeasonal, ExponentialSmoothing
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split as darts_train_test_split

from prophet import Prophet

import joblib
from tpot import TPOTRegressor
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator, H2ORandomForestEstimator, H2OGradientBoostingEstimator


def regression_scores(actual: np.ndarray, predicted: np.ndarray):

    results = {
        'predictions' : predicted ,
        'MAE': mean_absolute_error(actual, predicted),
        'MAE2': median_absolute_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted),
        'ME': np.mean(actual - predicted),
        'MSE': mean_squared_error(actual, predicted),
        'R2': r2_score(actual, predicted),
        'RMSE': math.sqrt(mean_squared_error(actual, predicted)),
    }

    return results

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

    # Get the selected features from SelectKBest
    selected_features = model_pipeline.named_steps['selectkbest'].get_support(indices=True)
    selected_feature_names = [features[i] for i in selected_features]

    # Predict future values
    last_row = df.iloc[-1][features]
    future = pd.DataFrame([last_row])
    prediction = model_pipeline.predict(future)

    return prediction[0], selected_feature_names

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
    full_df = full_df.drop(columns=['year_week','Week','EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price'])

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

def preprocess_exog_data(df, forecast_weeks):
    # Create a new dataframe exog_data by dropping columns that contain the string 'litres'
    exog_data = df.drop(columns=[col for col in df.columns if 'litres' in col])

    # Split exog_data into exog_train and exog_future
    exog_train = exog_data.iloc[:-forecast_weeks]
    exog_future = exog_data.iloc[-forecast_weeks:]

    # Ensure the number of columns in exog_future matches exog_train
    exog_future = exog_future.reindex(columns=exog_train.columns, fill_value=0)

    return exog_train, exog_future

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

    """
    #essayer un pmarima
        
    # Ajuster un modèle auto_arima simple
    arima = pm.auto_arima(train, error_action='ignore', trace=True,
                        suppress_warnings=True, seasonal=True, m=12)

    """

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

def train_models(df, model_name,model):
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

    
    if (model_name == 'ARIMA'):
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
        predictions, last_date = train_and_predict_arima(df, exog_future, best_order)

        # Get actual values for comparison
        actual_values = df_copy['litres'].iloc[-const.forecast_weeks:].values



        # Store predictions, last date, MAE, and MAPE in results
        result[model_name]['Predictions'] = predictions.tolist()
        result[model_name]['MAE'] = mae_score
        result[model_name]['MAPE'] = mape_score

        with open(f'result/by_model/{model_name}_{const.forecast_weeks}week.json', 'w') as json_file:
            json.dump(result[model_name], json_file, indent=4)
    
    else : 
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
                    prediction, selected_features = train_and_predict(df, features, target_col, pipeline)
                    print(f'MAPE for {model_name} with {target_col}, scaler {scaler_name}, scoring {scoring_name}: {mape_score:.2f}')

                    result[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
                    result[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
                    result[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = prediction

                    selected_features_set.update(selected_features)

                with open(f'result/by_model/{model_name}_{week}week.json', 'w') as json_file:
                    json.dump(result[model_name], json_file, indent=4)

def find_best_model_configs(result_dir, file_name, best_combinations, results_list):
    """
    Analyzes JSON result files to find the best model configurations for each model based on MAPE."""

    file_path = os.path.join(result_dir, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)

        best_mape = float('inf')
        best_combination = None
        best_predictions = None

        for scaler_name, scaler_data in data.items():
            for scoring_name, metrics in scaler_data.items():
                # Find the key for the last week
                last_week = max(metrics['MAPE'].keys(), key=lambda x: int(x.split('_')[1]))
                mape = metrics['MAPE'][last_week]
                if mape < best_mape:
                    best_mape = mape
                    mape_list = metrics['MAPE']
                    best_combination = (scaler_name, scoring_name, mape)
                    best_predictions = metrics['Prediction']

        if best_combination:
            scaler_name, scoring_name, mape = best_combination
            model_name = file_name.replace(".json", "") 
            best_combinations[model_name] = {
                'scaler': scaler_name,
                'scoring': scoring_name,
                'MAPE': mape_list,
                'predictions': best_predictions
            }
            results_list.append({
                'model': model_name,
                'scaler': scaler_name,
                'scoring': scoring_name,
                'MAPE': mape,
                'predictions': best_predictions
            })

    # Save the best combinations to a JSON file
    if best_combinations:
        with open(f'result/best_combinations_{const.forecast_weeks}week.json', 'w') as f:
            json.dump(best_combinations, f, indent=4)

    # Convert the results list to a DataFrame and save it to an Excel file
    if results_list:
        results_df = pd.DataFrame(results_list)
        # Sort the DataFrame by MAPE in ascending order
        results_df = results_df.sort_values(by='MAPE')
        results_df.to_excel(f'result/best_combinations_{const.forecast_weeks}week.xlsx', index=False)
        print(f" {file_name} results saved to xlsx")

def plot_model_performance(best_combinations_file, const):
    """
    Plots MAPE, Predictions, and MAE over time for each model using the best combinations JSON file.

    Args:
    - best_combinations_file (str): Path to the JSON file containing the best model configurations.
    - const (object): Object containing constants such as forecast_weeks.
    """
    # Load the best combinations JSON file
    with open(best_combinations_file, 'r') as f:
        best_combinations = json.load(f)

    df = pd.read_excel("spreadsheet/Final_Weekly_2009_2021.xlsx") 

    weeks = list(range(1, const.forecast_weeks + 1))

    # Initialize subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 18))

    # Directly access 'tab10' colormap colors
    colors = plt.cm.tab10.colors

    for i, (model_name, model_info) in enumerate(best_combinations.items()):
        mape = model_info['MAPE']
        predictions = model_info['predictions']

        # Convert MAPE from list of dictionaries to list of numeric values
        mape_values = [mape[f'week_{week}'] for week in weeks]

        # Plot MAPE over time
        axs[0].plot(weeks, mape_values, marker='o', linestyle='-', color=colors[i], label=f'{model_name} - MAPE')

        # Set logarithmic scale for MAPE plot
        axs[0].set_yscale('log')

        # Convert predictions from list of dictionaries to list of numeric values
        prediction_values = [predictions[f'week_{week}'] for week in weeks]

        # Plot Predictions over time
        axs[1].plot(weeks, prediction_values, marker='o', linestyle='-', color=colors[i], label=f'{model_name} - Predictions')

        # Load the actual values from the original data
        actual_col = df.iloc[-52:][const.target_column]


    #axs[1].plot(weeks, actual_col, marker='x', linestyle='-', color='black', label='Actual')


    # Set common labels and title for all subplots
    for ax in axs:
        ax.set_xlabel('Week')
        ax.grid(True)
        ax.legend()

    axs[0].set_ylabel('MAPE (log scale)')
    axs[0].set_title('MAPE over Time')

    axs[1].set_ylabel('Values')
    axs[1].set_title('Predictions vs Actuals over Time')

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'visualization/all_models_performance.png')

def train_darts_model(df, forecast_weeks):
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
    series = TimeSeries.from_dataframe(df[[const.target_column]], fill_missing_dates=True)

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

def plot_darts_model_performance(results, df):
    """
    Plots predictions and actual values for Darts models.

    Args:
    - results (dict): Dictionary containing the results from Darts models.
    - df (DataFrame): Original DataFrame containing actual values.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, (model_name, model_info) in enumerate(results.items()):
        predictions = model_info['Prediction']
        plt.plot(df.index[-len(predictions):], predictions, marker='o', linestyle='-', color=colors[i], label=f'{model_name} Predictions')

    # Plot actual values
    plt.plot(df.index[-len(predictions):], df[const.target_column][-len(predictions):], marker='x', linestyle='-', color='black', label='Actual')

    plt.xlabel('Date')
    plt.ylabel(const.target_column)
    plt.title('Darts Model Predictions')
    plt.legend()
    plt.savefig(f'visualization/Darts_Model_Predictions.png')
    plt.show()

def plot_prophet_model_performance(results, df):
    """
    Plots predictions and actual values for Prophet models.

    Args:
    - results (dict): Dictionary containing the results from Prophet models.
    - df (DataFrame): Original DataFrame containing actual values.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))

    # Convert results to DataFrame for plotting
    predictions_df = pd.DataFrame({
        'ds': df.index[-len(results["Predictions"]):],
        'Prediction': results["Predictions"]
    })

    # Plot actual values
    plt.plot(df.index[-len(results["Predictions"]):], df[const.target_column][-len(results["Predictions"]):], marker='x', linestyle='-', color='black', label='Actual')

    # Plot Prophet predictions
    plt.plot(predictions_df['ds'], predictions_df['Prediction'], marker='o', linestyle='-', color='blue', label='Prophet Predictions')

    plt.xlabel('Date')
    plt.ylabel(const.target_column)
    plt.title('Prophet Model Predictions')
    plt.legend()
    plt.show()

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
    df_prophet = df.reset_index().rename(columns={'Date': 'ds', const.target_column: 'y'})

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

def train_TPOT_model(df, target_column):

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    # Préparation des données
    X = df.drop(columns=target_column)
    y = df[target_column]

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
        r2 = r2_score(y_test, y_pred)
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


def train_H2O_model(df, const.TARGET_column):
    
    # Initialiser H2O
    h2o.init()

    # Charger les données
    df = pd.read_excel('spreadsheet/lagged_results.xlsx')
    data = h2o.H2OFrame(df)

    # Définir la variable cible et les variables explicatives
    target = "litres"
    features = data.columns
    features.remove(target)

    # Séparer les données en ensembles d'entraînement et de test
    train, test = data.split_frame(ratios=[0.8], seed=1234)

    # Modèle de Régression Linéaire avec régularisation
    model_lr = H2OGeneralizedLinearEstimator(alpha=0.5, lambda_=0.1)
    model_lr.train(x=features, y=target, training_frame=train)

    # Faire des prédictions sur l'ensemble de test
    predictions_lr = model_lr.predict(test)
    actuals_lr = test[target].as_data_frame().values.flatten()
    predicted_lr = predictions_lr.as_data_frame().values.flatten()
    mape_lr = mape(actuals_lr, predicted_lr)
    print(f"MAPE for Linear Regression Model: {mape_lr:.2f}%")

    # Évaluer les performances du modèle de régression linéaire
    performance_lr = model_lr.model_performance(test)
    print(performance_lr)

    # Modèle de Random Forest avec hyperparamètres optimisés
    model_rf = H2ORandomForestEstimator(
        ntrees=100,
        max_depth=20,
        min_rows=10
    )
    model_rf.train(x=features, y=target, training_frame=train)

    # Faire des prédictions sur l'ensemble de test
    predictions_rf = model_rf.predict(test)
    actuals_rf = test[target].as_data_frame().values.flatten()
    predicted_rf = predictions_rf.as_data_frame().values.flatten()
    mape_rf = mape(actuals_rf, predicted_rf)
    print(f"MAPE for Random Forest Model: {mape_rf:.2f}%")

    # Évaluer les performances du modèle de Random Forest
    performance_rf = model_rf.model_performance(test)
    print(performance_rf)

    # Modèle de Gradient Boosting
    model_gbm = H2OGradientBoostingEstimator()
    model_gbm.train(x=features, y=target, training_frame=train)

    # Faire des prédictions sur l'ensemble de test
    predictions_gbm = model_gbm.predict(test)
    actuals_gbm = test[target].as_data_frame().values.flatten()
    predicted_gbm = predictions_gbm.as_data_frame().values.flatten()
    mape_gbm = mape(actuals_gbm, predicted_gbm)
    print(f"MAPE for Gradient Boosting Model: {mape_gbm:.2f}%")

    # Évaluer les performances du modèle de Gradient Boosting
    performance_gbm = model_gbm.model_performance(test)
    print(performance_gbm)


if __name__ == "__main__":

    #df = load_and_preprocess_data()
    df = pd.read_excel('spreadsheet/lagged_results.xlsx')

    for model_name, model in const.models.items():
        train_models(df, model_name,model)
    
    find_best_model_configs()

    best_combinations = {}
    result_dir = 'result/by_model/'
    results_list = []
    best_combinations = {}

    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} does not exist.")
    else:
        json_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{const.forecast_weeks}week.json')]
        for file_name in json_files:
            find_best_model_configs(result_dir, file_name, best_combinations, results_list)

    df = pd.read_excel('spreadsheet/lagged_results.xlsx')

    train_models(df, 'ARIMA', None)

    # Train Darts models and get the results
    results = train_darts_model(df, const.forecast_weeks)

    # Plot the performance of Darts models
    plot_darts_model_performance(results, df)

    
    # Train Prophet model and get the results
    results = train_prophet_model(df, const.forecast_weeks)

    # Plot the performance of Prophet model
    plot_prophet_model_performance(results, df)

    train_TPOT_model(df, const.target_column)

    train_H2O_model(df, const.TARGET_COLUMN)



    
"""

    # Load the data from the JSON file with utf-8 encoding
    with open(f'result/best_combinations_{const.forecast_weeks}week.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')
    
    plt.figure(figsize=(12, 6))
    
    # Plot the predictions
    plt.plot(df['Date'], df['litres'][:52], marker='o', linestyle='-', color='black', label= 'Actual')

    plt.xlabel('Date')
    plt.ylabel('Litres')
    plt.title('ARIMA Model Predictions')
    plt.legend()
    plt.show()

    # Plot Predictions over time



    plot_model_performance(f'result/best_combinations_{const.forecast_weeks}week.json', const)
"""