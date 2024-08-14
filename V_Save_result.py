import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

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

import Constants as const


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_weekly_mape(model_name, targ_dir, y_true: np.ndarray, y_pred: np.ndarray, forecast_weeks: int) -> dict:
    weekly_errors = {}
    for week in range(1, forecast_weeks + 1):
        actual_value = y_true[week - 1]
        predicted_value = y_pred[week - 1]
        
        percentage_error = abs((actual_value - predicted_value) / actual_value) * 100
        weekly_errors[f'week_{week}'] = percentage_error


    # Print weekly MAPE
    for week, mape_value in weekly_errors.items():
        print(f'{model_name}  -  {week}: MAPE = {mape_value:.2f}%')

    # Optionally, save weekly MAPE to a file
    with open(f'result/{targ_dir}/weekly_mape/{model_name}weekly_mape.json', 'w') as f:
        json.dump(weekly_errors, f, indent=4)


    return weekly_errors


def calculate_weekly_percentage_errors(model_name, targ_dir, y_true: np.ndarray, y_pred: np.ndarray, forecast_weeks: int) -> dict:
    """
    Calculate the percentage error for each week individually.
    
    Args:
    - model_name (str): Name of the model being evaluated.
    - targ_dir (str): Target directory to save the results.
    - y_true (np.ndarray): True values for each week.
    - y_pred (np.ndarray): Predicted values for each week.
    - forecast_weeks (int): Number of weeks in the forecast period.
    
    Returns:
    - weekly_errors (dict): Dictionary of percentage errors for each week.
    """
    weekly_errors = {}
    
    for week in range(1, forecast_weeks + 1):
        actual_value = y_true[week - 1]
        predicted_value = y_pred[week - 1]
        
        if actual_value != 0:
            percentage_error = abs((actual_value - predicted_value) / actual_value) * 100
        else:
            percentage_error = np.nan  # Handle division by zero by setting to NaN
            
        weekly_errors[f'week_{week}'] = percentage_error

    # Print percentage errors for each week
    for week, error_value in weekly_errors.items():
        print(f'{model_name}  -  {week}: Percentage Error = {error_value:.2f}%')

    # Optionally, save the weekly errors to a file
    with open(f'result/{targ_dir}/weekly_errors/{model_name}_weekly_errors.json', 'w') as f:
        json.dump(weekly_errors, f, indent=4)

    return weekly_errors


def calc_scores(y_true, targ_dir) -> dict:

    for model_name, model in const.MODELS.items():
        with open(f'result/{targ_dir}/global_results.json', 'r') as f:
            data = json.load(f)
        
        with open(f'result/{targ_dir}/weekly_mape/{model_name}weekly_mape.json', 'r') as f:
            mape_dict = json.load(f)

        predicted = data[model_name]["Predictions"]
        mae_dict = {}
        mae2_dict = {}
        me_dict = {}
        mse_dict = {}
        rmse_dict = {}

        for week in range(1,  const.FORECAST_WEEKS + 1):
            actual_week = y_true[week - 1:week]
            predicted_week = predicted[week - 1:week]

            # Calculate metrics with safety checks
            mae_dict[f'week_{week}'] = mean_absolute_error(actual_week, predicted_week)
            mae2_dict[f'week_{week}'] = median_absolute_error(actual_week, predicted_week)
            mse = mean_squared_error(actual_week, predicted_week)
            mse_dict[f'week_{week}'] = mse
            rmse_dict[f'week_{week}'] = math.sqrt(mse)
            me_dict[f'week_{week}'] = np.mean(actual_week - predicted_week)

        data[model_name]['MAPE'] = mape_dict
        data[model_name]['MAE'] = mae_dict
        data[model_name]['MAE2'] =  mae2_dict
        data[model_name]['ME'] =  me_dict
        data[model_name]['MSE'] =  mse_dict
        data[model_name]['RMSE'] =  rmse_dict
 

    with open(f'result/{targ_dir}/global_results.json', 'w') as f:
        json.dump(data, f, indent=4)

    return data

def update_global_results(model_name, targ_dir):
    """
    Updates a global JSON file with the best results of a specific model.
    """
    # Define the path for the best model results file
    best_model_result_file = f'result/{targ_dir}/train_by_model/{model_name}_best_model.json'
    global_results_file = f'result/{targ_dir}/global_results.json'

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

def save_results_to_excel(output_excel_file, week, targ_dir):
    """
    Sauvegarde les meilleurs résultats du fichier JSON global dans un fichier Excel,
    triés par le MAPE le plus bas, et les enregistre également dans un fichier JSON.
    """

    results = []  # Initialiser la liste pour stocker les résultats pour Excel
    with open(f'result/{targ_dir}/global_results.json', 'r') as f:
        model_data = json.load(f)

    # Extraire les informations nécessaires
    for model_name in model_data.keys():
        with open(f'result/{targ_dir}/weekly_mape/{model_name}weekly_mape.json', 'r') as f:
            mape_data = json.load(f)

        results.append({
            'Model Name': model_name,
            'MAPE': mape_data[f'week_{week}'],
            'MAE' : model_data[model_name]['MAE Score'],
            'MAPE Score' : model_data[model_name]["MAPE_Score"],
            'Execution Time': model_data[model_name]['Execution Time'],
            'Scaler': model_data[model_name]['Scaler'],
            'Scoring': model_data[model_name]['Scoring'],
            'Lag': model_data[model_name]["Best lag"]
        })

    # Sauvegarder les résultats dans un fichier Excel et un fichier JSON si des résultats sont disponibles
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='MAPE', ascending=True)
        df_results.to_excel(output_excel_file, index=False)

        print(df_results)

        print(f"Results saved to {output_excel_file}.")

    else:
        print("No results to save.")

def plot_all_models_predictions(dates, y_true, targ_dir):

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("husl", len(const.MODELS))

    # Plotting actual values
    plt.plot(dates, y_true, label='Actual', color='black')
    i = 0
    for model_name in const.MODELS.keys():
        with open(f'result/{targ_dir}/train_by_model/{model_name}_best_model.json', 'r') as file:
            results = json.load(file)
        
        y_pred = results['Predictions']
        min_length = min(len(dates), len(y_true), len(y_pred))
        dates = dates[:min_length]
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        # Tracer les valeurs prédites pour chaque modèle avec une couleur différente
        plt.plot(dates, y_pred, label=model_name, color=palette[i])

        i+=1

    plt.title('Actual vs Predicted Values for All Models')
    plt.xlabel('Date')
    plt.ylabel(targ_dir)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'result/{targ_dir}/plot_predictions/predictions_all_models.png')
    plt.close()

def plot_predictions_by_model(y_pred, dates, y_true, model_name, targ_dir):
    # Convert dates to list if not already
    if not isinstance(dates, (list, np.ndarray)):
        dates = list(dates)

    min_length = min(len(dates), len(y_true), len(y_pred))
    dates = dates[:min_length]
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    plt.figure(figsize=(12, 6))

    # Plotting actual values
    plt.plot(dates, y_true, label='Actual', color='blue')

    # Plotting predicted values
    plt.plot(dates, y_pred, label='Predicted', color='red')

    plt.title(f'Actual vs Predicted Values - {model_name}')
    plt.xlabel('Date')
    plt.ylabel(targ_dir)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'result/{targ_dir}/plot_predictions/predictions_{model_name}.png')
    plt.close()

def plot_all_models_mape(dates, targ_dir):

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("husl", len(const.MODELS))

    i=0
    for model_name in const.MODELS.keys():
        with open(f'result/{targ_dir}/weekly_mape/{model_name}weekly_mape.json', 'r') as file:
            weekly_mape = json.load(file)

        mape_values = list(weekly_mape.values())

        # Ensure dates are the same length as mape_values
        min_length = min(len(dates), len(mape_values))
        dates = dates[:min_length]
        mape_values = mape_values[:min_length]

        # Plotting MAPE values for each model
        plt.plot(dates, mape_values, label=model_name, color=palette[i])
        i+=1

    plt.title('Weekly MAPE for All Models')
    plt.xlabel('Week')
    plt.ylabel('MAPE (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'result/{targ_dir}/plot_mape/mape_all_models.png')
    plt.close()

def plot_weekly_mape(dates, weekly_mape: dict,  model_name, targ_dir):
    mape_values = list(weekly_mape.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, mape_values, linestyle='-', color='b', label='Weekly MAPE', markersize=8) 
    plt.title('Weekly MAPE Over Time')
    plt.xlabel('Week')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'result/{targ_dir}/plot_mape/weekly_mape_{model_name}.png')
    plt.close()

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
