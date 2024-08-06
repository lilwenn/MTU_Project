import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Constants as const


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
                mape = metrics['mean_MAPE']
                predictions = metrics['Predictions']

                if mape < best_mape and 1 < mape :
                    best_mape = mape
                    best_iteration = w
                    best_combination = {
                        'MAPE': metrics['MAPE'],
                        'MAE': metrics['MAE'],
                        'MAE2': metrics['MAE2'],
                        'ME': metrics['ME'],
                        'MSE': metrics['MAE2'], 
                        'RMSE': metrics['RMSE'],
                        'Execution Time': metrics['Execution Time'],
                        'Scaler': scaler_name,
                        'Scoring': scoring_name,
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

def save_results_to_excel(output_excel_file, week):
    """
    Sauvegarde les meilleurs résultats du fichier JSON global dans un fichier Excel,
    triés par le MAPE le plus bas, et les enregistre également dans un fichier JSON.
    """

    results = []  # Initialiser la liste pour stocker les résultats pour Excel
    with open('result/global_results.json', 'r') as f:
        model_data = json.load(f)

    # Extraire les informations nécessaires
    for model_name in model_data.keys():
        results.append({
            'Model Name': model_name,
            'MAPE': model_data[model_name]['MAPE'][f'week_{week}'],
            'MAE': model_data[model_name]['MAE'][f'week_{week}'],
            'Execution Time': model_data[model_name]['Execution Time'],
            'Scaler': model_data[model_name]['Scaler'],
            'Scoring': model_data[model_name]['Scoring']
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

def plot_mape(json_file):
    """
    Crée un graphique pour les MAPE et l'enregistre en tant qu'image.
    
    Args:
        json_file (str): Chemin vers le fichier JSON contenant les données.
    """
    # Lire le fichier JSON
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Créer la figure pour les MAPE
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prétraiter les données pour le graphique MAPE
    for model_name, model_data in data.items():
        if model_name == "ARIMA":
            continue
        
        # Convertir les semaines en un format numérique pour l'axe x
        weeks = list(model_data.get("MAPE", {}).keys())
        week_numbers = [int(week.split('_')[1]) for week in weeks]
        
        # Tracer les MAPE si elles existent
        mape_values = list(model_data.get("MAPE", {}).values())
        if mape_values:
            ax.plot(week_numbers, mape_values, label=f'{model_name} MAPE')
    
    # Ajouter des labels et un titre au graphique MAPE
    ax.set_xlabel('Weeks')
    ax.set_ylabel('MAPE')
    ax.set_title('MAPE per Week for Each Model')
    ax.legend()
    
    # Enregistrer le graphique MAPE
    fig.savefig('visualization/mape_per_week.png')
    plt.close(fig)  # Fermer la figure pour libérer la mémoire

def plot_predictions(json_file,df):
    """
    Crée un graphique pour les prédictions et l'enregistre en tant qu'image.
    
    Args:
        json_file (str): Chemin vers le fichier JSON contenant les données.
    """
    # Lire le fichier JSON
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Créer la figure pour les prédictions
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prétraiter les données pour le graphique des prédictions
    for model_name, model_data in data.items():
        model_name = 'BayesianRidge'
        model_data = data['BayesianRidge']
        if model_name == "ARIMA":
            continue
        
        # Convertir les semaines en un format numérique pour l'axe x
        weeks = list(model_data.get("Predictions", {}).keys())
        week_numbers = [int(week.split('_')[1]) for week in weeks]
        
        # Tracer les prédictions si elles existent
        prediction_values = list(model_data.get("Predictions", {}).values())
        if prediction_values:
            ax.plot(week_numbers, prediction_values, label=f'{model_name}')


    actual_values = list(df[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values)
    ax.plot(week_numbers, actual_values, label='Actual', color='#000000')

    prediction_values = list(data['KNeighborsRegressor'].get("Predictions", {}).values())
    print(len(prediction_values))
    print(len(actual_values))




    # Ajouter des labels et un titre au graphique des prédictions
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions per Week for Each Model')
    ax.legend()
    
    # Enregistrer le graphique des prédictions
    fig.savefig('visualization/predictions_per_week.png')
    plt.close(fig)  




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

