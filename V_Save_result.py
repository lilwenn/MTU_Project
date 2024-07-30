import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Constants as const



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



    # Plot the performance of Darts models
    plot_darts_model_performance(results, df)
    plot_prophet_model_performance(results, df)

    
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