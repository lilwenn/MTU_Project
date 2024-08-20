import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from config import CONFIG  # Import CONFIG from the config module

class ResultAnalyzer:
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def find_global_best_configs(self, model_name, lag_list, non_ml_models):
        best_mape = float('inf')
        best_combination = None
        best_lag = None

        if model_name not in non_ml_models:
            for lag in lag_list:
                file_path = os.path.join(f'result/{self.target_dir}/train_by_model/{model_name}_{lag}_model_{CONFIG["FORECAST_WEEKS"]}.json')

                if not os.path.exists(file_path):
                    print(f"Result file for iteration {lag} does not exist: {file_path}")
                    continue

                with open(file_path, 'r') as f:
                    data = json.load(f)

                for scaler_name, scaler_data in data.items():
                    for scoring_name, metrics in scaler_data.items():
                        if "MAPE_Score" in metrics:  # Changez "MAPE_Score" en "MAPE"
                            mape = metrics["MAPE_Score"]  # Utilisez "MAPE" au lieu de "MAPE_Score"
                            if mape < best_mape:  # Supprimez la condition '1 < mape'
                                best_mape = mape
                                best_lag = lag
                                best_combination = {
                                    'Execution Time': metrics['Execution Time'],
                                    'Mean Train Score': metrics['Mean Train Score'],
                                    'Mean Test Score': metrics['Mean Test Score'],
                                    'MAPE_Score': metrics["MAPE_Score"], 
                                    'MAE Score': metrics['MAE'],
                                    'Predictions': metrics['Predictions'],
                                    'Scaler': scaler_name,
                                    'Scoring': scoring_name,
                                    'Selected Features': metrics['Selected Features'],
                                    'Best Parameters': metrics['Best Parameters'],
                                    'Best lag': best_lag
                                }


        output_file_path = f'result/{self.target_dir}/train_by_model/{model_name}_best_model.json'
        with open(output_file_path, 'w') as f:
            json.dump(best_combination, f, indent=4)

        print(f"Best combination for {model_name} across all iterations saved (Iteration {best_lag})")

        return best_combination

    def update_global_results(self, model_name):
        best_model_result_file = f'result/{self.target_dir}/train_by_model/{model_name}_best_model.json'
        global_results_file = f'result/{self.target_dir}/global_results.json'

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

    def calc_scores(self, y_true):
        with open(f'result/{self.target_dir}/global_results.json', 'r') as f:
            data = json.load(f)

        for model_name in CONFIG['MODELS']:
            with open(f'result/{self.target_dir}/weekly_mape/{model_name}_weekly_mape.json', 'r') as f:
                mape_dict = json.load(f)

            predicted = data[model_name]["Predictions"]
            mae_dict, mae2_dict, me_dict, mse_dict, rmse_dict = {}, {}, {}, {}, {}

            for week in range(1, CONFIG['FORECAST_WEEKS'] + 1):
                actual_week = y_true[week - 1:week]
                predicted_week = predicted[week - 1:week]

                mae_dict[f'week_{week}'] = mean_absolute_error(actual_week, predicted_week)
                mae2_dict[f'week_{week}'] = median_absolute_error(actual_week, predicted_week)
                mse = mean_squared_error(actual_week, predicted_week)
                mse_dict[f'week_{week}'] = mse
                rmse_dict[f'week_{week}'] = math.sqrt(mse)
                me_dict[f'week_{week}'] = np.mean(actual_week - predicted_week)

            data[model_name].update({
                'MAPE': mape_dict,
                'MAE': mae_dict,
                'MAE2': mae2_dict,
                'ME': me_dict,
                'MSE': mse_dict,
                'RMSE': rmse_dict
            })

        with open(f'result/{self.target_dir}/global_results.json', 'w') as f:
            json.dump(data, f, indent=4)

        return data

    def save_results_to_excel(self, output_excel_file, week):
        results = []
        with open(f'result/{self.target_dir}/global_results.json', 'r') as f:
            model_data = json.load(f)

        for model_name in model_data.keys():
            with open(f'result/{self.target_dir}/weekly_mape/{model_name}_weekly_mape.json', 'r') as f:
                mape_data = json.load(f)
            
            results.append({
                'Model Name': model_name,
                'MAPE Score': model_data[model_name]["MAPE_Score"],
                'MAPE 52': mape_data[f'week_{week}'],
                'MAPE 1': mape_data['week_1'],
                'MAE': model_data[model_name]['MAE Score'],
                'Execution Time': model_data[model_name]['Execution Time'],
                'Scaler': model_data[model_name]['Scaler'],
                'Scoring': model_data[model_name]['Scoring'],
                'Lag': model_data[model_name]["Best lag"]
            })

        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by='MAPE Score', ascending=True)
            df_results.to_excel(output_excel_file, index=False)
            print(df_results)
            print(f"Results saved to {output_excel_file}.")
        else:
            print("No results to save.")
            
    def calculate_weekly_mape(self, model_name, y_true: np.ndarray, y_pred: np.ndarray, forecast_weeks: int) -> dict:
        # Notez l'ajout de 'self' comme premier paramètre

        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Ensure the directory exists
        output_dir = f'result/{self.target_dir}/weekly_mape/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        weekly_errors = {}
        for week in range(1, forecast_weeks + 1):
            try:
                actual_value = y_true[week - 1]
                predicted_value = y_pred[week - 1]
                percentage_error = abs((actual_value - predicted_value) / actual_value) * 100
                weekly_errors[f'week_{week}'] = percentage_error
            except IndexError:
                print(f"Index {week - 1} out of range for y_true or y_pred.")
                weekly_errors[f'week_{week}'] = None  # Handle missing data gracefully

        # Print weekly MAPE
        for week, mape_value in weekly_errors.items():
            if mape_value is not None:
                print(f'{model_name} - {week}: MAPE = {mape_value:.2f}%')

        # Save weekly MAPE to a file
        with open(os.path.join(output_dir, f'{model_name}_weekly_mape.json'), 'w') as f:
            json.dump(weekly_errors, f, indent=4)

        return weekly_errors

