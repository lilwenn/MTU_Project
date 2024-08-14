import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Constants as const  

class Visualizer:
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def plot_all_models_predictions(self, dates, y_true):
        plt.figure(figsize=(12, 6))
        palette = sns.color_palette("husl", len(const.MODELS))

        # Plotting actual values
        plt.plot(dates, y_true, label='Actual', color='black')
        i = 0
        for model_name in const.MODELS.keys():
            with open(f'result/{self.target_dir}/train_by_model/{model_name}_best_model.json', 'r') as file:
                results = json.load(file)
            
            y_pred = results['Predictions']
            min_length = min(len(dates), len(y_true), len(y_pred))
            dates = dates[:min_length]
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]

            # Tracer les valeurs prédites pour chaque modèle avec une couleur différente
            plt.plot(dates, y_pred, label=model_name, color=palette[i])
            i += 1

        plt.title('Actual vs Predicted Values for All Models')
        plt.xlabel('Date')
        plt.ylabel(self.target_dir)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/{self.target_dir}/plot_predictions/predictions_all_models.png')
        plt.close()

    def plot_predictions_by_model(self, y_pred, dates, y_true, model_name):
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
        plt.ylabel(self.target_dir)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/{self.target_dir}/plot_predictions/predictions_{model_name}.png')
        plt.close()

    def plot_all_models_mape(self, dates):
        plt.figure(figsize=(12, 6))
        palette = sns.color_palette("husl", len(const.MODELS))

        i = 0
        for model_name in const.MODELS.keys():
            with open(f'result/{self.target_dir}/weekly_mape/{model_name}weekly_mape.json', 'r') as file:
                weekly_mape = json.load(file)

            mape_values = list(weekly_mape.values())

            # Ensure dates are the same length as mape_values
            min_length = min(len(dates), len(mape_values))
            dates = dates[:min_length]
            mape_values = mape_values[:min_length]

            # Plotting MAPE values for each model
            plt.plot(dates, mape_values, label=model_name, color=palette[i])
            i += 1

        plt.title('Weekly MAPE for All Models')
        plt.xlabel('Week')
        plt.ylabel('MAPE (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/{self.target_dir}/plot_mape/mape_all_models.png')
        plt.close()

    def plot_weekly_mape(self, dates, weekly_mape, model_name):
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
        plt.savefig(f'result/{self.target_dir}/plot_mape/weekly_mape_{model_name}.png')
        plt.close()
