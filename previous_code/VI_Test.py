import os
import json
import joblib
import pandas as pd
import numpy as np
import json

import time
import matplotlib.pyplot as plt

from sklearn.calibration import cross_val_predict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from sklearn.model_selection import cross_validate, train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import NBEATSModel, NaiveSeasonal, ExponentialSmoothing
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split as darts_train_test_split

from prophet import Prophet

from II_Data_visualization import plot_correlation_matrix  
from III_Preprocessing import load_and_preprocess_data, preprocess_arima_data
from IV_Main import split_data, train_model, find_best_model_configs
from V_Save_result import mape, calc_scores, plot_all_models_predictions, plot_all_models_mape, plot_weekly_mape, save_results_to_excel,  update_global_results, plot_predictions_by_model, calculate_weekly_mape

import Constants as const

from tpot import TPOTRegressor
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.model_selection import RandomizedSearchCV

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

# Chargement des données
# Remplacez cette ligne par le chargement de vos données
data = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')


for target in const.TARGET_COLUMN :
    # Sélection de la série temporelle à lisser
    series = data[target]

    # Initialisation des variables pour stocker les résultats
    window_sizes = range(5, 11, 2)  # Fenêtres impaires entre 5 et 50
    mse_values = []

    # Test de différentes tailles de fenêtre
    for window_size in window_sizes:
        # Application du filtre de Savitzky-Golay
        smoothed_series = savgol_filter(series, window_length=window_size, polyorder=3)
        
        # Calcul du MSE entre la série originale et la série lissée
        mse = mean_squared_error(series, smoothed_series)
        mse_values.append(mse)

    # Détermination de la meilleure fenêtre (celle avec le MSE le plus bas)
    best_window_size = window_sizes[np.argmin(mse_values)]

    # Affichage des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, mse_values, marker='o')
    plt.title('MSE en fonction de la taille de la fenêtre de Savitzky-Golay')
    plt.xlabel('Taille de la fenêtre')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.show()

    print(f"La meilleure taille de fenêtre est {best_window_size} avec un MSE de {min(mse_values):.4f}")

    # Application de la fenêtre optimale
    smoothed_series_best = savgol_filter(series, window_length=best_window_size, polyorder=3)

    # Affichage de la série lissée
    plt.figure(figsize=(10, 6))
    plt.plot(series, label='Série originale', alpha=0.5)
    plt.plot(smoothed_series_best, label='Série lissée (Savitzky-Golay)', color='red')
    plt.title('Comparaison de la série originale et de la série lissée')
    plt.xlabel('Temps')
    plt.ylabel('Prix du lait')
    plt.legend()
    plt.grid(True)
    plt.show()













"""


if __name__ == "__main__":
        
    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')
    

    if const.ACTION["smooth window test"]:
        for target in const.TARGET_COLUMN :
            targ_dir = const.TARGET_DIR[f'{target}']
            for smooth_window_val in const.SMOOTH_WINDOW :
                preprocess_df = load_and_preprocess_data(smooth_window_val, df, target)


                X_train, y_train, X_test, y_true = split_data(preprocess_df, target)
                test_dates = preprocess_df['Date'].iloc[len(X_train):len(X_train) + len(X_test)]

                for model_name, model in const.MODELS.items():
                    if const.ACTION["Train models"]:
                        results = train_model(X_train, y_train, X_test, y_true, model_name, model )

                if const.ACTION["Save models"]:
                    find_best_model_configs(model_name, const.WINDOWS_LIST, const.NON_ML)

                    with open(f'result/{targ_dir}/train_by_model/{model_name}_best_model.json', 'r') as file:
                        results = json.load(file)

                    weekly_mape = calculate_weekly_mape(model_name, targ_dir,np.array(y_true), np.array(results['Predictions']), const.FORECAST_WEEKS)

                    plot_predictions_by_model(results, test_dates, y_true, model_name, targ_dir)
                    plot_weekly_mape(test_dates, weekly_mape, model_name,targ_dir)
                    update_global_results(model_name, targ_dir)
    
"""