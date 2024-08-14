import os
import json
import joblib
import pandas as pd
import numpy as np
import json
import math

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
from III_Preprocessing import load_and_preprocess_data, preprocess_arima_data
from V_Save_result import mape, calc_scores, plot_all_models_predictions, plot_all_models_mape, plot_weekly_mape, save_results_to_excel,  update_global_results, plot_predictions_by_model, calculate_weekly_mape
import Constants as const

from tpot import TPOTRegressor
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dict(total_execution_time, cv_results, y_true, y_pred, selected_features, best_params):
    results = {
        'Execution Time': total_execution_time,
        'Mean Train Score': -np.mean(cv_results['train_score']),
        'Mean Test Score': -np.mean(cv_results['test_score']),
        'MAPE_Score': mape(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAE2': median_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'ME': np.mean(y_true - y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
        'Selected Features': selected_features,
        'Best Parameters': best_params,
        'Predictions': y_pred.tolist()  
    }
        
    return results


def split_data(df, target_column):
    # Assurer que les données sont triées chronologiquement
    df = df.sort_values('Date')
    
    # Séparer les caractéristiques et la cible
    features = [col for col in df.columns if col != target_column and col != 'Date']
    
    X = df[features]
    y = df[target_column]
    
    # Utiliser TimeSeriesSplit pour la séparation train/test
    tscv = TimeSeriesSplit(n_splits=2, test_size=const.FORECAST_WEEKS)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print(f'Size of the training dataset: {X_train.shape}')
    print(f'Size of the testing dataset: {X_test.shape}')
    
    return X_train, y_train, X_test, y_test


def build_pipeline(scoring_func, scoring_name, scaler):
    default_k = 5
    if scoring_func is not None:
        pipeline_steps = [
            ('scaler', scaler),
            ('selectkbest', SelectKBest(score_func=scoring_func, k=const.K_VALUES.get(scoring_name, default_k))),
            ('model', model)
        ]
    else:
        pipeline_steps = [
            ('scaler', scaler),
            ('model', model)
        ]
    
    return Pipeline(pipeline_steps)

def train_model(X_train, y_train, X_test, y_true, model_name, model, lag):
    result = {}
    for scaler_name, scaler in const.SCALERS.items():
        result[scaler_name] = {}
        for scoring_name, scoring_func in const.SCORING_METHODS.items():
            try:
                pipeline = build_pipeline(scoring_func, scoring_name, scaler)
                
                # Utiliser TimeSeriesSplit pour la validation croisée
                tscv = TimeSeriesSplit(n_splits=5)
                
                clf = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=const.HYPERPARAMETERS[model_name],
                    n_iter=5, 
                    cv=tscv,  # Utiliser TimeSeriesSplit
                    verbose=1,
                    n_jobs=-1,
                    error_score='raise'
                )

                start_time = time.time()
                clf.fit(X_train, y_train)
                end_time = time.time()
                total_execution_time = end_time - start_time

                best_estimator = clf.best_estimator_

                # Faire des prédictions sur l'ensemble de test
                y_pred = best_estimator.predict(X_test)

                if 'selectkbest' in best_estimator.named_steps:
                    selectkbest_step = best_estimator.named_steps['selectkbest']
                    selected_features = list(X_train.columns[selectkbest_step.get_support()])
                else:
                    selected_features = list(X_train.columns)

                best_params = clf.best_params_

                # Utiliser TimeSeriesSplit pour la validation croisée finale
                cv_results = cross_validate(best_estimator, X_train, y_train, 
                                            cv=tscv, scoring='neg_mean_squared_error', return_train_score=True)

                result[scaler_name][scoring_name] = create_dict(
                    total_execution_time, cv_results, y_true, y_pred, selected_features, best_params
                )

                print(f'Scaler: {scaler_name}, Scoring: {scoring_name}, Execution Time: {total_execution_time}, '
                      f'Mean Train Score: {np.mean(cv_results["train_score"])}, Mean Test Score: {np.mean(cv_results["test_score"])}\n \n ')
            
            except ValueError as e:
                print(f"Error with Scaler: {scaler_name}, Scoring: {scoring_name}: {e}")
                result[scaler_name][scoring_name] = {'Error': str(e)}

    print(f"All combinations of {model_name} for window = {lag} saved in json")
    with open(f'result/{targ_dir}/train_by_model/{model_name}_{lag}_model_{const.FORECAST_WEEKS}.json', 'w') as f:
        json.dump(result, f, indent=4)

    return result

def find_global_best_configs(model_name, lag_list, non_ml_models):
    """
    Analyzes JSON result files to find the best model configurations for each model based on the best MAPE across all iterations.
    """
    best_mape = float('inf')
    best_combination = None
    best_lag = None

    if model_name not in non_ml_models:
        for lag in lag_list:
            file_path = os.path.join(f'result/{targ_dir}/train_by_model/{model_name}_{lag}_model_{const.FORECAST_WEEKS}.json')

            if not os.path.exists(file_path):
                print(f"Result file for iteration {lag} does not exist: {file_path}")
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            for scaler_name, scaler_data in data.items():
                for scoring_name, metrics in scaler_data.items():
                    if "MAPE_Score" in metrics:
                        mape = metrics["MAPE_Score"]
                        if mape < best_mape and 1 < mape:
                            best_mape = mape
                            best_lag = lag
                            best_combination = {
                                'Execution Time': metrics['Execution Time'],
                                'Mean Train Score': metrics['Mean Train Score'],
                                'Mean Test Score': metrics['Mean Test Score'],
                                'MAPE_Score': metrics['MAPE_Score'],
                                'MAE Score': metrics['MAE'],
                                'Predictions': metrics['Predictions'],
                                'Scaler': scaler_name,
                                'Scoring': scoring_name,
                                'Selected Features': metrics['Selected Features'],
                                'Best Parameters': metrics['Best Parameters'],
                                'Best lag' : best_lag
                            }
                    else:
                        continue
                    
    output_file_path = f'result/{targ_dir}/train_by_model/{model_name}_best_model.json'
    with open(output_file_path, 'w') as f:
        json.dump(best_combination, f, indent=4)

    print(f"Best combination for {model_name} across all iterations saved (Iteration {best_lag})")

    return best_combination

if __name__ == "__main__":
    print('----------------------- START----------------------')
    
    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

    for target in const.TARGET_COLUMN :
        targ_dir = const.TARGET_DIR[f'{target}']

        for lag in const.LAG_LIST:

            preprocess_df = load_and_preprocess_data(lag, df, target)
            X_train, y_train, X_test, y_true = split_data(preprocess_df, target)
            test_dates = preprocess_df['Date'].iloc[len(X_train):len(X_train) + len(X_test)]

            for model_name, model in const.MODELS.items():
                if const.ACTION["Train models"]:
                    print(f'\n ----------- {model_name} Train - lag {lag}----------- \n')
                    results = train_model(X_train, y_train, X_test, y_true, model_name, model, lag)

        if const.ACTION["Save models"]:
            for model_name, model in const.MODELS.items():
                print(f'\n ----------- {model_name} saving ----------- \n')
                best_combination = find_global_best_configs(model_name, const.LAG_LIST, const.NON_ML)
                y_pred = best_combination['Predictions']

                weekly_mape = calculate_weekly_mape(model_name, targ_dir,np.array(y_true), np.array(y_pred), const.FORECAST_WEEKS)

                plot_predictions_by_model(y_pred, test_dates, y_true, model_name, targ_dir)
                plot_weekly_mape(test_dates, weekly_mape, model_name,targ_dir)
                update_global_results(model_name, targ_dir)


            calc_scores(y_true, targ_dir)
            output_excel_file = f'result/{targ_dir}/{const.FORECAST_WEEKS}w_global_model_results.xlsx' 
            save_results_to_excel(output_excel_file, const.FORECAST_WEEKS, targ_dir)

            plot_all_models_predictions(test_dates, y_true, targ_dir)
            plot_all_models_mape(test_dates, targ_dir)

    print('----------------------- END----------------------')


