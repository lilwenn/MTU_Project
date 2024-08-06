import os
import json
import joblib
import pandas as pd
import numpy as np
import json

import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
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
from V_Save_result import plot_predictions, plot_mape, save_results_to_excel, find_best_model_configs, update_global_results
import Constants as const

from tpot import TPOTRegressor
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima import model_selection
from sklearn.model_selection import RandomizedSearchCV

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calc_scores(actual: np.ndarray, predicted: np.ndarray, execution_time: float) -> dict:
    
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
        pred_dict[f'week_{week}'] = predicted_week[0] if len(predicted_week) > 0 else np.nan

        # Calculate metrics with safety checks
        mae_dict[f'week_{week}'] = mean_absolute_error(actual_week, predicted_week)
        mae2_dict[f'week_{week}'] = median_absolute_error(actual_week, predicted_week)
        mse = mean_squared_error(actual_week, predicted_week)
        mse_dict[f'week_{week}'] = mse
        rmse_dict[f'week_{week}'] = math.sqrt(mse)
        mape_dict[f'week_{week}'] = mape(actual_week, predicted_week) 
        print(f"mape au debut pour la semain {week} : {mape_dict[f'week_{week}']}")
        
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

    results['mean_MAPE'] = np.mean(list(mape_dict.values()))

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

    return y_test, y_pred, execution_time


def split_data(df):
    # Splitting into train and test sets
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    print(f'Size of the training dataset: {train_df.shape}')
    print(f'Size of the testing dataset: {test_df.shape}')

    target_column = const.TARGET_COLUMN
    features = [col for col in train_df.columns if col != target_column]
    
    # Split the training and testing datasets into X and y
    X_train = train_df[features]
    y_train = train_df[target_column]
    X_test = test_df[features]
    y_true = test_df[target_column]

    print(f'Features for training: {X_train.shape}')
    print(f'Target for training: {y_train.shape}')
    print(f'Features for testing: {X_test.shape}')
    print(f'True target for testing: {y_true.shape}')

    return X_train, y_train, X_test, y_true


def train_model(df, w):
    features = [col for col in df.columns if not col.startswith(f'{const.TARGET_COLUMN}_next_') and col != 'Date']
    target_col = const.TARGET_COLUMN
    result = {}

    for scaler_name, scaler in const.SCALERS.items():
        for scoring_name, scoring_func in const.SCORING_METHODS.items():
            default_scoring_func = f_regression
            default_k = 5

            pipeline = Pipeline([
                ('scaler', scaler),
                ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func,
                                            k=const.K_VALUES.get(scoring_name, default_k))),
                ('model', RandomForestRegressor())
            ])

            clf = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=const.HYPERPARAMETERS['RandomForestRegressor'],
                n_iter=10,
                cv=5,
                verbose=1,
                n_jobs=-1
            )

            start_time = time.time()
            clf.fit(df[features], df[target_col])
            end_time = time.time()
            total_execution_time = end_time - start_time

            best_estimator = clf.best_estimator_

            # Perform cross-validation
            cv_results = cross_validate(best_estimator, df[features], df[target_col], 
                                        cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                
            y_pred = cross_val_predict(best_estimator, df[features], df[target_col], cv=5)

            result[scaler_name] = {
                'Execution Time': total_execution_time,
                'Mean Train Score': -np.mean(cv_results['train_score']),
                'Mean Test Score': -np.mean(cv_results['test_score']),
                'MAPE': mape(df[target_col], y_pred),
                'Predictions': y_pred.tolist()  # Convert predictions to a list for JSON serialization
            }

            print(f'Scaler: {scaler_name}, Scoring: {scoring_name}, MAPE: {result[scaler_name]["MAPE"]:.2f}')



    return result


if __name__ == "__main__":
    print('----------------------- START----------------------')
    
    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

    preprocess_df = load_and_preprocess_data(4, df)
    X_train, y_train, X_test, y_true = split_data(preprocess_df)

    model_name = 'RandomForestRegressor'
    model = RandomForestRegressor()

    results = train_model(preprocess_df, 4)
    with open(f'result/test0608.json', 'w') as f:
        json.dump(results, f, indent=4)


    #train_models(preprocess_df, model_name, model, 4)





"""
    for model_name, model in const.MODELS.items():
        print(f' MODEL : {model_name}')
        if model_name not in const.NON_ML:
            for w in const.WINDOWS_LIST:
                preprocess_df = load_and_preprocess_data(w, df)
                X_train, y_train, X_test, y_true = split_data(preprocess_df)


                train_models(lagged_df, model_name, model,w)
        else:
            train_models(df, model_name, model, 0)

    # Find the best configurations for each model
    for model_name in const.MODELS.keys():
        find_best_model_configs(model_name, const.window_list, const.Non_ml)
        update_global_results(model_name)

    week_predict = 52
    output_excel_file = f'result/{week_predict}w_global_model_results.xlsx' 
    save_results_to_excel(output_excel_file, const.FORECAST_WEEKS)


    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')
    plot_mape('result/global_results.json')
    plot_predictions('result/global_results.json',df)

    print('----------------------- END----------------------')

"""






