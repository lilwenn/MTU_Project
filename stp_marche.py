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

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
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
        mape_dict[f'week_{week}'] = mean_absolute_percentage_error(actual_week, predicted_week) if np.any(predicted_week != 0) else np.nan
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

    results['Overall_MAPE'] = np.mean(list(mape_dict.values()))
    results['Overall_MAE'] = np.mean(list(mae_dict.values()))

    print(results['Overall_MAPE'])

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
    if 'selectkbest' in model_pipeline.named_steps:
        selected_features = model_pipeline.named_steps['selectkbest'].get_support(indices=True)
        selected_feature_names = [features[i] for i in selected_features]
    else:
        selected_feature_names = features

    # Predict future values
    last_row = df[features].iloc[-1]
    future = pd.DataFrame([last_row])
    prediction = model_pipeline.predict(future)

    return y_test, prediction[0]

def train_models(df, model_name, model, w):
    features = [col for col in df.columns if not col.startswith(f'{const.TARGET_COLUMN}_next_') and col != 'Date']
    result = {}

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

            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=const.HYPERPARAMETERS[model_name],
                n_iter=10,
                cv=5,
                verbose=2,
                n_jobs=-1
            )

            start_time = time.time()
            random_search.fit(df[features], df[const.TARGET_COLUMN])
            end_time = time.time()
            total_execution_time = end_time - start_time

            best_pipeline = random_search.best_estimator_

            pred_list = []
            for week in range(1, const.FORECAST_WEEKS + 1):
                target_col = f'{const.TARGET_COLUMN}_next_{week}weeks'
                y_test, prediction = train_and_predict(df, features, target_col, best_pipeline)
                pred_list.append(prediction)

            result[scaler_name][scoring_name] = calc_scores(df[const.TARGET_COLUMN].iloc[-const.FORECAST_WEEKS:].values, pred_list, total_execution_time)

    print(f"All combinations of {model_name} for window ={w} saved in json")
    with open(f'result/by_model/{model_name}_{w}_model_{const.FORECAST_WEEKS}.json', 'w') as f:
        json.dump(result, f, indent=4)



if __name__ == "__main__":
    print('----------------------- START----------------------')
    
    df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')
    for model_name, model in const.MODELS.items():

        print(f' MODEL : {model_name}')
        if model_name not in const.Non_ml:
            for w in const.window_list:
                lagged_df = load_and_preprocess_data(w, df)
                train_models(lagged_df, model_name, model,w)
        else:
            train_models(df, model_name, model, 0)

    print('----------------------- END----------------------')

