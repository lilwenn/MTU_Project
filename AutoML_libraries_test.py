import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import Constants as const

from darts import TimeSeries
from darts.models import ExponentialSmoothing, Prophet
from darts.dataprocessing.transformers import Scaler
#from evalml.automl import AutoMLSearch
from holidays.countries import Turkey

#from fbprophet import Prophet as FbProphet

from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.core.callbacks import SummaryCallback
from hyperts.framework.search_space.macro_search_space import DLForecastSearchSpace
from hyperts.framework.wrappers.dl_wrappers import SimpleRNNWrapper
from h2o.automl import H2OAutoML
import h2o
from tpot import TPOTRegressor

# Initialize H2O
h2o.init()

# Define additional functions for each library
def train_and_predict_darts(df, forecast_weeks):
    series = TimeSeries.from_dataframe(df, time_col='Date', value_cols='litres')
    model = ExponentialSmoothing()
    model.fit(series)
    prediction = model.predict(forecast_weeks)
    return prediction

"""
def train_and_predict_evalml(df, features, target_col):
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression', max_batches=1)
    automl.search()
    best_pipeline = automl.best_pipeline
    y_pred = best_pipeline.predict(X_test)
    return y_pred

def train_and_predict_fbprophet(df, forecast_weeks):
    df = df.rename(columns={'Date': 'ds', 'litres': 'y'})
    model = FbProphet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_weeks, freq='W')
    forecast = model.predict(future)
    return forecast['yhat'][-forecast_weeks:]"""

def train_and_predict_gluonts(df, forecast_weeks):
    train_data = ListDataset([{"start": df.index[0], "target": df['litres']}], freq="W")
    estimator = SimpleFeedForwardEstimator(num_hidden_dimensions=[10], prediction_length=forecast_weeks, trainer=Trainer(epochs=5))
    predictor = estimator.train(train_data)
    forecast_it, ts_it = predictor.predict(train_data, num_samples=100)
    forecast_entry = next(forecast_it)
    return forecast_entry.mean

def train_and_predict_hypernets(df, forecast_weeks):
    search_space = DLForecastSearchSpace()
    searcher = RandomSearcher(search_space, optimize_direction='min')
    estimator = SimpleRNNWrapper()
    hyper_model = estimator.fit(searcher, df['litres'], epochs=10)
    prediction = hyper_model.predict(forecast_weeks)
    return prediction

def train_and_predict_h2o(df, features, target_col):
    h2o_df = h2o.H2OFrame(df)
    train, test = h2o_df.split_frame(ratios=[0.8])
    aml = H2OAutoML(max_runtime_secs=300)
    aml.train(y=target_col, training_frame=train)
    preds = aml.leader.predict(test)
    return preds.as_data_frame()

def train_and_predict_tpot(df, features, target_col):
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
    tpot.fit(X_train, y_train)
    y_pred = tpot.predict(X_test)
    return y_pred

def train_models(df, model_name, model_func):
    try:
        print(f"Training model: {model_name}")
        if model_name in ['Darts', 'FbProphet', 'GluonTS', 'Hypernets']:
            forecast_weeks = const.forecast_weeks
            predictions = model_func(df, forecast_weeks)
        else:
            features = df.drop(columns=['Date', 'litres']).columns.tolist()
            target_col = 'litres'
            predictions = model_func(df, features, target_col)
        
        result_path = f'result/by_model/{model_name}_{const.forecast_weeks}week.json'
        with open(result_path, 'w') as file:
            json.dump(predictions.tolist(), file)
        
        print(f"Model {model_name} trained and results saved to {result_path}")
    except Exception as e:
        print(f"Failed to train model {model_name}: {e}")

def find_best_model_configs(result_dir, file_name, best_combinations, results_list):
    try:
        file_path = os.path.join(result_dir, file_name)
        with open(file_path, 'r') as file:
            results = json.load(file)
        
        model_name = file_name.split('_')[0]
        mae_score = mae(results['actual'], results['predicted'])
        
        results_list.append({
            'model': model_name,
            'file': file_name,
            'mae': mae_score
        })
        
        if model_name not in best_combinations or best_combinations[model_name]['mae'] > mae_score:
            best_combinations[model_name] = {
                'file': file_name,
                'mae': mae_score
            }
    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")

def plot_model_performance(results_file, const):
    try:
        with open(results_file, 'r') as file:
            best_combinations = json.load(file)
        
        model_names = list(best_combinations.keys())
        mae_scores = [best_combinations[model]['mae'] for model in model_names]
        
        plt.figure(figsize=(12, 6))
        plt.barh(model_names, mae_scores, color='skyblue')
        plt.xlabel('Mean Absolute Error (MAE)')
        plt.title(f'Model Performance Comparison ({const.forecast_weeks} Weeks Forecast)')
        plt.gca().invert_yaxis()
        plt.show()
    except Exception as e:
        print(f"Failed to plot model performance: {e}")


# Add these models to your constants
const.models['Darts'] = train_and_predict_darts
#const.models['EvalML'] = train_and_predict_evalml
#const.models['FbProphet'] = train_and_predict_fbprophet
const.models['GluonTS'] = train_and_predict_gluonts
const.models['Hypernets'] = train_and_predict_hypernets
const.models['H2O'] = train_and_predict_h2o
const.models['TPOT'] = train_and_predict_tpot


if __name__ == "__main__":
    df = pd.read_excel('spreadsheet/lagged_results.xlsx')

    for model_name, model in const.models.items():
        train_models(df, model_name, model)

    best_combinations = {}
    result_dir = 'result/by_model/'
    results_list = []

    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} does not exist.")
    else:
        json_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{const.forecast_weeks}week.json')]
        for file_name in json_files:
            find_best_model_configs(result_dir, file_name, best_combinations, results_list)
    
    with open(f'result/best_combinations_{const.forecast_weeks}week.json', 'w') as file:
        json.dump(best_combinations, file)
    
    plot_model_performance(f'result/best_combinations_{const.forecast_weeks}week.json', const)
