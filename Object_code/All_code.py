import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso, PassiveAggressiveRegressor, Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from scipy.stats import uniform, loguniform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def feature_importance(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model.feature_importances_

def pearson_corr(X, y):
    import numpy as np
    corr_matrix = np.corrcoef(X, y, rowvar=False)
    corr_with_target = np.abs(corr_matrix[:-1, -1])  
    return corr_with_target

CONFIG = {
    "BASE_DIR": BASE_DIR,
    "TARGET_COLUMN": ['litres'],
    "TARGET_DIR": {
        'litres': 'liter_results',
        'Ireland_Milk_Price': 'prices_results'
    },
    "FORECAST_WEEKS": 52,
    "HORIZON": 52,
    "LAG_LIST": [1, 2 , 3, 4, 5, 6, 7],
    "SMOOTH_WINDOW": 5,
    "NON_ML": ['Pmdarima', 'Darts', 'ARIMA'],
    "ACTION": {
        "time_series_smoothing": True,
        "shifting": True,
        "compare_lifting_methods": False,
        "Multi-step": False,
        "Train models": True,
        "Save models": True
    },
    "MODELS": {
        'BayesianRidge': BayesianRidge(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'ExtraTreeRegressor': ExtraTreeRegressor(),
        'GaussianProcessRegressor': GaussianProcessRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'Lasso': Lasso(),
        'LinearRegression': LinearRegression(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'Ridge': Ridge(),
    },

    "HYPERPARAMETERS": {
        'BayesianRidge': {
            'model__tol': uniform(1e-2, 1e-4),
            'model__alpha_1': uniform(1e-5, 1e-7),
            'model__alpha_2': uniform(1e-5, 1e-7),
            'model__lambda_1': uniform(1e-5, 1e-7),
            'model__lambda_2': uniform(1e-5, 1e-7)
        },
        'DecisionTreeRegressor': {
            'model__criterion': ['absolute_error', 'squared_error'],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
        },
        'ExtraTreeRegressor': {
            'model__criterion': ['squared_error', 'absolute_error', 'poisson'],
            'model__max_depth': [8, 16, 32, 64, 128, None],
            'model__splitter': ['best', 'random'],
            'model__max_features': [None, 'sqrt', 'log2']
        },
        'GaussianProcessRegressor': {
            'model__alpha': loguniform(1e-12, 1e-8),
            'model__n_restarts_optimizer': [0, 1, 2, 3],
            'model__normalize_y': [True, False]
        },
        'KNeighborsRegressor': {
            'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__weights': ['uniform', 'distance'],
            'model__p': [2, 3, 4]
        },
        'Lasso': {
            'model__alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
            'model__tol': [1e-2, 1e-3, 1e-4],
            'model__selection': ['random', 'cyclic']
        },
        'LinearRegression': {},
        'PassiveAggressiveRegressor': {
            'model__C': [0.01, 0.1, 1.0, 10],
            'model__epsilon': [0.001, 0.01, 0.1, 1.0],
            'model__tol': [1e-3, 1e-4, 1e-5],
            'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'model__fit_intercept': [True, False],
            'model__max_iter': [500, 1000, 1500, 2000, 3000]
        },
        'RandomForestRegressor': {
            'model__n_estimators': [50, 100, 200, 300, 400],
            'model__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'model__max_features': ['sqrt', 'log2', None, 0.5, 1.0],
            'model__max_depth': [None, 16, 32, 64, 128],
            'model__min_samples_split': [2, 10, 20]
        },
        'Ridge': {
            'model__alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
            'model__tol': [1e-2, 1e-3, 1e-4],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },
    "K_VALUES": {
        'F_regression': 15,
        'Mutual_info_regression': 25,
        'Pearson Correlation': 35,
        'Feature Importance': 5
    },
    "SCORING_METHODS": {
        'F_regression': f_regression,
        'Mutual_info_regression': mutual_info_regression,
        'Pearson Correlation': pearson_corr,
        'Feature Importance': feature_importance,
        'No scoring': None
    },
    "SCALERS": {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(),
        'No scaling': None,
    }
}


import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter
import logging
from config import CONFIG
from typing import List

logging.basicConfig(level=logging.INFO)

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column

    def load_and_preprocess_data(self, lag: int) -> pd.DataFrame:
        output_file = os.path.join(CONFIG['BASE_DIR'], f'spreadsheet/preproc_lag_{lag}.xlsx')
        
        if os.path.exists(output_file):
            return pd.read_excel(output_file)

        logging.info('Preprocessing...')
        logging.info(f'Size of the initial dataset: {self.df.shape}')

        nan_columns = self.df.columns[self.df.isna().any()].tolist()
        if nan_columns:
            logging.warning(f"Columns with NaN values: {nan_columns}")
        
        self.df = self._impute_missing_values()

        
        if CONFIG['ACTION']["time_series_smoothing"]:
            self.df = self._time_series_smoothing(CONFIG['SMOOTH_WINDOW'])

        self.df = self._new_features_creation()

        if CONFIG['ACTION']["shifting"]:
            self.df = self._create_lag_features(lag)

        if CONFIG['ACTION']["Multi-step"]:
            self.df = self._create_multi_step_features(CONFIG['FORECAST_WEEKS'])

        logging.info(f'Size of the dataset after preprocessing: {self.df.shape}')

        self.df.to_excel(output_file, index=False)
        logging.info(f"Preprocessed file saved as {output_file}.")

        return self.df
    

    def _impute_missing_values(self) -> pd.DataFrame:
        df = self.df
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        return df

    def _time_series_smoothing(self, window: int) -> pd.DataFrame:
        prep_data = self.df.copy()
        exclude_cols = ['Date', 'Week', self.target_column, 'litres', 'num_suppliers']
        impute_cols = prep_data.columns.difference(exclude_cols)

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        prep_data[impute_cols] = imputer.fit_transform(prep_data[impute_cols])
        
        feature_cols = prep_data.columns.difference(exclude_cols)

        data_SG = self._create_savgol_smoothing(prep_data[feature_cols], window_length=window, polyorder=2)
        data_SG = pd.concat([prep_data[exclude_cols], data_SG], axis=1)

        return data_SG

    def _new_features_creation(self) -> pd.DataFrame:
        df = self.df
        if 'num_suppliers' not in df.columns or 'litres' not in df.columns:
            raise KeyError("Required columns 'num_suppliers' or 'litres' are missing from the DataFrame.")

        # Ensure no duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        df['yield_per_supplier'] = df['litres'] / df['num_suppliers']
        df['cos_week'] = np.cos(df['Week'] * (2 * np.pi / 52))
        df['past_values'] = df[self.target_column].expanding().mean()

        return df

    def _create_lag_features(self, lag: int) -> pd.DataFrame:
        data = self.df.copy()
        lagged_cols = []
        for col in data.columns:
            if col not in ["Date", "Week", self.target_column]:
                for i in range(1, lag + 1):
                    lagged_col = data[col].shift(i)
                    lagged_col.name = f'{col}-{i}'
                    lagged_cols.append(lagged_col)

        lagged_data = pd.concat(lagged_cols, axis=1)
        data = pd.concat([data, lagged_data], axis=1)

        # Supprimez les lignes avec des valeurs NaN (qui sont les premières lignes après le décalage)
        data = data.dropna()

        return data
    
    def _create_multi_step_features(self, n_steps: int) -> pd.DataFrame:
        df = self.df.copy()
        for step in range(1, n_steps + 1):
            df[f'{self.target_column}_step_{step}'] = df[self.target_column].shift(-step)
        
        df.dropna(inplace=True)
        return df
    
        
    def split_data(self):
        df = self.df.sort_values('Date')  # Assurez-vous que les données sont triées par date
        numeric_df = df.select_dtypes(include=[float, int])

        train_size = len(numeric_df) - CONFIG['FORECAST_WEEKS']
        
        train_df = numeric_df.iloc[:train_size]
        test_df = numeric_df.iloc[train_size:]

        logging.info(f'Size of the training dataset: {train_df.shape}')
        logging.info(f'Size of the testing dataset: {test_df.shape}')

        features = [col for col in train_df.columns if col != self.target_column]
        
        X_train = train_df[features]
        y_train = train_df[self.target_column]
        X_test = test_df[features]
        y_true = test_df[self.target_column]

        return X_train, y_train, X_test, y_true

    @staticmethod
    def _create_savgol_smoothing(data: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
        sg_cols = []
        for col in data.columns:
            smoothed_col = savgol_filter(data[col].fillna(method='bfill'), window_length=window_length, polyorder=polyorder, mode='interp')
            smoothed_col = pd.Series(smoothed_col, index=data.index, name=f'{col}')
            sg_cols.append(smoothed_col)
        
        sg_data = pd.concat(sg_cols, axis=1)
        return sg_data


import pandas as pd
import os
import logging
from config import CONFIG
from PredictionProject import PredictionProject

logging.basicConfig(level=logging.INFO)

def main():
    logging.info('----------------------- START ----------------------')
    
    # Load initial data
    df = pd.read_excel(os.path.join(CONFIG['BASE_DIR'], 'spreadsheet/Final_Weekly_2009_2021.xlsx'))

    # Create and run a prediction project for each target column
    for target in CONFIG['TARGET_COLUMN']:
        logging.info(f"\nProcessing target: {target}")
        
        # Create a PredictionProject instance for the current target column
        project = PredictionProject(df, target)
        
        # Run the project
        project.run()

    logging.info('----------------------- END ----------------------')

if __name__ == "__main__":
    main()


import time
import json
import os
import math
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_validate, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List
import logging
from config import CONFIG

logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_true, target_column):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_true = y_true
        self.target_column = target_column

    def train_model(self, model_name: str, model: Any, lag: int, target_column) -> Dict[str, Any]:
        result = {}
        for scaler_name, scaler in CONFIG['SCALERS'].items():
            result[scaler_name] = {}
            for scoring_name, scoring_func in CONFIG['SCORING_METHODS'].items():
                try:
                    pipeline = self._build_pipeline(scoring_func, scoring_name, scaler, model)
                    
                    tscv = TimeSeriesSplit(n_splits=5)
                    
                    clf = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=CONFIG['HYPERPARAMETERS'][model_name],
                        n_iter=30, 
                        cv=tscv,
                        verbose=1,
                        n_jobs=-1,
                        error_score='raise'
                    )

                    start_time = time.time()
                    clf.fit(self.X_train, self.y_train)
                    end_time = time.time()
                    total_execution_time = end_time - start_time

                    best_estimator = clf.best_estimator_

                    y_pred = best_estimator.predict(self.X_test)
                    mape_score = self._mape(self.y_true, y_pred)

                    selected_features = self._get_selected_features(best_estimator)

                    best_params = clf.best_params_

                    cv_results = cross_validate(best_estimator, self.X_train, self.y_train, 
                                                cv=tscv, scoring='neg_mean_squared_error', return_train_score=True)

                    result[scaler_name][scoring_name] = self._create_result_dict(
                        total_execution_time, cv_results, self.y_true, y_pred, selected_features, best_params, mape_score
                    )

                    logging.info(f'Scaler: {scaler_name}, Scoring: {scoring_name}, Execution Time: {total_execution_time:.2f} s, '
                                 f'Mean Train Score: {-np.mean(cv_results["train_score"]):.4f}, Mean Test Score: {-np.mean(cv_results["test_score"]):.4f}')
                
                except Exception as e:
                    logging.error(f"Error with Scaler: {scaler_name}, Scoring: {scoring_name}: {e}")
                    result[scaler_name][scoring_name] = {'Error': str(e)}

        target_dir_value = CONFIG['TARGET_DIR'].get(self.target_column, 'default_value')
        result_dir = os.path.join(CONFIG['BASE_DIR'], 'result', f"{target_dir_value}/train_by_model/{model_name}_{lag}_model_{CONFIG['FORECAST_WEEKS']}.json")

        logging.info(f"All combinations of {model_name} for window = {lag} saved in json")
        with open(result_dir, 'w') as f:
            json.dump(result, f, indent=4)

        return result
    

    def _build_pipeline(self, scoring_func, scoring_name, scaler, model):
        default_k = 5
        if scoring_func is not None:
            pipeline_steps = [
                ('imputer', SimpleImputer(strategy='mean')),  # Ajoutez cette étape
                ('scaler', scaler),
                ('selectkbest', SelectKBest(score_func=scoring_func, k=CONFIG['K_VALUES'].get(scoring_name, default_k))),
                ('model', model)
            ]
        else:
            pipeline_steps = [
                ('imputer', SimpleImputer(strategy='mean')),  # Ajoutez cette étape
                ('scaler', scaler),
                ('model', model)
            ]
        
        return Pipeline(pipeline_steps)

    def _get_selected_features(self, estimator) -> List[str]:
        if 'selectkbest' in estimator.named_steps:
            selector = estimator.named_steps['selectkbest']
            feature_names = self.X_train.columns
            selected_features = feature_names[selector.get_support()].tolist()
            return selected_features
        else:
            return []

    def _create_result_dict(self, total_execution_time, cv_results, y_true, y_pred, selected_features, best_params, mape_score):
        results = {
            'Execution Time': total_execution_time,
            'Mean Train Score': -np.mean(cv_results['train_score']),
            'Mean Test Score': -np.mean(cv_results['test_score']),
            'MAPE_Score': mape_score,
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAE2': median_absolute_error(y_true, y_pred),
            'ME': np.mean(y_true - y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
            'Selected Features': selected_features,
            'Best Parameters': best_params,
            'Predictions': y_pred.tolist()
        }
            
        return results

    @staticmethod

    def _mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        non_zero_idx = y_true != 0
        if np.any(non_zero_idx):
            return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100
        else:
            return np.nan
        


import pandas as pd
import os
from typing import List
import logging
from config import CONFIG
from DataPreprocessor import DataPreprocessor
from ResultAnalyzer import ResultAnalyzer
from Visualizer import Visualizer
from ModelTrainer import ModelTrainer

logging.basicConfig(level=logging.INFO)

class PredictionProject:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.preprocessor = DataPreprocessor(df, target_column)
        self.result_analyzer = ResultAnalyzer(CONFIG['TARGET_DIR'][target_column])
        self.visualizer = Visualizer(CONFIG['TARGET_DIR'][target_column])

    def run(self):
        for lag in CONFIG['LAG_LIST']:
            preprocess_df = self.preprocessor.load_and_preprocess_data(lag)
            X_train, y_train, X_test, y_true = self.preprocessor.split_data()

            trainer = ModelTrainer(X_train, y_train, X_test, y_true, self.target_column)

            if CONFIG['ACTION']["Train models"]:
                for model_name, model in CONFIG['MODELS'].items():
                    trainer.train_model(model_name, model, lag, self.target_column)

        if CONFIG['ACTION']["Save models"]:
            for model_name in CONFIG['MODELS']:
                best_combination = self.result_analyzer.find_global_best_configs(model_name, CONFIG['LAG_LIST'], CONFIG['NON_ML'])
                y_pred = best_combination['Predictions']
                
                # Ajoutez ces lignes pour le débogage
                print(f"Length of y_true: {len(y_true)}, Length of y_pred: {len(y_pred)}")
                y_true_trimmed = y_true[:len(y_pred)]
                self.result_analyzer.calculate_weekly_mape(model_name, y_true_trimmed, y_pred, CONFIG['FORECAST_WEEKS'])
                self.result_analyzer.update_global_results(model_name)


            self.result_analyzer.calc_scores(y_true)
            self.result_analyzer.save_results_to_excel(
                os.path.join(CONFIG['BASE_DIR'], f'result/{self.result_analyzer.target_dir}/{CONFIG["FORECAST_WEEKS"]}w_global_model_results.xlsx'), 
                CONFIG['FORECAST_WEEKS']
            )

            test_dates = preprocess_df['Date'].iloc[-len(X_test):]
            self.visualizer.plot_all_models_predictions(test_dates, y_true)
            self.visualizer.plot_all_models_mape(test_dates)



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


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG  # Remplacez 'Constants as const' par 'config import CONFIG'

class Visualizer:
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def plot_all_models_predictions(self, dates, y_true):
        plt.figure(figsize=(12, 6))
        palette = sns.color_palette("husl", len(CONFIG['MODELS']))

        plt.plot(dates, y_true, label='Actual', color='black')
        for i, model_name in enumerate(CONFIG['MODELS'].keys()):
            with open(f'result/{self.target_dir}/train_by_model/{model_name}_best_model.json', 'r') as file:
                results = json.load(file)
            
            y_pred = results['Predictions']
            min_length = min(len(dates), len(y_true), len(y_pred))
            plt.plot(dates[:min_length], y_pred[:min_length], label=model_name, color=palette[i])

        plt.title('Actual vs Predicted Values for All Models')
        plt.xlabel('Date')
        plt.ylabel(self.target_dir)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/{self.target_dir}/plot_predictions/predictions_all_models.png')
        plt.close()

    def plot_predictions_by_model(self, y_pred, dates, y_true, model_name):
        min_length = min(len(dates), len(y_true), len(y_pred))
        dates = dates[:min_length]
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_true, label='Actual', color='blue')
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
        palette = sns.color_palette("husl", len(CONFIG['MODELS']))

        for i, model_name in enumerate(CONFIG['MODELS'].keys()):
            with open(f'result/{self.target_dir}/weekly_mape/{model_name}_weekly_mape.json', 'r') as file:
                weekly_mape = json.load(file)

            mape_values = list(weekly_mape.values())
            min_length = min(len(dates), len(mape_values))
            plt.plot(dates[:min_length], mape_values[:min_length], label=model_name, color=palette[i])

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
        plt.plot(dates[:len(mape_values)], mape_values, linestyle='-', color='b', label='Weekly MAPE', markersize=8)
        plt.title('Weekly MAPE Over Time')
        plt.xlabel('Week')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'result/{self.target_dir}/plot_mape/weekly_mape_{model_name}.png')
        plt.close()

        