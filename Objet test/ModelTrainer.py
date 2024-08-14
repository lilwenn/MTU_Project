import time
import json
import os
import math
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any
import logging
from config import CONFIG

logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_true):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_true = y_true

    def train_model(self, model_name: str, model: Any, lag: int) -> Dict[str, Any]:
        result = {}
        for scaler_name, scaler in CONFIG['SCALERS'].items():
            result[scaler_name] = {}
            for scoring_name, scoring_func in CONFIG['SCORING_METHODS'].items():
                try:
                    pipeline = self._build_pipeline(scoring_func, scoring_name, scaler, model)
                    
                    clf = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=CONFIG['HYPERPARAMETERS'][model_name],
                        n_iter=5, 
                        cv=30,
                        verbose=1,
                        n_jobs=-1,
                        error_score='raise'
                    )

                    start_time = time.time()
                    clf.fit(self.X_train, self.y_train)
                    total_execution_time = time.time() - start_time

                    best_estimator = clf.best_estimator_

                    selected_features = self._get_selected_features(best_estimator)

                    best_params = clf.best_params_

                    cv_results = cross_validate(
                        best_estimator, 
                        self.X_train, 
                        self.y_train, 
                        cv=5, 
                        scoring='neg_mean_squared_error', 
                        return_train_score=True
                    )
                    
                    y_pred = cross_val_predict(best_estimator, self.X_test, self.y_true, cv=5)

                    result[scaler_name][scoring_name] = self._create_result_dict(
                        total_execution_time, cv_results, self.y_true, y_pred, selected_features, best_params
                    )

                    logging.info(f'Scaler: {scaler_name}, Scoring: {scoring_name}, Execution Time: {total_execution_time:.2f} s, '
                                 f'Mean Train Score: {-np.mean(cv_results["train_score"]):.4f}, Mean Test Score: {-np.mean(cv_results["test_score"]):.4f}')
                
                except Exception as e:
                    logging.error(f"Error with Scaler: {scaler_name}, Scoring: {scoring_name}: {e}")
                    result[scaler_name][scoring_name] = {'Error': str(e)}

        result_dir = os.path.join(CONFIG['BASE_DIR'], 'result', f'train_by_model/{model_name}_{lag}_model_{CONFIG["FORECAST_WEEKS"]}.json')
        logging.info(f"All combinations of {model_name} for window = {lag} saved in json")
        with open(result_dir, 'w') as f:
            json.dump(result, f, indent=4)

        return result
    
    def _build_pipeline(self, scoring_func, scoring_name, scaler, model):
        default_k = 5
        if scoring_func is not None:
            pipeline_steps = [
                ('scaler', scaler),
                ('selectkbest', SelectKBest(score_func=scoring_func, k=CONFIG['K_VALUES'].get(scoring_name, default_k))),
                ('model', model)
            ]
        else:
            pipeline_steps = [
                ('scaler', scaler),
                ('model', model)
            ]
        
        return Pipeline(pipeline_steps)

    def _get_selected_features(self, estimator):
        """
        Extract selected features from the fitted SelectKBest component of the pipeline.
        
        Args:
        - estimator (Pipeline): The fitted pipeline with a SelectKBest component.
        
        Returns:
        - List[str]: List of selected feature names.
        """
        if 'selectkbest' in estimator.named_steps:
            selector = estimator.named_steps['selectkbest']
            feature_names = self.X_train.columns
            selected_features = feature_names[selector.get_support()].tolist()
            return selected_features
        else:
            return []

    def _create_result_dict(self, total_execution_time, cv_results, y_true, y_pred, selected_features, best_params):
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


def mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    
    Args:
    - y_true (array-like): True target values.
    - y_pred (array-like): Predicted values.
    
    Returns:
    - float: MAPE score.
    """
    non_zero_idx = y_true != 0
    if np.any(non_zero_idx):
        return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100
    else:
        return np.nan
