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
            
            trainer = ModelTrainer(X_train, y_train, X_test, y_true)
            
            for model_name, model in CONFIG['MODELS'].items():
                if CONFIG['ACTION']["Train models"]:
                    results = trainer.train_model(model_name, model, lag)
                
        if CONFIG['ACTION']["Save models"]:
            for model_name in CONFIG['MODELS']:
                best_combination = self.result_analyzer.find_global_best_configs(model_name, CONFIG['LAG_LIST'], CONFIG['NON_ML'])
                self.result_analyzer.update_global_results(model_name)
                
            self.result_analyzer.calc_scores(y_true)
            self.result_analyzer.save_results_to_excel(os.path.join(CONFIG['BASE_DIR'], f'result/{self.result_analyzer.target_dir}/{CONFIG["FORECAST_WEEKS"]}w_global_model_results.xlsx'), CONFIG['FORECAST_WEEKS'])
            
            test_dates = preprocess_df['Date'].iloc[len(X_train):len(X_train) + len(X_test)]
            self.visualizer.plot_all_models_predictions(test_dates, y_true)
            self.visualizer.plot_all_models_mape(test_dates)