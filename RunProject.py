import pandas as pd
import os
from typing import List
import logging
from config import CONFIG
from DataPreprocessor import DataPreprocessor
from ResultAnalyzer import ResultAnalyzer
from Visualizer import Visualizer
from ModelTrainer import ModelTrainer
from DatasetCreation import PriceDataProcessor, GrassDataProcessor, YieldDataProcessor, InflationDataProcessor, MeteoDataProcessor, FinalDataProcessor

logging.basicConfig(level=logging.INFO)

class PredictionProject:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.preprocessor = DataPreprocessor(df, target_column)
        self.result_analyzer = ResultAnalyzer(CONFIG['TARGET_DIR'][target_column])
        self.visualizer = Visualizer(CONFIG['TARGET_DIR'][target_column])

        # Initialize data processors
        self.price_processor = PriceDataProcessor()
        self.grass_processor = GrassDataProcessor()
        self.yield_processor = YieldDataProcessor()
        self.inflation_processor = InflationDataProcessor()
        self.meteo_processor = MeteoDataProcessor()
        self.final_processor = FinalDataProcessor()

    def run(self):
        if CONFIG['ACTION']["Data creation"]:
            print("Starting data creation...")

            # Run data processors
            print("Processing price data...")
            self.price_processor.process()

            print("Processing grass data...")
            self.grass_processor.process()

            print("Processing yield data...")
            self.yield_processor.process()

            print("Processing inflation data...")
            self.inflation_processor.process()

            print("Processing meteorological data...")
            self.meteo_processor.process()

            print("Finalizing data processing...")
            self.final_processor.process()

            print("Reloading data after processing...")
            self.df = pd.read_excel("spreadsheet/Final_Weekly_2009_2021.xlsx")
            self.preprocessor = DataPreprocessor(self.df, self.target_column)

        for lag in CONFIG['LAG_LIST']:
            print(f"Preprocessing data for a lag of {lag} weeks...")
            preprocess_df = self.preprocessor.load_and_preprocess_data(lag)
            X_train, y_train, X_test, y_true = self.preprocessor.split_data()

            trainer = ModelTrainer(X_train, y_train, X_test, y_true, self.target_column)

            if CONFIG['ACTION']["Train models"]:
                print(f"Training models for a lag of {lag} weeks...")
                for model_name, model in CONFIG['MODELS'].items():
                    print(f"Training model {model_name}...")
                    trainer.train_model(model_name, model, lag, self.target_column)

        if CONFIG['ACTION']["Save models"]:
            print("Saving models and results...")
            for model_name in CONFIG['MODELS']:
                print(f"Analyzing results for model {model_name}...")
                best_combination = self.result_analyzer.find_global_best_configs(model_name, CONFIG['LAG_LIST'], CONFIG['NON_ML'])
                y_pred = best_combination['Predictions']

                print(f"Calculating weekly MAPE for {model_name}...")
                print(f"Length of y_true: {len(y_true)}, Length of y_pred: {len(y_pred)}")
                y_true_trimmed = y_true[:len(y_pred)]
                self.result_analyzer.calculate_weekly_mape(model_name, y_true_trimmed, y_pred, CONFIG['FORECAST_WEEKS'])
                self.result_analyzer.update_global_results(model_name)

            print("Calculating global scores...")
            self.result_analyzer.calc_scores(y_true)

            print("Saving results to an Excel file...")
            self.result_analyzer.save_results_to_excel(
                os.path.join(CONFIG['BASE_DIR'], f'result/{self.result_analyzer.target_dir}/{CONFIG["FORECAST_WEEKS"]}w_global_model_results.xlsx'), 
                CONFIG['FORECAST_WEEKS']
            )

            print("Visualizing model predictions...")
            test_dates = preprocess_df['Date'].iloc[-len(X_test):]
            self.visualizer.plot_all_models_predictions(test_dates, y_true)
            self.visualizer.plot_all_models_mape(test_dates)

        print("Process completed.")

