import time
from test.utils import train_and_predict, evaluate_model, save_json
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from Constants import SCALERS, SCORING_METHODS, MODELS, BEST_K_VALUES, TARGET_COLUMN, FORECAST_WEEKS
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.features = [col for col in df.columns if not col.startswith(f'{TARGET_COLUMN}_next_') and col != 'Date']
        self.results = {}

    def train_model(self, model_name, model):
        self.results[model_name] = {}
        start_time = time.time()
        total_iterations = len(SCALERS) * len(SCORING_METHODS) * FORECAST_WEEKS
        progress_bar = tqdm(total=total_iterations, desc=f"Training {model_name}")

        for scaler_name, scaler in SCALERS.items():
            self.results[model_name][scaler_name] = {}
            for scoring_name, scoring_func in SCORING_METHODS.items():
                self._train_with_scaler_and_scoring(model_name, model, scaler_name, scaler, scoring_name, scoring_func, progress_bar)
        
        progress_bar.close()
        end_time = time.time()
        training_duration = end_time - start_time
        self.results[model_name]['Training_Time'] = training_duration
        save_json(self.results, f'result/by_model/{model_name}_results.json')

    def _train_with_scaler_and_scoring(self, model_name, model, scaler_name, scaler, scoring_name, scoring_func, progress_bar):
        self.results[model_name][scaler_name][scoring_name] = {
            'MAPE': {},
            'MAE': {},
            'Prediction': {}
        }
        default_scoring_func = f_regression
        default_k = 5
        pipeline = Pipeline([
            ('selectkbest', SelectKBest(score_func=scoring_func if scoring_func else default_scoring_func, k=BEST_K_VALUES.get(scoring_name, default_k))),
            ('scaler', scaler),
            ('model', model)
        ])
        for week in range(1, FORECAST_WEEKS + 1):
            target_col = f'{TARGET_COLUMN}_next_{week}weeks'
            y_test, y_pred = train_and_predict(self.df, self.features, target_col, pipeline)
            mape_score, mae_score = evaluate_model(y_test, y_pred)
            self.results[model_name][scaler_name][scoring_name]['MAPE'][f'week_{week}'] = mape_score
            self.results[model_name][scaler_name][scoring_name]['MAE'][f'week_{week}'] = mae_score
            self.results[model_name][scaler_name][scoring_name]['Prediction'][f'week_{week}'] = y_pred.tolist()
            progress_bar.update(1)
            
        print(f'MAPE for {model_name} with {target_col}, scaler {scaler_name}, scoring {scoring_name}: {mape_score:.2f}')


import pandas as pd
from Constants import MODELS
from III_Preprocessing import load_and_preprocess_data

if __name__ == "__main__":
    # df = load_and_preprocess_data()
    df = pd.read_excel('spreadsheet/lagged_results.xlsx')
    trainer = ModelTrainer(df)
    for model_name, model in MODELS.items():
        trainer.train_model(model_name, model)
