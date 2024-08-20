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