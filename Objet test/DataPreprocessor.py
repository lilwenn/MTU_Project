import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # To enable IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter
import logging
from config import CONFIG

logging.basicConfig(level=logging.INFO)

class DataPreprocessor:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def load_and_preprocess_data(self, lag: int) -> pd.DataFrame:
        """
        Load and preprocess data: clean, impute missing values, and create lagged features.
        
        Args:
        - lag (int): Number of past periods to create lag features.
        
        Returns:
        - df (DataFrame): Preprocessed DataFrame ready for model training.
        """
        output_file = os.path.join(CONFIG['BASE_DIR'], f'spreadsheet/preproc_lag_{lag}.xlsx')
        
        if os.path.exists(output_file):
            return pd.read_excel(output_file)

        logging.info('Preprocessing...')
        logging.info(f'Size of the initial dataset: {self.df.shape}')

        logging.info('     - Imputing')
        self.df = self._impute_missing_values()

        logging.info('     - Smoothing')
        if CONFIG['ACTION']["time_series_smoothing"]:
            self.df = self._time_series_smoothing(CONFIG['SMOOTH_WINDOW'])

        logging.info('     - New feature creation')
        self.df = self._new_features_creation()

        logging.info('     - Shifting')
        if CONFIG['ACTION']["shifting"]:
            self.df = self._create_lag_features(lag)

        logging.info('     - Multi-step Forecasting Features')
        if CONFIG['ACTION']["Multi-step"]:
            self.df = self._create_multi_step_features(CONFIG['FORECAST_WEEKS'])

        logging.info(f'Size of the dataset after preprocessing: {self.df.shape}')

        self.df.to_excel(output_file, index=False)
        logging.info(f"Preprocessed file saved as {output_file}.")

        return self.df

    def _impute_missing_values(self) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame using IterativeImputer.
        
        Returns:
        - df (DataFrame): DataFrame with missing values imputed.
        """
        df = self.df
        columns_with_nan = df.columns[df.isna().any()].tolist()
        
        if not columns_with_nan:
            logging.info("No missing values found in the DataFrame.")
            return df

        iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=100, random_state=42, tol=1e-3)
        df[columns_with_nan] = iterative_imputer.fit_transform(df[columns_with_nan])

        return df

    def _time_series_smoothing(self, window: int) -> pd.DataFrame:
        """
        Perform time series smoothing using different methods.
        
        Args:
        - window (int): Number of periods for smoothing.
        
        Returns:
        - data_SG (DataFrame): DataFrame with smoothed features.
        """
        prep_data = self.df.copy()
        exclude_cols = ['Date', 'Week', self.target_column, 'litres', 'num_suppliers']
        impute_cols = prep_data.columns.difference(exclude_cols)

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        prep_data[impute_cols] = imputer.fit_transform(prep_data[impute_cols])

        feature_cols = prep_data.columns.difference(exclude_cols)

        if CONFIG['ACTION']["compare lifting methods"]:
            data_MA = create_MA(prep_data[feature_cols], past_time=window)
            data_WMA = create_WMA(prep_data[feature_cols], window_size=20)
            data_EW = create_exponential_smoothing(prep_data[feature_cols], span=window)
        
        data_SG = create_savgol_smoothing(prep_data[feature_cols], window_length=window, polyorder=2)
        data_SG = pd.concat([prep_data[exclude_cols], data_SG], axis=1)

        """        if CONFIG['ACTION']["compare lifting methods"]:
                        plot_comparison(prep_data, data_MA, data_WMA, data_EW, data_SG, "Ireland_Milk_price")
        """
        return data_SG
    
    def _new_features_creation(self) -> pd.DataFrame:
        df = self.df
        df['yield_per_supplier'] = df['litres'] / df['num_suppliers']
        df['cos_week'] = np.cos(df['Week'] * (2 * np.pi / 52))
        df['past_values'] = df[self.target_column].expanding().mean()
        return df

    def _create_lag_features(self, lag: int) -> pd.DataFrame:
        """
        Create lagged features for time series forecasting.
        
        Args:
        - lag (int): Number of lag periods to create features for.
        
        Returns:
        - new_data (DataFrame): DataFrame with lagged features.
        """
        data = self.df
        lagged_cols = []
        for col in data.columns:
            if col not in ["Date", "Week", self.target_column]:
                for i in range(1, lag + 1):
                    lagged_col = data[col].shift(i)
                    lagged_col.name = f'{col}-{i}'
                    lagged_cols.append(lagged_col)

        lagged_data = pd.concat(lagged_cols, axis=1)
        data = pd.concat([data, lagged_data], axis=1)

        new_data = data.iloc[lag:]

        return new_data
    
    def _create_multi_step_features(self, n_steps: int) -> pd.DataFrame:
        df = self.df
        for step in range(1, n_steps + 1):
            df[f'{self.target_column}_step_{step}'] = df[self.target_column].shift(-step)
        
        df.dropna(inplace=True)
        return df

    def split_data(self):
        """
        Split the data into training and testing sets based on the number of test periods.
        
        Returns:
        - X_train, y_train: Features and target for the training set
        - X_test, y_true: Features and target for the test set
        """
        df = self.df
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

        logging.info(f'Features for training: {X_train.shape}')
        logging.info(f'Target for training: {y_train.shape}')
        logging.info(f'Features for testing: {X_test.shape}')
        logging.info(f'True target for testing: {y_true.shape}')

        return X_train, y_train, X_test, y_true


def create_MA(data, past_time):
    ma_cols = []
    for col in data.columns:
        ma_col = data[col].rolling(window=past_time, min_periods=1).mean()
        ma_col.name = f'{col}_MA'
        ma_cols.append(ma_col)
    
    ma_data = pd.concat(ma_cols, axis=1)
    return ma_data

def create_savgol_smoothing(data, window_length, polyorder):
    sg_cols = []
    for col in data.columns:
        smoothed_col = savgol_filter(data[col].fillna(method='bfill'), window_length=window_length, polyorder=polyorder, mode='interp')
        smoothed_col = pd.Series(smoothed_col, index=data.index, name=f'{col}')
        sg_cols.append(smoothed_col)
    
    sg_data = pd.concat(sg_cols, axis=1)
    return sg_data

def create_WMA(data, window_size):
    wma_cols = []
    for col in data.columns:
        weights = np.arange(1, window_size + 1)
        wma_col = data[col].rolling(window=window_size).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
        wma_col.name = f'{col}_WMA'
        wma_cols.append(wma_col)
    
    wma_data = pd.concat(wma_cols, axis=1).dropna()
    return wma_data

def create_exponential_smoothing(data, span):
    ewm_cols = []
    for col in data.columns:
        ewm_col = data[col].ewm(span=span).mean()
        ewm_col.name = f'{col}_EW'
        ewm_cols.append(ewm_col)
    
    ewm_data = pd.concat(ewm_cols, axis=1)
    return ewm_data
