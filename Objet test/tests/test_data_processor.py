import pytest
import pandas as pd
import numpy as np
from DataPreprocessor import DataPreprocessor
from config import CONFIG

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'target': np.random.rand(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })

def test_load_and_preprocess_data(sample_df):
    preprocessor = DataPreprocessor(sample_df, 'target')
    result = preprocessor.load_and_preprocess_data(lag=2)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_df) - 2  # Due to lag
    assert 'feature1-1' in result.columns
    assert 'feature2-2' in result.columns

def test_impute_missing_values(sample_df):
    sample_df.loc[0, 'feature1'] = np.nan
    preprocessor = DataPreprocessor(sample_df, 'target')
    result = preprocessor._impute_missing_values()
    
    assert result['feature1'].isnull().sum() == 0

def test_time_series_smoothing(sample_df):
    preprocessor = DataPreprocessor(sample_df, 'target')
    result = preprocessor._time_series_smoothing(window=3)
    
    assert not result.equals(sample_df)
    assert result.shape == sample_df.shape

def test_create_lag_features(sample_df):
    preprocessor = DataPreprocessor(sample_df, 'target')
    result = preprocessor._create_lag_features(lag=2)
    
    assert 'feature1-1' in result.columns
    assert 'feature2-2' in result.columns
    assert len(result) == len(sample_df) - 2

def test_split_data(sample_df):
    preprocessor = DataPreprocessor(sample_df, 'target')
    X_train, y_train, X_test, y_true = preprocessor.split_data()
    
    assert len(X_train) + len(X_test) == len(sample_df) - CONFIG['FORECAST_WEEKS']
    assert len(y_train) == len(X_train)
    assert len(y_true) == len(X_test)