import pytest
import pandas as pd
import numpy as np
import os
from DataPreprocessor import DataPreprocessor
from config import CONFIG

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'Week': range(1, 101),
        'litres': np.random.rand(100) * 1000,
        'num_suppliers': np.random.randint(50, 100, 100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
    })

@pytest.fixture
def preprocessor(sample_df):
    return DataPreprocessor(sample_df, 'litres')

def test_init(preprocessor):
    assert isinstance(preprocessor.df, pd.DataFrame)
    assert preprocessor.target_column == 'litres'

def test_load_and_preprocess_data(preprocessor, monkeypatch):
    def mock_impute_missing_values(self):
        return self.df

    def mock_time_series_smoothing(self, window):
        return self.df

    def mock_new_features_creation(self):
        return self.df

    def mock_create_lag_features(self, lag):
        return self.df

    monkeypatch.setattr(DataPreprocessor, '_impute_missing_values', mock_impute_missing_values)
    monkeypatch.setattr(DataPreprocessor, '_time_series_smoothing', mock_time_series_smoothing)
    monkeypatch.setattr(DataPreprocessor, '_new_features_creation', mock_new_features_creation)
    monkeypatch.setattr(DataPreprocessor, '_create_lag_features', mock_create_lag_features)

    result = preprocessor.load_and_preprocess_data(1)
    assert isinstance(result, pd.DataFrame)

def test_impute_missing_values(preprocessor):
    preprocessor.df.loc[0, 'feature1'] = np.nan
    result = preprocessor._impute_missing_values()
    assert not result['feature1'].isna().any()

def test_time_series_smoothing(preprocessor):
    result = preprocessor._time_series_smoothing(5)
    assert isinstance(result, pd.DataFrame)
    assert 'feature1' in result.columns
    assert 'feature2' in result.columns

def test_new_features_creation(preprocessor):
    result = preprocessor._new_features_creation()
    assert 'yield_per_supplier' in result.columns
    assert 'cos_week' in result.columns
    assert 'past_values' in result.columns

def test_create_lag_features(preprocessor):
    result = preprocessor._create_lag_features(2)
    assert 'feature1-1' in result.columns
    assert 'feature1-2' in result.columns
    assert 'feature2-1' in result.columns
    assert 'feature2-2' in result.columns


def test_create_multi_step_features(sample_df):
    preprocessor = DataPreprocessor(sample_df, 'litres')
    multi_step_df = preprocessor._create_multi_step_features(n_steps=2)
    
    assert 'litres_step_1' in multi_step_df.columns
    assert 'litres_step_2' in multi_step_df.columns

def test_split_data(sample_df):
    preprocessor = DataPreprocessor(sample_df, 'litres')
    X_train, y_train, X_test, y_true = preprocessor.split_data()
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_true, pd.Series)

def test_missing_columns(preprocessor):
    df_missing_columns = preprocessor.df.drop(columns=['num_suppliers'])
    preprocessor.df = df_missing_columns
    with pytest.raises(KeyError, match="Required columns 'num_suppliers' or 'litres' are missing from the DataFrame."):
        preprocessor._new_features_creation()

def test_impute_missing_values_with_no_na(preprocessor):
    preprocessor.df['feature1'] = np.random.rand(100)
    result = preprocessor._impute_missing_values()
    assert result['feature1'].isna().sum() == 0
