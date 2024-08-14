import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from config import CONFIG

@pytest.fixture(scope="session")
def sample_df():
    """Create a sample DataFrame that can be used across multiple test files."""
    return pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'target': np.random.rand(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })

@pytest.fixture(scope="session")
def sample_regression_data():
    """Create a sample regression dataset that can be used across multiple test files."""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test = X[:80], X[80:]
    y_train, y_true = y[:80], y[80:]
    return X_train, y_train, X_test, y_true

@pytest.fixture(scope="session")
def mock_config():
    """Create a mock configuration that can be used across multiple test files."""
    return {
        "TARGET_COLUMN": ["target"],
        "LAG_LIST": [1, 2, 3],
        "FORECAST_WEEKS": 4,
        "ACTION": {
            "time_series_smoothing": True,
            "shifting": True,
            "Multi-step": True,
            "Train models": True,
            "Save models": True
        },
        "MODELS": {
            "MockModel": lambda: None
        },
        "SCALERS": {
            "MockScaler": lambda: None
        },
        "SCORING_METHODS": {
            "mock_scoring": lambda: None
        },
        "K_VALUES": {
            "mock_scoring": 5
        },
        "HYPERPARAMETERS": {
            "MockModel": {
                "param1": [1, 2, 3],
                "param2": [4, 5, 6]
            }
        },
        "TARGET_DIR": {
            "target": "mock_target_results"
        }
    }

@pytest.fixture(autouse=True)
def mock_config_setup(monkeypatch, mock_config):
    """Automatically use the mock configuration for all tests."""
    monkeypatch.setattr("config.CONFIG", mock_config)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory that can be used across multiple test files."""
    return tmp_path