import pytest
import numpy as np
from sklearn.datasets import make_regression
from ModelTrainer import ModelTrainer
from config import CONFIG

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    X_train, X_test = X[:80], X[80:]
    y_train, y_true = y[:80], y[80:]
    return X_train, y_train, X_test, y_true

def test_train_model(sample_data):
    X_train, y_train, X_test, y_true = sample_data
    trainer = ModelTrainer(X_train, y_train, X_test, y_true)
    
    model_name = 'LinearRegression'
    model = CONFIG['MODELS'][model_name]
    result = trainer.train_model(model_name, model, lag=2)
    
    assert isinstance(result, dict)
    assert 'StandardScaler' in result
    assert 'MinMaxScaler' in result
    assert 'f_regression' in result['StandardScaler']

def test_build_pipeline(sample_data):
    X_train, y_train, X_test, y_true = sample_data
    trainer = ModelTrainer(X_train, y_train, X_test, y_true)
    
    pipeline = trainer._build_pipeline(
        CONFIG['SCORING_METHODS']['f_regression'],
        'f_regression',
        CONFIG['SCALERS']['StandardScaler'],
        CONFIG['MODELS']['LinearRegression']
    )
    
    assert 'scaler' in pipeline.named_steps
    assert 'selectkbest' in pipeline.named_steps
    assert 'model' in pipeline.named_steps

def test_create_result_dict(sample_data):
    X_train, y_train, X_test, y_true = sample_data
    trainer = ModelTrainer(X_train, y_train, X_test, y_true)
    
    result = trainer._create_result_dict(
        total_execution_time=1.0,
        cv_results={'train_score': [-0.1, -0.2], 'test_score': [-0.3, -0.4]},
        y_true=y_true,
        y_pred=y_true + 0.1,
        selected_features=['feature1', 'feature2'],
        best_params={'param1': 1, 'param2': 2}
    )
    
    assert 'Execution Time' in result
    assert 'Mean Train Score' in result
    assert 'Mean Test Score' in result
    assert 'MAPE_Score' in result
    assert 'Selected Features' in result
    assert 'Best Parameters' in result