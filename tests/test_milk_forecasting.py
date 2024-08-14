import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from IV_Main import create_dict, split_data, build_pipeline

import Constants as const

def test_create_dict():
    total_execution_time = 10
    cv_results = {
        'train_score': [-0.5, -0.4, -0.3],
        'test_score': [-0.6, -0.5, -0.4]
    }
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    selected_features = ['feature1', 'feature2']
    best_params = {'param1': 'value1'}

    result = create_dict(total_execution_time, cv_results, y_true, y_pred, selected_features, best_params)

    assert result['Execution Time'] == total_execution_time
    assert result['Mean Train Score'] == 0.4
    assert result['Mean Test Score'] == 0.5
    assert result['MAPE'] == 0.0  # Because y_true and y_pred are the same
    assert result['Selected Features'] == selected_features
    assert result['Best Parameters'] == best_params

def test_split_data():
    data = {
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [6, 5, 4, 3, 2, 1],
        'target': [10, 20, 30, 40, 50, 60]
    }
    df = pd.DataFrame(data)
    const.FORECAST_WEEKS = 2  # Assume this constant is set somewhere in your code
    X_train, y_train, X_test, y_true = split_data(df, 'target')

    assert X_train.shape == (4, 2)
    assert y_train.shape == (4,)
    assert X_test.shape == (2, 2)
    assert y_true.shape == (2,)

def test_build_pipeline():
    model = RandomForestRegressor()
    scaler = StandardScaler()
    pipeline = build_pipeline(None, '', scaler)

    assert pipeline is not None
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'scaler'
    assert pipeline.steps[1][0] == 'model'
