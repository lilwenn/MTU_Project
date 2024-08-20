import pytest
import os
import json
import numpy as np
import pandas as pd
from ResultAnalyzer import ResultAnalyzer
from unittest.mock import mock_open



@pytest.fixture
def result_analyzer():
    return ResultAnalyzer("test_target_dir")

@pytest.fixture
def sample_data():
    return {
        "model1": {
            "Predictions": [1, 2, 3, 4, 5],
            "MAPE_Score": 10.5,
            "MAE": 0.5,
            "Execution Time": 10.0,
            "Scaler": "StandardScaler",
            "Scoring": "f1",
            "Best lag": 3
        },
        "model2": {
            "Predictions": [1.5, 2.5, 3.5, 4.5, 5.5],
            "MAPE_Score": 9.5,
            "MAE": 0.4,
            "Execution Time": 12.0,
            "Scaler": "MinMaxScaler",
            "Scoring": "accuracy",
            "Best lag": 2
        }
    }



def test_calculate_weekly_mape(result_analyzer):
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])

    weekly_mape = result_analyzer.calculate_weekly_mape("test_model", y_true, y_pred, 5)

    assert len(weekly_mape) == 5
    for week, mape in weekly_mape.items():
        assert week.startswith("week_")
        assert isinstance(mape, float)

def test_calculate_weekly_mape_missing_data(result_analyzer):
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.2, 3.3])

    weekly_mape = result_analyzer.calculate_weekly_mape("test_model", y_true, y_pred, 5)

    assert len(weekly_mape) == 5
    assert weekly_mape["week_4"] is None
    assert weekly_mape["week_5"] is None

