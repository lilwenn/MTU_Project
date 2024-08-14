import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_regression, mutual_info_regression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TARGET_COLUMN": ["column1", "column2"],
    "LAG_LIST": [1, 2, 3, 4],
    "FORECAST_WEEKS": 4,
    "ACTION": {
        "time_series_smoothing": True,
        "shifting": True,
        "Multi-step": True,
        "Train models": True,
        "Save models": True
    },
    "MODELS": {
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "LinearRegression": LinearRegression()
    },
    "SCALERS": {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    },
    "SCORING_METHODS": {
        "f_regression": f_regression,
        "mutual_info_regression": mutual_info_regression
    },
    "K_VALUES": {
        "f_regression": 5,
        "mutual_info_regression": 5
    },
    "HYPERPARAMETERS": {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30]
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "TARGET_DIR": {
        "column1": "target1_results",
        "column2": "target2_results"
    }
}