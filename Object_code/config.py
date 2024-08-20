import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso, PassiveAggressiveRegressor, Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from scipy.stats import uniform, loguniform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def feature_importance(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model.feature_importances_

def pearson_corr(X, y):
    import numpy as np
    corr_matrix = np.corrcoef(X, y, rowvar=False)
    corr_with_target = np.abs(corr_matrix[:-1, -1])  
    return corr_with_target

CONFIG = {
    "BASE_DIR": BASE_DIR,
    "TARGET_COLUMN": ['litres'],
    "TARGET_DIR": {
        'litres': 'liter_results',
        'Ireland_Milk_Price': 'prices_results'
    },
    "FORECAST_WEEKS": 52,
    "HORIZON": 52,
    "LAG_LIST": [1, 2 , 3, 4, 5, 6, 7],
    "SMOOTH_WINDOW": 5,
    "NON_ML": ['Pmdarima', 'Darts', 'ARIMA'],
    "ACTION": {
        "time_series_smoothing": True,
        "shifting": True,
        "compare_lifting_methods": False,
        "Multi-step": False,
        "Train models": False,
        "Save models": True
    },
    "MODELS": {
        'BayesianRidge': BayesianRidge(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'ExtraTreeRegressor': ExtraTreeRegressor(),
        'GaussianProcessRegressor': GaussianProcessRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'Lasso': Lasso(),
        'LinearRegression': LinearRegression(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'Ridge': Ridge(),
    },

    "HYPERPARAMETERS": {
        'BayesianRidge': {
            'model__tol': uniform(1e-2, 1e-4),
            'model__alpha_1': uniform(1e-5, 1e-7),
            'model__alpha_2': uniform(1e-5, 1e-7),
            'model__lambda_1': uniform(1e-5, 1e-7),
            'model__lambda_2': uniform(1e-5, 1e-7)
        },
        'DecisionTreeRegressor': {
            'model__criterion': ['absolute_error', 'squared_error'],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
        },
        'ExtraTreeRegressor': {
            'model__criterion': ['squared_error', 'absolute_error', 'poisson'],
            'model__max_depth': [8, 16, 32, 64, 128, None],
            'model__splitter': ['best', 'random'],
            'model__max_features': [None, 'sqrt', 'log2']
        },
        'GaussianProcessRegressor': {
            'model__alpha': loguniform(1e-12, 1e-8),
            'model__n_restarts_optimizer': [0, 1, 2, 3],
            'model__normalize_y': [True, False]
        },
        'KNeighborsRegressor': {
            'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'model__weights': ['uniform', 'distance'],
            'model__p': [2, 3, 4]
        },
        'Lasso': {
            'model__alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
            'model__tol': [1e-2, 1e-3, 1e-4],
            'model__selection': ['random', 'cyclic']
        },
        'LinearRegression': {},
        'PassiveAggressiveRegressor': {
            'model__C': [0.01, 0.1, 1.0, 10],
            'model__epsilon': [0.001, 0.01, 0.1, 1.0],
            'model__tol': [1e-3, 1e-4, 1e-5],
            'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'model__fit_intercept': [True, False],
            'model__max_iter': [500, 1000, 1500, 2000, 3000]
        },
        'RandomForestRegressor': {
            'model__n_estimators': [50, 100, 200, 300, 400],
            'model__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'model__max_features': ['sqrt', 'log2', None, 0.5, 1.0],
            'model__max_depth': [None, 16, 32, 64, 128],
            'model__min_samples_split': [2, 10, 20]
        },
        'Ridge': {
            'model__alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
            'model__tol': [1e-2, 1e-3, 1e-4],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },
    "K_VALUES": {
        'F_regression': 15,
        'Mutual_info_regression': 25,
        'Pearson Correlation': 35,
        'Feature Importance': 5
    },
    "SCORING_METHODS": {
        'F_regression': f_regression,
        'Mutual_info_regression': mutual_info_regression,
        'Pearson Correlation': pearson_corr,
        'Feature Importance': feature_importance,
        'No scoring': None
    },
    "SCALERS": {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(),
        'No scaling': None,
    }
}
