

# ___________________________________________________________________________________________________
#                                        IMPORTATIONS
#____________________________________________________________________________________________________



from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import uniform, randint
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, LinearRegression, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import uniform, randint, loguniform
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression 

from III_preprocessing import feature_importance, pearson_corr


import numpy as np



# ___________________________________________________________________________________________________
#                                        CONSTANTS
#____________________________________________________________________________________________________


target_column = 'litres'
forecast_weeks = 52

models = {
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
    #'ARIMA': ARIMA(order=(1, 1, 1))
}


hyperparameters = {
    'BayesianRidge': {
        'model__tol': uniform(1e-2, 1e-4),
        'model__alpha_1': uniform(1e-5, 1e-7),
        'model__alpha_2': uniform(1e-5, 1e-7),
        'model__lambda_1': uniform(1e-5, 1e-7),
        'model__lambda_2': uniform(1e-5, 1e-7)
    },

    'DecisionTreeRegressor': {
    'model__criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
    'model__splitter': ['best', 'random'],
    'model__max_depth': [8, 16, 32, 64, 128, None],
    'model__max_features': [None, 'sqrt', 'log2'] 
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
        'model__max_iter': [500, 1000, 1500]
    },
    'RandomForestRegressor': {
        'model__n_estimators': [50, 100, 200, 300, 400],
        'model__criterion': ['mae', 'mse'],
        'model__max_features': ['auto','sqrt', 'log2', None, 0.5, 1.0],  
        'model__max_depth': [None, 16, 32, 64, 128],
        'model__min_samples_split': [2, 10, 20]
    },
    'Ridge': {
        'model__alpha': [0.2, 0.4, 0.6, 0.8, 1.0],
        'model__tol': [1e-2, 1e-3, 1e-4],
        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
}

k_values = {
    'F_regression': 15,
    'Mutual_info_regression': 25,
    'Pearson Correlation': 35,
    'Feature Importance': 5
}


scoring_methods = {
    'F_regression': f_regression,
    'Mutual_info_regression': mutual_info_regression,
    'Pearson Correlation': pearson_corr,
    'Feature Importance': feature_importance,
    'No scoring': None
}

scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'No scaling': None,
}
