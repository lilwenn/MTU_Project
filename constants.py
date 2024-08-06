

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

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler, RobustScaler
from scipy.stats import uniform, randint

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, LinearRegression, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import uniform, randint, loguniform
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression 



import numpy as np


def feature_importance(X, y):
    """
    Calculate feature importances using a RandomForestRegressor.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Features matrix.
        y (numpy.ndarray or pandas.Series): Target variable.

    Returns:
        numpy.ndarray: Feature importances.
    """
    model = RandomForestRegressor()
    model.fit(X, y)

    return model.feature_importances_


def pearson_corr(X, y):
    """
    Calculate Pearson correlation coefficients between features and target variable.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Features matrix.
        y (numpy.ndarray or pandas.Series): Target variable.

    Returns:
        numpy.ndarray: Pearson correlation coefficients.
    """
    corr_matrix = np.corrcoef(X, y, rowvar=False)
    corr_with_target = np.abs(corr_matrix[:-1, -1])  
    return corr_with_target



# ___________________________________________________________________________________________________
#                                        CONSTANTS
#____________________________________________________________________________________________________



TARGET_COLUMN = 'litres' # Ireland_Milk_price
FORECAST_WEEKS = 52
LAG = 1  
WINDOWS_LIST = [4]
NON_ML = ['Pmdarima', 'Darts' , 'ARIMA']


ACTION = {
    "time_series_smoothing" : True,
    "shifting": True,
    "compare lifting methods": False,
    "Multi-step" : True
}

MODELS = {
    'BayesianRidge': BayesianRidge(),
    #'DecisionTreeRegressor': DecisionTreeRegressor(),
    #'ExtraTreeRegressor': ExtraTreeRegressor(),
    #'GaussianProcessRegressor': GaussianProcessRegressor(),
    #'KNeighborsRegressor': KNeighborsRegressor(),
    #'Lasso': Lasso(),
    #'LinearRegression': LinearRegression(),
    #'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
    #'RandomForestRegressor': RandomForestRegressor(),
    #'Ridge': Ridge(),
    #'ARIMA': None,
    #'Pmdarima': None,
    #'Prophet': None,
    #'Darts': None,
    #'TPOT': None,
}


HYPERPARAMETERS = {
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
}

K_VALUES = {
    'F_regression': 15,
    'Mutual_info_regression': 25,
    'Pearson Correlation': 35,
    'Feature Importance': 5
}

SCORING_METHODS = {
    'F_regression': f_regression,
    #'Mutual_info_regression': mutual_info_regression,
    'Pearson Correlation': pearson_corr,
    #'Feature Importance': feature_importance,
    'No scoring': None
}

SCALERS = {
    #'MinMaxScaler': MinMaxScaler(),
    #'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer': QuantileTransformer(),
    'No scaling': None,
}
