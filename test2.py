import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV  
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, PassiveAggressiveRegressor
from sklearn.feature_selection import SelectKBest, f_regression 
from sklearn.impute import SimpleImputer
from scipy.stats import uniform, loguniform
import json

# Define MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to train and evaluate model
def train_and_evaluate(df, features, target_col, model_pipeline):
    X = df[features]
    y = df[target_col]

    # Split data into train, validation, and test sets (80-10-10)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 = 0.1

    # Train model on training set
    model_pipeline.fit(X_train, y_train)

    # Validate model on validation set
    y_val_pred = model_pipeline.predict(X_val)
    val_score = mean_squared_error(y_val, y_val_pred)

    # Test model on test set
    y_test_pred = model_pipeline.predict(X_test)
    test_mape_score = mape(y_test, y_test_pred)
    test_mae_score = mean_absolute_error(y_test, y_test_pred)

    return val_score, test_mape_score, test_mae_score

# Function to perform Randomized Search CV and get the best model
def perform_random_search(X_train, y_train, model_pipeline, param_distributions):
    random_search = RandomizedSearchCV(model_pipeline, param_distributions, n_iter=50, cv=3, n_jobs=-1, random_state=42, scoring='neg_mean_squared_error')
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Main function to train models
def train_models():
    target_column = 'litres'
    lag = 52

    # Load data
    full_df = pd.read_excel('spreadsheet/Final_Weekly_2009_2021.xlsx')

    # Data cleaning and preprocessing
    full_df = full_df.drop(columns=['EU_milk_price_without UK', 'feed_ex_port', 'Malta_milk_price', 'Croatia_milk_price', 'Malta_Milk_Price'])
    imputer = SimpleImputer(strategy='mean')
    full_df['yield_per_supplier'] = imputer.fit_transform(full_df[['yield_per_supplier']])
    columns_with_nan = full_df.columns[full_df.isna().any()].tolist()
    full_df = full_df.drop(columns=columns_with_nan)

    df = full_df.copy()

    # Lag features
    for i in range(1, lag+1):
        df[f'{target_column}_next_{i}weeks'] = df[target_column].shift(-i)
    df = df.dropna()

    # Features and target
    features = [col for col in df.columns if not col.startswith(f'{target_column}_next_') and col != 'Date']
    
    # Model and parameter configurations
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
        'Ridge': Ridge()
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
            'model__criterion': ['squared_error', 'absolute_error', 'poisson'], 
            'model__max_depth': [8, 16, 32, 64, 128, None],
            'model__splitter': ['best', 'random'],
            'model__max_features': ['auto', 'sqrt', 'log2']
        },
        'ExtraTreeRegressor': {
            'model__criterion': ['squared_error', 'absolute_error', 'poisson'],
            'model__max_depth': [8, 16, 32, 64, 128, None],
            'model__splitter': ['best', 'random'],
            'model__max_features': ['auto', 'sqrt', 'log2']
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
            'model__epsilon': [0.0, 0.5, 1.0],
            'model__tol': [1e-3, 1e-4, 1e-5],
            'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'model__fit_intercept': [True, False],
            'model__max_iter': [500, 1000, 1500]
        },
        'RandomForestRegressor': {
            'model__n_estimators': [50, 100],
            'model__max_features': ['auto', 'sqrt', 'log2']
        },
        'Ridge': {
            'model__alpha': [0.2, 0.5, 0.8],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    }

    scalers = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
    }

    scoring_methods = {
        'f_regression': f_regression,
        None: None
    }

    results = []

    # Training, validation, and testing over multiple iterations
    for iteration in range(30):
        for model_name, model in models.items():
            for scaler_name, scaler in scalers.items():
                for scoring_name, scoring_func in scoring_methods.items():
                    pipeline_steps = [('scaler', scaler)]
                    if scoring_func:
                        pipeline_steps.insert(0, ('selectkbest', SelectKBest(score_func=scoring_func, k=10)))
                    pipeline_steps.append(('model', model))
                    model_pipeline = Pipeline(pipeline_steps)

                    # Perform random search to get the best model
                    best_model = perform_random_search(df[features], df[f'{target_column}_next_1weeks'], model_pipeline, hyperparameters[model_name])

                    mape_scores = []
                    mae_scores = []

                    # Evaluate over multiple lags
                    for week in range(1, lag + 1):
                        target_col = f'{target_column}_next_{week}weeks'
                        val_score, test_mape_score, test_mae_score = train_and_evaluate(df, features, target_col, best_model)

                        mape_scores.append(test_mape_score)
                        mae_scores.append(test_mae_score)
                        print(f'Iteration {iteration + 1}, {model_name}, {scaler_name}, {scoring_name}, Week {week}, Validation Score: {val_score:.2f}, Test MAPE: {test_mape_score:.2f}, Test MAE: {test_mae_score:.2f}')

                    results.append({
                        'Iteration': iteration + 1,
                        'Model': model_name,
                        'Scaler': scaler_name,
                        'Feature Selection': scoring_name,
                        'MAPE': mape_scores,
                        'MAE': mae_scores
                    })

    # Save results to JSON
    with open('result/final_results2.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

train_models()
